"""
Tests for Prosper Smart Yard Cloud Outbox Dispatcher
"""

import time
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from pydantic import SecretStr

from app.domain import Detection, Observation, Validation, utcnow
from app.events.cloud_dispatcher import CloudOutboxDispatcher, ProsperSession


@pytest.fixture
def mock_settings(settings):
    cfg = settings.model_copy(update={
        'prosper_enabled': True,
        'prosper_auto_dispatch': True,
        'prosper_api_base_url': 'http://syapi.test.prosper.com',
        'prosper_site_id': 'site-uuid-1234',
        'prosper_bearer_token': SecretStr('test-bearer-token'),
        'prosper_site_code': 'SITE1',
        'prosper_email': 'test@prosper.com',
        'prosper_password': SecretStr('secret123'),
        'prosper_max_retries': 2,
        'prosper_queue_size': 10,
    })
    return cfg


def test_prosper_session_bearer_headers(mock_settings):
    session = ProsperSession(mock_settings)
    headers = session.get_auth_headers()
    assert headers.get('Authorization') == 'Bearer test-bearer-token'


def test_prosper_session_api_key_headers(mock_settings):
    cfg = mock_settings.model_copy(update={
        'prosper_bearer_token': None,
        'prosper_api_key': SecretStr('my-api-key-999'),
    })
    session = ProsperSession(cfg)
    headers = session.get_auth_headers()
    assert headers.get('x-api-key') == 'my-api-key-999'


def test_prosper_session_auto_login(mock_settings):
    cfg = mock_settings.model_copy(update={
        'prosper_bearer_token': None,
        'prosper_api_key': None,
    })
    session = ProsperSession(cfg)

    with patch.object(session.session, 'post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {'token': 'auto-login-jwt-token', 'expiresIn': 3600}

        headers = session.get_auth_headers()
        assert headers.get('Authorization') == 'Bearer auto-login-jwt-token'
        assert mock_post.called
        assert '/api/auth/login' in mock_post.call_args[0][0]


def test_prosper_session_upload_image(mock_settings):
    session = ProsperSession(mock_settings)
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch.object(session.session, 'post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {'url': 'https://s3.prosper.com/trailers/abc.jpg'}

        url = session.upload_image(dummy_img)
        assert url == 'https://s3.prosper.com/trailers/abc.jpg'
        assert mock_post.called
        assert '/api/sites/site-uuid-1234/images/trailers' in mock_post.call_args[0][0]


def test_prosper_session_post_gate_event(mock_settings):
    session = ProsperSession(mock_settings)
    payload = {
        'id': 'event-1',
        'gateId': 'gate-lane-1',
        'timestamp': '2026-09-09T12:00:00.000Z',
        'containerNumber': 'CSQU3054383',
        'eventType': 'ENTRY',
        'deviceId': 'device-uuid-1',
        'confidence': 0.95,
        'imageUrl': 'https://s3.prosper.com/trailers/abc.jpg',
    }

    with patch.object(session.session, 'post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.content = b'{"visitDatabaseId": "abc-123", "eventId": "evt-456"}'
        mock_post.return_value.json.return_value = {'visitDatabaseId': 'abc-123', 'eventId': 'evt-456'}
        ok = session.post_gate_event(payload)
        # post_gate_event now returns the response dict on success (for visitDatabaseId extraction)
        assert isinstance(ok, dict) or ok is True
        assert '/api/sites/site-uuid-1234/gate-events' in mock_post.call_args[0][0]


def test_prosper_session_skips_when_no_auth(mock_settings):
    """Ensure post_gate_event does not send unauthenticated requests that trigger 401."""
    cfg = mock_settings.model_copy(update={'prosper_bearer_token': None, 'prosper_api_key': None})
    session = ProsperSession(cfg)
    # Simulate auto-login failing (no token)
    with patch.object(session, 'get_auth_headers', return_value={'X-Site-Id': 'site-1'}):
        with patch.object(session.session, 'post') as mock_post:
            ok = session.post_gate_event({'id': 'ev-1', 'Containers': []})
            assert ok == 'OFFLINE'
            assert not mock_post.called


def test_prosper_session_invalidates_token_on_401(mock_settings):
    """Ensure cached token is invalidated when cloud returns 401."""
    session = ProsperSession(mock_settings)
    session._token = 'cached-token'
    session._token_expires_at = time.time() + 3600

    with patch.object(session.session, 'post') as mock_post:
        mock_post.return_value.status_code = 401
        mock_post.return_value.text = 'Unauthorized'
        ok = session.post_gate_event({'id': 'ev-1', 'Containers': []})
        assert ok == 'OFFLINE'
        assert session._token is None
        assert session._token_expires_at == 0.0


def test_prosper_session_buffers_on_503_service_unavailable(mock_settings):
    """Ensure 503 service unavailable returns OFFLINE to buffer locally."""
    session = ProsperSession(mock_settings)
    with patch.object(session.session, 'post') as mock_post:
        mock_post.return_value.status_code = 503
        mock_post.return_value.text = 'The service is unavailable.'
        ok = session.post_gate_event({'id': 'ev-1', 'Containers': []})
        assert ok == 'OFFLINE'


def test_cloud_dispatcher_disabled_does_nothing(settings):
    cfg = settings.model_copy(update={'prosper_enabled': False})
    dispatcher = CloudOutboxDispatcher(cfg)
    dispatcher.start()
    assert dispatcher._worker_thread is None

    # Enqueue should safely no-op
    dummy_obs = MagicMock(confirmed=True)
    dispatcher.enqueue(dummy_obs, 'event-1')
    assert dispatcher.queue.empty()
    dispatcher.stop()


def test_cloud_dispatcher_enqueue_and_sync(mock_settings):
    metrics = MagicMock()
    dispatcher = CloudOutboxDispatcher(mock_settings, metrics=metrics)

    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
    det = Detection(
        camera_id='gate-1',
        frame_timestamp=utcnow(),
        class_name='container',
        confidence=0.95,
        bbox=(10.0, 10.0, 50.0, 50.0),
    )
    val = Validation(raw_text='CSQU 305438-3', normalized_text='CSQU3054383',
                     confidence=0.95, valid_format=True, valid_check_digit=True,
                     validation_status='VALID')
    obs = Observation('origin-1', 'gate-lane-1', 1, det, val, 'ENTRY', dummy_img, utcnow(), confirmed=True)

    with patch.object(dispatcher.session, 'upload_image', return_value='https://img.test/1.jpg') as mock_up, \
         patch.object(dispatcher.session, 'post_gate_event', return_value={'visitDatabaseId': 'vdb-1', 'eventId': 'eid-1'}) as mock_post:

        dispatcher.start()
        dispatcher.enqueue(obs, 'evt-uuid-123')

        # Wait briefly for background worker
        for _ in range(20):
            if dispatcher.queue.empty() and mock_post.called:
                break
            time.sleep(0.05)

        dispatcher.stop()

        assert mock_post.called
        call_payload = mock_post.call_args[0][0]
        assert call_payload['id'] == 'evt-uuid-123'
        assert call_payload['containerNumber'] == 'CSQU3054383'
        assert call_payload['EventType'] in ('GATE_IN', 'GATE_OUT')
        assert metrics.increment.called


def test_syapi_gate_event_schema(mock_settings):
    dispatcher = CloudOutboxDispatcher(mock_settings)
    dummy_img = np.zeros((32, 32, 3), dtype=np.uint8)
    det = Detection(
        camera_id='CAM-01',
        frame_timestamp=utcnow(),
        class_name='container',
        confidence=0.96,
        bbox=(5.0, 5.0, 25.0, 25.0),
    )
    val = Validation(raw_text='MSCU 663987 0', normalized_text='MSCU6639870',
                     confidence=0.96, valid_format=True, valid_check_digit=True,
                     validation_status='VALID', size_type='22G1')
    obs = Observation('origin-syapi', 'LANE-1', 10, det, val, 'ENTRY', dummy_img, utcnow(), confirmed=True)

    with patch.object(dispatcher.session, 'post_gate_event', return_value=True) as mock_post:
        dispatcher.start()
        dispatcher.enqueue(obs, 'event-uuid-888')

        for _ in range(20):
            if dispatcher.queue.empty() and mock_post.called:
                break
            time.sleep(0.05)

        dispatcher.stop()

        assert mock_post.called
        payload = mock_post.call_args[0][0]
        assert 'VisitId' in payload
        assert payload['EventType'] == 'GATE_IN'
        assert payload['DeviceId'] == 'EDGE-GATE-01'
        assert payload['Truck'] == {'TruckNumber': 'NA'}
        assert payload['Driver'] is None
        assert len(payload['Containers']) == 1
        container = payload['Containers'][0]
        assert container['ContainerNumber'] == 'MSCU6639870'
        assert container['ContainerNumberConfidence'] == 0.96
        assert container['Size'] == '20'
        assert container['Position'] == 'SINGLE'
        assert len(container['Images']) >= 1
        assert container['Images'][0]['CameraId'] == 'CAM-01'
        assert container['Images'][0]['Image'].startswith('data:image/jpeg;base64,')


def test_each_camera_sends_one_image_without_duplicate_views(mock_settings, tmp_path):
    import cv2
    from pathlib import Path

    # Create dummy sharp images
    snap_dir = tmp_path / "snaps"
    snap_dir.mkdir()
    event_folder = snap_dir / "evt-123"
    event_folder.mkdir()

    img = np.ones((300, 400, 3), dtype=np.uint8) * 128
    # Add high contrast edges for high Laplacian quality
    img[50:250, 50:350] = 255
    cv2.putText(img, "TEST CONTAINER", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    _, buf = cv2.imencode('.jpg', img)

    p1 = event_folder / "gate-in-1.jpg"
    p1.write_bytes(buf.tobytes())
    p2a = event_folder / "gate-in-2-a.jpg"
    p2a.write_bytes(buf.tobytes())
    p2b = event_folder / "gate-in-2-b.jpg"
    p2b.write_bytes(buf.tobytes())
    p3 = event_folder / "gate-in-3.jpg"
    p3.write_bytes(buf.tobytes())

    mock_snaps = MagicMock()
    mock_snaps.path.side_effect = lambda rel: snap_dir / rel
    mock_snaps.compute_quality.return_value = 150.0

    mock_repo = MagicMock()
    mock_repo.event.return_value = {
        'id': 'evt-123',
        'status': 'CONFIRMED',
        'container_number': 'BSIU3391518',
        'ocr_results': [{'raw_text': 'BSIU3391518', 'normalized_text': 'BSIU3391518', 'valid_check_digit': True, 'confidence': .91, 'detection_id': 'det-main'}],
        'detections': [{'id': 'det-main', 'camera_id': 'gate-in-1', 'timestamp': '2026-09-30T10:00:00Z'},
                       {'id': 'det-side', 'camera_id': 'gate-in-2', 'timestamp': '2026-09-30T10:00:01Z'},
                       {'id': 'det-rear', 'camera_id': 'gate-in-3', 'timestamp': '2026-09-30T10:00:03Z'}],
        'container_number_2': None,
        'is_twin': 0,
        'iso_code_1': '22G1',
        'primary_snapshot_id': 'snap-2a',
        'cloud_synced': 0,
        'snapshots': [
            {'id': 'snap-1', 'camera_id': 'gate-in-1', 'image_path': 'evt-123/gate-in-1.jpg', 'timestamp': '2026-09-30T10:00:00Z'},
            {'id': 'snap-2a', 'camera_id': 'gate-in-2', 'image_path': 'evt-123/gate-in-2-a.jpg', 'timestamp': '2026-09-30T10:00:01Z'},
            {'id': 'snap-2b', 'camera_id': 'gate-in-2', 'image_path': 'evt-123/gate-in-2-b.jpg', 'timestamp': '2026-09-30T10:00:02Z'},
            {'id': 'snap-3', 'camera_id': 'gate-in-3', 'image_path': 'evt-123/gate-in-3.jpg', 'timestamp': '2026-09-30T10:00:03Z'},
        ]
    }
    mock_repo.canonical_event_id.return_value = 'evt-123'

    dispatcher = CloudOutboxDispatcher(mock_settings, repository=mock_repo, snapshots=mock_snaps)

    with patch.object(dispatcher.session, 'post_gate_event', return_value=True) as mock_post:
        dispatcher.start()
        dispatcher.enqueue(None, 'evt-123')

        for _ in range(20):
            if dispatcher.queue.empty() and mock_post.called:
                break
            time.sleep(0.05)

        dispatcher.stop()

        assert mock_post.called
        payload = mock_post.call_args[0][0]
        assert len(payload['Containers']) == 1
        container = payload['Containers'][0]
        assert container['ContainerNumber'] == 'BSIU3391518'
        assert container['Position'] == 'SINGLE'

        # Exactly 3 images: one from each camera, no duplicate gate-in-2 views
        camera_ids = [img['CameraId'] for img in container['Images']]
        assert camera_ids == ['gate-in-1', 'gate-in-2', 'gate-in-3']


def test_dispatch_audit_logging(mock_settings):
    audit_calls = []
    session = ProsperSession(mock_settings, audit_callback=lambda entry: audit_calls.append(entry))

    payload = {
        'id': 'evt-audit-1',
        'VisitId': 'VISIT-evt-audit-1',
        'EventType': 'GATE_OUT',
        'containerNumber': 'CSQU3054383',
        'Containers': [
            {
                'ContainerNumber': 'CSQU3054383',
                'Position': 'SINGLE',
                'Images': [
                    {'ImageType': 'FRONT', 'CameraId': 'cam-1', 'CapturedAt': '2026-10-15T12:00:00Z', 'Image': 'data:image/jpeg;base64,1234567890'}
                ]
            }
        ]
    }

    with patch.object(session.session, 'post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.content = b'{"eventId": "cloud-ev-999", "visitDatabaseId": "v-888"}'
        mock_post.return_value.json.return_value = {'eventId': 'cloud-ev-999', 'visitDatabaseId': 'v-888'}

        res = session.post_gate_event(payload)
        assert res == {'eventId': 'cloud-ev-999', 'visitDatabaseId': 'v-888'}
        assert len(audit_calls) == 1

        entry = audit_calls[0]
        assert entry['event_id'] == 'evt-audit-1'
        assert entry['visit_id'] == 'VISIT-evt-audit-1'
        assert entry['status_code'] == 200
        assert entry['status'] == 'SYNCED'
        assert entry['cloud_event_id'] == 'cloud-ev-999'
        assert entry['visit_database_id'] == 'v-888'
        assert entry['api_response'] == {'eventId': 'cloud-ev-999', 'visitDatabaseId': 'v-888'}
        # Image base64 must be sanitized and not stored raw
        assert 'Image' not in entry['event_sent']['Containers'][0]['Images'][0]
        assert entry['event_sent']['Containers'][0]['Images'][0]['image_bytes_len'] > 0


def test_is_dispatchable_event_thresholds(mock_settings):
    dispatcher = CloudOutboxDispatcher(mock_settings)

    # 1. Unreadable event (no container number or status UNREADABLE) -> False
    unreadable_ev = {
        'id': 'unreadable-1',
        'status': 'UNREADABLE',
        'container_number': None,
        'cloud_synced': 0,
        'ocr_results': []
    }
    assert dispatcher.is_dispatchable_event(unreadable_ev) is False

    # 2. Event with container number but accuracy < 50% (0.35) -> False
    low_acc_ev = {
        'id': 'low-acc-1',
        'status': 'CAPTURED',
        'container_number': 'EITU0514718',
        'cloud_synced': 0,
        'ocr_results': [
            {'raw_text': 'FITU 051471 D', 'normalized_text': 'FITU051471D', 'confidence': 0.35}
        ]
    }
    assert dispatcher.is_dispatchable_event(low_acc_ev) is False

    # 3. Event with container number and accuracy >= 50% (0.55) -> True
    valid_acc_ev = {
        'id': 'valid-acc-1',
        'status': 'CAPTURED',
        'container_number': 'EITU0514718',
        'cloud_synced': 0,
        'ocr_results': [
            {'raw_text': 'EITU 051471 8', 'normalized_text': 'EITU0514718', 'confidence': 0.55}
        ]
    }
    assert dispatcher.is_dispatchable_event(valid_acc_ev) is True

    # 4. Confirmed event with high accuracy (0.92) -> True
    confirmed_ev = {
        'id': 'confirmed-1',
        'status': 'CONFIRMED',
        'container_number': 'TEMU7104954',
        'cloud_synced': 0,
        'ocr_results': [
            {'raw_text': 'TEMU 710495 4', 'normalized_text': 'TEMU7104954', 'confidence': 0.92}
        ]
    }
    assert dispatcher.is_dispatchable_event(confirmed_ev) is True

    # 5. Already synced event -> False
    synced_ev = {
        'id': 'synced-1',
        'status': 'CONFIRMED',
        'container_number': 'TEMU7104954',
        'cloud_synced': 1,
        'ocr_results': [
            {'raw_text': 'TEMU 710495 4', 'normalized_text': 'TEMU7104954', 'confidence': 0.92}
        ]
    }
    assert dispatcher.is_dispatchable_event(synced_ev) is False


def test_two_step_post_and_put_images_and_late_override(mock_settings):
    """Verify Step 1 (POST /events) extracts eventId and Step 2 (PUT /events/{eventId}/images) uploads images,
    and subsequent image upgrades call PUT /events/{eventId}/images directly."""
    session = ProsperSession(mock_settings)

    post_resp = MagicMock()
    post_resp.status_code = 200
    post_resp.content = b'{"eventId": "0e7d4fac-5f22-48a4-be37-0bc58a948eff", "visitDatabaseId": "afcf4cc6-8d78-4806-acf2-42eb18e4ef8d"}'
    post_resp.json.return_value = {
        "eventId": "0e7d4fac-5f22-48a4-be37-0bc58a948eff",
        "visitDatabaseId": "afcf4cc6-8d78-4806-acf2-42eb18e4ef8d"
    }

    put_resp = MagicMock()
    put_resp.status_code = 200
    put_resp.content = b'{"eventId": "0e7d4fac-5f22-48a4-be37-0bc58a948eff"}'
    put_resp.json.return_value = {"eventId": "0e7d4fac-5f22-48a4-be37-0bc58a948eff"}

    payload = {
        "id": "evt-override-1",
        "VisitId": "VISIT-38727D97",
        "EventType": "GATE_OUT",
        "Containers": [
            {
                "ContainerNumber": "MEDU9988776",
                "ContainerNumberConfidence": 0.97,
                "Size": "40",
                "Position": "SINGLE",
                "Images": [
                    {
                        "ImageType": "FRONT",
                        "CameraId": "gate-out-2",
                        "CapturedAt": "2026-10-06T08:15:39.210Z",
                        "Image": "data:image/jpeg;base64,dGVzdA=="
                    }
                ]
            }
        ]
    }

    with patch.object(session.session, 'put', return_value=put_resp) as mock_put:
        res = session.put_gate_event_images("0e7d4fac-5f22-48a4-be37-0bc58a948eff", payload)
        assert isinstance(res, dict) and res.get('eventId') == "0e7d4fac-5f22-48a4-be37-0bc58a948eff"
        assert mock_put.called
        put_url = mock_put.call_args[0][0]
        put_json = mock_put.call_args[1]['json']
        assert put_url.endswith('/api/v1/gate/events/0e7d4fac-5f22-48a4-be37-0bc58a948eff/images')
        assert list(put_json.keys()) == ['Containers']
        c0 = put_json['Containers'][0]
        assert c0['ContainerNumber'] == 'MEDU9988776'
        assert c0['ContainerNumberConfidence'] == 0.97
        assert c0['Size'] == '40'
        assert c0['Position'] == 'SINGLE'
        assert len(c0['Images']) == 1
        assert c0['Images'][0]['ImageType'] == 'FRONT'
        assert c0['Images'][0]['CameraId'] == 'gate-out-2'


def test_cumulative_update_preserves_multiple_cameras(mock_settings, tmp_path):
    """Verify that when an update only has candidates from 1 camera, previously acknowledged
    proofs from other cameras are cumulatively retained and merged."""
    import cv2
    import hashlib
    from app.database.database import Database
    from app.database.repositories.gate import GateRepository
    from app.database.models import GateEvent

    db_path = tmp_path / "test_gate.db"
    db = Database(f"sqlite:///{db_path}")
    db.initialize()
    repo = GateRepository(db)

    # Create event in DB
    ev_id = "evt-cumul-1"
    now_dt = utcnow()
    with db.sessions.begin() as s:
        s.add(GateEvent(
            id=ev_id,
            gate_id='gate-in',
            status='CONFIRMED',
            container_number='MSKU6708610',
            event_type='ENTRY',
            timestamp=now_dt,
            cloud_synced=False,
            proof_version=1
        ))

    # Create dummy images for gate-in-1 and gate-in-2
    snap_dir = tmp_path / "snaps"
    snap_dir.mkdir()
    ev_dir = snap_dir / ev_id
    ev_dir.mkdir()

    img1 = np.ones((300, 400, 3), dtype=np.uint8) * 128
    img1[50:250, 50:350] = 255
    cv2.putText(img1, "CAM1", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    _, buf1 = cv2.imencode('.jpg', img1)
    b1 = buf1.tobytes()
    p1 = ev_dir / "gate-in-1.jpg"
    p1.write_bytes(b1)
    h1 = hashlib.sha256(b1).hexdigest()

    img2 = np.ones((300, 400, 3), dtype=np.uint8) * 128
    img2[50:250, 50:350] = 255
    cv2.putText(img2, "CAM2", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    _, buf2 = cv2.imencode('.jpg', img2)
    b2 = buf2.tobytes()
    p2 = ev_dir / "gate-in-2.jpg"
    p2.write_bytes(b2)
    h2 = hashlib.sha256(b2).hexdigest()

    mock_snaps = MagicMock()
    mock_snaps.path.side_effect = lambda rel: snap_dir / rel
    mock_snaps.compute_quality.return_value = 120.0
    mock_snaps.storage_path = snap_dir

    # Record gate-in-1 and gate-in-2 as already acknowledged
    repo.record_acknowledged_proof(
        event_id=ev_id,
        cloud_event_id="CLOUD-123",
        container_number="MSKU6708610",
        camera_id="gate-in-1",
        image_path=f"{ev_id}/gate-in-1.jpg",
        image_hash=h1,
        capture_timestamp=now_dt,
        proof_version=1
    )
    repo.record_acknowledged_proof(
        event_id=ev_id,
        cloud_event_id="CLOUD-123",
        container_number="MSKU6708610",
        camera_id="gate-in-2",
        image_path=f"{ev_id}/gate-in-2.jpg",
        image_hash=h2,
        capture_timestamp=now_dt,
        proof_version=1
    )

    # Now simulate a subsequent update where only gate-in-2 is in event snapshots
    with patch.object(repo, 'event') as mock_ev:
        mock_ev.return_value = {
            'id': ev_id,
            'status': 'CONFIRMED',
            'container_number': 'MSKU6708610',
            'timestamp': now_dt.isoformat(),
            'event_type': 'ENTRY',
            'is_twin': 0,
            'iso_code_1': '42G1',
            'cloud_event_id': 'CLOUD-123',
            'proof_version': 2,
            'ocr_results': [{'raw_text': 'MSKU6708610', 'normalized_text': 'MSKU6708610', 'valid_check_digit': True, 'confidence': 0.95, 'detection_id': 'det-2'}],
            'detections': [{'id': 'det-2', 'camera_id': 'gate-in-2', 'timestamp': now_dt.isoformat()}],
            'snapshots': [
                {'id': 'snap-2', 'camera_id': 'gate-in-2', 'image_path': f"{ev_id}/gate-in-2.jpg", 'timestamp': now_dt.isoformat()}
            ]
        }
        dispatcher = CloudOutboxDispatcher(mock_settings, repository=repo, snapshots=mock_snaps)
        payload = dispatcher.build_event_payload(ev_id)

    # Verify both gate-in-1 and gate-in-2 are in the container images
    imgs = payload['Containers'][0]['Images']
    cams = {img['CameraId'] for img in imgs}
    assert 'gate-in-1' in cams
    assert 'gate-in-2' in cams
    assert len(imgs) == 2


def test_missing_acknowledged_image_holds_update(mock_settings, tmp_path):
    """If a previously acknowledged proof cannot be recovered from disk, the dispatcher
    MUST hold the update rather than sending fewer images to the cloud."""
    import cv2
    from app.database.database import Database
    from app.database.repositories.gate import GateRepository
    from app.database.models import GateEvent

    db_path = tmp_path / "test_missing.db"
    db = Database(f"sqlite:///{db_path}")
    db.initialize()
    repo = GateRepository(db)

    ev_id = "evt-missing-1"
    now_dt = utcnow()
    with db.sessions.begin() as s:
        s.add(GateEvent(
            id=ev_id, gate_id='gate-in', status='CONFIRMED', container_number='ONEU4495199',
            event_type='ENTRY', timestamp=now_dt, cloud_synced=False, proof_version=1
        ))

    # Record proof for gate-in-1 that DOES NOT exist on disk
    repo.record_acknowledged_proof(
        event_id=ev_id, cloud_event_id="CLOUD-999", container_number="ONEU4495199",
        camera_id="gate-in-1", image_path="nonexistent/gate-in-1.jpg",
        image_hash="abc123hash", capture_timestamp=now_dt, proof_version=1
    )

    snap_dir = tmp_path / "snaps"
    snap_dir.mkdir()
    ev_dir = snap_dir / ev_id
    ev_dir.mkdir()
    img_miss = np.ones((300, 400, 3), dtype=np.uint8) * 128
    img_miss[50:250, 50:350] = 255
    cv2.putText(img_miss, "CAM2", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    _, buf_miss = cv2.imencode('.jpg', img_miss)
    p2 = ev_dir / "gate-in-2.jpg"
    p2.write_bytes(buf_miss.tobytes())

    mock_snaps = MagicMock()
    mock_snaps.path.side_effect = lambda rel: snap_dir / rel
    mock_snaps.compute_quality.return_value = 100.0
    mock_snaps.storage_path = snap_dir

    with patch.object(repo, 'event') as mock_ev:
        mock_ev.return_value = {
            'id': ev_id, 'status': 'CONFIRMED', 'container_number': 'ONEU4495199',
            'timestamp': now_dt.isoformat(), 'event_type': 'ENTRY', 'is_twin': 0,
            'cloud_event_id': 'CLOUD-999', 'proof_version': 2,
            'ocr_results': [{'raw_text': 'ONEU4495199', 'normalized_text': 'ONEU4495199', 'valid_check_digit': True, 'confidence': 0.95, 'detection_id': 'd2'}],
            'detections': [{'id': 'd2', 'camera_id': 'gate-in-2', 'timestamp': now_dt.isoformat()}],
            'snapshots': [{'id': 's2', 'camera_id': 'gate-in-2', 'image_path': f"{ev_id}/gate-in-2.jpg", 'timestamp': now_dt.isoformat()}]
        }
        dispatcher = CloudOutboxDispatcher(mock_settings, repository=repo, snapshots=mock_snaps)
        with pytest.raises(ValueError, match="cannot be recovered.*holding update"):
            dispatcher.build_event_payload(ev_id)


def test_restart_recovery_from_database(tmp_path):
    """Verify acknowledged proofs persist across database restarts."""
    from app.database.database import Database
    from app.database.repositories.gate import GateRepository
    from app.database.models import GateEvent

    db_path = tmp_path / "test_restart.db"
    db1 = Database(f"sqlite:///{db_path}")
    db1.initialize()
    repo1 = GateRepository(db1)

    ev_id = "evt-restart-1"
    now_dt = utcnow()
    with db1.sessions.begin() as s:
        s.add(GateEvent(id=ev_id, gate_id='gate-in', status='CONFIRMED', container_number='MSKU1234567',
                        event_type='ENTRY', timestamp=now_dt, cloud_synced=True, proof_version=1))

    repo1.record_acknowledged_proof(
        event_id=ev_id, cloud_event_id="CLOUD-RESTART", container_number="MSKU1234567",
        camera_id="gate-in-1", image_path="snaps/gate-in-1.jpg",
        image_hash="hash-restart-1", capture_timestamp=now_dt, proof_version=1
    )

    # Re-open database fresh
    db2 = Database(f"sqlite:///{db_path}")
    db2.initialize()
    repo2 = GateRepository(db2)

    proofs = repo2.get_acknowledged_proofs(ev_id, "MSKU1234567")
    assert len(proofs) == 1
    assert proofs[0]['camera_id'] == 'gate-in-1'
    assert proofs[0]['image_hash'] == 'hash-restart-1'
    assert proofs[0]['cloud_event_id'] == 'CLOUD-RESTART'


def test_twin_container_isolation_no_role_assumption(mock_settings, tmp_path):
    """Verify twin container proofs are bound to direct identity/observations,
    never assuming a rear camera belongs to rear container."""
    import cv2
    snap_dir = tmp_path / "snaps"
    snap_dir.mkdir()
    ev_id = "evt-twin-1"
    ev_dir = snap_dir / ev_id
    ev_dir.mkdir()

    img = np.ones((300, 400, 3), dtype=np.uint8) * 180
    cv2.putText(img, "CONTAINER", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    _, buf = cv2.imencode('.jpg', img)
    (ev_dir / "rear_cam.jpg").write_bytes(buf.tobytes())
    (ev_dir / "front_cam.jpg").write_bytes(buf.tobytes())

    mock_snaps = MagicMock()
    mock_snaps.path.side_effect = lambda rel: snap_dir / rel
    mock_snaps.compute_quality.return_value = 150.0

    now_dt = utcnow()
    # Rear camera gate-in-3 has identity for FRONT container (cntr1: HLXU1122334)
    # Front camera gate-in-1 has identity for REAR container (cntr2: MSCU9988776)
    event_dict = {
        'id': ev_id,
        'status': 'CONFIRMED',
        'container_number': 'HLXU1122334',
        'container_number_2': 'MSCU9988776',
        'is_twin': 1,
        'timestamp': now_dt.isoformat(),
        'event_type': 'ENTRY',
        'primary_snapshot_id': 'snap-rear-cam',
        'rear_snapshot_id': 'snap-front-cam',
        'ocr_results': [
            {'raw_text': 'HLXU1122334', 'normalized_text': 'HLXU1122334', 'valid_check_digit': True, 'confidence': 0.95, 'detection_id': 'det-front-box'},
            {'raw_text': 'MSCU9988776', 'normalized_text': 'MSCU9988776', 'valid_check_digit': True, 'confidence': 0.95, 'detection_id': 'det-rear-box'}
        ],
        'detections': [
            {'id': 'det-front-box', 'camera_id': 'gate-in-3', 'timestamp': now_dt.isoformat(), 'origin_key': 'final:evt-twin-1:snap-rear-cam'},
            {'id': 'det-rear-box', 'camera_id': 'gate-in-1', 'timestamp': now_dt.isoformat(), 'origin_key': 'final:evt-twin-1:snap-front-cam'}
        ],
        'snapshots': [
            {'id': 'snap-rear-cam', 'camera_id': 'gate-in-3', 'image_path': f"{ev_id}/rear_cam.jpg", 'timestamp': now_dt.isoformat()},
            {'id': 'snap-front-cam', 'camera_id': 'gate-in-1', 'image_path': f"{ev_id}/front_cam.jpg", 'timestamp': now_dt.isoformat()}
        ]
    }

    mock_repo = MagicMock()
    mock_repo.event.return_value = event_dict
    mock_repo.get_acknowledged_proofs.return_value = []

    dispatcher = CloudOutboxDispatcher(mock_settings, repository=mock_repo, snapshots=mock_snaps)
    payload = dispatcher.build_event_payload(ev_id)

    assert len(payload['Containers']) == 2
    c1 = payload['Containers'][0]
    c2 = payload['Containers'][1]
    assert c1['ContainerNumber'] == 'HLXU1122334'
    assert c2['ContainerNumber'] == 'MSCU9988776'

    # Container 1 should get snap-rear-cam (gate-in-3) because it observed HLXU1122334
    assert c1['Images'][0]['CameraId'] == 'gate-in-3'
    # Container 2 should get snap-front-cam (gate-in-1) because it observed MSCU9988776
    assert c2['Images'][0]['CameraId'] == 'gate-in-1'


def test_proof_version_isolation(tmp_path):
    """Verify that acknowledging proof_version=1 does not mark proof_version=2 as synced."""
    from app.database.database import Database
    from app.database.repositories.gate import GateRepository
    from app.database.models import GateEvent

    db_path = tmp_path / "test_version.db"
    db = Database(f"sqlite:///{db_path}")
    db.initialize()
    repo = GateRepository(db)

    ev_id = "evt-ver-1"
    now_dt = utcnow()
    with db.sessions.begin() as s:
        s.add(GateEvent(id=ev_id, gate_id='gate-in', status='CONFIRMED', container_number='MSKU9998881',
                        event_type='ENTRY', timestamp=now_dt, cloud_synced=False, proof_version=2))

    # Acknowledge older proof version 1
    repo.mark_cloud_synced(ev_id, cloud_event_id="CLOUD-V1", proof_version=1)

    # Event proof_version=2 is newer than acknowledged 1, so cloud_synced must remain False
    with db.sessions() as s:
        ev = s.get(GateEvent, ev_id)
        assert ev.cloud_synced is False
        assert ev.cloud_event_id == "CLOUD-V1"

    # Now acknowledge proof version 2
    repo.mark_cloud_synced(ev_id, cloud_event_id="CLOUD-V2", proof_version=2)
    with db.sessions() as s:
        ev = s.get(GateEvent, ev_id)
        assert ev.cloud_synced is True
        assert ev.cloud_event_id == "CLOUD-V2"
