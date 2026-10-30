from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import pytest

from app.events.cloud_dispatcher import CloudOutboxDispatcher, ProsperSession
from app.events.proof_updater import CloudProofUpgrader
from app.snapshots.outgoing_preview import OutgoingPreview
from app.snapshots.proof_revision import ProofRevision
from tests.test_pipeline_contract import overview_scene
from app.events.capture_planner import plan_captures


@pytest.fixture
def approved(settings, repository, snapshots, make_job):
    controller, cameras, group, frames, workers = overview_scene(settings, snapshots, make_job)
    repository.sync_cameras(cameras)
    sharp = frames['camera-1'].image.copy()
    for frame in frames.values():
        frame.image[:] = cv2.GaussianBlur(frame.image, (7, 7), 0)
    controller.observe(cameras[0], frames['camera-1'], group, cameras, workers, [])
    plan = plan_captures(group[:1], 2)[0]
    plan.regions[0].best_any[1].passage_metadata['shared_visit_folder'] = True
    event_id = repository.save_passage(plan, snapshots, 2)
    dispatcher = CloudOutboxDispatcher(settings, repository=repository, snapshots=snapshots)
    payload = dispatcher.build_event_payload(event_id)
    repository.mark_cloud_synced(event_id, 'cloud-existing', proof_version=1)
    session = Mock()
    session.put_gate_event_images.return_value = {'eventId': 'cloud-existing'}
    updater = CloudProofUpgrader(settings, repository, snapshots, dispatcher, session=session)
    updater._record_payload_proofs(event_id, 'cloud-existing', payload, 1)
    runtime = SimpleNamespace(settings=settings, model_status={'detector': 'READY'},
        _overview_visibility=Mock(return_value=True), _proof_position_matches=Mock(return_value=True))
    service = ProofRevision(repository, snapshots, runtime)
    path = snapshots.save_side_snapshot('visit-a', 'clearer-proof', sharp)
    snapshot_id = repository.add_or_update_snapshot(event_id, 'camera-1', path, timestamp=frames['camera-1'].timestamp)
    return SimpleNamespace(settings=settings, repo=repository, snapshots=snapshots, event_id=event_id,
        dispatcher=dispatcher, updater=updater, session=session, service=service, payload=payload,
        snapshot_id=snapshot_id, path=path, frames=frames, runtime=runtime)


def test_approved_revision_is_the_exact_ui_and_put_payload(approved):
    a = approved
    result = a.service.approve(a.event_id, a.snapshot_id, 'Reviewed clearer same-position frame')
    assert result['proof_version'] == 2
    assert not a.repo.event(a.event_id)['cloud_synced']
    shown = a.dispatcher.build_proof_preview(a.event_id)
    a.updater._integrate_candidate_image = Mock(side_effect=AssertionError('Must not select live OCR image'))
    response = a.updater.evaluate_and_upgrade(a.event_id, candidate_image=object(), camera_id='camera-2', force=True)
    assert response['status'] == 'UPGRADED'
    assert a.session.put_gate_event_images.call_args.args == ('cloud-existing', shown)
    old = a.payload['Containers'][0]['Images']
    new = shown['Containers'][0]['Images']
    assert [i['Image'] for i in old[1:]] == [i['Image'] for i in new[1:]]
    assert old[0]['Image'] != new[0]['Image']
    assert OutgoingPreview(a.snapshots.root).path(a.event_id).with_name(a.event_id + '.before-v2.json').exists()
    assert a.repo.event(a.event_id)['cloud_synced']
    assert {p['proof_version'] for p in a.repo.get_acknowledged_proofs(a.event_id)} == {2}


def test_local_only_allows_review_but_never_puts(approved):
    a = approved
    a.settings.local_test_mode = True
    assert a.service.approve(a.event_id, a.snapshot_id, 'Local review')['proof_version'] == 2
    assert a.updater.evaluate_and_upgrade(a.event_id)['status'] == 'LOCAL_ONLY'
    a.updater.enqueue_upgrade(a.event_id)
    assert a.updater._queue.empty()
    a.session.put_gate_event_images.assert_not_called()


def test_late_ocr_frame_is_not_a_fixed_position_replacement(approved):
    a = approved
    snap = a.repo.snapshot(a.snapshot_id)
    from app.database.models import Snapshot
    with a.repo.db.sessions.begin() as session:
        row = session.get(Snapshot, a.snapshot_id)
        row.timestamp += timedelta(seconds=5)
    with pytest.raises(ValueError, match='capture time'):
        a.service.approve(a.event_id, a.snapshot_id, 'Different moment')
    assert a.dispatcher.build_proof_preview(a.event_id) == a.payload
    a.session.put_gate_event_images.assert_not_called()


@pytest.mark.parametrize('response', ['OFFLINE', 'CONFLICT', 'REJECTED', {'success': False}])
def test_failure_does_not_acknowledge_new_revision(approved, response):
    a = approved
    a.service.approve(a.event_id, a.snapshot_id, 'Reviewed')
    a.session.put_gate_event_images.return_value = response
    assert not a.updater.evaluate_and_upgrade(a.event_id)['success']
    assert not a.repo.event(a.event_id)['cloud_synced']
    assert {p['proof_version'] for p in a.repo.get_acknowledged_proofs(a.event_id)} == {1}


def test_retry_reuses_approved_revision_without_changing_images(approved):
    a = approved
    a.service.approve(a.event_id, a.snapshot_id, 'Reviewed')
    a.session.put_gate_event_images.side_effect = ['OFFLINE', True]
    a.updater.evaluate_and_upgrade(a.event_id)
    assert a.updater.evaluate_and_upgrade(a.event_id)['proof_version'] == 2
    assert a.session.put_gate_event_images.call_args_list[0] == a.session.put_gate_event_images.call_args_list[1]


def test_unchanged_proof_is_noop_even_with_force(approved):
    a = approved
    assert a.updater.evaluate_and_upgrade(a.event_id, force=True)['status'] == 'NOOP'
    a.session.put_gate_event_images.assert_not_called()


def test_put_does_not_guess_unknown_size_or_round_confidence(approved):
    a = approved
    session = ProsperSession(a.settings)
    session.get_auth_headers = Mock(return_value={'Authorization': 'Bearer test'})
    session.session.put = Mock(return_value=SimpleNamespace(status_code=204, content=b''))
    payload = deepcopy(a.payload)
    payload['Containers'][0]['Size'] = None
    payload['Containers'][0]['ContainerNumberConfidence'] = .9234567
    session.put_gate_event_images('cloud-existing', payload)
    sent = session.session.put.call_args.kwargs['json']['Containers'][0]
    assert sent['Size'] is None and sent['ContainerNumberConfidence'] == .9234567
    assert [i['Image'] for i in sent['Images']] == [i['Image'] for i in payload['Containers'][0]['Images']]


def test_source_without_verified_container_cannot_be_approved(approved):
    a = approved
    a.runtime._overview_visibility.return_value = False
    with pytest.raises(ValueError, match='not verified'):
        a.service.approve(a.event_id, a.snapshot_id, 'Reviewed')
    assert a.dispatcher.build_proof_preview(a.event_id) == a.payload


def test_stale_review_cannot_overwrite_a_new_revision(approved):
    a = approved
    store = OutgoingPreview(a.snapshots.root)
    old_digest = store.digest(a.payload)
    a.service.approve(a.event_id, a.snapshot_id, 'First review')
    with pytest.raises(ValueError, match='changed during review'):
        store.revise_camera(a.event_id, 'camera-1', a.snapshots.path(a.path).read_bytes(),
            a.frames['camera-1'].timestamp.isoformat()+'Z', a.path, 'Stale review', expected_digest=old_digest)


def test_newer_revision_during_put_stays_pending(approved):
    a = approved
    a.service.approve(a.event_id, a.snapshot_id, 'Second version')
    def response(*_):
        OutgoingPreview(a.snapshots.root).revise_camera(a.event_id, 'camera-1',
            a.snapshots.path(a.path).read_bytes(), a.frames['camera-1'].timestamp.isoformat()+'Z',
            a.path, 'Third version approved while PUT was running')
        return True
    a.session.put_gate_event_images.side_effect = response
    assert a.updater.evaluate_and_upgrade(a.event_id)['status'] == 'NEWER_REVISION_PENDING'
    event = a.repo.event(a.event_id)
    assert event['proof_version'] == 3 and not event['cloud_synced']
    assert {p['proof_version'] for p in a.repo.get_acknowledged_proofs(a.event_id)} == {2}


def test_approved_set_cannot_shrink_previous_camera_coverage(approved):
    a = approved
    smaller = deepcopy(a.payload)
    smaller['Containers'][0]['Images'].pop()
    smaller['ProofVersion'] = 2
    with patch.object(OutgoingPreview, 'load', return_value=smaller):
        assert a.updater.evaluate_and_upgrade(a.event_id)['status'] == 'PROOF_COVERAGE_REGRESSION'
    a.session.put_gate_event_images.assert_not_called()


def test_put_rejects_duplicate_camera_images(approved):
    a = approved
    session = ProsperSession(a.settings)
    session.get_auth_headers = Mock(return_value={'Authorization': 'Bearer test'})
    session.session.put = Mock()
    payload = deepcopy(a.payload)
    payload['Containers'][0]['Images'].append(payload['Containers'][0]['Images'][0])
    assert session.put_gate_event_images('cloud-existing', payload) == 'REJECTED'
    session.session.put.assert_not_called()


def test_local_api_approval_refreshes_ui_without_upload(approved):
    from fastapi.testclient import TestClient
    from app.main import create_app
    a = approved
    a.settings.local_test_mode = True
    cameras = a.settings.cameras()
    # These three test camera IDs match the source snapshot records.
    from app.config.settings import CameraConfig
    cameras = [CameraConfig(id=f'camera-{i}', name=f'Camera {i}', gate_id='lane-1',
        source_env=f'CAMERA_{i}_RTSP', direction='ENTRY') for i in (1,2,3)]
    app = create_app(a.settings, a.repo.db, cameras)
    with TestClient(app) as client:
        app.state.runtime.model_status['detector'] = 'READY'
        app.state.runtime._overview_visibility = Mock(return_value=True)
        app.state.runtime._proof_position_matches = Mock(return_value=True)
        response = client.post('/api/gate-events/' + a.event_id + '/proof-revisions',
            json={'snapshot_id': a.snapshot_id, 'reason': 'Reviewed clearer fixed-position proof'})
        assert response.status_code == 200 and response.json()['proof_version'] == 2
        shown = client.get('/api/gate-events/' + a.event_id).json()
        assert shown['proof_frozen'] and not shown['cloud_synced']
        assert len(shown['cloud_proof_images']) == 3
        assert shown['local_test_mode']
        assert client.post('/api/gate-events/' + a.event_id + '/upgrade-proofs').json()['status'] == 'LOCAL_ONLY'


def test_nearby_time_does_not_allow_different_truck_position(approved):
    a = approved
    a.runtime._proof_position_matches.return_value = False
    with pytest.raises(ValueError, match='position differs'):
        a.service.approve(a.event_id, a.snapshot_id, 'Sharper but moved')
    assert a.dispatcher.build_proof_preview(a.event_id) == a.payload


def test_spatial_check_rejects_fast_truck_displacement(settings, camera, repository, snapshots, make_job):
    from app.runtime import GateRuntime
    from app.domain import Detection
    runtime = GateRuntime(settings, [camera], repository, snapshots)
    frame = make_job().frame
    runtime.detector = Mock()
    runtime.detector.detect_vehicles.return_value = []
    runtime.detector.detect.return_value = [Detection(camera.id, frame.timestamp, 'container', .95, (11,10,131,90))]
    assert runtime._proof_position_matches(frame, (10,10,130,90))
    runtime.detector.detect.return_value = [Detection(camera.id, frame.timestamp, 'container', .95, (20,10,140,90))]
    assert not runtime._proof_position_matches(frame, (10,10,130,90))
