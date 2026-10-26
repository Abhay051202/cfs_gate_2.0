"""
Unit tests for backend/app/events/proof_updater.py (CloudProofUpgrader).
"""

import base64
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest

from app.database.database import Database
from app.database.models import Base, GateEvent, Snapshot, utcnow
from app.database.repositories.gate import GateRepository
from app.events.proof_updater import CloudProofUpgrader
from app.snapshots.snapshot_manager import SnapshotManager


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_gate.db"
    db = Database(f"sqlite:///{db_file}")
    Base.metadata.create_all(db.engine)
    return db


@pytest.fixture
def repo(test_db):
    from app.database.models import Camera
    r = GateRepository(test_db)
    with r._write_lock, r.db.sessions.begin() as s:
        for cam_id in ("gate-in-1", "gate-in-2", "gate-in-3"):
            s.add(Camera(
                id=cam_id,
                name=cam_id,
                gate_id="lane-in",
                source_env=f"RTSP_{cam_id}",
                enabled=True,
                configuration={},
            ))
    return r


@pytest.fixture
def snapshot_mgr(tmp_path):
    mgr = SnapshotManager(tmp_path / "snapshots")
    mgr.categorized = True
    return mgr


@pytest.fixture
def mock_settings(tmp_path):
    s = MagicMock()
    s.snapshot_directory = str(tmp_path / "snapshots")
    s.cameras = [
        MagicMock(id="gate-in-1", role="FRONT_TOP"),
        MagicMock(id="gate-in-2", role="LEFT"),
        MagicMock(id="gate-in-3", role="REAR"),
    ]
    return s


def test_upgrader_skip_non_synced(repo, snapshot_mgr, mock_settings):
    """Events not yet synced or without cloud_event_id should be skipped."""
    event_id = str(uuid4())
    with repo._write_lock, repo.db.sessions.begin() as s:
        s.add(GateEvent(
            id=event_id,
            gate_id="lane-in",
            timestamp=utcnow(),
            container_number="MSCU1234567",
            event_type="ENTRY",
            status="CONFIRMED",
            cloud_synced=False,
            cloud_event_id=None,
        ))

    mock_session = MagicMock()
    upgrader = CloudProofUpgrader(mock_settings, repository=repo, snapshots=snapshot_mgr, session=mock_session)

    res = upgrader.evaluate_and_upgrade(event_id)
    assert res['success'] is False
    assert res['status'] == 'NOT_ELIGIBLE'
    assert not mock_session.put_gate_event_images.called


def test_upgrader_success_when_better_proof_arrives(repo, snapshot_mgr, mock_settings, tmp_path):
    """When a clearer/superior proof arrives for an already synced event, PUT /images is called."""
    event_id = str(uuid4())
    cloud_id = f"cloud-{uuid4()}"
    visit_id = f"visit-{uuid4()}"

    # 1. Create synced event in database
    with repo._write_lock, repo.db.sessions.begin() as s:
        s.add(GateEvent(
            id=event_id,
            gate_id="lane-in",
            timestamp=utcnow(),
            container_number="CAIU3834499",
            event_type="ENTRY",
            status="CONFIRMED",
            visit_id=visit_id,
            cloud_synced=True,
            cloud_event_id=cloud_id,
            proof_version=1,
        ))

    # 2. Add initial early snapshot (gate-in-1 approach)
    snap1_file = tmp_path / "snapshots" / "detected_in" / visit_id / "gate-in-1_context.jpg"
    snap1_file.parent.mkdir(parents=True, exist_ok=True)
    img1 = np.full((480, 640, 3), 100, dtype=np.uint8)
    import cv2
    cv2.imwrite(str(snap1_file), img1)
    repo.add_or_update_snapshot(event_id, "gate-in-1", f"detected_in/{visit_id}/gate-in-1_context.jpg")

    # Record initial acknowledged proof
    repo.record_acknowledged_proof(
        event_id=event_id,
        cloud_event_id=cloud_id,
        container_number="CAIU3834499",
        camera_id="gate-in-1",
        image_path=f"detected_in/{visit_id}/gate-in-1_context.jpg",
        image_hash=hashlib.sha256(b"dummy1").hexdigest(),
        capture_timestamp=utcnow(),
        proof_version=1,
    )

    # 3. New candidate frame arrives for gate-in-2 (crisp container proof)
    new_frame = np.full((720, 1280, 3), 180, dtype=np.uint8)
    cv2.rectangle(new_frame, (100, 100), (300, 300), (0, 0, 0), -1)

    mock_session = MagicMock()
    mock_session.put_gate_event_images.return_value = {"eventId": cloud_id, "uploadedImagesCount": 2}

    upgrader = CloudProofUpgrader(mock_settings, repository=repo, snapshots=snapshot_mgr, session=mock_session)

    res = upgrader.evaluate_and_upgrade(
        event_id=event_id,
        candidate_image=new_frame,
        camera_id="gate-in-2",
    )

    assert res['success'] is True
    assert res['status'] == 'UPGRADED'
    assert res['cloud_event_id'] == cloud_id
    assert res['proof_version'] == 2
    assert mock_session.put_gate_event_images.called

    call_args = mock_session.put_gate_event_images.call_args
    passed_cloud_id = call_args[0][0]
    payload = call_args[0][1]

    assert passed_cloud_id == cloud_id
    assert 'Containers' in payload
    c0 = payload['Containers'][0]
    assert c0['ContainerNumber'] == "CAIU3834499"

    # Verify CameraId uniqueness rule: No duplicate CameraId within the container
    camera_ids = [img['CameraId'] for img in c0['Images']]
    assert len(camera_ids) == len(set(camera_ids))
    assert "gate-in-2" in camera_ids

    # Verify DB updated
    ev_updated = repo.event(event_id)
    assert ev_updated['proof_version'] == 2

    acked = repo.get_acknowledged_proofs(event_id)
    assert len(acked) >= 2
    cams_in_ack = {a['camera_id'] for a in acked}
    assert "gate-in-2" in cams_in_ack


def test_upgrader_noop_when_proofs_match(repo, snapshot_mgr, mock_settings, tmp_path):
    """When proofs already match acknowledged hashes, return NOOP without network request."""
    event_id = str(uuid4())
    cloud_id = f"cloud-{uuid4()}"
    visit_id = f"visit-{uuid4()}"

    with repo._write_lock, repo.db.sessions.begin() as s:
        s.add(GateEvent(
            id=event_id,
            gate_id="lane-in",
            timestamp=utcnow(),
            container_number="MSCU9988776",
            event_type="ENTRY",
            status="CONFIRMED",
            visit_id=visit_id,
            cloud_synced=True,
            cloud_event_id=cloud_id,
            proof_version=1,
        ))

    snap_file = tmp_path / "snapshots" / "detected_in" / visit_id / "gate-in-1_proof.jpg"
    snap_file.parent.mkdir(parents=True, exist_ok=True)
    import cv2
    img = np.full((480, 640, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(snap_file), img)
    repo.add_or_update_snapshot(event_id, "gate-in-1", f"detected_in/{visit_id}/gate-in-1_proof.jpg")

    from app.snapshots.proof import proof_jpeg
    encoded = proof_jpeg(snap_file.read_bytes())
    img_hash = hashlib.sha256(encoded).hexdigest()

    repo.record_acknowledged_proof(
        event_id=event_id,
        cloud_event_id=cloud_id,
        container_number="MSCU9988776",
        camera_id="gate-in-1",
        image_path=f"detected_in/{visit_id}/gate-in-1_proof.jpg",
        image_hash=img_hash,
        capture_timestamp=utcnow(),
        proof_version=1,
    )

    mock_session = MagicMock()
    upgrader = CloudProofUpgrader(mock_settings, repository=repo, snapshots=snapshot_mgr, session=mock_session)

    res = upgrader.evaluate_and_upgrade(event_id)
    assert res['success'] is True
    assert res['status'] == 'NOOP'
    assert not mock_session.put_gate_event_images.called


def test_upgrader_async_queue_and_worker(repo, snapshot_mgr, mock_settings):
    """Test start(), enqueue_upgrade(), and stop() worker thread."""
    mock_session = MagicMock()
    upgrader = CloudProofUpgrader(mock_settings, repository=repo, snapshots=snapshot_mgr, session=mock_session)
    upgrader.start()
    assert upgrader._worker_thread is not None and upgrader._worker_thread.is_alive()

    # Enqueue a dummy upgrade
    upgrader.enqueue_upgrade("non-existent-id")
    upgrader.stop(timeout=1.0)
    assert upgrader._worker_thread is None


def test_upgrader_twin_event_upgrade(repo, snapshot_mgr, mock_settings, tmp_path):
    """Test that twin events upgrade both front and rear containers without CameraId collisions."""
    event_id = str(uuid4())
    cloud_id = f"cloud-{uuid4()}"
    visit_id = f"visit-{uuid4()}"

    with repo._write_lock, repo.db.sessions.begin() as s:
        s.add(GateEvent(
            id=event_id,
            gate_id="lane-in",
            timestamp=utcnow(),
            container_number="CAAU3179476",
            container_number_2="CAAU3179481",
            is_twin=True,
            iso_code_1="22G1",
            iso_code_2="22G1",
            event_type="ENTRY",
            status="CONFIRMED",
            visit_id=visit_id,
            cloud_synced=True,
            cloud_event_id=cloud_id,
            proof_version=1,
        ))

    import cv2
    v_dir = tmp_path / "snapshots" / "detected_in" / visit_id
    v_dir.mkdir(parents=True, exist_ok=True)

    img_front = np.full((480, 640, 3), 110, dtype=np.uint8)
    img_rear = np.full((480, 640, 3), 190, dtype=np.uint8)
    cv2.imwrite(str(v_dir / "gate-in-1_context.jpg"), img_front)
    cv2.imwrite(str(v_dir / "gate-in-3_rear.jpg"), img_rear)

    repo.add_or_update_snapshot(event_id, "gate-in-1", f"detected_in/{visit_id}/gate-in-1_context.jpg")
    repo.add_or_update_snapshot(event_id, "gate-in-3", f"detected_in/{visit_id}/gate-in-3_rear.jpg")

    mock_session = MagicMock()
    mock_session.put_gate_event_images.return_value = True

    upgrader = CloudProofUpgrader(mock_settings, repository=repo, snapshots=snapshot_mgr, session=mock_session)
    res = upgrader.evaluate_and_upgrade(event_id, force=True)

    assert res['success'] is True
    assert res['status'] == 'UPGRADED'
    assert mock_session.put_gate_event_images.called

    payload = mock_session.put_gate_event_images.call_args[0][1]
    containers = payload['Containers']
    assert len(containers) == 2
    assert containers[0]['ContainerNumber'] == "CAAU3179476"
    assert containers[0]['Position'] == "FRONT"
    assert containers[1]['ContainerNumber'] == "CAAU3179481"
    assert containers[1]['Position'] == "REAR"

    # Verify CameraId uniqueness per container
    for c in containers:
        cams = [img['CameraId'] for img in c['Images']]
        assert len(cams) == len(set(cams))


def test_api_upgrade_proofs_endpoint(repo, snapshot_mgr, mock_settings):
    """Test POST /api/gate-events/{event_id}/upgrade-proofs."""
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from app.api.events import router

    app = FastAPI()
    app.include_router(router)

    event_id = str(uuid4())
    mock_runtime = MagicMock()
    mock_runtime.proof_upgrader.evaluate_and_upgrade.return_value = {
        'success': True,
        'status': 'UPGRADED',
        'proof_version': 2,
    }

    app.state.runtime = mock_runtime
    app.state.repository = repo
    app.state.snapshots = snapshot_mgr

    client = TestClient(app)
    resp = client.post(f"/api/gate-events/{event_id}/upgrade-proofs?force=true")
    assert resp.status_code == 200
    data = resp.json()
    assert data['success'] is True
    assert data['status'] == 'UPGRADED'
    mock_runtime.proof_upgrader.evaluate_and_upgrade.assert_called_once_with(event_id, force=True)

