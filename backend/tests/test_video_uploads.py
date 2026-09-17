import time
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.config.settings import CameraConfig, CameraSourceConfig
from app.main import create_app
from app.camera.video_library import VideoLibrary
from tests.test_cameras_runtime import wait_until


@pytest.fixture
def video_bytes(tmp_path):
    import cv2
    path = tmp_path / 'test-clip.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 20, (160, 100))
    assert writer.isOpened()
    for index in range(100):
        image = np.full((100, 160, 3), index * 2, dtype=np.uint8)
        writer.write(image)
    writer.release()
    return path.read_bytes()


def upload(client, camera_id, content, name='test-clip.avi'):
    return client.post(f'/api/cameras/{camera_id}/video', content=content,
                       headers={'Content-Type': 'application/octet-stream', 'X-Filename': name})


def test_camera_source_dropdown_switches_slot_and_hides_rtsp(settings, repository, camera, video_bytes, monkeypatch):
    sources = [CameraSourceConfig(id='poe-entrance', name='PoE entrance', source_env='POE_ENTRANCE_RTSP'),
               CameraSourceConfig(id='poe-side', name='PoE side', source_env='POE_SIDE_RTSP')]
    monkeypatch.setenv('POE_ENTRANCE_RTSP', 'rtsp://operator:private-password@10.0.0.11/stream')
    monkeypatch.delenv('POE_SIDE_RTSP', raising=False)
    with TestClient(create_app(settings, repository.db, [camera], camera_sources=sources)) as client:
        available = client.get('/api/cameras/sources/available')
        assert available.status_code == 200
        assert [{k: s[k] for k in ('id', 'name', 'configured')} for s in available.json()] == [
            {'id': 'poe-entrance', 'name': 'PoE entrance', 'configured': True},
            {'id': 'poe-side', 'name': 'PoE side', 'configured': False}]
        assert 'private-password' not in available.text
        assert upload(client, camera.id, video_bytes).status_code == 200
        response = client.post(f'/api/cameras/{camera.id}/source', json={'source_id': 'poe-side'})
        assert response.status_code == 200
        slot = client.get('/api/cameras').json()[0]
        assert slot['source_type'] == 'rtsp'
        assert slot['active_source_id'] == 'poe-side'
        assert slot['active_source_name'] == 'PoE side'
        assert slot['video']['active'] is False
        assert settings.upload_directory.joinpath('camera-selections.json').is_file()
        assert client.post(f'/api/cameras/{camera.id}/source', json={'source_id': 'missing'}).status_code == 400


def test_four_uploads_play_pause_restart_restore(settings, repository, video_bytes):
    configs = [CameraConfig(id=f'camera-{i}', name=f'View {i}', gate_id=f'lane-{i}', source_env=f'UNCONFIGURED_{i}') for i in range(1, 5)]
    with TestClient(create_app(settings, repository.db, configs)) as client:
        runtime = client.app.state.runtime
        for camera in configs:
            response = upload(client, camera.id, video_bytes)
            assert response.status_code == 200, response.text
            assert response.json()['video']['filename'] == 'test-clip.avi'
        slots = client.get('/api/cameras').json()
        assert len(slots) == 4
        assert all(c['source_type'] == 'file' and c['playback_status'] == 'READY' and c['has_frame'] for c in slots)
        assert client.get('/api/cameras/camera-1/frame').status_code == 200
        assert client.post('/api/videos/play-all').status_code == 200
        wait_until(lambda: all(w.status()['playback_status'] == 'PLAYING' for w in runtime.manager.workers.values()))
        assert client.post('/api/cameras/camera-1/playback', json={'action': 'pause'}).status_code == 200
        # Allow a read already in progress to finish before checking pause.
        time.sleep(.1)
        at_pause = runtime.manager.workers['camera-1'].status()['video_position_seconds']
        other_before = runtime.manager.workers['camera-2'].status()['video_position_seconds']
        time.sleep(.12)
        assert runtime.manager.workers['camera-1'].status()['video_position_seconds'] == at_pause
        assert runtime.manager.workers['camera-2'].status()['video_position_seconds'] > other_before
        previous_source = runtime.manager.workers['camera-1'].source_id
        assert client.post('/api/cameras/camera-1/playback', json={'action': 'restart'}).status_code == 200
        assert runtime.manager.workers['camera-1'].source_id != previous_source
        assert client.post('/api/cameras/camera-1/playback', json={'action': 'restore-camera'}).status_code == 200
        slot = client.get('/api/cameras').json()[0]
        assert slot['source_type'] == 'rtsp' and slot['status'] == 'OFFLINE'
        assert not slot['video']['active']
        assert len(list(settings.upload_directory.glob('*.avi'))) == 4


def test_upload_validation_and_size_limit_leave_previous_video_intact(settings, repository, camera, video_bytes):
    settings.max_upload_mb = 1
    with TestClient(create_app(settings, repository.db, [camera])) as client:
        assert upload(client, camera.id, video_bytes).status_code == 200
        original_source = client.app.state.runtime.manager.workers[camera.id].source_id
        assert upload(client, camera.id, b'not-a-video', 'text.txt').status_code == 415
        assert upload(client, camera.id, b'not-a-video', 'text.avi').status_code == 422
        assert upload(client, camera.id, b'', 'empty.mp4').status_code == 400
        assert upload(client, camera.id, bytes(1024 * 1024 + 1)).status_code == 413
        assert upload(client, 'missing', video_bytes).status_code == 404
        assert client.app.state.runtime.manager.workers[camera.id].source_id == original_source
        assert len(list(settings.upload_directory.glob('*.avi'))) == 1


def test_video_assignment_restores_ready_after_service_restart(settings, repository, camera, video_bytes):
    with TestClient(create_app(settings, repository.db, [camera])) as client:
        assert upload(client, camera.id, video_bytes, '..%2Frecorded.avi').status_code == 200
        assert client.get('/api/cameras').json()[0]['video']['filename'] == 'recorded.avi'
    with TestClient(create_app(settings, repository.db, [camera])) as restarted:
        slot = restarted.get('/api/cameras').json()[0]
        assert slot['source_type'] == 'file' and slot['playback_status'] == 'READY'
        assert slot['video_position_seconds'] == 0


def test_upload_and_preview_work_while_database_unavailable(settings, repository, camera, video_bytes, monkeypatch):
    def unavailable():
        raise ConnectionError('Database unavailable')
    monkeypatch.setattr(repository.db, 'check', unavailable)
    with TestClient(create_app(settings, repository.db, [camera])) as client:
        assert client.get('/api/health').status_code == 503
        assert upload(client, camera.id, video_bytes).status_code == 200
        assert client.get('/api/cameras/camera-1/frame').status_code == 200
        assert client.post('/api/cameras/camera-1/playback', json={'action': 'play'}).status_code == 200
        assert client.post('/api/processing/start').status_code == 503


def test_cross_origin_mutations_rejected(settings, repository, camera, video_bytes):
    with TestClient(create_app(settings, repository.db, [camera])) as client:
        response = client.post('/api/cameras/camera-1/video', content=video_bytes,
                               headers={'X-Filename': 'clip.avi', 'Origin': 'https://unrelated.example'})
        assert response.status_code == 403
        assert not list(settings.upload_directory.glob('*.avi'))


def test_manifest_path_cannot_escape_upload_directory(tmp_path):
    library = VideoLibrary(tmp_path / 'uploads')
    with pytest.raises(ValueError):
        library.path('../outside.mp4')


def test_lane_video_sync_and_playback(settings, repository, video_bytes):
    configs = [
        CameraConfig(id='gate-in-1', name='Gate IN 1', gate_id='lane-in', source_env='CAM1'),
        CameraConfig(id='gate-in-2', name='Gate IN 2', gate_id='lane-in', source_env='CAM2'),
        CameraConfig(id='gate-in-3', name='Gate IN 3', gate_id='lane-in', source_env='CAM3'),
        CameraConfig(id='gate-in-4', name='Gate IN 4', gate_id='lane-in', source_env='CAM4'),
    ]
    with TestClient(create_app(settings, repository.db, configs)) as client:
        # Upload with sync_lane=true
        resp = client.post('/api/cameras/gate-in-1/video?sync_lane=true', content=video_bytes,
                           headers={'Content-Type': 'application/octet-stream', 'X-Filename': 'container.avi'})
        assert resp.status_code == 200
        assert resp.json()['sync_lane'] is True

        # Check that all 4 cameras in lane-in now have this video assigned
        cams = client.get('/api/cameras').json()
        assert len(cams) == 4
        for c in cams:
            assert c['source_type'] == 'file'
            assert c['video']['filename'] == 'container.avi'
            assert c['playback_status'] == 'READY'

        # Play all cameras in lane-in simultaneously
        play_resp = client.post('/api/lanes/lane-in/playback', json={'action': 'play'})
        assert play_resp.status_code == 200
        assert play_resp.json()['gate_id'] == 'lane-in'
