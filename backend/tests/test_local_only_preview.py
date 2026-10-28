import base64
from unittest.mock import Mock, patch

import pytest

from app.config.settings import Settings
from app.events.cloud_dispatcher import CloudOutboxDispatcher, ProsperSession
from app.events.proof_updater import CloudProofUpgrader
from app.snapshots.outgoing_preview import OutgoingPreview


def payload(image=b'jpeg-payload'):
    return {'id': 'event', 'Containers': [{'ContainerNumber': 'CSQU3054383',
        'Images': [{'CameraId': 'cam1', 'Image': 'data:image/jpeg;base64,' + base64.b64encode(image).decode()}]}]}


def test_local_mode_overrides_upload_flags(settings):
    cfg = Settings(_env_file=None, local_test_mode=True, prosper_enabled=True, prosper_auto_dispatch=True)
    assert not cfg.prosper_enabled and not cfg.prosper_auto_dispatch
    dispatcher = CloudOutboxDispatcher(cfg)
    dispatcher.resume()
    assert not dispatcher.enabled and not dispatcher.auto_dispatch
    assert dispatcher.dispatch_event('event')['status'] == 'DISABLED'


def test_direct_cloud_entry_points_cannot_send(settings):
    cfg = settings.model_copy(update={'local_test_mode': True, 'prosper_enabled': True})
    session = ProsperSession(cfg)
    with patch.object(session.session, 'request') as request:
        assert session.get_auth_headers() == {}
        assert session.upload_image(b'image') is None
        assert session.post_gate_event(payload()) == 'LOCAL_ONLY'
        assert session.put_gate_event_images('remote', payload()) == 'LOCAL_ONLY'
        request.assert_not_called()
    upgrader = CloudProofUpgrader(cfg, repository=Mock(), session=Mock())
    assert upgrader.evaluate_and_upgrade('event')['status'] == 'LOCAL_ONLY'
    upgrader.session.put_gate_event_images.assert_not_called()


def test_payload_survives_restart_without_reselecting_images(tmp_path):
    store = OutgoingPreview(tmp_path)
    first = store.freeze('event', payload(b'first-image'))
    assert OutgoingPreview(tmp_path).freeze('event', payload(b'later-image')) == first
    assert OutgoingPreview(tmp_path).load('event') == first


def test_ui_preview_equals_future_upload_payload(settings, snapshots):
    cfg = settings.model_copy(update={'local_test_mode': True})
    repo = Mock()
    repo.event.return_value = {'status': 'CONFIRMED'}
    dispatcher = CloudOutboxDispatcher(cfg, snapshots=snapshots, repository=repo)
    with patch.object(dispatcher, '_build_current_event_payload', return_value=payload()) as builder:
        preview = dispatcher.build_proof_preview('event')
        builder.return_value = payload(b'new-image')
        assert dispatcher.build_event_payload('event') == preview
        assert dispatcher.build_proof_preview('event') == preview
        assert builder.call_count == 1


def test_incomplete_event_does_not_freeze(settings, snapshots):
    cfg = settings.model_copy(update={'local_test_mode': True})
    repo = Mock()
    repo.event.return_value = {'status': 'NEEDS_REVIEW'}
    dispatcher = CloudOutboxDispatcher(cfg, snapshots=snapshots, repository=repo)
    with patch.object(dispatcher, '_build_current_event_payload', return_value=payload()):
        with pytest.raises(ValueError, match='confirmed event'):
            dispatcher.build_proof_preview('event')
    assert OutgoingPreview(snapshots.root).load('event') is None


def test_tampering_and_traversal_fail(tmp_path):
    store = OutgoingPreview(tmp_path)
    store.freeze('event', payload())
    path = store.path('event')
    path.write_text(path.read_text().replace('CSQU3054383', 'CHANGED'))
    with pytest.raises(ValueError, match='integrity'):
        store.load('event')
    with pytest.raises(ValueError):
        store.path('../escape')


def test_explicit_camera_revision_archives_original_and_preserves_other_camera(tmp_path):
    store = OutgoingPreview(tmp_path)
    original = payload(b'old')
    original['Containers'][0]['Images'].append({'CameraId': 'cam3', 'Image': 'old-camera-3'})
    store.freeze('event', original)
    revised = store.revise_camera('event', 'cam3', b'\xff\xd8original-jpeg',
        '2026-10-28T08:33:41Z', 'original.jpg', 'reviewed rear doors')
    assert revised['Containers'][0]['Images'][0] == original['Containers'][0]['Images'][0]
    assert revised['ProofVersion'] == 2
    assert (store.root / 'event.before-v2.json').is_file()
    assert store.freeze('event', payload(b'new-automatic-image')) == revised
