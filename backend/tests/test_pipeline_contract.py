from collections import Counter
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app.config.settings import CameraConfig
from app.domain import OCRRead
from app.events.capture_planner import plan_captures
from app.events.cloud_dispatcher import CloudOutboxDispatcher
from app.events.passages import Passage, Region
from app.ocr.validator import ContainerValidator
from app.snapshots.overview import VisitOverview


def passage(job, number, *, identity='truck-a', chassis=False, size=None):
    value = ContainerValidator().validate(OCRRead(number, .95))
    if size:
        value = replace(value, size_type=size)
    region = Region(votes=Counter({number: 2}), reads={number: (value, job)},
        best_any=(value, job), source_proofs={number: {job.frame.camera_id: (value, job)}})
    return Passage(identity, job.frame.camera_id, job.frame.source_id, job.gate_id,
        job.gate_id, job.direction, job.frame.timestamp, job.frame.timestamp,
        1, (10, 10, 130, 90), regions=[region], visit_id='visit-a',
        body_observed=True, chassis_observed=chassis)


def test_all_supported_numbers_survive_without_a_twin_label(repository, snapshots, make_job):
    a = passage(make_job(), 'CSQU3054383')
    b = passage(make_job(2), 'MSCU6639870', identity='other-text')
    c = passage(make_job(3), 'TGBU3796155', identity='third-text')
    a.regions += b.regions + c.regions
    plans = plan_captures([a], 2)
    assert {p.selected_numbers for p in plans} == {('CSQU3054383',), ('MSCU6639870',), ('TGBU3796155',)}
    for plan in plans:
        repository.save_passage(plan, snapshots, 2)
    events = repository.events()['items']
    assert len(events) == 3
    assert all(not e['is_twin'] for e in events)
    assert {e['container_number'] for e in events} == {'CSQU3054383', 'MSCU6639870', 'TGBU3796155'}


def test_two_twenty_foot_numbers_with_one_chassis_are_twins(make_job):
    a = passage(make_job(), 'CSQU3054383', chassis=True, size='22G1')
    a.regions += passage(make_job(2), 'MSCU6639870', size='22G1').regions
    plans = plan_captures([a], 2)
    assert len(plans) == 1 and len(plans[0].selected_numbers) == 2


def test_same_region_checksum_collision_is_retained_for_review(repository, snapshots, make_job):
    a = passage(make_job(), 'MEDU7007822')
    other = passage(make_job(2), 'WEDU7007822').regions[0]
    a.regions[0].votes.update(other.votes)
    a.regions[0].reads.update(other.reads)
    a.regions[0].source_proofs.update(other.source_proofs)
    plans = plan_captures([a], 2)
    assert len(plans) == 2 and all(p.identity_conflict for p in plans)
    for plan in plans:
        event = repository.event(repository.save_passage(plan, snapshots, 2))
        assert event['status'] == 'NEEDS_REVIEW' and event['identity_conflict']


def overview_scene(settings, snapshots, make_job, *, line=True, stale=False):
    settings.proof_capture_enabled = True
    job = make_job(direction='ENTRY')
    rng = np.random.default_rng(42)
    cameras, group, frames, workers = [], [], {}, {}
    for i in range(3):
        camera = CameraConfig(id=f'camera-{i+1}', name=f'View {i+1}', gate_id=job.gate_id,
            source_env=f'CAMERA_{i+1}_RTSP', direction='ENTRY', role='FRONT_TOP' if i == 0 else 'LEFT',
            proof_capture_axis='x' if line and i == 0 else None)
        cameras.append(camera)
        frame = replace(job.frame, camera_id=camera.id, source_id=f'source-{i}',
            image=rng.integers(10, 245, (100, 150, 3), dtype=np.uint8))
        p = passage(replace(job, frame=frame), 'CSQU3054383', identity=f'part-{i}')
        p.source_id = frame.source_id
        p.bbox = (15, 10, 135, 90)
        frames[camera.id] = frame
        group.append(p)
        workers[camera.id] = SimpleNamespace(nearest_frame=lambda timestamp, tolerance, f=frame: f)
    if stale:
        workers['camera-3'] = SimpleNamespace(nearest_frame=lambda *_: None)
    controller = VisitOverview(settings, snapshots)
    return controller, cameras, group, frames, workers


def test_three_camera_overview_does_not_require_three_ocr_results(settings, snapshots, make_job):
    controller, cameras, group, frames, workers = overview_scene(settings, snapshots, make_job)
    for p in group[1:]:
        p.regions.clear()  # Cameras see the body without reading its number.
    controller.observe(cameras[0], frames[cameras[0].id], group, cameras, workers, [])
    overview = controller.load('visit-a')
    assert overview['complete'] and overview['frozen']
    assert len(overview['images']) == 3
    previous = [image['sha256'] for image in overview['images']]
    frames['camera-1'].image[:] = 0
    controller.observe(cameras[0], frames['camera-1'], group, cameras, workers, [])
    assert [image['sha256'] for image in controller.load('visit-a')['images']] == previous


@pytest.mark.parametrize('line,stale,state', [(False, False, 'NEEDS_CALIBRATION'), (True, True, 'NEEDS_REVIEW')])
def test_incomplete_overview_is_visible_but_not_uploadable(settings, repository, snapshots, make_job, line, stale, state):
    controller, cameras, group, frames, workers = overview_scene(settings, snapshots, make_job, line=line, stale=stale)
    repository.sync_cameras(cameras)
    controller.observe(cameras[0], frames['camera-1'], group, cameras, workers, [])
    assert controller.load('visit-a')['state'] == state
    plan = plan_captures(group[:1], 2)[0]
    # Shared folder is recorded by the runtime's source metadata in production.
    plan.regions[0].best_any[1].passage_metadata['shared_visit_folder'] = True
    event_id = repository.save_passage(plan, snapshots, 2)
    dispatcher = CloudOutboxDispatcher(settings, repository=repository, snapshots=snapshots)
    preview = dispatcher.build_proof_preview(event_id)
    assert preview['Containers'][0]['Images'] and not preview['_proof_complete']
    with pytest.raises(ValueError):
        dispatcher.build_event_payload(event_id)


def test_overview_freezes_exact_payload_for_local_and_cloud(settings, repository, snapshots, make_job):
    controller, cameras, group, frames, workers = overview_scene(settings, snapshots, make_job)
    repository.sync_cameras(cameras)
    controller.observe(cameras[0], frames['camera-1'], group, cameras, workers, [])
    plan = plan_captures(group[:1], 2)[0]
    plan.regions[0].best_any[1].passage_metadata['shared_visit_folder'] = True
    event_id = repository.save_passage(plan, snapshots, 2)
    dispatcher = CloudOutboxDispatcher(settings, repository=repository, snapshots=snapshots)
    preview = dispatcher.build_proof_preview(event_id)
    assert preview['_proof_complete']
    assert preview == dispatcher.build_event_payload(event_id)
    assert len(preview['Containers'][0]['Images']) == 3
    assert preview['Containers'][0]['Size'] is None  # Unknown is not guessed as 20/40 ft.


def test_separate_container_records_have_separate_cloud_keys(settings, repository, snapshots, make_job):
    controller, cameras, group, frames, workers = overview_scene(settings, snapshots, make_job)
    repository.sync_cameras(cameras)
    controller.observe(cameras[0], frames['camera-1'], group, cameras, workers, [])
    extra = passage(replace(make_job(), frame=frames['camera-1'], direction='ENTRY'), 'MSCU6639870')
    group[0].regions += extra.regions
    dispatcher = CloudOutboxDispatcher(settings, repository=repository, snapshots=snapshots)
    payloads = []
    for plan in plan_captures(group[:1], 2):
        plan.regions[0].best_any[1].passage_metadata['shared_visit_folder'] = True
        event_id = repository.save_passage(plan, snapshots, 2)
        payloads.append(dispatcher.build_event_payload(event_id))
    assert len({p['VisitId'] for p in payloads}) == 2
    assert {p['_source_visit_id'] for p in payloads} == {'visit-a'}


def test_overlapping_trucks_do_not_borrow_unassigned_camera_views(settings, snapshots, make_job):
    controller, cameras, group, frames, workers = overview_scene(settings, snapshots, make_job)
    b = replace(group[0], id='truck-b', visit_id='visit-b')
    group = [group[0], b]  # Neither visit has a proven owner in cameras 2/3.
    controller.observe(cameras[0], frames['camera-1'], group, cameras, workers, [])
    assert not controller.load('visit-a')['complete']
    assert controller.load('visit-a')['missing_cameras'] == ['camera-2', 'camera-3']


def test_broken_visit_does_not_block_following_visit(settings, repository, snapshots, camera, make_job):
    from app.runtime import GateRuntime
    runtime = GateRuntime(settings, [camera], repository, snapshots)
    a = passage(make_job(), 'CSQU3054383')
    b = replace(a, id='truck-b', visit_id='visit-b')
    a.closed = b.closed = True
    runtime.passages.groups = Mock(return_value=[[a], [b]])
    runtime._queue_passage_candidates = Mock()
    runtime.passages.close_untracked_observers = Mock()
    runtime._finalize_passage_group = Mock(side_effect=[ValueError('bad plan'), None])
    runtime._finalize_passages()
    assert runtime._finalize_passage_group.call_count == 2
    assert runtime._finalization_failures['visit-a']['reason'] == 'bad plan'


def test_camera_buffer_rejects_stale_and_previous_connection_frames(settings, camera):
    from app.camera.camera_worker import CameraWorker
    worker = CameraWorker(camera, '', settings, Mock())
    worker._publish(np.zeros((100, 150, 3), dtype=np.uint8))
    current = worker.latest()
    assert worker.nearest_frame(current.timestamp, .3) == current
    assert worker.nearest_frame(current.timestamp + timedelta(seconds=2), .3) is None
    worker.source_id = 'reconnected'
    assert worker.nearest_frame(current.timestamp, .3) is None


def test_selected_frame_must_still_contain_the_container(settings, snapshots, make_job):
    controller, cameras, group, frames, workers = overview_scene(settings, snapshots, make_job, line=False)
    controller.observe(cameras[0], frames['camera-1'], group, cameras, workers, [])
    before = controller.load('visit-a')
    for p in group:
        frame = replace(frames[p.camera_id], timestamp=p.last_seen+timedelta(seconds=2), sequence=2)
        frames[p.camera_id] = frame
        workers[p.camera_id] = SimpleNamespace(nearest_frame=lambda *_args, f=frame: f)
        p.last_seen, p.bbox = frame.timestamp, (0,0,150,100)
    validator = Mock(return_value=False)
    controller.observe(cameras[0], frames['camera-1'], group, cameras, workers, [], validator)
    overview = controller.load('visit-a')
    assert overview == before
    assert validator.call_count == 3


def test_following_truck_does_not_keep_departed_truck_occupied(settings, camera, make_job):
    from app.events.passages import PassageCollector
    from app.domain import Detection
    settings.vehicle_visit_tracking = True
    frame = replace(make_job().frame, source_id='live', image=np.random.default_rng(8).integers(
        0, 255, (100, 150, 3), dtype=np.uint8))
    collector = PassageCollector(settings)
    old = Passage('a', camera.id, frame.source_id, camera.gate_id, camera.gate_id,
        camera.direction, frame.timestamp, frame.timestamp, 99, (5, 5, 65, 90), visit_id='visit-a')
    collector.passages[old.id] = old
    collector.active[(camera.id, frame.source_id, old.track_id)] = old
    for t in range(1, 11):
        current = replace(frame, sequence=t, timestamp=frame.timestamp+timedelta(seconds=t))
        collector.update(camera, current, [Detection(camera.id, current.timestamp, 'truck', .95, (80,5,145,90))])
    assert old.closed
    assert any(not p.closed and p.id != old.id for p in collector.passages.values())


def test_following_truck_cannot_borrow_departed_front_camera_visit(settings, camera, make_job):
    from app.events.passages import PassageCollector
    from app.domain import Detection
    settings.vehicle_visit_tracking = True
    frame = replace(make_job().frame, source_id='live', image=np.random.default_rng(9).integers(
        0,255,(100,150,3),dtype=np.uint8))
    collector = PassageCollector(settings)
    old = Passage('a', camera.id, frame.source_id, camera.gate_id, camera.gate_id,
        camera.direction, frame.timestamp, frame.timestamp, 99, (5,5,65,90),
        visit_id='visit-a', closed=True, closed_at=frame.timestamp)
    rear = replace(old, id='a-rear', camera_id='rear', source_id='rear-live', closed=False)
    collector.passages.update({old.id: old, rear.id: rear})
    collector.active[('rear','rear-live',99)] = rear
    collector._visit_cameras['visit-a'] = {camera.id, 'rear'}
    next_frame = replace(frame, sequence=2, timestamp=frame.timestamp+timedelta(seconds=1))
    collector.update(camera, next_frame, [Detection(camera.id, next_frame.timestamp, 'truck', .95, (80,5,145,90))])
    following = [p for p in collector.passages.values() if p.camera_id == camera.id and not p.closed]
    assert len(following) == 1 and following[0].visit_id != 'visit-a'


def test_api_shows_partial_proof_without_falsely_marking_it_frozen(settings, repository, snapshots, make_job):
    from fastapi.testclient import TestClient
    from app.main import create_app
    controller, cameras, group, frames, workers = overview_scene(settings, snapshots, make_job, stale=True)
    repository.sync_cameras(cameras)
    controller.observe(cameras[0], frames['camera-1'], group, cameras, workers, [])
    plan = plan_captures(group[:1], 2)[0]
    plan.regions[0].best_any[1].passage_metadata['shared_visit_folder'] = True
    event_id = repository.save_passage(plan, snapshots, 2)
    settings.local_test_mode = True
    with TestClient(create_app(settings, repository.db, cameras)) as client:
        event = client.get('/api/gate-events/' + event_id).json()
        assert not event['cloud_proof_complete'] and not event['proof_frozen']
        assert len(event['cloud_proof_images']) == 2
        assert 'camera-3' in event['cloud_proof_error']


def test_accuracy_report_counts_wrong_and_unreadable_images(tmp_path):
    import cv2
    from tools.evaluate_ocr import evaluate
    cv2.imwrite(str(tmp_path / 'number.jpg'), np.ones((40,200,3), dtype=np.uint8)*127)
    reader = Mock(read=Mock(side_effect=[OCRRead('MEDU7007822', .95), OCRRead('WEDU7007822', .99), OCRRead('', 0)]))
    rows = [{'image': 'number.jpg', 'container_number': 'MEDU7007822', 'condition': c}
            for c in ('day', 'night', 'blur')]
    report = evaluate(rows, tmp_path, reader)
    assert report['exact_accuracy'] == 1/3
    assert report['conditions']['night']['correct'] == 0
