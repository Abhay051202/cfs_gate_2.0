from copy import deepcopy
from unittest.mock import Mock

from app.domain import OCRRead
from app.ocr.acceptance import assess_identity
from app.ocr.validator import ContainerValidator, check_digit
from app.ocr.ocr_engine import PaddleOCREngine


def event(number='MEDU7007822', cameras=('cam1', 'cam3')):
    return {'container_number': number, 'detections': [
        {'id': str(i), 'camera_id': camera, 'timestamp': f'2026-10-28T08:29:{50+i}Z'}
        for i, camera in enumerate(cameras)], 'ocr_results': [
        {'detection_id': str(i), 'raw_text': number, 'confidence': .95}
        for i, _ in enumerate(cameras)]}


def test_reported_m_w_error_is_a_checksum_collision():
    assert check_digit('MEDU700782') == check_digit('WEDU700782') == 2
    assert ContainerValidator().validate(OCRRead('WEDU7007822', .96)).valid_check_digit
    assert not assess_identity(event('WEDU7007822', ('cam2',)))['accepted']
    assert assess_identity(event())['accepted']


def test_duplicate_same_frame_or_same_camera_is_not_independent_views():
    assert not assess_identity(event(cameras=('cam1', 'cam1')))['accepted']
    sample = event(cameras=('cam1',))
    sample['ocr_results'] *= 4
    assert not assess_identity(sample)['accepted']


def test_conflict_is_review_not_confidence_winner():
    sample = event()
    sample['ocr_results'][1]['raw_text'] = 'WEDU7007822'
    result = assess_identity(sample)
    assert not result['accepted']
    assert 'Conflicting' in result['reason']


def test_uncertain_read_is_not_confirmed_by_checksum():
    sample = event()
    sample['ocr_results'][1]['confidence'] = .60
    assert not assess_identity(sample)['accepted']


def test_ocr_preserves_actual_confidence_and_observed_text():
    engine = PaddleOCREngine.__new__(PaddleOCREngine)
    import threading
    import numpy as np
    engine._lock = threading.Lock()
    engine.engine = Mock(return_value=([['MEDU 700782 2', .60]], 0))
    read = engine.read(np.ones((50, 250, 3), dtype=np.uint8))
    assert read.confidence == .60
    assert read.raw_text == 'MEDU 700782 2'


def test_synthetic_valid_flag_cannot_replace_observed_identifier():
    sample = event()
    for r in sample['ocr_results']:
        r.update(raw_text='MEDU700782', normalized_text='MEDU7007822', valid_check_digit=True)
    assert not assess_identity(sample)['accepted']


def test_twins_each_need_independent_matching_evidence():
    first = event()
    second = event('CSQU3054383')
    for d in second['detections']:
        d['id'] += 'rear'
    for r in second['ocr_results']:
        r['detection_id'] += 'rear'
    sample = deepcopy(first)
    sample.update(is_twin=True, container_number_2='CSQU3054383')
    sample['detections'] += second['detections']
    sample['ocr_results'] += second['ocr_results']
    assert assess_identity(sample)['accepted']
    sample['ocr_results'].pop()
    assert not assess_identity(sample)['accepted']


def test_persisted_single_camera_misread_is_held_and_cannot_dispatch(settings, repository, snapshots, make_job):
    from dataclasses import replace
    from app.domain import Observation
    from app.events.cloud_dispatcher import CloudOutboxDispatcher
    repository.strict_ocr_acceptance = True
    job = make_job()
    validation = ContainerValidator().validate(OCRRead('WEDU7007822', .96))
    obs = Observation(job.origin_key, job.gate_id, job.track_id, job.detection,
        validation, job.direction, job.frame.image, job.started_at, True)
    event_id, _ = repository.save_observation(obs, snapshots, 30)
    detail = repository.event(event_id)
    assert detail['status'] == 'NEEDS_REVIEW'
    assert detail['container_number'] is None
    assert detail['candidate_number'] == 'WEDU7007822'
    cfg = settings.model_copy(update={'strict_ocr_acceptance': True})
    assert not CloudOutboxDispatcher(cfg).is_dispatchable_event(detail)
