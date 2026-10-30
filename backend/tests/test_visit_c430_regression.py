from dataclasses import replace
from datetime import timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch
import os

import cv2
import numpy as np
import pytest

from app.domain import Detection, OCRRead
from app.events.capture_planner import plan_captures
from app.ocr.container_ocr import stacked_number_strips
from app.ocr.job_store import OCRJobStore
from app.ocr.ocr_engine import PaddleOCREngine
from app.snapshots.visit_capture import VisitCapture
from tests.test_pipeline_contract import passage

SOURCE = Path(__file__).parent/'fixtures/regressions/c430-vertical-msbu.png'


def test_real_vertical_crop_keeps_upright_owner_and_serial_before_rotations():
    crop = cv2.imread(str(SOURCE))
    strips = stacked_number_strips(crop)
    assert strips and all(s.shape[1] > s.shape[0]*3 for s in strips)
    import threading
    engine = PaddleOCREngine.__new__(PaddleOCREngine)
    engine._lock = threading.Lock()
    engine.engine = Mock(side_effect=[([['L', .86]], 0), ([['MSBU1383355', .965]], 0)])
    read = engine.read(crop)
    assert read.raw_text == 'MSBU1383355'
    assert engine.engine.call_count == 2
    assert read.confidence == .965


@pytest.mark.skipif(os.environ.get('RUN_LOCAL_OCR_REGRESSION') != '1', reason='Opt-in real OCR with installed local weights')
def test_real_reader_recognizes_reported_visit():
    from app.config.settings import Settings
    engine = PaddleOCREngine(Settings(_env_file=None, local_test_mode=True, ocr_gpu=False, ocr_download_enabled=False))
    try:
        assert 'MSBU1383355' in engine.read(cv2.imread(str(SOURCE))).raw_text.replace(' ', '')
    finally:
        if hasattr(engine, 'close'):
            engine.close()


def test_live_visit_jobs_do_not_expire_while_waiting_for_ocr(tmp_path, make_job):
    store = OCRJobStore(tmp_path)
    active = make_job(origin='passage:active:region')
    obsolete = replace(make_job(2, origin='passage:obsolete:region'), frame=replace(active.frame, sequence=2))
    first = store.save(active, np.ones((30,40,3), dtype=np.uint8))
    second = store.save(obsolete, np.ones((30,40,3), dtype=np.uint8))
    with patch('app.ocr.job_store.time.time', return_value=active.frame.timestamp.replace(tzinfo=timezone.utc).timestamp()+120):
        store.clean_stale_pending(60, protected_origins=(active.origin_key,))
    with store.connection() as db:
        assert db.execute('SELECT state FROM jobs WHERE id=?',(first,)).fetchone()['state'] == 'PENDING'
        assert db.execute('SELECT state FROM jobs WHERE id=?',(second,)).fetchone()['state'] == 'SUPERSEDED'


def test_failed_one_letter_ocr_still_creates_an_unreadable_review_record(repository, snapshots, make_job):
    job = make_job()
    job = replace(job, detection=replace(job.detection, class_name='container_number'))
    p = passage(job, 'MSBU1383355')
    region = p.regions[0]
    from app.ocr.validator import ContainerValidator
    value = ContainerValidator().validate(OCRRead('L', .86))
    region.votes.clear()
    region.reads.clear()
    region.source_proofs.clear()
    region.best_any = (value, job)
    plans = plan_captures([p], 2)
    assert len(plans) == 1
    record = repository.event(repository.save_passage(plans[0], snapshots, 2))
    assert record and record['status'] == 'UNREADABLE'


def test_capture_continues_through_passage_and_rejects_empty_frames(settings, snapshots, camera, make_job):
    settings.visit_capture_enabled = True
    settings.visit_capture_max_frames = 6
    capture = VisitCapture(settings, snapshots)
    base = replace(make_job().frame, image=np.random.default_rng(43).integers(0,255,(100,640,3),dtype=np.uint8))
    p = passage(replace(make_job(), frame=base), 'MSBU1383355')
    for t in range(12):
        frame = replace(base, sequence=t+1, timestamp=base.timestamp+timedelta(seconds=t))
        p.bbox, p.last_seen = (10+t*10,10,130+t*10,90), frame.timestamp
        detection = Detection(camera.id, frame.timestamp, 'container_number', .9, (30+t*10,20,50+t*10,70))
        capture.observe(camera, frame, [p], [detection])
    files = sorted(snapshots.path('visit-a').glob('*_capture_*.jpg'))
    assert len(files) == 6
    before = {f.name: f.read_bytes() for f in files}
    frame = replace(base, timestamp=base.timestamp+timedelta(seconds=20), sequence=21)
    p.last_seen = frame.timestamp
    capture.observe(camera, frame, [p], [])
    assert before == {f.name: f.read_bytes() for f in files}
