"""Offline recovery of retained visit frames. Never connects to cameras/cloud.

Dry-run by default. --apply backs up both databases and all source files, then
creates review-only records and a recovered overview from the surviving frames.
It does not claim missing moments or a calibrated capture line were recovered.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config.settings import Settings
from app.database.database import Database
from app.database.models import GateEvent
from app.database.repositories.gate import GateRepository
from app.domain import Detection, Frame
from app.events.capture_planner import plan_captures
from app.events.passages import Passage, Region
from app.ocr.ocr_engine import PaddleOCREngine
from app.ocr.validator import ContainerValidator
from app.runtime import OCRJob
from app.snapshots.overview import atomic_json
from app.snapshots.snapshot_manager import SnapshotManager


def recover(visit, apply=False):
    if not re.fullmatch(r'[a-fA-F0-9-]{36}', visit):
        raise ValueError('Provide a visit UUID')
    root = ROOT / 'snapshots'
    source = next((root/c/visit for c in ('waiting','detected_in','detected_out') if (root/c/visit).is_dir()), None)
    if source is None or not source.resolve().is_relative_to(root.resolve()):
        raise ValueError('Visit folder not found inside the snapshot root')
    jobs_path, gate_path = ROOT/'uploads/ocr-spool/jobs.sqlite3', ROOT/'data/gate.db'
    with sqlite3.connect(jobs_path.resolve().as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute('SELECT * FROM jobs')
                if json.loads(r['metadata']).get('storage_visit_id') == visit]
    if not rows:
        raise ValueError('No retained OCR source metadata for this visit')
    backup = None
    if apply:
        backup = ROOT/'local-test/recoveries'/visit/datetime.now().strftime('%Y%m%d-%H%M%S')
        backup.mkdir(parents=True)
        shutil.copytree(source, backup/'original-images')
        for path in (jobs_path, gate_path):
            with sqlite3.connect(path) as src, sqlite3.connect(backup/path.name) as dst:
                src.backup(dst)
        atomic_json(backup/'original-jobs.json', rows)
    settings = Settings(_env_file=None, local_test_mode=True, ocr_gpu=False, ocr_download_enabled=False)
    reader, validator = PaddleOCREngine(settings), ContainerValidator()
    passages, report, source_frames = [], [], {}
    try:
        for row in rows:
            meta = json.loads(row['metadata'])
            image_path, crop_path = source/Path(row['image']).name, source/Path(row['crop']).name
            image, crop = cv2.imread(str(image_path)), cv2.imread(str(crop_path))
            if image is None or crop is None:
                continue
            raw = reader.read(crop)
            value = validator.validate(raw)
            report.append({'job_id': row['id'], 'camera_id': meta['frame']['camera_id'],
                'previous': json.loads(row['validation']) if row['validation'] else None,
                'new': asdict(value)})
            data = meta['frame']
            frame = Frame(data['camera_id'], datetime.fromisoformat(data['timestamp']), data['sequence'], image, data['source_id'])
            det = meta['detection']
            detection = Detection(det['camera_id'], frame.timestamp, det['class_name'], det['confidence'], tuple(det['bbox']))
            metadata = {**meta.get('passage_metadata', {}), 'shared_visit_folder': True}
            job = OCRJob(meta['origin_key'], meta['gate_id'], meta['track_id'], detection,
                frame, meta['direction'], datetime.fromisoformat(meta['started_at']), tuple(meta['roi']),
                event_id=visit, passage_id=meta['passage_id'], region_id=meta['region_id'], passage_metadata=metadata)
            region = Region(id=job.region_id, best_any=(value, job),
                context_only=metadata.get('context_only', False))
            if value.valid_check_digit:
                region.votes = Counter({value.normalized_text: 1})
                region.reads = {value.normalized_text: (value, job)}
                region.source_proofs = {value.normalized_text: {frame.camera_id: (value, job)}}
            passages.append(Passage(job.passage_id, frame.camera_id, frame.source_id,
                job.gate_id, metadata['lane_id'], job.direction, job.started_at, frame.timestamp,
                job.track_id, tuple(metadata['bbox']), regions=[region], closed=True,
                ambiguous=True, chassis_observed=False, body_observed=metadata.get('body_observed', False), visit_id=visit))
            source_frames.setdefault(frame.camera_id, (frame, metadata))
    finally:
        if hasattr(reader, 'close'):
            reader.close()
    result = {'visit_id': visit, 'applied': apply, 'readings': report, 'events': []}
    if not apply:
        return result
    snapshots = SnapshotManager(root)
    db = Database('sqlite+pysqlite:///' + gate_path.as_posix())
    repo = GateRepository(db)
    repo.strict_ocr_acceptance = True
    for plan in plan_captures(passages, 2):
        event_id = repo.save_passage(plan, snapshots, 2)
        with repo.db.sessions.begin() as session:
            event = session.get(GateEvent, event_id)
            event.status = 'NEEDS_REVIEW'
            event.capture_policy = 'RECOVERY_REVIEW'  # Excluded from automatic cloud outbox.
        repo.log_system_event('offline_visit_recovered', event_id=event_id,
            context={'visit_id': visit, 'backup': str(backup), 'review_required': True})
        snapshots.save_event_json(event_id, repo.event(event_id))
        result['events'].append(event_id)
    directory = snapshots.path(visit)
    previous = directory/'overview.json'
    if previous.is_file():
        shutil.copyfile(previous, directory/'overview.before-recovery.json')
    images = []
    for camera, (frame, metadata) in sorted(source_frames.items()):
        ok, encoded = cv2.imencode('.jpg', frame.image, [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok:
            continue
        name = camera+'_recovered_overview.jpg'
        data = encoded.tobytes()
        (directory/name).write_bytes(data)
        images.append({'camera_id': camera, 'file': name, 'captured_at': frame.timestamp.isoformat()+'Z',
            'source_id': frame.source_id, 'sequence': frame.sequence,
            'sha256': hashlib.sha256(data).hexdigest(), 'target_bbox': metadata.get('bbox'),
            'visibility_verified': False})
    atomic_json(directory/'overview.json', {'visit_id': visit, 'mode': 'RECOVERED_CAPTURE',
        'reference_camera_id': images[0]['camera_id'], 'trigger_at': images[0]['captured_at'],
        'clock_basis': 'HOST_RECEIVE_TIME', 'required_cameras': sorted(source_frames),
        'missing_cameras': [], 'unverified_cameras': sorted(source_frames), 'images': images,
        'frozen': True, 'complete': False, 'state': 'NEEDS_REVIEW', 'sync_tolerance_seconds': .3,
        'reason': 'Surviving original frames recovered; calibration and independent OCR confirmation still required'})
    with sqlite3.connect(jobs_path) as spool:
        for reading in report:
            spool.execute('UPDATE jobs SET validation=? WHERE id=?', (json.dumps(reading['new']), reading['job_id']))
        if len(result['events']) == 1:
            for row in rows:
                meta = json.loads(row['metadata'])
                meta['finalized'] = True
                meta['event_id'] = result['events'][0]
                spool.execute("UPDATE jobs SET metadata=?, event_id=?,state='DONE' WHERE id=?",
                    (json.dumps(meta), result['events'][0], row['id']))
    atomic_json(backup/'recovery-report.json', result)
    db.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('visit')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(recover(args.visit, args.apply), indent=2))
