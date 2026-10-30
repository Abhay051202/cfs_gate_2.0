"""One bounded, source-bound overview set per physical visit, independent of OCR."""
import hashlib
import json
import threading
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.' + uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class VisitOverview:
    def __init__(self, settings, snapshots):
        self.settings, self.snapshots = settings, snapshots
        self._positions = {}
        self._lock = threading.RLock()

    def load(self, visit):
        path = self.snapshots.path(f'{visit}/overview.json')
        return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None

    def observe(self, camera, frame, passages, cameras, workers, detections, validator=None):
        with self._lock:
            self._observe(camera, frame, passages, cameras, workers, detections, validator)

    def _observe(self, camera, frame, passages, cameras, workers, detections, validator=None):
        if not self.settings.proof_capture_enabled:
            return
        groups = {}
        for p in passages:
            if not p.saved and not p.closed and not p.capture_only:
                groups.setdefault(p.visit_id or p.id, []).append(p)
        live = {(visit, p.camera_id) for visit, group in groups.items() for p in group}
        self._positions = {k: v for k, v in self._positions.items() if k in live}
        for visit, group in groups.items():
            members = [p for p in group if p.camera_id == camera.id and p.source_id == frame.source_id]
            if len(members) != 1:
                continue
            owner = members[0]
            expected = [c for c in cameras if c.enabled and c.gate_id == owner.gate_id
                        and c.direction == owner.direction]
            configured = [c for c in expected if c.proof_capture_axis is not None]
            reference = min(configured or expected,
                key=lambda c: (c.role != 'FRONT_TOP', c.id), default=camera)
            if camera.id != reference.id:
                continue
            previous = self.load(visit)
            if previous and previous.get('frozen'):
                continue
            h, w = frame.image.shape[:2]
            # Prefer the target's container body over its truck cab. It is
            # associated by containment, not by taking the largest box in view.
            bx = owner.bbox
            boxes = [d.bbox for d in detections if d.class_name == 'container'
                     and bx[0] <= (d.bbox[0]+d.bbox[2])/2 <= bx[2]
                     and bx[1] <= (d.bbox[1]+d.bbox[3])/2 <= bx[3]]
            target = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), default=bx)
            score = max(0, min(w, target[2])-max(0, target[0])) * max(
                0, min(h, target[3])-max(0, target[1])) / (w*h)
            if not owner.body_observed or score < .005:
                continue
            axis = reference.proof_capture_axis
            trigger_at = frame.timestamp
            if axis is not None:
                i, extent = (0, w) if axis == 'x' else (1, h)
                position = (target[i]+target[i+2])/(2*extent)
                key = (visit, camera.id)
                prior = self._positions.get(key)
                old = prior[0] if prior else None
                self._positions[key] = (position, frame.timestamp)
                line = reference.proof_capture_position
                crossed = old is not None and ((old < line <= position) if reference.proof_capture_positive
                                               else (old > line >= position))
                if not crossed and abs(position-line) > .03:
                    continue
                if previous:
                    continue  # A missed camera cannot silently move the trigger.
                if crossed and position != old and 0 < (frame.timestamp-prior[1]).total_seconds() <= 1:
                    trigger_at = prior[1] + (frame.timestamp-prior[1]) * ((line-old)/(position-old))
            elif previous and (frame.timestamp - datetime.fromisoformat(
                    previous['trigger_at'].removesuffix('Z'))).total_seconds() < .5:
                continue
            competing = any(v != visit and any(p.gate_id == owner.gate_id for p in ps)
                            for v, ps in groups.items())
            images, missing, unverified, writes = [], [], [], []
            capture_times = []
            for observer in expected:
                worker = workers.get(observer.id)
                observed = (worker.nearest_frame(trigger_at, self.settings.proof_sync_tolerance_seconds)
                            if worker and hasattr(worker, 'nearest_frame') else
                            (frame if observer.id == camera.id else None))
                if observed is None or observed.timestamp < min(p.started for p in group):
                    missing.append(observer.id)
                    continue
                owners = [p for p in group if p.camera_id == observer.id and p.source_id == observed.source_id
                          and abs((p.last_seen-observed.timestamp).total_seconds()) <= 1]
                if competing and len(owners) != 1:
                    missing.append(observer.id)
                    continue
                visible = len(owners) == 1 and owners[0].body_observed
                if visible and validator is not None:
                    visible = bool(validator(observed, owners[0].bbox))
                visible = visible and self.snapshots.compute_quality(observed.image) >= self.settings.cloud_min_proof_quality
                if not visible:
                    unverified.append(observer.id)
                ok, encoded = cv2.imencode('.jpg', observed.image, [cv2.IMWRITE_JPEG_QUALITY, 94])
                if not ok:
                    missing.append(observer.id)
                    continue
                name = f'{observer.id}_overview.jpg'
                data = encoded.tobytes()
                writes.append((name, data))
                images.append({'camera_id': observer.id, 'file': name,
                    'captured_at': observed.timestamp.isoformat() + 'Z', 'source_id': observed.source_id,
                    'sequence': observed.sequence, 'sha256': hashlib.sha256(data).hexdigest(),
                    'target_bbox': list(owners[0].bbox) if len(owners) == 1 else None,
                    'visibility_verified': visible})
                capture_times.append(observed.timestamp)
            if not images:
                continue
            # Never replace a visible truck with empty/background pixels, even
            # if a stale/spurious body box reports a larger occupied area.
            reference_image = next((i for i in images if i['camera_id'] == camera.id), None)
            if not reference_image or not reference_image['visibility_verified']:
                continue
            verified = sum(i['visibility_verified'] for i in images)
            if previous:
                old_verified = sum(i['visibility_verified'] for i in previous['images'])
                if verified < old_verified or (verified == old_verified and score <= previous.get('occupancy_score', 0)):
                    continue
            fixed = axis is not None
            spread = (max(capture_times)-min(capture_times)).total_seconds()
            complete = fixed and not missing and not unverified and spread <= self.settings.proof_sync_tolerance_seconds
            manifest = {'visit_id': visit, 'reference_camera_id': camera.id,
                'trigger_at': trigger_at.isoformat() + 'Z',
                'mode': 'CAPTURE_LINE' if fixed else 'MAX_OCCUPANCY', 'occupancy_score': score,
                'sync_tolerance_seconds': self.settings.proof_sync_tolerance_seconds,
                'capture_spread_seconds': spread, 'clock_basis': 'HOST_RECEIVE_TIME',
                'required_cameras': [c.id for c in expected], 'missing_cameras': missing,
                'unverified_cameras': unverified, 'images': images,
                'frozen': fixed, 'complete': complete,
                'state': ('READY' if complete else 'NEEDS_REVIEW') if fixed else 'NEEDS_CALIBRATION'}
            directory = self.snapshots.ensure_event_dir(visit, category='waiting')
            for name, data in writes:
                temporary = directory / (name + '.tmp')
                temporary.write_bytes(data)
                temporary.replace(directory / name)
            atomic_json(directory / 'overview.json', manifest)
