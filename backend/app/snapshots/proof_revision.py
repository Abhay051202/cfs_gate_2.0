"""Review a clearer source frame without moving the overview capture trigger."""
import base64
from datetime import datetime

import cv2
import numpy as np

from app.domain import Frame
from app.snapshots.outgoing_preview import OutgoingPreview


class ProofRevision:
    def __init__(self, repository, snapshots, runtime):
        self.repository, self.snapshots, self.runtime = repository, snapshots, runtime
        self.store = OutgoingPreview(snapshots.root)

    def candidate(self, event_id, snapshot_id):
        event = self.repository.event(event_id)
        payload = self.store.load(event_id)
        if not event or event.get('status') != 'CONFIRMED' or not payload or not payload.get('_proof_complete'):
            raise ValueError('A confirmed, complete approved proof is required')
        snap = self.repository.snapshot(snapshot_id)
        if not snap or snap.get('event_id') != event_id:
            raise ValueError('Source frame does not belong to this event')
        camera = snap['camera_id']
        current = next((i for c in payload['Containers'] for i in c['Images'] if i['CameraId'] == camera), None)
        overview = payload.get('_overview') or {}
        source = next((i for i in overview.get('images', []) if i['camera_id'] == camera), None)
        if not current or not source or not source.get('target_bbox'):
            raise ValueError('No fixed-position target is available for this camera')
        parse = lambda value: datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        if abs((parse(snap['timestamp'])-parse(overview['trigger_at'])).total_seconds()) > overview.get('sync_tolerance_seconds', .3):
            raise ValueError('Source frame is outside the fixed capture time')
        path = self.snapshots.path(snap['image_path'])
        data = path.read_bytes()
        if not data.startswith(b'\xff\xd8'):
            raise ValueError('Use an original JPEG proof frame')
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        old = cv2.imdecode(np.frombuffer(base64.b64decode(current['Image'].split(',', 1)[1]), np.uint8), cv2.IMREAD_COLOR)
        if image is None or old is None or image.shape != old.shape:
            raise ValueError('Replacement dimensions must match the original overview')
        x1, y1, x2, y2 = map(int, source['target_bbox'])
        def quality(frame):
            roi = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
            return float(self.snapshots.compute_quality(roi)) if roi.size else 0
        score, previous = quality(image), quality(old)
        if score < self.runtime.settings.cloud_min_proof_quality or score <= previous * 1.05:
            raise ValueError('Replacement must be clearer in the container region')
        return event, payload, snap, source, image, data, score

    def candidates(self, event_id):
        event = self.repository.event(event_id)
        if not event:
            return []
        result = []
        for snap in event.get('snapshots', [])[:50]:
            try:
                _, _, snap, _, _, _, score = self.candidate(event_id, snap['id'])
                result.append({'snapshot_id': snap['id'], 'camera_id': snap['camera_id'],
                    'captured_at': snap['timestamp'], 'quality': score})
            except (ValueError, OSError, KeyError):
                continue
        return result

    def approve(self, event_id, snapshot_id, reason):
        if not reason.strip():
            raise ValueError('A review reason is required')
        event, approved, snap, source, image, data, _ = self.candidate(event_id, snapshot_id)
        if self.runtime.model_status.get('detector') != 'READY':
            raise ValueError('Detector must be ready to verify the replacement frame')
        timestamp = datetime.fromisoformat(snap['timestamp'].removesuffix('Z'))
        frame = Frame(snap['camera_id'], timestamp, source['sequence'], image, source['source_id'])
        if not self.runtime._overview_visibility(frame, tuple(source['target_bbox'])):
            raise ValueError('Container is not verified in the replacement frame')
        if not self.runtime._proof_position_matches(frame, tuple(source['target_bbox'])):
            raise ValueError('Truck position differs from the fixed overview proof')
        # Persist a retry before publishing new bytes so an interruption cannot
        # leave a newer approved file permanently outside the durable outbox.
        self.repository.mark_proof_pending(event_id, int(approved.get('ProofVersion') or 1))
        payload = self.store.revise_camera(event_id, snap['camera_id'], data,
            snap['timestamp'], snap['image_path'], reason, expected_digest=self.store.digest(approved))
        version = payload['ProofVersion']
        self.repository.mark_proof_pending(event_id, version)
        self.repository.log_system_event('proof_revision_approved', event_id=event_id,
            context={'proof_version': version, 'snapshot_id': snapshot_id, 'reason': reason})
        dispatcher = getattr(self.runtime, 'dispatcher', None)
        if (not self.runtime.settings.local_test_mode and dispatcher and
                dispatcher.enabled and dispatcher.auto_dispatch):
            if event.get('cloud_event_id') and getattr(self.runtime, 'proof_upgrader', None):
                self.runtime.proof_upgrader.enqueue_upgrade(event_id)
            else:
                dispatcher.enqueue(None, event_id)
        return {'success': True, 'proof_version': version,
            'payload_sha256': self.store.digest(payload),
            'local_test_mode': self.runtime.settings.local_test_mode,
            'cloud_update_pending': bool(event.get('cloud_event_id')),
            'message': 'Proof revision saved locally. Cloud acknowledgment is tracked separately.'}
