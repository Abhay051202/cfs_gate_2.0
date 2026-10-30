"""Bounded full-frame evidence throughout a visit, independently of OCR latency."""
import cv2

from app.detection.tracker import iou
from app.events.passages import contains_text
from app.snapshots.overview import atomic_json


class VisitCapture:
    def __init__(self, settings, snapshots):
        self.settings, self.snapshots, self.samples = settings, snapshots, {}

    def observe(self, camera, frame, passages, detections):
        if not self.settings.visit_capture_enabled:
            return
        live = {p.visit_id or p.id for p in passages if not p.saved}
        self.samples = {k: v for k, v in self.samples.items() if k[0] in live}
        for p in passages:
            if p.saved or p.closed or p.camera_id != camera.id or p.source_id != frame.source_id:
                continue
            if p.last_seen != frame.timestamp:
                continue
            visible = any((d.class_name == 'container' and d.confidence >= .35 and (iou(d.bbox, p.bbox) >= .1 or p.capture_only)) or
                (d.class_name == 'container_number' and d.confidence >= .35 and (contains_text(p.bbox, d.bbox) or p.capture_only)
                 and d.bbox[3]-d.bbox[1] >= frame.image.shape[0]*self.settings.minimum_text_trigger_height_ratio) or
                (d.class_name in ('truck', 'trailer') and d.confidence >= .40 and (iou(d.bbox, p.bbox) >= .15 or p.capture_only))
                for d in detections)
            if not visible:
                continue
            if p.capture_only:
                p.body_observed = True
            visit, key = p.visit_id or p.id, (p.visit_id or p.id, camera.id)
            previous = self.samples.get(key)
            if previous and (frame.timestamp-previous['timestamp']).total_seconds() < self.settings.visit_capture_interval_seconds:
                continue
            quality = float(self.snapshots.compute_quality(frame.image))
            if quality < self.settings.cloud_min_proof_quality:
                continue
            old = previous['bbox'] if previous else p.bbox
            stationary = (abs(p.bbox[0]+p.bbox[2]-old[0]-old[2])/(2*frame.image.shape[1]) <= self.settings.stationary_drift_threshold and
                          abs(p.bbox[1]+p.bbox[3]-old[1]-old[3])/(2*frame.image.shape[0]) <= self.settings.stationary_drift_threshold)
            if previous and previous['count'] >= 3 and stationary and iou(p.bbox, previous['bbox']) >= .9 and quality <= previous['quality']*1.15:
                continue  # Retain motion and meaningful improvements, not minutes of duplicates.
            slot = (previous['slot']+1) % self.settings.visit_capture_max_frames if previous else 0
            ok, data = cv2.imencode('.jpg', frame.image, [cv2.IMWRITE_JPEG_QUALITY, 94])
            if not ok:
                continue
            folder = self.snapshots.ensure_event_dir(visit, category='waiting')
            path = folder / f'{camera.id}_capture_{slot:03d}.jpg'
            temporary = path.with_suffix('.jpg.tmp')
            temporary.write_bytes(data.tobytes())
            temporary.replace(path)
            cntr_boxes = [d.bbox for d in detections if d.class_name in ('container', 'container_number')]
            best_target = max(cntr_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1])) if cntr_boxes else p.bbox
            atomic_json(path.with_suffix('.json'), {'visit_id': visit, 'camera_id': camera.id,
                'captured_at': frame.timestamp.isoformat()+'Z', 'source_id': frame.source_id,
                'sequence': frame.sequence, 'target_bbox': list(best_target), 'quality': quality,
                'has_container': bool(cntr_boxes)})
            self.samples[key] = {'timestamp': frame.timestamp, 'bbox': p.bbox, 'quality': quality,
                'slot': slot, 'count': (previous['count']+1 if previous else 1)}
