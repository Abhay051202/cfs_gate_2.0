"""Isolated feasibility experiment; never starts runtime or writes application records.

Compare the first detected number crop with five quality-selected video frames.
Not a full visit tracker or an end-to-end accuracy benchmark.
"""
import collections
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from app.config.settings import Settings
from app.detection.yolo_detector import YoloDetector
from app.domain import Frame, OCRRead
from app.ocr.container_ocr import inside_gate
from app.ocr.ocr_engine import PaddleOCREngine
from app.ocr.validator import ContainerValidator


def main():
    base = Path(__file__).resolve().parents[2]
    out = base / 'docs/video-pipeline-test-2026-10-21'
    out.mkdir(exist_ok=True)
    inventory = json.loads((base / 'docs/video-review-2026-10-21/inventory.json').read_text())
    # Night OUT, day IN visit churn, clear passage, empty foreground, day OUT.
    selected = list(range(122, 126)) + list(range(302, 317)) + list(range(403, 409)) + list(range(427, 430)) + list(range(472, 480))
    settings = Settings(prosper_enabled=False, prosper_auto_dispatch=False)
    detector = YoloDetector(settings)
    reader = PaddleOCREngine(SimpleNamespace(ocr_gpu=True))
    validator = ContainerValidator()
    calibration = json.loads((base / 'backend/lane_calibration.json').read_text())
    cv2.setNumThreads(1)
    fallbacks = collections.Counter()
    profiles = {}
    for name, session in [('recognition', reader.engine.text_rec.session), ('text_detection', reader.engine.text_det.infer)]:
        if not hasattr(session, 'fallback_session'):
            raise RuntimeError(name + ' TensorRT is unavailable; refusing CPU-only experiment')
        profiles[name] = {'min': list(session.min_shape), 'max': list(session.max_shape)}
        original = session.fallback_session
        def counted(data, original=original, name=name):
            fallbacks[name] += 1
            return original(data)
        session.fallback_session = counted
    raw_engine = reader.engine
    traces = []
    class Trace:
        def __call__(self, *args, **kwargs):
            result = raw_engine(*args, **kwargs)
            items = result[0] or []
            if kwargs.get('use_det', True):
                words = [(str(x[1]), float(x[2])) for x in items if len(x) >= 3]
            else:
                words = [(str(x[0]), float(x[1])) for x in items if len(x) >= 2]
            for text, confidence in words:
                valid = validator.validate(OCRRead(text, confidence))
                traces.append({'text': text, 'confidence': confidence, 'normalized': valid.normalized_text, 'checksum_valid': valid.valid_check_digit})
            if len(words) > 1:
                joined = ' '.join(x[0] for x in words)
                valid = validator.validate(OCRRead(joined, sum(x[1] for x in words)/len(words)))
                traces.append({'text': joined, 'confidence': valid.confidence, 'normalized': valid.normalized_text, 'checksum_valid': valid.valid_check_digit})
            return result
    reader.engine = Trace()
    results = []
    started = time.monotonic()
    for index in selected:
        item = inventory[index]
        path = base / 'backend/snapshots' / item['path']
        camera = re.search(r'(gate-(?:in|out)-\d+)', path.name).group(1)
        roi = calibration.get(camera, {}).get('occupancy_roi', [0, 0, 1, 1])
        capture = cv2.VideoCapture(str(path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        step = max(1, round(fps/2))
        candidates = []
        first = None
        frame_index = sampled = number_boxes = outside = 0
        inference_seconds = 0.0
        clip_start = time.monotonic()
        while True:
            ok, image = capture.read()
            if not ok:
                break
            n = frame_index
            frame_index += 1
            if n % step:
                continue
            sampled += 1
            before = time.monotonic()
            detections = detector.detect(Frame(camera, datetime(2026, 10, 21), n, image, 'offline-video-test'))
            inference_seconds += time.monotonic() - before
            for detection in detections:
                if detection.class_name != 'container_number':
                    continue
                number_boxes += 1
                if not inside_gate(detection.bbox, image.shape, roi):
                    outside += 1
                    continue
                x1, y1, x2, y2 = map(int, detection.bbox)
                px, py = max(2, int((x2-x1)*.08)), max(2, int((y2-y1)*.15))
                crop = image[max(0,y1-py):min(image.shape[0],y2+py),max(0,x1-px):min(image.shape[1],x2+px)].copy()
                if crop.size == 0:
                    continue
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                score = math.log1p((x2-x1)*(y2-y1))*math.log1p(sharp)*detection.confidence
                candidate = {'frame': n, 'offset': n/fps, 'bbox': list(detection.bbox), 'score': score, 'sharpness': sharp, 'crop': crop, 'image': image.copy()}
                if first is None:
                    first = candidate
                candidates.append(candidate)
                # Bound stored full frames even on longer recordings.
                candidates = sorted(candidates, key=lambda c:c['score'], reverse=True)[:30]
        capture.release()
        best = []
        for candidate in candidates:
            if all(abs(candidate['offset']-old['offset']) >= .7 for old in best):
                best.append(candidate)
            if len(best) == 5:
                break
        chosen = ([first] if first else []) + [c for c in best if c is not first]
        reads = []
        for candidate in chosen:
            traces.clear()
            before = time.monotonic()
            read = reader.read(candidate['crop'])
            elapsed = time.monotonic()-before
            prefix = 'clip-%03d-frame-%05d' % (index, candidate['frame'])
            cv2.imwrite(str(out/(prefix+'-crop.jpg')),candidate['crop'])
            cv2.imwrite(str(out/(prefix+'-proof.jpg')),candidate['image'])
            reads.append({
                'frame':candidate['frame'], 'offset':candidate['offset'], 'bbox':candidate['bbox'],
                'score':candidate['score'], 'first':candidate is first, 'selected':any(candidate is b for b in best),
                'returned_text':read.raw_text, 'returned_confidence':read.confidence,
                'raw_attempts':list(traces), 'ocr_seconds':elapsed, 'proof':prefix+'-proof.jpg', 'crop':prefix+'-crop.jpg'})
        result = {'index':index,'path':item['path'],'camera':camera,'roi':roi,'decoded':frame_index,'sampled':sampled,
                  'number_boxes':number_boxes,'outside_roi':outside,'detector_seconds':inference_seconds,
                  'elapsed_seconds':time.monotonic()-clip_start,'reads':reads}
        results.append(result)
        (out/'results.json').write_text(json.dumps(results,indent=2))
        print(json.dumps({'clip':index,'sampled':sampled,'crops':len(reads),'reads':[r['returned_text'] for r in reads],'fallbacks':dict(fallbacks)}),flush=True)
    (out/'run.json').write_text(json.dumps({'indices':selected,'elapsed_seconds':time.monotonic()-started,'fallbacks':dict(fallbacks),'trt_profiles':profiles,
        'scope':'Offline candidate-frame/OCR feasibility experiment; no visit grouping, runtime, database or cloud calls.'},indent=2))


if __name__ == '__main__':
    main()
