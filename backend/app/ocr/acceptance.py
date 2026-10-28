"""Conservative acceptance of observed identities, independent of checksum luck."""
from app.domain import OCRRead
from app.ocr.validator import ContainerValidator


def assess_identity(event, minimum=.90, minimum_cameras=2):
    detections = {d['id']: d for d in event.get('detections', []) if d.get('id')}
    numbers = [event.get('container_number')]
    if event.get('is_twin'):
        numbers.append(event.get('container_number_2'))
    supports, reasons = {}, []
    validator = ContainerValidator()
    for number in numbers:
        cameras, frames, alternatives = set(), set(), set()
        for reading in event.get('ocr_results', []):
            detection = detections.get(reading.get('detection_id'))
            if not detection:
                continue
            value = validator.validate(OCRRead(reading.get('raw_text') or '', float(reading.get('confidence') or 0)))
            if not value.valid_check_digit or value.confidence < minimum:
                continue
            if value.normalized_text != number:
                alternatives.add(value.normalized_text)
                continue
            camera, timestamp = detection.get('camera_id'), detection.get('timestamp')
            if camera and timestamp:
                cameras.add(camera)
                frames.add((camera, str(timestamp)))
        supports[number or 'UNREAD'] = {'cameras': sorted(cameras), 'independent_frames': len(frames)}
        if not number or len(frames) < 2 or len(cameras) < minimum_cameras:
            reasons.append(f'{number or "Unread identity"}: requires matching complete OCR from {minimum_cameras} cameras and at least two source frames')
        if alternatives and not event.get('is_twin'):
            reasons.append('Conflicting complete OCR identities require review: ' + ', '.join(sorted(alternatives)))
    return {'accepted': bool(numbers) and not reasons, 'supports': supports,
            'reason': '; '.join(reasons), 'minimum_confidence': minimum}
