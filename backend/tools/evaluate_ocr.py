"""Offline exact-match evaluation of independently labelled number images.

Input CSV: image,container_number,condition. Images may be original number
crops from any day/night/blur bucket. This does not measure truck association.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ocr.validator import ContainerValidator


def evaluate(rows, root, reader):
    root = Path(root).resolve()
    validator, results, buckets = ContainerValidator(), [], {}
    for row in rows:
        path = (root / row['image']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Image path escapes dataset root')
        expected = row['container_number'].strip().upper()
        from app.domain import OCRRead
        if not validator.validate(OCRRead(expected, 1)).valid_check_digit:
            raise ValueError(f'Invalid independently labelled ground truth: {expected}')
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Unreadable input image: {path.name}')
        started = time.perf_counter()
        raw = reader.read(image)
        value = validator.validate(raw)
        predicted = value.normalized_text if value.valid_check_digit else None
        exact = predicted == expected
        condition = row.get('condition') or 'unspecified'
        bucket = buckets.setdefault(condition, {'images': 0, 'correct': 0})
        bucket['images'] += 1
        bucket['correct'] += int(exact)
        results.append({'image': row['image'], 'expected': expected, 'predicted': predicted,
            'raw_text': raw.raw_text, 'confidence': raw.confidence, 'exact': exact,
            'latency_ms': round((time.perf_counter()-started)*1000, 2), 'condition': condition})
    count = len(results)
    if not count:
        raise ValueError('Provide at least one independently labelled image')
    for bucket in buckets.values():
        bucket['exact_accuracy'] = bucket['correct']/bucket['images']
    return {'scope': 'OCR image exact match; not end-to-end visit accuracy', 'images': count,
        'exact_accuracy': sum(r['exact'] for r in results)/count,
        'conditions': buckets, 'results': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('labels', type=Path)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    from app.config.settings import Settings
    from app.ocr.ocr_engine import create_ocr_engine
    settings = Settings(_env_file=None, local_test_mode=True, ocr_engine='paddleocr',
        ocr_gpu=args.gpu, ocr_download_enabled=False, ocr_fallback_engine='none')
    reader = create_ocr_engine(settings)
    try:
        with args.labels.open(encoding='utf-8-sig', newline='') as source:
            report = evaluate(list(csv.DictReader(source)), args.images, reader)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in report.items() if k != 'results'}))
    finally:
        if hasattr(reader, 'close'):
            reader.close()
