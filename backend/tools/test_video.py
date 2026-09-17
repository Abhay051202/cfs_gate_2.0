import sys
import cv2
import time
from pathlib import Path

# Add backend root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.config.settings import Settings
from app.detection.yolo_detector import YoloDetector
from app.ocr.ocr_engine import create_ocr_engine
from app.ocr.validator import ContainerValidator, extract_iso_size_code
from app.domain import Frame, utcnow

video_path = Path("uploads/7ccfe5f7f8a94e4c960e794566057eff.mp4")
if not video_path.is_file():
    print(f"Error: video file {video_path} does not exist!")
    sys.exit(1)

print(f"Loading video: {video_path}")
cap = cv2.VideoCapture(str(video_path))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = total_frames / fps
print(f"Video info: {width}x{height}, {total_frames} frames, {fps:.1f} FPS, {duration:.2f}s")

settings = Settings()
print(f"Settings loaded: model_device={settings.model_device}, ocr_engine={settings.ocr_engine}")

print("Initializing YOLO detector...")
detector = YoloDetector(settings)
print(f"Detector initialized. Model classes: {detector.model.names}")

print("Initializing OCR engine...")
ocr = create_ocr_engine(settings)
print("OCR engine initialized successfully.")

validator = ContainerValidator()

frame_idx = 0
step = max(1, int(fps / settings.inference_fps))  # e.g. sample at 3 FPS
detections_count = 0
ocr_readings = []

print(f"\n--- Processing video frames (sampling every {step} frames) ---")
start_time = time.time()

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    
    if frame_idx % step == 0:
        f = Frame(camera_id="test-cam", timestamp=utcnow(), sequence=frame_idx, image=img)
        dets = detector.detect(f)
        
        if dets:
            detections_count += len(dets)
            det_summary = ", ".join([f"{d.class_name} ({d.confidence:.2f})" for d in dets])
            print(f"Frame {frame_idx:4d} ({frame_idx/fps:5.2f}s): {len(dets)} detections -> {det_summary}")
            
            # If this is one of the high-confidence container detections, test OCR on it
            for d in dets:
                if d.class_name in settings.ocr_classes and d.confidence >= 0.4:
                    x1, y1, x2, y2 = [int(v) for v in d.bbox]
                    crop = img[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                    if crop.size > 0 and len(ocr_readings) < 3:
                        print(f"  -> Running OCR on crop ({crop.shape[1]}x{crop.shape[0]})...")
                        t0 = time.time()
                        ocr_res = ocr.read(crop)
                        elapsed = time.time() - t0
                        val = validator.validate(ocr_res)
                        iso_code, human_size = extract_iso_size_code(ocr_res.raw_text)
                        print(f"     Raw OCR: '{ocr_res.raw_text}' (took {elapsed:.2f}s, conf={ocr_res.confidence:.2f})")
                        print(f"     Validation: valid_format={val.valid_format}, valid_check_digit={val.valid_check_digit}, status={val.validation_status}")
                        if iso_code or human_size:
                            print(f"     Extracted Size Code: {iso_code} ({human_size})")
                        ocr_readings.append(ocr_res.raw_text)
    
    frame_idx += 1

cap.release()
total_time = time.time() - start_time
print(f"\n=======================================================")
print(f"TEST COMPLETED in {total_time:.2f}s")
print(f"Total frames processed: {frame_idx}")
print(f"Total detections found: {detections_count}")
print(f"OCR readings performed: {len(ocr_readings)}")
print(f"=======================================================")
