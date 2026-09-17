import math
from app.domain import Detection, Frame


class YoloDetector:
    def __init__(self, settings, model=None):
        self.settings = settings
        from app.device import resolve_optimal_device
        self.device, self.has_cuda = resolve_optimal_device(settings.model_device)
        if model is None:
            expected = {'pytorch': '.pt', 'onnx': '.onnx', 'tensorrt': '.engine'}[settings.model_backend]
            if settings.model_path.suffix != expected or not settings.model_path.is_file():
                raise ValueError('Model file missing or backend does not match extension')
            from ultralytics import YOLO
            model = YOLO(str(settings.model_path), task='detect')
        self.model = model

    def detect(self, frame: Frame) -> list[Detection]:
        try:
            results = self.model.predict(source=frame.image, conf=self.settings.confidence_threshold,
                                         imgsz=self.settings.model_image_size, device=self.device,
                                         verbose=False, rect=False)
        except Exception:
            if self.device != 'cpu':
                self.device = 'cpu'
                results = self.model.predict(source=frame.image, conf=self.settings.confidence_threshold,
                                             imgsz=self.settings.model_image_size, device='cpu',
                                             verbose=False, rect=False)
            else:
                raise
        height, width = frame.image.shape[:2]
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                name = str(result.names[int(box.cls.item())]).lower()
                name = self.settings.class_aliases.get(name, name)
                if name not in self.settings.target_classes:
                    continue
                confidence = float(box.conf.item())
                coords = [float(v) for v in box.xyxy[0].tolist()]
                if not all(math.isfinite(v) for v in coords + [confidence]):
                    continue
                x1, y1, x2, y2 = coords
                bbox = (max(0., x1), max(0., y1), min(float(width), x2), min(float(height), y2))
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    detections.append(Detection(frame.camera_id, frame.timestamp, name, confidence, bbox))
        return detections
