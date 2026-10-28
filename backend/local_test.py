"""Isolated Windows test launcher. Cloud access is unconditionally disabled."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.local-vendor'))


def application(process=False):
    from dotenv import dotenv_values
    from app.config.settings import Settings
    from app.main import create_app

    original = ROOT.parents[1] / 'ai-video-based-inventory' / 'backend'
    # Only reuse camera connection settings. No cloud credentials are imported.
    values = dotenv_values(original / '.env') if (original / '.env').is_file() else {}
    for index in range(1, 9):
        key = f'CAMERA_{index}_RTSP'
        if values.get(key):
            os.environ[key] = values[key]
    settings = Settings(_env_file=None,
        local_test_mode=True, prosper_enabled=False, prosper_auto_dispatch=False,
        camera_discovery_enabled=False, pipeline_enabled=process,
        database_url='sqlite+pysqlite:///' + (ROOT / 'data' / 'gate.db').as_posix(),
        snapshot_directory=ROOT / 'snapshots', upload_directory=ROOT / 'uploads',
        log_directory=ROOT / 'logs', cameras_file=ROOT / 'config' / 'cameras.yaml',
        model_path=original / 'models' / 'best.pt',
        passage_vehicle_model_path=original / 'models' / 'yolov8x.pt',
        model_device='cuda:0', ocr_engine='paddleocr', ocr_fallback_engine='none',
        # This Windows test has no compatible TensorRT OCR engines. Report/use
        # CPU OCR explicitly rather than claim GPU execution from a flag alone.
        ocr_gpu=False, ocr_download_enabled=False,
        camera_onvif_username=values.get('CAMERA_ONVIF_USERNAME') or '',
        camera_onvif_password=values.get('CAMERA_ONVIF_PASSWORD') or '')
    (ROOT / 'data').mkdir(parents=True, exist_ok=True)
    return create_app(settings=settings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', action='store_true', help='Enable live YOLO/OCR processing')
    parser.add_argument('--port', type=int, default=8002)
    args = parser.parse_args()
    import uvicorn
    os.chdir(ROOT)
    uvicorn.run(application(args.process), host='127.0.0.1', port=args.port, workers=1, access_log=False)
