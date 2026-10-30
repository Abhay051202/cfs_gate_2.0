import os
import sys
import socket
import subprocess
import platform
from pathlib import Path

# Safe encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Configure single-thread math limits and low-latency RTSP capture BEFORE native imports
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("ORT_ARENA_EXTEND_STRATEGY", "kSameAsRequested")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer+discardcorrupt|flags;low_delay|framedrop;1|max_delay;100000|probesize;131072|analyzeduration;500000"
)

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

# Auto-detect and switch to virtual environment Python if running under global python
venv_candidates = [
    APP_DIR / ".venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python"),
    APP_DIR / "venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python"),
    ROOT_DIR / ".venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python"),
]

for venv_py in venv_candidates:
    if venv_py.is_file() and Path(sys.executable).resolve() != venv_py.resolve():
        print(f"[*] Switching to virtual environment Python: {venv_py}")
        try:
            sys.exit(subprocess.call([str(venv_py), __file__] + sys.argv[1:]))
        except Exception as e:
            print(f"[!] Warning: Could not switch interpreter ({e}), continuing with current.")
        break

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
os.chdir(str(APP_DIR))

# Hardware Auto-Detection: NVIDIA Jetson AGX check
IS_JETSON = os.path.exists("/etc/nv_tegra_release")
if IS_JETSON:
    os.environ.setdefault("JETSON_NVDEC", "1")
    os.environ.setdefault("CAMERA_BACKEND", "gstreamer_nvdec")
    os.environ.setdefault("MODEL_DEVICE", "cuda")
    os.environ.setdefault("OCR_GPU", "true")
    # If TensorRT compiled models exist, activate them
    if (APP_DIR / "models" / "best.engine").is_file():
        os.environ.setdefault("MODEL_BACKEND", "tensorrt")
        os.environ.setdefault("MODEL_PATH", "models/best.engine")

def get_lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8002))
    lan_ip = get_lan_ip()
    platform_name = "NVIDIA Jetson AGX (Hardware NVDEC + TensorRT)" if IS_JETSON else f"{platform.system()} ({platform.machine()})"

    print("\n" + "=" * 70)
    print("         GATEVISION AI - REAL-TIME CONTAINER INGESTION")
    print("=" * 70)
    print(f"  * Platform Detected:              {platform_name}")
    print(f"  * Local Access (this machine):    http://localhost:{port}")
    print(f"                                   http://127.0.0.1:{port}")
    print(f"  * Network Access (LAN / Mobile):  http://{lan_ip}:{port}")
    print(f"  * API Swagger Docs:               http://{lan_ip}:{port}/docs")
    print("=" * 70)
    print(f"  Connected devices on your gate WiFi/LAN can open the dashboard at:")
    print(f"     -> http://{lan_ip}:{port}")
    print("=" * 70 + "\n")

    uvicorn.run("app.main:app", host="0.0.0.0", port=port, workers=1, reload=False, access_log=False)
