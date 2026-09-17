#!/usr/bin/env python3
"""
GateVision Universal Setup & Hardware Acceleration Manager
Detects system hardware (NVIDIA RTX 50-series Blackwell / Ada / Ampere, Intel NPU, CPU),
sets up the virtual environment with optimal PyTorch wheels, installs AI & backend
dependencies, configures databases, and builds the UI.
"""

import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"
IS_WINDOWS = platform.system() == "Windows"


def log(msg, tag="*"):
    print(f"[{tag}] {msg}", flush=True)


def log_header(title):
    print("\n" + "=" * 64, flush=True)
    print(f"   {title}", flush=True)
    print("=" * 64 + "\n", flush=True)


def get_venv_python(venv_dir: Path) -> Path:
    if IS_WINDOWS:
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def detect_hardware():
    """Detects GPU (including RTX 50-series Blackwell sm_120), NPU, and CPU."""
    info = {
        "gpu_detected": False,
        "gpu_name": None,
        "is_blackwell": False,  # RTX 5090, 5080, 5070, 5060 (sm_120)
        "npu_detected": False,
        "npu_name": None,
        "torch_install_cmd": ["torch", "torchvision"],
        "hardware_tier": "CPU",
    }

    # 1. Probe NVIDIA GPU via nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            output = subprocess.check_output(
                [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            if output:
                info["gpu_detected"] = True
                info["gpu_name"] = output.splitlines()[0].strip()
                gpu_lower = info["gpu_name"].lower()

                # Check for RTX 50-series / Blackwell architecture (sm_120)
                if any(x in gpu_lower for x in ["rtx 50", "5070", "5080", "5090", "5060", "b100", "b200", "blackwell"]):
                    info["is_blackwell"] = True
                    info["hardware_tier"] = f"NVIDIA Blackwell ({info['gpu_name']})"
                    # Blackwell requires CUDA 12.8+ wheel for sm_120 kernels
                    info["torch_install_cmd"] = [
                        "install", "--upgrade", "--pre", "torch", "torchvision",
                        "--index-url", "https://download.pytorch.org/whl/nightly/cu128"
                    ]
                else:
                    info["hardware_tier"] = f"NVIDIA CUDA ({info['gpu_name']})"
                    info["torch_install_cmd"] = [
                        "install", "torch", "torchvision",
                        "--index-url", "https://download.pytorch.org/whl/cu124"
                    ]
        except Exception:
            pass

    # 2. Probe NPU (Intel AI Boost / Neural Processing / AMD Ryzen AI)
    if IS_WINDOWS:
        try:
            ps_cmd = 'Get-PnpDevice | Where-Object { $_.FriendlyName -like "*AI Boost*" -or $_.FriendlyName -like "*Intel*NPU*" -or $_.FriendlyName -like "*Neural Processing*" -or $_.FriendlyName -like "*Ryzen AI*" } | Select-Object -ExpandProperty FriendlyName'
            res = subprocess.check_output(["powershell", "-NoProfile", "-Command", ps_cmd], text=True).strip()
            if res:
                info["npu_detected"] = True
                info["npu_name"] = res.splitlines()[0].strip()
        except Exception:
            pass
    else:
        # Linux check for /dev/accel*
        if os.path.exists("/dev/accel"):
            info["npu_detected"] = True
            info["npu_name"] = "Linux NPU Device (/dev/accel)"

    if not info["gpu_detected"]:
        if info["npu_detected"]:
            info["hardware_tier"] = f"Intel/AMD NPU ({info['npu_name']}) + CPU"
        else:
            info["hardware_tier"] = f"CPU ({platform.processor() or 'Standard'})"
        info["torch_install_cmd"] = ["install", "torch", "torchvision"]

    return info


def step_venv(venv_dir: Path) -> Path:
    log("Checking Python virtual environment...", "1/6")
    venv_python = get_venv_python(venv_dir)
    if not venv_python.exists():
        log(f"Creating virtual environment in {venv_dir}...", "...")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
    log(f"Virtual environment ready at: {venv_python}", "OK")
    return venv_python


def step_install_dependencies(venv_python: Path, hw_info: dict, skip_ai: bool = False):
    log("Upgrading pip...", "2/6")
    subprocess.check_call([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "--quiet"])

    log("Installing backend core and dev dependencies...", "...")
    dev_req = BACKEND_DIR / "requirements-dev.txt"
    if dev_req.exists():
        subprocess.check_call([str(venv_python), "-m", "pip", "install", "-r", str(dev_req), "--quiet"])

    if not skip_ai:
        log(f"Hardware Detected: {hw_info['hardware_tier']}", "AI")
        log(f"Installing PyTorch optimized for: {hw_info['hardware_tier']}...", "...")
        subprocess.check_call([str(venv_python), "-m", "pip"] + hw_info["torch_install_cmd"])

        ai_req = BACKEND_DIR / "requirements-ai.txt"
        if ai_req.exists():
            log("Installing vision and OCR dependencies (oLmOCR, Qwen2-VL, Ultralytics)...", "...")
            subprocess.check_call([str(venv_python), "-m", "pip", "install", "-r", str(ai_req), "--quiet"])


def clear_all_data():
    """Clears all events, detections, snapshots, uploads, and OCR spool jobs."""
    log("Clearing all databases, snapshots, uploads, and spool jobs...", "RESET")
    db_file = BACKEND_DIR / "data" / "gate.db"
    if db_file.exists():
        db_file.unlink()

    spool_db = BACKEND_DIR / "uploads" / "ocr-spool" / "jobs.sqlite3"
    if spool_db.exists():
        spool_db.unlink()

    for folder_name in ["snapshots", "uploads/ocr-spool/images"]:
        target = BACKEND_DIR / folder_name
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

    uploads = BACKEND_DIR / "uploads"
    if uploads.exists():
        for item in uploads.iterdir():
            if item.name == "ocr-spool":
                continue
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    venv_python = get_venv_python(BACKEND_DIR / ".venv")
    if venv_python.exists():
        try:
            subprocess.check_call([str(venv_python), "-m", "app.database.initialize"], cwd=str(BACKEND_DIR), stdout=subprocess.DEVNULL)
        except Exception:
            pass
    log("All data cleared and database schema reinitialized to fresh state.", "OK")


def step_configure_environment(venv_python: Path):
    log("Configuring backend environment and directories...", "3/6")
    env_file = BACKEND_DIR / ".env"
    env_example = BACKEND_DIR / ".env.example"
    if not env_file.exists() and env_example.exists():
        shutil.copy(str(env_example), str(env_file))
        log("Created backend/.env from .env.example", "OK")
    else:
        log("backend/.env already exists", "OK")

    # Create runtime folders
    for d in ["data", "uploads", "snapshots", "uploads/ocr-spool/images"]:
        (BACKEND_DIR / d).mkdir(parents=True, exist_ok=True)
    log("Runtime storage directories verified", "OK")

    # Initialize DB
    try:
        log("Initializing database schema...", "...")
        subprocess.check_call([str(venv_python), "-m", "app.database.initialize"], cwd=str(BACKEND_DIR))
        log("Database schema initialized", "OK")
    except Exception as e:
        log(f"Database init warning: {e}", "WARN")


def step_build_frontend(force_build: bool = False):
    log("Checking Angular frontend...", "4/6")
    dist_index = FRONTEND_DIR / "dist" / "gate" / "browser" / "index.html"
    if dist_index.exists() and not force_build:
        log("Angular frontend bundle already compiled", "OK")
        return

    npm_cmd = "npm.cmd" if IS_WINDOWS else "npm"
    if not shutil.which(npm_cmd):
        log("npm not found in PATH. Skipping frontend build (backend will run API mode).", "WARN")
        return

    log("Installing frontend npm dependencies...", "...")
    subprocess.check_call([npm_cmd, "ci"], cwd=str(FRONTEND_DIR))

    log("Building Angular production UI...", "...")
    subprocess.check_call([npm_cmd, "run", "build"], cwd=str(FRONTEND_DIR))
    log("Frontend built successfully", "OK")


def step_verify(venv_python: Path):
    log("Verifying AI acceleration and device resolution...", "5/6")
    test_code = """
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Compute Arch:    {torch.cuda.get_arch_list()}")
    try:
        x = torch.zeros(1, device='cuda')
        print("GPU Tensor Test: PASSED (Hardware Accelerated)")
    except Exception as e:
        print(f"GPU Tensor Test: FAILED ({e})")
"""
    subprocess.call([str(venv_python), "-c", test_code], cwd=str(BACKEND_DIR))


def run_full_setup(skip_ai: bool = False, force_frontend: bool = False):
    log_header("GateVision Universal Setup Manager")
    
    hw_info = detect_hardware()
    log(f"Target Platform: {platform.system()} ({platform.machine()})", "SYS")
    log(f"Detected Hardware Tier: {hw_info['hardware_tier']}", "HW")
    if hw_info["npu_detected"]:
        log(f"NPU Coprocessor: {hw_info['npu_name']} (Available for OpenVINO edge inference)", "NPU")

    venv_dir = BACKEND_DIR / ".venv"
    venv_python = step_venv(venv_dir)
    step_install_dependencies(venv_python, hw_info, skip_ai=skip_ai)
    step_configure_environment(venv_python)
    step_build_frontend(force_build=force_frontend)
    step_verify(venv_python)

    log_header("Setup Complete! Run the project with: python run.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GateVision Universal Setup Module")
    parser.add_argument("--skip-ai", action="store_true", help="Skip heavy AI model dependencies")
    parser.add_argument("--build-frontend", action="store_true", help="Force rebuild Angular frontend")
    parser.add_argument("--clear-data", action="store_true", help="Clear all database records, snapshots, spool jobs and uploads")
    args = parser.parse_args()

    if args.clear_data:
        clear_all_data()
        sys.exit(0)

    run_full_setup(skip_ai=args.skip_ai, force_frontend=args.build_frontend)
