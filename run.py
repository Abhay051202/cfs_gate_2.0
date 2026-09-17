#!/usr/bin/env python3
"""
GateVision Universal One-Command Project Runner
Checks for setup completion (auto-running setup_system.py if needed),
resolves hardware acceleration (NVIDIA RTX 50-series Blackwell / CUDA / CPU),
and launches the unified GateVision service with full UI and API support.
"""

import sys
import os
import subprocess
import shutil
import platform
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"
IS_WINDOWS = platform.system() == "Windows"


def get_venv_python():
    if IS_WINDOWS:
        return BACKEND_DIR / ".venv" / "Scripts" / "python.exe"
    return BACKEND_DIR / ".venv" / "bin" / "python"


def is_setup_complete():
    venv_python = get_venv_python()
    if not venv_python.exists():
        return False
    frontend_index = FRONTEND_DIR / "dist" / "gate" / "browser" / "index.html"
    if not frontend_index.exists():
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="GateVision One-Command Runner")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind (default: 8001)")
    parser.add_argument("--setup", action="store_true", help="Force run the setup module before starting")
    parser.add_argument("--skip-ai", action="store_true", help="Skip heavy AI setup if running setup")
    parser.add_argument("--clear-data", action="store_true", help="Clear all events, snapshots, spool jobs and uploads before starting")
    args = parser.parse_args()

    if args.clear_data:
        setup_script = ROOT_DIR / "setup_system.py"
        subprocess.call([sys.executable, str(setup_script), "--clear-data"])

    # If setup is requested or missing, run setup_system.py
    if args.setup or not is_setup_complete():
        print("\n[*] Initial setup is required or was requested. Launching Setup Manager...", flush=True)
        setup_script = ROOT_DIR / "setup_system.py"
        cmd = [sys.executable, str(setup_script)]
        if args.skip_ai:
            cmd.append("--skip-ai")
        ret = subprocess.call(cmd)
        if ret != 0:
            print("[!] Setup failed. Please review errors above.", file=sys.stderr)
            sys.exit(ret)

    venv_python = get_venv_python()
    if not venv_python.exists():
        print("[!] Fatal: Python virtual environment not found.", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 64)
    print("                GateVision CFS Gate Module")
    print("=" * 64)
    print(f"  Web Dashboard:  http://{args.host}:{args.port}")
    print(f"  API Docs:       http://{args.host}:{args.port}/docs")
    print(f"  Python Runtime: {venv_python}")
    print("=" * 64)
    print("  Press Ctrl+C to shut down cleanly.\n", flush=True)

    uvicorn_cmd = [
        str(venv_python), "-m", "uvicorn", "app.main:app",
        "--host", args.host,
        "--port", str(args.port),
        "--workers", "1",
        "--no-access-log"
    ]

    try:
        subprocess.run(uvicorn_cmd, cwd=str(BACKEND_DIR))
    except KeyboardInterrupt:
        print("\n[*] GateVision stopped cleanly.")


if __name__ == "__main__":
    main()
