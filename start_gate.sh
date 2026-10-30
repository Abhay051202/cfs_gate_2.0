#!/bin/bash
# GateVision AI - One-Command Launcher for NVIDIA Jetson AGX & Linux
# ------------------------------------------------------------------

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
cd "$BACKEND_DIR"

echo "======================================================================"
echo "          Starting GateVision AI Edge Appliance                       "
echo "======================================================================"

# Check if running on NVIDIA Jetson AGX
if [ -f "/etc/nv_tegra_release" ]; then
    echo "[*] NVIDIA Jetson Hardware Detected."
    if command -v nvpmodel >/dev/null 2>&1; then
        echo "[*] Setting Jetson Power Mode to MAXN (Maximum Performance)..."
        sudo nvpmodel -m 0 || true
    fi
    if command -v jetson_clocks >/dev/null 2>&1; then
        echo "[*] Locking CPU/GPU clocks to maximum speed..."
        sudo jetson_clocks || true
    fi
    export JETSON_NVDEC=1
    export CAMERA_BACKEND=gstreamer_nvdec
    export OCR_GPU=true
fi

# Activate Python Virtual Environment
if [ -f "$BACKEND_DIR/.venv/bin/activate" ]; then
    source "$BACKEND_DIR/.venv/bin/activate"
elif [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

export PYTHONPATH="$BACKEND_DIR"
python3 run_server.py

