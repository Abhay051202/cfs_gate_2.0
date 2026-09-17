#!/usr/bin/env python3
"""
Forwarder for GateVision Runner when called from inside backend/.
"""
from pathlib import Path
import sys
import subprocess

root_run = Path(__file__).resolve().parent.parent / "run.py"
if not root_run.exists():
    print(f"Error: Could not locate root runner at {root_run}", file=sys.stderr)
    sys.exit(1)

try:
    cmd = [sys.executable, str(root_run)] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
except KeyboardInterrupt:
    sys.exit(0)
