#!/usr/bin/env python3
"""
Forwarder for GateVision Setup Manager when called from inside backend/.
"""
from pathlib import Path
import sys
import subprocess

root_setup = Path(__file__).resolve().parent.parent / "setup_system.py"
if not root_setup.exists():
    print(f"Error: Could not locate root setup manager at {root_setup}", file=sys.stderr)
    sys.exit(1)

cmd = [sys.executable, str(root_setup)] + sys.argv[1:]
sys.exit(subprocess.call(cmd))
