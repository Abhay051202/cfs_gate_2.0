#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CMD="$(command -v python3 || command -v python || true)"

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python 3 not found in PATH." >&2
    exit 1
fi

"$PYTHON_CMD" "$ROOT_DIR/setup_system.py" "$@"
