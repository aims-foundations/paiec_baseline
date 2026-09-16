#!/usr/bin/env bash
# Package BLE with explicit public/private credential handling.
# Usage: ./tools/build_zip.sh --public|--private bundle|hf [owner/repository]
set -euo pipefail
SCRIPT_DIRECTORY=$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve().parent)' "$0")
exec python3 "$SCRIPT_DIRECTORY/ble_packaging.py" "$@"
