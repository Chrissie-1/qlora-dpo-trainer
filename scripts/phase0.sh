#!/usr/bin/env bash
# Phase 0: pull the dataset, build the splits, and prefetch the base model so
# the first training run is not also a 6 GB download.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=.venv/bin/python
"$PY" -m crucible.data all
"$PY" - <<'PYEOF'
from huggingface_hub import snapshot_download
from crucible.config import BASE_MODEL
path = snapshot_download(BASE_MODEL)
print("base model at", path)
PYEOF
