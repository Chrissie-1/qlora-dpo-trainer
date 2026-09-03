#!/usr/bin/env bash
# One-shot environment setup. torch must come from the CUDA wheel index; the
# PyPI default on Windows is CPU-only and bitsandbytes will refuse to load.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m venv .venv
PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=.venv/bin/python
"$PY" -m pip install --upgrade pip setuptools wheel
"$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -e ".[dev]"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
