#!/usr/bin/env bash
# Latency, all three stages back to back in one session. Measured piecemeal
# across sessions the numbers moved ~20% with thermal state and whatever else
# held the GPU, which is not a difference between the models.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
"$PY" -m crucible.evaluate latency --tag base --prompts 15
"$PY" -m crucible.evaluate latency --tag sft --adapter adapters/sft --prompts 15
"$PY" -m crucible.evaluate latency --tag dpo --adapter adapters/dpo --prompts 15
