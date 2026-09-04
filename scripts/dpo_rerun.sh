#!/usr/bin/env bash
# 78 pairs at accumulation 16 is four optimiser steps, which left the policy
# untouched (DPO loss 0.693 -> 0.688, perplexity identical to SFT to four
# decimals). Accumulation 4 and a LoRA-appropriate learning rate give 19 steps.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
stage() { echo; echo "===== $* ====="; date '+%H:%M:%S'; }

stage "train: DPO, grad-accum 4, lr 5e-5"
"$PY" -m crucible.dpo --grad-accum 4 --lr 5e-5

stage "generate: DPO, judge split"
"$PY" -m crucible.generate --split judge --tag dpo --adapter adapters/dpo

stage "perplexity and latency: DPO"
"$PY" -m crucible.evaluate ppl --tag dpo --adapter adapters/dpo
"$PY" -m crucible.evaluate latency --tag dpo --adapter adapters/dpo
stage "done"
