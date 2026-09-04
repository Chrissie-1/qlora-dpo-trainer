#!/usr/bin/env bash
# A/B the gradient-checkpointing tradeoff on identical data.
set -u
PY=/c/Users/ASUS/crucible/.venv/Scripts/python.exe
cd /c/Users/ASUS/crucible
for variant in off on; do
  echo "=== gradient checkpointing $variant ==="
  flag=""
  [ "$variant" = "off" ] && flag="--no-grad-ckpt"
  "$PY" -m crucible.sft --limit 32 --grad-accum 4 $flag --out "adapters/ab_$variant" \
    > "ab_$variant.log" 2>&1
  tr '\r' '\n' < "ab_$variant.log" \
    | grep -aE "train_runtime|peak GPU|out of memory|Error" | tail -3
done
