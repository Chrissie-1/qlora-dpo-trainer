#!/usr/bin/env bash
# The interleaved latency sweep needs the GPU to itself: sharing it with
# another job measures the contention, not the model. Wait for it to be free,
# then run. Gives up after 3 hours rather than waiting forever.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
deadline=$(( $(date +%s) + 10800 ))

while [ "$(date +%s)" -lt "$deadline" ]; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$used" -lt 1500 ]; then
    echo "GPU free (${used} MiB) at $(date '+%H:%M:%S') -- running sweep"
    "$PY" -m crucible.evaluate latency-sweep
    exit $?
  fi
  echo "$(date '+%H:%M:%S') GPU busy (${used} MiB), waiting"
  sleep 120
done
echo "gave up waiting for a free GPU"
exit 1
