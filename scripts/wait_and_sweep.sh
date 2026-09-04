#!/usr/bin/env bash
# The interleaved latency sweep needs the GPU to itself: sharing it measures
# contention, not the model.
#
# Both resources have to be free, not just one. A first version gated on VRAM
# alone and fired while another job still held the host memory, so the load died
# on "the paging file is too small" (os error 1455) rather than on anything to do
# with the GPU. Loading 3B in 4-bit needs roughly 7 GB of commit charge.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
deadline=$(( $(date +%s) + 21600 ))   # 6 hours

commit_free_gb() {
  powershell.exe -NoProfile -Command \
    "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)" 2>/dev/null | tr -d '\r'
}

while [ "$(date +%s)" -lt "$deadline" ]; do
  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  commit=$(commit_free_gb)
  commit=${commit:-0}
  if [ "$vram" -lt 1500 ] && [ "$commit" -ge 10 ]; then
    echo "GPU ${vram} MiB, commit ${commit} GB free at $(date '+%H:%M:%S') -- running sweep"
    "$PY" -m crucible.evaluate latency-sweep && exit 0
    echo "sweep failed; will retry"
  else
    echo "$(date '+%H:%M:%S') waiting: GPU ${vram} MiB, commit ${commit} GB free"
  fi
  sleep 180
done
echo "gave up waiting for a free machine"
exit 1
