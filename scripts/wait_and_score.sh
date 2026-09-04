#!/usr/bin/env bash
# Finish the judge scoring once gpt-oss-120b's daily token budget resets.
#
# There is no header that reports remaining daily tokens, so the probe is the
# work itself: scoring fails fast on a per-day rejection and every answered call
# is cached, so a premature attempt costs seconds and loses nothing. Retries
# every 30 minutes for 18 hours.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
deadline=$(( $(date +%s) + 64800 ))

while [ "$(date +%s)" -lt "$deadline" ]; do
  echo "=== scoring attempt $(date '+%Y-%m-%d %H:%M:%S') ==="
  if "$PY" -m crucible.score --gen base --gen sft --gen dpo --limit 100; then
    scored=$("$PY" -c "import csv,collections;print(dict(collections.Counter(r['source'] for r in csv.DictReader(open('results/judge_scores.csv',encoding='utf-8')))))")
    echo "scored: $scored"
    # Done when every stage has a full set; otherwise the quota cut it short.
    case "$scored" in
      *"'dpo': 100"*) echo "all stages scored"; "$PY" -m crucible.evaluate chart; exit 0 ;;
    esac
  fi
  echo "quota not yet reset (or partially spent); sleeping 30 min"
  sleep 1800
done
echo "gave up waiting for the judge quota"
exit 1
