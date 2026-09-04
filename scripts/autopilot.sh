#!/usr/bin/env bash
# Run Phases 3-4 end to end, unattended.
#
# Ordering rule: everything that only needs the GPU runs before anything that
# needs the judge. Groq's free tier caps a model at ~200k tokens per day, which
# this pipeline exhausts, and GPU work is the part that always makes progress.
#
# Every stage is guarded by the artifact it produces, so re-running after a
# failure or a quota wall resumes rather than repeating.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=.venv/bin/python

stage() { echo; echo "===== $* ====="; date '+%Y-%m-%d %H:%M:%S'; }

# --- GPU: generation for the preference pairs -------------------------------
if [ ! -f data/gen_dpo_base.jsonl ]; then
  stage "generate: base, DPO split (450)"
  "$PY" -m crucible.generate --split dpo --tag dpo_base --limit 450
fi

if [ ! -f data/gen_dpo_sft.jsonl ]; then
  stage "generate: SFT, DPO split (450)"
  "$PY" -m crucible.generate --split dpo --tag dpo_sft --limit 450 --adapter adapters/sft
fi

# --- GPU: the measurements that need no judge -------------------------------
stage "perplexity: base and SFT"
"$PY" -m crucible.evaluate ppl --tag base
"$PY" -m crucible.evaluate ppl --tag sft --adapter adapters/sft

stage "latency: base and SFT"
"$PY" -m crucible.evaluate latency --tag base
"$PY" -m crucible.evaluate latency --tag sft --adapter adapters/sft

# --- Judge: preferences, on the smaller model's separate quota ---------------
if [ ! -f data/prefs.jsonl ]; then
  stage "judge: pairwise preferences (each pair judged in both orders)"
  "$PY" -m crucible.prefs --a dpo_base --b dpo_sft
fi

# --- GPU: DPO, but only with enough surviving pairs to mean anything ---------
PAIRS=$(wc -l < data/prefs.jsonl 2>/dev/null || echo 0)
if [ ! -f adapters/dpo/adapter_model.safetensors ] && [ "$PAIRS" -ge 50 ]; then
  stage "train: DPO from the SFT adapter ($PAIRS pairs)"
  "$PY" -m crucible.dpo
elif [ ! -f adapters/dpo/adapter_model.safetensors ]; then
  stage "SKIPPING DPO: only $PAIRS preference pairs survived, too few to train on"
fi

if [ -f adapters/dpo/adapter_model.safetensors ]; then
  if [ ! -f data/gen_dpo.jsonl ]; then
    stage "generate: DPO, judge split"
    "$PY" -m crucible.generate --split judge --tag dpo --adapter adapters/dpo
  fi
  stage "perplexity and latency: DPO"
  "$PY" -m crucible.evaluate ppl --tag dpo --adapter adapters/dpo
  "$PY" -m crucible.evaluate latency --tag dpo --adapter adapters/dpo
fi

# --- Judge: scoring, whatever the daily budget allows ------------------------
# 100 prompts, not 250: at ~890 tokens per call, scoring three stages over 250
# prompts is 666k tokens against a 200k daily allowance. Partial results are
# cached, so a wall here costs nothing already paid for.
stage "judge: score the stages (100 prompts)"
GENS="--gen base --gen sft"
[ -f data/gen_dpo.jsonl ] && GENS="$GENS --gen dpo"
"$PY" -m crucible.score $GENS --limit 100 || echo "scoring incomplete: daily judge quota"

if [ -f results/judge_scores.csv ]; then
  stage "chart"
  "$PY" -m crucible.evaluate chart
fi

stage "done"
