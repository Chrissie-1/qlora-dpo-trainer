#!/usr/bin/env bash
# Run Phases 2-4 end to end, unattended.
#
# Every stage is guarded by the artifact it produces, so re-running after a
# failure resumes rather than repeating: generation and DPO training are hours
# of GPU time, and judge calls cost money and rate limit.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=.venv/bin/python

stage() { echo; echo "===== $* ====="; date '+%Y-%m-%d %H:%M:%S'; }

# --- Phase 2: base vs SFT on the held-out judge split ------------------------
if [ ! -f data/gen_base.jsonl ]; then
  stage "generate: base, judge split"
  "$PY" -m crucible.generate --split judge --tag base
fi

if [ ! -f data/gen_sft.jsonl ]; then
  stage "generate: SFT, judge split"
  "$PY" -m crucible.generate --split judge --tag sft --adapter adapters/sft
fi

if [ ! -f results/judge_scores_phase2.csv ]; then
  stage "judge: base vs SFT"
  "$PY" -m crucible.score --gen base --gen sft --out results/judge_scores_phase2.csv
fi

# --- Phase 3: preferences, then DPO -----------------------------------------
if [ ! -f data/gen_dpo_base.jsonl ]; then
  stage "generate: base, DPO split (450)"
  "$PY" -m crucible.generate --split dpo --tag dpo_base --limit 450
fi

if [ ! -f data/gen_dpo_sft.jsonl ]; then
  stage "generate: SFT, DPO split (450)"
  "$PY" -m crucible.generate --split dpo --tag dpo_sft --limit 450 --adapter adapters/sft
fi

if [ ! -f data/prefs.jsonl ]; then
  stage "judge: pairwise preferences (each pair judged in both orders)"
  "$PY" -m crucible.prefs --a dpo_base --b dpo_sft
fi

if [ ! -f adapters/dpo/adapter_model.safetensors ]; then
  stage "train: DPO from the SFT adapter"
  "$PY" -m crucible.dpo
fi

# --- Phase 4: evaluation -----------------------------------------------------
if [ ! -f data/gen_dpo.jsonl ]; then
  stage "generate: DPO, judge split"
  "$PY" -m crucible.generate --split judge --tag dpo --adapter adapters/dpo
fi

if [ ! -f results/judge_scores.csv ]; then
  stage "judge: base vs SFT vs DPO"
  "$PY" -m crucible.score --gen base --gen sft --gen dpo
fi

stage "perplexity"
"$PY" -m crucible.evaluate ppl --tag base
"$PY" -m crucible.evaluate ppl --tag sft --adapter adapters/sft
"$PY" -m crucible.evaluate ppl --tag dpo --adapter adapters/dpo

stage "latency"
"$PY" -m crucible.evaluate latency --tag base
"$PY" -m crucible.evaluate latency --tag sft --adapter adapters/sft
"$PY" -m crucible.evaluate latency --tag dpo --adapter adapters/dpo

stage "chart"
"$PY" -m crucible.evaluate chart

stage "done"
