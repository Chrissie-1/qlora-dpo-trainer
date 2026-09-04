PY := .venv/Scripts/python.exe

.PHONY: setup data sft gen-base gen-sft judge prefs dpo gen-dpo judge-all latency-sweep eval chart test lint all

setup:
	bash scripts/setup_env.sh

# Phase 0
data:
	$(PY) -m crucible.data all

# Phase 1. 3000 of the 8000 examples: measured 155 tok/s on the 4060, so the
# full split is a 7.2h epoch and this is 2.7h. See README for the tradeoff.
sft:
	$(PY) -m crucible.sft --limit 3000

# Phase 2
gen-base:
	$(PY) -m crucible.generate --split judge --tag base
gen-sft:
	$(PY) -m crucible.generate --split judge --tag sft --adapter adapters/sft
judge:
	$(PY) -m crucible.score --gen base --gen sft

# Phase 3
prefs:
	$(PY) -m crucible.generate --split dpo --tag dpo_base --limit 450
	$(PY) -m crucible.generate --split dpo --tag dpo_sft --limit 450 --adapter adapters/sft
	$(PY) -m crucible.prefs --a dpo_base --b dpo_sft
dpo:
	$(PY) -m crucible.dpo

# Phase 4
gen-dpo:
	$(PY) -m crucible.generate --split judge --tag dpo --adapter adapters/dpo
judge-all:
	$(PY) -m crucible.score --gen base --gen sft --gen dpo
latency-sweep:
	$(PY) -m crucible.evaluate latency-sweep

eval:
	$(PY) -m crucible.evaluate ppl --tag base
	$(PY) -m crucible.evaluate ppl --tag sft --adapter adapters/sft
	$(PY) -m crucible.evaluate ppl --tag dpo --adapter adapters/dpo
	$(PY) -m crucible.evaluate latency-sweep
chart:
	$(PY) -m crucible.evaluate chart

test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check crucible tests

# The whole loop, in order. Hours, not minutes.
all: data sft gen-base gen-sft judge prefs dpo gen-dpo judge-all eval chart
