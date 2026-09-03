# Crucible

A self-improving fine-tuning loop on one 8 GB laptop GPU: QLoRA supervised
fine-tuning, a larger model as the critic, and DPO trained on the critic's own
preferences.

The point of the project is not that fine-tuning improves a model — it is to
measure *whether it does*, with a judge that has not been told which response
came from which model, and to report the number that comes out either way.

```
UltraChat 10k
     │  Phase 0: ingest, four disjoint splits
     ▼
Qwen2.5-3B-Instruct ──── Phase 1: QLoRA SFT (nf4, LoRA on attention) ──▶ adapters/sft
     │                                                                       │
     │  Phase 2: both models answer the held-out judge split                  │
     ▼                                                                       ▼
        GPT-OSS 120B (Groq) scores every response 0-10  ──▶ results/judge_scores.csv
     │
     │  Phase 3: same judge ranks base vs SFT pairwise ──▶ data/prefs.jsonl
     ▼
DPO on those preferences ──▶ adapters/dpo
     │
     ▼  Phase 4: perplexity · judge scores · decode throughput ──▶ results/
```

## Status

Pipeline implemented and unit-tested; the training runs and their numbers are
in progress. **The results tables below are placeholders until the runs
finish** — no number appears here that was not produced by a command in this
repo.

## Setup

```bash
bash scripts/setup_env.sh          # venv + torch (cu128) + the training stack
cp .env.example .env               # then put your Groq key in it
make data                          # Phase 0
```

torch comes from the CUDA wheel index, not PyPI: the default Windows wheel is
CPU-only and bitsandbytes refuses to load against it.

## What each phase does

### Phase 0 — data (`crucible/data.py`)

Downloads `HuggingFaceH4/ultrachat_200k[:10000]` and reduces each conversation
to its first exchange, which gives an unambiguous supervision boundary: the
SFT loss mask and the DPO prompt both need to know exactly where the prompt
ends. The 10k pairs are shuffled once with a fixed seed and cut into four
**disjoint** splits:

| split | rows | used by |
|---|---|---|
| `sft` | 8000 | Phase 1 training |
| `eval` | 500 | Phase 4 perplexity |
| `judge` | 250 | Phase 2 and 4 judging |
| `dpo` | 1250 | Phase 3 preference pairs |

Disjoint matters: a judge score on prompts the SFT run trained on measures
memorisation, and DPO on its own SFT prompts re-optimises what SFT already fit.

### Phase 1 — QLoRA SFT (`crucible/sft.py`)

Base weights in nf4 with double quantisation and a bf16 compute dtype; LoRA
(r=16, α=32) on the attention projections only, as a concession to 8 GB.
Optimiser is `paged_adamw_8bit`, gradient checkpointing on, batch size 1 with
16-step accumulation.

Loss is computed on the assistant turn only. The prompt is tokenised twice —
once with the generation prefix, once with the response appended — and the
overlapping label positions are set to `-100`. Training on prompt tokens too
would teach the model to write user turns, which no later phase asks it for.

### Phase 2 — the critic (`crucible/judge.py`, `crucible/score.py`)

GPT-OSS 120B on Groq scores each response 0-10 on helpfulness, correctness and
clarity, at temperature 0, with JSON-mode replies. Every call is cached on disk
by `(judge model, mode, prompt, responses)`, so an interrupted run resumes for
free.

The headline number is the **paired** mean delta — the mean of (SFT − base)
over the prompts both models answered — with a standard error, not the
difference of two unpaired means. On 250 prompts those are different numbers,
and the paired one has less variance.

### Phase 3 — DPO on judge preferences (`crucible/prefs.py`, `crucible/dpo.py`)

The judge ranks base against SFT pairwise on the `dpo` split. Each pair is
judged **twice, with the two responses swapped**. If the judge picks the same
*position* both times it is expressing position bias rather than a preference,
and the pair is dropped. The kept fraction is reported, because it is the
honest measure of how much signal the critic actually produces: a judge that
agrees with itself on 55% of pairs is close to a coin flip, and DPO on coin
flips is noise.

DPO continues training the SFT adapter. No separate reference model is loaded —
with a PEFT policy, TRL uses the same base weights with the adapter disabled as
the reference, which is both correct here and the only thing that fits in 8 GB
alongside the policy.

### Phase 4 — evaluation (`crucible/evaluate.py`)

* **Perplexity** on the held-out `eval` split, prompt tokens masked, measured
  through the same 4-bit quantisation used in training — a bf16 perplexity
  would describe a model this pipeline never produces.
* **Judge scores** for base, SFT and DPO on the same 250 prompts.
* **Decode throughput**, single-stream, greedy, with peak GPU memory.

Perplexity is reported because it is cheap and standard, not because it is the
target. SFT should lower it; DPO optimises a preference margin and has no
reason to, and often raises it. Both go in the table as measured.

## Results

_Placeholders. Filled in from `results/metrics.json` and
`results/judge_scores.csv` when the runs complete._

| stage | perplexity (eval) | judge overall | paired Δ vs base | decode tok/s |
|---|---|---|---|---|
| base | — | — | — | — |
| SFT | — | — | — | — |
| DPO | — | — | — | — |

Training cost:

| run | wall clock | pairs/examples |
|---|---|---|
| SFT | — | — |
| DPO | — | — |

![judge scores](results/judge_scores.png)

## Deploying the result

`crucible/export.py merge` folds the adapter into the base weights and writes a
plain Transformers checkpoint — which [tessera](../tessera), the inference
stack this repo is a sibling of, can serve directly:

```bash
python -m crucible.export merge --adapter adapters/dpo --out exports/dpo-merged
TESSERA_MODEL=C:/Users/ASUS/crucible/exports/dpo-merged TESSERA_BACKEND=batched make -C ../tessera run-worker
```

The merge runs on CPU in fp16 on purpose: merging into 4-bit weights would mean
dequantise-add-requantise, which is not the model the adapter was trained
against.

`crucible/export.py onnx` exports that checkpoint through optimum for an
ONNX Runtime throughput comparison (needs `pip install -e ".[onnx]"`).

## Running the whole loop

```bash
make all     # data → sft → generate → judge → prefs → dpo → judge → eval → chart
```

or one phase at a time; every target in the `Makefile` is a single command you
can run by hand. Judge calls are cached, so re-running a phase after a crash
costs only the work that had not finished.

## Tests

```bash
make test    # CPU only, no model, no network
make lint
```

The tests cover the things that fail *silently*: the loss mask covering exactly
the prompt, padding lining up with `-100`, the A/B verdict being translated back
to the right response after the order swap, position-biased pairs being dropped,
score clamping, and the judge cache surviving a restart.

## Limitations

* One seed per configuration. Differences of a few tenths of a judge point are
  not separable from run-to-run variance, which is why the paired delta is
  reported with a standard error.
* The judge is a single model, scoring in a single language, with a fixed
  rubric. It has its own biases (length, formatting); the order swap controls
  for position bias only.
* 250 judged prompts is small. It is what a free Groq tier and one evening
  allow, and the confidence interval is reported rather than hidden.
* LoRA on attention only, 1024-token window, single epoch. All three are 8 GB
  concessions, not optimal choices.
