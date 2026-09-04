# Crucible — QLoRA fine-tuning with a judge-scored DPO loop

A self-improving fine-tuning loop on one 8 GB laptop GPU: QLoRA supervised
fine-tuning, a larger model as the critic, and DPO trained on the critic's own
preferences.

The point was never that fine-tuning improves a model. It was to *measure*
whether it does, with a judge that is not told which response came from which
model, and to report the number that comes out either way.

It came out negative. Details below.

```
UltraChat 10k
     │  Phase 0: ingest, four disjoint splits
     ▼
Qwen2.5-3B-Instruct ─── Phase 1: QLoRA SFT (nf4, LoRA on attention) ──▶ adapters/sft
     │                                                                      │
     │  Phase 2: both models answer the held-out judge split                 │
     ▼                                                                      ▼
        gpt-oss-120b scores every response 0-10  ──────▶ results/judge_scores.csv
     │
     │  Phase 3: gpt-oss-20b ranks base vs SFT pairwise ──▶ data/prefs.jsonl
     ▼
DPO on those preferences ──▶ adapters/dpo
     │
     ▼  Phase 4: perplexity · judge scores · decode throughput ──▶ results/
```

## The result

**Supervised fine-tuning lowered perplexity by 12.3% and made the model worse.**

| Stage | Perplexity (held-out) | Pairwise preference vs base | Judge score (paired Δ) |
|---|---|---|---|
| base | 3.609 | — | — |
| SFT | **3.165** (−12.3%) | **loses 70 of 78** (10% win rate) | −0.21 (95% CI −0.75 to +0.32) |
| DPO | *in progress* | | |

Perplexity fell because the model matched UltraChat's distribution better.
Quality fell because that distribution is *worse than where the model started*:
Qwen2.5-3B-**Instruct** is already instruction-tuned, and UltraChat's responses
come from a weaker teacher. Fine-tuning on them is not teaching, it is
regression toward the mean of the training corpus.

This is the finding the project exists to produce. A perplexity number alone
would have reported this run as a success.

Two instruments disagreed, and the disagreement is informative:

* **Pairwise comparison** — base preferred in 70 of 78 decided pairs (90%).
* **Absolute 0-10 scoring** — a paired delta of −0.21 with a confidence
  interval straddling zero, i.e. no detectable difference.

Absolute scoring is the blunter instrument. The judge rates a competent
two-sentence answer 9.3/10 despite a rubric telling it that competent-but-
unremarkable is a 6, so nearly all responses pile up in a narrow band near the
top and real differences compress into it. Asked which of two responses is
better, the same judge separates them cleanly. If you take one methodological
point from this repo, take that one.

### Judge reliability

97 pairs were judged, each **twice, with the two responses swapped**. 78 (80%)
produced the same winner both times; the other 20% flipped with position and
were discarded rather than trained on. That 80% is the number that says the
preference signal is mostly signal.

### Cost

| Run | Wall clock | Scale |
|---|---|---|
| SFT | 168.5 min | 2923 examples, 183 optimiser steps, peak 5.20 GiB |
| DPO | 11.9 min | 78 pairs |
| Generation | 18.7 min per 250 prompts | batch 16, 72.5 tok/s |

## What the 8 GB constraint actually taught

QLoRA on a 4060 is **memory-bound, not compute-bound**, and both obvious
speedups make it slower:

| Change | Expectation | Measured |
|---|---|---|
| Batch size 1 → 2 | faster | **slower**: 0.38 → 0.251 samples/s, peak 7.21 GiB |
| Gradient checkpointing off | ~30% faster | **~40× slower**: 16 examples took 28 min |

Past roughly 7.3 GiB, Windows oversubscribes the GPU into host memory and
throughput collapses. Checkpointing keeps peak VRAM at 5.20 GiB, and paying to
recompute the forward pass is far cheaper than paging weights over PCIe. The
failure mode is not an out-of-memory error — it is silent, and it looks like
your code being slow.

A related trap: the trainer's own ETA read **14.9 h** at step 4 of a run that
takes 7.2 h, because length-grouped batching schedules the longest batches
first. The token budget (4.01 M tokens ÷ ~155 tok/s) is the honest estimate.

## Reproducing it

```bash
bash scripts/setup_env.sh        # venv + torch (cu128) + the training stack
cp .env.example .env             # add your Groq key
make data                        # Phase 0
make sft                         # Phase 1  (~2.7 h for 3000 examples)
bash scripts/autopilot.sh        # Phases 2-4, resumable
```

torch comes from the CUDA wheel index, not PyPI: the default Windows wheel is
CPU-only and bitsandbytes refuses to load against it.

`autopilot.sh` runs every GPU stage before anything that touches the API, and
guards each stage with the artifact it produces, so a crash or a quota wall
resumes instead of repeating.

## Design decisions worth defending

**Disjoint splits.** 10k conversations shuffled once, then cut into `sft`
(8000), `eval` (500), `judge` (250) and `dpo` (1250). A judge score on prompts
the model trained on measures memorisation.

**Loss on the assistant turn only.** The prompt is tokenised twice — once with
the generation prefix, once with the response appended — and the overlapping
label positions are set to −100. Training on prompt tokens teaches the model to
write user turns.

**Two different judges.** Scoring uses gpt-oss-120b; the DPO preference signal
uses gpt-oss-20b. Partly necessity (see below), but it is the better
experiment: the model that produced the training signal is not the model that
evaluates the result, so a DPO gain cannot be the policy learning its
evaluator's quirks.

**No separate reference model.** With a PEFT policy, TRL uses the same base
weights with the adapter disabled as the DPO reference — correct here, and the
only thing that fits in 8 GB alongside the policy.

**Perplexity measured through the same 4-bit quantisation used in training.** A
bf16 perplexity would describe a model this pipeline never produces.

## The rate limit is part of the story

Groq's free tier allows ~200,000 judge tokens per model per day. At ~890 tokens
per call that is ~225 calls, and this pipeline wants ~1,950 (~1.2 M tokens).
gpt-oss-120b spent its entire daily allowance scoring 223 base responses.

That shaped the code more than any other constraint:

* every judge call is cached on disk by (model, mode, prompt, responses), so a
  re-run costs only what did not finish;
* calls are paced under a rolling one-minute token budget that settles against
  the API's reported usage, because retrying a 429 cannot fix a *rate*;
* a per-day rejection fails fast and cancels the queue, while a per-minute one
  waits — retrying a daily limit means re-asking a question whose answer cannot
  change until tomorrow;
* a quota wall keeps the results already paid for instead of discarding them.

## Limitations

* **The DPO run is small.** 78 preference pairs, bounded by the daily judge
  budget. At accumulation 16 that was four optimiser steps and moved nothing
  (perplexity identical to SFT to four decimals) — the re-run uses accumulation
  4. Either way, conclusions from 78 pairs are weak and labelled as such.
* **Scoring is partial.** base 100 prompts, SFT 25, DPO pending — the paired
  delta above rests on the 25 both stages share.
* One seed per configuration. Differences of a few tenths of a judge point are
  not separable from run-to-run variance, which is why the paired delta carries
  a confidence interval.
* Latency varies ~20% run to run on a laptop GPU under thermal load; the
  reported figures come from a single session so they are comparable to each
  other.
* SFT trained on 3000 of 8000 examples (2.7 h vs 7.2 h) to leave GPU time for
  the rest of the pipeline.

## Deploying the result

```bash
python -m crucible.export merge --adapter adapters/sft --out exports/sft-merged
TESSERA_MODEL=C:/Users/ASUS/crucible/exports/sft-merged TESSERA_BACKEND=batched
```

The merge folds LoRA into the base weights so
[tessera](https://github.com/Chrissie-1/tessera-llm-inference-paged-kv-cache-speculative-decoding)
can serve it. It runs on CPU in fp16 deliberately: merging into 4-bit weights
would mean dequantise-add-requantise, which is not the model the adapter was
trained against. It should also recover the throughput an unmerged adapter
costs — measured at 22% in one session (8.25 → 6.45 tok/s).

## Tests

```bash
make test    # CPU only, no model, no network
make lint
```

Ten tests covering what fails *silently*: the loss mask covering exactly the
prompt, padding aligned with −100, the A/B verdict translated back to the right
response after the order swap, position-biased pairs dropped, score clamping,
judge cache reuse, and `.env` never overriding a real environment variable.
