"""Phase 4: perplexity, latency and the summary chart.

Three subcommands:

    python -m crucible.evaluate ppl --adapter adapters/sft --tag sft
    python -m crucible.evaluate latency --adapter adapters/sft --tag sft
    python -m crucible.evaluate chart

Perplexity is measured on the held-out `eval` split with the prompt tokens
masked, so it scores the model on the responses it is supposed to produce
rather than on the prompts it is given. It is always measured through the same
4-bit quantisation used for training: a bf16 perplexity would be a number for a
model this pipeline never produces.

Perplexity is reported because it is cheap and standard, not because it is the
target. SFT on UltraChat moves the model toward one particular response style,
so perplexity on held-out UltraChat should fall; DPO optimises a preference
margin and has no reason to lower it, and often raises it. Both facts go in the
README as measured, not as expected.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from crucible.config import ADAPTERS, RESULTS
from crucible.data import load_split
from crucible.modeling import load_base, load_for_inference, load_tokenizer
from crucible.sft import encode

RESULTS_JSON = RESULTS / "metrics.json"


def _record(section: str, tag: str, payload: dict) -> None:
    """Merge one measurement into results/metrics.json."""
    data = {}
    if RESULTS_JSON.exists():
        data = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    data.setdefault(section, {})[tag] = payload
    RESULTS_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"recorded {section}.{tag} in {RESULTS_JSON}")


def perplexity(adapter: str | None, *, limit: int, split: str = "eval") -> dict:
    tok = load_tokenizer()
    model = load_for_inference(adapter)

    total_nll, total_tokens = 0.0, 0
    start = time.perf_counter()
    for i, pair in enumerate(load_split(split, limit)):
        example = encode(pair, tok)
        if example is None:
            continue
        input_ids = torch.tensor([example["input_ids"]], device=model.device)
        labels = torch.tensor([example["labels"]], device=model.device)
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits.float()

        # Standard causal shift: position t predicts token t+1.
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        counted = int((shift_labels != -100).sum())
        total_nll += float(loss)
        total_tokens += counted
        if (i + 1) % 25 == 0:
            running = float(torch.exp(torch.tensor(total_nll / total_tokens)))
            print(f"  {i + 1} examples, running ppl {running:.3f}", flush=True)

    mean_nll = total_nll / total_tokens
    return {
        "adapter": adapter or "base",
        "split": split,
        "examples": limit,
        "response_tokens": total_tokens,
        "nll": round(mean_nll, 4),
        "perplexity": round(float(torch.exp(torch.tensor(mean_nll))), 4),
        "seconds": round(time.perf_counter() - start, 1),
    }


def latency(adapter: str | None, *, quantized: bool, prompts: int, max_new_tokens: int) -> dict:
    """Single-stream decode speed: what a served request would actually see."""
    tok = load_tokenizer(for_generation=True)
    model = load_for_inference(adapter, quantized=quantized)

    rows = load_split("judge", prompts)
    per_prompt = []
    for row in rows:
        text = tok.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        batch = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start
        generated = out.shape[1] - batch["input_ids"].shape[1]
        per_prompt.append(generated / elapsed)

    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    return {
        "adapter": adapter or "base",
        "quantized": quantized,
        "prompts": len(per_prompt),
        "max_new_tokens": max_new_tokens,
        "tokens_per_second_mean": round(statistics.fmean(per_prompt), 2),
        "tokens_per_second_median": round(statistics.median(per_prompt), 2),
        "peak_gpu_gib": round(peak, 2),
    }


def latency_sweep(rounds: int, prompts: int, max_new_tokens: int) -> dict:
    """Compare base, SFT and DPO decode speed in one process.

    Measuring the three separately is worthless here: the same base config came
    back at 8.25, 6.66 and 4.34 tok/s across sessions, so run-to-run variance
    swamped the ~20% difference between the models, and one ordering even made
    DPO look faster than base -- impossible, since an adapter only adds work.

    So: load the base weights once, attach both adapters to them, and switch
    between the three with the model resident. Rounds are interleaved
    (base, sft, dpo, base, sft, dpo, ...) so drift in clocks or thermals is
    shared by all three rather than accruing to whichever ran last, and the
    reported figure is the median over rounds. A discarded warmup round absorbs
    the first-call compilation and clock ramp.
    """
    from peft import PeftModel

    tok = load_tokenizer(for_generation=True)
    base = load_base()
    model = PeftModel.from_pretrained(base, str(ADAPTERS / "sft"), adapter_name="sft")
    model.load_adapter(str(ADAPTERS / "dpo"), adapter_name="dpo")
    model.eval()
    model.config.use_cache = True

    rows = load_split("judge", prompts)
    batches = [
        tok(
            tok.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            ),
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model.device)
        for r in rows
    ]

    def measure() -> float:
        """Tokens per second over the prompt set, greedy, single stream."""
        generated, elapsed = 0, 0.0
        for batch in batches:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - start
            generated += out.shape[1] - batch["input_ids"].shape[1]
        return generated / elapsed

    def run(variant: str) -> float:
        if variant == "base":
            with model.disable_adapter():
                return measure()
        model.set_adapter(variant)
        return measure()

    variants = ["base", "sft", "dpo"]
    samples: dict[str, list[float]] = {v: [] for v in variants}

    for round_index in range(rounds + 1):
        for variant in variants:
            rate = run(variant)
            if round_index == 0:
                continue  # warmup round, discarded
            samples[variant].append(rate)
            print(f"  round {round_index} {variant:4} {rate:6.2f} tok/s", flush=True)

    result = {
        v: {
            "tokens_per_second_median": round(statistics.median(samples[v]), 2),
            "tokens_per_second_min": round(min(samples[v]), 2),
            "tokens_per_second_max": round(max(samples[v]), 2),
            "rounds": rounds,
            "prompts": prompts,
        }
        for v in variants
    }
    fastest = result["base"]["tokens_per_second_median"]
    for v in variants:
        median = result[v]["tokens_per_second_median"]
        result[v]["vs_base_pct"] = round(100 * (median - fastest) / fastest, 1)
    return result


def chart(scores_csv: str, out: str) -> None:
    import collections
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(scores_csv, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{scores_csv} is empty")

    axes = ["helpfulness", "correctness", "clarity", "overall"]
    counts = collections.Counter(r["source"] for r in rows)
    # A mean over a handful of responses is noise wearing a bar chart's clothes.
    MIN_SCORED = 10
    sources = [s for s in dict.fromkeys(r["source"] for r in rows) if counts[s] >= MIN_SCORED]
    skipped = [s for s in counts if counts[s] < MIN_SCORED]
    if skipped:
        print(f"omitting {skipped} from the chart: fewer than {MIN_SCORED} scored responses")
    if not sources:
        raise SystemExit("no source has enough scored responses to plot")
    means = {
        s: [statistics.fmean([float(r[a]) for r in rows if r["source"] == s]) for a in axes]
        for s in sources
    }
    errs = {
        s: [
            (
                statistics.stdev(v) / (len(v) ** 0.5)
                if len(v := [float(r[a]) for r in rows if r["source"] == s]) > 1
                else 0.0
            )
            for a in axes
        ]
        for s in sources
    }

    width = 0.8 / len(sources)
    positions = range(len(axes))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, source in enumerate(sources):
        offsets = [p + i * width - 0.4 + width / 2 for p in positions]
        label = f"{source} (n={counts[source]})"
        bars = ax.bar(offsets, means[source], width, yerr=errs[source], capsize=3, label=label)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(axes)
    ax.set_ylabel("judge score (0-10)")
    ax.set_ylim(0, 10)
    ax.set_title("Judge scores by pipeline stage (error bars: standard error of the mean)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ppl", help="perplexity on the held-out eval split")
    p.add_argument("--adapter", default=None)
    p.add_argument("--tag", required=True)
    p.add_argument("--limit", type=int, default=200)

    lat = sub.add_parser("latency", help="single-stream decode throughput")
    lat.add_argument("--adapter", default=None)
    lat.add_argument("--tag", required=True)
    lat.add_argument("--prompts", type=int, default=10)
    lat.add_argument("--max-new-tokens", type=int, default=128)
    lat.add_argument("--no-quant", action="store_true", help="load bf16 instead of 4-bit")

    sweep = sub.add_parser("latency-sweep", help="interleaved base/SFT/DPO decode speed")
    sweep.add_argument("--rounds", type=int, default=3)
    sweep.add_argument("--prompts", type=int, default=8)
    sweep.add_argument("--max-new-tokens", type=int, default=128)

    c = sub.add_parser("chart", help="bar chart of judge scores")
    c.add_argument("--scores", default=str(RESULTS / "judge_scores.csv"))
    c.add_argument("--out", default=str(RESULTS / "judge_scores.png"))

    args = ap.parse_args()

    if args.command == "ppl":
        result = perplexity(args.adapter, limit=args.limit)
        print(json.dumps(result, indent=2))
        _record("perplexity", args.tag, result)
    elif args.command == "latency":
        result = latency(
            args.adapter,
            quantized=not args.no_quant,
            prompts=args.prompts,
            max_new_tokens=args.max_new_tokens,
        )
        print(json.dumps(result, indent=2))
        _record("latency", args.tag, result)
    elif args.command == "latency-sweep":
        result = latency_sweep(args.rounds, args.prompts, args.max_new_tokens)
        print(json.dumps(result, indent=2))
        for tag, payload in result.items():
            _record("latency_sweep", tag, payload)
    else:
        chart(args.scores, args.out)


if __name__ == "__main__":
    main()
