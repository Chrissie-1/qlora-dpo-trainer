"""Phase 2 driver: judge one or more generation files and write judge_scores.csv.

Usage:

    python -m crucible.score --gen base --gen sft

Each `--gen TAG` reads data/gen_TAG.jsonl (written by crucible.generate). Every
response is scored 0-10 on three axes, and the per-source means plus the paired
delta land in results/judge_scores.csv.

The delta reported is the *paired* mean difference -- the mean of
(sft - base) over prompts both models answered -- not the difference of the two
means. On a small judge set the two are not the same number, and the paired one
is the one with less variance.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics

from crucible.config import DATA, RESULTS
from crucible.judge import Judge

AXES = ("helpfulness", "correctness", "clarity", "overall")


def load_generations(tag: str) -> list[dict]:
    path = DATA / f"gen_{tag}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} is missing -- run `python -m crucible.generate --tag {tag}`")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen", action="append", required=True, help="generation tag; repeatable")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(RESULTS / "judge_scores.csv"))
    args = ap.parse_args()

    judge = Judge(workers=args.workers)
    rows: list[dict] = []

    for tag in args.gen:
        generations = load_generations(tag)[: args.limit]
        scores = judge.map(
            lambda g: judge.score(g["prompt"], g["response"]),
            generations,
            desc=f"judging {tag}",
        )
        for gen, score in zip(generations, scores, strict=True):
            if score is None:
                continue
            rows.append(
                {
                    "prompt_id": gen["prompt_id"],
                    "source": tag,
                    **{axis: score[axis] for axis in AXES},
                    "reason": score["reason"],
                }
            )

    if not rows:
        raise SystemExit("no responses were scored -- the judge quota is spent")

    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["prompt_id", "source", *AXES, "reason"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} scored responses)")

    print("\nmean scores")
    print(f"{'source':<12}" + "".join(f"{axis:>14}" for axis in AXES))
    by_source = {}
    for tag in args.gen:
        subset = [r for r in rows if r["source"] == tag]
        if not subset:
            continue
        by_source[tag] = {r["prompt_id"]: r for r in subset}
        means = {axis: statistics.fmean([r[axis] for r in subset]) for axis in AXES}
        print(f"{tag:<12}" + "".join(f"{means[axis]:>14.2f}" for axis in AXES))

    # Paired deltas against the first tag, which is the baseline by convention.
    baseline, *others = args.gen
    for tag in others:
        # A stage the quota cut short has no rows at all; skip rather than fail.
        if baseline not in by_source or tag not in by_source:
            print(f"\nno paired comparison for {tag}: one of the two stages went unscored")
            continue
        shared = set(by_source[baseline]) & set(by_source[tag])
        if not shared:
            continue
        diffs = [by_source[tag][p]["overall"] - by_source[baseline][p]["overall"] for p in shared]
        mean = statistics.fmean(diffs)
        stdev = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        stderr = stdev / (len(diffs) ** 0.5) if diffs else 0.0
        wins = sum(1 for d in diffs if d > 0)
        losses = sum(1 for d in diffs if d < 0)
        print(
            f"\n{tag} vs {baseline} on {len(shared)} shared prompts:\n"
            f"  paired mean delta {mean:+.3f} overall (95% CI "
            f"{mean - 1.96 * stderr:+.3f} to {mean + 1.96 * stderr:+.3f})\n"
            f"  {wins} wins / {losses} losses / {len(diffs) - wins - losses} ties"
        )


if __name__ == "__main__":
    main()
