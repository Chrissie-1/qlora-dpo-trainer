"""Phase 3, step 1: turn judge preferences into a DPO dataset.

Reads two generation files over the same prompts, asks the judge which response
it prefers, and writes prompt/chosen/rejected triples.

Two rows are deliberately thrown away:

* ties -- there is no preference to learn from;
* pairs the judge ranked inconsistently under swapped presentation order, which
  `Judge.compare` already collapses to a tie.

The proportion kept is reported, because it is the honest measure of how much
signal the critic is actually producing. A judge that agrees with itself on 55%
of pairs is close to a coin flip, and DPO on coin flips is noise.
"""

from __future__ import annotations

import argparse
import json

from crucible.config import DATA, PREF_MODEL
from crucible.judge import Judge
from crucible.score import load_generations


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", default="base", help="generation tag for candidate A")
    ap.add_argument("--b", default="sft", help="generation tag for candidate B")
    ap.add_argument("--out", default=str(DATA / "prefs.jsonl"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--model", default=PREF_MODEL, help="judge used for the preferences")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    a_rows = {r["prompt_id"]: r for r in load_generations(args.a)}
    b_rows = {r["prompt_id"]: r for r in load_generations(args.b)}
    shared = [p for p in a_rows if p in b_rows]
    if args.limit:
        shared = shared[: args.limit]
    if not shared:
        raise SystemExit(f"no prompt overlap between gen_{args.a} and gen_{args.b}")

    judge = Judge(model=args.model, workers=args.workers)
    print(f"preference judge: {judge.model}")
    pairs = [(a_rows[p], b_rows[p]) for p in shared]
    verdicts = judge.map(
        lambda pair: judge.compare(pair[0]["prompt"], pair[0]["response"], pair[1]["response"]),
        pairs,
        desc=f"comparing {args.a} vs {args.b}",
    )

    kept, ties, empties = [], 0, 0
    unjudged = 0
    for (a, b), verdict in zip(pairs, verdicts, strict=True):
        if verdict is None:
            unjudged += 1
            continue
        if verdict["winner"] == "tie":
            ties += 1
            continue
        chosen, rejected = (b, a) if verdict["winner"] == "b" else (a, b)
        # An empty response is a generation failure, not a preference.
        if not chosen["response"].strip() or not rejected["response"].strip():
            empties += 1
            continue
        kept.append(
            {
                "prompt_id": a["prompt_id"],
                "prompt": a["prompt"],
                "chosen": chosen["response"],
                "rejected": rejected["response"],
                "chosen_source": chosen["source"],
            }
        )

    with open(args.out, "w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    chose_b = sum(1 for r in kept if r["chosen_source"] == args.b)
    print(f"wrote {args.out}")
    print(f"  {len(pairs) - unjudged} of {len(pairs)} pairs judged"
          + (f" ({unjudged} unjudged: quota)" if unjudged else ""))
    judged = len(pairs) - unjudged
    print(f"  {len(kept)} kept ({len(kept) / judged:.0%} of judged)" if judged else "  none judged")
    print(f"  {ties} dropped as ties or order-inconsistent, {empties} dropped as empty")
    if kept:
        print(f"  judge preferred {args.b} in {chose_b}/{len(kept)} decided pairs "
              f"({chose_b / len(kept):.0%})")


if __name__ == "__main__":
    main()
