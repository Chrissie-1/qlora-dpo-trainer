"""Phase 0: dataset ingestion and the splits every later phase draws from.

UltraChat rows are multi-turn. The pipeline uses only the first exchange of
each conversation, which makes a prompt/response pair with an unambiguous
supervision boundary -- the loss mask in `sft.py` and the DPO prompt in
`dpo.py` both need to know exactly where the prompt ends.

The four splits are disjoint, so a judge score on `judge` is never a score on
something the SFT run memorised, and DPO never re-optimises SFT's own prompts.
"""

from __future__ import annotations

import argparse
import json

from datasets import Dataset, load_dataset

from crucible.config import DATA, SEED

SOURCE = "HuggingFaceH4/ultrachat_200k"
SLICE = "train_sft[:10000]"
RAW = DATA / "ultrachat_10k.parquet"

# 8000 + 500 + 250 + 1250 == 10000, in that order after a seeded shuffle.
SPLIT_SIZES = {"sft": 8000, "eval": 500, "judge": 250, "dpo": 1250}


def ingest() -> None:
    """Download the 10k slice and cache it as parquet."""
    ds = load_dataset(SOURCE, split=SLICE)
    ds.to_parquet(RAW)
    print(f"wrote {RAW} ({len(ds)} rows)")


def _first_exchange(row: dict) -> dict | None:
    messages = row["messages"]
    if len(messages) < 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        return None
    prompt, response = messages[0]["content"].strip(), messages[1]["content"].strip()
    if not prompt or not response:
        return None
    return {"prompt_id": row["prompt_id"], "prompt": prompt, "response": response}


def build_splits() -> None:
    """Turn the raw parquet into four disjoint jsonl splits."""
    if not RAW.exists():
        raise SystemExit(f"{RAW} is missing -- run `python -m crucible.data ingest` first")

    ds = Dataset.from_parquet(str(RAW)).shuffle(seed=SEED)
    pairs = [p for p in (_first_exchange(r) for r in ds) if p is not None]
    print(f"{len(pairs)} usable single-turn pairs out of {len(ds)} conversations")

    start = 0
    for name, size in SPLIT_SIZES.items():
        chunk = pairs[start : start + size]
        if len(chunk) < size:
            print(f"warning: {name} got {len(chunk)} rows, wanted {size}")
        path = DATA / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in chunk:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {path} ({len(chunk)} rows)")
        start += size


def load_split(name: str, limit: int | None = None) -> list[dict]:
    path = DATA / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} is missing -- run `python -m crucible.data splits` first")
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    return rows[:limit] if limit else rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["ingest", "splits", "all"])
    args = ap.parse_args()
    if args.command in ("ingest", "all"):
        ingest()
    if args.command in ("splits", "all"):
        build_splits()


if __name__ == "__main__":
    main()
