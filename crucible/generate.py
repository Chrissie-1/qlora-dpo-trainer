"""Sampling responses from a model, with or without an adapter.

Phase 2 needs base and SFT responses to the same prompts; Phase 3 needs the
same thing to build preference pairs. Both go through here so the only
difference between the two sets of responses is the adapter, not the decoding
settings.

Generation is batched with left padding and the batch is sorted by prompt
length, which keeps padding waste down on a GPU that has none to spare. Output
order still matches the input split.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from crucible.config import DATA, MAX_NEW_TOKENS, SEED
from crucible.data import load_split
from crucible.modeling import load_for_inference, load_tokenizer

# Leaves room for MAX_NEW_TOKENS of continuation inside the 1024-token window
# the adapter was trained under.
MAX_PROMPT_TOKENS = 640


def generate_responses(
    prompts: list[str],
    *,
    adapter: str | None,
    batch_size: int = 16,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = MAX_NEW_TOKENS,
    seed: int = SEED,
) -> tuple[list[str], dict]:
    """Returns the responses in input order plus timing/throughput stats."""
    torch.manual_seed(seed)
    tok = load_tokenizer(for_generation=True)
    model = load_for_inference(adapter)

    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    lengths = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: lengths[i])

    responses: list[str | None] = [None] * len(texts)
    new_tokens = 0
    start = time.perf_counter()

    for chunk_start in range(0, len(order), batch_size):
        idx = order[chunk_start : chunk_start + batch_size]
        batch = tok(
            [texts[i] for i in idx],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_PROMPT_TOKENS,
            add_special_tokens=False,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                pad_token_id=tok.pad_token_id,
            )

        prompt_width = batch["input_ids"].shape[1]
        for row, i in enumerate(idx):
            completion = out[row][prompt_width:]
            new_tokens += int((completion != tok.pad_token_id).sum())
            responses[i] = tok.decode(completion, skip_special_tokens=True).strip()

        done = chunk_start + len(idx)
        print(f"  {done}/{len(order)} generated", flush=True)

    elapsed = time.perf_counter() - start
    stats = {
        "adapter": adapter or "base",
        "prompts": len(prompts),
        "seconds": round(elapsed, 2),
        "new_tokens": int(new_tokens),
        "tokens_per_second": round(new_tokens / elapsed, 2) if elapsed else 0.0,
        "batch_size": batch_size,
        "temperature": temperature,
    }
    return [r or "" for r in responses], stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="judge")
    ap.add_argument("--adapter", default=None, help="path to a PEFT adapter; omit for the base")
    ap.add_argument("--tag", required=True, help="names the output file: data/gen_TAG.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rows = load_split(args.split, args.limit)
    responses, stats = generate_responses(
        [r["prompt"] for r in rows],
        adapter=args.adapter,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    out = DATA / f"gen_{args.tag}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for row, response in zip(rows, responses, strict=True):
            fh.write(
                json.dumps(
                    {
                        "prompt_id": row["prompt_id"],
                        "prompt": row["prompt"],
                        "response": response,
                        "source": args.tag,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    stats_path = DATA / f"gen_{args.tag}.stats.json"
    Path(stats_path).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"wrote {out} and {stats_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
