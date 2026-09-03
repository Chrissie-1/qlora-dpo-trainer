"""Phase 3, step 2: DPO on the judge's preferences.

The policy starts from the SFT adapter and keeps training the same LoRA
weights. No separate reference model is loaded: with a PEFT policy, TRL uses
the same base weights with the adapter disabled as the reference, which is both
the correct reference for this setup and the only one that fits in 8 GB
alongside the policy.

TRL renamed the trainer's tokenizer argument to `processing_class` and moved
several DPOConfig fields between releases, so the keyword names are checked
against the installed signatures rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import math
import time

from datasets import Dataset
from peft import PeftModel
from trl import DPOConfig, DPOTrainer

from crucible.compat import supported
from crucible.config import ADAPTERS, DATA, MAX_SEQ_LEN, SEED
from crucible.modeling import load_base, load_tokenizer


def load_prefs(path: str, limit: int | None = None) -> Dataset:
    """Preference rows in TRL's conversational format."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            rows.append(
                {
                    "prompt": [{"role": "user", "content": row["prompt"]}],
                    "chosen": [{"role": "assistant", "content": row["chosen"]}],
                    "rejected": [{"role": "assistant", "content": row["rejected"]}],
                }
            )
    if limit:
        rows = rows[:limit]
    if not rows:
        raise SystemExit(f"{path} has no preference pairs")
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefs", default=str(DATA / "prefs.jsonl"))
    ap.add_argument("--init-adapter", default=str(ADAPTERS / "sft"))
    ap.add_argument("--out", default=str(ADAPTERS / "dpo"))
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tok = load_tokenizer()
    dataset = load_prefs(args.prefs, args.limit)
    print(f"{len(dataset)} preference pairs")

    # transformers 5 dropped warmup_ratio; DPOConfig inherits that signature.
    steps = math.ceil(len(dataset) * args.epochs / (args.batch_size * args.grad_accum))
    warmup_steps = max(5, round(0.03 * steps))
    print(f"{steps} optimiser steps, {warmup_steps} of them warmup")

    base = load_base(for_training=True)
    policy = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
    policy.print_trainable_parameters()

    config = DPOConfig(
        **supported(
            DPOConfig,
            {
                "output_dir": args.out,
                "beta": args.beta,
                "learning_rate": args.lr,
                "num_train_epochs": args.epochs,
                "per_device_train_batch_size": args.batch_size,
                "gradient_accumulation_steps": args.grad_accum,
                "lr_scheduler_type": "cosine",
                "warmup_steps": warmup_steps,
                "optim": "paged_adamw_8bit",
                "bf16": True,
                "gradient_checkpointing": True,
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
                "max_length": MAX_SEQ_LEN,
                "max_prompt_length": MAX_SEQ_LEN // 2,
                "logging_steps": 5,
                "save_strategy": "no",
                "report_to": [],
                "seed": SEED,
            },
        )
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=None,  # the adapter-disabled policy is the reference
        args=config,
        train_dataset=dataset,
        **supported(DPOTrainer, {"processing_class": tok, "tokenizer": tok}),
    )

    start = time.time()
    trainer.train()
    minutes = (time.time() - start) / 60

    trainer.model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved adapter to {args.out} after {minutes:.1f} min")

    with open(f"{args.out}/train_stats.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"minutes": round(minutes, 2), "pairs": len(dataset), "beta": args.beta},
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()
