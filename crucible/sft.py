"""Phase 1: QLoRA supervised fine-tuning on UltraChat.

Loss is computed on the assistant turn only. The prompt is tokenised twice --
once alone with the generation prefix, once with the response appended -- and
the label positions covered by the first are masked to -100. Training on the
prompt tokens as well would teach the model to generate user turns, which is
not what any later phase asks it for.
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import Trainer, TrainingArguments

from crucible.config import ADAPTERS, MAX_SEQ_LEN, SEED
from crucible.data import load_split
from crucible.modeling import attach_lora, load_base, load_tokenizer

IGNORE_INDEX = -100


def encode(pair: dict, tok) -> dict | None:
    """One prompt/response pair -> input_ids plus prompt-masked labels."""
    prefix = tok.apply_chat_template(
        [{"role": "user", "content": pair["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full = prefix + pair["response"] + tok.eos_token

    prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tok(full, add_special_tokens=False)["input_ids"][:MAX_SEQ_LEN]

    # A prompt that fills the window leaves nothing to supervise.
    if len(prefix_ids) >= len(full_ids):
        return None

    labels = list(full_ids)
    labels[: len(prefix_ids)] = [IGNORE_INDEX] * len(prefix_ids)
    return {"input_ids": full_ids, "labels": labels}


def build_dataset(split: str, tok, limit: int | None = None) -> list[dict]:
    rows = [encode(p, tok) for p in load_split(split, limit)]
    kept = [r for r in rows if r is not None]
    print(f"{split}: {len(kept)} examples kept of {len(rows)}")
    return kept


def collate(batch: list[dict], pad_id: int) -> dict:
    width = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, mask = [], [], []
    for b in batch:
        pad = width - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [IGNORE_INDEX] * pad)
        mask.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(mask),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="sft")
    ap.add_argument("--limit", type=int, default=None, help="train on a prefix of the split")
    ap.add_argument("--out", default=str(ADAPTERS / "sft"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    args = ap.parse_args()

    tok = load_tokenizer()
    dataset = build_dataset(args.split, tok, args.limit)

    model = attach_lora(load_base(for_training=True))

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        # Paged optimiser states spill to host memory instead of OOMing on a
        # long batch, which is the difference between finishing and not on 8 GB.
        optim="paged_adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=dataset,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )

    start = time.time()
    trainer.train()
    minutes = (time.time() - start) / 60

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved adapter to {args.out} after {minutes:.1f} min")


if __name__ == "__main__":
    main()
