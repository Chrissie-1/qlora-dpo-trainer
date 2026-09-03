"""Export a trained adapter for serving.

    python -m crucible.export merge --adapter adapters/dpo --out exports/dpo-merged
    python -m crucible.export onnx  --merged exports/dpo-merged --out exports/dpo-onnx

`merge` folds the LoRA weights into the base weights and writes a plain
Transformers checkpoint. It deliberately does *not* load the base in 4-bit:
merging into quantised weights means dequantising, adding, and re-quantising,
which changes the weights the adapter was trained against. The merge therefore
runs on CPU in fp16 -- slow, but it produces the checkpoint anything else can
load, including tessera's worker:

    TESSERA_MODEL=C:/Users/ASUS/crucible/exports/dpo-merged TESSERA_BACKEND=batched

`onnx` shells out to optimum's exporter. It needs the `onnx` extra
(`pip install -e ".[onnx]"`) and roughly 7 GB of free disk for a 3B model.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from crucible.config import BASE_MODEL
from crucible.modeling import load_tokenizer


def merge(adapter: str, out: str) -> None:
    print(f"loading {BASE_MODEL} in fp16 on CPU (this needs ~7 GB of RAM)")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.float16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(base, adapter)
    print("merging adapter weights")
    model = model.merge_and_unload()

    Path(out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    load_tokenizer().save_pretrained(out)
    print(f"wrote merged checkpoint to {out}")


def onnx(merged: str, out: str, opset: int) -> None:
    cmd = [
        sys.executable,
        "-m",
        "optimum.exporters.onnx",
        "--model",
        merged,
        "--task",
        "text-generation-with-past",
        "--opset",
        str(opset),
        out,
    ]
    print(" ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(
            "ONNX export failed. It needs the onnx extra: "
            'pip install -e ".[onnx]"'
        )
    print(f"wrote ONNX model to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    m = sub.add_parser("merge", help="fold a LoRA adapter into the base weights")
    m.add_argument("--adapter", required=True)
    m.add_argument("--out", required=True)

    o = sub.add_parser("onnx", help="export a merged checkpoint to ONNX")
    o.add_argument("--merged", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--opset", type=int, default=17)

    args = ap.parse_args()
    if args.command == "merge":
        merge(args.adapter, args.out)
    else:
        onnx(args.merged, args.out, args.opset)


if __name__ == "__main__":
    main()
