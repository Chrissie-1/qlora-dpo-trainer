"""Shared paths and constants.

Everything the pipeline writes lands under one of the three directories below,
all of which are gitignored: they are large and reproducible from scripts.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from .env without adding a dependency.

    Real environment variables win: an exported key is a deliberate override of
    the file, and silently preferring the file makes that impossible to debug.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ROOT / ".env")
DATA = ROOT / "data"
ADAPTERS = ROOT / "adapters"
RESULTS = ROOT / "results"

# Ungated, 3B, and already a chat model, so the SFT run refines an instruction
# format the base model can already produce rather than teaching it one from
# scratch. Swapping in meta-llama/Llama-3.2-3B-Instruct needs HF_TOKEN and the
# accepted licence; nothing else in the pipeline changes.
BASE_MODEL = os.environ.get("CRUCIBLE_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")

# Judge. GPT-OSS 120B on Groq: large enough that its ranking is not just the
# 3B model grading its own homework.
JUDGE_MODEL = os.environ.get("CRUCIBLE_JUDGE_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# 1024 covers the great majority of single-turn UltraChat exchanges and is what
# keeps a 3B QLoRA step inside 8 GB of VRAM.
MAX_SEQ_LEN = 1024
MAX_NEW_TOKENS = 384

SEED = 0

for _d in (DATA, ADAPTERS, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)
