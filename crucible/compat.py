"""Keyword filtering for APIs that move between library versions.

transformers 5 and TRL 1.x both renamed and removed arguments this pipeline
uses (`warmup_ratio`, `group_by_length`, `tokenizer`). Passing an unknown
keyword is a hard TypeError, so the call sites declare what they want and let
this drop whatever the installed version does not accept -- loudly, so a
silently missing setting shows up in the log.
"""

from __future__ import annotations

import inspect


def supported(cls, kwargs: dict) -> dict:
    accepted = set(inspect.signature(cls.__init__).parameters)
    dropped = sorted(k for k in kwargs if k not in accepted)
    if dropped:
        print(f"note: {cls.__name__} does not accept {dropped} in this version; ignoring")
    return {k: v for k, v in kwargs.items() if k in accepted}
