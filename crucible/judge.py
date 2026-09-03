"""Phase 2: the critic.

A judge call is the expensive, rate-limited, non-deterministic part of this
pipeline, so everything here is built around not making the same call twice:
replies are cached on disk keyed by (judge model, mode, prompt, responses), and
a re-run after a crash costs nothing for the work already done.

Two modes:

* score   -- absolute 0-10 on helpfulness, correctness and clarity. Used to
             compare base against SFT in Phase 2.
* compare -- pairwise preference between two responses. Used to build DPO pairs
             in Phase 3. Each pair is judged twice with the responses in both
             orders; a judge that picks the same *position* both times is
             expressing position bias, not a preference, so the pair is dropped.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from tqdm import tqdm

from crucible.config import GROQ_BASE_URL, JUDGE_MODEL, RESULTS

CACHE_PATH = RESULTS / "judge_cache.jsonl"

SCORE_SYSTEM = """You are a strict evaluator of AI assistant responses.
Score the response on three axes, each an integer from 0 to 10:

- helpfulness: does it actually address what was asked, at useful depth?
- correctness: is it factually and logically sound?
- clarity: is it well organised and easy to follow?

Be discriminating. A competent but unremarkable response is a 6, not a 9.
Reserve 9-10 for responses you could not improve.

Reply with JSON only, no prose around it:
{"helpfulness": int, "correctness": int, "clarity": int, "reason": "one sentence"}"""

COMPARE_SYSTEM = """You are comparing two AI assistant responses to the same prompt.
Pick the one that is more helpful, correct and clear. Ignore length unless the
extra length adds substance, and ignore which response came first.

Reply with JSON only, no prose around it:
{"winner": "A" or "B" or "tie", "reason": "one sentence"}"""


class JudgeError(RuntimeError):
    pass


def _key(mode: str, model: str, prompt: str, responses: list[str]) -> str:
    blob = json.dumps([mode, model, prompt, responses], ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


class Judge:
    def __init__(
        self,
        model: str = JUDGE_MODEL,
        *,
        cache_path: Path = CACHE_PATH,
        workers: int = 4,
        timeout: float = 120.0,
    ):
        self.model = model
        self.workers = workers
        self.cache_path = cache_path
        self._lock = threading.Lock()
        self._cache = self._load_cache()
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise JudgeError(
                "GROQ_API_KEY is not set. Get one at https://console.groq.com/keys "
                "and put it in .env (see .env.example)."
            )
        self._client = httpx.Client(
            base_url=GROQ_BASE_URL,
            headers={"Authorization": "Bearer " + api_key},
            timeout=timeout,
        )

    def _load_cache(self) -> dict:
        if not self.cache_path.exists():
            return {}
        cache = {}
        with self.cache_path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn last line from an interrupted run
                cache[row["key"]] = row["value"]
        return cache

    def _remember(self, key: str, value: dict) -> None:
        with self._lock:
            self._cache[key] = value
            with self.cache_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

    def _chat(self, system: str, user: str, *, attempts: int = 5) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
        }
        last = None
        for attempt in range(attempts):
            try:
                r = self._client.post("/chat/completions", json=payload)
                if r.status_code == 429 or r.status_code >= 500:
                    wait = float(r.headers.get("retry-after", 0) or 0) or min(
                        2**attempt + random.random(), 30
                    )
                    time.sleep(wait)
                    last = str(r.status_code) + ": " + r.text[:200]
                    continue
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except (httpx.HTTPError, KeyError) as exc:
                last = str(exc)
                time.sleep(min(2**attempt + random.random(), 30))
        raise JudgeError("judge call failed after " + str(attempts) + " attempts: " + str(last))

    @staticmethod
    def _parse(text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Reasoning models sometimes wrap the object in prose or a fence.
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise JudgeError("no JSON object in judge reply: " + text[:200]) from None
            return json.loads(match.group(0))

    def score(self, prompt: str, response: str) -> dict:
        """Absolute 0-10 scoring. Returns the three axes plus their mean."""
        key = _key("score", self.model, prompt, [response])
        if key in self._cache:
            return self._cache[key]

        user = "### Prompt\n" + prompt + "\n\n### Response\n" + response
        raw = self._parse(self._chat(SCORE_SYSTEM, user))
        axes = {}
        for axis in ("helpfulness", "correctness", "clarity"):
            try:
                axes[axis] = max(0.0, min(10.0, float(raw.get(axis, 0))))
            except (TypeError, ValueError):
                axes[axis] = 0.0
        value = {
            **axes,
            "overall": round(sum(axes.values()) / 3, 3),
            "reason": str(raw.get("reason", ""))[:300],
        }
        self._remember(key, value)
        return value

    def compare(self, prompt: str, a: str, b: str) -> dict:
        """Pairwise preference, judged in both orders to cancel position bias.

        Returns a dict whose "winner" is "a", "b" or "tie", referring to the
        arguments rather than to the labels the judge saw.
        """
        key = _key("compare", self.model, prompt, [a, b])
        if key in self._cache:
            return self._cache[key]

        def ask(first: str, second: str) -> dict:
            user = (
                "### Prompt\n" + prompt
                + "\n\n### Response A\n" + first
                + "\n\n### Response B\n" + second
            )
            return self._parse(self._chat(COMPARE_SYSTEM, user))

        forward = ask(a, b)
        reverse = ask(b, a)

        def pick(reply: dict, label_a: str) -> str:
            """Translate the judge's A/B answer back to 'a'/'b'."""
            choice = str(reply.get("winner", "tie")).strip().upper()
            if choice == "A":
                return label_a
            if choice == "B":
                return "b" if label_a == "a" else "a"
            return "tie"

        forward_choice = pick(forward, "a")
        reverse_choice = pick(reverse, "b")  # after the swap, label A was `b`

        consistent = forward_choice == reverse_choice and forward_choice != "tie"
        value = {
            "winner": forward_choice if consistent else "tie",
            "consistent": consistent,
            "forward": forward_choice,
            "reverse": reverse_choice,
            "reason": str(forward.get("reason", ""))[:300],
        }
        self._remember(key, value)
        return value

    def map(self, fn, items: list, desc: str = "judging") -> list:
        """Run `fn` over `items` on the thread pool, preserving input order."""
        results: list = [None] * len(items)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
            for future in tqdm(futures, total=len(items), desc=desc):
                results[futures[future]] = future.result()
        return results
