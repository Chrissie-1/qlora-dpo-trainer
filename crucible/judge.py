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
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from tqdm import tqdm

from crucible.config import GROQ_BASE_URL, JUDGE_MODEL, RESULTS

CACHE_PATH = RESULTS / "judge_cache.jsonl"

# Groq's free tier meters tokens per minute, not requests, and gpt-oss is a
# reasoning model whose hidden reasoning tokens are billed to that budget too.
# Retrying a 429 does not help when the limit is a rate: the fix is to pace the
# calls under it. 7000 leaves headroom under the 8000 TPM ceiling for the
# accounting drift between our estimate and Groq's.
TOKENS_PER_MINUTE = int(os.environ.get("CRUCIBLE_JUDGE_TPM", "7000"))


class TokenBudget:
    """A rolling one-minute token budget shared by every judge thread.

    Callers reserve an estimate before the request and settle the difference
    afterwards from the API's own usage figure, so a model that reasons more
    than expected slows the next call down instead of triggering a 429.
    """

    def __init__(self, tokens_per_minute: int = TOKENS_PER_MINUTE):
        self.limit = tokens_per_minute
        self._events: deque[list[float]] = deque()
        self._lock = threading.Lock()

    def _trim(self, now: float) -> None:
        while self._events and now - self._events[0][0] > 60:
            self._events.popleft()

    def reserve(self, estimate: int) -> list[float]:
        while True:
            with self._lock:
                now = time.monotonic()
                self._trim(now)
                if sum(e[1] for e in self._events) + estimate <= self.limit or not self._events:
                    entry = [now, float(estimate)]
                    self._events.append(entry)
                    return entry
                wait = 60 - (now - self._events[0][0])
            time.sleep(min(max(wait, 0.5), 10) + random.random())

    def settle(self, entry: list[float], actual: int) -> None:
        with self._lock:
            entry[1] = float(actual)

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
        workers: int = 2,
        timeout: float = 120.0,
    ):
        self.model = model
        self.workers = workers
        self.cache_path = cache_path
        self.budget = TokenBudget()
        # gpt-oss accepts a reasoning_effort hint; other models 400 on it, so it
        # is dropped permanently the first time the API rejects it.
        self._reasoning_effort: str | None = "low"
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

    def _chat(self, system: str, user: str, *, attempts: int = 6) -> str:
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
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort

        # Roughly four characters per token, plus the reply we are asking for.
        estimate = (len(system) + len(user)) // 4 + 200

        last = None
        for attempt in range(attempts):
            entry = self.budget.reserve(estimate)
            try:
                r = self._client.post("/chat/completions", json=payload)
                if r.status_code == 400 and "reasoning_effort" in r.text:
                    self._reasoning_effort = None
                    payload.pop("reasoning_effort", None)
                    continue
                if r.status_code == 429 or r.status_code >= 500:
                    # Spend the reservation: a refused call still consumed the
                    # budget upstream, and the retry-after header is the truth.
                    wait = float(r.headers.get("retry-after", 0) or 0) or min(
                        2**attempt + random.random(), 30
                    )
                    time.sleep(wait)
                    last = str(r.status_code) + ": " + r.text[:200]
                    continue
                r.raise_for_status()
                remaining = r.headers.get("x-ratelimit-remaining-requests")
                if remaining is not None and int(remaining) < 3:
                    raise JudgeError(
                        f"{self.model} has {remaining} requests left in its daily quota "
                        f"(resets in {r.headers.get('x-ratelimit-reset-requests', '?')}). "
                        "Cached results are kept; re-run when the quota resets."
                    )
                body = r.json()
                self.budget.settle(entry, body.get("usage", {}).get("total_tokens", estimate))
                return body["choices"][0]["message"]["content"]
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

    def map(self, fn, items: list, desc: str = "judging", *, tolerate: bool = True) -> list:
        """Run `fn` over `items` on the thread pool, preserving input order.

        With `tolerate`, a failed call becomes None instead of killing the run.
        The daily token quota is the expected failure here, and it arrives
        partway through: discarding the hundreds of results that did land --
        and the API budget they cost -- to re-raise one exception would be the
        expensive way to handle it. Callers report the gap.
        """
        results: list = [None] * len(items)
        failures = 0
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
            for future in tqdm(futures, total=len(items), desc=desc):
                try:
                    results[futures[future]] = future.result()
                except JudgeError as exc:
                    failures += 1
                    if failures == 1:
                        print(f"judge stopped answering: {exc}")
                    if not tolerate:
                        raise
        if failures:
            print(f"{failures}/{len(items)} judge calls did not complete")
        return results
