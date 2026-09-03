"""Tests for the parts of the pipeline that are wrong silently.

A mis-set loss mask or a mis-translated A/B verdict does not raise; it just
trains on the wrong thing and produces a plausible-looking number at the end.
Everything here runs on CPU in under a second -- no model, no network.
"""

from __future__ import annotations

import json

import pytest

from crucible import sft
from crucible.data import _first_exchange
from crucible.judge import Judge


class FakeTokenizer:
    """Whitespace tokeniser with a ChatML-shaped template.

    Token ids are irrelevant to what the tests check (which positions are
    masked, how padding lines up), so words are hashed to ids.
    """

    eos_token = "<eos>"
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        return " ".join(parts) + " "

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [abs(hash(w)) % 1000 + 1 for w in text.split()]}


def test_encode_masks_exactly_the_prompt():
    tok = FakeTokenizer()
    example = sft.encode({"prompt": "what is rain", "response": "condensed water"}, tok)

    labels = example["labels"]
    masked = [i for i, v in enumerate(labels) if v == sft.IGNORE_INDEX]

    # The mask is a prefix, and the unmasked tail is the response.
    assert masked == list(range(len(masked)))
    assert masked, "nothing was masked"
    assert len(masked) < len(labels), "everything was masked"
    for i in range(len(masked), len(labels)):
        assert labels[i] == example["input_ids"][i]


def test_encode_drops_examples_with_no_room_for_a_response(monkeypatch):
    monkeypatch.setattr(sft, "MAX_SEQ_LEN", 4)
    tok = FakeTokenizer()
    assert sft.encode({"prompt": "a much longer prompt than four", "response": "hi"}, tok) is None


def test_collate_pads_labels_with_ignore_index():
    batch = [
        {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
        {"input_ids": [4, 5], "labels": [-100, 5]},
    ]
    out = sft.collate(batch, pad_id=0)

    assert out["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]
    assert out["labels"].tolist() == [[-100, 2, 3], [-100, 5, -100]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_first_exchange_rejects_malformed_conversations():
    assert _first_exchange({"prompt_id": "x", "messages": []}) is None
    assert (
        _first_exchange(
            {"prompt_id": "x", "messages": [{"role": "assistant", "content": "hi"}]}
        )
        is None
    )
    assert (
        _first_exchange(
            {
                "prompt_id": "x",
                "messages": [
                    {"role": "user", "content": "  "},
                    {"role": "assistant", "content": "hi"},
                ],
            }
        )
        is None
    )
    ok = _first_exchange(
        {
            "prompt_id": "x",
            "messages": [
                {"role": "user", "content": " q "},
                {"role": "assistant", "content": " a "},
                {"role": "user", "content": "later turn"},
            ],
        }
    )
    assert ok == {"prompt_id": "x", "prompt": "q", "response": "a"}


class StubJudge(Judge):
    """A Judge whose API calls are a scripted list of reply strings."""

    def __init__(self, replies, tmp_path):
        self.replies = list(replies)
        super().__init__(cache_path=tmp_path / "cache.jsonl", workers=1)

    def _chat(self, system, user, *, attempts=5):
        return self.replies.pop(0)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")


def test_compare_keeps_a_verdict_that_survives_the_order_swap(tmp_path):
    # Judge picks the first-listed response both times -> it is really picking
    # the same *response*: A first (a), then B when swapped (still a).
    judge = StubJudge(
        [json.dumps({"winner": "A"}), json.dumps({"winner": "B"})], tmp_path
    )
    verdict = judge.compare("q", "response-a", "response-b")

    assert verdict["winner"] == "a"
    assert verdict["consistent"] is True


def test_compare_drops_a_position_biased_verdict(tmp_path):
    # Judge says "A" both times -- that is the position, not a preference.
    judge = StubJudge(
        [json.dumps({"winner": "A"}), json.dumps({"winner": "A"})], tmp_path
    )
    verdict = judge.compare("q", "response-a", "response-b")

    assert verdict["winner"] == "tie"
    assert verdict["consistent"] is False
    assert (verdict["forward"], verdict["reverse"]) == ("a", "b")


def test_scores_are_clamped_and_averaged(tmp_path):
    judge = StubJudge(
        [json.dumps({"helpfulness": 12, "correctness": 6, "clarity": "bad", "reason": "r"})],
        tmp_path,
    )
    score = judge.score("q", "a")

    assert score["helpfulness"] == 10.0  # clamped from 12
    assert score["clarity"] == 0.0  # unparseable becomes 0
    assert score["overall"] == pytest.approx((10 + 6 + 0) / 3, abs=1e-3)


def test_judge_results_are_cached_across_instances(tmp_path):
    first = StubJudge([json.dumps({"helpfulness": 5, "correctness": 5, "clarity": 5})], tmp_path)
    first.score("q", "a")

    # No replies scripted: a cache miss would pop from an empty list and raise.
    second = StubJudge([], tmp_path)
    assert second.score("q", "a")["overall"] == 5.0


def test_parse_recovers_json_from_surrounding_prose():
    parsed = Judge._parse('Here is my verdict:\n```json\n{"winner": "B"}\n```')
    assert parsed == {"winner": "B"}
