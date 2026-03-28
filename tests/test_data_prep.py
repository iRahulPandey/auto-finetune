"""
tests/test_data_prep.py — Unit tests for data preparation utilities.

No LLM calls, no disk I/O (split_train_eval works in memory).
"""

import pytest

from _core.data_prep import _hash_examples, format_chat_examples, split_train_eval


def _make_examples(labels: list[str]) -> list[dict]:
    """Build minimal chat-formatted examples with the given labels."""
    return [
        {
            "messages": [
                {"role": "system", "content": "Classify the input."},
                {"role": "user", "content": f"example input {i}"},
                {"role": "assistant", "content": label},
            ]
        }
        for i, label in enumerate(labels)
    ]


# ── _hash_examples ────────────────────────────────────────────────────────────

def test_hash_examples_returns_12_chars():
    examples = [{"input": "hello", "output": "urgent"}]
    h = _hash_examples(examples)
    assert len(h) == 12


def test_hash_examples_deterministic():
    examples = [{"input": "hello", "output": "urgent"}]
    assert _hash_examples(examples) == _hash_examples(examples)


def test_hash_examples_changes_with_content():
    a = [{"input": "hello", "output": "urgent"}]
    b = [{"input": "hello", "output": "not_urgent"}]
    assert _hash_examples(a) != _hash_examples(b)


def test_hash_examples_order_sensitive():
    a = [{"input": "a", "output": "1"}, {"input": "b", "output": "2"}]
    b = [{"input": "b", "output": "2"}, {"input": "a", "output": "1"}]
    # json.dumps with sort_keys=True on the list still preserves list order
    assert _hash_examples(a) != _hash_examples(b)


# ── format_chat_examples ──────────────────────────────────────────────────────

def test_format_chat_examples_structure():
    raw = [{"input": "Is this urgent?", "output": "urgent"}]
    result = format_chat_examples(raw, "Classify the input.", "qwen2.5-0.5b")
    assert len(result) == 1
    msgs = result[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]


def test_format_chat_examples_content():
    raw = [{"input": "Server down!", "output": "urgent"}]
    result = format_chat_examples(raw, "You classify emails.", "qwen2.5-0.5b")
    msgs = result[0]["messages"]
    assert msgs[0]["content"] == "You classify emails."
    assert msgs[1]["content"] == "Server down!"
    assert msgs[2]["content"] == "urgent"


def test_format_chat_examples_preserves_count():
    raw = [{"input": f"input {i}", "output": "urgent"} for i in range(20)]
    result = format_chat_examples(raw, "Classify.", "qwen2.5-0.5b")
    assert len(result) == 20


def test_format_chat_examples_empty():
    assert format_chat_examples([], "System prompt.", "qwen2.5-0.5b") == []


# ── split_train_eval ──────────────────────────────────────────────────────────

def test_split_preserves_all_examples():
    examples = _make_examples(["urgent"] * 60 + ["not_urgent"] * 60)
    train, eval_set = split_train_eval(examples)
    assert len(train) + len(eval_set) == 120


def test_split_eval_contains_both_classes():
    examples = _make_examples(["urgent"] * 60 + ["not_urgent"] * 60)
    _, eval_set = split_train_eval(examples)
    eval_labels = {m["content"] for ex in eval_set for m in ex["messages"] if m["role"] == "assistant"}
    assert "urgent" in eval_labels
    assert "not_urgent" in eval_labels


def test_split_is_deterministic():
    examples = _make_examples(["urgent"] * 50 + ["not_urgent"] * 50)
    train1, eval1 = split_train_eval(examples)
    train2, eval2 = split_train_eval(examples)
    assert train1 == train2
    assert eval1 == eval2


def test_split_different_seeds_differ():
    examples = _make_examples(["urgent"] * 50 + ["not_urgent"] * 50)
    train1, _ = split_train_eval(examples, seed=42)
    train2, _ = split_train_eval(examples, seed=99)
    assert train1 != train2


def test_split_small_dataset_higher_eval_ratio():
    # < 80 examples → eval ratio bumped to at least 30%
    examples = _make_examples(["urgent"] * 30 + ["not_urgent"] * 30)
    train, eval_set = split_train_eval(examples)
    eval_ratio = len(eval_set) / (len(train) + len(eval_set))
    assert eval_ratio >= 0.25  # should be close to 30%


def test_split_generation_task_random_split():
    # Unique outputs → detected as generation, uses random split
    examples = _make_examples([f"unique output {i}" for i in range(50)])
    train, eval_set = split_train_eval(examples)
    assert len(train) + len(eval_set) == 50
    assert len(eval_set) >= 2
