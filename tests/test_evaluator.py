"""
tests/test_evaluator.py — Unit tests for pure metric functions in evaluator.py.

All tests use synthetic data and never call the LLM or load a model.
compute_accuracy is tested with high-accuracy inputs so the strict path
(>= 0.8) is taken and the LLM judge is never invoked.
"""

import json
import pytest
from unittest.mock import patch

from _core.evaluator import (
    _normalize_text,
    _json_match,
    _labels_match,
    compute_f1_macro,
    compute_f1_weighted,
    compute_f1_token,
    compute_rouge_l,
    compute_bleu,
    compute_json_field_accuracy,
    compute_accuracy,
)


# ── _normalize_text ──────────────────────────────────────────────────────────

def test_normalize_text_lowercases():
    assert _normalize_text("URGENT") == "urgent"


def test_normalize_text_strips_whitespace():
    assert _normalize_text("  hello world  ") == "hello world"


def test_normalize_text_removes_punctuation():
    assert _normalize_text("NOT_URGENT.") == "noturgent"


def test_normalize_text_collapses_spaces():
    assert _normalize_text("  too   many   spaces  ") == "too many spaces"


def test_normalize_text_empty():
    assert _normalize_text("") == ""


# ── _json_match ──────────────────────────────────────────────────────────────

def test_json_match_equal_dicts():
    assert _json_match('{"a": 1, "b": 2}', '{"b": 2, "a": 1}') is True


def test_json_match_different_values():
    assert _json_match('{"a": 1}', '{"a": 2}') is False


def test_json_match_invalid_json():
    assert _json_match("not json", '{"a": 1}') is False


def test_json_match_both_invalid():
    assert _json_match("bad", "also bad") is False


def test_json_match_whitespace_differences():
    assert _json_match('{"vendor": "Acme"}', '{ "vendor" : "Acme" }') is True


# ── _labels_match ────────────────────────────────────────────────────────────

def test_labels_match_exact():
    assert _labels_match("urgent", "urgent") is True


def test_labels_match_case_insensitive():
    assert _labels_match("Urgent", "urgent") is True


def test_labels_match_with_punctuation():
    assert _labels_match("urgent!", "urgent") is True


def test_labels_match_different_labels():
    assert _labels_match("urgent", "not_urgent") is False


def test_labels_match_llm_code_prefix():
    assert _labels_match("LLM06: sensitive info", "LLM06: something else") is True


def test_labels_match_different_llm_codes():
    assert _labels_match("LLM01: prompt injection", "LLM06: data exposure") is False


def test_labels_match_json_objects():
    a = '{"vendor": "Acme", "total": 100}'
    b = '{"total": 100, "vendor": "Acme"}'
    assert _labels_match(a, b) is True


# ── compute_accuracy ─────────────────────────────────────────────────────────

def test_compute_accuracy_perfect():
    preds = ["urgent", "not_urgent", "urgent"]
    refs  = ["urgent", "not_urgent", "urgent"]
    assert compute_accuracy(preds, refs) == 1.0


def test_compute_accuracy_partial():
    preds = ["urgent", "not_urgent", "urgent", "urgent"]
    refs  = ["urgent", "not_urgent", "not_urgent", "urgent"]
    # 3/4 = 0.75 — strict path is < 0.8 so mock judge to avoid API call
    with patch("_core.evaluator._judge_batch", return_value=[True, True, False, True]):
        result = compute_accuracy(preds, refs)
    assert result == pytest.approx(0.75)


def test_compute_accuracy_empty():
    assert compute_accuracy([], []) == 0.0


# ── compute_f1_macro ─────────────────────────────────────────────────────────

def test_f1_macro_perfect():
    preds = ["urgent", "not_urgent", "urgent", "not_urgent"]
    refs  = ["urgent", "not_urgent", "urgent", "not_urgent"]
    assert compute_f1_macro(preds, refs) == pytest.approx(1.0)


def test_f1_macro_all_wrong():
    preds = ["not_urgent", "not_urgent"]
    refs  = ["urgent", "urgent"]
    assert compute_f1_macro(preds, refs) == pytest.approx(0.0)


def test_f1_macro_empty():
    assert compute_f1_macro([], []) == 0.0


def test_f1_macro_three_classes():
    preds = ["a", "b", "c", "a", "b"]
    refs  = ["a", "b", "c", "b", "a"]
    result = compute_f1_macro(preds, refs)
    assert 0.0 < result <= 1.0


# ── compute_f1_weighted ──────────────────────────────────────────────────────

def test_f1_weighted_perfect():
    preds = ["urgent"] * 3 + ["not_urgent"] * 2
    refs  = ["urgent"] * 3 + ["not_urgent"] * 2
    assert compute_f1_weighted(preds, refs) == pytest.approx(1.0)


def test_f1_weighted_empty():
    assert compute_f1_weighted([], []) == 0.0


def test_f1_weighted_between_zero_and_one():
    preds = ["urgent", "urgent", "not_urgent"]
    refs  = ["urgent", "not_urgent", "not_urgent"]
    result = compute_f1_weighted(preds, refs)
    assert 0.0 <= result <= 1.0


# ── compute_f1_token ─────────────────────────────────────────────────────────

def test_f1_token_identical():
    preds = ["the cat sat on the mat"]
    refs  = ["the cat sat on the mat"]
    assert compute_f1_token(preds, refs) == pytest.approx(1.0)


def test_f1_token_partial_overlap():
    preds = ["the cat sat on the mat"]
    refs  = ["the cat on the mat"]
    result = compute_f1_token(preds, refs)
    assert 0.0 < result < 1.0


def test_f1_token_no_overlap():
    preds = ["hello world"]
    refs  = ["completely different text"]
    assert compute_f1_token(preds, refs) == pytest.approx(0.0)


def test_f1_token_empty():
    assert compute_f1_token([], []) == 0.0


# ── compute_rouge_l ──────────────────────────────────────────────────────────

def test_rouge_l_identical():
    preds = ["feat: add login feature"]
    refs  = ["feat: add login feature"]
    assert compute_rouge_l(preds, refs) == pytest.approx(1.0)


def test_rouge_l_partial():
    preds = ["feat: add feature"]
    refs  = ["feat: add login feature"]
    result = compute_rouge_l(preds, refs)
    assert 0.0 < result < 1.0


def test_rouge_l_empty():
    assert compute_rouge_l([], []) == 0.0


# ── compute_bleu ─────────────────────────────────────────────────────────────

def test_bleu_identical():
    preds = ["fix: resolve the null pointer exception in auth"]
    refs  = ["fix: resolve the null pointer exception in auth"]
    assert compute_bleu(preds, refs) == pytest.approx(1.0)


def test_bleu_between_zero_and_one():
    preds = ["fix: resolve null pointer"]
    refs  = ["fix: resolve the null pointer exception in auth module"]
    result = compute_bleu(preds, refs)
    assert 0.0 <= result <= 1.0


def test_bleu_empty():
    assert compute_bleu([], []) == 0.0


# ── compute_json_field_accuracy ──────────────────────────────────────────────

def test_json_field_accuracy_perfect():
    preds = ['{"vendor": "Acme", "total": 100}']
    refs  = ['{"vendor": "Acme", "total": 100}']
    assert compute_json_field_accuracy(preds, refs) == pytest.approx(1.0)


def test_json_field_accuracy_partial():
    preds = ['{"vendor": "Acme", "total": 999}']
    refs  = ['{"vendor": "Acme", "total": 100}']
    # vendor matches, total doesn't → 1/2 = 0.5
    assert compute_json_field_accuracy(preds, refs) == pytest.approx(0.5)


def test_json_field_accuracy_case_insensitive_strings():
    preds = ['{"vendor": "acme corp"}']
    refs  = ['{"vendor": "Acme Corp"}']
    assert compute_json_field_accuracy(preds, refs) == pytest.approx(1.0)


def test_json_field_accuracy_invalid_prediction():
    preds = ["not valid json"]
    refs  = ['{"vendor": "Acme", "total": 100}']
    # All fields missed → 0.0
    assert compute_json_field_accuracy(preds, refs) == pytest.approx(0.0)


def test_json_field_accuracy_empty():
    assert compute_json_field_accuracy([], []) == 0.0
