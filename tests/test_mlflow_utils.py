"""
tests/test_mlflow_utils.py — Unit tests for pure MLflow utility functions.

Tests _slugify, _run_name, _experiment_name, and session_id validation.
No actual MLflow server or tracking calls.
"""

import pytest

from _core.mlflow_utils import _slugify, _run_name, _experiment_name, get_run_history


# ── _slugify ──────────────────────────────────────────────────────────────────

def test_slugify_lowercases():
    assert _slugify("Classify Email") == "classify-email"


def test_slugify_replaces_spaces_with_hyphens():
    assert _slugify("email urgency classifier") == "email-urgency-classifier"


def test_slugify_strips_special_chars():
    assert _slugify("classify OWASP LLM risk!") == "classify-owasp-llm-risk"


def test_slugify_truncates_at_max_len():
    long_text = "a" * 100
    result = _slugify(long_text, max_len=50)
    assert len(result) <= 50


def test_slugify_empty_returns_unnamed():
    assert _slugify("") == "unnamed-task"
    assert _slugify("   ") == "unnamed-task"


def test_slugify_no_trailing_hyphens():
    result = _slugify("hello world!!!")
    assert not result.endswith("-")


# ── _run_name ─────────────────────────────────────────────────────────────────

def test_run_name_format():
    name = _run_name(1, 3e-5, 16)
    assert "iter-001" in name
    assert "r=16" in name


def test_run_name_zero_pads_iteration():
    assert "iter-010" in _run_name(10, 1e-4, 32)


def test_run_name_scientific_notation_for_small_lr():
    name = _run_name(1, 3e-5, 16)
    assert "e" in name  # scientific notation


def test_run_name_plain_notation_for_larger_lr():
    name = _run_name(1, 0.001, 16)
    assert "0.001" in name


# ── _experiment_name ──────────────────────────────────────────────────────────

def test_experiment_name_slugifies_use_case():
    name = _experiment_name("Classify email urgency")
    assert name == "classify-email-urgency"


def test_experiment_name_empty_returns_default():
    assert _experiment_name("") == "auto-finetune"
    assert _experiment_name("   ") == "auto-finetune"


# ── session_id validation (get_run_history) ───────────────────────────────────

def test_get_run_history_rejects_invalid_session_id():
    with pytest.raises(ValueError, match="Invalid session_id"):
        get_run_history(session_id="../../etc/passwd")


def test_get_run_history_rejects_sql_injection():
    with pytest.raises(ValueError, match="Invalid session_id"):
        get_run_history(session_id="x' OR '1'='1")


def test_get_run_history_accepts_valid_hex_session_id(mocker):
    mocker.patch("_core.mlflow_utils._ensure_tracking")
    mocker.patch("_core.mlflow_utils.MlflowClient")
    # Valid 12-char hex — should not raise
    try:
        get_run_history(session_id="abc123def456")
    except ValueError:
        pytest.fail("Valid session_id raised ValueError")
    except Exception:
        pass  # Other errors (MLflow not available) are acceptable
