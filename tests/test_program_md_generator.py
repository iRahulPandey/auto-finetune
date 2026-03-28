"""
tests/test_program_md_generator.py — Unit tests for program.md generation.
"""

import pytest

from _core.config import RunConfig
from _core.program_md_generator import generate_program_md


def test_generate_contains_use_case():
    rc = RunConfig(use_case="Classify email urgency", task_type="classification")
    content = generate_program_md(rc)
    assert "Classify email urgency" in content


def test_generate_contains_model_key():
    rc = RunConfig(use_case="test", model_key="qwen2.5-0.5b")
    content = generate_program_md(rc)
    assert "qwen2.5-0.5b" in content


def test_generate_contains_metric():
    rc = RunConfig(use_case="test", task_type="classification")
    content = generate_program_md(rc)
    assert "accuracy" in content


def test_generate_contains_stopping_criteria():
    rc = RunConfig(use_case="test", max_iterations=15)
    content = generate_program_md(rc)
    assert "15" in content


def test_generate_contains_search_space_section():
    rc = RunConfig(use_case="test")
    content = generate_program_md(rc)
    assert "Search Space" in content
    assert "learning_rate" in content.lower() or "Learning rate" in content


def test_generate_contains_constraints_section():
    rc = RunConfig(use_case="test")
    content = generate_program_md(rc)
    assert "Constraints" in content


def test_generate_contains_layer_rationale():
    rc = RunConfig(use_case="test", task_type="extraction")
    content = generate_program_md(rc)
    assert "k_proj" in content


def test_generate_classification_recommends_q_v():
    rc = RunConfig(use_case="test", task_type="classification")
    content = generate_program_md(rc)
    assert "q_proj" in content
    assert "v_proj" in content


def test_generate_target_threshold_in_content():
    rc = RunConfig(use_case="test", task_type="classification", target_threshold=0.90)
    content = generate_program_md(rc)
    assert "0.90" in content or "0.9" in content


def test_generate_returns_string():
    rc = RunConfig(use_case="test")
    content = generate_program_md(rc)
    assert isinstance(content, str)
    assert len(content) > 100
