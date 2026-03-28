"""
tests/test_config.py — Unit tests for RunConfig and config constants.
"""

import pytest

from _core.config import RunConfig, SUPPORTED_MODELS, TASK_TYPES, CONSTRAINTS, SEARCH_SPACE


# ── RunConfig defaults ────────────────────────────────────────────────────────

def test_runconfig_default_metric_classification():
    rc = RunConfig(use_case="test", task_type="classification")
    assert rc.metric_name == "accuracy"


def test_runconfig_default_metric_generation():
    rc = RunConfig(use_case="test", task_type="generation")
    assert rc.metric_name == "rouge_l"


def test_runconfig_default_metric_extraction():
    rc = RunConfig(use_case="test", task_type="extraction")
    assert rc.metric_name == "json_field_accuracy"


def test_runconfig_default_threshold_classification():
    rc = RunConfig(use_case="test", task_type="classification")
    assert rc.target_threshold == 0.95


def test_runconfig_default_threshold_generation():
    rc = RunConfig(use_case="test", task_type="generation")
    assert rc.target_threshold == 0.85


def test_runconfig_default_target_modules_classification():
    rc = RunConfig(use_case="test", task_type="classification")
    assert "q_proj" in rc.target_modules
    assert "v_proj" in rc.target_modules


def test_runconfig_default_target_modules_extraction():
    rc = RunConfig(use_case="test", task_type="extraction")
    assert set(rc.target_modules) == {"q_proj", "v_proj", "k_proj", "o_proj"}


def test_runconfig_lora_alpha_is_rank_times_two():
    rc = RunConfig(use_case="test", lora_rank=16)
    assert rc.lora_alpha == 32


def test_runconfig_custom_metric_not_overwritten():
    rc = RunConfig(use_case="test", task_type="classification", metric_name="f1_macro")
    assert rc.metric_name == "f1_macro"


# ── RunConfig.validate ────────────────────────────────────────────────────────

def test_validate_valid_config():
    rc = RunConfig(use_case="test", model_key="qwen2.5-0.5b", task_type="classification")
    assert rc.validate() == []


def test_validate_unknown_model():
    rc = RunConfig(use_case="test", model_key="gpt-99")
    errors = rc.validate()
    assert any("model" in e.lower() for e in errors)


def test_validate_unknown_task_type():
    rc = RunConfig(use_case="test")
    rc.task_type = "regression"  # bypass __post_init__
    errors = rc.validate()
    assert any("task" in e.lower() for e in errors)


def test_validate_learning_rate_too_high():
    rc = RunConfig(use_case="test", learning_rate=0.1)
    errors = rc.validate()
    assert any("learning_rate" in e for e in errors)


def test_validate_learning_rate_too_low():
    rc = RunConfig(use_case="test", learning_rate=1e-8)
    errors = rc.validate()
    assert any("learning_rate" in e for e in errors)


def test_validate_lora_rank_too_high():
    rc = RunConfig(use_case="test", lora_rank=128)
    errors = rc.validate()
    assert any("lora_rank" in e for e in errors)


def test_validate_epochs_out_of_range():
    rc = RunConfig(use_case="test", num_train_epochs=10)
    errors = rc.validate()
    assert any("num_train_epochs" in e for e in errors)


def test_validate_max_iterations_invalid():
    rc = RunConfig(use_case="test", max_iterations=0)
    errors = rc.validate()
    assert any("max_iterations" in e for e in errors)


# ── Model properties ──────────────────────────────────────────────────────────

def test_model_info_returns_dict():
    rc = RunConfig(use_case="test", model_key="qwen2.5-0.5b")
    info = rc.model_info
    assert "hf_id" in info
    assert "context_length" in info


def test_hf_model_id():
    rc = RunConfig(use_case="test", model_key="qwen2.5-0.5b")
    assert "Qwen" in rc.hf_model_id


# ── Constants sanity checks ───────────────────────────────────────────────────

def test_all_supported_models_have_hf_id():
    for key, info in SUPPORTED_MODELS.items():
        assert "hf_id" in info, f"{key} missing hf_id"


def test_all_task_types_have_default_metric():
    for task, info in TASK_TYPES.items():
        assert "default_metric" in info, f"{task} missing default_metric"


def test_search_space_learning_rates_within_constraints():
    for lr in SEARCH_SPACE["learning_rate"]:
        assert CONSTRAINTS["min_learning_rate"] <= lr <= CONSTRAINTS["max_learning_rate"]


def test_search_space_ranks_within_constraints():
    for rank in SEARCH_SPACE["lora_rank"]:
        assert rank <= CONSTRAINTS["max_lora_rank"]
