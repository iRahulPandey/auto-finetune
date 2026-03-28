"""
tests/test_agent_loop.py — Unit tests for agent loop validation and config parsing.

No LLM calls, no subprocess execution, no file I/O beyond reading finetune.py.
"""

import pytest

from _core.agent_loop import (
    _ALLOWED_BIAS,
    _ALLOWED_OPTIM,
    _ALLOWED_SCHEDULERS,
    _extract_config_from_finetune,
    _read_finetune_py,
    _validate_proposed_config,
)


def _good_config(**overrides) -> dict:
    base = {
        "hypothesis": "Testing a lower learning rate to reduce overfitting.",
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
        },
        "training_args": {
            "learning_rate": 3e-5,
            "num_train_epochs": 2,
            "lr_scheduler_type": "cosine",
            "optim": "adamw_torch",
        },
    }
    base.update(overrides)
    return base


# ── Valid config passes ───────────────────────────────────────────────────────


def test_valid_config_no_errors():
    assert _validate_proposed_config(_good_config()) == []


def test_valid_all_four_target_modules():
    cfg = _good_config()
    cfg["lora_config"]["target_modules"] = ["q_proj", "v_proj", "k_proj", "o_proj"]
    assert _validate_proposed_config(cfg) == []


def test_valid_all_schedulers():
    for sched in _ALLOWED_SCHEDULERS:
        cfg = _good_config()
        cfg["training_args"]["lr_scheduler_type"] = sched
        assert _validate_proposed_config(cfg) == [], f"Scheduler {sched!r} should be valid"


def test_valid_all_bias_values():
    for bias in _ALLOWED_BIAS:
        cfg = _good_config()
        cfg["lora_config"]["bias"] = bias
        assert _validate_proposed_config(cfg) == [], f"Bias {bias!r} should be valid"


def test_valid_all_optim_values():
    for optim in _ALLOWED_OPTIM:
        cfg = _good_config()
        cfg["training_args"]["optim"] = optim
        assert _validate_proposed_config(cfg) == [], f"Optim {optim!r} should be valid"


# ── Numeric constraints ───────────────────────────────────────────────────────


def test_lora_rank_too_high():
    cfg = _good_config()
    cfg["lora_config"]["r"] = 128
    errors = _validate_proposed_config(cfg)
    assert any("lora_rank" in e for e in errors)


def test_learning_rate_too_high():
    cfg = _good_config()
    cfg["training_args"]["learning_rate"] = 0.1
    errors = _validate_proposed_config(cfg)
    assert any("learning_rate" in e for e in errors)


def test_learning_rate_too_low():
    cfg = _good_config()
    cfg["training_args"]["learning_rate"] = 1e-9
    errors = _validate_proposed_config(cfg)
    assert any("learning_rate" in e for e in errors)


def test_epochs_too_high():
    cfg = _good_config()
    cfg["training_args"]["num_train_epochs"] = 10
    errors = _validate_proposed_config(cfg)
    assert any("num_train_epochs" in e for e in errors)


def test_epochs_too_low():
    cfg = _good_config()
    cfg["training_args"]["num_train_epochs"] = 0
    errors = _validate_proposed_config(cfg)
    assert any("num_train_epochs" in e for e in errors)


# ── Missing hypothesis ────────────────────────────────────────────────────────


def test_missing_hypothesis():
    cfg = _good_config()
    del cfg["hypothesis"]
    errors = _validate_proposed_config(cfg)
    assert any("hypothesis" in e.lower() for e in errors)


# ── String field allowlist validation ─────────────────────────────────────────


def test_invalid_scheduler_rejected():
    cfg = _good_config()
    cfg["training_args"]["lr_scheduler_type"] = "warmup_only"
    errors = _validate_proposed_config(cfg)
    assert any("lr_scheduler_type" in e for e in errors)


def test_injection_attempt_in_scheduler_rejected():
    cfg = _good_config()
    cfg["training_args"]["lr_scheduler_type"] = 'cosine"\nimport os\nos.system("rm -rf /")\n#'
    errors = _validate_proposed_config(cfg)
    assert any("lr_scheduler_type" in e for e in errors)


def test_invalid_bias_rejected():
    cfg = _good_config()
    cfg["lora_config"]["bias"] = "everything"
    errors = _validate_proposed_config(cfg)
    assert any("bias" in e for e in errors)


def test_invalid_optim_rejected():
    cfg = _good_config()
    cfg["training_args"]["optim"] = "sgd_custom"
    errors = _validate_proposed_config(cfg)
    assert any("optim" in e for e in errors)


def test_invalid_target_modules_rejected():
    cfg = _good_config()
    cfg["lora_config"]["target_modules"] = ["q_proj", "embed_tokens"]
    errors = _validate_proposed_config(cfg)
    assert any("target_modules" in e for e in errors)


# ── Multiple errors reported together ────────────────────────────────────────


def test_multiple_errors_all_reported():
    cfg = _good_config()
    cfg["lora_config"]["r"] = 512
    cfg["training_args"]["learning_rate"] = 99.0
    cfg["training_args"]["num_train_epochs"] = 99
    errors = _validate_proposed_config(cfg)
    assert len(errors) >= 3


# ── _extract_config_from_finetune ─────────────────────────────────────────────


def test_extract_config_has_hypothesis():
    source = _read_finetune_py()
    config = _extract_config_from_finetune(source)
    assert "hypothesis" in config
    assert isinstance(config["hypothesis"], str)
    assert len(config["hypothesis"]) > 0


def test_extract_config_has_lora_config():
    source = _read_finetune_py()
    config = _extract_config_from_finetune(source)
    assert "lora_config" in config
    lora = config["lora_config"]
    assert "r" in lora
    assert "target_modules" in lora


def test_extract_config_has_training_args():
    source = _read_finetune_py()
    config = _extract_config_from_finetune(source)
    assert "training_args" in config
    ta = config["training_args"]
    assert "learning_rate" in ta
    assert "num_train_epochs" in ta
