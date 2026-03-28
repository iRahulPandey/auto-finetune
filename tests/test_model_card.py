"""
tests/test_model_card.py — Unit tests for HuggingFace model card generation.
"""

from app import _build_model_card


def _sample_params() -> dict:
    return {
        "lora_r": "16",
        "lora_alpha": "32",
        "lora_dropout": "0.05",
        "target_modules": '["q_proj", "v_proj"]',
        "learning_rate": "3e-5",
        "num_train_epochs": "2",
        "lr_scheduler_type": "cosine",
        "batch_size": "2",
        "gradient_accumulation_steps": "4",
        "warmup_ratio": "0.1",
    }


def test_model_card_contains_frontmatter():
    card = _build_model_card(
        repo_id="user/test-model",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        use_case="Classify emails",
        metric_name="accuracy",
        metric_value=0.92,
        lora_params=_sample_params(),
        training_params=_sample_params(),
        system_prompt="You are an email classifier.",
        hypothesis="Higher rank improves accuracy",
        iteration=3,
    )
    assert "library_name: peft" in card
    assert "base_model: Qwen/Qwen2.5-0.5B-Instruct" in card
    assert "auto-finetune" in card


def test_model_card_contains_usage_code():
    card = _build_model_card(
        repo_id="user/test-model",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        use_case="Classify emails",
        metric_name="accuracy",
        metric_value=0.92,
        lora_params=_sample_params(),
        training_params=_sample_params(),
        system_prompt="You are an email classifier.",
        hypothesis="Higher rank",
        iteration=3,
    )
    assert "from peft import PeftModel" in card
    assert 'PeftModel.from_pretrained(model, "user/test-model")' in card
    assert "AutoTokenizer.from_pretrained" in card
    assert "model.generate" in card


def test_model_card_contains_metrics():
    card = _build_model_card(
        repo_id="user/test-model",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        use_case="Classify emails",
        metric_name="accuracy",
        metric_value=0.8765,
        lora_params=_sample_params(),
        training_params=_sample_params(),
        system_prompt="",
        hypothesis="",
        iteration=5,
    )
    assert "**accuracy**" in card
    assert "**0.8765**" in card
    assert "Iteration | 5" in card


def test_model_card_contains_lora_config():
    card = _build_model_card(
        repo_id="user/test-model",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        use_case="Extract fields",
        metric_name="json_field_accuracy",
        metric_value=0.85,
        lora_params=_sample_params(),
        training_params=_sample_params(),
        system_prompt="",
        hypothesis="",
        iteration=1,
    )
    assert "Rank (r) | 16" in card
    assert "Alpha | 32" in card
    assert "Dropout | 0.05" in card


def test_model_card_contains_training_args():
    card = _build_model_card(
        repo_id="user/test-model",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        use_case="Summarize text",
        metric_name="rouge_l",
        metric_value=0.71,
        lora_params=_sample_params(),
        training_params=_sample_params(),
        system_prompt="",
        hypothesis="",
        iteration=2,
    )
    assert "Learning rate | 3e-5" in card
    assert "Epochs | 2" in card
    assert "Scheduler | cosine" in card


def test_model_card_includes_system_prompt_when_present():
    card = _build_model_card(
        repo_id="user/test-model",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        use_case="Classify emails",
        metric_name="accuracy",
        metric_value=0.9,
        lora_params=_sample_params(),
        training_params=_sample_params(),
        system_prompt="You are an email classifier.",
        hypothesis="",
        iteration=1,
    )
    assert "You are an email classifier." in card
    assert '{"role": "system"' in card


def test_model_card_no_system_prompt():
    card = _build_model_card(
        repo_id="user/test-model",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        use_case="Classify emails",
        metric_name="accuracy",
        metric_value=0.9,
        lora_params=_sample_params(),
        training_params=_sample_params(),
        system_prompt="",
        hypothesis="",
        iteration=1,
    )
    assert "No system prompt recorded" in card
    assert '{"role": "system"' not in card


def test_model_card_contains_hypothesis():
    card = _build_model_card(
        repo_id="user/test-model",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        use_case="Classify",
        metric_name="accuracy",
        metric_value=0.9,
        lora_params=_sample_params(),
        training_params=_sample_params(),
        system_prompt="",
        hypothesis="Increasing rank to 32 should capture more features",
        iteration=4,
    )
    assert "Increasing rank to 32 should capture more features" in card
