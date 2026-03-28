"""
_core/hf_utils.py — HuggingFace Hub utilities for model card generation.
"""


def build_model_card(
    repo_id: str,
    base_model: str,
    use_case: str,
    metric_name: str,
    metric_value: float,
    lora_params: dict,
    training_params: dict,
    system_prompt: str,
    hypothesis: str,
    iteration: int,
) -> str:
    """Generate a HuggingFace model card (README.md) for a pushed LoRA adapter."""
    _target_mods = lora_params.get("target_modules", '["q_proj", "v_proj"]')
    _repo_short = repo_id.split("/")[-1]
    _dash = "\u2014"

    _lora_r = lora_params.get("lora_r", _dash)
    _lora_a = lora_params.get("lora_alpha", _dash)
    _lora_d = lora_params.get("lora_dropout", _dash)
    _lr = training_params.get("learning_rate", _dash)
    _ep = training_params.get("num_train_epochs", _dash)
    _sched = training_params.get("lr_scheduler_type", _dash)
    _bs = training_params.get("batch_size", _dash)
    _ga = training_params.get("gradient_accumulation_steps", _dash)
    _wr = training_params.get("warmup_ratio", _dash)

    if system_prompt:
        _sp_block = f'# System prompt used during training\nsystem_prompt = """{system_prompt}"""\n'
        _msg_line = '    {"role": "system", "content": system_prompt},\n'
    else:
        _sp_block = "# No system prompt recorded\nsystem_prompt = None\n"
        _msg_line = ""

    _hyp = hypothesis or "No hypothesis recorded."

    card = f"""---
library_name: peft
base_model: {base_model}
tags:
  - lora
  - auto-finetune
  - fine-tuned
license: mit
---

# {_repo_short}

A LoRA adapter fine-tuned with [auto-finetune](https://github.com/iRahulPandey/auto-finetune) \
{_dash} autonomous LoRA hyperparameter search powered by an LLM agent.

## Task

{use_case}

## Results

| Metric | Value |
|--------|-------|
| **{metric_name}** | **{metric_value:.4f}** |
| Iteration | {iteration} |

## LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | {_lora_r} |
| Alpha | {_lora_a} |
| Dropout | {_lora_d} |
| Target modules | {_target_mods} |

## Training Arguments

| Parameter | Value |
|-----------|-------|
| Learning rate | {_lr} |
| Epochs | {_ep} |
| Scheduler | {_sched} |
| Batch size | {_bs} |
| Gradient accumulation | {_ga} |
| Warmup ratio | {_wr} |

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "{base_model}"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(model, "{repo_id}")

{_sp_block}
# Run inference
messages = [
{_msg_line}    {{"role": "user", "content": "YOUR INPUT HERE"}},
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
outputs = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True))
```

## Agent Hypothesis

> {_hyp}

## Training Details

This adapter was trained autonomously by an LLM agent that iteratively searched over LoRA hyperparameters.
The agent reads a generated `program.md`, proposes configurations, trains, evaluates on a frozen eval set,
and iterates until the target metric is reached or max iterations are exhausted.

Built with [auto-finetune](https://github.com/iRahulPandey/auto-finetune).
"""
    return card
