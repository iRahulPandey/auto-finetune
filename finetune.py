"""
finetune.py — The experiment surface. This is the ONLY file the agent edits.

The agent modifies the LoRA config section (LORA_CONFIG and TRAINING_ARGS)
between iterations. Everything else is fixed infrastructure.

Usage:
    python finetune.py --config config.json

The config.json is written by agent_loop.py before each run.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer

from _core.config import DEVICE, DTYPE, DTYPE_STR
from _core.evaluator import evaluate_in_process

# ═══════════════════════════════════════════════════════════════════════════
# LoRA CONFIG — Agent edits this section between iterations
# ═══════════════════════════════════════════════════════════════════════════

HYPOTHESIS = "Starting with the smallest learning rate (1e-05) and rank 16 with q_proj+v_proj should provide stable learning for email urgency classification without overfitting."

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "v_proj"],
    "task_type": "CAUSAL_LM",
    "bias": "none",
}

TRAINING_ARGS = {
    "learning_rate": 1e-05,
    "num_train_epochs": 3,
    "lr_scheduler_type": "cosine",
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.1,
    "logging_steps": 10,
    "save_strategy": "no",
    "optim": "adamw_torch",
    "remove_unused_columns": False,
    "report_to": "none",
}

MAX_SEQ_LENGTH = 512

# ═══════════════════════════════════════════════════════════════════════════
# FIXED INFRASTRUCTURE — Agent does NOT edit below this line
# ═══════════════════════════════════════════════════════════════════════════


def _apply_dtype_flags(args: dict, num_examples: int = 0) -> dict:
    """Inject the correct precision flag and resolve warmup_ratio → warmup_steps."""
    args = args.copy()

    # Remove any precision flags the agent may have hardcoded
    args.pop("bf16", None)
    args.pop("fp16", None)

    if DTYPE_STR == "bf16":
        args["bf16"] = True
    elif DTYPE_STR == "fp16":
        args["fp16"] = True
    # fp32: no flag needed

    # MPS: no fused optimizers, reduce batch size for 8GB RAM
    if DEVICE == "mps":
        args["optim"] = "adamw_torch"
        args.setdefault("per_device_train_batch_size", 2)
        args.setdefault("gradient_accumulation_steps", 8)

    # Convert warmup_ratio → warmup_steps (warmup_ratio deprecated in TRL v5.2)
    warmup_ratio = args.pop("warmup_ratio", None)
    if warmup_ratio and num_examples > 0:
        batch_size = args.get("per_device_train_batch_size", 4)
        grad_accum = args.get("gradient_accumulation_steps", 4)
        num_epochs = args.get("num_train_epochs", 2)
        total_steps = max(1, (num_examples // (batch_size * grad_accum)) * num_epochs)
        args["warmup_steps"] = max(1, int(total_steps * warmup_ratio))
    elif warmup_ratio:
        args["warmup_steps"] = 10  # safe fallback

    return args


def _load_model(base_model_id: str):
    """Load model with correct device placement for CUDA/MPS/CPU."""
    if DEVICE == "cuda":
        return AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=DTYPE,
            device_map="auto",
            trust_remote_code=True,
        )
    elif DEVICE == "mps":
        # device_map="auto" doesn't support MPS — load to CPU then move
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=DTYPE,
            trust_remote_code=True,
        )
        return model.to("mps")
    else:
        return AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=DTYPE,
            trust_remote_code=True,
        )


def load_run_config(config_path: str) -> dict:
    """Load the runtime config written by agent_loop.py."""
    with open(config_path, "r") as f:
        return json.load(f)


def format_chat(example, tokenizer):
    """Apply the model's chat template to a single example."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def run_finetune(
    base_model_id: str,
    train_path: str,
    output_dir: str,
    max_seq_length: int = MAX_SEQ_LENGTH,
    eval_path: str | None = None,
    metric_name: str | None = None,
    **kwargs,
) -> dict:
    """
    Execute a single fine-tuning run with the current LORA_CONFIG and TRAINING_ARGS.
    If eval_path and metric_name are provided, evaluates immediately after training
    using the model already in memory (avoids a second model load).

    Returns:
        dict with: adapter_path, train_loss, train_samples,
                   and optionally: metric_name, metric_value, mismatches
    """
    print(f"Device: {DEVICE} | Dtype: {DTYPE} | Precision flag: {DTYPE_STR}")

    # Flush any stale memory at subprocess start
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Set max length here — SFTConfig dropped max_seq_length in TRL 0.12+
    tokenizer.model_max_length = max_seq_length

    # Load dataset
    dataset = load_dataset("json", data_files=train_path, split="train")

    # Format with chat template
    dataset = dataset.map(
        lambda ex: format_chat(ex, tokenizer),
        remove_columns=dataset.column_names,
    )

    # Load base model (device-aware)
    model = _load_model(base_model_id)

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias=LORA_CONFIG.get("bias", "none"),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments (inject correct precision, resolve warmup, device flags)
    resolved_args = _apply_dtype_flags(TRAINING_ARGS, num_examples=len(dataset))

    if DEVICE == "mps":
        resolved_args["dataloader_pin_memory"] = False

    training_args = SFTConfig(
        output_dir=str(output_path),
        dataset_text_field="text",
        **resolved_args,
    )

    # processing_class replaces the deprecated tokenizer= kwarg
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    train_result = trainer.train()

    # Save adapter (suppress benign PEFT vocab-check warning for Qwen models)
    import warnings

    adapter_path = output_path / "adapter"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not find a config file")
        model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    result = {
        "adapter_path": str(adapter_path),
        "train_loss": round(train_result.training_loss, 6),
        "train_samples": len(dataset),
        "hypothesis": HYPOTHESIS,
        "lora_config": LORA_CONFIG.copy(),
        "training_args": {k: v for k, v in TRAINING_ARGS.items()},
    }

    # ── In-process eval (reuses model already in memory — no second load) ────
    if eval_path and metric_name:
        # Task-aware generation length: classification labels are short,
        # extraction/generation outputs can be 200+ tokens (JSON, commit messages)
        _task_type = kwargs.get("task_type", "classification")
        _max_new_tokens = 30 if _task_type == "classification" else 256
        print(
            f"\nEvaluating ({metric_name}) with model already in memory... "
            f"(task={_task_type}, max_new_tokens={_max_new_tokens})"
        )
        eval_result = evaluate_in_process(
            model=model,
            tokenizer=tokenizer,
            eval_path=eval_path,
            metric_name=metric_name,
            max_new_tokens=_max_new_tokens,
        )
        result["metric_name"] = eval_result["metric_name"]
        result["metric_value"] = eval_result["metric_value"]
        result["num_eval_examples"] = eval_result["num_examples"]
        result["mismatches"] = eval_result.get("mismatches", [])
        result["per_class_accuracy"] = eval_result.get("per_class_accuracy", {})
        print(f"Eval {metric_name}: {eval_result['metric_value']:.4f}")
        if eval_result.get("mismatches"):
            print("Sample mismatches (first 3):")
            for mm in eval_result["mismatches"][:3]:
                print(f"  expected: {mm['expected'][:80]}")
                print(f"  got:      {mm['predicted'][:80]}")
                print()

    # Cleanup
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="auto-finetune training run")
    parser.add_argument("--config", type=str, required=True, help="Path to run config JSON")
    parser.add_argument(
        "--output-dir", type=str, default="./adapters/current", help="Output directory"
    )
    parser.add_argument(
        "--eval-path", type=str, default=None, help="Path to eval.jsonl for in-process eval"
    )
    parser.add_argument(
        "--metric-name", type=str, default=None, help="Metric to compute during eval"
    )
    args = parser.parse_args()

    run_config = load_run_config(args.config)

    result = run_finetune(
        base_model_id=run_config["hf_model_id"],
        train_path=run_config["train_path"],
        output_dir=args.output_dir,
        eval_path=args.eval_path,
        metric_name=args.metric_name,
        task_type=run_config.get("task_type", "classification"),
    )

    # Write result for agent_loop.py to read
    result_path = Path(args.output_dir) / "train_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nTraining complete. Loss: {result['train_loss']}")
    print(f"Adapter saved to: {result['adapter_path']}")
    print(f"Hypothesis: {result['hypothesis']}")

    return result


if __name__ == "__main__":
    main()
