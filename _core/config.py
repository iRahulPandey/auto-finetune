"""
config.py — Central configuration for auto-finetune.

All constants, search spaces, model definitions, and paths live here.
Nothing else in the codebase should hardcode these values.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch


# ── Device Detection ─────────────────────────────────────────────────────────

def get_device() -> str:
    """Detect best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype() -> torch.dtype:
    """bf16 on CUDA, fp16 on MPS, fp32 on CPU."""
    device = get_device()
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def get_dtype_str() -> str:
    """String version for TrainingArguments."""
    device = get_device()
    if device == "cuda":
        return "bf16"
    if device == "mps":
        return "fp16"
    return "fp32"


DEVICE = get_device()
DTYPE = get_dtype()
DTYPE_STR = get_dtype_str()


# ── Paths ────────────────────────────────────────────────────────────────────
# NOTE: config.py lives in _core/, so PROJECT_ROOT is one level up.

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ADAPTERS_DIR = PROJECT_ROOT / "adapters"
BEST_ADAPTER_DIR = ADAPTERS_DIR / "best"
LOGS_DIR = PROJECT_ROOT / "logs"
MLFLOW_TRACKING_URI = f"file://{PROJECT_ROOT / 'mlruns'}"
PROGRAM_MD_PATH = PROJECT_ROOT / "program.md"


# ── Supported Models ─────────────────────────────────────────────────────────

SUPPORTED_MODELS = {
    "qwen2.5-0.5b": {
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "short_name": "Qwen2.5-0.5B",
        "context_length": 4096,
        "dtype": "bfloat16",
    },
    "qwen2.5-1.5b": {
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "short_name": "Qwen2.5-1.5B",
        "context_length": 4096,
        "dtype": "bfloat16",
    },
    "phi-3-mini": {
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "short_name": "Phi-3-Mini-3.8B",
        "context_length": 4096,
        "dtype": "bfloat16",
    },
    "llama-3.2-1b": {
        "hf_id": "meta-llama/Llama-3.2-1B-Instruct",
        "short_name": "Llama-3.2-1B",
        "context_length": 4096,
        "dtype": "bfloat16",
    },
    "llama-3.2-3b": {
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "short_name": "Llama-3.2-3B",
        "context_length": 4096,
        "dtype": "bfloat16",
    },
}


# ── Task Types & Metrics ────────────────────────────────────────────────────

TASK_TYPES = {
    "classification": {
        "metrics": ["accuracy", "f1_macro", "f1_weighted"],
        "default_metric": "accuracy",
        "description": "Classify input into one of N categories",
    },
    "generation": {
        "metrics": ["rouge_l", "bleu", "exact_match"],
        "default_metric": "rouge_l",
        "description": "Generate free-form text output",
    },
    "extraction": {
        "metrics": ["json_field_accuracy", "f1_token", "exact_match", "rouge_l"],
        "default_metric": "json_field_accuracy",
        "description": "Extract structured info from unstructured text",
    },
}


# ── Layer Selection Rationale ────────────────────────────────────────────────
# Task-aware reasoning for which transformer projections to apply LoRA to.
# Used by program_md_generator and the agent prompt so the LLM can reason
# about WHY certain layers matter instead of searching them randomly.

LAYER_RATIONALE: dict[str, dict] = {
    "classification": {
        "recommended": ["q_proj", "v_proj"],
        "high_capacity": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "rationale": (
            "Classification needs compact attention routing. q+v captures query-value "
            "selection efficiently. o_proj helps if the model must suppress irrelevant "
            "tokens. k_proj rarely helps classification and adds parameter cost without "
            "benefit on small label sets."
        ),
        "escalate_when": "metric stagnates after 3+ iterations with q+v only",
    },
    "extraction": {
        "recommended": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "high_capacity": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "rationale": (
            "Extraction requires precise token matching. k_proj is critical because keys "
            "determine what the model attends to for field boundaries. o_proj controls how "
            "attended values project into output — essential for structured JSON fidelity. "
            "Use all four projections from the start."
        ),
        "escalate_when": "N/A — start with full set",
    },
    "generation": {
        "recommended": ["q_proj", "v_proj", "o_proj"],
        "high_capacity": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "rationale": (
            "Generation needs fluent output projection. o_proj is the most impactful "
            "layer for controlling output style and coherence. k_proj adds diversity "
            "in what the model keys on. Start with q+v+o; add k only if train_loss "
            "plateaus above 0.5 after 3+ epochs."
        ),
        "escalate_when": "train_loss > 0.5 after 3+ epochs",
    },
}


# ── Hyperparameter Search Space ──────────────────────────────────────────────

SEARCH_SPACE = {
    "learning_rate": [1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3],
    "lora_rank": [8, 16, 32, 64],
    "lora_alpha_multiplier": 2,  # alpha = rank * multiplier
    "num_train_epochs": [1, 2, 3, 4, 5],
    "lr_scheduler_type": ["cosine", "linear", "constant_with_warmup"],
    "target_modules": [
        ["q_proj", "v_proj"],
        ["q_proj", "v_proj", "k_proj"],
        ["q_proj", "v_proj", "o_proj"],
        ["q_proj", "v_proj", "k_proj", "o_proj"],
    ],
    "lora_dropout": [0.0, 0.05, 0.1],
    "warmup_ratio": [0.05, 0.1, 0.15, 0.2],
    # batch_size x gradient_acc = effective batch size
    # Keep batch <=4 for MPS compatibility; compensate with gradient_acc
    "per_device_train_batch_size": [1, 2, 4],
    "gradient_accumulation_steps": [2, 4, 8],
    "max_seq_length": 512,
}

# ── Hard Constraints (agent must not exceed) ─────────────────────────────────

CONSTRAINTS = {
    "max_lora_rank": 64,
    "min_learning_rate": 1e-5,
    "max_learning_rate": 1e-3,
    "min_epochs": 1,
    "max_epochs": 5,
    "eval_split_ratio": 0.2,  # 80/20 train/eval
}


# ── Agent Config ─────────────────────────────────────────────────────────────

AGENT_CONFIG = {
    "claude_model": "claude-sonnet-4-20250514",
    "max_tokens": 4096,
    "temperature": 0.3,  # Low temp: decisions should follow the data, not be creative
    "stagnation_threshold": 5,  # warn agent after 5 consecutive no-improvement runs
}


# ── Default Run Config ───────────────────────────────────────────────────────

@dataclass
class RunConfig:
    """Runtime configuration for a single fine-tuning session."""

    use_case: str = ""
    model_key: str = "qwen2.5-0.5b"
    task_type: str = "classification"
    max_iterations: int = 10
    target_threshold: Optional[float] = None
    metric_name: str = ""  # auto-set from task_type if empty

    # LoRA defaults (starting point for iteration 1)
    learning_rate: float = 3e-5
    lora_rank: int = 16
    num_train_epochs: int = 2
    lr_scheduler_type: str = "cosine"
    target_modules: list = field(default_factory=list)  # set by __post_init__ from task_type

    def __post_init__(self):
        if not self.metric_name:
            task_info = TASK_TYPES.get(self.task_type, TASK_TYPES["classification"])
            self.metric_name = task_info["default_metric"]
        if self.target_threshold is None:
            self.target_threshold = 0.95 if self.task_type == "classification" else 0.85
        if not self.target_modules:
            self.target_modules = LAYER_RATIONALE.get(
                self.task_type, LAYER_RATIONALE["classification"]
            )["recommended"]

    @property
    def model_info(self) -> dict:
        return SUPPORTED_MODELS[self.model_key]

    @property
    def hf_model_id(self) -> str:
        return self.model_info["hf_id"]

    @property
    def lora_alpha(self) -> int:
        return self.lora_rank * SEARCH_SPACE["lora_alpha_multiplier"]

    def validate(self) -> list[str]:
        """Return list of validation errors, empty if valid."""
        errors = []
        if self.model_key not in SUPPORTED_MODELS:
            errors.append(f"Unknown model: {self.model_key}")
        if self.task_type not in TASK_TYPES:
            errors.append(f"Unknown task type: {self.task_type}")
        if not (CONSTRAINTS["min_learning_rate"] <= self.learning_rate <= CONSTRAINTS["max_learning_rate"]):
            errors.append(f"learning_rate {self.learning_rate} out of range")
        if self.lora_rank > CONSTRAINTS["max_lora_rank"]:
            errors.append(f"lora_rank {self.lora_rank} exceeds max {CONSTRAINTS['max_lora_rank']}")
        if not (CONSTRAINTS["min_epochs"] <= self.num_train_epochs <= CONSTRAINTS["max_epochs"]):
            errors.append(f"num_train_epochs {self.num_train_epochs} out of range")
        if self.max_iterations < 1:
            errors.append("max_iterations must be >= 1")
        return errors
