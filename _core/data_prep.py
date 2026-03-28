"""
data_prep.py — Fixed data infrastructure. The agent NEVER touches this file.

Responsibilities:
  1. Accept raw input/output example pairs from the user
  2. Call Claude API to synthesize an optimal system prompt from the examples
  3. Validate data quality: class balance, label consistency, minimum counts
  4. Augment minority classes with Claude-generated synthetic examples
  5. Format examples into the model's chat template
  6. Stratified split into train/eval (80/20) preserving class balance
  7. Save to disk as JSONL
"""

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from . import llm_client
from .config import (
    CONSTRAINTS,
    DATA_DIR,
    SUPPORTED_MODELS,
    RunConfig,
)

# Minimum examples per class before augmentation kicks in.
# 20 ensures each class has enough diversity for LoRA fine-tuning.
# With <10 per class, the model memorises rather than generalises —
# this is the #1 cause of flat/oscillating accuracy curves.
MIN_EXAMPLES_PER_CLASS = 20


def _analyze_class_distribution(examples: list[dict]) -> dict:
    """Analyze label distribution and return diagnostic info."""
    labels = [ex["output"].strip() for ex in examples]
    dist = Counter(labels)
    total = len(labels)

    warnings = []
    if total < 20:
        warnings.append(f"Very few examples ({total}). 50+ recommended for reliable fine-tuning.")

    minority_classes = {
        label: count for label, count in dist.items() if count < MIN_EXAMPLES_PER_CLASS
    }
    if minority_classes:
        for label, count in minority_classes.items():
            warnings.append(
                f"Class '{label}' has only {count} example(s) "
                f"(minimum {MIN_EXAMPLES_PER_CLASS} recommended). "
                f"Will augment with synthetic examples."
            )

    return {
        "distribution": dict(dist),
        "total": total,
        "num_classes": len(dist),
        "minority_classes": minority_classes,
        "warnings": warnings,
    }


def _augment_minority_classes(
    examples: list[dict],
    use_case: str,
    min_per_class: int = MIN_EXAMPLES_PER_CLASS,
) -> list[dict]:
    """Generate synthetic examples for underrepresented classes using Claude.

    For each class with fewer than min_per_class examples, asks Claude to
    generate realistic additional examples that are diverse and different from
    the existing ones. This prevents model collapse to majority classes.
    """
    dist = Counter(ex["output"].strip() for ex in examples)
    classes_to_augment = {label: count for label, count in dist.items() if count < min_per_class}

    if not classes_to_augment:
        return examples

    provider_label = llm_client.get_stage_config(llm_client.STAGE_DATA_PREP).label
    print(f"  Using {provider_label} for augmentation")

    augmented = list(examples)

    for label, count in classes_to_augment.items():
        need = min_per_class - count
        class_examples = [ex for ex in examples if ex["output"].strip() == label]

        # Show all existing examples for this class
        example_text = "\n".join(
            f"  Input: {ex['input'][:300]}\n  Output: {ex['output']}" for ex in class_examples
        )

        # Also show a few examples from OTHER classes for contrast
        other_examples = [ex for ex in examples if ex["output"].strip() != label][:3]
        contrast_text = "\n".join(
            f"  Input: {ex['input'][:300]}\n  Output: {ex['output']}" for ex in other_examples
        )

        prompt = f"""You are helping create training data for a fine-tuned classifier.

Task: {use_case}

The class "{label}" only has {count} example(s) — we need {need} more.

Existing examples for "{label}":
{example_text}

Examples from OTHER classes (for contrast — do NOT generate these):
{contrast_text}

Generate exactly {need} NEW, diverse examples for the class "{label}".

CRITICAL REQUIREMENTS for diversity:
- Each example MUST describe a DIFFERENT scenario/industry/context
- Vary the length: some short (1 sentence), some medium (2-3 sentences)
- Vary the phrasing style: questions, descriptions, requests, proposals
- Cover different industries: healthcare, finance, education, retail, government, tech, etc.
- Cover different risk patterns, not just the same pattern repeated
- The output MUST be exactly "{label}" (no variations)

Return ONLY a JSON array:
[{{"input": "...", "output": "{label}"}}, ...]"""

        try:
            raw = llm_client.generate(
                prompt=prompt,
                stage=llm_client.STAGE_DATA_PREP,
                model_hint="fast",
                max_tokens=4096,
                temperature=0.9,
            )

            # Extract JSON array
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                new_examples = json.loads(match.group(0))
                # Validate: every output must be exactly the target label
                valid = [
                    ex
                    for ex in new_examples
                    if isinstance(ex, dict)
                    and "input" in ex
                    and ex.get("output", "").strip() == label
                ]
                augmented.extend(valid[:need])
                print(
                    f"  Augmented class '{label}': {count} → {count + len(valid[:need])} examples"
                )
            else:
                print(f"  Warning: Could not parse augmentation response for '{label}'")

        except Exception as e:
            print(f"  Warning: Augmentation failed for '{label}': {e}")

    return augmented


def _hash_examples(examples: list[dict]) -> str:
    """Deterministic hash for reproducible splits."""
    raw = json.dumps(examples, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def synthesize_system_prompt(
    use_case: str,
    examples: list[dict],
) -> str:
    """
    Stage 1: Use LLM to generate an optimal system prompt
    from the user's use case description and example pairs.
    Uses whatever provider is configured in llm_client.
    """
    example_text = "\n".join(
        f"Input: {ex['input']}\nOutput: {ex['output']}"
        for ex in examples[:10]  # send at most 10 examples to keep prompt short
    )

    prompt = f"""You are an expert prompt engineer. A user wants to fine-tune a small LLM for this task:

Task: {use_case}

Here are some example input/output pairs:

{example_text}

Write an optimal system prompt for this task. The prompt should:
1. Be clear and concise (under 200 words)
2. Specify the exact output format expected
3. Include any constraints (e.g. "respond with only the label, no explanation")
4. Be written as a direct instruction to the model

Return ONLY the system prompt text, nothing else."""

    provider_label = llm_client.get_stage_config(llm_client.STAGE_DATA_PREP).label
    print(f"  Synthesizing system prompt with {provider_label}...")

    return llm_client.generate(
        prompt=prompt,
        stage=llm_client.STAGE_DATA_PREP,
        model_hint="smart",
        max_tokens=1024,
        temperature=0.3,
    )


def format_chat_examples(
    examples: list[dict],
    system_prompt: str,
    model_key: str,
) -> list[dict]:
    """
    Stage 2: Format examples into chat-template-compatible dicts.
    Each example becomes:
      {
        "messages": [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": input},
          {"role": "assistant", "content": output}
        ]
      }
    """
    formatted = []
    for ex in examples:
        formatted.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": ex["input"]},
                    {"role": "assistant", "content": ex["output"]},
                ]
            }
        )
    return formatted


def split_train_eval(
    formatted_examples: list[dict],
    eval_ratio: float = CONSTRAINTS["eval_split_ratio"],
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Stratified split. Every class is represented in both train and eval
    so the model is tested on all labels, not just the majority class.
    Eval set is FROZEN — never changes across iterations.

    For small datasets (<80 total examples), we use a higher eval ratio (30%)
    and guarantee at least 2 examples per class in eval — otherwise the metric
    is statistically meaningless.
    """
    rng = random.Random(seed)

    # Group by assistant (output) label
    by_class: dict[str, list[dict]] = defaultdict(list)
    for ex in formatted_examples:
        label = next(
            (m["content"].strip() for m in ex["messages"] if m["role"] == "assistant"),
            "__unknown__",
        )
        by_class[label].append(ex)

    total = len(formatted_examples)

    # Detect generation/extraction tasks: if most outputs are unique,
    # stratified split makes no sense — use a simple random split instead.
    if len(by_class) > total * 0.5:
        # More than half the outputs are unique → random split
        if total < 80:
            eval_ratio = max(eval_ratio, 0.30)
        all_examples = list(formatted_examples)
        rng.shuffle(all_examples)
        n_eval = max(2, int(total * eval_ratio))
        eval_set = all_examples[:n_eval]
        train_set = all_examples[n_eval:]
        print("  Unique outputs detected — using random split instead of stratified")
    else:
        # Classification: stratified split preserving class balance
        if total < 80:
            eval_ratio = max(eval_ratio, 0.30)

        train_set = []
        eval_set = []

        for label, items in by_class.items():
            rng.shuffle(items)
            # At least 2 eval examples per class (if the class has >=3 items)
            min_eval = 2 if len(items) >= 3 else 1
            n_eval = max(min_eval, int(len(items) * eval_ratio))
            # Never take all items for eval — leave at least 1 for training
            n_eval = min(n_eval, len(items) - 1)
            eval_set.extend(items[:n_eval])
            train_set.extend(items[n_eval:])

    # Shuffle both sets so training isn't grouped by class
    rng.shuffle(train_set)
    rng.shuffle(eval_set)

    # Log the split
    train_labels = Counter(
        next(m["content"].strip() for m in ex["messages"] if m["role"] == "assistant")
        for ex in train_set
    )
    eval_labels = Counter(
        next(m["content"].strip() for m in ex["messages"] if m["role"] == "assistant")
        for ex in eval_set
    )
    print(f"  Stratified split: {len(train_set)} train / {len(eval_set)} eval")
    print(f"  Train classes: {dict(train_labels)}")
    print(f"  Eval classes:  {dict(eval_labels)}")

    return train_set, eval_set


def save_datasets(
    train_set: list[dict],
    eval_set: list[dict],
    session_id: str,
) -> tuple[Path, Path]:
    """Save train and eval sets as JSONL files."""
    session_dir = DATA_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    train_path = session_dir / "train.jsonl"
    eval_path = session_dir / "eval.jsonl"

    for path, dataset in [(train_path, train_set), (eval_path, eval_set)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return train_path, eval_path


def prepare_data(
    use_case: str,
    examples: list[dict],
    model_key: str,
    session_id: Optional[str] = None,
    skip_prompt_synthesis: bool = False,
    system_prompt: Optional[str] = None,
    enable_augmentation: bool = False,
) -> dict:
    """
    Full data preparation pipeline.

    Args:
        use_case: Description of the fine-tuning task
        examples: List of {"input": str, "output": str} dicts
        model_key: Key from SUPPORTED_MODELS
        session_id: Optional session identifier (auto-generated if None)
        skip_prompt_synthesis: If True, skip Claude API call (for testing)
        system_prompt: Pre-defined system prompt (skips synthesis if provided)
        enable_augmentation: If True, augment minority classes via Claude API.
            Off by default — example datasets are pre-balanced.

    Returns:
        dict with keys: system_prompt, train_path, eval_path,
                        train_count, eval_count, session_id
    """
    if session_id is None:
        session_id = _hash_examples(examples)

    # ── Data quality analysis ─────────────────────────────────────────────
    print("\n=== Data Quality Analysis ===")
    analysis = _analyze_class_distribution(examples)
    print(f"  Total examples: {analysis['total']}")
    print(f"  Classes ({analysis['num_classes']}):")
    for label, count in sorted(analysis["distribution"].items(), key=lambda x: -x[1]):
        pct = count / analysis["total"] * 100
        bar = "#" * int(pct / 2)
        print(f"    {count:>3} ({pct:4.0f}%) {bar} {label}")

    for warning in analysis["warnings"]:
        print(f"  WARNING: {warning}")

    # ── Augment minority classes ──────────────────────────────────────────
    if analysis["minority_classes"] and enable_augmentation:
        print(f"\n=== Augmenting {len(analysis['minority_classes'])} minority class(es) ===")
        examples = _augment_minority_classes(examples, use_case)
        # Re-analyze after augmentation
        post_analysis = _analyze_class_distribution(examples)
        print(f"  After augmentation: {post_analysis['total']} total examples")
        for label, count in sorted(post_analysis["distribution"].items(), key=lambda x: -x[1]):
            print(f"    {count:>3}  {label}")
    elif analysis["minority_classes"]:
        print("\n  Augmentation disabled — skipping. Enable in sidebar if classes are imbalanced.")

    # Stage 1: Synthesize system prompt
    if system_prompt is None:
        if skip_prompt_synthesis:
            system_prompt = f"You are a helpful assistant. Task: {use_case}"
        else:
            system_prompt = synthesize_system_prompt(use_case, examples)

    # Stage 2: Format into chat template
    formatted = format_chat_examples(examples, system_prompt, model_key)

    # Stratified split (preserves class balance in both train and eval)
    train_set, eval_set = split_train_eval(formatted)

    # Save
    train_path, eval_path = save_datasets(train_set, eval_set, session_id)

    # Also save the system prompt
    prompt_path = DATA_DIR / session_id / "system_prompt.txt"
    prompt_path.write_text(system_prompt, encoding="utf-8")

    # Save raw examples for reference
    raw_path = DATA_DIR / session_id / "raw_examples.json"
    raw_path.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "system_prompt": system_prompt,
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "train_count": len(train_set),
        "eval_count": len(eval_set),
        "session_id": session_id,
        "data_analysis": analysis,
    }
