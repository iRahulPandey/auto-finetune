"""
evaluator.py — Evaluation on the frozen eval set. Agent NEVER modifies this file.

Supports multiple metrics depending on task type:
  - classification: accuracy, f1_macro, f1_weighted
  - generation: rouge_l, bleu, exact_match
  - extraction: exact_match, f1_token, rouge_l

Accuracy uses Claude-as-judge for semantic matching when strict exact match
is insufficient (e.g. model outputs "This is urgent" vs reference "urgent").
"""

import json
import os
import re
import string
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from . import llm_client


def _load_eval_set(eval_path: str) -> list[dict]:
    """Load the frozen eval JSONL. Each line has a 'messages' key."""
    examples = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _extract_prompt_and_reference(example: dict) -> tuple[list[dict], str]:
    """
    From a chat-formatted example, extract the prompt messages
    (system + user) and the reference output (assistant).
    """
    messages = example["messages"]
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    reference = next(m["content"] for m in messages if m["role"] == "assistant")
    return prompt_messages, reference


def _generate_prediction(
    model,
    tokenizer,
    prompt_messages: list[dict],
    max_new_tokens: int = 30,
) -> str:
    """Generate a single prediction from prompt messages.

    max_new_tokens defaults to 30 (not 256) — classification labels are
    typically 1-10 tokens. Generating too many tokens lets the model ramble
    past the label, harming accuracy.
    """
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    import warnings
    with torch.no_grad(), warnings.catch_warnings():
        # Suppress "generation flags are not valid" for top_p/top_k in greedy mode
        warnings.filterwarnings("ignore", message="The following generation flags")
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for eval determinism
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Post-process: extract just the first line (label) — the model sometimes
    # generates explanation text after the label which tanks exact-match accuracy.
    prediction = prediction.split("\n")[0].strip()

    return prediction


# ── Metric Implementations ───────────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    """Lowercase, strip whitespace, remove punctuation."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _json_match(prediction: str, reference: str) -> bool:
    """Compare two strings as JSON objects. Returns True if both parse
    to equivalent dicts/lists, ignoring key order and whitespace."""
    try:
        pred_json = json.loads(prediction.strip())
        ref_json = json.loads(reference.strip())
        return pred_json == ref_json
    except (json.JSONDecodeError, ValueError):
        return False


def _labels_match(prediction: str, reference: str) -> bool:
    """Check if prediction matches reference, with smart matching.

    Tries multiple strategies in order:
      1. JSON comparison (for extraction tasks with JSON output)
      2. Exact text match after normalization
      3. Label code matching (e.g. "LLM06" prefix)
    """
    # Try JSON comparison first — handles extraction tasks where
    # prediction and reference are both JSON but differ in whitespace/key order
    if (reference.strip().startswith("{") or reference.strip().startswith("[")):
        if _json_match(prediction, reference):
            return True

    pred_norm = _normalize_text(prediction)
    ref_norm = _normalize_text(reference)

    # Exact match after normalization
    if pred_norm == ref_norm:
        return True

    # Extract label codes (e.g. "LLM06" from "LLM06:sensitive-information-disclosure")
    pred_code = re.match(r'(llm\d{2})', pred_norm)
    ref_code = re.match(r'(llm\d{2})', ref_norm)

    if pred_code and ref_code:
        return pred_code.group(1) == ref_code.group(1)

    return False


def _judge_batch(
    predictions: list[str],
    references: list[str],
    batch_size: int = 20,
) -> list[bool]:
    """Use LLM-as-judge to evaluate whether predictions match references.

    Sends predictions in batches to keep latency manageable. Each batch asks
    the LLM to return a JSON array of true/false verdicts.

    Falls back to pattern matching if the API call fails.
    Uses whatever provider is configured for the evaluator stage.
    """
    stage_cfg = llm_client.get_stage_config(llm_client.STAGE_EVALUATOR)
    print(f"  Judge provider: {stage_cfg.label}")

    all_verdicts: list[bool] = []

    for start in range(0, len(predictions), batch_size):
        batch_preds = predictions[start:start + batch_size]
        batch_refs = references[start:start + batch_size]

        pairs = []
        for i, (pred, ref) in enumerate(zip(batch_preds, batch_refs)):
            pairs.append(f"  {i}: prediction={json.dumps(pred[:200])} | expected={json.dumps(ref[:200])}")
        pairs_str = "\n".join(pairs)

        prompt = f"""You are evaluating a fine-tuned model's outputs. For each pair below, decide if the prediction is semantically correct — i.e. it conveys the same answer/label/meaning as the expected output.

Be lenient about formatting differences: "urgent" and "This is classified as urgent" are both correct if the expected answer is "urgent". But "not urgent" when expected is "urgent" is wrong.

Pairs:
{pairs_str}

Return ONLY a JSON array of booleans, one per pair. Example for 3 pairs: [true, false, true]"""

        try:
            raw = llm_client.generate(
                prompt=prompt,
                stage=llm_client.STAGE_EVALUATOR,
                model_hint="fast",
                max_tokens=1024,
                temperature=0.0,
            )

            # Extract JSON array from response
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                verdicts = json.loads(match.group(0))
                if len(verdicts) == len(batch_preds):
                    all_verdicts.extend(bool(v) for v in verdicts)
                    continue

            # If parsing failed, fall back for this batch
            print(f"  Judge parse error for batch starting at {start}, using exact match fallback")
            all_verdicts.extend(
                _normalize_text(p) == _normalize_text(r)
                for p, r in zip(batch_preds, batch_refs)
            )

        except Exception as e:
            print(f"  Judge API error: {e} — using exact match fallback for batch {start}")
            all_verdicts.extend(
                _normalize_text(p) == _normalize_text(r)
                for p, r in zip(batch_preds, batch_refs)
            )

    return all_verdicts


def compute_accuracy(predictions: list[str], references: list[str]) -> float:
    """Classification accuracy using Claude-as-judge for semantic matching.

    Strategy:
      1. Try strict exact match first (fast, free).
      2. If strict accuracy >= 80%, trust it — the model is outputting clean labels.
      3. Otherwise, use Claude (haiku) as a judge to evaluate semantic equivalence.
      4. Fall back to exact match if no API key is available.
    """
    if not predictions:
        return 0.0

    # Fast path: strict label match (handles partial codes like "LLM06" vs full labels)
    strict_correct = sum(
        1 for p, r in zip(predictions, references)
        if _labels_match(p, r)
    )
    strict_acc = strict_correct / len(predictions)

    # If strict match is already high, no need for the judge
    if strict_acc >= 0.8:
        return strict_acc

    # Use LLM-as-judge for semantic evaluation
    eval_cfg = llm_client.get_stage_config(llm_client.STAGE_EVALUATOR)
    if eval_cfg.is_claude() and not os.environ.get("ANTHROPIC_API_KEY"):
        print("  No ANTHROPIC_API_KEY — using exact match for accuracy")
        return strict_acc
    if eval_cfg.is_ollama() and not eval_cfg.ollama_model:
        print("  No Ollama model configured for evaluator — using exact match")
        return strict_acc

    try:
        print(f"  Strict accuracy={strict_acc:.2f} < 0.80 — using LLM-as-judge ({eval_cfg.label})")
        verdicts = _judge_batch(predictions, references)
        judge_acc = sum(verdicts) / len(verdicts)
        print(f"  Judge accuracy: {judge_acc:.4f} (strict was {strict_acc:.4f})")
        return judge_acc
    except Exception as e:
        print(f"  Judge failed ({e}) — falling back to strict accuracy")
        return strict_acc


def compute_f1_macro(predictions: list[str], references: list[str]) -> float:
    """Macro F1 across all unique labels."""
    labels = set(_normalize_text(r) for r in references)
    if not labels:
        return 0.0

    f1_scores = []
    for label in labels:
        tp = sum(
            1 for p, r in zip(predictions, references)
            if _normalize_text(p) == label and _normalize_text(r) == label
        )
        fp = sum(
            1 for p, r in zip(predictions, references)
            if _normalize_text(p) == label and _normalize_text(r) != label
        )
        fn = sum(
            1 for p, r in zip(predictions, references)
            if _normalize_text(p) != label and _normalize_text(r) == label
        )
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def compute_f1_weighted(predictions: list[str], references: list[str]) -> float:
    """Weighted F1 by label frequency."""
    labels = set(_normalize_text(r) for r in references)
    if not labels:
        return 0.0

    total = len(references)
    weighted_f1 = 0.0

    for label in labels:
        support = sum(1 for r in references if _normalize_text(r) == label)
        tp = sum(
            1 for p, r in zip(predictions, references)
            if _normalize_text(p) == label and _normalize_text(r) == label
        )
        fp = sum(
            1 for p, r in zip(predictions, references)
            if _normalize_text(p) == label and _normalize_text(r) != label
        )
        fn = sum(
            1 for p, r in zip(predictions, references)
            if _normalize_text(p) != label and _normalize_text(r) == label
        )
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        weighted_f1 += f1 * (support / total)

    return weighted_f1


def compute_exact_match(predictions: list[str], references: list[str]) -> float:
    """Alias for accuracy — exact match after normalization."""
    return compute_accuracy(predictions, references)


def _get_tokens(text: str) -> list[str]:
    """Tokenize by whitespace for token-level F1."""
    return _normalize_text(text).split()


def compute_f1_token(predictions: list[str], references: list[str]) -> float:
    """Token-level F1 (useful for extraction tasks)."""
    if not predictions:
        return 0.0

    f1_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = _get_tokens(pred)
        ref_tokens = _get_tokens(ref)

        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            f1_scores.append(0.0)
            continue

        precision = num_common / len(pred_tokens) if pred_tokens else 0.0
        recall = num_common / len(ref_tokens) if ref_tokens else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)


def _lcs_length(x: list[str], y: list[str]) -> int:
    """Longest common subsequence length."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge_l(predictions: list[str], references: list[str]) -> float:
    """ROUGE-L F1 score."""
    if not predictions:
        return 0.0

    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = _get_tokens(pred)
        ref_tokens = _get_tokens(ref)

        if not ref_tokens:
            scores.append(0.0)
            continue

        lcs = _lcs_length(pred_tokens, ref_tokens)
        precision = lcs / len(pred_tokens) if pred_tokens else 0.0
        recall = lcs / len(ref_tokens) if ref_tokens else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        scores.append(f1)

    return sum(scores) / len(scores)


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """Simplified BLEU-4 score."""
    import math

    if not predictions:
        return 0.0

    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = _get_tokens(pred)
        ref_tokens = _get_tokens(ref)

        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue

        # n-gram precisions for n=1..4
        precisions = []
        for n in range(1, 5):
            pred_ngrams = Counter(
                tuple(pred_tokens[i:i + n]) for i in range(len(pred_tokens) - n + 1)
            )
            ref_ngrams = Counter(
                tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1)
            )
            clipped = sum(min(count, ref_ngrams[ng]) for ng, count in pred_ngrams.items())
            total = sum(pred_ngrams.values())
            precisions.append(clipped / total if total > 0 else 0.0)

        if any(p == 0 for p in precisions):
            scores.append(0.0)
            continue

        log_avg = sum(math.log(p) for p in precisions) / 4
        brevity = min(1.0, len(pred_tokens) / len(ref_tokens))
        bp = math.exp(1 - 1 / brevity) if brevity < 1 else 1.0
        scores.append(bp * math.exp(log_avg))

    return sum(scores) / len(scores)


def compute_json_field_accuracy(predictions: list[str], references: list[str]) -> float:
    """Field-level accuracy for JSON extraction outputs.

    For each prediction/reference pair:
      1. Parse both as JSON dicts.
      2. Compare each expected key's value — count matching fields.
      3. Score = total matching fields / total expected fields across all examples.

    Falls back to LLM-as-judge for pairs where JSON parsing fails
    (model may output valid-but-different formatting).
    """
    if not predictions:
        return 0.0

    total_fields = 0
    matching_fields = 0
    unparseable_preds = []
    unparseable_refs = []
    unparseable_indices = []

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        try:
            ref_obj = json.loads(ref.strip())
        except (json.JSONDecodeError, ValueError):
            # Reference isn't JSON — fall back to token F1 for this pair
            unparseable_refs.append(ref)
            unparseable_preds.append(pred)
            unparseable_indices.append(i)
            continue

        try:
            pred_obj = json.loads(pred.strip())
        except (json.JSONDecodeError, ValueError):
            # Prediction isn't valid JSON (truncated or garbled)
            # Count all reference fields as misses
            if isinstance(ref_obj, dict):
                total_fields += len(ref_obj)
            else:
                total_fields += 1
            continue

        # Both parsed — compare field by field
        if isinstance(ref_obj, dict) and isinstance(pred_obj, dict):
            for key, ref_val in ref_obj.items():
                total_fields += 1
                pred_val = pred_obj.get(key)
                if pred_val == ref_val:
                    matching_fields += 1
                elif isinstance(ref_val, str) and isinstance(pred_val, str):
                    # Lenient string comparison: case-insensitive, strip whitespace
                    if ref_val.strip().lower() == pred_val.strip().lower():
                        matching_fields += 1
        elif ref_obj == pred_obj:
            # Non-dict JSON (list, scalar) — exact match
            total_fields += 1
            matching_fields += 1
        else:
            total_fields += 1

    # For unparseable pairs, use token F1 as fallback
    if unparseable_preds:
        for p, r in zip(unparseable_preds, unparseable_refs):
            total_fields += 1
            pred_tokens = set(_normalize_text(p).split())
            ref_tokens = set(_normalize_text(r).split())
            if ref_tokens and pred_tokens:
                overlap = pred_tokens & ref_tokens
                precision = len(overlap) / len(pred_tokens)
                recall = len(overlap) / len(ref_tokens)
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    matching_fields += f1  # weighted contribution

    accuracy = matching_fields / total_fields if total_fields > 0 else 0.0

    # Log diagnostic
    print(f"  JSON field accuracy: {matching_fields:.0f}/{total_fields} fields "
          f"({accuracy:.4f})")

    return accuracy


# ── Metric Registry ──────────────────────────────────────────────────────────

METRIC_FUNCTIONS = {
    "accuracy": compute_accuracy,
    "f1_macro": compute_f1_macro,
    "f1_weighted": compute_f1_weighted,
    "exact_match": compute_exact_match,
    "json_field_accuracy": compute_json_field_accuracy,
    "f1_token": compute_f1_token,
    "rouge_l": compute_rouge_l,
    "bleu": compute_bleu,
}


# ── Main Evaluation Entry Point ──────────────────────────────────────────────

def _load_model_for_eval(base_model_id: str, adapter_path: str | None):
    """Load model with correct device placement for CUDA/MPS/CPU."""
    from .config import DEVICE, DTYPE

    if DEVICE == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=DTYPE,
            device_map="auto",
            trust_remote_code=True,
        )
    elif DEVICE == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=DTYPE,
            trust_remote_code=True,
        )
        model = model.to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=DTYPE,
            trust_remote_code=True,
        )

    # Apply LoRA adapter if provided
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        if DEVICE == "mps":
            model = model.to("mps")

    return model


def evaluate_in_process(
    model,
    tokenizer,
    eval_path: str,
    metric_name: str,
    max_new_tokens: int = 30,
) -> dict:
    """Evaluate using a model that is ALREADY loaded in memory.

    This is the fast path — used when finetune.py calls eval right after
    training, avoiding a second model load. The model can be a PeftModel
    or a merged model; both work for generation.

    Returns:
        dict with: metric_name, metric_value, predictions, references, num_examples,
                   mismatches (list of dicts showing where prediction != reference)
    """
    if metric_name not in METRIC_FUNCTIONS:
        raise ValueError(f"Unknown metric: {metric_name}. Available: {list(METRIC_FUNCTIONS.keys())}")

    eval_examples = _load_eval_set(eval_path)
    if not eval_examples:
        raise ValueError(f"Empty eval set at {eval_path}")

    model.eval()
    predictions = []
    references = []

    for example in eval_examples:
        prompt_messages, reference = _extract_prompt_and_reference(example)
        prediction = _generate_prediction(model, tokenizer, prompt_messages, max_new_tokens)
        predictions.append(prediction)
        references.append(reference)

    metric_fn = METRIC_FUNCTIONS[metric_name]
    metric_value = metric_fn(predictions, references)

    # Log mismatches so agent can see WHY accuracy is low
    mismatches = []
    for i, (p, r) in enumerate(zip(predictions, references)):
        if not _labels_match(p, r):
            # For long outputs show where they diverge, not just truncated starts
            p_show = p[:200] if len(p) <= 200 else p[:100] + " ... " + p[-50:]
            r_show = r[:200] if len(r) <= 200 else r[:100] + " ... " + r[-50:]
            mismatches.append({"idx": i, "predicted": p_show, "expected": r_show})

    # Per-class accuracy breakdown (critical for diagnosing which classes fail)
    from collections import Counter
    class_correct: dict[str, int] = Counter()
    class_total: dict[str, int] = Counter()
    for p, r in zip(predictions, references):
        r_key = r.strip()
        class_total[r_key] += 1
        if _labels_match(p, r):
            class_correct[r_key] += 1

    per_class = {}
    for label in sorted(class_total.keys()):
        correct = class_correct.get(label, 0)
        total = class_total[label]
        per_class[label] = f"{correct}/{total}"
    print(f"  Per-class accuracy: {per_class}")

    return {
        "metric_name": metric_name,
        "metric_value": round(metric_value, 6),
        "predictions": predictions,
        "references": references,
        "num_examples": len(eval_examples),
        "mismatches": mismatches[:10],  # first 10 for debugging
        "per_class_accuracy": per_class,
    }


def evaluate(
    base_model_id: str,
    adapter_path: str | None,
    eval_path: str,
    metric_name: str,
    max_new_tokens: int = 30,
    device: str = "auto",
) -> dict:
    """
    Run evaluation on the frozen eval set.

    Args:
        base_model_id: HuggingFace model ID
        adapter_path: Path to LoRA adapter (None = base model only)
        eval_path: Path to eval.jsonl
        metric_name: Which metric to compute (must be in METRIC_FUNCTIONS)
        max_new_tokens: Max tokens to generate per example
        device: Device placement (ignored — auto-detected from config.DEVICE)

    Returns:
        dict with: metric_name, metric_value, predictions, references, num_examples
    """
    if metric_name not in METRIC_FUNCTIONS:
        raise ValueError(f"Unknown metric: {metric_name}. Available: {list(METRIC_FUNCTIONS.keys())}")

    # Load eval set
    eval_examples = _load_eval_set(eval_path)
    if not eval_examples:
        raise ValueError(f"Empty eval set at {eval_path}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_model_for_eval(base_model_id, adapter_path)
    model.eval()

    # Generate predictions
    predictions = []
    references = []

    for example in eval_examples:
        prompt_messages, reference = _extract_prompt_and_reference(example)
        prediction = _generate_prediction(model, tokenizer, prompt_messages, max_new_tokens)
        predictions.append(prediction)
        references.append(reference)

    # Compute metric
    metric_fn = METRIC_FUNCTIONS[metric_name]
    metric_value = metric_fn(predictions, references)

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "metric_name": metric_name,
        "metric_value": round(metric_value, 6),
        "predictions": predictions,
        "references": references,
        "num_examples": len(eval_examples),
    }
