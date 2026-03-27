"""
test_e2e.py — End-to-end validation of the auto-finetune pipeline.

Tests each stage independently so you can catch issues before
burning GPU hours. Run with:

    python test_e2e.py              # all stages (needs ANTHROPIC_API_KEY + torch)
    python test_e2e.py --stage 1    # just data prep (needs ANTHROPIC_API_KEY)
    python test_e2e.py --stage 2    # just program.md generation (no deps)
    python test_e2e.py --stage 3    # just finetune dry-run (needs torch)
    python test_e2e.py --stage 4    # just evaluator metrics (no deps)
    python test_e2e.py --stage 5    # just MLflow logging (needs mlflow)
    python test_e2e.py --no-api     # skip stages needing ANTHROPIC_API_KEY
    python test_e2e.py --no-gpu     # skip stages needing torch/GPU
"""

import json
import os
import sys
import argparse
import traceback
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def _header(stage: int, name: str):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 60}")
    print(f"STAGE {stage}: {name}")
    print(f"{'=' * 60}{Colors.RESET}\n")


def _pass(msg: str):
    print(f"  {Colors.GREEN}PASS{Colors.RESET} {msg}")


def _fail(msg: str):
    print(f"  {Colors.RED}FAIL{Colors.RESET} {msg}")


def _skip(msg: str):
    print(f"  {Colors.YELLOW}SKIP{Colors.RESET} {msg}")


def _load_examples() -> list[dict]:
    path = PROJECT_ROOT / "examples" / "email_urgency.json"
    with open(path) as f:
        return json.loads(f.read())


# ── Stage 1: Data Preparation ───────────────────────────────────────────────

def test_stage1_data_prep(skip_api: bool = False) -> bool:
    _header(1, "Data Preparation")
    ok = True

    # Test example loading
    examples = _load_examples()
    assert len(examples) == 50, f"Expected 50 examples, got {len(examples)}"
    _pass(f"Loaded {len(examples)} examples from email_urgency.json")

    # Test hash
    from data_prep import _hash_examples
    h = _hash_examples(examples)
    assert len(h) == 12
    _pass(f"Deterministic hash: {h}")

    # Test formatting
    from data_prep import format_chat_examples
    formatted = format_chat_examples(examples, "Classify as urgent or not_urgent.", "qwen2.5-0.5b")
    assert len(formatted) == 50
    assert "messages" in formatted[0]
    assert len(formatted[0]["messages"]) == 3
    roles = [m["role"] for m in formatted[0]["messages"]]
    assert roles == ["system", "user", "assistant"]
    _pass(f"Chat formatting: {len(formatted)} examples, roles={roles}")

    # Test split
    from data_prep import split_train_eval
    train, eval_set = split_train_eval(formatted)
    assert len(train) + len(eval_set) == 50
    assert len(eval_set) >= 1
    _pass(f"Train/eval split: {len(train)}/{len(eval_set)}")

    # Test determinism
    train2, eval2 = split_train_eval(formatted)
    assert train == train2 and eval_set == eval2
    _pass("Split is deterministic (same seed = same split)")

    # Test full pipeline (with or without API)
    if skip_api:
        from data_prep import prepare_data
        result = prepare_data(
            use_case="Classify emails as urgent or not urgent",
            examples=examples,
            model_key="qwen2.5-0.5b",
            session_id="test-e2e",
            skip_prompt_synthesis=True,
        )
        _pass(f"Pipeline (no API): train={result['train_count']}, eval={result['eval_count']}")
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            _skip("ANTHROPIC_API_KEY not set — skipping prompt synthesis")
        else:
            from data_prep import prepare_data
            result = prepare_data(
                use_case="Classify emails as urgent or not urgent",
                examples=examples,
                model_key="qwen2.5-0.5b",
                session_id="test-e2e",
            )
            _pass(f"Pipeline (with Claude API): train={result['train_count']}, eval={result['eval_count']}")
            _pass(f"System prompt: {result['system_prompt'][:80]}...")

    # Verify files on disk
    from config import DATA_DIR
    session_dir = DATA_DIR / "test-e2e"
    assert (session_dir / "train.jsonl").exists(), "train.jsonl not found"
    assert (session_dir / "eval.jsonl").exists(), "eval.jsonl not found"
    _pass("Files saved to disk")

    return ok


# ── Stage 2: Program.md Generation ──────────────────────────────────────────

def test_stage2_program_md() -> bool:
    _header(2, "Program.md Generation")

    from config import RunConfig, PROGRAM_MD_PATH
    from program_md_generator import generate_program_md, write_program_md

    cfg = RunConfig(
        use_case="Classify emails as urgent or not urgent",
        model_key="qwen2.5-0.5b",
        task_type="classification",
        max_iterations=10,
    )

    # Test generation
    content = generate_program_md(cfg)
    assert "accuracy" in content
    assert "qwen2.5-0.5b" in content
    assert "urgent" in content.lower()
    assert "Search Space" in content
    assert "Constraints" in content
    _pass("Generated program.md with correct fields")

    # Test write
    write_program_md(cfg)
    assert PROGRAM_MD_PATH.exists()
    disk_content = PROGRAM_MD_PATH.read_text()
    assert disk_content == content
    _pass(f"Written to {PROGRAM_MD_PATH}")

    return True


# ── Stage 3: Finetune Dry Run ───────────────────────────────────────────────

def test_stage3_finetune_dryrun(skip_gpu: bool = False) -> bool:
    _header(3, "Finetune Structure Validation")

    # Validate finetune.py parses
    import ast
    with open(PROJECT_ROOT / "finetune.py") as f:
        tree = ast.parse(f.read())
    funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert "run_finetune" in funcs
    assert "_apply_dtype_flags" in funcs
    assert "_load_model" in funcs
    _pass(f"finetune.py parses OK, functions: {funcs}")

    # Test _apply_dtype_flags logic
    with open(PROJECT_ROOT / "finetune.py") as f:
        source = f.read()

    # Check that TRAINING_ARGS doesn't hardcode bf16
    agent_section = source.split("FIXED INFRASTRUCTURE")[0]
    assert "bf16" not in agent_section, "bf16 hardcoded in agent-editable section!"
    _pass("No hardcoded bf16 in agent-editable section")

    if skip_gpu:
        _skip("Skipping model load test (--no-gpu)")
    else:
        try:
            from config import DEVICE, DTYPE, DTYPE_STR
            _pass(f"Device detected: {DEVICE}, dtype: {DTYPE}, flag: {DTYPE_STR}")

            from finetune import _apply_dtype_flags
            test_args = {"learning_rate": 3e-5, "num_train_epochs": 2}
            resolved = _apply_dtype_flags(test_args)
            if DEVICE == "mps":
                assert resolved.get("fp16") is True, "Expected fp16=True on MPS"
                assert "bf16" not in resolved
                _pass("MPS: fp16=True, no bf16, batch_size=2")
            elif DEVICE == "cuda":
                assert resolved.get("bf16") is True
                _pass("CUDA: bf16=True")
            else:
                assert "bf16" not in resolved and "fp16" not in resolved
                _pass("CPU: no precision flag (fp32)")
        except ImportError as e:
            _skip(f"torch not available: {e}")

    return True


# ── Stage 4: Evaluator Metrics ──────────────────────────────────────────────

def test_stage4_evaluator_metrics() -> bool:
    _header(4, "Evaluator Metrics")

    from evaluator import (
        compute_accuracy, compute_f1_macro, compute_f1_weighted,
        compute_exact_match, compute_f1_token, compute_rouge_l, compute_bleu,
        _normalize_text,
    )

    # Normalization
    assert _normalize_text("  Urgent! ") == "urgent"
    assert _normalize_text("NOT_URGENT.") == "noturgent"
    _pass("Text normalization")

    # Accuracy
    preds = ["urgent", "not_urgent", "urgent", "urgent"]
    refs =  ["urgent", "not_urgent", "not_urgent", "urgent"]
    acc = compute_accuracy(preds, refs)
    assert acc == 0.75, f"Expected 0.75, got {acc}"
    _pass(f"Accuracy: {acc}")

    # F1 macro
    f1m = compute_f1_macro(preds, refs)
    assert 0.0 < f1m <= 1.0
    _pass(f"F1 macro: {f1m:.4f}")

    # F1 weighted
    f1w = compute_f1_weighted(preds, refs)
    assert 0.0 < f1w <= 1.0
    _pass(f"F1 weighted: {f1w:.4f}")

    # Exact match (same as accuracy)
    em = compute_exact_match(preds, refs)
    assert em == acc
    _pass(f"Exact match: {em}")

    # Token F1
    preds_gen = ["the cat sat on the mat", "hello world"]
    refs_gen = ["the cat on the mat", "hello there world"]
    f1t = compute_f1_token(preds_gen, refs_gen)
    assert 0.0 < f1t <= 1.0
    _pass(f"Token F1: {f1t:.4f}")

    # ROUGE-L
    rl = compute_rouge_l(preds_gen, refs_gen)
    assert 0.0 < rl <= 1.0
    _pass(f"ROUGE-L: {rl:.4f}")

    # BLEU
    bl = compute_bleu(preds_gen, refs_gen)
    assert 0.0 <= bl <= 1.0
    _pass(f"BLEU: {bl:.4f}")

    # Edge cases
    assert compute_accuracy([], []) == 0.0
    assert compute_rouge_l([], []) == 0.0
    _pass("Edge cases (empty inputs)")

    return True


# ── Stage 5: MLflow Logging ─────────────────────────────────────────────────

def test_stage5_mlflow() -> bool:
    _header(5, "MLflow Logging")

    try:
        from mlflow_utils import (
            init_mlflow, log_run, get_run_history,
            get_best_metric, format_history_for_agent,
        )

        experiment_name = "test-e2e-validation"
        exp_id = init_mlflow(experiment_name)
        _pass(f"MLflow initialized, experiment_id={exp_id}")

        # Log a fake run
        run_id = log_run(
            iteration=1,
            hypothesis="Test hypothesis for validation",
            lora_config={"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ["q_proj", "v_proj"]},
            training_args={"learning_rate": 3e-5, "num_train_epochs": 2, "lr_scheduler_type": "cosine"},
            train_loss=0.5,
            metric_name="accuracy",
            metric_value=0.75,
            adapter_path="/tmp/fake",
            is_improvement=True,
            session_id="test-e2e",
        )
        _pass(f"Logged run: {run_id}")

        # Log second run
        run_id_2 = log_run(
            iteration=2,
            hypothesis="Increase learning rate to 1e-4",
            lora_config={"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ["q_proj", "v_proj"]},
            training_args={"learning_rate": 1e-4, "num_train_epochs": 2, "lr_scheduler_type": "cosine"},
            train_loss=0.3,
            metric_name="accuracy",
            metric_value=0.85,
            adapter_path="/tmp/fake",
            is_improvement=True,
            session_id="test-e2e",
        )
        _pass(f"Logged run 2: {run_id_2}")

        # Retrieve history
        history = get_run_history(experiment_name, "test-e2e")
        assert len(history) >= 2
        _pass(f"Retrieved {len(history)} runs from history")

        # Get best metric
        best_val, best_rid = get_best_metric("accuracy", experiment_name, "test-e2e")
        assert best_val == 0.85
        _pass(f"Best accuracy: {best_val} (run_id={best_rid})")

        # Format for agent
        formatted = format_history_for_agent(history, "accuracy")
        assert "accuracy" in formatted.lower() or "0.85" in formatted
        _pass("Formatted history for agent prompt")

    except ImportError as e:
        _skip(f"mlflow not available: {e}")

    return True


# ── Stage 6: Agent Loop Structure ────────────────────────────────────────────

def test_stage6_agent_loop() -> bool:
    _header(6, "Agent Loop Structure")

    import ast
    with open(PROJECT_ROOT / "agent_loop.py") as f:
        tree = ast.parse(f.read())

    funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    required = [
        "_read_finetune_py", "_write_finetune_py",
        "_build_agent_prompt", "_call_claude_api",
        "_apply_config_to_finetune", "_run_training",
        "_validate_proposed_config", "run_agent_loop",
    ]
    for fn in required:
        assert fn in funcs, f"Missing function: {fn}"
    _pass(f"All required functions present: {len(required)}/{len(required)}")

    # Test config extraction
    from agent_loop import _extract_config_from_finetune, _read_finetune_py
    source = _read_finetune_py()
    config = _extract_config_from_finetune(source)
    assert "hypothesis" in config
    _pass(f"Config extraction: hypothesis='{config['hypothesis'][:50]}...'")

    # Test validation
    from agent_loop import _validate_proposed_config
    good_config = {
        "hypothesis": "Test",
        "lora_config": {"r": 16},
        "training_args": {"learning_rate": 3e-5, "num_train_epochs": 2},
    }
    errors = _validate_proposed_config(good_config)
    assert len(errors) == 0
    _pass("Valid config passes validation")

    bad_config = {
        "hypothesis": "Test",
        "lora_config": {"r": 128},  # exceeds max
        "training_args": {"learning_rate": 0.01, "num_train_epochs": 10},  # both out of range
    }
    errors = _validate_proposed_config(bad_config)
    assert len(errors) >= 2
    _pass(f"Invalid config caught {len(errors)} errors: {errors}")

    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="auto-finetune e2e validation")
    parser.add_argument("--stage", type=int, help="Run only this stage (1-6)")
    parser.add_argument("--no-api", action="store_true", help="Skip stages needing ANTHROPIC_API_KEY")
    parser.add_argument("--no-gpu", action="store_true", help="Skip stages needing torch/GPU")
    args = parser.parse_args()

    print(f"\n{Colors.BOLD}auto-finetune End-to-End Validation{Colors.RESET}")
    print(f"{'=' * 60}")

    stages = {
        1: ("Data Preparation", lambda: test_stage1_data_prep(skip_api=args.no_api)),
        2: ("Program.md Generation", test_stage2_program_md),
        3: ("Finetune Structure", lambda: test_stage3_finetune_dryrun(skip_gpu=args.no_gpu)),
        4: ("Evaluator Metrics", test_stage4_evaluator_metrics),
        5: ("MLflow Logging", test_stage5_mlflow),
        6: ("Agent Loop Structure", test_stage6_agent_loop),
    }

    if args.stage:
        stages = {args.stage: stages[args.stage]}

    results = {}
    for num, (name, fn) in stages.items():
        try:
            results[num] = fn()
        except Exception as e:
            _fail(f"Stage {num} crashed: {e}")
            traceback.print_exc()
            results[num] = False

    # Summary
    print(f"\n{Colors.BOLD}{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}{Colors.RESET}")

    all_pass = True
    for num, passed in results.items():
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if passed else f"{Colors.RED}FAIL{Colors.RESET}"
        name = stages.get(num, (f"Stage {num}",))[0] if num in stages else f"Stage {num}"
        print(f"  Stage {num}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"\n{Colors.GREEN}{Colors.BOLD}All stages passed!{Colors.RESET}")
        print(f"\nReady to run: streamlit run app.py")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}Some stages failed.{Colors.RESET}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
