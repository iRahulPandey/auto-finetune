"""
agent_loop.py — The autonomous search loop powered by Claude API.

This is the core of auto-finetune. It reads program.md, reads finetune.py,
proposes a hypothesis, edits the LoRA config in finetune.py, runs training,
evaluates on the frozen eval set, logs to MLflow, keeps improvements or
reverts regressions, and repeats.

The mapping from AutoResearch:
  - Claude Code agent  -> Claude API call (this file)
  - git commit         -> MLflow log_run + adapter save
  - git revert         -> discard adapter, keep finetune.py as-is
"""

import json
import re
import subprocess
import sys
from typing import Optional, Callable
from dataclasses import dataclass

from . import llm_client

from .config import (
    RunConfig,
    AGENT_CONFIG,
    CONSTRAINTS,
    PROGRAM_MD_PATH,
    ADAPTERS_DIR,
    PROJECT_ROOT,
)
from .mlflow_utils import (
    init_mlflow,
    log_run,
    get_run_history,
    save_best_adapter,
    register_best_model,
    format_history_for_agent,
    generate_config_table,
    format_config_table_for_agent,
)
# evaluate() no longer called here — finetune.py runs eval in-process
# to avoid loading the model twice per iteration


_DIAGNOSIS_PROMPT_TEMPLATE = """\
You are an ML debugging assistant. A fine-tuning iteration just completed with a low {metric_name} of {metric_value:.4f}.

## Config used
LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}, modules={target_modules}
Training: lr={learning_rate}, epochs={num_epochs}, sched={scheduler}, bs={batch_size}×ga={grad_acc}

## Sample prediction mismatches ({n_mismatches} total)
{mismatch_block}

## Task type: {task_type}

Diagnose the most likely failure mode in 1-2 sentences. Be specific and actionable.
Focus on: underfitting vs overfitting, learning rate too high/low, wrong target modules,
insufficient epochs, label format errors, or model collapse.
Do NOT say "try more experiments". Give a concrete hypothesis about what went wrong."""


@dataclass
class IterationResult:
    """Result of a single iteration."""
    iteration: int
    hypothesis: str
    metric_value: float
    train_loss: float
    is_improvement: bool
    lora_config: dict
    training_args: dict
    adapter_path: Optional[str]
    run_id: str
    error: Optional[str] = None
    diagnosis: Optional[str] = None


def _read_finetune_py() -> str:
    """Read the current finetune.py source."""
    return (PROJECT_ROOT / "finetune.py").read_text(encoding="utf-8")


def _write_finetune_py(content: str) -> None:
    """Write updated finetune.py."""
    (PROJECT_ROOT / "finetune.py").write_text(content, encoding="utf-8")


def _extract_config_from_finetune(source: str) -> dict:
    """Extract current LORA_CONFIG and TRAINING_ARGS from finetune.py source."""
    config = {}

    # Extract HYPOTHESIS
    match = re.search(r'HYPOTHESIS\s*=\s*"([^"]*)"', source)
    if match:
        config["hypothesis"] = match.group(1)

    # Extract LORA_CONFIG dict
    match = re.search(r'LORA_CONFIG\s*=\s*(\{[^}]+\})', source, re.DOTALL)
    if match:
        raw = match.group(1)
        raw = re.sub(r'#.*', '', raw)  # remove comments
        raw = raw.replace("'", '"')
        raw = re.sub(r',\s*}', '}', raw)  # trailing commas
        try:
            config["lora_config"] = json.loads(raw)
        except json.JSONDecodeError:
            pass

    # Extract TRAINING_ARGS dict
    match = re.search(r'TRAINING_ARGS\s*=\s*(\{[^}]+\})', source, re.DOTALL)
    if match:
        raw = match.group(1)
        raw = re.sub(r'#.*', '', raw)
        raw = raw.replace("'", '"')
        raw = raw.replace("True", "true").replace("False", "false")
        raw = re.sub(r',\s*}', '}', raw)
        try:
            config["training_args"] = json.loads(raw)
        except json.JSONDecodeError:
            pass

    return config


def _build_agent_prompt(
    program_md: str,
    run_history_str: str,
    config_table_str: str,
    best_metric: float,
    metric_name: str,
    iteration: int,
    max_iterations: int,
    stagnation_count: int,
    last_per_class: str = "",
    last_diagnosis: Optional[str] = None,
) -> str:
    """Build the prompt for the Claude API agent."""
    iter_display = "∞" if max_iterations >= 9999 else str(max_iterations)
    progress_pct = iteration / max_iterations if max_iterations < 9999 else 0
    is_final = (iteration == max_iterations and max_iterations < 9999)

    if progress_pct < 0.4:
        strategy = "EXPLORE: try configs that are different from what's been run. Vary lr, rank, modules freely."
    elif progress_pct < 0.75:
        strategy = "EXPLOIT: start from the best run so far and make one targeted change at a time."
    else:
        strategy = "REFINE: make small surgical changes (±lr 20%, ±1 epoch, different dropout) from the best config."

    stagnation_note = ""
    if stagnation_count >= AGENT_CONFIG["stagnation_threshold"]:
        stagnation_note = (
            f"\n⚠ STAGNATION: {stagnation_count} runs without improvement. "
            "Pick something meaningfully different — different lr range, different modules, or different rank."
        )

    return f"""You are an autonomous ML engineer running a LoRA hyperparameter search.

## Task (program.md)
{program_md}

## Run History
{run_history_str}

## Config Table
{config_table_str}

## Current State
- Iteration {iteration} / {iter_display} — Strategy: {strategy}{stagnation_note}
- Best {metric_name} so far: {best_metric:.6f}
{f"- Per-class accuracy last run: {last_per_class}" if last_per_class else ""}
{"- WARNING: model collapsing to one label (0/ in per-class). Change modules or lower lr." if last_per_class and "0/" in last_per_class else ""}
{f"- Last run diagnosis: {last_diagnosis}" if last_diagnosis else ""}
{"- FINAL ITERATION: summarise what was learned in your hypothesis." if is_final else ""}

## Instructions
1. Read the run history and config table above.
2. Pick a config_id from the PENDING list — or propose a custom config if the table misses a clearly better region.
3. If picking from table: use its exact parameters. If custom: set config_id to null.
4. Don't repeat a config that's already been run.

## Output — return ONLY this JSON:
```json
{{
    "config_id": <int or null>,
    "hypothesis": "<one sentence: what you expect and why>",
    "lora_config": {{
        "r": <8|16|32|64>,
        "lora_alpha": <int, 2x r>,
        "lora_dropout": <0.0|0.05|0.1>,
        "target_modules": <["q_proj","v_proj"] or similar>,
        "task_type": "CAUSAL_LM",
        "bias": "none"
    }},
    "training_args": {{
        "learning_rate": <{CONSTRAINTS['min_learning_rate']}–{CONSTRAINTS['max_learning_rate']}>,
        "num_train_epochs": <{CONSTRAINTS['min_epochs']}–{CONSTRAINTS['max_epochs']}>,
        "lr_scheduler_type": "<cosine|linear|constant_with_warmup>",
        "per_device_train_batch_size": <1|2|4>,
        "gradient_accumulation_steps": <2|4|8>,
        "warmup_ratio": <0.05|0.1|0.15|0.2>,
        "logging_steps": 10,
        "save_strategy": "no",
        "optim": "adamw_torch",
        "remove_unused_columns": false,
        "report_to": "none"
    }}
}}
```"""


def _call_agent_llm(prompt: str) -> dict:
    """Call the configured LLM for the agent stage and parse JSON response."""
    agent_cfg = llm_client.get_stage_config(llm_client.STAGE_AGENT)
    print(f"  Agent LLM: {agent_cfg.label}")

    raw = llm_client.generate(
        prompt=prompt,
        stage=llm_client.STAGE_AGENT,
        model_hint="smart",
        max_tokens=AGENT_CONFIG["max_tokens"],
        temperature=AGENT_CONFIG["temperature"],
    )

    # Extract JSON from response (handle markdown code blocks)
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if json_match:
        raw = json_match.group(1)

    # Try to find raw JSON object
    if not raw.startswith("{"):
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)

    return json.loads(raw)


def _apply_config_to_finetune(proposed: dict) -> str:
    """
    Generate the updated finetune.py source with the proposed config.
    Only modifies HYPOTHESIS, LORA_CONFIG, and TRAINING_ARGS.
    """
    source = _read_finetune_py()

    # Update HYPOTHESIS
    hypothesis = proposed["hypothesis"].replace('"', '\\"')
    source = re.sub(
        r'HYPOTHESIS\s*=\s*"[^"]*"',
        f'HYPOTHESIS = "{hypothesis}"',
        source,
    )

    # Update LORA_CONFIG
    lora = proposed["lora_config"]
    lora_str = f"""LORA_CONFIG = {{
    "r": {lora['r']},
    "lora_alpha": {lora['lora_alpha']},
    "lora_dropout": {lora['lora_dropout']},
    "target_modules": {json.dumps(lora['target_modules'])},
    "task_type": "CAUSAL_LM",
    "bias": "{lora.get('bias', 'none')}",
}}"""
    source = re.sub(
        r'LORA_CONFIG\s*=\s*\{[^}]+\}',
        lora_str,
        source,
        flags=re.DOTALL,
    )

    # Update TRAINING_ARGS — use every parameter Claude proposed, no silent overrides
    ta = proposed["training_args"]
    ta_str = f"""TRAINING_ARGS = {{
    "learning_rate": {ta['learning_rate']},
    "num_train_epochs": {ta['num_train_epochs']},
    "lr_scheduler_type": "{ta['lr_scheduler_type']}",
    "per_device_train_batch_size": {ta.get('per_device_train_batch_size', 4)},
    "gradient_accumulation_steps": {ta.get('gradient_accumulation_steps', 4)},
    "warmup_ratio": {ta.get('warmup_ratio', 0.1)},
    "logging_steps": {ta.get('logging_steps', 10)},
    "save_strategy": "no",
    "optim": "{ta.get('optim', 'adamw_torch')}",
    "remove_unused_columns": False,
    "report_to": "none",
}}"""
    source = re.sub(
        r'TRAINING_ARGS\s*=\s*\{[^}]+\}',
        ta_str,
        source,
        flags=re.DOTALL,
    )

    return source


def _run_training(
    run_config: RunConfig,
    session_id: str,
    iteration: int = 0,
    on_output: Optional[Callable[[str], None]] = None,
    eval_path: Optional[str] = None,
    metric_name: Optional[str] = None,
) -> dict:
    """Execute finetune.py as a subprocess, streaming stdout line-by-line.

    If eval_path and metric_name are provided, the subprocess runs eval
    immediately after training while the model is still in memory (avoids
    a second model load that was wasting ~5s per iteration).

    on_output(line) is called for every stdout line so the UI can show
    live progress (model download, epoch loss, etc.).
    """
    config_path = PROJECT_ROOT / "data" / session_id / "run_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    run_config_dict = {
        "hf_model_id": run_config.hf_model_id,
        "train_path": str(PROJECT_ROOT / "data" / session_id / "train.jsonl"),
        "task_type": run_config.task_type,
    }
    config_path.write_text(json.dumps(run_config_dict, indent=2))

    output_dir = ADAPTERS_DIR / "runs" / session_id / f"iter_{iteration:03d}"

    cmd = [
        sys.executable, str(PROJECT_ROOT / "finetune.py"),
        "--config", str(config_path),
        "--output-dir", str(output_dir),
    ]
    if eval_path and metric_name:
        cmd.extend(["--eval-path", eval_path, "--metric-name", metric_name])

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge stderr so on_output sees everything
        text=True,
        cwd=str(PROJECT_ROOT),
        bufsize=1,                  # line-buffered
    )

    captured_lines: list[str] = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        captured_lines.append(line)
        print(line)                 # always echo to agent_loop stdout
        if on_output:
            on_output(line)

    proc.wait()
    if proc.returncode != 0:
        tail = "\n".join(captured_lines[-40:])
        raise RuntimeError(f"Training subprocess exited {proc.returncode}:\n{tail}")

    result_path = output_dir / "train_result.json"
    with open(result_path) as f:
        return json.load(f)


def _config_fingerprint(proposed: dict) -> str:
    """Create a hashable fingerprint from a proposed config for dedup."""
    lora = proposed.get("lora_config", {})
    ta = proposed.get("training_args", {})
    parts = [
        str(ta.get("learning_rate", 0)),
        str(lora.get("r", 0)),
        str(ta.get("num_train_epochs", 0)),
        str(ta.get("lr_scheduler_type", "")),
        str(sorted(lora.get("target_modules", []))),
        str(ta.get("per_device_train_batch_size", 0)),
        str(ta.get("gradient_accumulation_steps", 0)),
        str(ta.get("warmup_ratio", 0)),
        str(lora.get("lora_dropout", 0)),
    ]
    return "|".join(parts)


def _mutate_config(proposed: dict) -> dict:
    """Apply a random small mutation to avoid exact config repetition."""
    import random
    proposed = json.loads(json.dumps(proposed))  # deep copy

    # Pick a random parameter to mutate
    mutations = [
        lambda p: p["training_args"].__setitem__("warmup_ratio",
            random.choice([v for v in [0.05, 0.1, 0.15, 0.2]
                           if v != p["training_args"].get("warmup_ratio")])),
        lambda p: p["lora_config"].__setitem__("lora_dropout",
            random.choice([v for v in [0.0, 0.05, 0.1]
                           if v != p["lora_config"].get("lora_dropout")])),
        lambda p: p["training_args"].__setitem__("gradient_accumulation_steps",
            random.choice([v for v in [2, 4, 8]
                           if v != p["training_args"].get("gradient_accumulation_steps")])),
    ]
    random.choice(mutations)(proposed)
    proposed["hypothesis"] += " [auto-mutated to avoid exact repeat]"
    return proposed


def _diagnose_failure(
    mismatches: list[dict],
    metric_value: float,
    metric_name: str,
    lora_config: dict,
    training_args: dict,
    task_type: str,
) -> Optional[str]:
    """Call the LLM to diagnose why a run underperformed.

    Only called when metric_value is below 50% of expected range to avoid
    burning tokens on successful or near-successful runs.
    Returns a short diagnosis string, or None if the call fails.
    """
    if not mismatches:
        return None

    mismatch_lines = []
    for mm in mismatches[:5]:
        exp = mm.get("expected", "")[:80]
        got = mm.get("predicted", "")[:80]
        mismatch_lines.append(f"  expected: {exp!r}")
        mismatch_lines.append(f"  got:      {got!r}")
        mismatch_lines.append("")
    mismatch_block = "\n".join(mismatch_lines).rstrip()

    prompt = _DIAGNOSIS_PROMPT_TEMPLATE.format(
        metric_name=metric_name,
        metric_value=metric_value,
        lora_r=lora_config.get("r", "?"),
        lora_alpha=lora_config.get("lora_alpha", "?"),
        lora_dropout=lora_config.get("lora_dropout", "?"),
        target_modules=lora_config.get("target_modules", []),
        learning_rate=training_args.get("learning_rate", "?"),
        num_epochs=training_args.get("num_train_epochs", "?"),
        scheduler=training_args.get("lr_scheduler_type", "?"),
        batch_size=training_args.get("per_device_train_batch_size", "?"),
        grad_acc=training_args.get("gradient_accumulation_steps", "?"),
        task_type=task_type,
        n_mismatches=len(mismatches),
        mismatch_block=mismatch_block,
    )

    try:
        diagnosis = llm_client.generate(
            prompt=prompt,
            stage=llm_client.STAGE_AGENT,
            model_hint="smart",
            max_tokens=256,
            temperature=0.2,
        )
        return diagnosis.strip()[:500]
    except Exception as e:
        print(f"  [diagnosis] LLM call failed: {e}")
        return None


def _validate_proposed_config(proposed: dict) -> list[str]:
    """Validate that the proposed config respects constraints."""
    errors = []

    lora = proposed.get("lora_config", {})
    ta = proposed.get("training_args", {})

    if lora.get("r", 0) > CONSTRAINTS["max_lora_rank"]:
        errors.append(f"lora_rank {lora['r']} exceeds max {CONSTRAINTS['max_lora_rank']}")

    lr = ta.get("learning_rate", 0)
    if not (CONSTRAINTS["min_learning_rate"] <= lr <= CONSTRAINTS["max_learning_rate"]):
        errors.append(f"learning_rate {lr} out of range")

    epochs = ta.get("num_train_epochs", 0)
    if not (CONSTRAINTS["min_epochs"] <= epochs <= CONSTRAINTS["max_epochs"]):
        errors.append(f"num_train_epochs {epochs} out of range")

    if "hypothesis" not in proposed:
        errors.append("Missing hypothesis")

    return errors


def run_agent_loop(
    run_config: RunConfig,
    session_id: str,
    on_iteration_complete: Optional[Callable[[IterationResult], None]] = None,
    on_training_output: Optional[Callable[[str], None]] = None,
    on_status: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Execute the full autonomous search loop.

    Args:
        run_config: The RunConfig for this session
        session_id: Unique session identifier
        on_iteration_complete: Called after each complete iteration
        on_training_output: Called for each stdout line from finetune.py
        on_status: Called with short status strings ("Calling Claude API...", etc.)

    Returns:
        dict with: best_metric, best_run_id, total_iterations, history
    """
    def _status(msg: str) -> None:
        print(msg)
        if on_status:
            on_status(msg)

    # One MLflow experiment per use-case (slug of task description)
    init_mlflow(use_case=run_config.use_case)

    # ── Free any leftover GPU/MPS memory from previous sessions ──────────
    import gc
    import torch
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _status("Memory cleared before training loop.")

    # Pre-download model weights so subprocess iterations skip the HF Hub check.
    # huggingface_hub.snapshot_download returns the local cache path; we set
    # TRANSFORMERS_OFFLINE=1 so subsequent from_pretrained calls never hit the network.
    _status("Pre-downloading model weights (one-time)...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(run_config.hf_model_id, local_files_only=False)
        import os as _os
        _os.environ["HF_HUB_OFFLINE"] = "1"
        _status(f"Model cached locally. Subprocess iterations will skip network calls.")
    except Exception as e:
        _status(f"Pre-download skipped ({e}) — will download in subprocess.")

    # Read program.md once — agents reads this to understand the task
    program_md = PROGRAM_MD_PATH.read_text(encoding="utf-8")

    best_metric_value = 0.0
    best_run_id = None
    best_adapter_path = None
    stagnation_count = 0
    results = []
    tried_fingerprints: set[str] = set()  # dedup: prevent exact config repeats
    last_per_class: str = ""  # per-class accuracy from last iteration
    last_diagnosis: Optional[str] = None  # failure diagnosis from last iteration

    eval_path = str(PROJECT_ROOT / "data" / session_id / "eval.jsonl")

    # ── Generate or load pre-planned config table ──────────────────────────
    config_table_path = PROJECT_ROOT / "data" / session_id / "config_table.json"
    if config_table_path.exists():
        _status("Loading existing config table...")
        with open(config_table_path) as f:
            config_table = json.load(f)
        _status(f"Config table loaded: {sum(1 for c in config_table if c['status'] == 'completed')} done, "
                f"{sum(1 for c in config_table if c['status'] == 'pending')} pending")
    else:
        _status(f"Generating pre-planned config table ({run_config.max_iterations} configs)...")
        config_table = generate_config_table(run_config.max_iterations, task_type=run_config.task_type)
        config_table_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_table_path, "w") as f:
            json.dump(config_table, f, indent=2)
        _status(f"Config table generated: {len(config_table)} diverse configurations planned")

    def _save_config_table():
        with open(config_table_path, "w") as f:
            json.dump(config_table, f, indent=2)

    for iteration in range(1, run_config.max_iterations + 1):
        iter_display = "∞" if run_config.max_iterations >= 9999 else str(run_config.max_iterations)
        pending_count = sum(1 for c in config_table if c["status"] == "pending")
        _status(f"{'=' * 60}")
        _status(
            f"ITERATION {iteration}/{iter_display}  |  "
            f"best so far: {best_metric_value:.4f}  |  "
            f"stagnation: {stagnation_count}  |  "
            f"table: {pending_count} pending"
        )
        _status(f"{'=' * 60}")

        picked_id = None  # track which table entry was selected
        try:
            # Get run history and format for LLM
            history = get_run_history(session_id=session_id)
            history_str = format_history_for_agent(history, run_config.metric_name)
            config_table_str = format_config_table_for_agent(config_table, run_config.metric_name)

            prompt = _build_agent_prompt(
                program_md=program_md,
                run_history_str=history_str,
                config_table_str=config_table_str,
                best_metric=best_metric_value,
                metric_name=run_config.metric_name,
                iteration=iteration,
                max_iterations=run_config.max_iterations,
                stagnation_count=stagnation_count,
                last_per_class=last_per_class,
                last_diagnosis=last_diagnosis,
            )

            # Call Claude API for next config
            _status(f"[{iteration}/{run_config.max_iterations}] Asking Claude to pick from config table...")
            proposed = _call_agent_llm(prompt)

            # If LLM picked a config_id, mark it in the table
            picked_id = proposed.get("config_id")
            if picked_id is not None:
                for cfg in config_table:
                    if cfg["config_id"] == picked_id and cfg["status"] == "pending":
                        cfg["status"] = "running"
                        _status(f"[{iteration}] Picked config #{picked_id} from table")
                        break
                else:
                    _status(f"[{iteration}] config_id #{picked_id} not found or already done — using as custom")
                    picked_id = None

            # Validate
            validation_errors = _validate_proposed_config(proposed)
            if validation_errors:
                _status(f"Invalid config from Claude: {validation_errors} — skipping")
                # Revert table status if we picked one
                if picked_id is not None:
                    for cfg in config_table:
                        if cfg["config_id"] == picked_id:
                            cfg["status"] = "pending"
                continue

            # Dedup: if this exact config was already tried, auto-mutate
            fp = _config_fingerprint(proposed)
            dedup_attempts = 0
            while fp in tried_fingerprints and dedup_attempts < 5:
                _status(f"[{iteration}] Config already tried — auto-mutating to avoid repeat")
                proposed = _mutate_config(proposed)
                fp = _config_fingerprint(proposed)
                dedup_attempts += 1
            tried_fingerprints.add(fp)

            hypothesis = proposed["hypothesis"]
            ta_p = proposed["training_args"]
            lc_p = proposed["lora_config"]
            _status(f"[{iteration}] Hypothesis: {hypothesis}")
            _status(
                f"[{iteration}] Config: lr={ta_p.get('learning_rate')} "
                f"rank={lc_p.get('r')} alpha={lc_p.get('lora_alpha')} "
                f"ep={ta_p.get('num_train_epochs')} sched={ta_p.get('lr_scheduler_type')} "
                f"warmup={ta_p.get('warmup_ratio')} "
                f"bs={ta_p.get('per_device_train_batch_size')}×ga={ta_p.get('gradient_accumulation_steps')} "
                f"dropout={lc_p.get('lora_dropout')} "
                f"modules={lc_p.get('target_modules')}"
            )

            # Apply config to finetune.py
            updated_source = _apply_config_to_finetune(proposed)
            _write_finetune_py(updated_source)

            # Run training + eval in one subprocess (model loaded ONCE, not twice)
            _status(f"[{iteration}] Starting training + eval subprocess...")
            train_result = _run_training(
                run_config, session_id, iteration,
                on_output=on_training_output,
                eval_path=eval_path,
                metric_name=run_config.metric_name,
            )
            train_loss = train_result["train_loss"]
            adapter_path = train_result["adapter_path"]
            metric_value = train_result.get("metric_value", 0.0)
            _status(
                f"[{iteration}] Done — loss={train_loss:.4f}  "
                f"{run_config.metric_name}={metric_value:.4f}"
            )

            # Capture per-class accuracy for next iteration's prompt
            per_class = train_result.get("per_class_accuracy", {})
            if per_class:
                last_per_class = str(per_class)

            # Update config table entry if we picked from it
            if picked_id is not None:
                for cfg in config_table:
                    if cfg["config_id"] == picked_id:
                        cfg["status"] = "completed"
                        cfg["result_metric"] = metric_value
                        cfg["result_loss"] = train_loss
                        break
                _save_config_table()

            # Log mismatches for debugging if available
            mismatches = train_result.get("mismatches", [])
            if mismatches:
                _status(f"[{iteration}] Sample mismatches ({len(mismatches)} total, showing 3):")
                for mm in mismatches[:3]:
                    exp = mm["expected"]
                    got = mm["predicted"]
                    # Find where they actually diverge
                    diff_pos = next(
                        (i for i, (a, b) in enumerate(zip(exp, got)) if a != b),
                        min(len(exp), len(got)),
                    )
                    if diff_pos > 40:
                        _status(f"  diverges at char {diff_pos}:")
                        _status(f"    expected: ...{exp[max(0,diff_pos-15):diff_pos+40]}")
                        _status(f"    got:      ...{got[max(0,diff_pos-15):diff_pos+40]}")
                    elif len(got) < len(exp) and got == exp[:len(got)]:
                        _status(f"  truncated at {len(got)} chars (expected {len(exp)})")
                    else:
                        _status(f"  expected: {exp[:100]}")
                        _status(f"  got:      {got[:100]}")

            # Diagnose failure when metric is poor and we have mismatch data
            iteration_diagnosis: Optional[str] = None
            if mismatches and metric_value < 0.5:
                _status(f"[{iteration}] Running failure diagnosis...")
                iteration_diagnosis = _diagnose_failure(
                    mismatches=mismatches,
                    metric_value=metric_value,
                    metric_name=run_config.metric_name,
                    lora_config=proposed["lora_config"],
                    training_args=proposed["training_args"],
                    task_type=run_config.task_type,
                )
                if iteration_diagnosis:
                    _status(f"[{iteration}] Diagnosis: {iteration_diagnosis}")
                    last_diagnosis = iteration_diagnosis

            is_improvement = metric_value > best_metric_value

            if is_improvement:
                _status(
                    f"[{iteration}/{run_config.max_iterations}] "
                    f"IMPROVED  {run_config.metric_name}: "
                    f"{best_metric_value:.4f} -> {metric_value:.4f}"
                )
                best_metric_value = metric_value
                best_adapter_path = adapter_path
                save_best_adapter(adapter_path)
                stagnation_count = 0
            else:
                _status(
                    f"[{iteration}/{run_config.max_iterations}] "
                    f"No improvement ({metric_value:.4f}).  "
                    f"Best stays: {best_metric_value:.4f}"
                )
                stagnation_count += 1

            # Log to MLflow — always, regardless of improvement
            run_id = log_run(
                iteration=iteration,
                hypothesis=hypothesis,
                lora_config=proposed["lora_config"],
                training_args=proposed["training_args"],
                train_loss=train_loss,
                metric_name=run_config.metric_name,
                metric_value=metric_value,
                adapter_path=adapter_path,
                is_improvement=is_improvement,
                session_id=session_id,
                use_case=run_config.use_case,
                base_model_id=run_config.hf_model_id,
                diagnosis=iteration_diagnosis,
            )

            if is_improvement:
                best_run_id = run_id

            iter_result = IterationResult(
                iteration=iteration,
                hypothesis=hypothesis,
                metric_value=metric_value,
                train_loss=train_loss,
                is_improvement=is_improvement,
                lora_config=proposed["lora_config"],
                training_args=proposed["training_args"],
                adapter_path=adapter_path,
                run_id=run_id,
                diagnosis=iteration_diagnosis,
            )
            results.append(iter_result)

            if on_iteration_complete:
                on_iteration_complete(iter_result)

            # Free memory between iterations (subprocess already cleaned up
            # its own model, but gc + cache flush catches any leaked refs)
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Early stopping check
            if run_config.target_threshold and metric_value > run_config.target_threshold:
                print(f"\nTARGET REACHED! {run_config.metric_name} = {metric_value:.6f} > {run_config.target_threshold}")
                break

        except Exception as e:
            print(f"ERROR in iteration {iteration}: {e}")
            # Revert table entry if we picked from it
            if picked_id is not None:
                for cfg in config_table:
                    if cfg["config_id"] == picked_id and cfg["status"] == "running":
                        cfg["status"] = "pending"  # re-queue for retry
                _save_config_table()
            iter_result = IterationResult(
                iteration=iteration,
                hypothesis="ERROR",
                metric_value=0.0,
                train_loss=0.0,
                is_improvement=False,
                lora_config={},
                training_args={},
                adapter_path=None,
                run_id="",
                error=str(e),
            )
            results.append(iter_result)
            if on_iteration_complete:
                on_iteration_complete(iter_result)
            continue

    # Register best model
    if best_run_id:
        print(f"\nRegistering best model (run_id={best_run_id})...")
        register_best_model(best_run_id)

    summary = {
        "best_metric_name": run_config.metric_name,
        "best_metric_value": best_metric_value,
        "best_run_id": best_run_id,
        "best_adapter_path": str(best_adapter_path) if best_adapter_path else None,
        "total_iterations": len(results),
        "improvements": sum(1 for r in results if r.is_improvement),
        "errors": sum(1 for r in results if r.error),
        "session_id": session_id,
    }

    # Save summary
    summary_path = PROJECT_ROOT / "data" / session_id / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'=' * 60}")
    print("SEARCH COMPLETE")
    print(f"Best {run_config.metric_name}: {best_metric_value:.6f}")
    print(f"Total iterations: {len(results)}")
    print(f"Improvements: {summary['improvements']}")
    print(f"Best adapter: {ADAPTERS_DIR / 'best'}")
    print(f"{'=' * 60}")

    return summary
