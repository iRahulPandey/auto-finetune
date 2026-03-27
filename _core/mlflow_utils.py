"""
mlflow_utils.py — MLflow experiment logging and model registry.

One MLflow experiment per use-case (slug of the task description).
Sessions within the same use-case share an experiment; runs are named
"iter-001 | lr=1e-4 | r=16" so the MLflow UI is immediately readable.

Every function sets the tracking URI before touching MLflow — no
reliance on ambient state.
"""

import os
import re
import shutil
import time
from pathlib import Path
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient

from .config import MLFLOW_TRACKING_URI, BEST_ADAPTER_DIR


def _slugify(text: str, max_len: int = 50) -> str:
    """Turn a free-form task description into a clean experiment name.

    "Classify OWASP LLM risk categories" -> "classify-owasp-llm-risk-categories"
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    slug = text[:max_len].rstrip("-")
    return slug or "unnamed-task"


def _experiment_name(use_case: str) -> str:
    return _slugify(use_case) if use_case.strip() else "auto-finetune"


def _run_name(iteration: int, lr: float, rank: int) -> str:
    """Short, scannable run name visible in the MLflow UI."""
    lr_str = f"{lr:.0e}" if lr < 0.001 else str(lr)
    return f"iter-{iteration:03d} | lr={lr_str} | r={rank}"


def _ensure_tracking() -> None:
    """Always set tracking URI before any MLflow operation.

    Priority: MLFLOW_TRACKING_URI env var > config default.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(uri)


def init_mlflow(use_case: str = "") -> str:
    """Ensure the experiment for this use-case exists. Returns experiment name."""
    _ensure_tracking()
    exp_name = _experiment_name(use_case)
    if mlflow.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(exp_name)
    mlflow.set_experiment(exp_name)
    return exp_name


def log_run(
    iteration: int,
    hypothesis: str,
    lora_config: dict,
    training_args: dict,
    train_loss: float,
    metric_name: str,
    metric_value: float,
    adapter_path: str,
    is_improvement: bool,
    session_id: str,
    use_case: str = "",
    base_model_id: str = "",
    extra_metrics: Optional[dict] = None,
    diagnosis: Optional[str] = None,
) -> str:
    """
    Log a single iteration to MLflow. Always re-sets tracking URI + experiment
    so this works correctly even if called from a fresh import context.

    Returns: run_id
    """
    _ensure_tracking()
    mlflow.set_experiment(_experiment_name(use_case))

    lr = training_args.get("learning_rate", 0)
    rank = lora_config.get("r", 0)
    with mlflow.start_run(run_name=_run_name(iteration, lr, rank)) as run:

        # ── Tags (searchable, shown in run list) ──────────────────────────────
        mlflow.set_tag("session_id", session_id)
        mlflow.set_tag("iteration", str(iteration))
        mlflow.set_tag("hypothesis", hypothesis[:500])
        mlflow.set_tag("is_improvement", str(is_improvement))
        mlflow.set_tag("adapter_path", adapter_path)
        mlflow.set_tag("metric_name", metric_name)
        mlflow.set_tag("use_case", use_case[:200])
        mlflow.set_tag("base_model_id", base_model_id)
        mlflow.set_tag("timestamp", str(int(time.time())))
        if diagnosis:
            mlflow.set_tag("failure_diagnosis", diagnosis[:500])

        # ── Params ────────────────────────────────────────────────────────────
        mlflow.log_param("lora_r", lora_config.get("r"))
        mlflow.log_param("lora_alpha", lora_config.get("lora_alpha"))
        mlflow.log_param("lora_dropout", lora_config.get("lora_dropout"))
        mlflow.log_param("target_modules", str(lora_config.get("target_modules")))
        mlflow.log_param("learning_rate", training_args.get("learning_rate"))
        mlflow.log_param("num_train_epochs", training_args.get("num_train_epochs"))
        mlflow.log_param("lr_scheduler_type", training_args.get("lr_scheduler_type"))
        mlflow.log_param("batch_size", training_args.get("per_device_train_batch_size"))
        mlflow.log_param("gradient_accumulation_steps", training_args.get("gradient_accumulation_steps"))
        mlflow.log_param("warmup_ratio", training_args.get("warmup_ratio"))

        # ── Metrics ───────────────────────────────────────────────────────────
        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric(metric_name, metric_value)
        mlflow.log_metric("iteration", float(iteration))
        mlflow.log_metric("is_best", 1.0 if is_improvement else 0.0)

        if extra_metrics:
            for k, v in extra_metrics.items():
                mlflow.log_metric(k, v)

        return run.info.run_id


def _all_experiment_ids(client: MlflowClient) -> list[str]:
    """Return IDs of all user-created experiments (excludes MLflow's Default)."""
    experiments = client.search_experiments()
    return [
        e.experiment_id for e in experiments
        if e.name != "Default" and e.lifecycle_stage == "active"
    ]


def get_all_experiments() -> list[dict]:
    """Return all MLflow experiments with their runs, grouped by experiment.

    Each dict: experiment_id, experiment_name, sessions (list of session dicts),
    total_runs, best_metric, best_run_id.
    """
    _ensure_tracking()
    client = MlflowClient()
    experiments = client.search_experiments()

    result = []
    for exp in experiments:
        if exp.name == "Default" or exp.lifecycle_stage != "active":
            continue

        # Get all runs for this experiment
        all_runs = []
        page_token = None
        while True:
            runs_page = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["attributes.start_time ASC"],
                max_results=1000,
                page_token=page_token,
            )
            all_runs.extend(runs_page)
            page_token = runs_page.token if hasattr(runs_page, "token") else None
            if not page_token:
                break

        if not all_runs:
            continue

        # Group runs by session_id
        sessions: dict[str, dict] = {}
        parsed_runs = []
        for run in all_runs:
            tags = run.data.tags
            run_dict = {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "iteration": int(tags.get("iteration", 0)),
                "hypothesis": tags.get("hypothesis", ""),
                "is_improvement": tags.get("is_improvement", "False") == "True",
                "adapter_path": tags.get("adapter_path", ""),
                "metric_name": tags.get("metric_name", ""),
                "use_case": tags.get("use_case", ""),
                "base_model_id": tags.get("base_model_id", ""),
                "session_id": tags.get("session_id", ""),
                "timestamp": int(tags.get("timestamp", 0)),
                "metrics": dict(run.data.metrics),
                "params": dict(run.data.params),
            }
            parsed_runs.append(run_dict)

            sid = run_dict["session_id"]
            if not sid:
                continue
            if sid not in sessions:
                sessions[sid] = {
                    "session_id": sid,
                    "use_case": run_dict["use_case"],
                    "base_model_id": run_dict["base_model_id"],
                    "metric_name": run_dict["metric_name"],
                    "best_metric": 0.0,
                    "best_run_id": None,
                    "best_adapter_path": "",
                    "run_count": 0,
                    "runs": [],
                    "timestamp": run_dict["timestamp"],
                }
            sessions[sid]["run_count"] += 1
            sessions[sid]["runs"].append(run_dict)

            metric_val = run_dict["metrics"].get(run_dict["metric_name"], 0.0)
            if metric_val > sessions[sid]["best_metric"]:
                sessions[sid]["best_metric"] = metric_val
                sessions[sid]["best_run_id"] = run_dict["run_id"]
                sessions[sid]["best_adapter_path"] = run_dict["adapter_path"]

        # Experiment-level best
        best_metric = 0.0
        best_run_id = None
        metric_name = ""
        base_model_id = ""
        for s in sessions.values():
            if s["best_metric"] > best_metric:
                best_metric = s["best_metric"]
                best_run_id = s["best_run_id"]
            if not metric_name:
                metric_name = s["metric_name"]
            if not base_model_id:
                base_model_id = s["base_model_id"]

        sorted_sessions = sorted(
            sessions.values(), key=lambda s: s["timestamp"], reverse=True
        )

        result.append({
            "experiment_id": exp.experiment_id,
            "experiment_name": exp.name,
            "sessions": sorted_sessions,
            "total_runs": len(all_runs),
            "total_sessions": len(sessions),
            "best_metric": best_metric,
            "best_run_id": best_run_id,
            "metric_name": metric_name,
            "base_model_id": base_model_id,
            "all_runs": sorted(parsed_runs, key=lambda r: r.get("iteration", 0)),
        })

    # Sort experiments by most recent activity
    result.sort(
        key=lambda e: max((s["timestamp"] for s in e["sessions"]), default=0),
        reverse=True,
    )
    return result


def get_run_history(
    session_id: Optional[str] = None,
    use_case: Optional[str] = None,
    max_results: int = 0,
) -> list[dict]:
    """
    Retrieve run history from MLflow, sorted by iteration ascending.

    If use_case is given, searches only that experiment.
    If session_id is given, filters to that session.
    Otherwise returns runs from all experiments.

    max_results=0 (default) means fetch ALL runs using pagination.
    """
    _ensure_tracking()
    client = MlflowClient()

    if use_case:
        experiment = client.get_experiment_by_name(_experiment_name(use_case))
        if experiment is None:
            return []
        exp_ids = [experiment.experiment_id]
    else:
        exp_ids = _all_experiment_ids(client)
        if not exp_ids:
            return []

    filter_str = f"tags.session_id = '{session_id}'" if session_id else ""

    # Paginate through all results to avoid the 200-run cap
    all_runs = []
    page_token = None
    page_size = 1000  # MLflow max per page

    while True:
        result = client.search_runs(
            experiment_ids=exp_ids,
            filter_string=filter_str,
            order_by=["attributes.start_time ASC"],
            max_results=page_size,
            page_token=page_token,
        )
        all_runs.extend(result)

        # Check for more pages
        page_token = result.token if hasattr(result, "token") else None
        if not page_token or (max_results > 0 and len(all_runs) >= max_results):
            break

    if max_results > 0:
        all_runs = all_runs[:max_results]

    history = []
    for run in all_runs:
        tags = run.data.tags
        history.append({
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "iteration": int(tags.get("iteration", 0)),
            "hypothesis": tags.get("hypothesis", ""),
            "is_improvement": tags.get("is_improvement", "False") == "True",
            "adapter_path": tags.get("adapter_path", ""),
            "metric_name": tags.get("metric_name", ""),
            "use_case": tags.get("use_case", ""),
            "base_model_id": tags.get("base_model_id", ""),
            "session_id": tags.get("session_id", ""),
            "timestamp": int(tags.get("timestamp", 0)),
            "diagnosis": tags.get("failure_diagnosis", ""),
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
        })

    return sorted(history, key=lambda r: (r["session_id"], r["iteration"]))


def get_all_sessions(max_results: int = 200) -> list[dict]:
    """
    Return a deduplicated list of sessions from MLflow run history.
    Each session dict: session_id, use_case, base_model_id, best_metric, best_run_id, run_count
    """
    history = get_run_history(max_results=max_results)
    sessions: dict[str, dict] = {}

    for run in history:
        sid = run["session_id"]
        if not sid:
            continue
        if sid not in sessions:
            sessions[sid] = {
                "session_id": sid,
                "use_case": run["use_case"],
                "base_model_id": run["base_model_id"],
                "metric_name": run["metric_name"],
                "best_metric": 0.0,
                "best_run_id": None,
                "best_adapter_path": "",
                "run_count": 0,
                "runs": [],
                "timestamp": run["timestamp"],
            }
        sessions[sid]["run_count"] += 1
        sessions[sid]["runs"].append(run)

        metric_val = run["metrics"].get(run["metric_name"], 0.0)
        if metric_val > sessions[sid]["best_metric"]:
            sessions[sid]["best_metric"] = metric_val
            sessions[sid]["best_run_id"] = run["run_id"]
            sessions[sid]["best_adapter_path"] = run["adapter_path"]

    return sorted(sessions.values(), key=lambda s: s["timestamp"], reverse=True)


def get_best_metric(
    metric_name: str,
    session_id: Optional[str] = None,
) -> tuple[float, Optional[str]]:
    """Get best metric value and run_id. Returns (0.0, None) if no runs."""
    history = get_run_history(session_id)
    if not history:
        return 0.0, None

    best_value, best_run_id = 0.0, None
    for run in history:
        value = run["metrics"].get(metric_name, 0.0)
        if value > best_value:
            best_value = value
            best_run_id = run["run_id"]

    return best_value, best_run_id


def save_best_adapter(adapter_path: str) -> str:
    """Copy the best adapter to adapters/best/."""
    src = Path(adapter_path)
    dst = BEST_ADAPTER_DIR
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return str(dst)


def register_best_model(
    run_id: str,
    model_name: str = "auto-finetune-best",
) -> Optional[str]:
    """Register the best run's adapter in MLflow Model Registry as Production."""
    try:
        _ensure_tracking()
        client = MlflowClient()
        model_uri = f"runs:/{run_id}/adapter"
        result = mlflow.register_model(model_uri, model_name)
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Production",
            archive_existing_versions=True,
        )
        return result.version
    except Exception as e:
        print(f"Warning: Could not register model in MLflow: {e}")
        return None


def compute_parameter_insights(history: list[dict], metric_name: str) -> str:
    """Bayesian-inspired parameter analysis: compute which parameter values
    correlate with better metrics and lower loss.

    This is the "surrogate model" — instead of fitting a Gaussian Process,
    we compute simple statistics that tell the LLM exactly which settings
    work and which don't, so it doesn't have to guess from a raw table.

    Returns a structured text block with:
      1. Per-parameter value performance (avg metric, avg loss, n trials)
      2. Best-performing value for each parameter (marked)
      3. A suggested "expected improvement" config combining best values
      4. Untried parameter values (exploration opportunities)
    """
    from collections import defaultdict

    if len(history) < 3:
        return "(Not enough history for parameter analysis yet — need 3+ runs.)"

    # Parameters to analyze — these are the ones the agent can control
    param_keys = [
        ("learning_rate", "lr"),
        ("lora_r", "rank"),
        ("num_train_epochs", "epochs"),
        ("lr_scheduler_type", "scheduler"),
        ("target_modules", "modules"),
        ("batch_size", "batch_sz"),
        ("lora_dropout", "dropout"),
    ]

    lines: list[str] = []
    lines.append("=== PARAMETER PERFORMANCE ANALYSIS (Bayesian-inspired) ===")
    lines.append("For each parameter, avg metric & loss across all runs using that value:")
    lines.append("")

    suggested_config: dict[str, str] = {}
    best_metric_per_param: dict[str, tuple[str, float]] = {}

    for param_key, display_name in param_keys:
        value_metrics: dict[str, list[float]] = defaultdict(list)
        value_losses: dict[str, list[float]] = defaultdict(list)

        for run in history:
            val = run["params"].get(param_key, "unknown")
            metric = run["metrics"].get(metric_name, 0.0)
            loss = run["metrics"].get("train_loss", 999.0)
            val_str = str(val)
            value_metrics[val_str].append(metric)
            value_losses[val_str].append(loss)

        if not value_metrics:
            continue

        # Compute averages and find best
        avg_metrics = {v: sum(ms) / len(ms) for v, ms in value_metrics.items()}
        avg_losses = {v: sum(ls) / len(ls) for v, ls in value_losses.items()}
        best_val = max(avg_metrics, key=avg_metrics.get)
        worst_val = min(avg_metrics, key=avg_metrics.get)

        suggested_config[param_key] = best_val
        best_metric_per_param[param_key] = (best_val, avg_metrics[best_val])

        lines.append(f"  {display_name} ({param_key}):")
        # Sort by avg metric descending
        for val in sorted(avg_metrics.keys(), key=lambda v: avg_metrics[v], reverse=True):
            n = len(value_metrics[val])
            marker = ""
            if val == best_val:
                marker = "  ← BEST"
            elif val == worst_val and val != best_val:
                marker = "  ← WORST"
            lines.append(
                f"    {val:>20}: avg_{metric_name}={avg_metrics[val]:.4f}  "
                f"avg_loss={avg_losses[val]:.4f}  (n={n} trials){marker}"
            )
        lines.append("")

    # ── Suggested next config (acquisition function: combine best values) ──
    lines.append("=== SUGGESTED NEXT CONFIG (Expected Improvement) ===")
    lines.append("Combining the best-performing value for each parameter:")
    for param_key, display_name in param_keys:
        if param_key in suggested_config:
            val = suggested_config[param_key]
            score = best_metric_per_param[param_key][1]
            lines.append(f"  {display_name:>12} = {val}  (avg {metric_name}={score:.4f} when using this)")

    # ── Check if this exact combination has been tried ──
    tried_combos = set()
    for run in history:
        combo = tuple(str(run["params"].get(pk, "?")) for pk, _ in param_keys)
        tried_combos.add(combo)

    suggested_combo = tuple(suggested_config.get(pk, "?") for pk, _ in param_keys)
    if suggested_combo in tried_combos:
        lines.append("")
        lines.append("  NOTE: This exact combination was already tried.")
        lines.append("  → Try the SECOND-best value for one parameter to explore nearby.")
    else:
        lines.append("")
        lines.append("  This combination has NOT been tried — high expected improvement.")

    # ── Exploration gaps: parameter values never tried ──
    from .config import SEARCH_SPACE
    search_space_map = {
        "learning_rate": [str(v) for v in SEARCH_SPACE.get("learning_rate", [])],
        "lora_r": [str(v) for v in SEARCH_SPACE.get("lora_rank", [])],
        "num_train_epochs": [str(v) for v in SEARCH_SPACE.get("num_train_epochs", [])],
        "lr_scheduler_type": SEARCH_SPACE.get("lr_scheduler_type", []),
        "batch_size": [str(v) for v in SEARCH_SPACE.get("per_device_train_batch_size", [])],
    }

    gaps = []
    for param_key, display_name in param_keys:
        if param_key not in search_space_map:
            continue
        all_values = set(search_space_map[param_key])
        tried_values = set()
        for run in history:
            tried_values.add(str(run["params"].get(param_key, "")))
        untried = all_values - tried_values
        if untried:
            gaps.append(f"  {display_name}: never tried {sorted(untried)}")

    if gaps:
        lines.append("")
        lines.append("=== EXPLORATION GAPS (untried values) ===")
        for g in gaps:
            lines.append(g)

    return "\n".join(lines)


def generate_config_table(max_iterations: int, task_type: str = "classification") -> list[dict]:
    """Generate a diverse, pre-planned configuration table for systematic search.

    Uses stratified sampling to ensure all important parameter values are covered.
    The table is designed so that:
      - Every learning_rate value is tried at least once
      - Every batch_size value is tried at least once
      - Every target_modules combo is tried at least once
      - Every rank value is tried
      - The task-recommended module set appears 3x more in early coverage configs
      - Configs are diverse — no two share more than 3 identical params

    Returns a list of config dicts, each with lora_config + training_args.
    """
    import random

    from .config import SEARCH_SPACE, LAYER_RATIONALE

    lrs = SEARCH_SPACE["learning_rate"]
    ranks = SEARCH_SPACE["lora_rank"]
    epochs_list = SEARCH_SPACE["num_train_epochs"]
    schedulers = SEARCH_SPACE["lr_scheduler_type"]
    modules = SEARCH_SPACE["target_modules"]

    # Weight recommended modules 3x higher in the random pool so the agent
    # sees task-appropriate layers explored early rather than by chance
    recommended = LAYER_RATIONALE.get(task_type, LAYER_RATIONALE["classification"])["recommended"]
    weighted_modules = modules + [recommended, recommended, recommended]
    dropouts = SEARCH_SPACE["lora_dropout"]
    warmups = SEARCH_SPACE["warmup_ratio"]
    batch_sizes = SEARCH_SPACE["per_device_train_batch_size"]
    grad_accums = SEARCH_SPACE["gradient_accumulation_steps"]

    # Phase 1: Coverage configs — ensure every value of every key param is tried
    coverage_configs: list[dict] = []

    # One config per learning_rate
    for i, lr in enumerate(lrs):
        cfg = {
            "learning_rate": lr,
            "r": ranks[i % len(ranks)],
            "num_train_epochs": epochs_list[i % len(epochs_list)],
            "lr_scheduler_type": schedulers[i % len(schedulers)],
            "target_modules": modules[i % len(modules)],
            "lora_dropout": dropouts[i % len(dropouts)],
            "warmup_ratio": warmups[i % len(warmups)],
            "per_device_train_batch_size": batch_sizes[i % len(batch_sizes)],
            "gradient_accumulation_steps": grad_accums[i % len(grad_accums)],
        }
        coverage_configs.append(cfg)

    # One config per batch_size (ensure coverage if not already covered)
    for i, bs in enumerate(batch_sizes):
        cfg = {
            "learning_rate": lrs[(i + 2) % len(lrs)],
            "r": ranks[(i + 1) % len(ranks)],
            "num_train_epochs": epochs_list[(i + 1) % len(epochs_list)],
            "lr_scheduler_type": schedulers[(i + 1) % len(schedulers)],
            "target_modules": modules[(i + 1) % len(modules)],
            "lora_dropout": dropouts[(i + 1) % len(dropouts)],
            "warmup_ratio": warmups[(i + 1) % len(warmups)],
            "per_device_train_batch_size": bs,
            "gradient_accumulation_steps": grad_accums[(i + 1) % len(grad_accums)],
        }
        coverage_configs.append(cfg)

    # One config per target_modules combo
    for i, mod in enumerate(modules):
        cfg = {
            "learning_rate": lrs[(i + 3) % len(lrs)],
            "r": ranks[(i + 2) % len(ranks)],
            "num_train_epochs": epochs_list[(i + 2) % len(epochs_list)],
            "lr_scheduler_type": schedulers[(i + 2) % len(schedulers)],
            "target_modules": mod,
            "lora_dropout": dropouts[(i + 2) % len(dropouts)],
            "warmup_ratio": warmups[(i + 2) % len(warmups)],
            "per_device_train_batch_size": batch_sizes[(i + 2) % len(batch_sizes)],
            "gradient_accumulation_steps": grad_accums[(i + 2) % len(grad_accums)],
        }
        coverage_configs.append(cfg)

    # Phase 2: Random diverse configs to fill remaining budget
    # Use random sampling from the full space
    random.seed(42)  # reproducible
    max_random = max(0, max_iterations - len(coverage_configs))
    random_configs: list[dict] = []

    for _ in range(max_random * 3):  # oversample, then deduplicate
        cfg = {
            "learning_rate": random.choice(lrs),
            "r": random.choice(ranks),
            "num_train_epochs": random.choice(epochs_list),
            "lr_scheduler_type": random.choice(schedulers),
            "target_modules": random.choice(weighted_modules),
            "lora_dropout": random.choice(dropouts),
            "warmup_ratio": random.choice(warmups),
            "per_device_train_batch_size": random.choice(batch_sizes),
            "gradient_accumulation_steps": random.choice(grad_accums),
        }
        random_configs.append(cfg)

    # Deduplicate all configs
    def _cfg_key(c: dict) -> str:
        return "|".join([
            str(c["learning_rate"]),
            str(c["r"]),
            str(c["num_train_epochs"]),
            str(c["lr_scheduler_type"]),
            str(sorted(c["target_modules"])),
            str(c["per_device_train_batch_size"]),
            str(c["gradient_accumulation_steps"]),
            str(c["warmup_ratio"]),
            str(c["lora_dropout"]),
        ])

    seen = set()
    unique_configs: list[dict] = []
    for cfg in coverage_configs + random_configs:
        key = _cfg_key(cfg)
        if key not in seen:
            seen.add(key)
            unique_configs.append(cfg)

    # Trim to max_iterations
    table = unique_configs[:max_iterations]

    # Convert to the full format expected by the system
    result = []
    for i, cfg in enumerate(table):
        result.append({
            "config_id": i + 1,
            "status": "pending",
            "priority": i + 1,  # initial priority = order
            "result_metric": None,
            "result_loss": None,
            "lora_config": {
                "r": cfg["r"],
                "lora_alpha": cfg["r"] * 2,
                "lora_dropout": cfg["lora_dropout"],
                "target_modules": cfg["target_modules"],
                "task_type": "CAUSAL_LM",
                "bias": "none",
            },
            "training_args": {
                "learning_rate": cfg["learning_rate"],
                "num_train_epochs": cfg["num_train_epochs"],
                "lr_scheduler_type": cfg["lr_scheduler_type"],
                "per_device_train_batch_size": cfg["per_device_train_batch_size"],
                "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
                "warmup_ratio": cfg["warmup_ratio"],
                "logging_steps": 10,
                "save_strategy": "no",
                "optim": "adamw_torch",
                "remove_unused_columns": False,
                "report_to": "none",
            },
        })

    return result


def format_config_table_for_agent(
    config_table: list[dict],
    metric_name: str,
    top_n_pending: int = 10,
) -> str:
    """Format the config table for the LLM to pick from.

    Shows:
      1. Completed configs (with results, sorted by metric desc)
      2. Top N pending configs (by priority) for the LLM to choose from
    """
    completed = [c for c in config_table if c["status"] == "completed"]
    pending = [c for c in config_table if c["status"] == "pending"]

    completed_sorted = sorted(
        completed,
        key=lambda c: c.get("result_metric") or 0.0,
        reverse=True,
    )
    pending_sorted = sorted(pending, key=lambda c: c["priority"])

    lines: list[str] = []

    lines.append(f"=== CONFIG TABLE: {len(completed)} done, {len(pending)} remaining ===")
    lines.append("")

    # Completed configs
    if completed_sorted:
        lines.append(f"--- COMPLETED (sorted by {metric_name} desc) ---")
        for c in completed_sorted[:10]:
            lc = c["lora_config"]
            ta = c["training_args"]
            lines.append(
                f"  #{c['config_id']:>3} | {metric_name}={c.get('result_metric', 0):.4f} "
                f"loss={c.get('result_loss', 0):.4f} | "
                f"lr={ta['learning_rate']} r={lc['r']} ep={ta['num_train_epochs']} "
                f"sched={ta['lr_scheduler_type']} "
                f"bs={ta['per_device_train_batch_size']}×ga={ta['gradient_accumulation_steps']} "
                f"warmup={ta['warmup_ratio']} dropout={lc['lora_dropout']} "
                f"modules={lc['target_modules']}"
            )
        lines.append("")

    # Pending configs
    if pending_sorted:
        show = pending_sorted[:top_n_pending]
        lines.append(f"--- NEXT {len(show)} PENDING (by priority — pick one) ---")
        for c in show:
            lc = c["lora_config"]
            ta = c["training_args"]
            lines.append(
                f"  #{c['config_id']:>3} (pri={c['priority']}) | "
                f"lr={ta['learning_rate']} r={lc['r']} ep={ta['num_train_epochs']} "
                f"sched={ta['lr_scheduler_type']} "
                f"bs={ta['per_device_train_batch_size']}×ga={ta['gradient_accumulation_steps']} "
                f"warmup={ta['warmup_ratio']} dropout={lc['lora_dropout']} "
                f"modules={lc['target_modules']}"
            )
        lines.append("")
        lines.append(f"  ({len(pending) - len(show)} more pending configs not shown)")

    return "\n".join(lines)


def reprioritize_config_table(
    config_table: list[dict],
    history: list[dict],
    metric_name: str,
) -> list[dict]:
    """Re-rank pending configs based on Bayesian parameter insights.

    For each pending config, compute an "expected score" based on how well
    each of its parameter values has performed historically. Higher expected
    score = higher priority (lower number).
    """
    from collections import defaultdict

    if len(history) < 2:
        return config_table

    # Build per-parameter-value average metric
    param_keys = [
        ("learning_rate", lambda c: str(c["training_args"]["learning_rate"])),
        ("lora_r", lambda c: str(c["lora_config"]["r"])),
        ("num_train_epochs", lambda c: str(c["training_args"]["num_train_epochs"])),
        ("lr_scheduler_type", lambda c: str(c["training_args"]["lr_scheduler_type"])),
        ("target_modules", lambda c: str(sorted(c["lora_config"]["target_modules"]))),
        ("batch_size", lambda c: str(c["training_args"]["per_device_train_batch_size"])),
        ("gradient_accumulation_steps", lambda c: str(c["training_args"]["gradient_accumulation_steps"])),
    ]

    # Compute value->avg_metric from history
    value_avg: dict[str, dict[str, float]] = {}
    for param_key, _ in param_keys:
        metrics_by_val: dict[str, list[float]] = defaultdict(list)
        for run in history:
            val = str(run["params"].get(param_key, "?"))
            m = run["metrics"].get(metric_name, 0.0)
            metrics_by_val[val].append(m)
        value_avg[param_key] = {
            v: sum(ms) / len(ms) for v, ms in metrics_by_val.items()
        }

    # Score each pending config
    for cfg in config_table:
        if cfg["status"] != "pending":
            continue

        score = 0.0
        n_known = 0
        n_novel = 0
        for param_key, extractor in param_keys:
            val = extractor(cfg)
            if val in value_avg.get(param_key, {}):
                score += value_avg[param_key][val]
                n_known += 1
            else:
                # Bonus for novelty: untried values get a small exploration bonus
                n_novel += 1

        # Expected score = avg of known param scores + exploration bonus
        if n_known > 0:
            expected = score / n_known
        else:
            expected = 0.5  # no data at all: neutral

        # Exploration bonus: configs with more untried params get a bump
        exploration_bonus = n_novel * 0.05
        cfg["_expected_score"] = expected + exploration_bonus

    # Sort pending by expected score (desc) and assign priorities
    pending = [c for c in config_table if c["status"] == "pending"]
    pending.sort(key=lambda c: c.get("_expected_score", 0), reverse=True)

    for rank, cfg in enumerate(pending, 1):
        cfg["priority"] = rank
        cfg.pop("_expected_score", None)

    return config_table


def format_history_for_agent(history: list[dict], metric_name: str) -> str:
    """Format run history for the Claude agent with three ranked sections.

    Section 1 — TOP CONFIGS by eval metric (exploit these).
    Section 2 — LOWEST TRAIN LOSS configs (sometimes metric lags loss).
    Section 3 — Full chronological table with ALL parameters visible.
    """
    if not history:
        return "No previous runs found."

    def _row(run: dict) -> str:
        m = run["metrics"]
        p = run["params"]
        mods = p.get("target_modules", "?")
        if isinstance(mods, str) and mods.startswith("["):
            try:
                import ast
                mods = "+".join(ast.literal_eval(mods))
            except Exception:
                pass
        return (
            f"  iter={run['iteration']:>3} "
            f"{metric_name}={m.get(metric_name, 0.0):.4f} "
            f"loss={m.get('train_loss', 0.0):.4f} "
            f"lr={str(p.get('learning_rate', '?')):>10} "
            f"r={str(p.get('lora_r', '?')):>2} "
            f"ep={str(p.get('num_train_epochs', '?')):>1} "
            f"sched={str(p.get('lr_scheduler_type', '?')):>22} "
            f"bs={str(p.get('batch_size', '?')):>1} "
            f"modules=[{mods}] "
            f"{'★BEST' if run['is_improvement'] else ''}"
        )

    # Sort by metric descending → top performers
    by_metric = sorted(
        [r for r in history if r["metrics"].get(metric_name, 0.0) > 0],
        key=lambda r: r["metrics"].get(metric_name, 0.0),
        reverse=True,
    )

    # Sort by train_loss ascending → best optimisation signal
    by_loss = sorted(
        [r for r in history if r["metrics"].get("train_loss", 9999) < 9999],
        key=lambda r: r["metrics"].get("train_loss", 9999),
    )

    lines: list[str] = []

    # ── Section 1: Top configs by eval metric ────────────────────────────────
    lines.append(f"=== TOP {min(5, len(by_metric))} BY {metric_name.upper()} (exploit these) ===")
    if by_metric:
        for run in by_metric[:5]:
            lines.append(_row(run))
            lines.append(f"    hypothesis: {run['hypothesis'][:100]}")
            if run.get("diagnosis"):
                lines.append(f"    diagnosis: {run['diagnosis'][:150]}")
    else:
        lines.append("  (none yet)")
    lines.append("")

    # ── Section 2: Lowest train loss ─────────────────────────────────────────
    lines.append("=== TOP 3 BY LOWEST TRAIN LOSS (low loss = model is learning) ===")
    if by_loss:
        for run in by_loss[:3]:
            lines.append(_row(run))
            lines.append(f"    hypothesis: {run['hypothesis'][:100]}")
            if run.get("diagnosis"):
                lines.append(f"    diagnosis: {run['diagnosis'][:150]}")
    else:
        lines.append("  (none yet)")
    lines.append("")

    # ── Section 3: Full chronological table ───────────────────────────────────
    lines.append("=== ALL RUNS (chronological) ===")
    header = (
        f"  {'iter':>4} | {metric_name:>8} | {'loss':>6} | {'lr':>10} | "
        f"{'r':>2} | {'ep':>2} | {'scheduler':>22} | {'bs':>2} | modules"
    )
    lines.append(header)
    lines.append("  " + "-" * 100)
    for run in history:
        m = run["metrics"]
        p = run["params"]
        mods = p.get("target_modules", "?")
        if isinstance(mods, str) and mods.startswith("["):
            try:
                import ast
                mods = "+".join(ast.literal_eval(mods))
            except Exception:
                pass
        lines.append(
            f"  {run['iteration']:>4} | "
            f"{m.get(metric_name, 0.0):>8.4f} | "
            f"{m.get('train_loss', 0.0):>6.4f} | "
            f"{str(p.get('learning_rate', '?')):>10} | "
            f"{str(p.get('lora_r', '?')):>2} | "
            f"{str(p.get('num_train_epochs', '?')):>2} | "
            f"{str(p.get('lr_scheduler_type', '?')):>22} | "
            f"{str(p.get('batch_size', '?')):>2} | "
            f"[{mods}]"
            f"{'  ★' if run['is_improvement'] else ''}"
        )

    return "\n".join(lines)
