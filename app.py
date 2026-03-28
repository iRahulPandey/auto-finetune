"""
app.py — auto-finetune: two-tab Streamlit product.

Tab 1 — Train: configure, upload examples, run autonomous LoRA search.
Tab 2 — Inference Lab: browse all completed runs, pick any combination of
         fine-tuned adapters + base model, run side-by-side inference.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import streamlit as st

from _core.config import (
    RunConfig,
    SUPPORTED_MODELS,
    TASK_TYPES,
    PROJECT_ROOT,
    DEVICE,
    DTYPE,
)

# ── Load .env if present (runs before any env var checks) ────────────────────

def _load_dotenv(dotenv_path: Path) -> None:
    """Minimal .env loader — sets missing env vars from key=value lines."""
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

_load_dotenv(PROJECT_ROOT / ".env")

from _core.data_prep import prepare_data
from _core.program_md_generator import write_program_md
from _core.agent_loop import run_agent_loop, IterationResult
from _core.mlflow_utils import get_all_experiments
from _core.evaluator import _labels_match, _json_match, compute_json_field_accuracy
from _core import llm_client
from _core.llm_client import (
    LLMConfig, StageConfig, list_ollama_models, check_ollama_server,
)


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="auto-finetune",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# No custom CSS — use Streamlit defaults


# ── Session state init ────────────────────────────────────────────────────────

defaults = {
    "train_stage": "setup",   # setup | running | complete
    "iterations": [],
    "summary": None,
    "session_id": None,
    "data_info": None,
    "run_config": None,
    "examples": [],
    # Inference Lab
    "infer_models": {},       # adapter_path → (model, tokenizer) cache
    "infer_results": {},      # run_id → generated text
    "infer_results_list": [], # list of (label, text) tuples from last generation
    "infer_base_text": "",
    "infer_input": "",
    "infer_last_input": "",   # last input text for display
    "infer_expected": "",     # expected output from eval set (if applicable)
    "infer_batch_results": [],  # batch eval results
    "infer_generation_count": 0,  # incremented on every Generate click to bust widget key cache
    # Data augmentation (off by default — example datasets are pre-balanced)
    "enable_augmentation": False,
    # LLM provider per stage
    "ollama_url": "http://127.0.0.1:11434",
    "ollama_model": "",
    "provider_data_prep": "claude",
    "provider_agent": "claude",
    "provider_evaluator": "claude",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── MLflow tracking URI ───────────────────────────────────────────────────────
# Use MLFLOW_TRACKING_URI from .env if set, otherwise default to local mlruns.
# To use a remote server: set MLFLOW_TRACKING_URI=http://your-server:5000 in .env

_mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI") or f"file://{PROJECT_ROOT / 'mlruns'}"
os.environ["MLFLOW_TRACKING_URI"] = _mlflow_uri

# ── Apply per-stage LLM config ────────────────────────────────────────────────

def _build_stage_config(stage_key: str) -> StageConfig:
    """Build a StageConfig from session state for a given stage."""
    provider = st.session_state.get(f"provider_{stage_key}", "claude")
    return StageConfig(
        provider=provider,
        ollama_url=st.session_state.ollama_url,
        ollama_model=st.session_state.ollama_model,
    )


# Show wordmark + status badges in the sidebar
with st.sidebar:
    st.write("**auto-finetune**")
    st.divider()

    if _mlflow_uri.startswith("file://"):
        st.caption("MLflow: local file storage")
    else:
        st.caption(f"MLflow: [{_mlflow_uri}]({_mlflow_uri})")

    _badge_parts = []
    for sk, sl in [("data_prep", "Prep"), ("agent", "Agent"), ("evaluator", "Eval")]:
        prov = st.session_state.get(f"provider_{sk}", "claude")
        _badge_parts.append(f"{sl}: {'Ollama' if prov == 'ollama' else 'Claude'}")
    st.caption(" · ".join(_badge_parts))


# ── Example parsing helpers ───────────────────────────────────────────────────

def _is_valid_example(e: dict) -> bool:
    return "input" in e and "output" in e

def _parse_examples(text: str) -> list[dict]:
    text = text.strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [e for e in data if _is_valid_example(e)]
        except json.JSONDecodeError:
            pass
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
            if _is_valid_example(e):
                out.append(e)
        except json.JSONDecodeError:
            pass
    return out

_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


def _parse_file(f) -> list[dict]:
    import csv, io
    raw = f.read()
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise ValueError(f"File too large ({len(raw) // 1024} KB). Maximum is 10 MB.")
    content = raw.decode("utf-8")
    name = f.name.lower()
    if name.endswith(".json"):
        data = json.loads(content)
        if isinstance(data, list):
            return [e for e in data if _is_valid_example(e)]
    elif name.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(content))
        return [{"input": r["input"], "output": r["output"]}
                for r in reader if _is_valid_example(r)]
    return _parse_examples(content)


# ── Example datasets ──────────────────────────────────────────────────────────

EXAMPLE_DATASETS = [
    {
        "file": "email_urgency.json",
        "name": "Email Urgency Classifier",
        "task_type": "classification",
        "use_case": "Classify emails and messages as urgent or not_urgent based on content severity",
        "description": "200 diverse emails labeled as urgent (server outages, security breaches, data loss) or not_urgent (lunch orders, meeting notes, FYIs).",
        "badge": "Classification",
    },
    {
        "file": "invoice_extraction.json",
        "name": "Invoice Data Extraction",
        "task_type": "extraction",
        "use_case": "Extract structured fields (vendor, customer, total, line items, due date) from invoice text",
        "description": "200 diverse invoices (formal, compact, narrative, tabular formats) with structured JSON output containing vendor, customer, amounts, dates, and line items.",
        "badge": "Extraction",
    },
    {
        "file": "commit_message_generation.json",
        "name": "Commit Message Generator",
        "task_type": "generation",
        "use_case": "Generate conventional commit messages from code change descriptions",
        "description": "200 code change descriptions with conventional commit messages covering feat, fix, refactor, test, docs, perf, ci, and chore types.",
        "badge": "Generation",
    },
]

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_train, tab_infer, tab_lora = st.tabs(["Finetune", "Inference Lab", "LoRA Card"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — TRAIN
# ════════════════════════════════════════════════════════════════════════════════

with tab_train:

    # ── Sidebar config ─────────────────────────────────────────────────────────
    with st.sidebar:
        # ── Data source ────────────────────────────────────────────────────────
        st.subheader("Data")

        data_mode = st.radio(
            "data_mode",
            ["Example dataset", "My own data"],
            horizontal=True,
            key="_data_mode",
            label_visibility="collapsed",
        )

        if data_mode == "Example dataset":
            _example_names = ["— select a dataset —"] + [ds["name"] for ds in EXAMPLE_DATASETS]
            _chosen_idx = st.selectbox(
                "Dataset",
                range(len(_example_names)),
                format_func=lambda i: _example_names[i],
                key="_example_select",
                label_visibility="collapsed",
            )
            if _chosen_idx > 0:
                _ds = EXAMPLE_DATASETS[_chosen_idx - 1]
                # Only reload when selection changes, to not overwrite user edits
                if st.session_state.get("_prev_example_select") != _chosen_idx:
                    ds_path = PROJECT_ROOT / "examples" / _ds["file"]
                    try:
                        with open(ds_path, encoding="utf-8") as _fp:
                            st.session_state.examples = json.load(_fp)
                    except (FileNotFoundError, json.JSONDecodeError) as _e:
                        st.warning(f"Could not load {_ds['file']}: {_e}")
                    st.session_state["_use_case_input"] = _ds["use_case"]
                    if _ds["task_type"] in TASK_TYPES:
                        st.session_state["_task_type_select"] = _ds["task_type"]
                st.session_state["_prev_example_select"] = _chosen_idx
                st.caption(_ds["description"])
            else:
                st.session_state["_prev_example_select"] = 0
        else:
            # Switched to own data — clear any example-loaded data
            if st.session_state.get("_prev_data_mode") == "Example dataset":
                st.session_state.examples = []
                st.session_state["_prev_example_select"] = 0
        st.session_state["_prev_data_mode"] = data_mode

        st.divider()

        # ── Training config ─────────────────────────────────────────────────────
        st.subheader("Training Config")

        use_case = st.text_area(
            "Describe your task",
            placeholder="e.g. Classify support emails as urgent or not urgent",
            height=90,
            key="_use_case_input",
        )
        task_type = st.selectbox(
            "Task type",
            options=list(TASK_TYPES.keys()),
            format_func=lambda x: f"{x} — {TASK_TYPES[x]['description']}",
            key="_task_type_select",
        )
        model_key = st.selectbox(
            "Base model",
            options=list(SUPPORTED_MODELS.keys()),
            format_func=lambda x: SUPPORTED_MODELS[x]["short_name"],
            help="Qwen 0.5B recommended for MacBook Air. Fits in ~3 GB RAM.",
        )
        run_until_threshold = st.checkbox(
            "Run until threshold (no iteration limit)",
            value=False,
            help="Keep searching until the metric reaches the early-stop threshold. "
                 "Stop manually with Ctrl-C or by closing the app.",
        )
        max_iterations = st.slider(
            "Search iterations",
            min_value=3, max_value=100, value=10,
            disabled=run_until_threshold,
        )
        if run_until_threshold:
            st.caption("Iterations: **unlimited** — stops when threshold is reached.")
            max_iterations = 9999
        target_threshold = st.number_input(
            "Early-stop threshold",
            min_value=0.0, max_value=1.0,
            value=0.95 if task_type == "classification" else 0.85,
            step=0.05,
        )

        st.divider()

        # ── Augmentation ────────────────────────────────────────────────────────
        enable_aug = st.toggle(
            "Auto-balance classes (augmentation)",
            value=st.session_state.enable_augmentation,
            help="Off by default. Example datasets are already balanced.",
            key="_enable_augmentation_toggle",
        )
        st.session_state.enable_augmentation = enable_aug
        if enable_aug:
            st.caption(
                "Uses **Claude API** to generate synthetic examples when a "
                "class has fewer than 20 samples. Requires ANTHROPIC_API_KEY."
            )

        # ── LLM Providers ───────────────────────────────────────────────────────
        st.subheader("LLM Providers")

        ollama_url = st.text_input(
            "Ollama server URL",
            value=st.session_state.ollama_url,
            placeholder="http://127.0.0.1:11434",
            key="_ollama_url_input",
        )
        st.session_state.ollama_url = ollama_url

        # Cache Ollama check — re-run only when URL changes, not on every render
        if st.session_state.get("_ollama_url_cached") != ollama_url:
            _ollama_ok: bool = check_ollama_server(ollama_url)
            _ollama_models: list[str] = list_ollama_models(ollama_url) if _ollama_ok else []
            st.session_state["_ollama_ok_cached"] = _ollama_ok
            st.session_state["_ollama_models_cached"] = _ollama_models
            st.session_state["_ollama_url_cached"] = ollama_url
        else:
            _ollama_ok = st.session_state["_ollama_ok_cached"]
            _ollama_models = st.session_state["_ollama_models_cached"]

        if _ollama_ok and _ollama_models:
            prev_model = st.session_state.ollama_model
            default_idx = 0
            if prev_model in _ollama_models:
                default_idx = _ollama_models.index(prev_model)
            selected_model = st.selectbox(
                "Ollama model",
                options=_ollama_models,
                index=default_idx,
                key="_ollama_model_select",
            )
            st.session_state.ollama_model = selected_model
            st.success(f"Ollama: {len(_ollama_models)} model(s)")
        elif _ollama_ok:
            st.warning("Ollama running but no models. Run `ollama pull llama3.1`")
        else:
            st.caption(f"Ollama not reachable at {ollama_url}")

        _stage_info = {
            "data_prep": ("Data prep", "System prompt synthesis + augmentation (if enabled)"),
            "agent": ("HP search agent", "Picks hyperparameter configs each iteration"),
            "evaluator": ("Evaluator judge", "Semantic match scoring on eval set"),
        }

        for stage_key, (stage_label, stage_help) in _stage_info.items():
            ss_key = f"provider_{stage_key}"
            current = st.session_state.get(ss_key, "claude")
            provider = st.radio(
                stage_label,
                ["claude", "ollama"],
                format_func=lambda x: "Claude" if x == "claude" else "Ollama",
                index=0 if current == "claude" else 1,
                horizontal=True,
                help=stage_help,
                key=f"_provider_{stage_key}_radio",
            )
            st.session_state[ss_key] = provider

            if provider == "ollama" and not _ollama_ok:
                st.error("Start Ollama first: `ollama serve`")
            elif provider == "ollama" and not _ollama_models:
                st.warning("No Ollama models available")

        # Re-apply config after sidebar changes
        llm_client.configure(LLMConfig(
            data_prep=_build_stage_config("data_prep"),
            agent=_build_stage_config("agent"),
            evaluator=_build_stage_config("evaluator"),
        ))

        st.divider()

        # API key status
        _all_ollama = all(
            st.session_state.get(f"provider_{s}", "claude") == "ollama"
            for s in ["data_prep", "agent", "evaluator"]
        )
        _needs_api_key = not _all_ollama

        if os.environ.get("ANTHROPIC_API_KEY"):
            st.success("ANTHROPIC_API_KEY ✓")
        else:
            if _needs_api_key:
                st.error("Add your key to `.env`:\n```\nANTHROPIC_API_KEY=sk-ant-…\n```")
            else:
                st.info("Fully local mode — no API key needed")
        st.caption(f"Device: **{DEVICE.upper()}**")

    # ── Setup stage ────────────────────────────────────────────────────────────
    if st.session_state.train_stage == "setup":

        st.title("Finetune model")
        st.write("Describe your task, upload examples, and let the agent find the best LoRA configuration.")

        examples: list[dict] = list(st.session_state.get("examples", []))

        if data_mode == "My own data":
            st.subheader("Training Data")
            st.markdown(
                "Provide 20–200 input/output pairs as a JSON array or JSONL. "
                "Each object needs an `input` and `output` field."
            )
            tab_paste, tab_upload = st.tabs(["Paste", "Upload file"])

            with tab_paste:
                raw = st.text_area(
                    "Examples (JSON array or JSONL)", height=280,
                    placeholder='[\n  {"input": "...", "output": "..."},\n  ...\n]',
                )
                pasted = _parse_examples(raw) if raw else []
                if pasted:
                    st.success(f"Parsed **{len(pasted)}** examples")
                    examples = pasted
                    st.session_state.examples = examples

            with tab_upload:
                uploaded = st.file_uploader(
                    "Upload .json / .jsonl / .csv", type=["json", "jsonl", "csv"]
                )
                if uploaded:
                    file_examples = _parse_file(uploaded)
                    if file_examples:
                        st.success(f"Parsed **{len(file_examples)}** examples from {uploaded.name}")
                        examples = file_examples
                        st.session_state.examples = examples
                    else:
                        st.error("Could not parse — ensure 'input' and 'output' columns exist.")

        else:
            # Example dataset mode — data is already loaded via sidebar
            _chosen_idx = st.session_state.get("_example_select", 0)
            if _chosen_idx > 0 and examples:
                _ds = EXAMPLE_DATASETS[_chosen_idx - 1]
                st.subheader("Loaded Dataset")
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"**{_ds['name']}**")
                    st.caption(_ds["description"])
                with col_b:
                    st.metric("Examples", len(examples))
            elif _chosen_idx == 0:
                st.info("Select a dataset from the sidebar to get started, or switch to **My own data**.")

        if examples:
            with st.expander(f"Preview — {len(examples)} examples", expanded=False):
                for i, ex in enumerate(examples[:4]):
                    c1, c2 = st.columns(2)
                    c1.text_area("Input", ex["input"], height=70, key=f"pi{i}", disabled=True)
                    c2.text_area("Output", ex["output"], height=70, key=f"po{i}", disabled=True)
                if len(examples) > 4:
                    st.caption(f"… and {len(examples) - 4} more")

        st.divider()
        can_start = (
            len(examples) >= 2
            and use_case.strip()
            and os.environ.get("ANTHROPIC_API_KEY")
        )
        col_btn, col_warn = st.columns([1, 3])
        with col_btn:
            start = st.button(
                "Start Training", type="primary",
                disabled=not can_start, use_container_width=True,
            )
        with col_warn:
            if not use_case.strip():
                st.warning("Enter a task description in the sidebar.")
            elif len(examples) < 2:
                st.warning("Provide at least 2 examples (20+ recommended).")
            elif not os.environ.get("ANTHROPIC_API_KEY"):
                st.warning("Set `ANTHROPIC_API_KEY` in your environment.")
            elif len(examples) < 20:
                st.info(f"{len(examples)} examples loaded. 20+ recommended.")

        if start:
            rc = RunConfig(
                use_case=use_case,
                model_key=model_key,
                task_type=task_type,
                max_iterations=max_iterations,
                target_threshold=target_threshold,
            )
            errors = rc.validate()
            if errors:
                st.error(f"Config errors: {errors}")
            else:
                st.session_state.run_config = rc
                st.session_state.examples = examples
                st.session_state.iterations = []
                st.session_state.train_stage = "running"
                st.rerun()

    # ── Running stage ──────────────────────────────────────────────────────────
    elif st.session_state.train_stage == "running":
        rc = st.session_state.run_config
        examples = st.session_state.examples

        iter_label = "Unlimited" if rc.max_iterations >= 9999 else str(rc.max_iterations)
        st.title(rc.use_case[:70])
        st.write(f"{SUPPORTED_MODELS[rc.model_key]['short_name']} · {DEVICE.upper()} · {rc.metric_name} · {iter_label} iterations")

        # ── Live status banner ──────────────────────────────────────────────
        banner        = st.empty()
        iter_metrics  = st.empty()   # scoreboard updated after each iteration
        log_expander  = st.expander("Training log", expanded=True)
        log_area      = log_expander.empty()

        log_lines: list[str] = []

        def _append_log(line: str) -> None:
            log_lines.append(line)
            # Show last 35 lines so it feels like a live terminal
            log_area.code("\n".join(log_lines[-35:]), language=None)

        def on_iter(result: IterationResult) -> None:
            st.session_state.iterations.append(result)
            good = [r for r in st.session_state.iterations if not r.error]
            if good:
                best_val = max(r.metric_value for r in good)
                rows = "".join(
                    f"| {r.iteration} "
                    f"| {'★' if r.is_improvement else ' '} "
                    f"| {r.metric_value:.4f} "
                    f"| {r.training_args.get('learning_rate','')} "
                    f"| {r.lora_config.get('r','')} "
                    f"| {r.hypothesis[:55]} |\n"
                    for r in good
                )
                iter_metrics.markdown(
                    f"**Best {rc.metric_name}: {best_val:.4f}**  "
                    f"({len(good)} iteration{'s' if len(good)!=1 else ''} complete)\n\n"
                    f"| Iter | ★ | {rc.metric_name} | LR | Rank | Hypothesis |\n"
                    f"|------|---|--------|----|------|------------|\n"
                    + rows
                )

        # ── Free memory before training ──────────────────────────────────
        # Clear any cached inference models from the Inference Lab tab
        # and flush MPS/CUDA memory so the training subprocess gets a clean slate.
        import torch
        import gc
        if st.session_state.infer_models:
            _append_log("Clearing cached inference models to free memory...")
            st.session_state.infer_models = {}
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _aug_on = st.session_state.get("enable_augmentation", False)
        _prep_label = llm_client.get_stage_config(llm_client.STAGE_DATA_PREP).label
        _prep_msg = f"Preparing data and synthesizing system prompt via {_prep_label}"
        if _aug_on:
            _prep_msg += " (augmentation ON)…"
        else:
            _prep_msg += "…"
        banner.info(_prep_msg)
        try:
            data_info = prepare_data(
                use_case=rc.use_case,
                examples=examples,
                model_key=rc.model_key,
                enable_augmentation=_aug_on,
            )
            st.session_state.data_info = data_info
            st.session_state.session_id = data_info["session_id"]

            _append_log(
                f"Data ready — {data_info['train_count']} train / "
                f"{data_info['eval_count']} eval examples"
            )
            _append_log(
                f"System prompt synthesized. "
                f"MLflow experiment: {rc.use_case[:50]}"
            )
            banner.info(
                f"Data ready · {data_info['train_count']} train / "
                f"{data_info['eval_count']} eval · Search starting…"
            )
            write_program_md(rc)

            summary = run_agent_loop(
                run_config=rc,
                session_id=data_info["session_id"],
                on_iteration_complete=on_iter,
                on_training_output=_append_log,
                on_status=_append_log,
            )
            st.session_state.summary = summary
            st.session_state.train_stage = "complete"
            st.rerun()

        except Exception as e:
            import traceback
            _append_log(f"ERROR: {e}")
            _append_log(traceback.format_exc())
            banner.error(f"Training failed: {e}")
            st.session_state.train_stage = "setup"
            time.sleep(3)
            st.rerun()

    # ── Complete stage ─────────────────────────────────────────────────────────
    elif st.session_state.train_stage == "complete":
        summary = st.session_state.summary
        rc = st.session_state.run_config

        st.title("Training complete")
        st.caption(f"Session: {summary['session_id'][:12]}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"Best {summary['best_metric_name']}",
                    f"{summary['best_metric_value']:.4f}")
        col2.metric("Iterations", summary["total_iterations"])
        col3.metric("Improvements", summary["improvements"])
        col4.metric("Errors", summary["errors"])

        iters = [r for r in st.session_state.iterations if not r.error]
        if iters:
            import pandas as pd
            df = pd.DataFrame([{
                "Iter": r.iteration,
                summary["best_metric_name"]: round(r.metric_value, 4),
                "Loss": round(r.train_loss, 4),
                "Best": "★" if r.is_improvement else "",
                "LR": r.training_args.get("learning_rate", ""),
                "Rank": r.lora_config.get("r", ""),
                "Hypothesis": r.hypothesis[:90],
            } for r in iters])

            st.subheader(f"{summary['best_metric_name']} over iterations")
            st.line_chart(df.set_index("Iter")[summary["best_metric_name"]])

            st.subheader("All iterations")
            st.dataframe(df, width='stretch', hide_index=True)

        st.info(
            f"All adapters saved to `adapters/runs/{summary['session_id'][:12]}/`. "
            f"Switch to the **Inference Lab** tab to compare them."
        )

        if st.button("Start New Training Session"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — INFERENCE LAB
# ════════════════════════════════════════════════════════════════════════════════

with tab_infer:

    st.title("Inference Lab")
    st.write("Pick an experiment, select runs, and compare outputs side-by-side.")

    # ── Load all experiments from MLflow ───────────────────────────────────────

    @st.cache_data(ttl=10)
    def _load_experiments():
        try:
            return get_all_experiments()
        except Exception:
            return []

    experiments = _load_experiments()

    if not experiments:
        st.info(
            "No experiments found yet. "
            "Run a training session first, then come back here."
        )
        st.stop()

    # ── Experiment + session + run selector ────────────────────────────────────

    col_sidebar, col_main = st.columns([1, 2], gap="large")

    with col_sidebar:
        st.subheader("Experiment")

        def _exp_label(i: int) -> str:
            e = experiments[i]
            name = e["experiment_name"]
            if len(name) > 40:
                name = name[:40] + "…"
            return f"{name} ({e['total_runs']} runs, {e['total_sessions']} sessions)"

        selected_exp_idx = st.selectbox(
            "Experiment",
            range(len(experiments)),
            format_func=_exp_label,
        )

        experiment = experiments[selected_exp_idx]

        st.caption(
            f"Model: `{experiment['base_model_id'].split('/')[-1]}` · "
            f"Metric: `{experiment['metric_name']}` · "
            f"Best: **{experiment['best_metric']:.4f}** · "
            f"Total runs: {experiment['total_runs']}"
        )

        # If experiment has multiple sessions, let user filter by session
        exp_sessions = experiment["sessions"]
        if len(exp_sessions) > 1:
            session_options = ["All sessions"] + [
                f"Session {s['session_id'][:8]} ({s['run_count']} runs)"
                for s in exp_sessions
            ]
            selected_session_label = st.selectbox(
                "Filter by session",
                session_options,
                key="infer_session_filter",
            )
            if selected_session_label == "All sessions":
                runs = experiment["all_runs"]
                active_session = None
            else:
                idx = session_options.index(selected_session_label) - 1
                active_session = exp_sessions[idx]
                runs = active_session["runs"]
        else:
            active_session = exp_sessions[0] if exp_sessions else None
            runs = experiment["all_runs"]

        st.markdown("---")

        # Always offer base model as option
        include_base = st.checkbox("Include base model (no adapter)", value=True)

        # Run selection — up to 3 fine-tuned runs
        st.markdown("**Select up to 3 fine-tuned runs:**")
        selected_run_ids = []

        best_run_id = experiment["best_run_id"]
        metric_name = experiment["metric_name"]

        for run in sorted(runs, key=lambda r: r["iteration"]):
            m = run["metrics"].get(metric_name, 0.0)
            is_best = run["run_id"] == best_run_id
            adapter_exists = run["adapter_path"] and Path(run["adapter_path"]).exists()

            label_parts = [f"Iter {run['iteration']}  {metric_name}={m:.4f}"]
            if is_best:
                label_parts.append("★ best")
            if not adapter_exists:
                label_parts.append("⚠ no adapter")

            disabled = not adapter_exists or (
                len(selected_run_ids) >= 3
                and run["run_id"] not in selected_run_ids
            )
            checked = st.checkbox(
                "  ·  ".join(label_parts),
                key=f"sel_{run['run_id']}",
                disabled=disabled and run["run_id"] not in selected_run_ids,
                help=run["hypothesis"][:200],
            )
            if checked and adapter_exists:
                selected_run_ids.append(run["run_id"])

    # ── Inference panel ────────────────────────────────────────────────────────

    with col_main:

        n_cols = (1 if include_base else 0) + len(selected_run_ids)
        if n_cols == 0:
            st.info("Select the base model and/or at least one run on the left to start.")
            st.stop()

        # ── Load examples from eval data ──────────────────────────────────────
        @st.cache_data(ttl=60)
        def _load_eval_examples(session_ids: tuple) -> list[dict]:
            """Load examples from eval.jsonl files for the given sessions."""
            import json as _json
            examples = []
            seen_inputs = set()
            for sid in session_ids:
                eval_path = PROJECT_ROOT / "data" / sid / "eval.jsonl"
                if not eval_path.exists():
                    continue
                for line in eval_path.read_text(encoding="utf-8").strip().split("\n"):
                    if not line.strip():
                        continue
                    try:
                        row = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    msgs = row.get("messages", [])
                    user_msg = next((m["content"] for m in msgs if m["role"] == "user"), None)
                    expected = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
                    if user_msg and user_msg not in seen_inputs:
                        seen_inputs.add(user_msg)
                        examples.append({
                            "input": user_msg,
                            "expected": expected or "",
                        })
            return examples

        _session_ids = tuple(s["session_id"] for s in exp_sessions)
        eval_examples = _load_eval_examples(_session_ids)

        # ── Tabs: Single Test | Examples | Batch Eval ─────────────────────────
        infer_tab_single, infer_tab_examples, infer_tab_batch = st.tabs([
            "Single Test", "Examples from Dataset", "Batch Evaluation",
        ])

        # ── Shared helpers (defined once, used across tabs) ───────────────────
        base_model_id = experiment["base_model_id"]

        # Compute once — used in both Examples and Batch Eval tabs
        _eval_labels = {e["expected"] for e in eval_examples if e["expected"]}
        _is_classification = bool(_eval_labels) and all(len(l) < 50 for l in _eval_labels)

        # Load system prompt
        _system_prompt = ""
        _sp_sessions = [active_session] if active_session else exp_sessions
        for _sp_s in _sp_sessions:
            if not _sp_s:
                continue
            sp_path = PROJECT_ROOT / "data" / _sp_s["session_id"] / "system_prompt.txt"
            if sp_path.exists():
                _system_prompt = sp_path.read_text(encoding="utf-8").strip()
                break

        def _build_messages(user_text: str) -> list[dict]:
            msgs = []
            if _system_prompt:
                msgs.append({"role": "system", "content": _system_prompt})
            msgs.append({"role": "user", "content": user_text})
            return msgs

        def _get_model(adapter_path: Optional[str]):
            """Load base model once; subsequent adapters load only LoRA weights (~MB).

            Uses PEFT's multi-adapter API so the GB base model is read from disk
            exactly once per base_model_id, regardless of how many adapters are
            selected for comparison.

            Returns (model, tokenizer, active_adapter_name).
            active_adapter_name is None for base model inference.
            """
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            # Keyed by base_model_id so switching experiments uses the right weights
            shared_key = f"__shared_{base_model_id}__"
            cache = st.session_state.infer_models

            if shared_key not in cache:
                with st.spinner("Loading base model…"):
                    tok = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
                    if tok.pad_token is None:
                        tok.pad_token = tok.eos_token
                    extra_kwargs = {"device_map": "auto"} if DEVICE == "cuda" else {}
                    mdl = AutoModelForCausalLM.from_pretrained(
                        base_model_id, dtype=DTYPE, trust_remote_code=True, **extra_kwargs,
                    )
                    if DEVICE == "mps":
                        mdl = mdl.to("mps")
                    mdl.eval()
                cache[shared_key] = {"model": mdl, "tok": tok, "adapters": {}}

            entry = cache[shared_key]

            if not adapter_path:
                return entry["model"], entry["tok"], None

            if adapter_path not in entry["adapters"]:
                from peft import PeftModel
                adapter_name = f"run_{len(entry['adapters'])}"
                with st.spinner("Loading adapter weights…"):
                    if not entry["adapters"]:
                        # First adapter: wrap base model in PeftModel
                        peft = PeftModel.from_pretrained(
                            entry["model"], adapter_path, adapter_name=adapter_name,
                        )
                        peft.eval()
                        entry["model"] = peft
                    else:
                        # Subsequent adapters: only LoRA weights loaded, no base reload
                        entry["model"].load_adapter(adapter_path, adapter_name=adapter_name)
                entry["adapters"][adapter_path] = adapter_name

            adapter_name = entry["adapters"][adapter_path]
            entry["model"].set_adapter(adapter_name)
            return entry["model"], entry["tok"], adapter_name

        def _generate(model, tokenizer, messages_: list[dict], active_adapter: Optional[str] = None) -> str:
            import torch
            from peft import PeftModel
            _is_cls = metric_name in ("accuracy", "f1_macro", "f1_weighted")
            _max_tok = 30 if _is_cls else 256
            prompt_text = tokenizer.apply_chat_template(
                messages_, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(prompt_text, return_tensors="pt")
            device_obj = next(model.parameters()).device

            def _run() -> str:
                with torch.no_grad():
                    out = model.generate(
                        **inputs.to(device_obj),
                        max_new_tokens=_max_tok,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                return tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                ).strip()

            # Base model inference through a PeftModel: disable all LoRA layers
            if isinstance(model, PeftModel) and active_adapter is None:
                with model.disable_adapter():
                    return _run()
            return _run()

        def _clean_prediction(text: str) -> str:
            """Strip markdown code fences and whitespace from model output."""
            import re as _re
            t = text.strip()
            # Remove ```json ... ``` wrappers
            t = _re.sub(r'^```(?:json)?\s*\n?', '', t)
            t = _re.sub(r'\n?```\s*$', '', t)
            return t.strip()

        def _check_match(prediction: str, expected: str) -> bool:
            """JSON-aware comparison: tries JSON match, then normalized text."""
            pred = _clean_prediction(prediction)
            exp = expected.strip()
            # Try full JSON match first
            if _json_match(pred, exp):
                return True
            # Try JSON field accuracy >= 0.8 (partial credit)
            if exp.lstrip().startswith("{") or exp.lstrip().startswith("["):
                score = compute_json_field_accuracy([pred], [exp])
                if score >= 0.8:
                    return True
            # Fall back to _labels_match (normalized text, label codes)
            return _labels_match(pred, exp)

        def _build_col_specs() -> list[tuple]:
            specs = []
            if include_base:
                specs.append(("Base model (no adapter)", None))
            for rid in selected_run_ids:
                run_obj = next((r for r in runs if r["run_id"] == rid), None)
                if run_obj:
                    m = run_obj["metrics"].get(metric_name, 0.0)
                    label = (
                        f"Iter {run_obj['iteration']}  "
                        f"{metric_name}={m:.4f}"
                        + (" ★" if run_obj["run_id"] == best_run_id else "")
                    )
                    specs.append((label, run_obj["adapter_path"]))
            return specs

        def _run_single_inference(user_text: str, expected: str = "") -> None:
            """Run inference on a single input across all selected models."""
            st.session_state.infer_generation_count += 1
            st.session_state.infer_results_list = []

            col_specs = _build_col_specs()
            results_list = []
            n_models = len(col_specs)
            with st.spinner(f"Generating from {n_models} model{'s' if n_models != 1 else ''}…"):
                messages = _build_messages(user_text)
                for label, adapter_path in col_specs:
                    model, tokenizer, active_adapter = _get_model(adapter_path)
                    text = _generate(model, tokenizer, messages, active_adapter)
                    results_list.append((label, text))

            st.session_state.infer_results_list = results_list
            st.session_state.infer_last_input = user_text
            st.session_state.infer_expected = expected

        # ══════════════════════════════════════════════════════════════════════
        #  TAB 1: Single Test (free-form input)
        # ══════════════════════════════════════════════════════════════════════
        with infer_tab_single:
            st.subheader("Test Input")
            user_input = st.text_area(
                "Enter something to test",
                height=100,
                placeholder="Try an example from your dataset…",
                key="infer_input_box",
            )

            generate_btn = st.button(
                "Generate",
                type="primary",
                disabled=not user_input.strip(),
                key="single_gen_btn",
            )

            if generate_btn and user_input.strip():
                _run_single_inference(user_input)

        # ══════════════════════════════════════════════════════════════════════
        #  TAB 2: Examples from Dataset
        # ══════════════════════════════════════════════════════════════════════
        with infer_tab_examples:
            st.subheader("Examples from Dataset")
            if not eval_examples:
                st.info("No evaluation examples found for this experiment.")
            else:
                st.caption(
                    f"{len(eval_examples)} examples available from the eval set. "
                    "Click any example to run it through your selected models."
                )

                # Group by expected label if labels are short (classification task)
                if _is_classification:
                    # Show grouped by class
                    for label_val in sorted(_eval_labels):
                        class_examples = [e for e in eval_examples if e["expected"] == label_val]
                        with st.expander(
                            f"**{label_val}** ({len(class_examples)} examples)",
                            expanded=False,
                        ):
                            for idx, ex in enumerate(class_examples[:5]):
                                preview = ex["input"][:120] + ("…" if len(ex["input"]) > 120 else "")
                                if st.button(
                                    preview,
                                    key=f"ex_{label_val}_{idx}",
                                    use_container_width=True,
                                ):
                                    _run_single_inference(ex["input"], expected=ex["expected"])
                            if len(class_examples) > 5:
                                st.caption(f"… and {len(class_examples) - 5} more")
                else:
                    # Show flat list
                    for idx, ex in enumerate(eval_examples[:15]):
                        preview = ex["input"][:120] + ("…" if len(ex["input"]) > 120 else "")
                        expected_tag = f"  [expected: {ex['expected'][:40]}]" if ex["expected"] else ""
                        if st.button(
                            f"{preview}{expected_tag}",
                            key=f"ex_flat_{idx}",
                            use_container_width=True,
                        ):
                            _run_single_inference(ex["input"], expected=ex["expected"])
                    if len(eval_examples) > 15:
                        st.caption(f"Showing 15 of {len(eval_examples)} examples.")

        # ══════════════════════════════════════════════════════════════════════
        #  TAB 3: Batch Evaluation
        # ══════════════════════════════════════════════════════════════════════
        with infer_tab_batch:
            st.subheader("Batch Evaluation")
            if not eval_examples:
                st.info("No evaluation examples found for this experiment.")
            else:
                st.caption(
                    "Run all selected models on multiple examples and compare accuracy."
                )
                n_batch = st.slider(
                    "Number of examples",
                    min_value=5,
                    max_value=min(50, len(eval_examples)),
                    value=min(10, len(eval_examples)),
                    key="batch_n_slider",
                )
                batch_btn = st.button(
                    f"Run batch eval on {n_batch} examples",
                    type="primary",
                    key="batch_eval_btn",
                )

                if batch_btn:
                    import random as _rnd
                    import pandas as pd

                    col_specs = _build_col_specs()
                    if not col_specs:
                        st.warning("Select at least one model.")
                    else:
                        # Sample examples (stratified if classification)
                        if _is_classification and len(_eval_labels) > 1:
                            # Stratified sample
                            per_class = max(1, n_batch // len(_eval_labels))
                            sampled = []
                            for lv in sorted(_eval_labels):
                                cls_exs = [e for e in eval_examples if e["expected"] == lv]
                                sampled.extend(_rnd.sample(cls_exs, min(per_class, len(cls_exs))))
                            _rnd.shuffle(sampled)
                            sampled = sampled[:n_batch]
                        else:
                            sampled = _rnd.sample(eval_examples, min(n_batch, len(eval_examples)))

                        progress = st.progress(0, text="Running batch evaluation…")
                        batch_rows = []
                        model_correct = {label: 0 for label, _ in col_specs}
                        total = len(sampled)

                        for ex_i, ex in enumerate(sampled):
                            messages = _build_messages(ex["input"])
                            row = {
                                "Input": ex["input"][:80] + ("…" if len(ex["input"]) > 80 else ""),
                                "Expected": ex["expected"],
                            }
                            for label, adapter_path in col_specs:
                                model, tokenizer, active_adapter = _get_model(adapter_path)
                                pred = _generate(model, tokenizer, messages, active_adapter)
                                # Smart comparison: JSON-aware with partial credit
                                is_correct = _check_match(pred, ex["expected"])
                                if is_correct:
                                    model_correct[label] += 1
                                row[label] = pred + (" ✓" if is_correct else " ✗")
                            batch_rows.append(row)
                            progress.progress((ex_i + 1) / total, text=f"Example {ex_i + 1}/{total}")

                        progress.empty()

                        # Accuracy summary
                        st.markdown("#### Accuracy Summary")
                        acc_cols = st.columns(len(col_specs))
                        for i, (label, _) in enumerate(col_specs):
                            acc = model_correct[label] / total if total else 0
                            with acc_cols[i]:
                                st.metric(label[:30], f"{acc:.0%}", f"{model_correct[label]}/{total}")

                        # Detailed results table
                        st.markdown("#### Detailed Results")
                        st.dataframe(
                            pd.DataFrame(batch_rows),
                            width='stretch',
                            hide_index=True,
                        )

                        st.session_state.infer_batch_results = batch_rows

        # ── Display single inference results (shared across Single & Examples tabs) ──
        if st.session_state.get("infer_results_list"):
            st.markdown("---")
            st.subheader("Output")

            if st.session_state.get("infer_last_input"):
                input_preview = st.session_state.infer_last_input
                if len(input_preview) > 200:
                    input_preview = input_preview[:200] + "…"
                st.markdown(f"> **Input:** {input_preview}")

            expected = st.session_state.get("infer_expected", "")
            if expected:
                st.markdown(f"> **Expected:** `{expected}`")

            results_list = st.session_state.infer_results_list
            gen = st.session_state.infer_generation_count
            cols = st.columns(len(results_list))
            for i, (col, (label, text)) in enumerate(zip(cols, results_list)):
                with col:
                    st.markdown(f"**{label}**")
                    # Check correctness if we have expected
                    if expected:
                        if _check_match(text, expected):
                            st.success("Correct ✓")
                        else:
                            st.error("Incorrect ✗")
                    st.text_area(
                        "Output",
                        text,
                        height=200,
                        disabled=True,
                        key=f"out_{gen}_{i}",
                        label_visibility="collapsed",
                    )

        # ── Run details expander ───────────────────────────────────────────────
        if selected_run_ids:
            with st.expander("Selected run details"):
                import pandas as pd
                detail_rows = []
                for rid in selected_run_ids:
                    run_obj = next((r for r in runs if r["run_id"] == rid), None)
                    if run_obj:
                        p = run_obj["params"]
                        m = run_obj["metrics"]
                        detail_rows.append({
                            "Iter": run_obj["iteration"],
                            metric_name: round(m.get(metric_name, 0), 4),
                            "Loss": round(m.get("train_loss", 0), 4),
                            "LR": p.get("learning_rate", ""),
                            "Rank": p.get("lora_r", ""),
                            "Epochs": p.get("num_train_epochs", ""),
                            "Scheduler": p.get("lr_scheduler_type", ""),
                            "Hypothesis": run_obj["hypothesis"][:100],
                        })
                if detail_rows:
                    st.dataframe(pd.DataFrame(detail_rows),
                                 width='stretch', hide_index=True)

        # ── Clear model cache ──────────────────────────────────────────────────
        if st.session_state.infer_models:
            cached_count = len(st.session_state.infer_models)
            if st.button(f"Clear model cache ({cached_count} loaded)", type="secondary"):
                import torch
                st.session_state.infer_models = {}
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                st.success("Cache cleared.")
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — LORA CARD
# ════════════════════════════════════════════════════════════════════════════════

with tab_lora:

    st.title("LoRA Card")
    st.write("Select any completed run to inspect its full configuration and system prompt.")

    try:
        _lora_experiments = _load_experiments()
    except Exception:
        _lora_experiments = []

    if not _lora_experiments:
        st.info("No experiments found yet. Run a training session first, then come back here.")
        st.stop()

    lora_col_sel, lora_col_card = st.columns([1, 2], gap="large")

    with lora_col_sel:
        st.subheader("Select run")

        def _lora_exp_label(i: int) -> str:
            e = _lora_experiments[i]
            name = e["experiment_name"]
            return f"{name[:40]} ({e['total_runs']} runs)"

        lora_exp_idx = st.selectbox(
            "Experiment",
            range(len(_lora_experiments)),
            format_func=_lora_exp_label,
            key="lora_exp_idx",
        )
        lora_exp = _lora_experiments[lora_exp_idx]

        lora_sessions = lora_exp["sessions"]
        if len(lora_sessions) > 1:
            _lora_sess_opts = ["All sessions"] + [
                f"Session {s['session_id'][:8]} ({s['run_count']} runs)"
                for s in lora_sessions
            ]
            lora_sess_sel = st.selectbox(
                "Session",
                _lora_sess_opts,
                key="lora_sess_sel",
            )
            if lora_sess_sel == "All sessions":
                lora_runs = lora_exp["all_runs"]
                _lora_active_sess = None
            else:
                _si = _lora_sess_opts.index(lora_sess_sel) - 1
                _lora_active_sess = lora_sessions[_si]
                lora_runs = _lora_active_sess["runs"]
        else:
            _lora_active_sess = lora_sessions[0] if lora_sessions else None
            lora_runs = lora_exp["all_runs"]

        _lora_metric = lora_exp["metric_name"]
        _lora_best_id = lora_exp["best_run_id"]

        def _lora_run_label(r: dict) -> str:
            m = r["metrics"].get(_lora_metric, 0.0)
            star = " ★" if r["run_id"] == _lora_best_id else ""
            return f"Iter {r['iteration']}  {_lora_metric}={m:.4f}{star}"

        lora_runs_sorted = sorted(lora_runs, key=lambda r: r["iteration"])
        # Default to best run
        _default_run_idx = next(
            (i for i, r in enumerate(lora_runs_sorted) if r["run_id"] == _lora_best_id),
            0,
        )
        lora_run_idx = st.selectbox(
            "Run",
            range(len(lora_runs_sorted)),
            format_func=lambda i: _lora_run_label(lora_runs_sorted[i]),
            index=_default_run_idx,
            key="lora_run_idx",
        )
        lora_run = lora_runs_sorted[lora_run_idx]

    with lora_col_card:

        import datetime as _dt

        p = lora_run["params"]
        m = lora_run["metrics"]
        _lora_metric_val = m.get(_lora_metric, 0.0)
        _is_best_run = lora_run["run_id"] == _lora_best_id
        _ts = lora_run.get("timestamp", 0)
        _ts_str = (
            _dt.datetime.fromtimestamp(_ts).strftime("%Y-%m-%d %H:%M")
            if _ts else "—"
        )

        # ── Header ─────────────────────────────────────────────────────────────
        _star = " ★ Best run" if _is_best_run else ""
        st.subheader(f"Iteration {lora_run['iteration']}{_star}")
        st.caption(
            f"Run ID: `{lora_run['run_id'][:12]}…`  ·  "
            f"Session: `{lora_run['session_id'][:8]}…`  ·  "
            f"Logged: {_ts_str}"
        )

        # ── Key metrics strip ──────────────────────────────────────────────────
        _mc1, _mc2, _mc3, _mc4 = st.columns(4)
        _mc1.metric(_lora_metric, f"{_lora_metric_val:.4f}")
        _mc2.metric("Train loss", f"{m.get('train_loss', 0):.4f}")
        _mc3.metric("Epochs", p.get("num_train_epochs", "—"))
        _mc4.metric("LR", p.get("learning_rate", "—"))

        st.divider()

        # ── Task info ──────────────────────────────────────────────────────────
        st.markdown("**Task**")
        st.write(lora_run.get("use_case") or "—")
        st.caption(f"Base model: `{lora_run.get('base_model_id', '—')}`")

        st.divider()

        # ── LoRA config ────────────────────────────────────────────────────────
        st.markdown("**LoRA configuration**")
        _lc1, _lc2, _lc3 = st.columns(3)
        _lc1.metric("Rank (r)", p.get("lora_r", "—"))
        _lc2.metric("Alpha", p.get("lora_alpha", "—"))
        _lc3.metric("Dropout", p.get("lora_dropout", "—"))
        st.caption(f"Target modules: `{p.get('target_modules', '—')}`")

        st.divider()

        # ── Training args ──────────────────────────────────────────────────────
        st.markdown("**Training arguments**")
        _ta1, _ta2, _ta3, _ta4 = st.columns(4)
        _ta1.metric("Scheduler", p.get("lr_scheduler_type", "—"))
        _ta2.metric("Batch size", p.get("batch_size", "—"))
        _ta3.metric("Grad accum", p.get("gradient_accumulation_steps", "—"))
        _ta4.metric("Warmup ratio", p.get("warmup_ratio", "—"))

        st.divider()

        # ── Hypothesis ─────────────────────────────────────────────────────────
        st.markdown("**Hypothesis**")
        st.write(lora_run.get("hypothesis") or "—")

        # ── Failure diagnosis (only shown when present) ────────────────────────
        _diag = lora_run.get("diagnosis", "")
        if _diag:
            st.divider()
            st.markdown("**Failure diagnosis**")
            st.warning(_diag)

        st.divider()

        # ── System prompt ──────────────────────────────────────────────────────
        _lora_sp = ""
        _sp_sid = lora_run.get("session_id", "")
        if _sp_sid:
            _sp_path = PROJECT_ROOT / "data" / _sp_sid / "system_prompt.txt"
            if _sp_path.exists():
                _lora_sp = _sp_path.read_text(encoding="utf-8").strip()

        with st.expander("System prompt", expanded=False):
            if _lora_sp:
                st.text(_lora_sp)
            else:
                st.caption("System prompt not found on disk.")

        # ── Adapter path ───────────────────────────────────────────────────────
        _adapter = lora_run.get("adapter_path", "")
        with st.expander("Adapter path", expanded=False):
            if _adapter and Path(_adapter).exists():
                st.code(_adapter)
            elif _adapter:
                st.caption(f"Path recorded but not found on disk: `{_adapter}`")
            else:
                st.caption("No adapter path recorded.")

        # ── Export as JSON ─────────────────────────────────────────────────────
        _export = {
            "run_id": lora_run["run_id"],
            "session_id": lora_run["session_id"],
            "iteration": lora_run["iteration"],
            "use_case": lora_run.get("use_case", ""),
            "base_model_id": lora_run.get("base_model_id", ""),
            "hypothesis": lora_run.get("hypothesis", ""),
            "metrics": {_lora_metric: _lora_metric_val, "train_loss": m.get("train_loss", 0)},
            "lora_config": {
                "r": p.get("lora_r"),
                "lora_alpha": p.get("lora_alpha"),
                "lora_dropout": p.get("lora_dropout"),
                "target_modules": p.get("target_modules"),
            },
            "training_args": {
                "learning_rate": p.get("learning_rate"),
                "num_train_epochs": p.get("num_train_epochs"),
                "lr_scheduler_type": p.get("lr_scheduler_type"),
                "per_device_train_batch_size": p.get("batch_size"),
                "gradient_accumulation_steps": p.get("gradient_accumulation_steps"),
                "warmup_ratio": p.get("warmup_ratio"),
            },
            "system_prompt": _lora_sp,
            "adapter_path": _adapter,
            "diagnosis": _diag,
            "timestamp": _ts_str,
        }
        st.download_button(
            label="Export as JSON",
            data=json.dumps(_export, indent=2),
            file_name=f"lora_card_iter{lora_run['iteration']}_{lora_run['run_id'][:8]}.json",
            mime="application/json",
        )
