# CLAUDE.md — auto-finetune

## What this project is

UI-first autonomous LoRA fine-tuning. Inspired by Karpathy's autoresearch, but instead of hand-writing program.md, the user describes their use case in a Streamlit UI and the system generates everything: system prompt, program.md, search space, stopping criteria. An LLM agent then proposes hyperparameter experiments, trains LoRA adapters, evaluates them, and iterates — all tracked in MLflow.

## Architecture

```
app.py                → Streamlit UI (two tabs: Train + Inference Lab)
finetune.py           → ONLY file the agent edits (LoRA config + training loop)
_core/config.py       → Constants, search spaces, device detection
_core/data_prep.py    → Data validation, system prompt synthesis, train/eval split
_core/llm_client.py   → Unified LLM provider (Claude/Ollama per stage)
_core/agent_loop.py   → Autonomous search loop (reads program.md, edits finetune.py)
_core/evaluator.py    → Frozen eval metrics (agent NEVER modifies this)
_core/mlflow_utils.py → MLflow experiment tracking, model registry
_core/program_md_generator.py → Auto-generates program.md from RunConfig
test_e2e.py           → End-to-end stage validation
```

## Critical rules

1. **finetune.py is the only agent-editable file.** The agent loop patches the LORA_CONFIG and TRAINING_ARGS dicts between iterations. Everything else is fixed infrastructure.

2. **evaluator.py is never modified by the agent.** It defines the scoring contract. If you change metrics, do it manually and re-run all experiments for fair comparison.

3. **program.md is auto-generated.** Created by `program_md_generator.py` from RunConfig. Never edit manually — it gets overwritten each session.

4. **The eval set is frozen.** Created once during data_prep. Every iteration scores against the same examples. Don't mutate `data/<session>/eval.jsonl` mid-session.

## Task types and metrics

- **classification**: accuracy, f1_macro, f1_weighted. `max_new_tokens=30`.
- **extraction**: json_field_accuracy (primary), f1_token, exact_match. `max_new_tokens=256`. Outputs are JSON objects — use `_json_match()` or `compute_json_field_accuracy()` for comparison, never raw string equality.
- **generation**: rouge_l (primary), bleu. `max_new_tokens=256`.

## LLM provider system

Three pipeline stages, each independently configurable:

| Stage | What it does | Config key |
|-------|-------------|------------|
| data_prep | System prompt synthesis, augmentation | `provider_data_prep` |
| agent | Hypothesis generation, finetune.py editing | `provider_agent` |
| evaluator | LLM-as-judge for semantic matching | `provider_evaluator` |

All LLM calls go through `_core/llm_client.py`. Call `llm_client.generate(prompt, stage=STAGE_*)`. Never import anthropic or call Ollama directly.

## Data flow

```
User uploads JSON → data_prep.py validates + splits → data/<session>/train.jsonl + eval.jsonl
                  → program_md_generator.py → program.md
                  → agent_loop.py reads program.md, patches finetune.py
                  → subprocess runs finetune.py --config run_config.json
                  → evaluator scores on eval.jsonl
                  → mlflow logs params + metrics + adapter path
                  → agent sees results, proposes next hypothesis
```

## Key patterns

### Comparison logic for JSON outputs
Always use `_labels_match()` or `_check_match()` — never `pred == expected`. For extraction tasks, JSON may differ in whitespace, key order, or minor field values. Use `json_field_accuracy >= 0.8` for partial credit.

### Subprocess training
`agent_loop.py` runs `finetune.py` as a subprocess, passing config via a temp JSON file. The `run_config.json` includes `task_type` so finetune.py can set appropriate `max_new_tokens`.

### Session state (Streamlit)
Provider configs, augmentation toggle, Ollama URL/model are stored in `st.session_state`. The sidebar builds `StageConfig` objects via `_build_stage_config()` and applies them with `llm_client.configure()`.

`llm_client.configure()` is called once inside the sidebar block on every render. Do NOT add a second call at module level — it is redundant.

### UI rules
- No custom CSS injected via `st.markdown(..., unsafe_allow_html=True)`. Use plain Streamlit primitives only (`st.title`, `st.subheader`, `st.write`, `st.caption`, `st.divider`). Custom CSS breaks Streamlit's built-in icon fonts (Material Symbols), causing expander arrows to render as raw text.
- Tabs: "Finetune" (train tab) and "Inference Lab".
- The app uses `layout="wide"` and `initial_sidebar_state="expanded"`.

### Shared computed values in Inference Lab
`_eval_labels` and `_is_classification` are computed once at the `col_main` scope and reused across all three inference sub-tabs (Single Test, Examples, Batch Eval). Do not recompute them inside individual tabs.

## File size guidelines

- Functions: < 50 lines
- Files: 200-400 typical, 800 max
- Nesting: < 4 levels

## Common gotchas

- `max_new_tokens` must be task-aware: 30 for classification, 256 for extraction/generation. Hardcoding a single value truncates JSON outputs or wastes tokens on labels.
- Stratified train/eval split fails for generation/extraction tasks (outputs are unique). Detect >50% unique outputs → fall back to random split.
- `_normalize_text()` strips all punctuation — it destroys JSON structure. Always try `_json_match()` before falling back to normalized text comparison.
- MPS (Apple Silicon) needs `float16`, not `bfloat16`. CUDA uses `bfloat16`. CPU uses `float32`. All handled in `config.py`.
- Data augmentation is off by default. Only enable when classes are imbalanced. Requires Claude API.

## Dependencies

torch, transformers, peft, trl, accelerate, mlflow, anthropic, streamlit, pandas, requests. Managed via `pyproject.toml` + uv.

## Running

```bash
cp .env.example .env    # add ANTHROPIC_API_KEY
uv sync
streamlit run app.py
```

## Testing

```bash
uv run python test_e2e.py
```

Validates each stage independently: data prep → program.md generation → finetune dry-run → evaluator → MLflow logging.
