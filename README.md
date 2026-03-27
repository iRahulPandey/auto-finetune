# auto-finetune

Autonomous LoRA fine-tuning with an LLM agent running hyperparameter search. You bring examples, the agent does the rest.

Inspired by Karpathy's [autoresearch](https://github.com/karpathy/autoresearch). Same philosophy — let the agent loop over hypotheses, train, evaluate, iterate — but applied to fine-tuning instead of research. The key difference: **autoresearch asks you to write `program.md` yourself. auto-finetune generates it from a UI.** You describe your use case in plain English, upload data, and the system synthesizes the constraints, search space, and evaluation plan automatically. No YAML, no config files, no prompt engineering. Just a Streamlit app.

This matters because most people who want to fine-tune a model aren't specific about hyperparameters — they know what they want the model to do, not what learning rate to use. The UI captures intent ("classify email urgency" or "extract invoice fields") and translates it into a complete training program.

## How it works

```
 ┌───────────────────────────────────┐
 │  Streamlit UI                     │
 │  Describe use case + upload data  │
 └──────────────┬────────────────────┘
                ▼
 ┌──────────────────────────────────┐
 │  Data prep                       │
 │  Validate → synthesize system    │
 │  prompt → split train/eval       │
 └──────────────┬───────────────────┘
                ▼
 ┌──────────────────────────────────┐
 │  Auto-generate program.md        │
 │  (search space, constraints,     │
 │   stopping criteria — from UI)   │
 └──────────────┬───────────────────┘
                ▼
 ┌──────────────────────────────────┐
 │  Agent loop                      │
 │  Read program.md + prior results │
 │  → propose hypothesis            │
 │  → edit finetune.py              │
 │  → train LoRA adapter            │
 │  → evaluate on frozen eval set   │
 │  → log to MLflow                 │
 │  → repeat                        │
 └──────────────┬───────────────────┘
                ▼
 ┌──────────────────────────────────┐
 │  Inference Lab                   │
 │  Compare base vs fine-tuned      │
 │  models side-by-side             │
 └──────────────────────────────────┘
```

The agent can only edit `finetune.py`. Everything else is fixed infrastructure.

## auto-finetune vs autoresearch

| | autoresearch | auto-finetune |
|---|---|---|
| **Input** | You write `program.md` by hand | UI generates `program.md` from your description + data |
| **Domain** | Open-ended research | LoRA fine-tuning |
| **User** | Researchers who know what they want | Anyone with input/output examples |
| **Loop** | Agent writes code, runs experiments | Agent tweaks LoRA config, trains, evaluates |
| **Tracking** | Git commits | MLflow experiments |
| **Output** | Research artifacts | Trained LoRA adapters ready for inference |

Both share the same core idea: give an agent a tight sandbox (one editable file), a scoring function, and a loop. Let it explore.

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/user/auto-finetune.git
cd auto-finetune

cp .env.example .env          # add your ANTHROPIC_API_KEY
uv sync                       # install deps
streamlit run app.py          # launch
```

Open the UI, describe your use case, upload examples, click train. That's it.

### Fully local mode

If you have [Ollama](https://ollama.com) running, you can route any pipeline stage through local models instead of Claude — zero API cost.

```bash
ollama serve
ollama pull llama3.2
```

Toggle each stage (data prep, HP search agent, evaluator) between Claude and Ollama in the sidebar. Mix and match or go fully local.

## Project structure

```
auto-finetune/
├── app.py                  # Streamlit UI (train tab + inference lab)
├── finetune.py             # The experiment surface — ONLY file the agent edits
├── test_e2e.py             # End-to-end stage validation
├── _core/
│   ├── config.py           # Constants, search spaces, device detection
│   ├── data_prep.py        # Validation, system prompt synthesis, train/eval split
│   ├── llm_client.py       # Unified Claude/Ollama provider per stage
│   ├── agent_loop.py       # Autonomous search loop
│   ├── evaluator.py        # Frozen metrics (agent never touches this)
│   ├── mlflow_utils.py     # Experiment tracking, model registry
│   └── program_md_generator.py  # Auto-generates program.md from UI config
├── examples/               # Sample datasets to try immediately
├── pyproject.toml          # Dependencies
└── .env.example            # API key template
```

Runtime artifacts (git-ignored): `data/`, `adapters/`, `mlruns/`, `program.md`

## Task types

| Type | Example | Primary metric | Output |
|------|---------|---------------|--------|
| Classification | Email urgency (low/medium/high) | accuracy | Short label |
| Extraction | Invoice fields from text | json_field_accuracy | JSON object |
| Generation | Commit message from diff | rouge_l | Free text |

Auto-detected from your data. The system picks appropriate metrics, token limits, and evaluation strategies.

## Design decisions

**UI-first.** The user never writes config files. The UI captures intent (use case description, task type, target metric) and generates everything: system prompt, `program.md`, search space, stopping criteria. This is the main departure from autoresearch, where `program.md` is hand-authored.

**One editable file.** The agent only touches `finetune.py`. Small diffs, safe, reviewable. The rest is infrastructure.

**Frozen eval set.** Created once during data prep, never changes. Every iteration scores against the same examples.

**Per-stage LLM routing.** Each pipeline stage (data prep, agent, evaluator) can independently use Claude or Ollama. Burn API credits only where it matters.

**Task-aware evaluation.** Classification gets `max_new_tokens=30` and accuracy. Extraction gets `max_new_tokens=256` and `json_field_accuracy` (field-level partial credit). Generation gets `rouge_l`. Falls back to LLM-as-judge when strict metrics are insufficient.

## Example datasets

Three built-in datasets in `examples/`:

- **Email Urgency** — 200 emails classified as urgent/not_urgent/needs_review
- **Invoice Extraction** — 200 invoices → structured JSON (number, vendor, total, items)
- **Commit Messages** — 200 git diffs → conventional commit messages

## Requirements

- Python 3.10+
- Apple Silicon Mac (MPS) or NVIDIA GPU (CUDA). CPU works but is slow.
- Anthropic API key (or Ollama for fully local mode)
- ~4 GB free disk space for adapters

## License

MIT
