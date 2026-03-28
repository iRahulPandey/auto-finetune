# auto-finetune

You have input/output examples. You want a small model that does exactly that task. You don't want to think about learning rates.

This project automates that. You describe your task in plain English, upload examples, and an LLM agent searches for the best LoRA hyperparameters by training, evaluating, and iterating — all tracked in MLflow. The best adapter is saved when it's done.

<video src="demo/auto-finetune-demo.mp4" controls width="100%"></video>

---

## The idea

Inspired directly by Karpathy's [autoresearch](https://github.com/karpathy/autoresearch). Same loop: agent reads a `program.md`, proposes a hypothesis, runs an experiment, reads the results, updates its belief, repeats.

**The one change**: in autoresearch, you write `program.md` by hand. Here, you describe your use case in a UI and the system generates `program.md` — including the search space, stopping criteria, layer selection rationale, and evaluation plan. No YAML. No config files. Just plain English and examples.

Everything else follows from that: the agent reads the generated `program.md`, edits exactly one file (`finetune.py`), trains a LoRA adapter, scores it on a frozen eval set, and decides what to try next.

---

## How it differs from AutoML and autoresearch

**AutoML** (AutoSklearn, TPOT, H2O) is designed for tabular data. It searches over fixed pipelines — feature encoders, classifiers, ensembles — using classical HPO algorithms like Bayesian optimization or evolutionary search. It doesn't touch neural networks, doesn't understand task semantics, and can't reason about *why* something failed.

**autoresearch** searches over arbitrary Python code. The agent can write new experiments from scratch, test novel algorithms, discover new approaches. The search space is unbounded. Great for research. Overkill (and unsafe) for "I just need this model to extract invoice fields."

**auto-finetune** sits between them. The search space is fixed and small (learning rate, LoRA rank, target modules, epochs, scheduler). The base model is fixed. The evaluation metric is fixed. The agent can't write new code — it can only change numbers in two dictionaries in `finetune.py`. This constraint is the point. It makes the loop safe, fast, and auditable. You can read every change the agent made.


---

## The loop

```
User: "Classify email urgency" + 200 labeled examples
              │
              ▼
┌─────────────────────────────┐
│  data_prep.py               │
│  · validate examples        │
│  · synthesize system prompt │  ← LLM call
│  · split 80/20 train/eval   │
│  · freeze eval set to disk  │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  program_md_generator.py    │
│  · write program.md         │  ← from UI inputs, not hand-authored
│    - task goal              │
│    - search space           │
│    - layer selection rules  │
│    - stopping criteria      │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  agent_loop.py  (repeats up to N iterations)                │
│                                                             │
│  1. Read program.md + MLflow run history                    │
│  2. LLM proposes: HYPOTHESIS + new LORA_CONFIG + TRAINING_ARGS │  ← Claude/Ollama
│  3. Patch finetune.py with the new config                   │
│  4. Run: python finetune.py --config run_config.json        │
│     · load base model + apply LoRA                          │
│     · train on train.jsonl                                  │
│     · evaluate on frozen eval.jsonl  ← in same process     │
│     · save adapter to adapters/runs/<session>/iter-N/       │
│  5. Log params + metrics + hypothesis to MLflow             │
│  6. If metric < 0.5: call LLM to diagnose failure           │
│  7. If no improvement for 5 runs: warn agent                │
│  8. If metric > threshold: stop early                       │
│  9. If improvement: keep adapter. Else: discard.            │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Best adapter               │
│  · saved to adapters/best/  │
│  · registered in MLflow     │
│    as "production"          │
└─────────────────────────────┘
```

The agent reads `finetune.py` before each iteration. The only thing it can change is:

```python
HYPOTHESIS = "..."          # one sentence explaining the bet

LORA_CONFIG = {
    "r": 16,                # rank
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
}

TRAINING_ARGS = {
    "learning_rate": 3e-5,
    "num_train_epochs": 2,
    "lr_scheduler_type": "cosine",
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.1,
}
```

That's the entire search surface. The agent cannot import new libraries, change the model, touch the eval set, or modify the training loop. It just changes numbers. The tight constraint is what makes this safe to run autonomously.

---

## Key files

```
auto-finetune/
│
├── app.py                   UI: three tabs — Finetune, Inference Lab, LoRA Card
├── finetune.py              The ONLY file the agent edits (LORA_CONFIG + TRAINING_ARGS)
│
├── _core/
│   ├── config.py            Search space, device detection, supported models, RunConfig
│   ├── data_prep.py         Validates examples, synthesizes system prompt, creates train/eval split
│   ├── agent_loop.py        The search loop — proposes configs, trains, evaluates, iterates
│   ├── evaluator.py         Frozen eval metrics — agent never touches this
│   ├── mlflow_utils.py      Experiment tracking, run history formatting, model registry
│   ├── llm_client.py        Unified LLM provider (Claude or Ollama) per pipeline stage
│   └── program_md_generator.py  Auto-generates program.md from RunConfig
│
├── examples/                Three built-in datasets (email, invoice, git commits)
├── program.md               Generated each session — the agent's operating instructions
├── pyproject.toml
└── .env.example
```

**`finetune.py`** is the experiment surface. Think of it as a single cell in a Jupyter notebook that the agent keeps rewriting. The rest of the project is infrastructure that never changes during a run.

**`evaluator.py`** is the scoring contract. It is never modified by the agent. If you change metrics, you must re-run all experiments for fair comparison.

**`program.md`** is generated fresh at the start of each session. It contains the task description, search space, layer selection reasoning, stopping criteria, and agent behavior rules. The agent reads it before every iteration. Never edit it by hand — it gets overwritten.

**`_core/config.py`** is the single source of truth for all constants. Search space bounds, supported models, task types, layer selection rationale. Nothing else in the codebase hardcodes these values.

---

## Install and run

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/user/auto-finetune.git
cd auto-finetune

cp .env.example .env
# add ANTHROPIC_API_KEY to .env

uv sync
streamlit run app.py
```

Go to the **Finetune** tab. Describe your use case. Upload examples (JSON, JSONL, or CSV with `input`/`output` columns). Pick a model and task type. Click **Start training**.

### Fully local (no API key needed)

Run any or all pipeline stages through [Ollama](https://ollama.com) instead of Claude:

```bash
ollama serve
ollama pull llama3.2
```

In the sidebar, toggle each stage (data prep / agent / evaluator) between Claude and Ollama independently. You can run the whole pipeline for $0.

### MLflow

Experiments are stored locally in `mlruns/` by default. To use a remote MLflow server, set `MLFLOW_TRACKING_URI` in `.env`:

```
MLFLOW_TRACKING_URI=http://your-server:5000
```

No server setup required for local use — it just works.

---

## Task types

| Type | Example use case | Primary metric | `max_new_tokens` |
|------|-----------------|---------------|-----------------|
| **classification** | Email urgency, sentiment, intent | accuracy | 30 |
| **extraction** | Invoice fields, named entities, form parsing | json_field_accuracy | 256 |
| **generation** | Commit messages, summaries, rewrites | rouge_l | 256 |

The system detects the task type from your selection and picks appropriate metrics, token limits, and eval strategy automatically.

For extraction tasks, the evaluator uses field-level partial credit (`json_field_accuracy >= 0.8`) rather than exact match — a predicted JSON that gets 4 out of 5 fields right scores 0.8, not 0.0.

---

## Supported models

| Key | Model | Parameters |
|-----|-------|-----------|
| `qwen2.5-0.5b` | Qwen/Qwen2.5-0.5B-Instruct | 0.5B |
| `qwen2.5-1.5b` | Qwen/Qwen2.5-1.5B-Instruct | 1.5B |
| `phi-3-mini` | microsoft/Phi-3-mini-4k-instruct | 3.8B |
| `llama-3.2-1b` | meta-llama/Llama-3.2-1B-Instruct | 1B |
| `llama-3.2-3b` | meta-llama/Llama-3.2-3B-Instruct | 3B |

Start with `qwen2.5-0.5b` — it trains fast and hits surprisingly good accuracy on narrow tasks. Move up to 1.5B or 3B if the 0.5B plateaus.

---

## What the UI gives you

**Finetune tab** — configure, upload, run. Live log of each iteration with metric progress. Iteration table at the end.

**Inference Lab** — pick any trained adapter (or the base model) and test inputs side-by-side. Runs up to 3 adapters simultaneously. Uses PEFT's multi-adapter API so the base model weights load once regardless of how many adapters you compare.

**LoRA Card** — select any completed run and see its full configuration: LoRA params, training args, system prompt, hypothesis, failure diagnosis. Export to JSON for documentation or reproducibility.

---

## Failure mode diagnosis

When a run scores below 0.5, the agent calls the LLM a second time with a sample of prediction mismatches and asks for a concrete failure diagnosis: underfitting vs overfitting, wrong target modules, label format errors, learning rate too high. The diagnosis is logged to MLflow and shown on the LoRA Card. It feeds into the next iteration's prompt.

---

## Design principles

**One editable file.** The agent edits exactly one file, exactly two dicts. Small diffs. Auditable. Reversible. You can read the entire search history as a sequence of small config changes.

**Frozen eval set.** Created once during data prep and never touched again. Every iteration scores against the same examples, so metric comparisons across iterations are meaningful.

**The UI generates the program.** The main departure from autoresearch. Users know what they want the model to do, not what learning rate to use. The UI captures intent and translates it into a complete training program.

**Per-stage LLM routing.** Data prep, agent, and evaluator each independently use Claude or Ollama. Run the expensive reasoning (agent) on Claude; run cheap classification eval on a local model.

**No server dependencies.** MLflow writes to `mlruns/` by default. No Docker. No daemon. No ports to open.

---

## Requirements

- Python 3.10+
- Apple Silicon (MPS) or NVIDIA GPU (CUDA). CPU works but is slow for anything above 0.5B.
- ~4 GB disk space for adapters per session
- Anthropic API key, or Ollama for fully local mode

---

## License

MIT
