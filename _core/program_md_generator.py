"""
program_md_generator.py — Auto-generates program.md from user inputs.

This is the equivalent of the human-written program.md in AutoResearch,
but generated programmatically from the Streamlit form inputs.
The agent reads this file to understand its search strategy.
"""

from .config import (
    CONSTRAINTS,
    LAYER_RATIONALE,
    PROGRAM_MD_PATH,
    SEARCH_SPACE,
    RunConfig,
)


def generate_program_md(run_config: RunConfig) -> str:
    """Generate the program.md content from a RunConfig."""

    metric_name = run_config.metric_name
    use_case = run_config.use_case
    model_key = run_config.model_key
    task_type = run_config.task_type
    target_threshold = run_config.target_threshold
    max_iterations = run_config.max_iterations

    lr_options = SEARCH_SPACE["learning_rate"]
    lora_rank_options = SEARCH_SPACE["lora_rank"]
    epoch_options = SEARCH_SPACE["num_train_epochs"]

    content = f"""# Fine-Tuning Session

## Goal
Maximize {metric_name} on the frozen eval set for:
- **Task**: "{use_case}"
- **Model**: {model_key}
- **Task Type**: {task_type}
- **Target metric**: {metric_name} > {target_threshold}

## Search Space — Explore in This Priority Order

1. **Learning rate**: {lr_options}
   (Most impactful. Try smallest first, then increase.)

2. **LoRA rank**: {lora_rank_options}
   (Higher rank = more capacity but slower. Start at 16.)

3. **Number of epochs**: {epoch_options}
   (Risk of overfitting increases with epochs on small datasets.)

4. **LR scheduler**: {SEARCH_SPACE["lr_scheduler_type"]}
   (Only explore if learning rate changes aren't yielding improvements.)

5. **Target modules**: {SEARCH_SPACE["target_modules"]}
   Treat this as a first-class hyperparameter — explore freely from iteration 1.
   Recommended starting point for {task_type}: {LAYER_RATIONALE[task_type]["recommended"]}
   Rationale: {LAYER_RATIONALE[task_type]["rationale"]}
   Escalate to {LAYER_RATIONALE[task_type]["high_capacity"]} when: {LAYER_RATIONALE[task_type]["escalate_when"]}

## Layer Selection Reasoning

| Layer set | Best for |
|-----------|----------|
| `["q_proj", "v_proj"]` | Classification — minimal budget, label routing |
| `["q_proj", "v_proj", "o_proj"]` | Generation — output projection critical for fluency |
| `["q_proj", "v_proj", "k_proj", "o_proj"]` | Extraction — all projections needed for field precision |

**This task ({task_type}) → start with `{LAYER_RATIONALE[task_type]["recommended"]}`**
If metric stagnates after 3 iterations, switch layer set before changing rank or lr.

## Constraints — DO NOT CHANGE THESE

- Do not modify evaluator.py
- Do not change the eval dataset path
- Do not change the base model ({model_key})
- Do not add new Python dependencies
- Do not change the HYPOTHESIS string format (it must be a single descriptive sentence)
- LoRA rank must not exceed {CONSTRAINTS["max_lora_rank"]} (memory constraint)
- learning_rate must stay between {CONSTRAINTS["min_learning_rate"]} and {CONSTRAINTS["max_learning_rate"]}
- num_train_epochs must stay between {CONSTRAINTS["min_epochs"]} and {CONSTRAINTS["max_epochs"]}

## Agent Behavior Rules

- Read the MLflow run history before proposing each config
- If the last 3 runs all showed no improvement, switch to a different hyperparameter dimension
- Always state your hypothesis in the HYPOTHESIS string before running
- If metric_value > {target_threshold}: stop early and report
- On the final iteration: write a summary comment in HYPOTHESIS explaining what was learned

## Stopping Criteria

- Stop after {max_iterations} iterations
- Stop early if {metric_name} > {target_threshold}
- Final output: best adapter saved to ./adapters/best/, registered in MLflow as "production"

## Current Best

- **Best {metric_name}**: (will be updated by agent_loop)
- **Best config**: (will be updated by agent_loop)

## Run History

| Iteration | Hypothesis | {metric_name} | LR | Rank | Epochs | Scheduler | Status |
|-----------|-----------|{"-" * len(metric_name)}--|------|------|--------|-----------|--------|
"""
    return content


def write_program_md(run_config: RunConfig) -> None:
    """Generate and write program.md to disk."""
    content = generate_program_md(run_config)
    PROGRAM_MD_PATH.write_text(content, encoding="utf-8")
