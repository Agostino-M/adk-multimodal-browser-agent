# Optimization — GEPA planner prompt tuning

This folder contains the GEPA-based pipeline used to optimize the **planner prompt** of the browser agent.

GEPA ([paper](https://arxiv.org/abs/2507.19457), [repo](https://github.com/gepa-ai/gepa)) is a reflective prompt optimizer: it runs the agent on a training set, lets a *reflection LM* analyze failures, mutates the prompt, and keeps the best candidate against a validation set.

## Contents

| File | Role |
|---|---|
| `gepa_optimization.py` | Main entry point. Defines the `BrowserAgentGEPAAdapter`, the judge-LM scorer, the resume logic, and the CLI. |
| `gepa_eval_sets.py` | Loads the task CSV, filters enabled rows, and splits into train / val. |
| `best_prompts/` | Best prompts produced by past runs (e.g. `35b_gepa_s1_candidate1.txt`). |
| `optimization_runs/` | Per-run output dirs (`best_prompt.txt`, `gepa_result.json`, `gepa_state.bin`, logs). Ignored by git. |

> **Note**: `dataset_unified.csv` is **not** included in the repo (see [Dataset](#dataset) below).

## Dataset

The pipeline expects a CSV at the path passed to `--csv` (default: `optimization/dataset_unified.csv`).

The original `dataset_unified.csv` used for the runs in `best_prompts/` is **not redistributed** in this repo. It mixes hand-written tasks with samples drawn from third-party benchmarks (e.g. GAIA, which is gated on HuggingFace and does not allow redistribution). To reproduce the runs you have to rebuild your own CSV with the same schema.

### Schema

```
enabled,id,level,web,web_name,input,task_id,ground_truth,
original_ground_truth,verified_ground_truth,verification_method,
verification_confidence,verification_notes
```

| Column | Required | Notes |
|---|---|---|
| `enabled` | yes | `True` / `False`. Rows with `False` are skipped by `load_tasks_from_csv`. |
| `id` | yes | Stable unique row id (used for session naming and per-example reporting). |
| `web_name` | yes | Source benchmark tag (e.g. `CUSTOM`, `GAIA`). Filterable via `--web_names`. |
| `input` | yes | The task prompt fed to the agent. |
| `ground_truth` | yes | Reference answer used by the judge LM. |
| `web` | optional | Starting URL. If set, it's appended to `input` as `\nwebsite: <url>` before sending it to the agent. |
| `level`, `task_id`, `original_ground_truth`, `verified_ground_truth`, `verification_*` | optional | Bookkeeping fields, not used by the pipeline. |

### Rebuilding the dataset

The original CSV was assembled by hand. There is no end-to-end build script. For benchmark sources, fetch each one from its canonical location (and accept its license) before adding rows to the CSV. The repo's `evaluation/` folder may contain ad-hoc helper scripts/notebooks that were used as utilities during dataset curation.

## How it works

1. **Load dataset** — `gepa_eval_sets.load_and_split_dataset()` reads the CSV given via `--csv` and produces train / val splits (default 70/30). See [Dataset](#dataset) for the expected schema.
2. **Seed candidate** — by default the seed prompt is `browser_agent.prompt.planner_prompt`. You can override it with `--seed_prompt_file` to continue a previous run.
3. **GEPA loop** (`BrowserAgentGEPAAdapter`):
   - Runs the browser agent end-to-end on each example with the current candidate planner prompt.
   - Scores the final answer with a **judge LM** (`score_prediction`, prompt embedded in `gepa_optimization.py`).
   - Feeds GEPA the trajectories so the **reflection LM** can propose a mutated prompt.
4. **Persist results** — writes `best_prompt.txt` and `gepa_result.json` to the run directory. GEPA also saves its own `candidates.json` and `gepa_state.bin` (used for resume).

## Environment

`gepa_optimization.py` loads env files in this order (first one wins):

```
browser_agent/.env   # canonical: task LM, reflection LM
evaluation/.env      # fallback for shared vars
.env                 # last-resort fallback
```

Required:

| Var | Used for |
|---|---|
| `API_KEY`, `API_BASE`, `MODEL_NAME` | Task LM (the model that drives the agent) |
| `REFLECTION_MODEL_NAME` (+ `REFLECTION_API_BASE`, `REFLECTION_API_KEY`) | Reflection LM. Defaults to the task LM if unset. Use a stronger model here for better mutations. |
| `JUDGE_MODEL_NAME` (+ `JUDGE_API_BASE`, `JUDGE_API_KEY`) | Judge LM used by `score_prediction`. Defaults to the task LM. Override to keep judging consistent across runs (e.g. always score with the benchmark judge). |

## Run

From the repo root:

```bash
python -m optimization.gepa_optimization \
  --csv optimization/dataset_unified.csv \
  --output optimization_runs/my_run \
  --max_metric_calls 60 \
  --reflection_minibatch_size 4 \
  --train_ratio 0.7 \
  --headless
```

Useful flags (full list in `parse_args()`):

| Flag | Purpose |
|---|---|
| `--web_names` | Comma-separated `web_name` filter to restrict the dataset (e.g. `CUSTOM,GAIA`). |
| `--train_sample_size` / `--val_sample_size` | Cap train/val size for quick experiments (0 = full). |
| `--timeout` | Per-task wall-clock budget in seconds before marking the task failed. |
| `--smoke_eval_size` | Tasks used for the baseline-vs-best smoke check at the end (0 to skip). |
| `--seed_prompt_file` | Seed the next run from a previous `best_prompt.txt` (session chaining). |
| `--reflection_model` / `--reflection_api_base` / `--reflection_api_key` | Override reflection LM at the CLI without touching `.env`. |
| `--display_progress_bar` | Show GEPA progress bar. |

## Resume

If `--output` already contains a `gepa_state.bin`, the script detects it and **skips the redundant seed valset evaluation** (which GEPA always re-runs at startup but then discards in favor of the loaded state). This saves hours when resuming an interrupted run.

## Outputs

After a successful run, in the output dir you will find:

```
best_prompt.txt        # best planner prompt found
gepa_result.json       # best_idx, best_validation_score, smoke scores
candidates.json        # all candidates tried (GEPA-managed)
gepa_state.bin         # GEPA state, used for resume
run_log.json / .txt    # logs
```

## References

- GEPA paper: <https://arxiv.org/abs/2507.19457>
- GEPA repo: <https://github.com/gepa-ai/gepa>
- Planner prompt under optimization: `browser_agent/prompt.py` → `planner_prompt`
