# Evaluation — benchmark runner, LLM judge, retry protocol

This folder runs the browser agent over the task dataset and scores the results with an **LLM-as-a-Judge**. It is the counterpart of [`optimization/`](../optimization/README.md): same dataset schema, same judge scoring logic.

## Contents

| File | Role |
|---|---|
| `run_test.py` | Batch runner. Instantiates a session per enabled task, drives the agent, and writes one JSONL line per task (`task_id`, `content`, `duration_min`). |
| `run_eval.py` | Scores a results JSONL against the dataset ground truth with the judge LM and writes an aggregate JSON report. |
| `judge_reliability.py` | Test–retest reliability of the judge: runs the judge N times per prediction and reports per-task consistency and pairwise agreement. |
| `data/dataset_unified.csv` | Task dataset (**not redistributed** — see [`optimization/README.md`](../optimization/README.md#dataset)). |

## Dataset

`run_test.py` and `run_eval.py` read the same CSV schema documented in [`optimization/README.md`](../optimization/README.md#schema). Only rows with `enabled=True` are executed. Ground truth for scoring is taken from the `ground_truth` column.

> `data/dataset_unified.csv` is git-ignored: it mixes hand-written tasks with samples from gated benchmarks (e.g. GAIA) that cannot be redistributed. Rebuild it locally following the optimization README.

## Judge LM

The judge is configured via the `.env` in this folder (separate from `browser_agent/.env`, so the judge model can differ from the agent model):

| Var | Used for |
|---|---|
| `MODEL_NAME` | Judge model ID (the `openai/` prefix is stripped before the call) |
| `API_BASE` | OpenAI-compatible chat-completions endpoint |
| `API_KEY` | Bearer token |

Fixed scoring parameters (in `run_eval.py`): `temperature=0.1`, `enable_thinking=False`, `max_retries=3`, HTTP `--timeout` (default 240 s). Scoring is binary (`score` 0/1) with a textual reason; the rules are lenient on numeric precision and equivalent formats, strict on entity names and completeness. If the JSON response is malformed, a text-extraction fallback recovers the score; after `max_retries` failures the task is marked `score=0`.

## Usage

Run the agent over the dataset, then score the output:

```bash
python evaluation/run_test.py \
    --csv_file evaluation/data/dataset_unified.csv \
    --output evaluation/results.jsonl \
    --timeout 2000 --headless

python evaluation/run_eval.py \
    --results evaluation/results.jsonl \
    --csv_file evaluation/data/dataset_unified.csv \
    --output evaluation/evaluation_report.json
```

`run_test.py` flags: `--web_names` (comma-separated category filter), `--n_test` (cap task count), `--timeout` (per-task seconds, default 2700), `--headless`, `--agent` (`browser_agent` or `browser_agent_react`).

### Output formats

`results.jsonl` — one line per task:

```json
{"task_id": "GitHub--14", "content": "mjbvz; Copilot; mrleemurray", "duration_min": 2.6}
```

`content` is the Planner's `final_answer`, or a structured diagnostic string on timeout/exception. The report adds `average_score`, `evaluated_tasks`, and a per-task record (`score`, `reason`, `prediction`, `ground_truth`).

## Retry protocol

Web tasks are flaky (network, layout drift, CAPTCHAs), so a single run underestimates the true success rate. The reported scores come from an **exhaustive retry protocol**:

1. Run the full dataset once (round 1 = single-shot).
2. Score it, identify the failing tasks.
3. Set `enabled=True` only for the failing tasks, `enabled=False` for the rest; re-run.
4. **Smart-merge**: keep the passing predictions from the previous merged result, override only the tasks that now pass in the new round.
5. Repeat until two consecutive rounds add zero net passes (Δ=0) — the plateau.

The `enabled` column lets you re-run only the failing subset without editing the dataset structure. Attempts/task is the cumulative count of non-ERROR runs divided by dataset size, used as an approximate effort metric.

## Judge reliability

To measure how stable the judge's verdicts are (the self-evaluation bias is discussed in the thesis), re-score an existing results file several times and inspect the agreement:

```bash
python evaluation/judge_reliability.py \
    --results evaluation/results.jsonl \
    --csv_file evaluation/data/dataset_unified.csv \
    --output_dir evaluation/reliability_runs \
    --n_rounds 10
```

It writes `round_NN.jsonl` per round and prints a summary with per-task consistency and pairwise agreement.
