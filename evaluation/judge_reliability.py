"""Test-retest reliability of the LLM-as-a-Judge.

Loads a results JSONL file and runs the judge N times on each (task, prediction,
ground_truth) triple, then aggregates per-task consistency.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from collections import Counter
import requests
from dotenv import load_dotenv

# Use the same env as run_eval.py
ENV_PATH = Path(__file__).resolve().with_name(".env")
load_dotenv(dotenv_path=ENV_PATH)

MODEL_NAME = os.getenv("MODEL_NAME")
API_BASE = os.getenv("API_BASE")
API_KEY = os.getenv("API_KEY")

if not (MODEL_NAME and API_BASE and API_KEY):
    raise ValueError("Missing env vars: MODEL_NAME / API_BASE / API_KEY")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return items


def read_tasks_from_csv(file_path: str) -> List[Dict[str, str]]:
    tasks: List[Dict[str, str]] = []
    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append(row)
    return tasks


def call_judge(input_task: str, prediction: str, ground_truth: str, timeout_s: int = 180) -> Dict[str, Any]:
    """Single judge call. Returns dict with score and reason. No retries: a failure is recorded as such."""

    def sanitize(s: str) -> str:
        return s.replace("\\", "\\\\").replace("\n", " ").replace("\r", " ").replace("\t", " ")

    sanitized_input = sanitize(input_task)
    sanitized_pred = sanitize(prediction)
    sanitized_gt = sanitize(ground_truth)

    if not sanitized_gt:
        return {"score": 0.0, "reason": "No ground truth", "raw": "skipped"}
    if sanitized_pred.startswith("TIMEOUT: task exceeded "):
        return {"score": 0.0, "reason": "Task timed out", "raw": "skipped"}

    prompt = (
        "Evaluate if the PREDICTION correctly answers the INPUT TASK by comparing it with the GROUND TRUTH.\n\n"
        "SCORING RULES — read carefully before scoring:\n\n"
        "Score 1 (correct) when:\n"
        "- The prediction contains the correct answer, even if surrounded by extra explanation or context.\n"
        "- The value matches the ground truth within reasonable numeric precision (e.g., 39.25 matches 'around 39.2'; 63.63 matches 63.64; 65.27% matches 65.25%).\n"
        "- The answer is expressed in equivalent format or notation: '9,675 days' matches 9675; '17 thousand' matches 17000; '0.078 kg' matches '78 g'; '$1,099' matches '1099 dollars'.\n"
        "- The ground truth contains a clear typo or minor spelling error and the prediction gives the factually correct form (e.g., prediction 'Polybius Plaza' matches ground truth 'Ploybius Plaza').\n"
        "- Mathematical or scientific expressions are algebraically/analytically equivalent even if written in a different form.\n\n"
        "Score 0 (incorrect) when:\n"
        "- The prediction is missing one or more required pieces of information explicitly asked in the INPUT TASK.\n"
        "- The prediction names the wrong entity (wrong person, institution, product, location) even if the answer structure looks similar.\n"
        "- The numeric value is outside reasonable rounding tolerance of the ground truth.\n\n"
        "IMPORTANT: Return ONLY a valid JSON object with exactly these keys:\n"
        '{"score": 1, "reason": "brief explanation"} or {"score": 0, "reason": "brief explanation"}\n\n'
        f"INPUT TASK: {sanitized_input}\n\n"
        f"GROUND TRUTH: {sanitized_gt}\n\n"
        f"PREDICTION: {sanitized_pred}\n\n"
        "Reason should be 1-2 sentences explaining which rule above determined your score and what could be improved if the score is 0."
    )

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME.replace("openai/", ""),
        "messages": [
            {"role": "system", "content": "You are a precise evaluator for an AI benchmark. Always respond with valid JSON containing only 'score' (0 or 1) and 'reason' (string). Be lenient on numeric precision and format differences, but strict on entity names and completeness. Do not include any other text or formatting."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        response = requests.post(
            f"{API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()

        try:
            parsed = json.loads(content)
            score = float(parsed.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            reason = str(parsed.get("reason", "No reason"))
            return {"score": score, "reason": reason, "raw": content}
        except json.JSONDecodeError:
            # fallback parse
            content_lower = content.lower()
            score = 0.0
            if '"score": 1' in content or "'score': 1" in content or 'score: 1' in content:
                score = 1.0
            return {"score": score, "reason": content[:200], "raw": content}
    except requests.RequestException as e:
        return {"score": -1.0, "reason": f"REQUEST_ERROR: {e}", "raw": ""}
    except Exception as e:
        return {"score": -1.0, "reason": f"ERROR: {type(e).__name__}: {e}", "raw": ""}


def run_reliability_test(results_path: str, csv_file: str, n_rounds: int, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    results = _load_jsonl(results_path)
    tasks_meta = read_tasks_from_csv(csv_file)
    tasks_by_id = {t["id"]: t for t in tasks_meta}

    # Build list of triples to evaluate
    triples: List[Dict[str, Any]] = []
    for item in results:
        tid = str(item.get("task_id", "")).strip()
        if tid not in tasks_by_id:
            continue
        pred = str(item.get("content", "")).strip()
        triples.append({
            "task_id": tid,
            "input_task": tasks_by_id[tid]["input"],
            "ground_truth": tasks_by_id[tid]["ground_truth"],
            "prediction": pred,
        })

    logging.info(f"Loaded {len(triples)} triples from {results_path}")
    logging.info(f"Will run {n_rounds} rounds. Output to {output_dir}/")

    for r in range(n_rounds):
        round_path = os.path.join(output_dir, f"round_{r:02d}.jsonl")
        if os.path.exists(round_path):
            logging.info(f"Skipping existing round {r}")
            continue
        logging.info(f"--- Round {r} ---")
        t_start = time.time()
        with open(round_path, "w", encoding="utf-8") as f:
            for i, tr in enumerate(triples):
                judgement = call_judge(tr["input_task"], tr["prediction"], tr["ground_truth"])
                record = {
                    "task_id": tr["task_id"],
                    "round": r,
                    "score": judgement["score"],
                    "reason": judgement["reason"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                if (i + 1) % 20 == 0:
                    logging.info(f"  round {r} progress: {i+1}/{len(triples)}")
        dt = time.time() - t_start
        logging.info(f"Round {r} done in {dt/60:.1f} min")


def aggregate(output_dir: str, n_rounds: int, report_path: str) -> None:
    # task_id -> list of scores across rounds
    per_task: Dict[str, List[float]] = {}
    for r in range(n_rounds):
        round_path = os.path.join(output_dir, f"round_{r:02d}.jsonl")
        if not os.path.exists(round_path):
            logging.warning(f"Missing round {r}")
            continue
        with open(round_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                per_task.setdefault(rec["task_id"], []).append(rec["score"])

    n_tasks = len(per_task)
    consistent_full = 0  # all rounds agree
    consistent_modal_rate: List[float] = []
    pairwise_agreement_num = 0
    pairwise_agreement_den = 0
    score_means: Dict[str, float] = {}

    inconsistent_list: List[Dict[str, Any]] = []

    for tid, scores in per_task.items():
        valid = [s for s in scores if s >= 0]
        if not valid:
            continue
        counter = Counter(valid)
        most_common_score, count = counter.most_common(1)[0]
        consistency = count / len(valid)
        consistent_modal_rate.append(consistency)
        if len(set(valid)) == 1:
            consistent_full += 1
        else:
            inconsistent_list.append({
                "task_id": tid,
                "scores": valid,
                "modal_score": most_common_score,
                "consistency": consistency,
            })
        score_means[tid] = sum(valid) / len(valid)

        # Pairwise agreement
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                pairwise_agreement_den += 1
                if valid[i] == valid[j]:
                    pairwise_agreement_num += 1

    pairwise_agreement = pairwise_agreement_num / max(pairwise_agreement_den, 1)
    avg_consistency = sum(consistent_modal_rate) / max(len(consistent_modal_rate), 1)

    report = {
        "n_tasks_evaluated": n_tasks,
        "n_rounds": n_rounds,
        "n_tasks_fully_consistent": consistent_full,
        "frac_tasks_fully_consistent": consistent_full / max(n_tasks, 1),
        "avg_modal_consistency": avg_consistency,
        "pairwise_agreement": pairwise_agreement,
        "n_inconsistent_tasks": len(inconsistent_list),
        "inconsistent_tasks": sorted(inconsistent_list, key=lambda x: x["consistency"]),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logging.info(f"=== Reliability summary ===")
    logging.info(f"  Tasks evaluated: {n_tasks}")
    logging.info(f"  Rounds: {n_rounds}")
    logging.info(f"  Fully consistent tasks: {consistent_full}/{n_tasks} = {consistent_full/n_tasks*100:.1f}%")
    logging.info(f"  Avg modal consistency: {avg_consistency*100:.1f}%")
    logging.info(f"  Pairwise agreement: {pairwise_agreement*100:.1f}%")
    logging.info(f"  Inconsistent tasks: {len(inconsistent_list)}")
    logging.info(f"Report saved to: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="JSONL with predictions")
    parser.add_argument("--csv_file", required=True, help="Dataset CSV")
    parser.add_argument("--output_dir", required=True, help="Where to store round_NN.jsonl")
    parser.add_argument("--n_rounds", type=int, default=10)
    parser.add_argument("--report", default=None)
    parser.add_argument("--aggregate_only", action="store_true", help="Skip judging, only aggregate")
    args = parser.parse_args()

    report_path = args.report or os.path.join(args.output_dir, "reliability_report.json")

    if not args.aggregate_only:
        run_reliability_test(args.results, args.csv_file, args.n_rounds, args.output_dir)

    aggregate(args.output_dir, args.n_rounds, report_path)


if __name__ == "__main__":
    main()
