import argparse
import csv
import json
import logging
import os
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv

logging.basicConfig(level=logging.DEBUG)

ENV_PATH = Path(__file__).resolve().with_name(".env")
print(ENV_PATH)
load_dotenv(dotenv_path=ENV_PATH)

MODEL_NAME = os.getenv("MODEL_NAME")
if not MODEL_NAME:
    raise ValueError("MODEL_NAME not found")

API_BASE = os.getenv("API_BASE")
if not API_BASE:
    raise ValueError("API_BASE not found")

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found.")

@dataclass
class TaskEvaluation:
    task_id: str
    score: bool
    reason: str
    prediction: Any
    ground_truth: Any


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def _extract_prediction(result_item: Dict[str, Any], prediction_field: str) -> str:
    if prediction_field in result_item:
        return str(result_item[prediction_field]).strip()
    if "events" in result_item:
        return json.dumps(result_item["events"], ensure_ascii=False)
    return json.dumps(result_item, ensure_ascii=False)

def read_tasks_from_csv(file_path):
    """Function for reading CSV dataset"""
    tasks = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            tasks.append(row)
    return tasks

def _llm_score_chatcompletion(input_task: str, prediction: str, ground_truth: str, timeout_s: int) -> TaskEvaluation:
    prompt = (
        "Evaluate the prediction comparing it with ground truth for input task.\n"
        "Return only valid JSON with keys: score (0-1 boolean), reason (string).\n"
        f"INPUT TASK:\n{input_task}\n\n"
        f"GROUND_TRUTH:\n{ground_truth}\n\n"
        f"PREDICTION:\n{prediction}\n"
    )
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME.replace("openai/", ""),
        "messages": [
            {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of a prediction compared to the ground truth. You have to understand the give a score of 1 if the prediction is correct and 0 if it is not, along with a brief reason for the score."},
            {"role": "user", "content": prompt},
        ],
    }
    
    response = requests.post(
        f"{API_BASE}/chat/completions",
        headers=headers,
        json=payload,
        #timeout=timeout_s,
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    score = float(parsed.get("score", 0.0))
    score = max(0.0, min(1.0, score))
    reason = str(parsed.get("reason", "No reason provided."))
    return TaskEvaluation("", score, reason, prediction, ground_truth)


def evaluate(results_path: str, csv_file: str, output_path: str, timeout_s: int):
    results = _load_json(results_path)
    if not isinstance(results, list):
        raise ValueError("Results JSON must be a list of objects.")

    tasks = read_tasks_from_csv(csv_file)
    evaluations: List[TaskEvaluation] = []

    for item in results:
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id", "")).strip()

        prediction = _extract_prediction(item, prediction_field="content")
        input_task = next((task['input'] for task in tasks if task['id'] == task_id), "")
        ground_truth = next((task['ground_truth'] for task in tasks if task['id'] == task_id), "")
        logging.info(f"Evaluating task {task_id}:\n- Input: {input_task} \n- Prediction: {prediction} \n- Ground truth: {ground_truth}")

        task_eval = _llm_score_chatcompletion(
            input_task=input_task,
            prediction=prediction,
            ground_truth=ground_truth,
            timeout_s=timeout_s,
        )

        task_eval.task_id = task_id
        evaluations.append(task_eval)
        logging.info(f"Task {task_id} score={task_eval.score}")

    report = {
        "results_path": results_path,
        "ground_truth_path": csv_file,
        "average_score": sum(e.score for e in evaluations) / len(evaluations) if evaluations else 0.0,
        "evaluated_tasks": len(evaluations),
        "tasks": [
            {
                "task_id": e.task_id,
                "score": e.score,
                "reason": e.reason,
                "prediction": e.prediction,
                "ground_truth": e.ground_truth,
            }
            for e in evaluations
        ],
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple evaluator for browser-agent test results.")
    parser.add_argument("--results", default="results.json", help="Path to results JSON file.")
    parser.add_argument("--csv_file", default="./data/dataset_unified.csv", help="Path to tasks dataset file.")
    parser.add_argument("--output", default="evaluation_report.json", help="Path to save evaluation report.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(
        results_path=args.results,
        csv_file=args.csv_file,
        output_path=args.output,
        timeout_s=args.timeout,
    )
    logging.info(
        "Evaluation complete: %s tasks, average_score=%.3f -> %s",
        report["evaluated_tasks"],
        report["average_score"],
        args.output,
    )


if __name__ == "__main__":
    main()
