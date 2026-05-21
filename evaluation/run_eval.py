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


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

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

def _llm_score_chatcompletion(input_task: str, prediction: str, ground_truth: str, timeout_s: int, max_retries: int = 3) -> TaskEvaluation:

    # Sanitize inputs to avoid JSON parsing issues
    def sanitize_text(text: str) -> str:
        """Remove or escape problematic characters for LLM prompts"""
        text = text.replace('\\', '\\\\')
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        return text

    sanitized_input = sanitize_text(input_task)
    sanitized_prediction = sanitize_text(prediction)
    sanitized_ground_truth = sanitize_text(ground_truth)

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
        f"GROUND TRUTH: {sanitized_ground_truth}\n\n"
        f"PREDICTION: {sanitized_prediction}\n\n"
        "Reason should be 1-2 sentences explaining which rule above determined your score and what could be improved if the score is 0."
    )

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME.replace("openai/", ""),
        "messages": [
            {"role": "system", "content": "You are a precise evaluator for an AI benchmark. Always respond with valid JSON containing only 'score' (0 or 1) and 'reason' (string). Be lenient on numeric precision and format differences, but strict on entity names and completeness. Do not include any other text or formatting."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,  # Low temperature for consistent scoring
    }

    for attempt in range(max_retries):
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

            # Try to parse JSON
            try:
                parsed = json.loads(content)
                score = float(parsed.get("score", 0.0))
                score = max(0.0, min(1.0, score))  # Clamp to 0-1
                reason = str(parsed.get("reason", "No reason provided."))
                return TaskEvaluation("", score, reason, prediction, ground_truth)
            except json.JSONDecodeError:
                # Try to extract score and reason manually from text
                content_lower = content.lower()
                if 'score' in content_lower:
                    # Look for score patterns
                    if '"score": 1' in content or "'score': 1" in content or 'score: 1' in content:
                        score = 1.0
                    elif '"score": 0' in content or "'score': 0" in content or 'score: 0' in content:
                        score = 0.0
                    else:
                        score = 0.0

                    # Extract reason
                    reason = content
                    if '"reason"' in content:
                        # Try to extract reason value
                        try:
                            reason_start = content.find('"reason"') + 10
                            reason_end = content.find('"', reason_start + 1)
                            if reason_end > reason_start:
                                reason = content[reason_start:reason_end]
                        except:
                            pass
                    elif 'reason:' in content_lower:
                        reason = content.split('reason:', 1)[1].strip().split('\n')[0]

                    return TaskEvaluation("", score, reason[:200], prediction, ground_truth)

                # If we can't parse, retry or return default
                if attempt == max_retries - 1:
                    logging.warning(f"Failed to parse LLM response after {max_retries} attempts: {content[:200]}...")
                    return TaskEvaluation("", 0.0, f"Failed to parse LLM response: {content[:100]}...", prediction, ground_truth)

        except requests.RequestException as e:
            logging.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return TaskEvaluation("", 0.0, f"Request failed: {str(e)}", prediction, ground_truth)
        except Exception as e:
            logging.warning(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return TaskEvaluation("", 0.0, f"Unexpected error: {str(e)}", prediction, ground_truth)

    # Fallback
    return TaskEvaluation("", 0.0, "Max retries exceeded", prediction, ground_truth)


def evaluate(results_path: str, csv_file: str, output_path: str, timeout_s: int, max_retries: int = 3):
    results = _load_jsonl(results_path)

    tasks = read_tasks_from_csv(csv_file)
    evaluations: List[TaskEvaluation] = []
    eval_records: List[Dict[str, Any]] = []

    # Open intermediate file for crash-recovery: each entry is a JSONL line.
    intermediate_file = output_path + ".tmp"
    output_handle = open(intermediate_file, "w", encoding="utf-8")

    try:
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
                max_retries=max_retries,
            )

            task_eval.task_id = task_id
            evaluations.append(task_eval)
            logging.info(f"Task {task_id} score={task_eval.score:.1f} - {task_eval.reason[:50]}...")

            eval_dict = {
                "input_task": input_task,
                "task_id": task_eval.task_id,
                "score": task_eval.score,
                "reason": task_eval.reason,
                "prediction": task_eval.prediction,
                "ground_truth": task_eval.ground_truth,
            }
            eval_records.append(eval_dict)
            # Write one JSON line per entry for crash-recovery (JSONL format).
            output_handle.write(json.dumps(eval_dict, ensure_ascii=True) + "\n")
            output_handle.flush()

        output_handle.close()

        # Build final report using in-memory data — no re-read needed.
        report = {
            "results_path": results_path,
            "ground_truth_path": csv_file,
            "average_score": sum(e.score for e in evaluations) / len(evaluations) if evaluations else 0.0,
            "evaluated_tasks": len(evaluations),
            "tasks": eval_records,
        }

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)

        try:
            os.remove(intermediate_file)
        except OSError:
            pass

        return report

    except Exception as e:
        output_handle.close()
        logging.error(f"Evaluation failed: {e}")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple evaluator for browser-agent test results.")
    parser.add_argument("--results", default="results.jsonl", help="Path to results JSON file.")
    parser.add_argument("--csv_file", default="./data/dataset_unified.csv", help="Path to tasks dataset file.")
    parser.add_argument("--output", default="evaluation_report.json", help="Path to save evaluation report.")
    parser.add_argument("--timeout", type=int, default=240, help="HTTP timeout in seconds.")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum retries for LLM evaluation calls.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(
        results_path=args.results,
        csv_file=args.csv_file,
        output_path=args.output,
        timeout_s=args.timeout,
        max_retries=args.max_retries,
    )
    logging.info(
        "Evaluation complete: %s tasks, average_score=%.3f -> %s",
        report["evaluated_tasks"],
        report["average_score"],
        args.output,
    )


if __name__ == "__main__":
    main()
