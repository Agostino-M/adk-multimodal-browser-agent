"""Minimal GEPA optimizer for planner prompt only."""
 
import argparse
import asyncio
import concurrent.futures
import json
import logging
import os
import requests
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
 
from dotenv import load_dotenv
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from litellm import completion
 
try:
    from gepa import optimize
    from gepa.core.adapter import EvaluationBatch, GEPAAdapter
except ImportError as exc:
    raise RuntimeError("GEPA is required. Install with: pip install gepa") from exc
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set headless mode before browser_agent imports (BrowserManager is instantiated at module level)
if "--headless" in sys.argv:
    os.environ["SHOW_BROWSER"] = "false"

ROOT_DIR = Path(__file__).resolve().parent.parent
# Order matters with override=False: the FIRST file that sets a var wins.
# browser_agent/.env is the canonical production env (task LM, reflection LM),
# evaluation/.env is only used as fallback for shared vars.
ENV_CANDIDATES = [
    ROOT_DIR / "browser_agent" / ".env",
    ROOT_DIR / "evaluation" / ".env",
    ROOT_DIR / ".env",
]
for env_path in ENV_CANDIDATES:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
 
from browser_agent.agent import app as base_app
from browser_agent.agent import root_agent as base_root_agent
from browser_agent.callbacks import validate_planner_tools
from browser_agent.prompt import planner_prompt
from optimization.gepa_eval_sets import load_and_split_dataset
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
 
API_KEY = os.getenv("API_KEY") or os.getenv("GOOGLE_API_KEY")
API_BASE = os.getenv("API_BASE")
MODEL_NAME = os.getenv("MODEL_NAME")
if not API_KEY or not MODEL_NAME:
    raise ValueError("Missing API key or MODEL_NAME in .env")

# Reflection LM can be a different (stronger) model than the task LM.
# Fall back to the task model if not configured.
REFLECTION_MODEL_NAME = os.getenv("REFLECTION_MODEL_NAME", MODEL_NAME)
REFLECTION_API_BASE = os.getenv("REFLECTION_API_BASE", API_BASE)
REFLECTION_API_KEY = os.getenv("REFLECTION_API_KEY", API_KEY)

# Judge LM (used by score_prediction to score agent outputs). Defaults to
# the task model but can be overridden to keep the benchmark judge consistent
# (e.g., task LM = 35B for optimization, judge = 9B for benchmark parity).
JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", MODEL_NAME)
JUDGE_API_BASE = os.getenv("JUDGE_API_BASE", API_BASE)
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY", API_KEY)

TASK_TIMEOUT_S: int = 1800  # overridden by --timeout CLI arg

# Persistent event loop for all agent evaluations.
# Using a single loop across tasks avoids destroying/recreating loops between calls,
# which would cancel open OTel spans and trigger ContextVar mismatch errors.
_eval_loop = asyncio.new_event_loop()
_eval_thread = threading.Thread(target=_eval_loop.run_forever, daemon=True)
_eval_thread.start()
 
 
@dataclass
class ScoreResult:
    score: float
    reason: str
 
 
def format_task_input(example: Dict[str, Any]) -> str:
    task_input = example.get("input", "")
    if example.get("web"):
        task_input += f"\nwebsite: {example['web']}"
    return task_input
 
 
def get_browser_agent_app(candidate_prompt: str):
    base_root_agent.instruction = candidate_prompt
    base_root_agent.before_tool_callback = validate_planner_tools
    return base_app
 
 
async def run_agent_on_task_async(candidate_prompt: str, example: Dict[str, Any]) -> str:
    task_id = example.get("id", "unknown")
    task_input = example.get("input", "")
    if example.get("web"):
        task_input += "\nwebsite: " + example.get("web", "")
 
    session_service = InMemorySessionService()
    session_id = f"gepa_{task_id}_{hash(candidate_prompt) % 10000}"
 
    try:
        await session_service.delete_session(
            app_name="browser_agent",
            session_id=session_id,
            user_id="gepa_optimizer",
        )
    except Exception:
        pass
 
    await session_service.create_session(
        app_name="browser_agent",
        session_id=session_id,
        user_id="gepa_optimizer",
    )
    runner = Runner(app=get_browser_agent_app(candidate_prompt), session_service=session_service)
    task_input = format_task_input(example)
 
    async def _collect_response() -> str:
        last_text = ""
        async for event in runner.run_async(
            user_id="gepa_optimizer",
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text=task_input)]),
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    last_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    last_text = f"Agent escalated: {event.error_message or 'No message'}"
        return last_text or "No response received from agent"

    try:
        return await asyncio.wait_for(_collect_response(), timeout=TASK_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.warning("Task %s timed out after %ss — marking as failed.", task_id, TASK_TIMEOUT_S)
        return f"TIMEOUT: task exceeded {TASK_TIMEOUT_S}s"
    except Exception as exc:
        logger.warning("Agent run failed on task %s: %s", task_id, exc)
        return f"Error: {exc}"

 
def run_agent_on_task_sync(candidate_prompt: str, example: Dict[str, Any]) -> str:
    future = asyncio.run_coroutine_threadsafe(
        run_agent_on_task_async(candidate_prompt, example),
        _eval_loop,
    )
    try:
        return future.result(timeout=TASK_TIMEOUT_S + 60)
    except concurrent.futures.TimeoutError:
        future.cancel()
        task_id = example.get("id", "unknown")
        logger.warning("Task %s timed out in sync wrapper after %ss.", task_id, TASK_TIMEOUT_S)
        return f"TIMEOUT: task exceeded {TASK_TIMEOUT_S}s"
 
 
def score_prediction(task_input: str, prediction: str, ground_truth: str, max_retries: int = 3) -> ScoreResult:
    def sanitize_text(text: str) -> str:
        text = text.replace('\\', '\\\\')
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        return text

    sanitized_input = sanitize_text(task_input)
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

    headers = {"Authorization": f"Bearer {JUDGE_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": JUDGE_MODEL_NAME.replace("openai/", ""),
        "messages": [
            {"role": "system", "content": "You are a precise evaluator for an AI benchmark. Always respond with valid JSON containing only 'score' (0 or 1) and 'reason' (string). Be lenient on numeric precision and format differences, but strict on entity names and completeness. Do not include any other text or formatting."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{JUDGE_API_BASE}/chat/completions",
                headers=headers,
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()

            try:
                parsed = json.loads(content)
                score = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
                return ScoreResult(score=score, reason=str(parsed.get("reason", "No reason provided.")))
            except json.JSONDecodeError:
                content_lower = content.lower()
                if 'score' in content_lower:
                    if '"score": 1' in content or "'score': 1" in content or 'score: 1' in content:
                        score = 1.0
                    elif '"score": 0' in content or "'score': 0" in content or 'score: 0' in content:
                        score = 0.0
                    else:
                        score = 0.0
                    reason = content
                    if '"reason"' in content:
                        try:
                            reason_start = content.find('"reason"') + 10
                            reason_end = content.find('"', reason_start + 1)
                            if reason_end > reason_start:
                                reason = content[reason_start:reason_end]
                        except Exception:
                            pass
                    elif 'reason:' in content_lower:
                        reason = content.split('reason:', 1)[1].strip().split('\n')[0]
                    return ScoreResult(score=score, reason=reason[:200])
                if attempt == max_retries - 1:
                    logger.warning("Failed to parse judge response after %d attempts: %s", max_retries, content[:200])
                    return ScoreResult(score=0.0, reason=f"Failed to parse judge response: {content[:100]}")

        except requests.RequestException as exc:
            logger.warning("Judge request failed (attempt %d/%d): %s", attempt + 1, max_retries, exc)
            if attempt == max_retries - 1:
                return ScoreResult(score=0.0, reason=f"Request failed: {exc}")
        except Exception as exc:
            logger.warning("Scoring failure (attempt %d/%d): %s", attempt + 1, max_retries, exc)
            if attempt == max_retries - 1:
                return ScoreResult(score=0.0, reason=f"Scoring error: {exc}")

    return ScoreResult(score=0.0, reason="Max retries exceeded")
 
 
def evaluate_prompt_on_examples(candidate_prompt: str, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_example: Dict[str, Dict[str, Any]] = {}
    scores: List[float] = []
    for example in examples:
        example_id = str(example.get("id", "unknown"))
        prediction = run_agent_on_task_sync(candidate_prompt, example)
        result = score_prediction(
            task_input=example.get("input", ""),
            prediction=prediction,
            ground_truth=example.get("ground_truth", ""),
        )
        per_example[example_id] = {
            "score": result.score,
            "reason": result.reason,
            "prediction": prediction,
        }
        scores.append(result.score)
    avg_score = (sum(scores) / len(scores)) if scores else 0.0
    return {"average_score": avg_score, "example_results": per_example}
 
 
class BrowserAgentGEPAAdapter(GEPAAdapter[Dict[str, Any], Dict[str, Any], Dict[str, Any]]):
    # Resume optimization: when True, skips the first full-batch evaluate() call
    # because GEPA always re-runs the seed valset eval before checking for
    # gepa_state.bin (the results would be discarded anyway by the loaded state).
    # Saves hours of wasted compute when resuming an interrupted run.
    skip_first_full_batch: bool = False
    _first_batch_skipped: bool = False

    def evaluate(
        self,
        batch: list[Dict[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Dict[str, Any], Dict[str, Any]]:
        if self.skip_first_full_batch and not self._first_batch_skipped:
            self._first_batch_skipped = True
            logger.info(
                "RESUME MODE: skipping wasted seed valset eval (%d tasks). "
                "Loaded gepa_state.bin will overwrite these dummies.",
                len(batch),
            )
            return EvaluationBatch(
                outputs=[{"prediction": "SKIP_RESUME"} for _ in batch],
                scores=[0.0] * len(batch),
                trajectories=None,
            )

        prompt_text = candidate.get("planner_prompt", "")
        outputs: list[Dict[str, Any]] = []
        scores: list[float] = []
        trajectories: list[Dict[str, Any]] | None = [] if capture_traces else None

        for example in batch:
            prediction = run_agent_on_task_sync(prompt_text, example)
            scored = score_prediction(
                task_input=example.get("input", ""),
                prediction=prediction,
                ground_truth=example.get("ground_truth", ""),
            )
            outputs.append({"prediction": prediction})
            scores.append(scored.score)
 
            if trajectories is not None:
                trajectories.append(
                    {
                        "input": example.get("input", ""),
                        "prediction": prediction,
                        "ground_truth": example.get("ground_truth", ""),
                        "feedback": scored.reason,
                        "score": scored.score,
                    }
                )
 
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)
 
    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Dict[str, Any], Dict[str, Any]],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        if not eval_batch.trajectories:
            raise ValueError("Trajectories are required for reflective dataset generation.")
 
        comp = components_to_update[0]
        records: list[Mapping[str, Any]] = []
 
        for traj in eval_batch.trajectories:
            records.append(
                {
                    "Inputs": traj.get("input", ""),
                    "Generated Outputs": traj.get("prediction", ""),
                    "Feedback": traj.get("feedback", ""),
                }
            )
 
        return {comp: records}
 
def custom_reflection_lm(prompt: str | list[dict]) -> str:
    """Reflection LM with VPN-drop tolerance.

    Uses REFLECTION_* env vars when set (so we can run a stronger model
    for reflection while keeping the agent on the task model). Retries
    transient network errors so a brief VPN drop doesn't kill the run.
    """
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    import time as _time
    last_err: Exception | None = None
    for attempt in range(4):  # 4 attempts with generous per-call timeout
        t0 = _time.time()
        try:
            response = completion(
                model=REFLECTION_MODEL_NAME,
                messages=messages,
                api_key=REFLECTION_API_KEY,
                api_base=REFLECTION_API_BASE,
                timeout=1800,  # 30 min: reflection prompts are long + reflection LM may be slow
                # Disable Qwen3 reasoning (<think> blocks): faster generation
                # and clean output that GEPA can use directly as the new prompt.
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            elapsed = _time.time() - t0
            logger.info("Reflection LM call ok (attempt %d, %.1fs)", attempt + 1, elapsed)
            return response.choices[0].message.content
        except Exception as exc:
            elapsed = _time.time() - t0
            last_err = exc
            wait = min(120, 15 * (2 ** attempt))  # 15s, 30s, 60s, 120s
            logger.warning(
                "Reflection LM failed (attempt %d/4, %.1fs): %s — retrying in %ss",
                attempt + 1, elapsed, exc, wait,
            )
            _time.sleep(wait)
    raise RuntimeError(f"Reflection LM failed after 4 retries: {last_err}") from last_err
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal GEPA optimizer for planner prompt.")
    parser.add_argument("--csv", default=str(Path(__file__).parent / "dataset_unified.csv"), help="Dataset CSV path.")
    parser.add_argument("--web_names", type=str, default=None, help="Comma-separated web_name filter.")
    parser.add_argument("--output", default="optimization_runs/gepa_planner_minimal", help="Output directory.")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--max_metric_calls", type=int, default=60, help="GEPA evaluation budget.")
    parser.add_argument("--reflection_minibatch_size", type=int, default=4, help="GEPA reflection minibatch size.")
    parser.add_argument("--train_sample_size", type=int, default=0, help="Optional train subset size (0=all).")
    parser.add_argument("--val_sample_size", type=int, default=0, help="Optional val subset size (0=all).")
    parser.add_argument("--smoke_eval_size", type=int, default=3, help="Validation examples for baseline vs best check.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--display_progress_bar", action="store_true", help="Show GEPA progress.")
    parser.add_argument("--timeout", type=int, default=2000, help="Max seconds per task before marking it failed (default: 1800 = 30 min).")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode.")
    parser.add_argument("--reflection_model", type=str, default=None, help="Override REFLECTION_MODEL_NAME env (e.g. coder_openai).")
    parser.add_argument("--reflection_api_base", type=str, default=None, help="Override REFLECTION_API_BASE env.")
    parser.add_argument("--reflection_api_key", type=str, default=None, help="Override REFLECTION_API_KEY env.")
    parser.add_argument("--seed_prompt_file", type=str, default=None, help="Path to a best_prompt.txt to use as seed instead of the default planner_prompt. Enables session-chaining: pass the previous run's best_prompt.txt to continue optimization.")
    return parser.parse_args()
 
 
def main() -> None:
    args = parse_args()
    global TASK_TIMEOUT_S, REFLECTION_MODEL_NAME, REFLECTION_API_BASE, REFLECTION_API_KEY
    TASK_TIMEOUT_S = args.timeout
    if args.reflection_model:    REFLECTION_MODEL_NAME = args.reflection_model
    if args.reflection_api_base: REFLECTION_API_BASE = args.reflection_api_base
    if args.reflection_api_key:  REFLECTION_API_KEY = args.reflection_api_key
    logger.info("=" * 60)
    logger.info("Task LM       : %s @ %s", MODEL_NAME, API_BASE)
    logger.info("Reflection LM : %s @ %s", REFLECTION_MODEL_NAME, REFLECTION_API_BASE)
    logger.info("Judge LM      : %s @ %s", JUDGE_MODEL_NAME, JUDGE_API_BASE)
    logger.info("=" * 60)
    web_names = [w.strip() for w in args.web_names.split(",")] if args.web_names else None
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    train_examples, val_examples = load_and_split_dataset(
        csv_path=args.csv,
        web_names=web_names,
        train_ratio=args.train_ratio,
        random_seed=args.seed,
    )
    if args.train_sample_size > 0:
        train_examples = train_examples[: args.train_sample_size]
    if args.val_sample_size > 0:
        val_examples = val_examples[: args.val_sample_size]
    if not train_examples:
        raise ValueError("No training examples loaded.")
 
    logger.info("Train examples: %s | Val examples: %s", len(train_examples), len(val_examples))
    adapter = BrowserAgentGEPAAdapter()

    # Resume detection: GEPA always re-runs the seed valset eval before checking
    # for gepa_state.bin. If we're resuming, the results would be discarded — so
    # skip the eval entirely by short-circuiting the adapter on the first call.
    state_bin = output_dir / "gepa_state.bin"
    if state_bin.exists():
        size_kb = state_bin.stat().st_size / 1024
        logger.info("=" * 60)
        logger.info("RESUME detected: %s exists (%.1f KB)", state_bin, size_kb)
        logger.info("Adapter will skip the wasted seed valset eval.")
        logger.info("=" * 60)
        adapter.skip_first_full_batch = True
    else:
        logger.info("Fresh start: no gepa_state.bin found, will do full seed valset eval.")

    # Seed prompt: either the default planner_prompt, or a best_prompt.txt
    # from a previous session (for chaining short runs together).
    if args.seed_prompt_file:
        seed_path = Path(args.seed_prompt_file)
        if not seed_path.exists():
            raise FileNotFoundError(f"--seed_prompt_file not found: {seed_path}")
        seed_text = seed_path.read_text(encoding="utf-8")
        logger.info("Seeding from file: %s (%d chars)", seed_path, len(seed_text))
        seed_candidate = {"planner_prompt": seed_text}
    else:
        logger.info("Seeding from browser_agent.prompt.planner_prompt")
        seed_candidate = {"planner_prompt": planner_prompt}
 
    logger.info("Starting GEPA optimization")
    result = optimize(
        seed_candidate=seed_candidate,
        trainset=train_examples,
        valset=val_examples,
        adapter=adapter,
        task_lm=None,
        reflection_lm=custom_reflection_lm,
        reflection_minibatch_size=args.reflection_minibatch_size,
        max_metric_calls=args.max_metric_calls,
        run_dir=str(output_dir),
        display_progress_bar=args.display_progress_bar,
        cache_evaluation=False,
        seed=args.seed,
        raise_on_exception=False,
    )
 
    best_candidate = result.best_candidate
    best_prompt = best_candidate.get("planner_prompt", "")
    best_val_score = float(result.val_aggregate_scores[result.best_idx]) if result.val_aggregate_scores else 0.0
 
    (output_dir / "best_prompt.txt").write_text(best_prompt, encoding="utf-8")

    # Smoke eval: skip when smoke_eval_size <= 0 to save wall time when you
    # plan to validate on the full benchmark anyway (run_test.py + run_eval.py).
    if args.smoke_eval_size > 0:
        smoke_set = (val_examples or train_examples)[: args.smoke_eval_size]
        baseline_smoke = evaluate_prompt_on_examples(seed_candidate["planner_prompt"], smoke_set)
        best_smoke = evaluate_prompt_on_examples(best_prompt, smoke_set)
        smoke_fields = {
            "smoke_num_examples": len(smoke_set),
            "baseline_smoke_avg_score": baseline_smoke["average_score"],
            "best_smoke_avg_score": best_smoke["average_score"],
        }
        smoke_log = f"baseline={baseline_smoke['average_score']:.3f} best={best_smoke['average_score']:.3f}"
    else:
        logger.info("Skipping smoke eval (smoke_eval_size=0).")
        smoke_fields = {
            "smoke_num_examples": 0,
            "baseline_smoke_avg_score": None,
            "best_smoke_avg_score": None,
        }
        smoke_log = "skipped"

    report = {
        "best_idx": result.best_idx,
        "best_validation_score": best_val_score,
        **smoke_fields,
        "best_candidate": {"planner_prompt": best_prompt},
    }
    with open(output_dir / "gepa_result.json", "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)

    logger.info("Finished GEPA optimization")
    logger.info("Best validation score: %.3f", best_val_score)
    logger.info("Smoke scores: %s", smoke_log)
    logger.info("Saved artifacts in %s", output_dir)
 
 
if __name__ == "__main__":
    main()
 