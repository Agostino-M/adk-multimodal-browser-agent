import argparse
import asyncio
import csv
import requests
import json
import time
import logging
import sys
import os
from typing import AsyncIterator

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set headless mode before browser_agent imports
if "--headless" in sys.argv:
    os.environ["SHOW_BROWSER"] = "false"

from google.genai import types
from google.adk.events import Event
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from browser_agent.agent import root_agent, app

from browser_agent.browser import BrowserManager

#logging.basicConfig(level=logging.DEBUG)

def read_tasks_from_csv(file_path):
    """Function for reading CSV dataset"""
    tasks = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("enabled", "true").lower() == "true":
                tasks.append(row)
    return tasks

def create_session(session_id, user_id="user_test"):
    """Function for creating a session for the task"""
    url = f"http://localhost:8000/apps/browser_agent/users/{user_id}/sessions/{session_id}"
    headers = {"Content-Type": "application/json"}
    #data = {
    #    "key1": "value1",  # handle params
    #    "key2": 42
    #}
    
    response = requests.post(url, headers=headers)
    if response.status_code == 200:
        logging.debug(f"Session created: response:{response}")
        return response.json()
    else:
        logging.error(f"Error during creation of session for the task {session_id}")
        return None

async def invoke_agent(runner, task_id, session_id, task_input, user_id="user_test"):
    """Function for invoking  the agent"""
    '''url = "http://localhost:8000/run"
    headers = {"Content-Type": "application/json"}
    data = {
        "appName": "browser_agent",
        "userId": user_id,
        "sessionId": session_id,
        "newMessage": {
            "role": "user",
            "parts": [{
                "text": task_input
            }]
        }
    }
    logging.debug(f"Invoking llm with url={url}, data={data}")

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        logging.error(f"Error during invocation of agent for the task {task_id}")
        return None
    '''
    
    return 

def get_events(session_id, user_id="user_test"):
    """Function for retrieving events"""
    url = f"http://localhost:8000/apps/browser_agent/users/{user_id}/sessions/{session_id}/events"
    response = requests.get(url)
    if response.status_code == 200:
        events = response.json()
        return events
    else:
        logging.error(f"Error during retrieval of events for the session {session_id}")
        return None

def delete_session(session_id, user_id="user_test"):
    """Function for deleting the session"""
    url = f"http://localhost:8000/apps/browser_agent/users/{user_id}/sessions/{session_id}"
    response = requests.delete(url)
    if response.status_code == 200:
        logging.info(f"Session {session_id} deleted successfully.")
    else:
        logging.error(f"Error during deletion of session {session_id}. Status code: {response.status_code}")


async def collect_final_response(events_iterator: AsyncIterator[Event]) -> str:
    """Drain the event stream and return the last final response text.
    The planner's conclusive answer is always the last is_final_response() event;
    """
    last_text = ""
    async for event in events_iterator:
        if event.is_final_response():
            if event.content and event.content.parts:
                last_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:
                last_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
    return last_text


async def run_tasks_and_save_results(runner: Runner, session_service: InMemorySessionService, csv_file: str, result_file: str, web_names: list, n_test: int, task_timeout: int, user_id="user_test"):
    """Main function for executing all tasks"""
    from browser_agent.subagents.execution_agent import browser as _task_browser

    logging.info(f"Reading csv")
    tasks = read_tasks_from_csv(csv_file)
    if web_names is not None:
        tasks = [task for task in tasks if task.get("web_name") in web_names]
    if n_test is not None:
        tasks = tasks[:n_test]
    logging.info(f"Found {len(tasks)} tasks")

    with open(result_file, "w", encoding="utf-8") as result_fh:
        for i, task in enumerate(tasks):
            logging.info(f"Task {i}: {task}")
            task_id = task["id"]
            session_id = f"s_{task_id}"
            task_input = task["input"]
            task_input += "\nsuggested website: " + task['web'] if task.get('web') else ""

            # Create session
            # session = delete_session(session_id)
            # session = create_session(session_id)
            await session_service.delete_session(app_name="browser_agent", session_id=session_id, user_id="user_test")
            await session_service.create_session(app_name="browser_agent", session_id=session_id, user_id="user_test")

            # Invoke agent for task
            # events = await invoke_agent(runner, task_id, session_id, task_input)
            events_iterator: AsyncIterator[Event] = runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(role="user", parts=[types.Part(text=task_input)]),
            )

            task_start = time.time()
            try:
                final_response_text = await asyncio.wait_for(
                    collect_final_response(events_iterator),
                    timeout=task_timeout,
                )
                result = {
                    "task_id": task_id,
                    "content": final_response_text,
                    "duration_min": round((time.time() - task_start) / 60, 2),
                }
            except asyncio.TimeoutError:
                logging.warning(f"Task {task_id} timed out after {task_timeout}s — marking as failed.")
                result = {
                    "task_id": task_id,
                    "content": f"TIMEOUT: task exceeded {task_timeout}s.",
                    "duration_min": round((time.time() - task_start) / 60, 2),
                }
            finally:
                try:
                    await _task_browser.close()
                    logging.info(f"Browser closed and reset after task {task_id}.")
                except Exception as e:
                    logging.warning(f"Browser close failed after task {task_id}: {e}")

            result_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            result_fh.flush()
            logging.info(f"Task {task_id} saved to {result_file}.")

            time.sleep(1)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple runner for browser-agent test.")
    parser.add_argument("--csv_file", default="./data/dataset_unified.csv", help="Path to tasks dataset file.")
    parser.add_argument("--output", default="results.jsonl", help="Path to save test results JSON file.")
    parser.add_argument("--web_names", type=str, default=None, help="Comma-separated names of the web to test, e.g. 'CUSTOM,GAIA'.")
    parser.add_argument("--n_test", type=int, default=None, help="Number of tests to run.")
    parser.add_argument("--timeout", type=int, default=60*45, help="Max seconds per task before marking it as failed.")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode.")
    return parser.parse_args()

async def main():
    args = parse_args()
    # Convert comma-separated string to list if provided
    web_names = [w.strip() for w in args.web_names.split(",")] if args.web_names else None
    csv_file = args.csv_file
    result_file = args.output
    n_test = args.n_test
    task_timeout = args.timeout

    runner = Runner(
        app=app,
        session_service=InMemorySessionService(),
    )

    from browser_agent.subagents.execution_agent import browser as _browser
    try:
        await run_tasks_and_save_results(runner, runner.session_service, csv_file, result_file, web_names, n_test, task_timeout)
    finally:
        await _browser.close()


if __name__ == "__main__":
    asyncio.run(main())
