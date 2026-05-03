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

async def run_tasks_and_save_results(runner: Runner, session_service: InMemorySessionService, csv_file: str, result_file: str, web_name: str, n_test: int, user_id="user_test"):
    """Main function for executing all tasks"""
    logging.info(f"Reading csv")
    tasks = read_tasks_from_csv(csv_file)
    if web_name is not None:
        tasks = [task for task in tasks if task.get("web_name") == web_name]
    if n_test is not None:
        tasks = tasks[:n_test]
    logging.info(f"Found {len(tasks)} tasks")
    browser = BrowserManager()
    task_events = []

    for i, task in enumerate(tasks):
        logging.info(f"Task {i}: {task}")
        task_id = task["id"]
        session_id = f"s_{task_id}"
        task_input = task["input"]
        # concatenate input task with web field
        task_input += task['web']

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

        # events = get_events(session_id)
        async for event in events_iterator:
            # Key Concept: is_final_response() marks the concluding message for the turn
            if event.is_final_response():
                if event.content and event.content.parts:
                    # Extract text from the first part
                    final_response_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    # Handle potential errors/escalations
                    final_response_text = (
                        f"Agent escalated: {event.error_message or 'No specific message.'}"
                    )
                task_events.append({"task_id": task_id, "content": final_response_text})

        time.sleep(1)

        # Save results in file JSON
        #print(task_events)
        with open(result_file, "w") as f:
            json.dump(task_events, f)
            f.write("\n")

        logging.info(f"Results saved in {result_file}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple runner for browser-agent test.")
    parser.add_argument("--csv_file", default="./data/dataset_unified.csv", help="Path to tasks dataset file.")
    parser.add_argument("--output", default="results.json", help="Path to save test results JSON file.")
    parser.add_argument("--web_name", type=str, default=None, help="Name of the web to test.")
    parser.add_argument("--n-test", type=int, default=None, help="Number of tests to run.")
    return parser.parse_args()

async def main():
    args = parse_args()
    csv_file = args.csv_file
    result_file = args.output
    web_name = args.web_name
    n_test = args.n_test

    runner = Runner(
        app=app,
        session_service=InMemorySessionService(),
    )

    await run_tasks_and_save_results(runner, runner.session_service, csv_file, result_file, web_name, n_test)


if __name__ == "__main__":
    asyncio.run(main())
