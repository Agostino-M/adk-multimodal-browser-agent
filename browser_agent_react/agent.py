import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from langfuse import get_client
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import AgentTool
from google.adk.apps.app import App
from google.adk.plugins.multimodal_tool_results_plugin import MultimodalToolResultsPlugin
from google.adk.plugins.reflect_retry_tool_plugin import ReflectAndRetryToolPlugin, TrackingScope

from browser_agent.event_compaction import event_compaction
from browser_agent_react.subagents.execution_agent import execution_agent
from browser_agent_react.callbacks import validate_planner_tools
from browser_agent_react.prompt import planner_prompt
from browser_agent.state import (
    add_subtasks,
    complete_session,
    remove_subtask,
    set_current_subtask,
    set_goal,
)

logging.basicConfig(level=logging.DEBUG)

ENV_PATH = Path(__file__).resolve().with_name(".env")
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

langfuse = get_client()
GoogleADKInstrumentor().instrument()

planner_tools = [
    set_goal,
    set_current_subtask,
    add_subtasks,
    remove_subtask,
    complete_session,
    AgentTool(agent=execution_agent),
]

root_agent = LlmAgent(
    name="planner_orchestrator",
    model=LiteLlm(
        api_base=API_BASE,
        api_key=API_KEY,
        model=MODEL_NAME,
        chat_template_kwargs={"enable_thinking": False},
        num_retries=3,
    ),
    instruction=planner_prompt,
    tools=planner_tools,
    include_contents="none",
    #before_model_callback=lambda callback_context, llm_request: logging.info(f"Before planner orchestrator model call: {llm_request}"),
    before_tool_callback=validate_planner_tools,
    after_model_callback=event_compaction,

)

retry_plugin = ReflectAndRetryToolPlugin(
    max_retries=2,
    tracking_scope=TrackingScope.GLOBAL,
    throw_exception_if_retry_exceeded=False,
)

app = App(
    name="browser_agent_react",
    root_agent=root_agent,
    plugins=[MultimodalToolResultsPlugin(), retry_plugin],
)
