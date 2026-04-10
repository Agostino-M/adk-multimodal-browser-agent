import logging
import os
from dotenv import load_dotenv
from langfuse import get_client
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import AgentTool
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.plugins.multimodal_tool_results_plugin import MultimodalToolResultsPlugin

from browser_agent.subagents.execution_agent import execution_agent
from browser_agent.subagents.verification_agent import verification_agent
from browser_agent.callbacks import validate_planner_tools
from browser_agent.prompt import planner_prompt
from browser_agent.state import (
    add_subtasks,
    complete_session,
    remove_subtask,
    set_current_subtask,
    set_goal,
)

logging.basicConfig(level=logging.DEBUG)

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found")

ENG_API_KEY = os.getenv("ENG_API_KEY")
if not ENG_API_KEY:
    raise ValueError("ENG_API_KEY not found.")

langfuse = get_client()
GoogleADKInstrumentor().instrument()

execute_verify_pipeline = SequentialAgent(
    name="run_execute_verify_step",
    description="Agent that runs execution and verification in one shot for the current subtask.",
    sub_agents=[execution_agent, verification_agent],
)

planner_tools = [
    set_goal,
    set_current_subtask,
    add_subtasks,
    remove_subtask,
    complete_session,
    AgentTool(agent=execute_verify_pipeline),
]

root_agent = LlmAgent(
    name="planner_orchestrator",
    model=LiteLlm(
        api_base="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model=MODEL_NAME,
    ),
    instruction=planner_prompt,
    tools=planner_tools,
    include_contents="none",
    #before_model_callback=lambda callback_context, llm_request: logging.info(f"Before planner orchestrator model call: {llm_request}"),
    before_tool_callback=validate_planner_tools,
)

app = App(
    name="browser_agent",
    root_agent=root_agent,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=3,  # Trigger compaction every 3 new invocations.
        overlap_size=1          # Include last invocation from the previous window.
    ),
    plugins=[MultimodalToolResultsPlugin()],
)
