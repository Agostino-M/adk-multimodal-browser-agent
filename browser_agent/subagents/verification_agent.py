import os
from pathlib import Path
from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from browser_agent.prompt import verification_prompt
from ..callbacks import inject_current_task
from browser_agent.state import update_current_subtask


ENV_PATH = Path(__file__).parent.resolve().with_name(".env")
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

verification_agent = LlmAgent(
    name="verification_agent",
    model=LiteLlm(
        api_base=API_BASE,
        api_key=API_KEY,
        model=MODEL_NAME,
        chat_template_kwargs={"enable_thinking": False},
    ),
    instruction=verification_prompt,
    tools=[update_current_subtask],
    include_contents="none",
    before_model_callback=inject_current_task,
)
