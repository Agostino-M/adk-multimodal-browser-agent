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

ENG_API_KEY = os.getenv("ENG_API_KEY")
if not ENG_API_KEY:
    raise ValueError("ENG_API_KEY not found.")


verification_agent = LlmAgent(
    name="verification_agent",
    model=LiteLlm(
        api_base="http://172.30.22.153/v1",
        api_key=ENG_API_KEY,
        model="openai/Qwen3_5_9B",
        chat_template_kwargs={"enable_thinking": False},
    ),
    instruction=verification_prompt,
    tools=[update_current_subtask],
    include_contents="none",
    before_model_callback=inject_current_task,
)
