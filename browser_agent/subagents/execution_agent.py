import os
from pathlib import Path
from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from browser_agent.event_compaction import event_compaction
from browser_agent.prompt import web_execution_prompt
from browser_agent.callbacks import inject_current_task, validate_execution_tools
from browser_agent.browser import BrowserManager

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

browser = BrowserManager(show_browser=True)
browser_tools = [
    browser.click,
    browser.type,
    browser.scroll,
    browser.goto_url,
    browser.get_state,
    browser.press_key,
    browser.wait,
    browser.close,
]

execution_agent = LlmAgent(
    name="execution_agent",
    model=LiteLlm(
        api_base=API_BASE,
        api_key=API_KEY,
        model=MODEL_NAME,
        chat_template_kwargs={"enable_thinking": False},
    ),
    instruction=web_execution_prompt,
    output_key="execution_output",
    tools=browser_tools,
    include_contents="none",
    before_model_callback=inject_current_task,
    before_tool_callback=validate_execution_tools,
    after_model_callback=event_compaction,
)
