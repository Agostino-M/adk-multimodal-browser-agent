import os
from pathlib import Path
from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from browser_agent.prompt import web_execution_prompt
from browser_agent.callbacks import inject_current_task, validate_execution_tools
from browser_agent.browser import BrowserManager

ENV_PATH = Path(__file__).parent.resolve().with_name(".env")
print(ENV_PATH)
load_dotenv(dotenv_path=ENV_PATH)

ENG_API_KEY = os.getenv("ENG_API_KEY")
if not ENG_API_KEY:
    raise ValueError("ENG_API_KEY not found.")

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
        api_base="http://172.30.22.153/v1",
        api_key=ENG_API_KEY,
        model="openai/Qwen3_5_9B",
        chat_template_kwargs={"enable_thinking": False},
    ),
    instruction=web_execution_prompt,
    output_key="execution_output",
    tools=browser_tools,
    include_contents="none",
    before_model_callback=inject_current_task,
    before_tool_callback=validate_execution_tools,
)
