import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.apps.app import App
from google.adk.plugins.multimodal_tool_results_plugin import MultimodalToolResultsPlugin

from browser_agent.browser import BrowserManager
from browser_agent.prompt import prompt_base

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found")

browser = BrowserManager(show_browser=True)

# Register tools
browser_tools = [
    browser.click,
    browser.type,
    browser.scroll,
    browser.goto_url,
    browser.get_state,
    browser.wait,
    browser.close,
]


root_agent = LlmAgent(
    tools=browser_tools,
    model=LiteLlm(
        api_base="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model="openai/qwen/qwen3-vl-30b-a3b-thinking",
    ),
    instruction=prompt_base,
    name="web_agent_test",
    description="Navigates the web and performs actions based on user instructions",
)

app = App(
    name="browser_agent",
    root_agent=root_agent,
    plugins=[MultimodalToolResultsPlugin()],
)
