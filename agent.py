import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService

from browser import BrowserManager
from prompt import prompt_base

load_dotenv()

browser = BrowserManager(show_browser=True)

# Register tools
browser_tools = [
    browser.click_by_coordinates,
    browser.click_by_selector,
    browser.click_by_text,
    browser.type,
    browser.scroll,
    browser.goto_url,
    browser.get_state,
]

async def handle_event(event):
    if not event.content or not event.content.parts:
        return None, None

    final_text = None
    function_call = None

    for part in event.content.parts:

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "author": event.author,
            "type": type(event).__name__,
        }

        if hasattr(part, "thought") and part.thought:
            log_entry["thought"] = part.text
            print(f"\n🧠 [RAGIONAMENTO]: {part.text}")

        elif part.text:
            log_entry["final_text"] = part.text
            final_text = part.text
            print(f"\n✅ [RISPOSTA FINALE]: {part.text}")

        if hasattr(part, "function_call") and part.function_call:
            fc = part.function_call
            function_call = fc
            log_entry["function_call"] = {
                "name": fc.name,
                "args": fc.args,
                "id": fc.id
            }
            print("\n🔵 FUNCTION CALL")
            print(json.dumps(log_entry["function_call"], indent=2))

        if hasattr(part, "function_response") and part.function_response:
            fr = part.function_response
            log_entry["function_response"] = fr.response
            print("\n🟢 FUNCTION RESPONSE")
            print(json.dumps(fr.response, indent=2))

        with open("agent_log.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    return final_text, function_call


root_agent = LlmAgent(
    tools=browser_tools,
    model=LiteLlm(
        model="openai/qwen3-vl:8b",
        api_base="http://localhost:11434/v1",
        api_key="ollama"
    ),
    instruction=prompt_base,
    name="web_agent_test",
    description="Navigates the web and performs actions based on user instructions",
)

async def main():
    await browser.init()

    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent, app_name="BrowserTest", session_service=session_service
    )

    # Simulazione output di un planner esterno
    """structured_tasks = [
        {
            "task": "goto url https://google.com",
        },
        {
            "task": "Click a button that semantically rejects or declines cookies.",
        },
        {
            "task": "Click the main Google search input field.",
        },
        {
            "task": "Type 'CHATGPT' into the active search input.",
        }
    ]"""

    structured_tasks = [{"task": "goto https://httpbin.org/forms/post"},{"task": "compile form with 'Agostino' as Customer name"},{"task": "submit form"}]

    from google.genai import types
    import json

    try:
        with open("agent_log.jsonl", "a") as f:
            f.write("Start session\n")

        for task in structured_tasks:

            # Stateless session
            session = await session_service.create_session(
                app_name="BrowserTest", user_id="test_user"
            )

            print(f"\n===== TASK: {task} =====")

            img_b64 = await browser._take_screenshot()
            dom = await browser._extract_interactive_elements()
            from pprint import pprint
            pprint(dom)

            new_message = types.Content(
                role="user",
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/png",
                            data=img_b64
                        )
                    ),
                    types.Part(text="Interactive DOM elements:"),
                    types.Part(text=json.dumps(dom)),
                    types.Part(text=f"Current URL: {browser.active_page.url}"),
                    types.Part(text=f"""
                        CURRENT ATOMIC TASK
                        {task}
                        STRICT REQUIREMENTS:
                        - Accomplish ONLY this task.
                        - The action must directly satisfy the task.
                        - Ignore unrelated buttons.
                        - Ignore login unless explicitly requested.
                        - If already completed, respond ONLY with: TASK_COMPLETED.
                        - Do not start other tasks.
                        """)
                ]
            )

            with open("agent_log.jsonl", "a") as f:
                f.write(f"Task: {task}\n")

            async for event in runner.run_async(
                user_id="test_user",
                session_id=session.id,
                new_message=new_message
            ):
                final_text, function_call = await handle_event(event)

    except Exception as e:
        print(f"Errore: {e}")
    finally:
        with open("agent_log.jsonl", "a") as f:
            f.write("End session\n")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
