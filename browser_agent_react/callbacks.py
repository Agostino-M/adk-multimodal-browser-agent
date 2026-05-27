import logging
import re
from google.genai import types
from typing import Any, Dict
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.base_tool import BaseTool, ToolContext
from google.adk.models import LlmRequest, LlmResponse

from browser_agent.state import _get_current_subtask, _load_state, _save_state

PLANNER_TOOL_NAMES = [
    "set_goal",
    "set_current_subtask",
    "add_subtasks",
    "remove_subtask",
    "complete_session",
    "execution_agent",
]

EXECUTION_TOOL_NAMES = [
    "click",
    "type",
    "scroll",
    "goto_url",
    "get_state",
    "switch_page",
    "press_key",
    "wait",
    "close",
    "update_current_subtask",
]


def inject_current_task(callback_context: CallbackContext, llm_request: LlmRequest):
    """Inject the current subtask description into the system instruction."""
    state = _load_state(callback_context)
    current = _get_current_subtask(state)
    task_description = current.description if current else "NO TASK"

    injected_marker = "=== CURRENT TASK ==="
    original_instruction = llm_request.config.system_instruction
    if isinstance(original_instruction, str):
        text = original_instruction
    elif original_instruction and hasattr(original_instruction, "parts"):
        text = original_instruction.parts[0].text or ""
    else:
        text = ""

    if injected_marker in text:
        text = text.split(injected_marker)[0]

    text += f"\n{injected_marker}\n{task_description}\n"
    llm_request.config.system_instruction = text
    return None


def handle_agent_retry(callback_context: CallbackContext, llm_request: LlmRequest):
    """Enforce a maximum number of iterations; force a final diagnostic response when hit."""
    MAX_ITERATIONS = 40
    injected_marker = "=== ITERATION CONTROL ==="

    state = _load_state(callback_context)

    if state.step_count is None:
        logging.info(f"Agent '{callback_context.agent_name}' step count not found")
        state.step_count = 0

    state.step_count += 1
    logging.info(f"Agent '{callback_context.agent_name}' step count: {state.step_count}")

    if state.step_count >= MAX_ITERATIONS:
        state.terminate_execution_agent = True
        logging.info("Sent termination signal in state due to max iterations reached.")

        termination_prompt = f"""
        {injected_marker}
        You have reached the maximum number of iterations ({MAX_ITERATIONS}).

        This is your FINAL turn. Do NOT call browser tools.

        Call update_current_subtask(done=False, blockers="Max iterations reached without task completion") and stop.
        """
        llm_request.contents.append(
            types.Content(
                role="user", parts=[types.Part.from_text(text=termination_prompt)]
            )
        )

    _save_state(callback_context, state)
    return None


def stop_agent_after_max_iterations(callback_context: CallbackContext, llm_response: LlmResponse):
    """After-model callback: stop agent execution on termination signal."""
    state = _load_state(callback_context)
    if state.terminate_execution_agent:
        logging.info("Termination signal detected.")
        callback_context.actions.end_of_agent = True
        callback_context.actions.escalate = True
    return None


def validate_planner_tools(tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext):
    """Enforce deterministic constraints for the planner orchestrator."""
    tool_name = tool.name
    logging.info(f"Invoked tool: {tool_name} with args: {args}")
    if tool_name not in PLANNER_TOOL_NAMES:
        return f"Tool '{tool_name}' does not exist. Available tools: {PLANNER_TOOL_NAMES}"

    state = _load_state(tool_context)

    if tool_name == "set_goal":
        if state.goal is not None:
            return "Cannot reset goal: goal already set. Use the existing goal or adjust subtasks instead."

    elif tool_name == "complete_session":
        if any(not t.done for t in state.subtasks):
            incomplete_count = sum(1 for t in state.subtasks if not t.done)
            return f"Cannot complete session: there are {incomplete_count} incomplete subtasks. Complete all subtasks first."

    elif tool_name == "execution_agent":
        state.terminate_execution_agent = False
        state.step_count = 0
        _save_state(tool_context, state)

        if state.current_subtask_id is None:
            return "Cannot run execution step: no current subtask set. Use 'set_current_subtask' to set an active task."

    return None


def validate_execution_tools(tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext):
    """Enforce constraints for the execution agent."""
    tool_name = tool.name
    logging.info(f"Invoked tool: {tool_name} with args: {args}")
    if tool_name not in EXECUTION_TOOL_NAMES:
        return f"Tool '{tool_name}' does not exist. Available tools: {EXECUTION_TOOL_NAMES}"

    if tool_name == "close":
        user_message = (
            tool_context.user_content.parts[0].text
            if tool_context.user_content and tool_context.user_content.parts
            else ""
        )
        logging.debug(f"Request to close browser detected. User message: {user_message}")

        close_browser_patterns = [
            r"\bclose\s+browser\b",
            r"\bshut\s+down\s+browser\b",
            r"\bexit\s+browser\b",
            r"\bclose\s+the\s+browser\b",
            r"\bclose\s+the\s+web\s+browser\b",
            r"\bshutdown\s+browser\b",
            r"\bclose\s+session\b",
            r"\bexit\s+session\b",
            r"\bend\s+session\b",
        ]

        if not any(re.search(pattern, user_message, re.IGNORECASE) for pattern in close_browser_patterns):
            return "Error: The 'close_browser' tool was invoked, but the prompt did not explicitly request it."

    return None
