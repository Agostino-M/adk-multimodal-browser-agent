import logging
import re
from google.genai import types
from typing import Any, Dict
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.base_tool import BaseTool, ToolContext
from google.adk.models import LlmRequest, LlmResponse

from browser_agent.state import _get_current_subtask, _load_state, _save_state

# Planner tool names for validation
PLANNER_TOOL_NAMES = [
    "set_goal",
    "set_current_subtask",
    "add_subtasks",
    "remove_subtask",
    "complete_session",
    "run_execute_verify_step"
]

# Browser tool names for validation
BROWSER_TOOL_NAMES = [
    "click",
    "type",
    "scroll",
    "goto_url",
    "get_state",
    "switch_page",
    "press_key",
    "wait",
    "close"
]

def inject_current_task(callback_context: CallbackContext, llm_request: LlmRequest):
    """
    Before model callback to inject the current subtask description into the system instruction,
    allowing the model to have direct access to the current task.
    """
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
    #logging.info(f"Before {callback_context.agent_name} model call: {llm_request}")
 
    return None

def handle_agent_retry(callback_context: CallbackContext, llm_request: LlmRequest):
    """
    Before model callback to enforce a maximum number of iterations.
    If the limit is reached, modifies the system instruction to force
    a final diagnostic response and activates termination mode.
    """ 

    MAX_ITERATIONS = 40
    injected_marker = "=== ITERATION CONTROL ==="

    state = _load_state(callback_context)

    if state.step_count is None:
        logging.info(f"Agent '{callback_context.agent_name}' step count not found")
        state.step_count = 0

    state.step_count += 1
    logging.info(f"Agent '{callback_context.agent_name}' step count: {state.step_count}")

    # original_instruction = llm_request.config.system_instruction
    # if isinstance(original_instruction, str):
    #    text = original_instruction
    # elif original_instruction and hasattr(original_instruction, "parts"):
    #    text = original_instruction.parts[0].text or ""
    # else:
    #    text = ""

    # if injected_marker in text:
    #    text = text.split(injected_marker)[0]

    if state.step_count >= MAX_ITERATIONS:
        state.terminate_execution_agent = True
        logging.info("Sent termination signal in state due to max iterations reached.")

        termination_prompt = f"""
        {injected_marker}
        You have reached the maximum number of iterations ({MAX_ITERATIONS}).

        This is your FINAL turn. Do NOT call tools.

        Provide a structured output with:
        - current state
        - what you tried
        - failure reason

        Respond in JSON format with:
        {{
        "status": "max retries reached",
        "iterations": {state.step_count},
        "summary": "...",
        "failure_reason": "..."
        }}
        """
        # text += termination_prompt
        llm_request.contents.append(
            types.Content(
                role="user", parts=[types.Part.from_text(text=termination_prompt)]
            )
        )

    _save_state(callback_context, state)

    return None

def stop_agent_after_max_iterations(callback_context: CallbackContext, llm_response: LlmResponse):
    """
    After model callback to check the response for a termination signal and set a flag in the state to stop further execution.
    """
    state = _load_state(callback_context)
    if state.terminate_execution_agent:
        logging.info("Termination signal detected.")

        #llm_response.content.parts[0].text = json.dumps({
        #    "status": "max retries reached",
        #    "iterations": state.step_count,
        #    "summary": "Agent stopped due to iteration limit",
        #    "failure_reason": "Maximum iterations reached without task completion",
        #})

        callback_context.actions.end_of_agent = True
        callback_context.actions.escalate = True
    
    #if state.terminate_execution_agent:
    #    logging.info("Termination signal detected.")
    #    #raise TimeoutError(f"Max iterations reached, response: {llm_response}")

    return None

def validate_planner_tools(tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext):
    """
    Before tool callback to enforce deterministic constraints for the planner orchestrator.
    Returns an informative message if the action is invalid, preventing execution and informing the model.
    """
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
    
    elif tool_name == "run_execute_verify_step":
        state.terminate_execution_agent = False
        state.step_count = 0
        _save_state(tool_context, state)

        if state.current_subtask_id is None:
            return "Cannot run execution step: no current subtask set. Use 'set_current_subtask' to set an active task."
    return None


def validate_execution_tools(tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext):
    """
    Before tool callback to enforce constraints for the execution agent, ensuring it only uses browser tools and follows execution rules.
    """
    tool_name = tool.name
    logging.info(f"Invoked tool: {tool_name} with args: {args}")
    if tool_name not in BROWSER_TOOL_NAMES:
        return f"Tool '{tool_name}' does not exist. Available tools: {BROWSER_TOOL_NAMES}"
    logging.debug(f"user_content:{tool_context.user_content},actions:{tool_context.actions},state:{tool_context.state},session:{tool_context.session},")
    # Check if prompt requests to close browser
    if tool_name == "close":
        user_message = tool_context.user_content.parts[0].text if tool_context.user_content and tool_context.user_content.parts else ""
        logging.debug(f"Request to close browser detected. User message: {user_message}")

        close_browser_patterns = [
            r"\bclose\s+browser\b",  # "close browser"
            r"\bshut\s+down\s+browser\b",  # "shut down browser"
            r"\bexit\s+browser\b",  # "exit browser"
            r"\bclose\s+the\s+browser\b",  # "close the browser"
            r"\bclose\s+the\s+web\s+browser\b",  # "close the web browser"
            r"\bshutdown\s+browser\b",  # "shutdown browser"
            r"\bclose\s+session\b",  # "close session"
            r"\bexit\s+session\b",  # "exit session"
            r"\bend\s+session\b",  # "end session"
        ]
        
        if not any(re.search(pattern, user_message, re.IGNORECASE) for pattern in close_browser_patterns):
            return "Error: The 'close_browser' tool was invoked, but the prompt did not explicitly request it."

    return None
