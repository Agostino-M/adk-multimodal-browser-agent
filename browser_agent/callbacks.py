import logging
from typing import Any, Dict
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.base_tool import BaseTool, ToolContext
from google.adk.models import LlmRequest

from browser_agent.state import _get_current_subtask, _load_state


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
    logging.info(f"Before {callback_context.agent_name} model call: {llm_request}")
 
    return None


def validate_planner_tools(tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext):
    """
    Before tool callback to enforce deterministic constraints for the planner orchestrator.
    Returns an informative message if the action is invalid, preventing execution and informing the model.
    """
    state = _load_state(tool_context)
    
    tool_name = tool.name
    
    if tool_name == "set_goal":
        if state.goal is not None:
            return "Cannot reset goal: goal already set. Use the existing goal or adjust subtasks instead."
    
    elif tool_name == "complete_session":
        if any(not t.done for t in state.subtasks):
            incomplete_count = sum(1 for t in state.subtasks if not t.done)
            return f"Cannot complete session: there are {incomplete_count} incomplete subtasks. Complete all subtasks first."
    
    return None
