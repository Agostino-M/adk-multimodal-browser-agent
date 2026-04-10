from google.adk.tools.tool_context import ToolContext

from typing import List, Optional
from pydantic import BaseModel, Field


class Subtask(BaseModel):
    """
    A single subtask inside the agent's execution plan.
    """

    id: str
    description: str
    results: Optional[str] = None
    blockers: Optional[str] = None
    retry: int = 0
    done: bool = False

    def mark_done(self, results: Optional[str] = None):
        self.done = True
        if results:
            self.results = results

    def mark_failed(self, blockers: str):
        self.blockers = blockers
        self.retry += 1


class SessionState(BaseModel):
    """
    Output schema returned by the agent at every reasoning cycle.
    """

    goal: Optional[str] = None
    subtasks: List[Subtask] = Field(default_factory=list)
    current_subtask_id: Optional[str] = None
    next_subtask_id: int = 1  # counter for ids
    final_answer: Optional[str] = None
    summary_of_actions: Optional[str] = None


def _load_state(tool_context: ToolContext) -> SessionState:
    raw_state = (
        tool_context.state.to_dict()
        if hasattr(tool_context.state, "to_dict")
        else {}
    )
    return SessionState(**raw_state)

def _save_state(tool_context: ToolContext, state: SessionState):
    tool_context.state.update(state.model_dump(exclude_none=True))

def _get_current_subtask(state: SessionState) -> Optional[Subtask]:
    if not state.current_subtask_id:
        return None
    return next(
        (t for t in state.subtasks if t.id == state.current_subtask_id),
        None
    )

def set_goal(tool_context: ToolContext, goal: str):
    """
    Set the main goal of the session in the state. This should be called only at the beginning of the session to define what we want to achieve.
    """
    state = _load_state(tool_context)

    if state.goal is not None:
        return {"status": "ignored", "message": "Goal already set"}

    state.goal = goal
    _save_state(tool_context, state)

    return state

def add_subtasks(tool_context: ToolContext, new_subtasks: List[str]):
    """
    Add new subtasks to the session state. This can be used both for initial task decomposition and for adding new tasks in case of blockers.
    """
    state = _load_state(tool_context)

    created = []
    for desc in new_subtasks:
        subtask = Subtask(id=f"T{state.next_subtask_id:03d}", description=desc)
        state.next_subtask_id += 1
        state.subtasks.append(subtask)
        created.append(subtask)

    # set first task if none active
    if state.current_subtask_id is None and state.subtasks:
        state.current_subtask_id = state.subtasks[0].id

    _save_state(tool_context, state)

    return state

def set_current_subtask(tool_context: ToolContext, task_id: str):
    """
    Set the current active subtask by its ID from the list of subtasks in the session state.
    This is useful to switch to a specific subtask when needed.
    """
    state = _load_state(tool_context)

    current_task = next((task for task in state.subtasks if task.id == task_id), None)

    if not current_task:
        return {"status": "error", "message": f"Subtask {task_id} not found."}

    state.current_subtask_id = current_task.id

    _save_state(tool_context, state)
    return state

def remove_subtask(tool_context: ToolContext, task_id: str):
    """
    Remove a subtask from the session state by its ID.
    Only removes tasks that are not completed.
    """
    state = _load_state(tool_context)

    idx = next((i for i, t in enumerate(state.subtasks) if t.id == task_id), None)
    if idx is None:
        return {"status": "error", "message": f"Subtask {task_id} not found."}

    task = state.subtasks[idx]

    # Do not remove completed tasks
    if task.done:
        return {"status": "ignored", "message": "Cannot remove completed task"}

    state.subtasks.pop(idx)

    # Update current_subtask_id if necessary
    if state.current_subtask_id == task_id:
        state.current_subtask_id = None
    #    if state.subtasks:
    #        if idx < len(state.subtasks):
    #            state.current_subtask_id = state.subtasks[idx].id
    #        else:
    #            # fallback:last available task
    #            state.current_subtask_id = state.subtasks[-1].id
    #    else:
    #        state.current_subtask_id = None

    _save_state(tool_context, state)
    return state

def complete_session(tool_context: ToolContext, final_result: str, performed_actions: str):
    """
    Mark the session as completed by setting the final answer and a summary of performed actions in the state. This should be called when the agent has achieved the goal and has no more subtasks to perform.
    """
    state = _load_state(tool_context)

    state.final_answer = final_result
    state.summary_of_actions = performed_actions
    state.current_subtask_id = None

    tool_context.actions.escalate = True

    _save_state(tool_context, state)
    return state

def update_current_subtask(tool_context: ToolContext, done: bool, results: Optional[str] = None, blockers: Optional[str] = None):
    state = _load_state(tool_context)

    current = _get_current_subtask(state)

    if current is None:
        return {"status": "error", "message": "No current subtask"}

    if done:
        current.mark_done(results)

        # Find index of current task
        #idx = next(i for i, t in enumerate(state.subtasks) if t.id == current.id)

        # Move to next
        #if idx + 1 < len(state.subtasks):
        #    state.current_subtask_id = state.subtasks[idx + 1].id
        #else:
        #    state.current_subtask_id = None

    else:
        # error / retry
        if blockers:
            current.mark_failed(blockers)
        else:
            current.retry += 1

    _save_state(tool_context, state)

    return state
