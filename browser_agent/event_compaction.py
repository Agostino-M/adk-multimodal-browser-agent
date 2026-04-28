import logging
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from google.adk.agents.callback_context import CallbackContext
from google.adk.events.event import Event
from google.adk.models import LiteLlm, LlmResponse
from google.adk.apps.compaction import LlmEventSummarizer
from google.adk.sessions.session import Session

ENV_PATH = Path(__file__).resolve().with_name(".env")
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

model = LiteLlm(
    api_base=API_BASE,
    api_key=API_KEY,
    model=MODEL_NAME,
    chat_template_kwargs={"enable_thinking": False},
)

EVENT_COMPACTION_WINDOW = 4
EVENT_COMPACTION_OVERLAP = 1

summarizer = LlmEventSummarizer(llm=model)


def _event_is_relevant_for_summary(event: Event) -> bool:
    if event.actions and event.actions.compaction:
        return False
    # check on author, type, tool calls, etc to further filter out non-relevant events if needed
    return True


def _latest_compaction_end_timestamp(events: List[Event]) -> float:
    for event in reversed(events):
        if event.actions and event.actions.compaction:
            end_timestamp = event.actions.compaction.end_timestamp
            if end_timestamp is not None:
                return end_timestamp
    return 0.0


def _event_function_call_ids(event: Event) -> set[str]:
    function_call_ids: set[str] = set()
    for function_call in event.get_function_calls():
        if function_call.id:
            function_call_ids.add(function_call.id)
    return function_call_ids


def _event_function_response_ids(event: Event) -> set[str]:
    function_response_ids: set[str] = set()
    for function_response in event.get_function_responses():
        if function_response.id:
            function_response_ids.add(function_response.id)
    return function_response_ids


def _pending_function_call_ids(events: List[Event]) -> set[str]:
    all_call_ids: set[str] = set()
    all_response_ids: set[str] = set()
    for event in events:
        all_call_ids.update(_event_function_call_ids(event))
        all_response_ids.update(_event_function_response_ids(event))
    return all_call_ids - all_response_ids


def _has_pending_function_call(event: Event, pending_ids: set[str]) -> bool:
    call_ids = _event_function_call_ids(event)
    return bool(call_ids and not call_ids.isdisjoint(pending_ids))


def _truncate_events_before_pending_function_call(
    events: List[Event], pending_ids: set[str]
) -> List[Event]:
    for index, event in enumerate(events):
        if _has_pending_function_call(event, pending_ids):
            return events[:index]
    return events


def _select_sliding_window_events(
    session: Session,
    window: int,
    overlap: int,
) -> List[Event]:
    # Instead of using invocation IDs, count relevant events since last compaction
    last_compacted_end_timestamp = _latest_compaction_end_timestamp(session.events)
    logging.debug(f"Last compacted end timestamp: {last_compacted_end_timestamp}")

    # Get all events after last compaction
    recent_events = [
        event for event in session.events
        if event.timestamp > last_compacted_end_timestamp
    ]
    logging.debug(f"Recent events since last compaction: {len(recent_events)}")

    # Filter to relevant events (user/model textual)
    relevant_recent_events = [
        event for event in recent_events
        if _event_is_relevant_for_summary(event)
    ]
    logging.debug(f"Relevant recent events (user/model textual): {len(relevant_recent_events)}")

    if len(relevant_recent_events) < window:
        logging.debug(f"Not enough relevant events ({len(relevant_recent_events)} < {window}), skipping compaction.")
        return []

    # Take the last 'window' relevant events, plus some overlap
    events_to_compact = relevant_recent_events[-(window + overlap):]
    logging.debug(f"Selected {len(events_to_compact)} events for compaction")

    # Also include any tool events that are between these relevant events
    # to maintain context
    if events_to_compact:
        first_timestamp = events_to_compact[0].timestamp
        last_timestamp = events_to_compact[-1].timestamp
        
        all_events_in_window = [
            event for event in session.events
            if first_timestamp <= event.timestamp <= last_timestamp
            and not (event.actions and event.actions.compaction)
        ]
        
        # Filter out events before any pending function calls
        pending_ids = _pending_function_call_ids(session.events)
        all_events_in_window = _truncate_events_before_pending_function_call(
            all_events_in_window, pending_ids
        )
        
        logging.debug(f"Total events in window (including tools): {len(all_events_in_window)}")
        return all_events_in_window
    
    return []


async def event_compaction(callback_context: CallbackContext, llm_response: LlmResponse):
    """
    After-model callback that summarizes a sliding window of recent events.

    This callback:
    - uses a fixed window size and overlap to preserve continuity
    - filters out previous compaction events
    - excludes non-textual tool events by default
    - appends a compacted event back to the session
    """
    invocation_context = callback_context._invocation_context
    session = invocation_context.session
    session_service = invocation_context.session_service

    logging.info(f"Event compaction triggered. Session has {len(session.events)} events.")

    events_to_compact = _select_sliding_window_events(
        session=session,
        window=EVENT_COMPACTION_WINDOW,
        overlap=EVENT_COMPACTION_OVERLAP,
    )
    if not events_to_compact:
        logging.info("No recent events to compact.")
        return None

    compaction_event = await summarizer.maybe_summarize_events(
        events=events_to_compact,
    )
    if compaction_event:
        compaction_event.author = callback_context.agent_name or 'model'
        await session_service.append_event(session=session, event=compaction_event)
        logging.info(
            "Appended compaction event for %d events (window=%d overlap=%d).",
            len(events_to_compact),
            EVENT_COMPACTION_WINDOW,
            EVENT_COMPACTION_OVERLAP,
        )
    return None

