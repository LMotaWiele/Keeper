"""
Agent runner — bridges the companion loop and the LangGraph agent.

Flow:
  1. companion.process(user_id, text) → builds context from all 5 pillars
  2. LangGraph agent runs with that context → produces a response
  3. companion.post_process() → updates all pillars with what happened

This is the single entry point for all message handling.
"""
from __future__ import annotations

import logging
import time

from langchain_core.messages import HumanMessage

from agent.graph import companion_graph
from core.loop import companion

log = logging.getLogger(__name__)


async def run_agent(user_id: int, user_text: str) -> str:
    """
    Process a user message through the full architecture.

    Returns the assistant's response text.
    """
    start = time.monotonic()

    # 1. Companion builds context from all five pillars
    context = await companion.process(user_id, user_text)

    # 2. Run the LangGraph agent with that context
    initial_state = {
        "user_id": user_id,
        "messages": [HumanMessage(content=user_text)],
        "system_prompt": context["system_prompt"],
        "tool_calls_pending": False,
        "response_text": "",
        "tool_calls_made": 0,
    }

    result = await companion_graph.ainvoke(initial_state)

    response_text = result.get("response_text", "")
    tool_calls_made = result.get("tool_calls_made", 0)

    # Fallback: if response_text is empty, grab the last AI message content
    if not response_text:
        from langchain_core.messages import AIMessage
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                response_text = msg.content
                break

    if not response_text:
        response_text = "I'm not sure what to say right now."

    elapsed_ms = (time.monotonic() - start) * 1000

    # 3. Companion updates all pillars with what happened
    await companion.post_process(
        user_id=user_id,
        user_text=user_text,
        response_text=response_text,
        tool_calls_made=tool_calls_made,
        processing_time_ms=elapsed_ms,
    )

    log.info(
        "Agent completed — user=%s, tools=%d, time=%.0fms, engagement=%s",
        user_id, tool_calls_made, elapsed_ms, context.get("engagement_level", "?"),
    )

    return response_text