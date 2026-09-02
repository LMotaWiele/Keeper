"""Agent runner — process → graph → post_process. Single entry for user turns."""
from __future__ import annotations

import logging
import time

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agent.graph import companion_graph, message_text
from core.loop import companion

log = logging.getLogger(__name__)


def _as_messages(raw: list) -> list[BaseMessage]:
    """Coerce working-memory dicts into LangChain messages."""
    out: list[BaseMessage] = []
    for m in raw or []:
        if isinstance(m, BaseMessage):
            out.append(m)
            continue
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "assistant":
            out.append(AIMessage(content=content))
        elif role == "system":
            out.append(SystemMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out


def _usage_from_messages(messages: list) -> tuple[int, int]:
    """Sum input/output tokens from AI message metadata."""
    inp = out = 0
    for msg in messages or []:
        usage = getattr(msg, "usage_metadata", None) or {}
        if not usage:
            meta = getattr(msg, "response_metadata", None) or {}
            usage = meta.get("usage") or meta.get("token_usage") or {}
        inp += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        out += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    return inp, out


async def run_agent(user_id: int, user_text: str) -> str:
    """Run one user message through the architecture. Returns reply text."""
    start = time.monotonic()

    context = await companion.process(user_id, user_text)

    messages = _as_messages(context.get("messages") or [])
    if not messages or message_text(messages[-1].content) != user_text:
        messages.append(HumanMessage(content=user_text))

    initial_state = {
        "user_id": user_id,
        "messages": messages,
        "system_prompt": context["system_prompt"],
        "tool_calls_pending": False,
        "response_text": "",
        "tool_calls_made": 0,
        "tier": companion.conversation_tier().value,
    }

    result = await companion_graph.ainvoke(initial_state)

    response_text = result.get("response_text", "") or ""
    if not response_text:
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage):
                response_text = message_text(msg.content)
                if response_text:
                    break

    if not response_text:
        response_text = "I'm not sure what to say right now."

    elapsed_ms = (time.monotonic() - start) * 1000
    input_tokens, output_tokens = _usage_from_messages(result.get("messages", []))

    await companion.post_process(
        user_id=user_id,
        user_text=user_text,
        response_text=response_text,
        tool_calls_made=result.get("tool_calls_made", 0),
        processing_time_ms=elapsed_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    log.info(
        "Agent completed — user=%s, tools=%d, time=%.0fms, engagement=%s",
        user_id, result.get("tool_calls_made", 0), elapsed_ms,
        context.get("engagement_level", "?"),
    )

    return response_text
