"""
LangGraph agent — the companion's brain.

Graph topology:
  ┌─────────────┐
  │  load_context│  ← Reads SOUL.md + all memory layers into state
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │    reason   │  ← LLM decides: respond directly OR call a tool
  └──────┬──────┘
         │
    ┌────▼────┐
    │ tool?   │─── yes ──► tool_executor ──► reason (loop)
    └────┬────┘
         │ no
  ┌──────▼──────┐
  │  consolidate│  ← After responding, summarise & update mid/long-term memory
  └─────────────┘
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from config.settings import config
from memory.mid_term import mid_term
from memory.long_term import long_term
from memory.short_term import short_term
from tools import ALL_TOOLS


# ── State ─────────────────────────────────────────────────────────────────────

class CompanionState(TypedDict):
    user_id: int
    messages: Annotated[list[BaseMessage], add_messages]
    soul: str                    # contents of SOUL.md
    mid_term_context: str        # formatted mid-term memory block
    long_term_context: str       # formatted long-term memory block
    tool_calls_pending: bool


# ── LLM ───────────────────────────────────────────────────────────────────────

llm = ChatAnthropic(
    model=config.llm_model,
    anthropic_api_key=config.anthropic_api_key,
    temperature=0.7,
    max_tokens=2048,
)

llm_with_tools = llm.bind_tools(ALL_TOOLS)


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def load_context(state: CompanionState) -> dict[str, Any]:
    """Load SOUL.md and memory context into state."""
    user_id = state["user_id"]

    # Soul
    soul_path = config.soul_file_path
    soul = soul_path.read_text() if soul_path.exists() else "# No SOUL.md found."

    # Last message as the query for long-term search
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )

    mid_ctx = await mid_term.format_for_prompt(user_id)
    lt_ctx = await long_term.format_for_prompt(user_id, last_human)

    return {
        "soul": soul,
        "mid_term_context": mid_ctx,
        "long_term_context": lt_ctx,
    }


def _build_system_prompt(state: CompanionState) -> str:
    parts = [state["soul"]]

    if state.get("mid_term_context"):
        parts.append(state["mid_term_context"])

    if state.get("long_term_context"):
        parts.append(state["long_term_context"])

    parts.append(
        "\n---\n"
        "You have tools available: web search and memory management. "
        "Use web_search when you need current information. "
        "Use memory tools to save important facts/preferences/summaries proactively — "
        "your future self depends on them.\n"
        "Reply naturally, as a companion would. Never break character."
    )

    return "\n\n".join(parts)


async def reason(state: CompanionState) -> dict[str, Any]:
    """Core LLM reasoning step."""
    system_prompt = _build_system_prompt(state)
    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    response = await llm_with_tools.ainvoke(messages)

    has_tool_calls = bool(getattr(response, "tool_calls", None))
    return {
        "messages": [response],
        "tool_calls_pending": has_tool_calls,
    }


async def tool_executor(state: CompanionState) -> dict[str, Any]:
    """Execute any tool calls the LLM requested."""
    from langchain_core.tools import BaseTool

    last_ai: AIMessage = next(
        m for m in reversed(state["messages"]) if isinstance(m, AIMessage)
    )

    tool_map: dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}
    tool_messages: list[ToolMessage] = []

    for tc in last_ai.tool_calls:
        tool = tool_map.get(tc["name"])
        if tool is None:
            result = f"Unknown tool: {tc['name']}"
        else:
            try:
                # Inject user_id into memory tools automatically
                args = tc["args"].copy()
                if "user_id" in tool.args_schema.model_fields:
                    args.setdefault("user_id", state["user_id"])
                result = await tool.ainvoke(args)
            except Exception as exc:
                result = f"Tool error: {exc}"

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )

    return {"messages": tool_messages, "tool_calls_pending": False}


async def consolidate(state: CompanionState) -> dict[str, Any]:
    """
    After the final reply, ask the LLM to extract any new facts/preferences
    worth saving, and store a session summary in long-term memory.
    """
    user_id = state["user_id"]

    # Build a brief conversation transcript
    transcript_parts = []
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            transcript_parts.append(f"User: {m.content}")
        elif isinstance(m, AIMessage) and m.content:
            transcript_parts.append(f"Assistant: {m.content}")
    transcript = "\n".join(transcript_parts[-10:])  # last 10 turns

    extraction_prompt = (
        "You are a memory curator. Given this conversation excerpt, output a JSON object with:\n"
        '- "facts": list of factual statements about the user worth remembering\n'
        '- "preferences": list of preferences the user expressed\n'
        '- "summary": a 1-2 sentence summary of this exchange\n'
        "Output ONLY valid JSON, no other text.\n\n"
        f"Conversation:\n{transcript}"
    )

    try:
        extraction = await llm.ainvoke([HumanMessage(content=extraction_prompt)])
        data = json.loads(extraction.content)

        for fact in data.get("facts", []):
            await mid_term.store(user_id, "fact", fact, importance=0.7)

        for pref in data.get("preferences", []):
            await mid_term.store(user_id, "preference", pref, importance=0.6)

        summary = data.get("summary", "")
        if summary:
            await long_term.store(user_id, summary, metadata={"type": "session_summary"})

    except Exception:
        pass  # Consolidation is best-effort

    return {}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_reason(state: CompanionState) -> str:
    return "tool_executor" if state["tool_calls_pending"] else "consolidate"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph() -> Any:
    g = StateGraph(CompanionState)

    g.add_node("load_context", load_context)
    g.add_node("reason", reason)
    g.add_node("tool_executor", tool_executor)
    g.add_node("consolidate", consolidate)

    g.set_entry_point("load_context")
    g.add_edge("load_context", "reason")
    g.add_conditional_edges("reason", route_after_reason, {
        "tool_executor": "tool_executor",
        "consolidate": "consolidate",
    })
    g.add_edge("tool_executor", "reason")   # loop back after tool use
    g.add_edge("consolidate", END)

    return g.compile()


companion_graph = build_graph()
