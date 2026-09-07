"""LangGraph agent — reason ⇄ tools. Context is built by the companion loop."""
from __future__ import annotations

import logging
from typing import Annotated, Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from config.settings import config
from core.llm import Tier, get_llm
from tools import ALL_TOOLS

log = logging.getLogger(__name__)


class CompanionState(TypedDict):
    user_id: int
    messages: Annotated[list[BaseMessage], add_messages]
    system_prompt: str
    tool_calls_pending: bool
    response_text: str
    tool_calls_made: int
    tier: str
    forced_answer: bool


_GRAPH_CACHE: dict[str, Any] = {}


def _llm_for_tier(tier: Tier):
    """Build (and cache) a tool-bound chat model for a conversation tier."""
    if tier.value not in _GRAPH_CACHE:
        model = {
            Tier.HIGH: config.model_high,
            Tier.MID: config.model_mid,
            Tier.LOW: config.model_low,
        }[tier]
        llm = get_llm("conversation", model=model)
        _GRAPH_CACHE[tier.value] = llm.bind_tools(ALL_TOOLS)
    return _GRAPH_CACHE[tier.value]


def message_text(content: Any) -> str:
    """Flatten string or content-block lists to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
            elif hasattr(block, "text"):
                parts.append(getattr(block, "text") or "")
        return "\n".join(p for p in parts if p)
    return ""


def _tool_accepts_user_id(tool: Any) -> bool:
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return False
    fields = getattr(schema, "model_fields", None) or getattr(schema, "__fields__", None) or {}
    return "user_id" in fields


async def load_context(state: CompanionState) -> dict[str, Any]:
    """Passthrough — runner already filled system_prompt and messages."""
    return {}


async def reason(state: CompanionState) -> dict[str, Any]:
    """One LLM step: reply and/or request tools."""
    try:
        tier = Tier(state.get("tier") or Tier.HIGH.value)
    except ValueError:
        tier = Tier.HIGH
    llm_with_tools = _llm_for_tier(tier)
    messages = [SystemMessage(content=state["system_prompt"])] + state["messages"]
    response = await llm_with_tools.ainvoke(messages)

    has_tool_calls = bool(getattr(response, "tool_calls", None))
    tool_count = state.get("tool_calls_made", 0)
    if has_tool_calls:
        tool_count += len(response.tool_calls)

    return {
        "messages": [response],
        "tool_calls_pending": has_tool_calls,
        "response_text": "" if has_tool_calls else message_text(response.content),
        "tool_calls_made": tool_count,
    }


async def tool_executor(state: CompanionState) -> dict[str, Any]:
    """Run requested tools. Inject user_id when the schema expects it."""
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
                args = dict(tc.get("args") or {})
                if _tool_accepts_user_id(tool):
                    args.setdefault("user_id", state["user_id"])
                result = await tool.ainvoke(args)
            except Exception as exc:
                result = f"Tool error: {exc}"

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )

    return {"messages": tool_messages, "tool_calls_pending": False}


async def force_answer(state: CompanionState) -> dict[str, Any]:
    """Tool budget exhausted — answer from what is already in the transcript."""
    llm = get_llm("conversation")          # unbound: no tools
    messages = (
        [SystemMessage(content=state["system_prompt"])]
        + state["messages"]
        + [HumanMessage(content=(
            "[system] Tool budget exhausted for this turn. Answer the user now, "
            "using the tool results already in this transcript. State plainly what "
            "you could not determine. Do not describe what you still intend to read."
        ))]
    )
    response = await llm.ainvoke(messages)
    made = state.get("tool_calls_made", 0)
    log.warning("FORCED ANSWER after %d tool calls", made)
    return {
        "messages": [response],
        "response_text": message_text(response.content),
        "forced_answer": True,
        "tool_calls_pending": False,
    }


async def finalize(state: CompanionState) -> dict[str, Any]:
    """Terminal node — runner reads response_text after the graph ends."""
    if state.get("tool_calls_pending") and not state.get("forced_answer"):
        log.error(
            "finalize reached with tool_calls_pending and no forced_answer "
            "(made=%s) — graph must not end here",
            state.get("tool_calls_made"),
        )
    return {}


MAX_TOOL_CALLS = 16


def route_after_reason(state: CompanionState) -> str:
    if state["tool_calls_pending"]:
        if state.get("tool_calls_made", 0) >= MAX_TOOL_CALLS:
            return "force_answer"
        return "tool_executor"
    return "finalize"


def build_graph() -> Any:
    g = StateGraph(CompanionState)
    g.add_node("load_context", load_context)
    g.add_node("reason", reason)
    g.add_node("tool_executor", tool_executor)
    g.add_node("force_answer", force_answer)
    g.add_node("finalize", finalize)
    g.set_entry_point("load_context")
    g.add_edge("load_context", "reason")
    g.add_conditional_edges("reason", route_after_reason, {
        "tool_executor": "tool_executor",
        "force_answer": "force_answer",
        "finalize": "finalize",
    })
    g.add_edge("tool_executor", "reason")
    g.add_edge("force_answer", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


companion_graph = build_graph()
