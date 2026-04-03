"""
LangGraph agent — the companion's brain.

Updated to integrate with the ConsciousArchitecture. The graph no longer
builds its own context — it receives a rich context dict from
companion.process() that includes all five pillars.

Graph topology:
  ┌─────────────┐
  │  load_context│  ← Receives pre-built context from the companion loop
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
  │   finalize  │  ← Signals the runner to call post_process
  └─────────────┘
"""
from __future__ import annotations

from typing import Annotated, Any

from langchain_anthropic import ChatAnthropic
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
from tools import ALL_TOOLS


# ── State ─────────────────────────────────────────────────────────────────

class CompanionState(TypedDict):
    user_id: int
    messages: Annotated[list[BaseMessage], add_messages]
    system_prompt: str
    tool_calls_pending: bool
    response_text: str               # final response for post_process
    tool_calls_made: int              # count for post_process


# ── LLM ───────────────────────────────────────────────────────────────────

llm = ChatAnthropic(
    model=config.llm_model,
    anthropic_api_key=config.anthropic_api_key,
    temperature=0.7,
    max_tokens=2048,
)

llm_with_tools = llm.bind_tools(ALL_TOOLS)


# ── Nodes ─────────────────────────────────────────────────────────────────

async def load_context(state: CompanionState) -> dict[str, Any]:
    """
    Context is already built by the companion loop and passed in
    via the initial state. This node just confirms it's present.
    """
    # system_prompt and messages are set by the runner before invocation
    return {}


async def reason(state: CompanionState) -> dict[str, Any]:
    """Core LLM reasoning step."""
    messages = [SystemMessage(content=state["system_prompt"])] + state["messages"]

    response = await llm_with_tools.ainvoke(messages)

    has_tool_calls = bool(getattr(response, "tool_calls", None))
    tool_count = state.get("tool_calls_made", 0)
    if has_tool_calls:
        tool_count += len(response.tool_calls)

    # Track the latest text response
    response_text = response.content if isinstance(response.content, str) else ""

    return {
        "messages": [response],
        "tool_calls_pending": has_tool_calls,
        "response_text": response_text,
        "tool_calls_made": tool_count,
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


async def finalize(state: CompanionState) -> dict[str, Any]:
    """
    Final node — nothing to do here. The runner reads response_text
    from the state and calls companion.post_process() after the graph
    completes.

    This replaces the old consolidate node. Consolidation now happens
    in the companion's background loops instead of blocking every response.
    """
    return {}


# ── Routing ───────────────────────────────────────────────────────────────

def route_after_reason(state: CompanionState) -> str:
    return "tool_executor" if state["tool_calls_pending"] else "finalize"


# ── Graph ─────────────────────────────────────────────────────────────────

def build_graph() -> Any:
    g = StateGraph(CompanionState)

    g.add_node("load_context", load_context)
    g.add_node("reason", reason)
    g.add_node("tool_executor", tool_executor)
    g.add_node("finalize", finalize)

    g.set_entry_point("load_context")
    g.add_edge("load_context", "reason")
    g.add_conditional_edges("reason", route_after_reason, {
        "tool_executor": "tool_executor",
        "finalize": "finalize",
    })
    g.add_edge("tool_executor", "reason")
    g.add_edge("finalize", END)

    return g.compile()


companion_graph = build_graph()