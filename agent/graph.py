"""LangGraph agent — reason ⇄ tools. Context is built by the companion loop."""
from __future__ import annotations

from typing import Annotated, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from config.settings import config
from tools import ALL_TOOLS


class CompanionState(TypedDict):
    user_id: int
    messages: Annotated[list[BaseMessage], add_messages]
    system_prompt: str
    tool_calls_pending: bool
    response_text: str
    tool_calls_made: int


llm = ChatAnthropic(
    model=config.llm_model,
    anthropic_api_key=config.anthropic_api_key,
    temperature=0.7,
    max_tokens=2048,
)

llm_with_tools = llm.bind_tools(ALL_TOOLS)


def message_text(content: Any) -> str:
    """Flatten Anthropic string or content-block lists to plain text."""
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
    messages = [SystemMessage(content=state["system_prompt"])] + state["messages"]
    response = await llm_with_tools.ainvoke(messages)

    has_tool_calls = bool(getattr(response, "tool_calls", None))
    tool_count = state.get("tool_calls_made", 0)
    if has_tool_calls:
        tool_count += len(response.tool_calls)

    return {
        "messages": [response],
        "tool_calls_pending": has_tool_calls,
        "response_text": message_text(response.content),
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


async def finalize(state: CompanionState) -> dict[str, Any]:
    """Terminal node — runner reads response_text after the graph ends."""
    return {}


MAX_TOOL_CALLS = 8


def route_after_reason(state: CompanionState) -> str:
    if state["tool_calls_pending"] and state.get("tool_calls_made", 0) < MAX_TOOL_CALLS:
        return "tool_executor"
    return "finalize"


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
