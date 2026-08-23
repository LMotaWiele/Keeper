"""Web search — Tavily, gated by the daily search budget."""
from __future__ import annotations

import os
from typing import Any

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

from config.settings import config

os.environ["TAVILY_API_KEY"] = config.tavily_api_key

_tavily = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
    include_images=False,
    name="web_search",
    description=(
        "Search the web for current information. Use when you need recent news, "
        "facts you're uncertain about, or anything that might have changed since "
        "your training. Returns titles, URLs, and relevant snippets."
    ),
)


async def run_search(query: str) -> Any:
    """Run a Tavily search if the daily curiosity-scaled quota allows."""
    from core.loop import companion

    if not companion.search_budget.can_search(companion.state.curiosity):
        return "Search budget exhausted for today. Use existing knowledge instead."
    result = await _tavily.ainvoke({"query": query})
    companion.search_budget.record_search()
    return result


@tool
async def web_search(query: str) -> str:
    """Search the web for current information. Use for news, facts you're unsure about, or anything that may have changed since training."""
    return str(await run_search(query))


# Back-compat alias used by older imports.
web_search_tool = web_search
