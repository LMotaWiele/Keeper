"""
Web search tool — powered by Tavily.

The companion uses this when it needs current information it doesn't already
know. Follows the SOUL.md principle: be resourceful before asking.
"""
from __future__ import annotations

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

from config.settings import config

import os
os.environ["TAVILY_API_KEY"] = config.tavily_api_key


# LangChain-compatible Tavily tool (returns top-k results with snippets + URLs)
web_search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,       # Tavily's own answer synthesis
    include_raw_content=False,
    include_images=False,
    name="web_search",
    description=(
        "Search the web for current information. Use when you need recent news, "
        "facts you're uncertain about, or anything that might have changed since "
        "your training. Returns titles, URLs, and relevant snippets."
    ),
)
