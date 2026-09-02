"""Tools bound into the LangGraph agent."""
from tools.introspection import INTROSPECTION_TOOLS
from tools.memory_tools import MEMORY_TOOLS
from tools.web_search import web_search

ALL_TOOLS = [
    web_search,
    *MEMORY_TOOLS,
    *INTROSPECTION_TOOLS,
]

__all__ = ["ALL_TOOLS"]
