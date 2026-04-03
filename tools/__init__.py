"""
Tools available to the LangGraph agent.
 
Add new tools here — the agent picks them up automatically.
"""
from tools.memory_tools import MEMORY_TOOLS
from tools.web_search import web_search_tool
 
ALL_TOOLS = [
    web_search_tool,
    *MEMORY_TOOLS,
]
 
__all__ = ["ALL_TOOLS"]