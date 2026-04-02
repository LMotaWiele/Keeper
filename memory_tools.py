"""
Memory tools — let the agent explicitly read/write its own memory layers.

These are LangChain @tool functions the LangGraph node can call during a turn.
"""
from __future__ import annotations

from langchain_core.tools import tool

from memory.short_term import short_term
from memory.mid_term import mid_term
from memory.long_term import long_term


@tool
async def save_fact(user_id: int, fact: str, tags: list[str] | None = None) -> str:
    """
    Save an important fact about the user to mid-term memory.
    Use for things like name, job, preferences, important life events.
    """
    ep_id = await mid_term.store(user_id, "fact", fact, tags=tags, importance=0.8)
    return f"Saved fact #{ep_id}: {fact}"


@tool
async def save_preference(user_id: int, preference: str) -> str:
    """Save a user preference (likes, dislikes, communication style) to mid-term memory."""
    ep_id = await mid_term.store(user_id, "preference", preference, importance=0.7)
    return f"Saved preference #{ep_id}"


@tool
async def save_long_term_memory(user_id: int, content: str, tags: dict | None = None) -> str:
    """
    Embed and save a piece of text to long-term vector memory.
    Use for conversation summaries, important discussions, documents the user shared.
    """
    doc_id = await long_term.store(user_id, content, metadata=tags)
    return f"Stored in long-term memory (id={doc_id})"


@tool
async def search_long_term_memory(user_id: int, query: str) -> str:
    """
    Search long-term memory for anything semantically related to the query.
    Use before answering questions that might reference past conversations.
    """
    memories = await long_term.search(user_id, query, n_results=5)
    if not memories:
        return "No relevant long-term memories found."
    lines = [f"- (relevance {m['relevance']}) {m['content']}" for m in memories]
    return "Relevant memories:\n" + "\n".join(lines)


@tool
async def recall_facts(user_id: int) -> str:
    """Retrieve all stored facts and preferences about the user from mid-term memory."""
    facts = await mid_term.get_facts(user_id)
    prefs = await mid_term.get_preferences(user_id)
    if not facts and not prefs:
        return "No facts or preferences stored yet."
    lines = []
    for f in facts:
        lines.append(f"[fact] {f['content']}")
    for p in prefs:
        lines.append(f"[preference] {p['content']}")
    return "\n".join(lines)


MEMORY_TOOLS = [save_fact, save_preference, save_long_term_memory, search_long_term_memory, recall_facts]
