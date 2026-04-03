"""
Memory tools — let the agent explicitly read/write its own memory layers.

Updated to use the new episodic + semantic memory system.
The agent can now store facts with emotional weight, search semantic
patterns, and recall with state-aware retrieval.
"""
from __future__ import annotations

from langchain_core.tools import tool

from memory.episodic import episodic
from memory.semantic import semantic


@tool
async def save_fact(user_id: int, fact: str, tags: list[str] | None = None) -> str:
    """
    Save an important fact about the user to episodic memory.
    Use for things like name, job, preferences, important life events.
    Facts are stored with high importance so they resist decay.
    """
    ep_id = await episodic.store(
        user_id, "fact", fact, tags=tags, importance=0.8, emotional_weight=0.6
    )
    return f"Saved fact #{ep_id}: {fact}"


@tool
async def save_preference(user_id: int, preference: str) -> str:
    """Save a user preference (likes, dislikes, communication style) to episodic memory."""
    ep_id = await episodic.store(
        user_id, "preference", preference, importance=0.7, emotional_weight=0.5
    )
    return f"Saved preference #{ep_id}"


@tool
async def save_long_term_memory(user_id: int, content: str, tags: dict | None = None) -> str:
    """
    Store a crystallised pattern or summary in semantic (long-term) memory.
    Use for conversation summaries, important insights, or knowledge
    the user has shared that should persist indefinitely.
    """
    doc_id = await semantic.store(
        user_id, content, metadata=tags, pattern_type="agent_stored", confidence=0.8
    )
    return f"Stored in semantic memory (id={doc_id})"


@tool
async def search_long_term_memory(user_id: int, query: str) -> str:
    """
    Search semantic memory for anything related to the query.
    Use before answering questions that might reference past conversations
    or established knowledge about the user.
    """
    memories = await semantic.search(user_id, query, n_results=5)
    if not memories:
        return "No relevant semantic memories found."
    lines = [
        f"- (relevance {m['relevance']:.2f}, confidence {m['confidence']:.1f}) {m['content']}"
        for m in memories
    ]
    return "Relevant memories:\n" + "\n".join(lines)


@tool
async def recall_facts(user_id: int) -> str:
    """Retrieve all stored facts and preferences about the user from episodic memory."""
    facts = await episodic.get_facts(user_id)
    prefs = await episodic.get_preferences(user_id)
    if not facts and not prefs:
        return "No facts or preferences stored yet."
    lines = []
    for f in facts:
        strength = f.get("effective_strength", 1.0)
        fade = " (fading)" if strength < 0.5 else ""
        lines.append(f"[fact] {f['content']}{fade}")
    for p in prefs:
        lines.append(f"[preference] {p['content']}")
    return "\n".join(lines)


MEMORY_TOOLS = [
    save_fact,
    save_preference,
    save_long_term_memory,
    search_long_term_memory,
    recall_facts,
]