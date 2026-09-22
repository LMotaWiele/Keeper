"""Memory tools — explicit fact/preference/semantic read-write for the agent."""
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
    from core.timeutil import memory_time_prefix, parse_iso, utcnow
    from memory.timebind import resolve_event_time

    now = utcnow()
    lines = []
    for m in memories:
        meta = m.get("metadata") or {}
        raw = meta.get("event_at") or meta.get("stored_at") or ""
        try:
            event_dt = parse_iso(raw) if raw else now
        except Exception:
            event_dt, _ = resolve_event_time(m["content"], now)
        prefix = memory_time_prefix(event_dt, now)
        lines.append(
            f"- (relevance {m['relevance']:.2f}, confidence {m['confidence']:.1f}) "
            f"{prefix} {m['content']}"
        )
    return "Relevant memories:\n" + "\n".join(lines)


@tool
async def recall_facts(user_id: int) -> str:
    """Retrieve all stored facts and preferences about the user from episodic memory."""
    facts = await episodic.get_facts(user_id)
    prefs = await episodic.get_preferences(user_id)
    if not facts and not prefs:
        return "No facts or preferences stored yet."
    lines = []
    ids: list[str] = []
    for f in facts:
        strength = f.get("effective_strength", 1.0)
        fade = " (fading)" if strength < 0.5 else ""
        _, prefix = episodic._event_fields(f)
        lines.append(f"[fact] {prefix} {f['content']}{fade}")
        ids.append(f.get("episode_id") or str(f["id"]))
    for p in prefs:
        _, prefix = episodic._event_fields(p)
        lines.append(f"[preference] {prefix} {p['content']}")
        ids.append(p.get("episode_id") or str(p["id"]))
    if ids:
        await episodic.increment_counters(ids, [])
    return "\n".join(lines)


MEMORY_TOOLS = [
    save_fact,
    save_preference,
    save_long_term_memory,
    search_long_term_memory,
    recall_facts,
]