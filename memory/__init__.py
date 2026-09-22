"""Memory facade — working + episodic + semantic + consolidator. See docs/memory.md."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from memory.working import WorkingMemory, working_memory
from memory.episodic import EpisodicMemory, episodic, EpisodeType
from memory.semantic import SemanticMemory, semantic
from memory.consolidator import MemoryConsolidator, consolidator


def _infer_source(type: EpisodeType, tags: list[str] | None) -> str:
    tags = tags or []
    if "user_input" in tags:
        return "user_turn"
    if type == "self_observation" or "self_model" in tags:
        return "self_observation"
    if "response" in tags:
        return "keeper_response"
    if "environment" in tags:
        return "environment_event"
    if "tool" in tags:
        return "tool_result"
    if type == "research" or "autonomous" in tags or type == "opinion":
        return "autonomous_artifact"
    return "user_turn"


class MemorySystem:
    """
    Unified memory interface. The rest of the system talks to this —
    it routes to the right layer internally.
    """

    def __init__(
        self,
        working: WorkingMemory | None = None,
        episodic_store: EpisodicMemory | None = None,
        semantic_store: SemanticMemory | None = None,
        consolidator_instance: MemoryConsolidator | None = None,
    ):
        self.working = working or working_memory
        self.episodic = episodic_store or episodic
        self.semantic = semantic_store or semantic
        self.consolidator = consolidator_instance or consolidator

    async def init(self) -> None:
        """Initialise persistent stores (SQLite schema, embeddings warmup)."""
        await self.episodic.init()
        await self.semantic.warmup()

    # ── Store ─────────────────────────────────────────────────────────────

    async def store_episode(
        self,
        user_id: int,
        content: str,
        internal_state: Any = None,
        salience: float = 0.5,
        type: EpisodeType = "event",
        tags: list[str] | None = None,
        source: str | None = None,
    ) -> str:
        """
        Store an experience. Returns the UUID episode_id.
        Encoding strength is no longer taken from InternalState.compute_salience().
        """
        state_snapshot = {}
        if internal_state is not None:
            if hasattr(internal_state, "snapshot"):
                state_snapshot = internal_state.snapshot()
            elif isinstance(internal_state, dict):
                state_snapshot = internal_state

        inferred = source or _infer_source(type, tags)
        return await self.episodic.store(
            user_id=user_id,
            type=type,
            content=content,
            tags=tags,
            importance=salience,
            emotional_weight=0.5,
            state_snapshot=state_snapshot,
            source=inferred,
        )

    def store_working(
        self,
        user_id: int,
        role: str,
        content: str,
        salience: float | None = None,
    ) -> None:
        """Add a message to the active working memory window."""
        self.working.add(user_id, role, content, salience=salience)

    async def store_semantic(
        self,
        user_id: int,
        content: str,
        pattern_type: str = "general",
        confidence: float = 0.7,
        metadata: dict | None = None,
    ) -> str:
        """Store a crystallised pattern directly into semantic memory."""
        return await self.semantic.store(
            user_id=user_id,
            content=content,
            pattern_type=pattern_type,
            confidence=confidence,
            metadata=metadata,
        )

    # ── Recall ────────────────────────────────────────────────────────────

    async def recall(
        self,
        user_id: int,
        query: str,
        current_state: Any = None,
        n_episodic: int = 5,
        n_semantic: int = 4,
    ) -> dict:
        """
        Multi-layer recall. Returns results from both episodic and
        semantic memory, plus the current working memory window.

        The current internal state biases episodic retrieval — e.g.
        high-arousal states surface high-emotion memories.
        """
        state_bias = None
        if current_state is not None:
            if hasattr(current_state, "snapshot"):
                state_bias = current_state.snapshot()
            elif isinstance(current_state, dict):
                state_bias = current_state

        episodic_results = await self.episodic.recall(
            user_id, limit=n_episodic, state_bias=state_bias
        )
        semantic_results = await self.semantic.search(
            user_id, query, n_results=n_semantic
        )
        working_items = self.working.get_items(user_id)

        return {
            "working": working_items,
            "episodic": episodic_results,
            "semantic": semantic_results,
        }

    # ── Consolidation ─────────────────────────────────────────────────────

    async def consolidate(self, user_id: int, since_hours: int = 24) -> dict:
        """Run one consolidation cycle: extract patterns + apply decay."""
        self.working.idle_tick(user_id)
        return await self.consolidator.run_cycle(user_id, since_hours)

    # ── Prompt building ───────────────────────────────────────────────────

    async def build_memory_context(
        self,
        user_id: int,
        query: str,
        now: datetime | None = None,
    ) -> tuple[str, dict]:
        """
        Build a complete memory context block for the system prompt.
        Combines episodic + semantic. Does not bump recall_count.
        """
        parts = []
        exclude = [item.content for item in self.working.get_items(user_id)]
        ep_block, used, stats = await self.episodic.format_for_prompt(
            user_id, exclude_texts=exclude, now=now,
        )
        if ep_block:
            parts.append(ep_block)

        sem_block = await self.semantic.format_for_prompt(
            user_id, query, exclude_texts=exclude, now=now,
        )
        if sem_block:
            parts.append(sem_block)

        # Injection is not recall — only used_count moves.
        if used:
            await self.episodic.increment_counters([], used)
        text = "\n\n".join(parts) if parts else ""
        stats = dict(stats)
        stats["mem_chars"] = len(text)
        return text, stats

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def clear_working(self, user_id: int) -> None:
        self.working.clear(user_id)


# Singleton
memory_system = MemorySystem()
