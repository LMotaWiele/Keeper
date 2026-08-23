"""Memory facade — working + episodic + semantic + consolidator. See docs/memory.md."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from memory.working import WorkingMemory, working_memory
from memory.episodic import EpisodicMemory, episodic, EpisodeType
from memory.semantic import SemanticMemory, semantic
from memory.consolidator import MemoryConsolidator, consolidator


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
        """Initialise persistent stores (SQLite schema, etc.)."""
        await self.episodic.init()

    # ── Store ─────────────────────────────────────────────────────────────

    async def store_episode(
        self,
        user_id: int,
        content: str,
        internal_state: Any = None,
        salience: float = 0.5,
        type: EpisodeType = "event",
        tags: list[str] | None = None,
    ) -> int:
        """
        Store an experience. If an internal state is provided,
        its arousal/valence are used to compute emotional weight
        and the full state is snapshotted.
        """
        emotional_weight = 0.5
        state_snapshot = {}

        if internal_state is not None:
            # internal_state is expected to have compute_salience() and snapshot()
            # but we gracefully handle dicts or missing methods
            if hasattr(internal_state, "compute_salience"):
                emotional_weight = internal_state.compute_salience()
            elif isinstance(internal_state, dict):
                arousal = internal_state.get("arousal", 0.5)
                valence = internal_state.get("valence", 0.5)
                emotional_weight = (arousal + abs(valence - 0.5)) / 1.5

            if hasattr(internal_state, "snapshot"):
                state_snapshot = internal_state.snapshot()
            elif isinstance(internal_state, dict):
                state_snapshot = internal_state

        return await self.episodic.store(
            user_id=user_id,
            type=type,
            content=content,
            tags=tags,
            importance=salience,
            emotional_weight=emotional_weight,
            state_snapshot=state_snapshot,
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
        return await self.consolidator.run_cycle(user_id, since_hours)

    # ── Prompt building ───────────────────────────────────────────────────

    async def build_memory_context(self, user_id: int, query: str) -> str:
        """
        Build a complete memory context block for the system prompt.
        Combines all three layers into a coherent narrative.
        """
        parts = []

        # Episodic — what I remember
        ep_block = await self.episodic.format_for_prompt(user_id)
        if ep_block:
            parts.append(ep_block)

        # Semantic — what I've learned
        sem_block = await self.semantic.format_for_prompt(user_id, query)
        if sem_block:
            parts.append(sem_block)

        return "\n\n".join(parts) if parts else ""

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def clear_working(self, user_id: int) -> None:
        self.working.clear(user_id)


# Singleton
memory_system = MemorySystem()
