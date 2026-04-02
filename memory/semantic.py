"""
Semantic memory — abstracted knowledge derived from experience.

Replaces the old long_term.py. Still backed by ChromaDB for vector search,
but now serves a distinct role: this is where *patterns* live, not raw
experiences. The consolidator extracts meaning from episodic memory and
crystallises it here.

Think of episodic memory as "I remember that conversation about cooking"
and semantic memory as "this person is a confident cook who prefers
Mediterranean flavours."

Key differences from the old long_term:
  - Stores crystallised patterns, not raw conversation dumps
  - Metadata tracks source episodes and confidence scores
  - Supports incremental reinforcement (same pattern seen again → stronger)
  - Patterns can be invalidated when contradicted by new episodes
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings

from config.settings import config


def _content_id(content: str) -> str:
    """Deterministic ID from content — same text always maps to same doc."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class SemanticMemory:
    """
    Persistent vector store for crystallised knowledge.

    Each document represents a pattern or piece of abstracted knowledge
    rather than a raw conversation excerpt. Metadata tracks provenance
    (which episodes it was derived from) and confidence.
    """

    def __init__(self, db_path: Path | None = None):
        self.db_path = str(db_path or config.chroma_db_path)
        self._client: chromadb.PersistentClient | None = None
        self._embeddings: OpenAIEmbeddings | None = None

    # ── Internals ─────────────────────────────────────────────────────────

    def _get_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def _get_embeddings(self) -> OpenAIEmbeddings:
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=config.embedding_model,
                openai_api_key=config.openai_api_key,
            )
        return self._embeddings

    def _collection(self, user_id: int) -> chromadb.Collection:
        return self._get_client().get_or_create_collection(
            name=f"semantic_user_{user_id}",
            metadata={"hnsw:space": "cosine"},
        )

    # ── Store ─────────────────────────────────────────────────────────────

    async def store(
        self,
        user_id: int,
        content: str,
        metadata: dict | None = None,
        source_episode_ids: list[int] | None = None,
        pattern_type: str = "general",
        confidence: float = 0.7,
    ) -> str:
        """
        Embed and store a piece of crystallised knowledge.

        If the exact same content already exists (by content hash),
        reinforce it instead of duplicating — bump the confidence
        and merge source episode IDs.
        """
        doc_id = _content_id(content)
        col = self._collection(user_id)

        # Check if this pattern already exists
        existing = col.get(ids=[doc_id], include=["metadatas"])
        if existing and existing["ids"]:
            # Reinforce — bump confidence, merge sources
            old_meta = existing["metadatas"][0] if existing["metadatas"] else {}
            old_confidence = old_meta.get("confidence", 0.5)
            old_sources = old_meta.get("source_episode_ids", [])
            if isinstance(old_sources, str):
                import json
                old_sources = json.loads(old_sources)

            new_confidence = min(1.0, old_confidence + 0.1)
            merged_sources = list(set(old_sources + (source_episode_ids or [])))

            col.update(
                ids=[doc_id],
                metadatas=[{
                    **old_meta,
                    "confidence": new_confidence,
                    "source_episode_ids": str(merged_sources),
                    "reinforced_at": datetime.utcnow().isoformat(),
                    "reinforcement_count": old_meta.get("reinforcement_count", 0) + 1,
                }],
            )
            return doc_id

        # New pattern — embed and store
        embedding = self._get_embeddings().embed_query(content)
        col.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{
                "stored_at": datetime.utcnow().isoformat(),
                "pattern_type": pattern_type,
                "confidence": confidence,
                "source_episode_ids": str(source_episode_ids or []),
                "reinforcement_count": 0,
                "invalidated": "false",
                **(metadata or {}),
            }],
        )
        return doc_id

    # ── Search ────────────────────────────────────────────────────────────

    async def search(
        self,
        user_id: int,
        query: str,
        n_results: int = 5,
        min_confidence: float = 0.3,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Semantic search — returns the closest patterns to the query.
        Filters out invalidated patterns and low-confidence ones.
        """
        col = self._collection(user_id)
        if col.count() == 0:
            return []

        # Build where filter
        where_filter = where or {}

        embedding = self._get_embeddings().embed_query(query)
        results = col.query(
            query_embeddings=[embedding],
            n_results=min(n_results * 2, col.count()),  # over-fetch to filter
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"],
        )

        memories = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            confidence = meta.get("confidence", 0.5)
            invalidated = meta.get("invalidated", "false") == "true"

            if invalidated or confidence < min_confidence:
                continue

            memories.append({
                "content": doc,
                "metadata": meta,
                "confidence": confidence,
                "relevance": round(1 - dist, 3),
            })

        # Sort by combined relevance + confidence score
        memories.sort(
            key=lambda m: m["relevance"] * 0.6 + m["confidence"] * 0.4,
            reverse=True,
        )
        return memories[:n_results]

    # ── Invalidation ──────────────────────────────────────────────────────

    async def invalidate(self, user_id: int, doc_id: str, reason: str = "") -> None:
        """
        Mark a pattern as invalidated (contradicted by newer evidence).
        We don't delete — the consolidator can review invalidated patterns
        and either remove them or update them.
        """
        col = self._collection(user_id)
        existing = col.get(ids=[doc_id], include=["metadatas"])
        if existing and existing["ids"]:
            meta = existing["metadatas"][0]
            meta["invalidated"] = "true"
            meta["invalidated_at"] = datetime.utcnow().isoformat()
            meta["invalidation_reason"] = reason
            col.update(ids=[doc_id], metadatas=[meta])

    async def cleanup_invalidated(self, user_id: int) -> int:
        """Remove patterns that have been invalidated. Returns count removed."""
        col = self._collection(user_id)
        # ChromaDB where filter
        results = col.get(
            where={"invalidated": "true"},
            include=["metadatas"],
        )
        if results and results["ids"]:
            col.delete(ids=results["ids"])
            return len(results["ids"])
        return 0

    # ── Prompt formatting ─────────────────────────────────────────────────

    async def format_for_prompt(self, user_id: int, query: str, n: int = 4) -> str:
        """Return semantically relevant patterns as a prompt block."""
        memories = await self.search(user_id, query, n_results=n)
        if not memories:
            return ""

        lines = ["## Things I've learned (semantic memory)"]
        for mem in memories:
            rel = mem["relevance"]
            conf = mem["confidence"]
            ptype = mem["metadata"].get("pattern_type", "general")
            lines.append(f"- ({ptype}, confidence {conf:.1f}) {mem['content']}")
        return "\n".join(lines)

    # ── Direct access ─────────────────────────────────────────────────────

    async def delete(self, user_id: int, doc_id: str) -> None:
        col = self._collection(user_id)
        col.delete(ids=[doc_id])

    async def count(self, user_id: int) -> int:
        return self._collection(user_id).count()


# Singleton
semantic = SemanticMemory()
