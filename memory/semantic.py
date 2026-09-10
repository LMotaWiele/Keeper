"""Semantic memory — Chroma patterns, not transcripts. Same text reinforces instead of duplicating."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from core.timeutil import utcnow, parse_iso
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from fastembed import TextEmbedding

from config.settings import config

log = logging.getLogger(__name__)


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
        self._embeddings: TextEmbedding | None = None
        self._embedding_dim: int | None = None

    # ── Internals ─────────────────────────────────────────────────────────

    def _get_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def _get_embeddings(self) -> TextEmbedding:
        if self._embeddings is None:
            self._embeddings = TextEmbedding(
                model_name=config.embedding_model,
                cache_dir=str(config.embedding_cache_dir),
            )
        return self._embeddings

    async def _embed(self, text: str) -> list[float]:
        def _run() -> list[float]:
            raw = next(iter(self._get_embeddings().embed([text])))
            return [float(x) for x in raw]
        return await asyncio.to_thread(_run)

    def _collection(self, user_id: int) -> chromadb.Collection:
        return self._get_client().get_or_create_collection(
            name=f"semantic_v2_user_{user_id}",
            metadata={"hnsw:space": "cosine"},
        )

    async def warmup(self) -> None:
        """Download/load the embedding model so the first user turn is not stalled."""
        vec = await self._embed("warmup")
        self._embedding_dim = len(vec)
        log.info(
            "Embeddings ready — dim=%d model=%s",
            self._embedding_dim,
            config.embedding_model,
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
                try:
                    old_sources = json.loads(old_sources)
                except json.JSONDecodeError:
                    old_sources = []

            new_confidence = min(1.0, old_confidence + 0.1)
            merged_sources = list(set(str(s) for s in (old_sources + (source_episode_ids or []))))

            col.update(
                ids=[doc_id],
                metadatas=[{
                    **old_meta,
                    "confidence": new_confidence,
                    "source_episode_ids": json.dumps(merged_sources),
                    "episode_ids": json.dumps(merged_sources),
                    "reinforced_at": utcnow().isoformat(),
                    "reinforcement_count": old_meta.get("reinforcement_count", 0) + 1,
                    "invalidated": "false",
                }],
            )
            await self._register_refs(merged_sources, doc_id)
            return doc_id

        # New pattern — embed and store
        embedding = await self._embed(content)
        source_ids = [str(s) for s in (source_episode_ids or [])]
        col.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{
                "stored_at": utcnow().isoformat(),
                "pattern_type": pattern_type,
                "confidence": confidence,
                "source_episode_ids": json.dumps(source_ids),
                "episode_ids": json.dumps(source_ids),
                "reinforcement_count": 0,
                "invalidated": "false",
                **(metadata or {}),
            }],
        )
        await self._register_refs(source_ids, doc_id)
        return doc_id

    async def _register_refs(self, episode_ids: list[str], doc_id: str) -> None:
        from memory.episodic import episodic
        await episodic.add_episode_refs(episode_ids, "semantic", doc_id)

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

        embedding = await self._embed(query)
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
                "id": _content_id(doc),
                "content": doc,
                "metadata": meta,
                "confidence": confidence,
                "relevance": round(1 - dist, 3),
                "stored_at": meta.get("stored_at", ""),
            })

        # Sort by combined relevance + confidence score
        memories.sort(
            key=lambda m: m["relevance"] * 0.6 + m["confidence"] * 0.4,
            reverse=True,
        )
        return memories[:n_results]

    async def list_review_candidates(
        self,
        user_id: int,
        max_confidence: float = 0.4,
    ) -> list[dict]:
        """Invalidated or low-confidence patterns for consolidation_review."""
        col = self._collection(user_id)
        if col.count() == 0:
            return []
        results = col.get(include=["documents", "metadatas"])
        out = []
        for doc_id, doc, meta in zip(
            results.get("ids") or [],
            results.get("documents") or [],
            results.get("metadatas") or [],
        ):
            confidence = meta.get("confidence", 0.5)
            invalidated = meta.get("invalidated", "false") == "true"
            if invalidated or confidence < max_confidence:
                out.append({
                    "id": doc_id,
                    "content": doc,
                    "metadata": meta,
                    "confidence": confidence,
                    "invalidated": invalidated,
                    "stored_at": meta.get("stored_at", ""),
                })
        return out

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
            meta["invalidated_at"] = utcnow().isoformat()
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
