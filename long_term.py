"""
Long-term memory — semantic vector store (ChromaDB).

Stores embeddings of important past conversations, documents, and knowledge.
Queried via similarity search to surface relevant context for any new message.
Persists indefinitely — this is the companion's "deep memory."
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings

from config.settings import config


class LongTermMemory:
    def __init__(self, db_path: Path | None = None):
        self.db_path = str(db_path or config.chroma_db_path)
        self._client: chromadb.PersistentClient | None = None
        self._embeddings: OpenAIEmbeddings | None = None

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
            name=f"user_{user_id}",
            metadata={"hnsw:space": "cosine"},
        )

    def _doc_id(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def store(
        self,
        user_id: int,
        content: str,
        metadata: dict | None = None,
    ) -> str:
        """Embed and store a piece of text. Returns the document ID."""
        doc_id = self._doc_id(content)
        embedding = self._get_embeddings().embed_query(content)
        col = self._collection(user_id)
        col.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[
                {
                    "stored_at": datetime.utcnow().isoformat(),
                    **(metadata or {}),
                }
            ],
        )
        return doc_id

    async def search(
        self,
        user_id: int,
        query: str,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """Semantic search — returns the closest memories to the query."""
        col = self._collection(user_id)
        if col.count() == 0:
            return []

        embedding = self._get_embeddings().embed_query(query)
        results = col.query(
            query_embeddings=[embedding],
            n_results=min(n_results, col.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        memories = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            memories.append(
                {
                    "content": doc,
                    "metadata": meta,
                    "relevance": round(1 - dist, 3),  # cosine → similarity
                }
            )
        return memories

    async def format_for_prompt(self, user_id: int, query: str, n: int = 4) -> str:
        """Return semantically relevant memories as a prompt block."""
        memories = await self.search(user_id, query, n_results=n)
        if not memories:
            return ""

        lines = ["## Relevant long-term memories"]
        for mem in memories:
            rel = mem["relevance"]
            lines.append(f"- (relevance {rel:.2f}) {mem['content']}")
        return "\n".join(lines)

    async def delete(self, user_id: int, doc_id: str) -> None:
        col = self._collection(user_id)
        col.delete(ids=[doc_id])


# Singleton
long_term = LongTermMemory()
