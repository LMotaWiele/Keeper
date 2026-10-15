"""Shared local embeddings and deterministic cosine matching (no model calls)."""
from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch through the existing semantic embedder, off the event loop."""
    if not texts:
        return []
    from memory.semantic import semantic

    def run():
        return [[float(x) for x in vec] for vec in semantic._get_embeddings().embed(texts)]

    return await asyncio.to_thread(run)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Embedding dimensions differ")
    denominator = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / denominator if denominator else 0.0


def nearest(vec: Sequence[float], candidates: Sequence[Sequence[float]]) -> tuple[int | None, float]:
    """Return the first best match, or (None, 0.0) for an empty set."""
    if not candidates:
        return None, 0.0
    scores = [cosine(vec, candidate) for candidate in candidates]
    index = max(range(len(scores)), key=scores.__getitem__)
    return index, scores[index]
