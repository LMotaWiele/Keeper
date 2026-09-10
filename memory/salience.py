"""Evidence-based episode salience. Pure function over metadata + ref counts."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from config.settings import config
from core.timeutil import parse_iso, utcnow


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def episode_salience(
    meta: dict[str, Any],
    now: datetime | None = None,
    ref_count: int = 0,
) -> float:
    """Salience from facts about the episode, not the affect vector."""
    now = now or utcnow()
    created_raw = meta.get("created_at") or ""
    try:
        created = parse_iso(str(created_raw))
        age_days = max((now - created).total_seconds() / 86400.0, 0.0)
    except Exception:
        age_days = 0.0

    used_count = int(meta.get("used_count") or 0)
    novelty = float(meta.get("novelty") if meta.get("novelty") is not None else 1.0)
    source = str(meta.get("source") or "")
    source_prior = config.SALIENCE_SOURCE_PRIOR.get(source, 0.3)
    recall_utility = min(used_count / 3.0, 1.0)
    ref_score = min(ref_count / 2.0, 1.0)
    age_penalty = min(age_days / 180.0, 1.0)

    return clamp(
        config.W_SOURCE * source_prior
        + config.W_NOVELTY * novelty
        + config.W_RECALL * recall_utility
        + config.W_REF * ref_score
        - config.W_AGE * age_penalty,
    )


def salience_terms(
    meta: dict[str, Any],
    now: datetime | None = None,
    ref_count: int = 0,
) -> dict[str, float]:
    """Component terms for deletion logs."""
    now = now or utcnow()
    created_raw = meta.get("created_at") or ""
    try:
        created = parse_iso(str(created_raw))
        age_days = max((now - created).total_seconds() / 86400.0, 0.0)
    except Exception:
        age_days = 0.0
    used_count = int(meta.get("used_count") or 0)
    novelty = float(meta.get("novelty") if meta.get("novelty") is not None else 1.0)
    source = str(meta.get("source") or "")
    source_prior = config.SALIENCE_SOURCE_PRIOR.get(source, 0.3)
    recall_utility = min(used_count / 3.0, 1.0)
    ref_score = min(ref_count / 2.0, 1.0)
    age_penalty = min(age_days / 180.0, 1.0)
    return {
        "source_prior": source_prior,
        "novelty": novelty,
        "recall_utility": recall_utility,
        "ref_score": ref_score,
        "age_days": age_days,
        "age_penalty": age_penalty,
        "w_source": config.W_SOURCE * source_prior,
        "w_novelty": config.W_NOVELTY * novelty,
        "w_recall": config.W_RECALL * recall_utility,
        "w_ref": config.W_REF * ref_score,
        "w_age": config.W_AGE * age_penalty,
    }
