"""User-life tools — commitments and wellbeing. Bound into ALL_TOOLS."""
from __future__ import annotations

import json

from langchain_core.tools import tool


def _tracker():
    from core.loop import companion
    return companion.user_life


def _source_episode_id() -> str | None:
    from core.loop import companion
    uid = getattr(companion, "_current_user_id", None)
    mapping = getattr(companion, "_current_turn_episode_id", None) or {}
    if uid is None:
        return None
    return mapping.get(uid)


def _dump(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


@tool
async def record_user_commitment(
    commitment: str,
    evidence: str,
    deadline: str | None = None,
) -> str:
    """
    Record something the user said they will do.

    evidence must be the user's own words, not a paraphrase of the commitment.
    deadline is optional (ISO date or relative: 'tomorrow', 'end of week').
    Do not invent a user_id; there is one user.
    Do not announce the record, do not restate a deadline, and do not add
    a confirmation turn. Continue the conversation he was having.
    """
    life = _tracker()
    payload = life.record_commitment(
        text=commitment,
        evidence=evidence,
        deadline=deadline,
        source_episode_id=_source_episode_id(),
    )
    if payload.get("ok") and payload.get("id") is not None:
        await life.register_episode_ref(
            payload.get("source_episode_id"), int(payload["id"])
        )
    return _dump(payload)


@tool
async def update_commitment_status(
    commitment_id: int,
    status: str,
    note: str | None = None,
) -> str:
    """
    Mark a commitment active, done, abandoned, or missed.

    Abandoning is frictionless — do not ask for confirmation, do not follow up.
    Unknown ids return the currently open ids; never fail silently.
    Do not announce the update.
    """
    life = _tracker()
    return _dump(life.update_status(int(commitment_id), status, note=note))


@tool
async def get_active_commitments() -> str:
    """Return open items he said he would do.

    Call this only when he asks what he was going to do. Do not call it to
    open a status check.
    """
    life = _tracker()
    items = life.get_active_commitments()
    return _dump({
        "ok": True,
        "commitments": [
            {
                "id": c.id,
                "text": c.text,
                "deadline": c.deadline,
                "deadline_raw": c.deadline_raw,
                "status": c.status,
                "evidence": c.evidence,
            }
            for c in items
        ],
    })


@tool
async def log_wellbeing_snapshot(
    score: float,
    evidence: str,
    source: str,
    note: str | None = None,
) -> str:
    """
    Log a wellbeing reading in [0.0, 1.0].

    source must be 'user_reported' when the user stated it, or 'inferred'
    when derived from tone. Inferred snapshots are stored for audit only
    and are excluded from trends.
    """
    life = _tracker()
    return _dump(life.log_wellbeing(score, evidence, source, note=note))


USER_LIFE_TOOLS = [
    record_user_commitment,
    update_commitment_status,
    get_active_commitments,
    log_wellbeing_snapshot,
]
