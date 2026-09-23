"""World-model tools. Quote required. No user_id. No delete."""
from __future__ import annotations

import json

from langchain_core.tools import tool


def _world():
    from core.loop import companion
    return companion.user_world


def _source_turn_id() -> str | None:
    from core.loop import companion
    uid = getattr(companion, "_current_user_id", None)
    mapping = getattr(companion, "_current_turn_episode_id", None) or {}
    if uid is None:
        return None
    return mapping.get(uid)


def _dump(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


@tool
def record_world_fact(domain: str, key: str, value: str, quote: str) -> str:
    """Record a fact he stated. quote must be his exact words from this turn.

    Do not announce the write. Do not infer mood or motive.
    """
    return _dump(_world().record_world_fact(
        domain, key, value, quote, source_turn_id=_source_turn_id(),
    ))


@tool
def confirm_world_fact(key: str, quote: str) -> str:
    """He restated a fact you already hold. quote must be his exact words."""
    return _dump(_world().confirm_world_fact(
        key, quote, source_turn_id=_source_turn_id(),
    ))


@tool
def supersede_world_fact(key: str, new_value: str, quote: str) -> str:
    """A fact changed. Writes a new slot and retires the old one. No delete."""
    return _dump(_world().supersede_world_fact(
        key, new_value, quote, source_turn_id=_source_turn_id(),
    ))


@tool
def get_world_model(domains: str | None = None) -> str:
    """Read the world model. domains is an optional comma-separated filter."""
    wanted = None
    if domains and domains.strip():
        wanted = [d.strip() for d in domains.split(",") if d.strip()]
    slots = _world().get_world_model(wanted)
    return _dump({
        "ok": True,
        "slots": [
            {
                "domain": s.domain,
                "key": s.key,
                "value": s.value,
                "quote": s.quote,
                "stated_at": s.stated_at,
                "confirmed_at": s.confirmed_at,
            }
            for s in slots
        ],
    })


WORLD_TOOLS = [
    record_world_fact,
    confirm_world_fact,
    supersede_world_fact,
    get_world_model,
]
