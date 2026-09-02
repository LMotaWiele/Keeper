"""Machine-checkable completion conditions for bounded goals."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from core.timeutil import utcnow, parse_iso
from typing import Any

from langchain_core.messages import HumanMessage

from core.llm import get_llm
from core.opinions import OpinionOrigin

log = logging.getLogger(__name__)

CONDITION_TYPES = {
    "opinion_registered",       # params: {"domain": str}
    "semantic_pattern_stored",  # params: {"query": str, "min_confidence": float}
    "episode_tagged",           # params: {"tag": str}
    "commitment_resolved",      # params: {"description": str}
    "proposal_created",         # params: {}
    "user_confirmed",           # params: {"question": str}
}

_REQUIRED_PARAMS: dict[str, tuple[str, ...]] = {
    "opinion_registered": ("domain",),
    "semantic_pattern_stored": ("query", "min_confidence"),
    "episode_tagged": ("tag",),
    "commitment_resolved": ("description",),
    "proposal_created": (),
    "user_confirmed": ("question",),
}


@dataclass
class CompletionCondition:
    type: str
    params: dict
    description: str          # human-readable, shown in prompts
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CompletionCondition":
        return cls(
            type=data.get("type", ""),
            params=dict(data.get("params") or {}),
            description=data.get("description", ""),
            created_at=data.get("created_at") or utcnow().isoformat(),
        )


def validate_condition(raw: Any) -> CompletionCondition | None:
    """Return a condition if type and params are valid, else None."""
    if not isinstance(raw, dict):
        return None
    ctype = raw.get("type")
    if ctype not in CONDITION_TYPES:
        return None
    params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
    for key in _REQUIRED_PARAMS[ctype]:
        if key not in params or params[key] in (None, ""):
            return None
    if ctype == "semantic_pattern_stored":
        try:
            params = {**params, "min_confidence": float(params["min_confidence"])}
        except (TypeError, ValueError):
            return None
    description = (raw.get("description") or "").strip() or f"{ctype} satisfied"
    return CompletionCondition(
        type=ctype,
        params=params,
        description=description,
        created_at=raw.get("created_at") or utcnow().isoformat(),
    )


def _iso(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _parse_tags(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(t) for t in raw]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(t) for t in parsed]
        except json.JSONDecodeError:
            return [raw] if raw else []
    return []


def _word_overlap(a: str, b: str) -> float:
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), 1)


async def check(
    cond: CompletionCondition | dict,
    *,
    goal: Any,
    companion: Any,
    user_id: int,
) -> tuple[bool, str]:
    """Return (satisfied, evidence_string)."""
    if isinstance(cond, dict):
        parsed = validate_condition(cond)
        if parsed is None:
            return False, "invalid completion condition"
        cond = parsed

    goal_created = _iso(getattr(goal, "created_at", ""))
    ctype = cond.type
    params = cond.params

    if ctype == "opinion_registered":
        return _check_opinion(companion, params, goal_created)
    if ctype == "semantic_pattern_stored":
        return await _check_semantic(user_id, params, goal_created)
    if ctype == "episode_tagged":
        return await _check_episode(companion, user_id, params, goal_created)
    if ctype == "commitment_resolved":
        return _check_commitment(companion, params)
    if ctype == "proposal_created":
        return _check_proposal(companion, goal_created)
    if ctype == "user_confirmed":
        return await _check_user_confirmed(companion, user_id, params)
    return False, f"unknown condition type {ctype}"


def _check_opinion(companion: Any, params: dict, goal_created: str) -> tuple[bool, str]:
    domain = str(params.get("domain", "")).strip().lower()
    if not domain:
        return False, "missing domain"
    registry = getattr(getattr(companion, "self_model", None), "opinions", None)
    if registry is None:
        return False, "no opinion registry"
    allowed = {OpinionOrigin.EXTERNAL, OpinionOrigin.INDEPENDENT}
    for op in registry.opinions.values():
        op_domain = (op.domain or "").lower()
        if op_domain != domain and domain not in op_domain and op_domain not in domain:
            continue
        if op.origin not in allowed:
            continue
        if op.conviction < 0.4:
            continue
        if (op.formed_at or "") < goal_created:
            continue
        return True, f"{op.id}: {op.position}"
    return False, f"no matching opinion in {domain}"


async def _check_semantic(user_id: int, params: dict, goal_created: str) -> tuple[bool, str]:
    from memory.semantic import semantic
    query = str(params.get("query", "")).strip()
    min_conf = float(params.get("min_confidence", 0.5))
    hits = await semantic.search(user_id, query, n_results=3)
    for hit in hits:
        if hit.get("confidence", 0) < min_conf:
            continue
        if (hit.get("stored_at") or "") <= goal_created:
            continue
        return True, hit.get("content", "")
    return False, f"no pattern matching {query!r}"


async def _check_episode(
    companion: Any, user_id: int, params: dict, goal_created: str,
) -> tuple[bool, str]:
    tag = str(params.get("tag", "")).strip()
    if not tag:
        return False, "missing tag"
    episodic = getattr(getattr(companion, "memory", None), "episodic", None)
    if episodic is None:
        return False, "no episodic store"
    episodes = await episodic.get_recent(user_id, limit=80)
    for ep in episodes:
        tags = _parse_tags(ep.get("tags"))
        if tag not in tags:
            continue
        if (ep.get("created_at") or "") <= goal_created:
            continue
        content = (ep.get("content") or "")[:120]
        return True, f"{ep.get('id')}: {content}"
    return False, f"no episode tagged {tag!r}"


def _check_commitment(companion: Any, params: dict) -> tuple[bool, str]:
    needle = str(params.get("description", "")).strip()
    tracker = getattr(companion, "user_life", None)
    if tracker is None or not needle:
        return False, "no commitments"
    best = None
    best_score = 0.0
    for c in tracker.commitments:
        if c.status not in ("completed", "abandoned"):
            continue
        score = _word_overlap(needle, c.description)
        if score > best_score:
            best_score = score
            best = c
    if best is not None and best_score >= 0.4:
        return True, f"{best.description} ({best.status})"
    return False, f"no resolved commitment matching {needle!r}"


def _check_proposal(companion: Any, goal_created: str) -> tuple[bool, str]:
    theorizer = getattr(companion, "theorizer", None)
    if theorizer is None:
        return False, "no theorizer"
    for p in theorizer.pending_proposals:
        if (p.get("generated_at") or "") > goal_created:
            return True, p.get("title") or "proposal"
    return False, "no new proposals"


USER_CONFIRMED_PROMPT = """\
You are checking whether the USER explicitly confirmed something.

Question to confirm: {question}

Recent turns (Human = user, AI = system):
{turns}

Did the USER explicitly confirm this (a clear yes, agreement, or equivalent)?
You MUST quote the exact Human turn that is the confirmation.
If you cannot quote a Human turn, it is not confirmed — the system confirming
itself does not count.

Output ONLY JSON:
{{"satisfied": true or false, "quote": "verbatim Human turn or empty string"}}
"""


async def _check_user_confirmed(
    companion: Any, user_id: int, params: dict,
) -> tuple[bool, str]:
    question = str(params.get("question", "")).strip()
    if not question:
        return False, "missing question"
    working = getattr(getattr(companion, "memory", None), "working", None)
    if working is None:
        return False, "no working memory"
    items = working.get_items(user_id)[-20:]
    human_turns = [
        getattr(i, "content", "")
        for i in items
        if getattr(i, "role", "") in ("user", "human")
    ]
    turns = "\n".join(
        f"{'Human' if getattr(i, 'role', '') in ('user', 'human') else 'AI'}: "
        f"{getattr(i, 'content', '')[:400]}"
        for i in items
    )
    prompt = USER_CONFIRMED_PROMPT.format(question=question, turns=turns or "(none)")
    try:
        from core.json_utils import parse_json_lenient
        result = await get_llm("completion_check", json_mode=True).ainvoke(
            [HumanMessage(content=prompt)]
        )
        parsed = parse_json_lenient(result.content)
        if not isinstance(parsed, dict):
            return False, "completion check failed"
    except Exception as exc:
        log.debug("user_confirmed check failed: %s", exc)
        return False, "completion check failed"

    quote = str(parsed.get("quote") or "").strip()
    if not parsed.get("satisfied"):
        return False, "user has not confirmed"
    if not quote:
        return False, "no Human turn quoted"
    if not any(quote in t or t in quote for t in human_turns if t):
        return False, "quoted text is not a Human turn"
    return True, quote
