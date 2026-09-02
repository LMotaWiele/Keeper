"""Pairwise Elo ratings over action *types* for unbounded goals."""
from __future__ import annotations

import json
import logging
import random
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from core.timeutil import utcnow, parse_iso
from typing import Any

from langchain_core.messages import HumanMessage

from core.llm import get_llm

log = logging.getLogger(__name__)

ACTION_TYPES = [
    "web_research",         # SearXNG + synthesis + opinion formation
    "self_theorize",        # SelfTheorizer cycle
    "future_simulate",      # FutureSimulator on a candidate action
    "codebase_read",        # CodebaseIndex module read + analysis
    "memory_synthesis",     # LLM pass over recalled episodes → semantic pattern
    "consolidation_review", # review invalidated / low-confidence patterns
]

ACTION_PRIORS = {
    "web_research":         1250,
    "memory_synthesis":     1210,
    "codebase_read":        1200,
    "consolidation_review": 1190,
    "future_simulate":      1180,
    "self_theorize":        1150,
}

ELO_K = 24
ELO_SEED = 1200
UNPRODUCTIVE_ELO = 1000

RATING_PROMPT = """\
You are comparing actions taken in pursuit of a goal, to learn which *kinds* of action actually move it.

Goal: {name} — {description}
What full saturation looks like: {ceiling_description}

New action (`{action_type}`): {summary}
Artifacts it produced: {artifact_refs}

Prior actions:
{priors}

For each prior action, decide which of the two moved the goal further toward saturation. Judge by what was actually produced, not by how the action describes itself. An action that produced a specific external finding beats one that restated existing knowledge. Ties are allowed and should be common.

Output only: [{{"opponent_index": 0, "winner": "new"|"prior"|"tie", "why": "one clause"}}]
"""


@dataclass
class ActionRecord:
    id: str
    goal_id: str
    action_type: str
    summary: str
    artifact_refs: list[str]
    cost_usd: float
    elo: float
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ActionRecord":
        return cls(
            id=data.get("id") or uuid.uuid4().hex[:10],
            goal_id=data.get("goal_id", ""),
            action_type=data.get("action_type", ""),
            summary=data.get("summary", ""),
            artifact_refs=list(data.get("artifact_refs") or []),
            cost_usd=float(data.get("cost_usd") or 0.0),
            elo=float(data.get("elo") or ELO_SEED),
            timestamp=data.get("timestamp") or utcnow().isoformat(),
        )


def _expected(new_elo: float, opp_elo: float) -> float:
    return 1.0 / (1.0 + 10 ** ((opp_elo - new_elo) / 400.0))


def _apply_elo(new_elo: float, opp_elo: float, score: float) -> tuple[float, float]:
    exp = _expected(new_elo, opp_elo)
    new_elo = new_elo + ELO_K * (score - exp)
    opp_elo = opp_elo + ELO_K * ((1.0 - score) - (1.0 - exp))
    return new_elo, opp_elo


def _recompute_aggregates(goal: Any) -> None:
    by_type: dict[str, list[dict]] = {}
    for rec in goal.action_log:
        by_type.setdefault(rec.get("action_type", ""), []).append(rec)
    ratings: dict[str, dict] = {}
    for atype, recs in by_type.items():
        elos = [float(r.get("elo") or ELO_SEED) for r in recs]
        spend = sum(float(r.get("cost_usd") or 0.0) for r in recs)
        ratings[atype] = {
            "elo": sum(elos) / len(elos) if elos else float(ACTION_PRIORS.get(atype, ELO_SEED)),
            "n": len(recs),
            "spend_usd": spend,
        }
    goal.action_ratings = ratings


def select_action_type(goal: Any, unavailable: set[str] | None = None) -> str:
    """Epsilon-greedy over action types, with forced exploration of unseen types."""
    blocked = unavailable or set()
    types = [t for t in ACTION_TYPES if t not in blocked] or list(ACTION_TYPES)
    ratings = goal.action_ratings or {}
    for t in types:
        n = int((ratings.get(t) or {}).get("n") or 0)
        if n < 2:
            return t
    ns = [int((ratings.get(t) or {}).get("n") or 0) for t in types]
    epsilon = 0.25 if min(ns) < 10 else 0.10
    if random.random() < epsilon:
        return random.choice(types)
    return max(
        types,
        key=lambda t: float(
            (ratings.get(t) or {}).get("elo") or ACTION_PRIORS.get(t, ELO_SEED)
        ),
    )


async def rate_action(goal: Any, record: ActionRecord, companion: Any) -> None:
    """Score a new action against prior ones and update type aggregates."""
    if not record.artifact_refs:
        record.elo = UNPRODUCTIVE_ELO
        goal.action_log.append(record.to_dict())
        goal.action_log = goal.action_log[-50:]
        _recompute_aggregates(goal)
        log.info(
            "unproductive action type=%s goal=%s (no artifacts)",
            record.action_type, goal.name,
        )
        return

    log_recs = list(goal.action_log)
    opponents: list[dict] = []
    if log_recs:
        ranked = sorted(log_recs, key=lambda r: float(r.get("elo") or 0), reverse=True)
        top = ranked[:2]
        rest = [r for r in log_recs if r not in top]
        random_picks = random.sample(rest, k=min(2, len(rest))) if rest else []
        seen_ids = set()
        for r in top + random_picks:
            rid = r.get("id")
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            opponents.append(r)

    if len(opponents) < 1:
        record.elo = float(ACTION_PRIORS.get(record.action_type, ELO_SEED))
        goal.action_log.append(record.to_dict())
        goal.action_log = goal.action_log[-50:]
        _recompute_aggregates(goal)
        return

    priors_text = "\n".join(
        f"[{i}] ({r.get('action_type')}) {r.get('summary')} — artifacts: {r.get('artifact_refs')}"
        for i, r in enumerate(opponents)
    )
    prompt = RATING_PROMPT.format(
        name=goal.name,
        description=goal.description,
        ceiling_description=goal.ceiling_description or goal.description,
        action_type=record.action_type,
        summary=record.summary,
        artifact_refs=record.artifact_refs,
        priors=priors_text,
    )

    comparisons: list[dict] = []
    try:
        from core.json_utils import parse_json_lenient
        result = await get_llm("action_rating", json_mode=True).ainvoke(
            [HumanMessage(content=prompt)]
        )
        parsed = parse_json_lenient(result.content)
        if isinstance(parsed, dict):
            parsed = parsed.get("comparisons") or parsed.get("results") or []
        if isinstance(parsed, list):
            comparisons = [c for c in parsed if isinstance(c, dict)]
    except Exception as exc:
        log.debug("action rating LLM failed: %s", exc)

    new_elo = float(ACTION_PRIORS.get(record.action_type, ELO_SEED))
    opp_by_id = {r.get("id"): r for r in goal.action_log}
    for cmp_ in comparisons:
        try:
            idx = int(cmp_.get("opponent_index", -1))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(opponents):
            continue
        opp = opponents[idx]
        winner = str(cmp_.get("winner") or "tie").lower()
        score = {"new": 1.0, "prior": 0.0, "tie": 0.5}.get(winner, 0.5)
        opp_elo = float(opp.get("elo") or ELO_SEED)
        new_elo, opp_elo = _apply_elo(new_elo, opp_elo, score)
        stored = opp_by_id.get(opp.get("id"))
        if stored is not None:
            stored["elo"] = opp_elo
        opp["elo"] = opp_elo

    record.elo = new_elo
    goal.action_log.append(record.to_dict())
    goal.action_log = goal.action_log[-50:]
    _recompute_aggregates(goal)
