"""Bounded/unbounded goals, completion, and Elo rating."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from core.opinions import OpinionOrigin, OpinionRegistry
from core.timeutil import utcnow
from goals.action_rating import (
    ACTION_PRIORS,
    ActionRecord,
    UNPRODUCTIVE_ELO,
    rate_action,
)
from goals.completion import check, validate_condition
from goals.system import Goal, GoalStatus, GoalSystem


class _FakeBudget:
    budget_fraction_used = 0.0


class _FakeCompanion:
    def __init__(self, opinions: OpinionRegistry):
        self.self_model = type("SM", (), {"opinions": opinions})()
        self.theorizer = type("Th", (), {"pending_proposals": []})()
        self.user_life = type("UL", (), {"commitments": []})()
        self.memory = type("Mem", (), {"episodic": None, "working": None})()
        self.api_budget = _FakeBudget()


def test_unbounded_never_completes():
    gs = GoalSystem()
    g = Goal(
        id="u1",
        name="continuous_self_improvement",
        description="keep going",
        kind="unbounded",
        ceiling_description="saturation",
        budget_share=0.15,
        salience=0.9,
    )
    gs.instrumental.append(g)

    async def _run():
        updates = await gs.check_completions(_FakeCompanion(OpinionRegistry()), user_id=1)
        assert updates == []
        assert g.status == GoalStatus.ACTIVE
        assert g.is_unbounded
        g.advance(5.0, "lots of work")
        assert g.status == GoalStatus.ACTIVE
        assert g.completed_at is None

    asyncio.run(_run())


def test_bounded_completes_when_condition_satisfied():
    gs = GoalSystem()
    created = utcnow().isoformat()
    g = Goal(
        id="b1",
        name="form_ai_opinion",
        description="research AI consciousness",
        kind="bounded",
        completion_condition={
            "type": "opinion_registered",
            "params": {"domain": "ai_consciousness"},
            "description": "external opinion on ai_consciousness",
            "created_at": created,
        },
        created_at=datetime.now(timezone.utc),
    )
    gs.instrumental.append(g)
    reg = OpinionRegistry()

    async def _run():
        companion = _FakeCompanion(reg)
        updates = await gs.check_completions(companion, user_id=1)
        assert updates == []
        assert g.status == GoalStatus.ACTIVE

        await reg.form_opinion(
            domain="ai_consciousness",
            position="Functional but not phenomenal",
            reasoning="from research",
            origin=OpinionOrigin.EXTERNAL,
            conviction=0.7,
        )
        updates = await gs.check_completions(companion, user_id=1)
        assert len(updates) == 1
        assert updates[0]["completed"] is True
        assert g.status == GoalStatus.COMPLETED
        assert g in gs.completed

    asyncio.run(_run())


def test_invalid_condition_creates_unbounded():
    gs = GoalSystem()

    class FakeLLM:
        async def ainvoke(self, messages):
            class R:
                content = (
                    '[{"name": "vague", "description": "be generally better",'
                    ' "parent_goal": "understand", "salience": 0.5, "tags": []}]'
                )
            return R()

    async def _run():
        with patch("goals.system.get_llm", return_value=FakeLLM()):
            state = type("S", (), {
                "to_prompt_context": lambda self: "state",
                "drives": type("D", (), {"active_summary": lambda self: "drives"})(),
            })()
            new = await gs.generate_instrumental(state, None, user_id=1)
        assert len(new) == 1
        assert new[0].is_unbounded
        assert new[0].budget_share == 0.05
        assert new[0].ceiling_description

    asyncio.run(_run())


def test_unproductive_action_skips_llm_and_gets_1000():
    g = Goal(
        id="u2", name="improve", description="d", kind="unbounded",
        ceiling_description="sat",
    )
    rec = ActionRecord(
        id="a1", goal_id=g.id, action_type="self_theorize",
        summary="narrated a lot", artifact_refs=[],
        cost_usd=0.0, elo=0.0, timestamp=utcnow().isoformat(),
    )

    async def _run():
        with patch("goals.action_rating.get_llm") as mocked:
            await rate_action(g, rec, companion=None)
            mocked.assert_not_called()
        assert rec.elo == UNPRODUCTIVE_ELO
        assert g.action_log[-1]["elo"] == UNPRODUCTIVE_ELO

    asyncio.run(_run())


def test_elo_moves_in_the_right_direction():
    g = Goal(
        id="u3", name="improve", description="d", kind="unbounded",
        ceiling_description="sat",
    )
    prior = {
        "id": "old",
        "goal_id": g.id,
        "action_type": "self_theorize",
        "summary": "restated existing knowledge",
        "artifact_refs": ["proposal:0"],
        "cost_usd": 0.01,
        "elo": 1200.0,
        "timestamp": utcnow().isoformat(),
    }
    g.action_log.append(prior)
    rec = ActionRecord(
        id="new", goal_id=g.id, action_type="web_research",
        summary="found a specific external paper",
        artifact_refs=["opinion:abc", "episode:12"],
        cost_usd=0.02, elo=0.0, timestamp=utcnow().isoformat(),
    )

    class FakeLLM:
        async def ainvoke(self, messages):
            class R:
                content = '[{"opponent_index": 0, "winner": "new", "why": "external finding"}]'
            return R()

    async def _run():
        with patch("goals.action_rating.get_llm", return_value=FakeLLM()):
            await rate_action(g, rec, companion=None)
        seed = ACTION_PRIORS["web_research"]
        assert rec.elo > seed, (rec.elo, seed)
        stored_prior = next(r for r in g.action_log if r["id"] == "old")
        assert stored_prior["elo"] < 1200.0

    asyncio.run(_run())


def test_validate_condition_rejects_unknown_type():
    assert validate_condition({"type": "vibes", "params": {}, "description": "x"}) is None
    ok = validate_condition({
        "type": "opinion_registered",
        "params": {"domain": "ai"},
        "description": "an opinion exists",
    })
    assert ok is not None
    assert ok.type == "opinion_registered"


if __name__ == "__main__":
    test_unbounded_never_completes()
    print("✓ unbounded goals never complete")
    test_bounded_completes_when_condition_satisfied()
    print("✓ bounded goal completes when condition is satisfied")
    test_invalid_condition_creates_unbounded()
    print("✓ generated goal without valid condition is unbounded")
    test_unproductive_action_skips_llm_and_gets_1000()
    print("✓ no-artifact action gets 1000 and skips LLM")
    test_elo_moves_in_the_right_direction()
    print("✓ Elo updates in the right direction")
    test_validate_condition_rejects_unknown_type()
    print("✓ condition validation")
    print("All goal tests passed")
