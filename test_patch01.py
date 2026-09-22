"""KEEPER_PATCH_01 — salience, forgetting, state injection, action_bias."""
from __future__ import annotations

import asyncio
import tempfile
from datetime import timedelta
from pathlib import Path

from config import settings as settings_mod
from core.action_bias import (
    apply_verdict,
    eligible_hypotheses,
    ensure_hypothesis_fields,
    select_sticky,
    trial_eligible_after_reply,
)
from core.events import UserMessageEvent
from core.internal_state import InternalState
from core.timeutil import utcnow
from memory.episodic import EpisodicMemory
from memory.salience import episode_salience


def _isolate_chroma(store: EpisodicMemory) -> None:
    async def _nov(user_id, content):
        return 1.0

    async def _up(**kwargs):
        return None

    store._compute_novelty = _nov  # type: ignore[method-assign]
    store._chroma_upsert = _up  # type: ignore[method-assign]
    store._chroma_delete = lambda *a, **k: None  # type: ignore[method-assign]


def test_episode_salience_source_prior_discriminates():
    now = utcnow()
    created = (now - timedelta(days=10)).isoformat()
    user = episode_salience(
        {"created_at": created, "source": "user_turn", "novelty": 1.0, "used_count": 0},
        now=now,
        ref_count=0,
    )
    keeper = episode_salience(
        {"created_at": created, "source": "keeper_response", "novelty": 1.0, "used_count": 0},
        now=now,
        ref_count=0,
    )
    assert user > keeper
    assert 0.0 <= keeper <= 1.0


def test_episode_salience_used_and_refs_raise():
    now = utcnow()
    created = (now - timedelta(days=10)).isoformat()
    base = episode_salience(
        {"created_at": created, "source": "user_turn", "novelty": 0.5, "used_count": 0},
        now=now,
        ref_count=0,
    )
    used = episode_salience(
        {"created_at": created, "source": "user_turn", "novelty": 0.5, "used_count": 3},
        now=now,
        ref_count=0,
    )
    refs = episode_salience(
        {"created_at": created, "source": "user_turn", "novelty": 0.5, "used_count": 0},
        now=now,
        ref_count=2,
    )
    assert used > base
    assert refs > base


def test_forget_skips_young_episodes():
    orig_dry = settings_mod.config.FORGET_DRY_RUN
    orig_min = settings_mod.config.FORGET_MIN_AGE_DAYS
    orig_thr = settings_mod.config.FORGET_THRESHOLD
    settings_mod.config.FORGET_DRY_RUN = False
    settings_mod.config.FORGET_MIN_AGE_DAYS = 7
    settings_mod.config.FORGET_THRESHOLD = 1.0
    try:
        with tempfile.TemporaryDirectory() as td:
            store = EpisodicMemory(db_path=Path(td) / "ep.db")
            _isolate_chroma(store)

            async def _run():
                await store.init()
                eid = await store.store(
                    user_id=1, type="event", content="fresh", source="user_turn",
                )
                assert await store.forget_by_evidence(1) == 0
                assert await store.count(1) == 1
                import aiosqlite
                async with aiosqlite.connect(store.db_path) as conn:
                    await conn.execute(
                        "UPDATE episodes SET created_at = ? WHERE episode_id = ?",
                        ((utcnow() - timedelta(days=30)).isoformat(), eid),
                    )
                    await conn.commit()
                assert await store.forget_by_evidence(1) == 1
                assert await store.count(1) == 0

            asyncio.run(_run())
    finally:
        settings_mod.config.FORGET_DRY_RUN = orig_dry
        settings_mod.config.FORGET_MIN_AGE_DAYS = orig_min
        settings_mod.config.FORGET_THRESHOLD = orig_thr


def test_forget_dry_run_does_not_delete():
    orig_dry = settings_mod.config.FORGET_DRY_RUN
    orig_thr = settings_mod.config.FORGET_THRESHOLD
    settings_mod.config.FORGET_DRY_RUN = True
    settings_mod.config.FORGET_THRESHOLD = 1.0
    try:
        with tempfile.TemporaryDirectory() as td:
            store = EpisodicMemory(db_path=Path(td) / "ep.db")
            _isolate_chroma(store)

            async def _run():
                await store.init()
                eid = await store.store(
                    user_id=1, type="event", content="old-looking",
                    source="keeper_response",
                )
                import aiosqlite
                async with aiosqlite.connect(store.db_path) as conn:
                    await conn.execute(
                        "UPDATE episodes SET created_at = ? WHERE episode_id = ?",
                        ((utcnow() - timedelta(days=40)).isoformat(), eid),
                    )
                    await conn.commit()
                assert await store.forget_by_evidence(1) == 0
                assert await store.count(1) == 1

            asyncio.run(_run())
    finally:
        settings_mod.config.FORGET_DRY_RUN = orig_dry
        settings_mod.config.FORGET_THRESHOLD = orig_thr


def test_reference_count():
    with tempfile.TemporaryDirectory() as td:
        store = EpisodicMemory(db_path=Path(td) / "ep.db")
        _isolate_chroma(store)

        async def _run():
            await store.init()
            eid = await store.store(
                user_id=1, type="event", content="src", source="user_turn",
            )
            assert await store.reference_count(eid) == 0
            await store.add_episode_ref(eid, "semantic", "doc1")
            await store.add_episode_ref(eid, "opinion", "op1")
            await store.add_episode_ref(eid, "user_life", "c1")
            assert await store.reference_count(eid) == 3

        asyncio.run(_run())


def test_state_injection_omits_empty_lists():
    state = InternalState()
    orig_f = settings_mod.config.STATE_FIELDS_INJECTED
    orig_d = settings_mod.config.DRIVES_INJECTED
    try:
        settings_mod.config.STATE_FIELDS_INJECTED = []
        settings_mod.config.DRIVES_INJECTED = []
        assert state.to_prompt_context() == ""

        settings_mod.config.STATE_FIELDS_INJECTED = ["arousal", "fatigue"]
        settings_mod.config.DRIVES_INJECTED = []
        text = state.to_prompt_context()
        assert "Arousal:" in text
        assert "Fatigue:" in text
        assert "Curiosity:" not in text
        assert "Active drives" not in text
    finally:
        settings_mod.config.STATE_FIELDS_INJECTED = orig_f
        settings_mod.config.DRIVES_INJECTED = orig_d


def test_action_bias_sticky_selection():
    model = {
        "model_version": 12,
        "hypotheses": [
            {"statement": "a", "confidence": 0.5, "tested": False},
            {"statement": "b", "confidence": 0.9, "tested": False},
        ],
    }
    first = select_sticky(model)
    assert first["statement"] == "b"
    hid = first["id"]
    model["hypotheses"][0]["confidence"] = 0.99
    again = select_sticky(model)
    assert again["id"] == hid


def test_invalid_does_not_break_held_streak():
    orig = settings_mod.config.ACTION_BIAS_STREAK_LEN
    settings_mod.config.ACTION_BIAS_STREAK_LEN = 3
    try:
        model = {
            "hypotheses": [{
                "id": "h-v12-01",
                "statement": "x",
                "confidence": 0.9,
                "tested": False,
                "actionable": None,
                "bias_text": "Name a file.",
                "trial_log": [],
            }],
            "standing_constraints": [],
        }
        apply_verdict(model, "h-v12-01", 1, "HELD")
        apply_verdict(model, "h-v12-01", 2, "INVALID")
        apply_verdict(model, "h-v12-01", 3, "HELD")
        apply_verdict(model, "h-v12-01", 4, "HELD")
        h = model["hypotheses"][0]
        assert h["tested"] is True
        assert h["actionable"] is True
        assert model["standing_constraints"]
    finally:
        settings_mod.config.ACTION_BIAS_STREAK_LEN = orig


def test_three_violated_marks_not_actionable():
    model = {
        "hypotheses": [{
            "id": "h-v12-02",
            "statement": "y",
            "confidence": 0.8,
            "tested": False,
            "actionable": None,
            "capability_verified": True,
            "bias_text": "Ask one question.",
            "trial_log": [],
        }],
        "active_bias_hypothesis_id": "h-v12-02",
    }
    apply_verdict(model, "h-v12-02", 1, "VIOLATED")
    apply_verdict(model, "h-v12-02", 2, "VIOLATED")
    apply_verdict(model, "h-v12-02", 3, "VIOLATED")
    h = model["hypotheses"][0]
    assert h["tested"] is True
    assert h["actionable"] is False
    assert model["active_bias_hypothesis_id"] is None


def test_trial_skips_short_and_tool_only():
    assert not trial_eligible_after_reply(
        response_text="ok", tool_calls_only=False, error=False
    )
    long_reply = (
        "one two three four five six seven eight nine ten "
        "eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
    )
    assert len(long_reply.split()) >= 20
    assert not trial_eligible_after_reply(
        response_text=long_reply,
        tool_calls_only=True,
        error=False,
    )
    assert trial_eligible_after_reply(
        response_text=long_reply,
        tool_calls_only=False,
        error=False,
    )


def test_compute_salience_still_exists():
    s = InternalState()
    import memory.state_trace as st
    orig = st.record
    st.record = lambda *a, **k: None
    try:
        s.update(UserMessageEvent(content="hi", novelty=0.9, complexity=0.4))
        v = s.compute_salience()
        assert 0.0 <= v <= 1.0
    finally:
        st.record = orig


def test_ensure_hypothesis_fields_assigns_ids():
    hyps = ensure_hypothesis_fields(
        [{"statement": "z", "confidence": 0.7, "tested": False}],
        version=12,
    )
    assert hyps[0]["id"] == "h-v12-01"
    assert hyps[0]["actionable"] is None
    assert hyps[0]["trial_log"] == []
    assert hyps[0]["capability_verified"] is False
    assert hyps[0]["capability_note"] is None
    assert eligible_hypotheses(hyps)
