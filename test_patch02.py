"""KEEPER_PATCH_02 — user-life tools, VOID verdict, capability gate, injection."""
from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage

from agent.graph import tool_executor
from config import settings as settings_mod
from core.action_bias import (
    apply_verdict,
    eligible_hypotheses,
    ensure_hypothesis_fields,
    insert_trial,
    void_capability_absent_trials,
    write_trial_verdict,
)
from core.internal_state import InternalState
from core.timeutil import utcnow
from core.user_life import (
    UserLifeTracker,
    evidence_is_paraphrase,
    parse_deadline,
)
from memory.episodic import EpisodicMemory
from tools import ALL_TOOLS
from tools.user_life_tools import USER_LIFE_TOOLS

TZ = ZoneInfo("Europe/Amsterdam")
USER_WORDS = "I'll get the scope doc out by Friday, that's a promise."


def _isolate_chroma(store: EpisodicMemory) -> None:
    async def _nov(user_id, content):
        return 1.0

    async def _up(**kwargs):
        return None

    store._compute_novelty = _nov  # type: ignore[method-assign]
    store._chroma_upsert = _up  # type: ignore[method-assign]
    store._chroma_delete = lambda *a, **k: None  # type: ignore[method-assign]


def _life(td: str) -> UserLifeTracker:
    tracker = UserLifeTracker(db_path=Path(td) / "life.db")
    tracker.ensure_schema()
    return tracker


def test_user_life_tools_bound_without_user_id():
    names = {t.name for t in ALL_TOOLS}
    expected = {
        "record_user_commitment",
        "update_commitment_status",
        "get_active_commitments",
        "log_wellbeing_snapshot",
    }
    assert expected <= names
    for tool in USER_LIFE_TOOLS:
        fields = getattr(tool.args_schema, "model_fields", {}) or {}
        assert "user_id" not in fields, tool.name


def test_record_commitment_tomorrow_resolves_amsterdam():
    now = datetime(2026, 9, 10, 1, 0, tzinfo=TZ)
    with tempfile.TemporaryDirectory() as td:
        life = _life(td)
        payload = life.record_commitment(
            "ship the scope doc",
            USER_WORDS,
            deadline="tomorrow",
            now=now,
        )
        assert payload["ok"] is True
        assert payload["deadline"]
        resolved = datetime.fromisoformat(payload["deadline"])
        assert resolved.date().isoformat() == "2026-09-11"
        assert resolved.tzinfo is not None
        offset = resolved.utcoffset()
        assert offset is not None
        # CEST in September is UTC+2
        assert offset.total_seconds() == 2 * 3600
        assert payload["deadline_raw"] == "tomorrow"
        assert payload["deadline_display"]


def test_record_commitment_midnight_boundary():
    now = datetime(2026, 9, 10, 23, 30, tzinfo=TZ)
    dt = parse_deadline("tomorrow", now=now)
    assert dt.date().isoformat() == "2026-09-11"


def test_record_commitment_rejects_empty_and_paraphrase_evidence():
    with tempfile.TemporaryDirectory() as td:
        life = _life(td)
        empty = life.record_commitment("ship the doc", "   ")
        assert empty["ok"] is False
        dup = life.record_commitment("ship the doc", "ship the doc")
        assert dup["ok"] is False
        assert evidence_is_paraphrase("ship the doc", "ship the doc")


def test_record_commitment_unparseable_deadline_errors():
    with tempfile.TemporaryDirectory() as td:
        life = _life(td)
        payload = life.record_commitment(
            "ship the doc", USER_WORDS, deadline="blargle-not-a-date",
        )
        assert payload["ok"] is False
        assert "blargle-not-a-date" in payload["error"]


def test_update_unknown_id_lists_open_ids():
    with tempfile.TemporaryDirectory() as td:
        life = _life(td)
        created = life.record_commitment("ship the doc", USER_WORDS)
        payload = life.update_status(99999, "done")
        assert payload["ok"] is False
        assert "99999" in payload["error"]
        assert str(created["id"]) in payload["error"]


def test_update_invalid_status_lists_four():
    with tempfile.TemporaryDirectory() as td:
        life = _life(td)
        created = life.record_commitment("ship the doc", USER_WORDS)
        payload = life.update_status(created["id"], "in progress")
        assert payload["ok"] is False
        for status in ("active", "done", "abandoned", "missed"):
            assert status in payload["error"]


def test_terminal_to_terminal_rejected_abandoned_is_frictionless():
    with tempfile.TemporaryDirectory() as td:
        life = _life(td)
        created = life.record_commitment("ship the doc", USER_WORDS)
        first = life.update_status(created["id"], "abandoned")
        assert first["ok"] is True
        assert first["status"] == "abandoned"
        second = life.update_status(created["id"], "done")
        assert second["ok"] is False
        assert "abandoned" in second["error"]


def test_commitment_block_absent_when_empty():
    with tempfile.TemporaryDirectory() as td:
        life = _life(td)
        block = life.on_session_start("s1")
        assert block == ""
        assert life.to_prompt_context() == ""


def test_commitment_injection_cap_and_session_cache():
    with tempfile.TemporaryDirectory() as td:
        life = _life(td)
        now = datetime(2026, 9, 10, 12, 0, tzinfo=TZ)
        for i in range(7):
            life.record_commitment(
                f"task {i}",
                f"Lucas said he will do task {i} this week.",
                deadline="tomorrow",
                now=now,
            )
        orig = settings_mod.config.COMMITMENTS_INJECTED_MAX
        settings_mod.config.COMMITMENTS_INJECTED_MAX = 5
        try:
            block = life.on_session_start("s1", now=now)
            assert block.startswith("Open commitments:")
            assert block.count("#") == 5
            life.record_commitment(
                "late extra",
                "Lucas said he will do the extra thing too.",
                now=now,
            )
            assert life.to_prompt_context() == block
        finally:
            settings_mod.config.COMMITMENTS_INJECTED_MAX = orig


def test_abandoned_never_surfaces():
    with tempfile.TemporaryDirectory() as td:
        life = _life(td)
        created = life.record_commitment("old thing", USER_WORDS)
        life.update_status(created["id"], "abandoned")
        assert life.get_active_commitments() == []
        assert life.on_session_start("s1") == ""


def test_overdue_surfaces_once_per_cooldown():
    orig_cd = settings_mod.config.OVERDUE_SURFACE_COOLDOWN_H
    settings_mod.config.OVERDUE_SURFACE_COOLDOWN_H = 2
    try:
        with tempfile.TemporaryDirectory() as td:
            life = _life(td)
            now = datetime(2026, 9, 10, 12, 0, tzinfo=TZ)
            past = (now - timedelta(hours=1)).isoformat()
            conn = life._conn()
            try:
                conn.execute(
                    """INSERT INTO user_commitment
                       (text, evidence, created_at, deadline, status)
                       VALUES (?, ?, ?, ?, 'active')""",
                    ("overdue thing", USER_WORDS, (now - timedelta(days=3)).isoformat(), past),
                )
                conn.commit()
            finally:
                conn.close()
            block1 = life.on_session_start("s1", now=now)
            assert "overdue thing" in block1
            life.on_session_end()
            block2 = life.on_session_start("s2", now=now + timedelta(hours=1))
            assert block2 == ""
            life.on_session_end()
            # Still inside 24h grace, outside 2h cooldown.
            block3 = life.on_session_start("s3", now=now + timedelta(hours=3))
            assert "overdue thing" in block3
    finally:
        settings_mod.config.OVERDUE_SURFACE_COOLDOWN_H = orig_cd


def test_auto_missed_after_grace_surfaces_once():
    with tempfile.TemporaryDirectory() as td:
        life = _life(td)
        now = datetime(2026, 9, 10, 12, 0, tzinfo=TZ)
        deadline = (now - timedelta(hours=25)).isoformat()
        conn = life._conn()
        try:
            conn.execute(
                """INSERT INTO user_commitment
                   (text, evidence, created_at, deadline, status)
                   VALUES (?, ?, ?, ?, 'active')""",
                ("missed thing", USER_WORDS, (now - timedelta(days=4)).isoformat(), deadline),
            )
            conn.commit()
        finally:
            conn.close()
        n = life.mark_overdue_missed(now=now)
        assert n == 1
        items = life.get_active_commitments(now=now)
        assert len(items) == 1
        assert items[0].status == "missed"
        block = life.on_session_start("s1", now=now)
        assert "missed thing" in block
        life.on_session_end()
        assert life.get_active_commitments(now=now) == []
        assert life.on_session_start("s2", now=now + timedelta(hours=1)) == ""


def test_inferred_wellbeing_excluded_from_trends():
    orig = settings_mod.config.WELLBEING_INFERRED_IN_TRENDS
    settings_mod.config.WELLBEING_INFERRED_IN_TRENDS = False
    try:
        with tempfile.TemporaryDirectory() as td:
            life = _life(td)
            life.log_wellbeing(0.2, "sounded tired", "inferred")
            life.log_wellbeing(0.9, "I feel great today", "user_reported")
            rows = life.wellbeing_trend_rows()
            assert len(rows) == 1
            assert rows[0].source == "user_reported"
            assert rows[0].score == 0.9
            bad = life.log_wellbeing(1.5, "I feel great today", "user_reported")
            assert bad["ok"] is False
            src = life.log_wellbeing(0.5, "I feel fine", "guess")
            assert src["ok"] is False
    finally:
        settings_mod.config.WELLBEING_INFERRED_IN_TRENDS = orig


def test_commitment_ref_count_path():
    with tempfile.TemporaryDirectory() as td:
        store = EpisodicMemory(db_path=Path(td) / "ep.db")
        _isolate_chroma(store)

        async def _run():
            await store.init()
            eid = await store.store(
                user_id=1, type="event", content="user promised", source="user_turn",
            )
            life = _life(td)
            payload = life.record_commitment(
                "ship the doc", USER_WORDS, source_episode_id=eid,
            )
            await store.add_episode_ref(
                eid, "user_life", f"commitment:{payload['id']}"
            )
            assert await store.reference_count(eid) == 1

        asyncio.run(_run())


def test_void_excluded_from_streak_and_never_overwritten():
    orig_path = settings_mod.config.midterm_db_path
    orig_len = settings_mod.config.ACTION_BIAS_STREAK_LEN
    settings_mod.config.ACTION_BIAS_STREAK_LEN = 3
    try:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "ep.db"
            settings_mod.config.midterm_db_path = db
            tid = insert_trial("turn-1", "h-v56-01", "do a thing")
            n = void_capability_absent_trials(
                ("h-v56-01",),
                before_ts=(utcnow() + timedelta(days=1)).isoformat(),
            )
            assert n == 1
            write_trial_verdict(tid, "HELD", "should not stick")
            import sqlite3
            conn = sqlite3.connect(str(db))
            try:
                verdict, reason = conn.execute(
                    "SELECT verdict, reason FROM action_bias_trial WHERE trial_id = ?",
                    (tid,),
                ).fetchone()
            finally:
                conn.close()
            assert verdict == "VOID"
            assert "capability absent" in reason

            model = {
                "hypotheses": [{
                    "id": "h-v56-01",
                    "tested": False,
                    "actionable": None,
                    "capability_verified": False,
                    "bias_text": "Name a file.",
                    "trial_log": [],
                }],
            }
            apply_verdict(model, "h-v56-01", tid, "VOID")
            apply_verdict(model, "h-v56-01", 2, "HELD")
            apply_verdict(model, "h-v56-01", 3, "HELD")
            apply_verdict(model, "h-v56-01", 4, "HELD")
            # VOID skipped like INVALID; three HELDs still resolve
            assert model["hypotheses"][0]["actionable"] is True
    finally:
        settings_mod.config.midterm_db_path = orig_path
        settings_mod.config.ACTION_BIAS_STREAK_LEN = orig_len


def test_violated_without_capability_does_not_write_actionable_false():
    orig = settings_mod.config.ACTION_BIAS_STREAK_LEN
    settings_mod.config.ACTION_BIAS_STREAK_LEN = 3
    try:
        model = {
            "hypotheses": [{
                "id": "h-v57-01",
                "tested": False,
                "actionable": None,
                "capability_verified": False,
                "bias_text": "Call record_user_commitment.",
                "trial_log": [],
            }],
            "active_bias_hypothesis_id": "h-v57-01",
        }
        apply_verdict(model, "h-v57-01", 1, "VIOLATED")
        apply_verdict(model, "h-v57-01", 2, "VIOLATED")
        apply_verdict(model, "h-v57-01", 3, "VIOLATED")
        h = model["hypotheses"][0]
        assert h["tested"] is False
        assert h["actionable"] is None
        assert h["needs_capability_review"] is True
        assert model["active_bias_hypothesis_id"] is None
        assert not eligible_hypotheses(model["hypotheses"])
    finally:
        settings_mod.config.ACTION_BIAS_STREAK_LEN = orig


def test_state_fields_default_empty_omits_affect_block():
    orig_f = settings_mod.config.STATE_FIELDS_INJECTED
    orig_d = settings_mod.config.DRIVES_INJECTED
    try:
        settings_mod.config.STATE_FIELDS_INJECTED = []
        settings_mod.config.DRIVES_INJECTED = []
        assert InternalState().to_prompt_context() == ""
        settings_mod.config.DRIVES_INJECTED = ["understand"]
        text = InternalState().to_prompt_context()
        assert "Internal state" not in text
        assert "Arousal:" not in text
    finally:
        settings_mod.config.STATE_FIELDS_INJECTED = orig_f
        settings_mod.config.DRIVES_INJECTED = orig_d


def test_ensure_hypothesis_fields_capability_gate():
    hyps = ensure_hypothesis_fields(
        [{"statement": "z", "confidence": 0.7, "tested": False}],
        version=12,
    )
    assert hyps[0]["capability_verified"] is False
    assert hyps[0]["needs_capability_review"] is False


def test_tool_executor_calls_record_user_commitment():
    with tempfile.TemporaryDirectory() as td:
        life = _life(td)

        async def _run():
            with patch("tools.user_life_tools._tracker", lambda: life), \
                 patch("tools.user_life_tools._source_episode_id", lambda: None):
                state = {
                    "user_id": 1,
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[{
                                "name": "record_user_commitment",
                                "args": {
                                    "commitment": "ship the scope doc",
                                    "evidence": USER_WORDS,
                                    "deadline": "tomorrow",
                                },
                                "id": "call-1",
                                "type": "tool_call",
                            }],
                        )
                    ],
                    "system_prompt": "",
                    "tool_calls_pending": True,
                    "response_text": "",
                    "tool_calls_made": 1,
                    "tier": "high",
                    "forced_answer": False,
                }
                out = await tool_executor(state)
                content = out["messages"][0].content
                payload = json.loads(content)
                assert payload["ok"] is True
                assert payload["id"]
                assert payload["deadline"]

        asyncio.run(_run())
