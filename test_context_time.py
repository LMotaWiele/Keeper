"""Prompt de-dupe and temporal binding — clock-frozen, no live LLM."""
from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from environment.time_sense import TimeSense
from memory.episodic import EpisodicMemory
from memory.timebind import resolve_event_time
from memory.working import WorkingMemory
from core.timeutil import age_phrase, memory_time_prefix, to_user_local

TZ = ZoneInfo("Europe/Amsterdam")
UTC = timezone.utc
UID = 1


def _isolate_chroma(store: EpisodicMemory) -> None:
    async def _nov(user_id, content):
        return 1.0

    async def _up(**kwargs):
        return None

    store._compute_novelty = _nov  # type: ignore[method-assign]
    store._chroma_upsert = _up  # type: ignore[method-assign]
    store._chroma_delete = lambda *a, **k: None  # type: ignore[method-assign]


def _store(path: Path) -> EpisodicMemory:
    store = EpisodicMemory(db_path=path)
    _isolate_chroma(store)
    asyncio.run(store.init())
    return store


def test_working_memory_keeps_recent_assistants_and_pins_old_users():
    wm = WorkingMemory(dialogue_window=12, pin_capacity=8)
    base = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    old_texts = [
        "commissioned mural for my aunt in Rotterdam last spring",
        "renounced theism after sitting with the problem of evil",
        "negotiating an October rental in the slow housing market",
        "Jev skip-gates replaced a pile of text-generating calls",
        "income runway is the only non-negotiable this quarter",
        "Amsterdam nights still feel longer than the UTC clock",
        "the unnamed conflict still organizes every other goal",
        "DeepSeek flash was the previous low-tier conversation model",
    ]
    for i, text in enumerate(old_texts):
        wm.add(
            UID, "user", text,
            salience=1.0,
            timestamp=base + timedelta(hours=i),
        )
    recent = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
    for i in range(12):
        wm.add(
            UID, "assistant", f"recent assistant {i}",
            salience=0.5,
            timestamp=recent + timedelta(minutes=i),
        )
    dialogue, pins = wm.split_window(UID)
    assert len(dialogue) == 12
    assert all(i.role == "assistant" for i in dialogue)
    assert len(pins) == 8
    assert all(i.role == "user" for i in pins)
    assert all(p not in dialogue for p in pins)


def test_session_end_drops_unused_old_pins_and_calls_decay():
    wm = WorkingMemory(dialogue_window=12, pin_capacity=8)
    now = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
    old = now - timedelta(days=15)
    wm.add(UID, "user", "stale pin about a forgotten commission", salience=1.0, timestamp=old)
    for i in range(12):
        wm.add(
            UID, "user", f"today turn {i} unique xyz{i}",
            salience=0.7,
            timestamp=now - timedelta(minutes=12 - i),
        )
    dialogue, pins = wm.split_window(UID)
    assert any("stale pin" in p.content for p in pins)
    for item in wm._buf(UID):
        if "stale pin" in item.content:
            item.last_used_at = old
            item.salience = 0.9
    wm.on_session_end(UID, now=now)
    _, pins_after = wm.split_window(UID)
    assert not any("stale pin" in p.content for p in pins_after)
    # decay ran — recent user salience dropped from 0.7
    recent = [i for i in wm.get_items(UID) if "today turn" in i.content]
    assert recent
    assert all(i.salience < 0.7 for i in recent)

    # used-this-session 15-day pin is kept
    wm2 = WorkingMemory(dialogue_window=12, pin_capacity=8)
    wm2.add(UID, "user", "old but shown pin unique-aaa", salience=1.0, timestamp=old)
    for i in range(12):
        wm2.add(
            UID, "user", f"fresh {i} unique-bbb{i}",
            salience=0.7,
            timestamp=now - timedelta(minutes=12 - i),
        )
    for item in wm2._buf(UID):
        if "shown pin" in item.content:
            item.last_used_at = now
    wm2.on_session_end(UID, now=now)
    _, pins2 = wm2.split_window(UID)
    assert any("shown pin" in p.content for p in pins2)


def test_langchain_messages_use_utterance_local_time_not_now():
    wm = WorkingMemory(dialogue_window=12, pin_capacity=8)
    uttered = datetime(2026, 9, 7, 14, 39, tzinfo=UTC)
    now = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
    wm.add(
        UID, "user",
        "I had to work on a commissioned project yesterday",
        timestamp=uttered,
    )
    msgs = wm.to_langchain_messages(UID, now=now)
    assert len(msgs) == 1
    content = msgs[0]["content"]
    assert "2026-09-07" in content
    assert "16:39" in content
    assert "CEST" in content
    # prefix is the utterance date, not today
    assert not content.startswith("[2026-09-18")


def test_format_for_prompt_drops_transcript_echo_keeps_fact():
    with tempfile.TemporaryDirectory() as td:
        store = _store(Path(td) / "ep.db")
        uttered = datetime(2026, 9, 18, 11, 39, tzinfo=UTC)
        user_text = (
            "I have a place for October, the commission is stalled "
            "at the very last steps."
        )

        async def _run():
            await store.store(
                UID, "event", f"User said: {user_text}",
                tags=["user_input"], importance=0.9, created_at=uttered,
            )
            await store.store(
                UID, "self_observation",
                f"self_observation: Given '{user_text[:40]}', responded with: ok",
                importance=0.9, created_at=uttered,
            )
            await store.store(
                UID, "event", f"Responded: First, good that October is locked down.",
                tags=["response"], importance=0.5, created_at=uttered,
            )
            await store.store(
                UID, "fact", user_text, importance=0.8, created_at=uttered,
            )
            block, used, stats = await store.format_for_prompt(
                UID,
                exclude_texts=[user_text],
                now=datetime(2026, 9, 18, 12, 0, tzinfo=UTC),
            )
            return block, used, stats

        block, used, stats = asyncio.run(_run())
        assert "User said:" not in block
        assert "self_observation" not in block
        assert "[fact]" in block
        assert "2026-09-18" in block
        assert "place for October" in block
        assert stats["dropped_overlap"] >= 1


def test_prompt_assembly_does_not_increment_recall_count():
    from memory import MemorySystem
    from memory.working import WorkingMemory

    with tempfile.TemporaryDirectory() as td:
        store = _store(Path(td) / "ep.db")
        wm = WorkingMemory()
        ms = MemorySystem(working=wm, episodic_store=store)

        async def _run():
            eid = await store.store(
                UID, "fact",
                "Committed to shipping today or tomorrow",
                importance=0.8,
                created_at=datetime(2026, 9, 9, 11, 29, tzinfo=UTC),
            )
            for _ in range(20):
                await ms.build_memory_context(UID, "how is the commission")
            async with __import__("aiosqlite").connect(store.db_path) as db:
                db.row_factory = __import__("aiosqlite").Row
                async with db.execute(
                    "SELECT recall_count, used_count FROM episodes WHERE episode_id = ?",
                    (eid,),
                ) as cur:
                    row = await cur.fetchone()
            return dict(row)

        row = asyncio.run(_run())
        assert row["recall_count"] == 0
        assert row["used_count"] >= 1


def test_resolver_yesterday_is_previous_local_date():
    uttered = datetime(2026, 9, 7, 14, 39, tzinfo=UTC)
    event_at, local_date = resolve_event_time(
        "I had to work on a commissioned project yesterday and that went well",
        uttered,
    )
    assert local_date == "2026-09-06"
    now = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
    prefix = memory_time_prefix(event_at, now)
    assert "2026-09-06" in prefix
    assert "ago" in prefix
    assert "yesterday" not in prefix  # not unbound; age is Nd ago from Sep 18
    assert age_phrase(event_at, now).endswith("d ago")


def test_resolver_tonight_stays_on_utterance_local_date():
    uttered = datetime(2026, 9, 6, 16, 37, tzinfo=UTC)
    event_at, local_date = resolve_event_time(
        "Has a paid commission that needs to be finished tonight.",
        uttered,
    )
    assert local_date == "2026-09-06"
    local = to_user_local(event_at)
    assert local.date().isoformat() == "2026-09-06"
    now = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
    assert "tonight" not in memory_time_prefix(event_at, now)


def test_environment_clock_is_user_local():
    ts = TimeSense()
    now = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)  # Friday 14:00 CEST
    block = ts.to_prompt_context(now=now)
    assert "CEST" in block or "CET" in block
    assert "Friday" in block
    assert "UTC" in block
    assert "14:00" in block  # Amsterdam
    assert "12:00" in block  # UTC in parentheses


def test_backfill_fills_event_local_date_from_created_at():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ep.db"
        store = _store(path)
        created = "2026-09-07T14:39:00+00:00"
        content = "I had to work on a commissioned project yesterday"

        con = sqlite3.connect(path)
        con.execute(
            "INSERT INTO episodes (user_id, type, content, created_at, importance) "
            "VALUES (?, ?, ?, ?, ?)",
            (UID, "event", content, created, 0.5),
        )
        con.commit()
        con.close()

        n = asyncio.run(store.backfill_event_times())
        assert n >= 1
        con = sqlite3.connect(path)
        row = con.execute(
            "SELECT event_at, event_local_date FROM episodes WHERE content = ?",
            (content,),
        ).fetchone()
        con.close()
        assert row[1] == "2026-09-06"
        now = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
        event_at = datetime.fromisoformat(row[0])
        prefix = memory_time_prefix(event_at, now)
        assert "2026-09-06" in prefix
        assert "ago" in prefix
