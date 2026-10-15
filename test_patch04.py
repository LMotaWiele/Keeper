"""SPEC-0004 stage 1: mocked unit checks only (no network or embeddings)."""
from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from config.settings import config
from core.timeutil import parse_iso
from goals.autonomous import AutonomousEngine
from goals.system import Goal, GoalSystem
from memory.semantic import SemanticMemory
from memory.similarity import cosine, nearest
from scripts.purge_synthesis_flood import flood_candidates

COPY = Path("/home/george-contis/keeper-data-copy-20261015")
T = parse_iso("2026-10-14T12:00:00+00:00")


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    # Export all four before any test can touch storage. Tests below use mocks
    # or their own temporary SQLite database, never the data copy itself.
    for env, attr, value in (
        ("DATA_DIR", "data_dir", COPY),
        ("MIDTERM_DB_PATH", "midterm_db_path", COPY / "episodic.db"),
        ("CHROMA_DB_PATH", "chroma_db_path", COPY / "chroma"),
        ("EMBEDDING_CACHE_DIR", "embedding_cache_dir", COPY / "fastembed"),
    ):
        monkeypatch.setenv(env, str(value))
        monkeypatch.setattr(config, attr, value)
    monkeypatch.setattr(config, "synth_gate_enabled", True, raising=False)
    monkeypatch.setattr(config, "synth_min_evidence_episodes", 2, raising=False)
    monkeypatch.setattr(config, "synth_min_evidence_days", 2, raising=False)
    monkeypatch.setattr(config, "semantic_dedup_sim", 0.9, raising=False)


def episode(i=1, at=T, **overrides):
    return {"id": i, "episode_id": f"ep-{i}", "created_at": at.isoformat(),
            "source": "user_turn", "type": "event", "content": "I stayed up past bedtime.", **overrides}


def harness(monkeypatch, episodes):
    import core.llm
    import goals.action_rating
    from memory.semantic import semantic

    goal = Goal(id="g", name="life", description="Support wellbeing", kind="unbounded")
    engine = AutonomousEngine(SimpleNamespace(active_instrumental=[goal]), None, None, None)
    engine._synthesis_inputs = AsyncMock(return_value=episodes)
    llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content='{}')))
    monkeypatch.setattr(core.llm, "get_llm", lambda *a, **k: llm)
    write = AsyncMock(return_value="doc-1")
    monkeypatch.setattr(semantic, "store", write)
    budget = SimpleNamespace(daily_limit_eur=10, USD_TO_EUR=1, spent_today_eur=0,
                             allows=lambda *a, **k: True)
    monkeypatch.setitem(sys.modules, "core.loop", SimpleNamespace(
        companion=SimpleNamespace(api_budget=budget, codebase=None)))
    monkeypatch.setattr(goals.action_rating, "rate_action", AsyncMock())
    seen = []

    def select(goal, unavailable):
        seen.append(set(unavailable))
        return None if "memory_synthesis" in unavailable else "memory_synthesis"

    monkeypatch.setattr(goals.action_rating, "select_action_type", select)
    return engine, goal, llm, write, seen


def test_v1_watermark_gate_selection_and_legacy(monkeypatch):
    engine, goal, llm, write, seen = harness(monkeypatch, [episode()])
    goal.last_synthesis_at = T
    goal.last_synthesis_episode_at = T
    assert asyncio.run(engine.take_action(1)) is None
    assert "memory_synthesis" in seen[-1]
    engine._synthesis_inputs.return_value.append(episode(2, T + timedelta(seconds=1)))
    asyncio.run(engine.take_action(1))
    assert "memory_synthesis" not in seen[-1]
    goal.last_synthesis_at = goal.last_synthesis_episode_at = None
    asyncio.run(engine.take_action(1))
    assert "memory_synthesis" not in seen[-1]


def test_v1_goal_watermarks_round_trip(tmp_path):
    system = GoalSystem()
    goal = Goal(id="g", name="life", description="wellbeing", last_synthesis_at=T,
                last_synthesis_episode_at=T - timedelta(days=1))
    system.instrumental = [goal]
    path = tmp_path / "goals.json"
    system.save(path)
    restored = GoalSystem()
    restored.load(path)
    assert restored.instrumental[0].last_synthesis_at == T
    assert restored.instrumental[0].last_synthesis_episode_at == T - timedelta(days=1)
    assert Goal.from_dict({"id": "legacy", "name": "old", "description": "old"}).last_synthesis_at is None


def test_v1_source_query_ignores_finite_window_and_legacy_internal_rows(monkeypatch, tmp_path):
    path = tmp_path / "episodes.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE episodes (id INTEGER, episode_id TEXT, user_id INTEGER, "
                   "content TEXT, source TEXT, type TEXT, created_at TEXT)")
        db.execute("INSERT INTO episodes VALUES (1, 'ep-1', 1, 'user', 'user_turn', 'event', ?)", (T.isoformat(),))
        for i in range(2, 110):
            db.execute("INSERT INTO episodes VALUES (?, ?, 1, 'internal', 'user_turn', 'self_observation', ?)",
                       (i, f"ep-{i}", (T + timedelta(seconds=i)).isoformat()))
        for i, kind in enumerate(("keeper_response", "environment_event", "tool_result", "autonomous_artifact", "research", "opinion"), 110):
            db.execute("INSERT INTO episodes VALUES (?, ?, 1, 'internal', 'user_turn', ?, ?)",
                       (i, f"ep-{i}", kind, T.isoformat()))
        db.execute("INSERT INTO episodes VALUES (200, 'ep-200', 1, 'auto', 'autonomous_artifact', 'event', ?)", (T.isoformat(),))
        db.execute("INSERT INTO episodes VALUES (201, 'ep-201', 2, 'other user', 'user_turn', 'event', ?)", (T.isoformat(),))
    engine = AutonomousEngine(None, None, SimpleNamespace(episodic=SimpleNamespace(db_path=str(path))), None)
    result = asyncio.run(engine._synthesis_inputs(1))
    assert [ep["episode_id"] for ep in result] == ["ep-1"]


@pytest.mark.parametrize("payload", [
    {"content": "pattern", "evidence": [0]},
    {"content": "pattern"},
    {"content": "pattern", "evidence": None},
    {"content": "pattern", "evidence": "0,1"},
    {"content": "pattern", "evidence": [True, 1]},
    {"content": "pattern", "evidence": [0, 1.0]},
    {"content": "pattern", "evidence": [0, 99]},
    {"content": "pattern", "evidence": [-1, 1]},
    {"content": "pattern", "evidence": [0, 0]},
    {"content": "pattern", "evidence": [[0], 1]},
    {"content": "pattern", "evidence": [{"id": 0}, 1]},
    {"content": 12, "evidence": [0, 1]},
    [{"content": "pattern", "evidence": [0, 1]}],
    "not json", None,
])
def test_v2_malformed_or_insufficient_evidence_rejects(monkeypatch, caplog, payload):
    engine, goal, llm, write, _ = harness(monkeypatch, [episode(), episode(2, T + timedelta(days=1))])
    llm.ainvoke.return_value.content = json.dumps(payload)
    with caplog.at_level("INFO"):
        result = asyncio.run(engine.take_action(1))
    assert result["action"].startswith("Synthesis rejected: insufficient evidence")
    assert "SYNTH_REJECT" in caplog.text
    write.assert_not_awaited()
    assert goal.last_synthesis_at is not None
    assert goal.last_synthesis_episode_at == T + timedelta(days=1)


def test_v2_same_amsterdam_day_rejects(monkeypatch):
    # Different UTC dates but the same Amsterdam local date.
    a = parse_iso("2026-10-14T23:30:00+00:00")
    b = parse_iso("2026-10-15T10:00:00+00:00")
    engine, goal, llm, write, _ = harness(monkeypatch, [episode(at=a), episode(2, b)])
    llm.ainvoke.return_value.content = json.dumps({"content": "pattern", "evidence": [0, 1]})
    result = asyncio.run(engine.take_action(1))
    assert "days=1" in result["action"]
    write.assert_not_awaited()


def test_v2_two_days_store_exact_cited_ids(monkeypatch):
    engine, goal, llm, write, _ = harness(monkeypatch, [episode(), episode(2, T + timedelta(days=1)), episode(3, T + timedelta(days=2))])
    llm.ainvoke.return_value.content = json.dumps({"content": "pattern", "type": "behavioral", "evidence": [0, 1, 0]})
    result = asyncio.run(engine.take_action(1))
    assert result["artifact_refs"] == ["semantic:doc-1"]
    kwargs = write.await_args.kwargs
    assert kwargs["source_episode_ids"] == ["ep-1", "ep-2"]
    assert kwargs["metadata"] == {"origin": "autonomous_synthesis", "goal": "life"}
    # Watermark all considered input, not just the model's cited subset.
    assert goal.last_synthesis_episode_at == T + timedelta(days=2)


def test_v4_ten_cycle_single_bedtime_incident(monkeypatch, tmp_path):
    engine, goal, llm, write, seen = harness(monkeypatch, [episode()])
    path = tmp_path / "incident.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE episodes (id INTEGER, episode_id TEXT, user_id INTEGER, "
                   "content TEXT, source TEXT, type TEXT, created_at TEXT)")
        db.execute("INSERT INTO episodes VALUES (1, 'ep-1', 1, 'I stayed up past bedtime', 'user_turn', 'event', ?)",
                   (T.isoformat(),))
        for i in range(2, 20):
            db.execute("INSERT INTO episodes VALUES (?, ?, 1, 'autonomous research', 'autonomous_artifact', 'research', ?)",
                       (i, f"ep-{i}", (T + timedelta(seconds=i)).isoformat()))
    engine.memory = SimpleNamespace(episodic=SimpleNamespace(db_path=str(path)))
    engine._synthesis_inputs = AutonomousEngine._synthesis_inputs.__get__(engine)
    llm.ainvoke.return_value.content = json.dumps({"content": "Repeatedly works too late", "evidence": [0]})
    for _ in range(10):
        asyncio.run(engine.take_action(1))
    write.assert_not_awaited()
    assert llm.ainvoke.await_count == 1
    assert all("memory_synthesis" in unavailable for unavailable in seen[1:])


class FakeCollection:
    def __init__(self):
        self.docs = {}
        self.query_calls = []

    def get(self, ids=None, include=None):
        ids = [i for i in (ids or list(self.docs)) if i in self.docs]
        return {"ids": ids, "documents": [self.docs[i]["content"] for i in ids],
                "metadatas": [dict(self.docs[i]["metadata"]) for i in ids],
                "embeddings": [self.docs[i]["embedding"] for i in ids]}

    def count(self):
        return len(self.docs)

    def upsert(self, ids, embeddings, documents, metadatas):
        for i, vec, text, meta in zip(ids, embeddings, documents, metadatas):
            self.docs[i] = {"embedding": vec, "content": text, "metadata": meta}

    def update(self, ids, metadatas):
        for i, meta in zip(ids, metadatas):
            self.docs[i]["metadata"] = meta

    def query(self, query_embeddings, n_results, where, include):
        self.query_calls.append(where)
        assert where == {"invalidated": {"$ne": "true"}}
        valid = [(i, d) for i, d in self.docs.items() if d["metadata"].get("invalidated") != "true"]
        ranked = sorted(valid, key=lambda pair: cosine(query_embeddings[0], pair[1]["embedding"]), reverse=True)[:n_results]
        return {"ids": [[i for i, _ in ranked]], "metadatas": [[d["metadata"] for _, d in ranked]],
                "distances": [[1 - cosine(query_embeddings[0], d["embedding"]) for _, d in ranked]]}


def semantic_fixture(monkeypatch):
    store, col = SemanticMemory(), FakeCollection()
    monkeypatch.setattr(store, "_collection", lambda uid: col)
    monkeypatch.setattr(store, "_register_refs", AsyncMock())
    vectors = {
        "I push past bedtime to keep working.": [1.0, 0.0],
        "I stay up late to continue my work.": [0.999, 0.001],
        "The user keeps working beyond bedtime.": [0.999, 0.002],
        "The user enjoys mountain hiking.": [0.0, 1.0],
    }
    monkeypatch.setattr(store, "_embed", AsyncMock(side_effect=lambda text: vectors[text]))
    return store, col, list(vectors)


def test_v3_unit_paraphrases_merge_confidence_only_with_new_source(monkeypatch):
    store, col, texts = semantic_fixture(monkeypatch)

    async def run():
        first = await store.store(1, texts[0], confidence=0.6)
        assert await store.store(1, texts[1]) == first
        assert col.count() == 1
        assert col.docs[first]["metadata"]["confidence"] == 0.6
        assert await store.store(1, texts[2], source_episode_ids=[1]) == first
        assert col.docs[first]["metadata"]["confidence"] == pytest.approx(0.7)
        assert await store.store(1, texts[0], source_episode_ids=["1"]) == first
        assert col.docs[first]["metadata"]["confidence"] == pytest.approx(0.7)
        await store.store(1, texts[0], source_episode_ids=[2])
        assert col.docs[first]["metadata"]["confidence"] == pytest.approx(0.8)
        assert set(json.loads(col.docs[first]["metadata"]["source_episode_ids"])) == {"1", "2"}
        assert col.docs[first]["metadata"]["reinforcement_count"] == 4
        assert await store.store(1, texts[3]) != first
        assert col.count() == 2

    asyncio.run(run())
    # Exact-hash reinforcement does not embed; each non-exact write embeds once.
    assert store._embed.await_count == 4


def test_v3_explicit_none_disables_similarity_and_invalidated_excluded(monkeypatch):
    store, col, texts = semantic_fixture(monkeypatch)

    async def run():
        first = await store.store(1, texts[0])
        second = await store.store(1, texts[1], dedup_sim=None)
        assert second != first
        assert not col.query_calls
        await store.invalidate(1, first)
        await store.invalidate(1, second)
        third = await store.store(1, texts[2])
        assert third not in {first, second}
        assert col.query_calls

    asyncio.run(run())


def test_similarity_pure_and_async_shared_embedder(monkeypatch):
    from memory.semantic import semantic
    from memory.similarity import embed_texts

    assert cosine([1, 0], [1, 0]) == 1
    assert cosine([0, 0], [1, 1]) == 0
    assert nearest([1, 0], [[0, 1], [1, 0]]) == (1, 1)
    assert nearest([1, 0], []) == (None, 0)
    with pytest.raises(ValueError):
        cosine([1], [1, 2])
    calls = []

    def embed(texts):
        calls.append(texts)
        return [[1, 0] for _ in texts]

    monkeypatch.setattr(semantic, "_get_embeddings", lambda: SimpleNamespace(embed=embed))
    assert asyncio.run(embed_texts([])) == []
    assert asyncio.run(embed_texts(["a", "b"])) == [[1.0, 0.0], [1.0, 0.0]]
    assert calls == [["a", "b"]]


def test_purge_a_union_b_only_source_less_neighbours():
    goals = [{"action_log": [
        {"action_type": "memory_synthesis", "artifact_refs": ["semantic:a"]},
        {"action_type": "web_research", "artifact_refs": ["semantic:other"]},
    ]}]
    docs = [
        {"id": i, "embedding": vec, "metadata": {"source_episode_ids": sources}}
        for i, vec, sources in (("a", [1, 0], "[]"), ("b", [1, 0.01], "[]"),
                                ("evidenced", [1, 0], '["ep-1"]'), ("other", [0, 1], "[]"))
    ]
    a, b = flood_candidates(goals, docs, 0.9)
    assert a == {"a"}
    assert a | b == {"a", "b"}
    assert flood_candidates([], docs, 0.9) == (set(), set())


@pytest.mark.parametrize("escape", ["midterm_db_path", "chroma_db_path", "embedding_cache_dir", None])
def test_v17_purge_path_guard_before_loader(monkeypatch, capsys, escape):
    from scripts.repair_common import require_isolated_paths
    from scripts.purge_synthesis_flood import run

    envs = {"midterm_db_path": "MIDTERM_DB_PATH", "chroma_db_path": "CHROMA_DB_PATH",
            "embedding_cache_dir": "EMBEDDING_CACHE_DIR"}
    if escape:
        outside = COPY.parent / "escape"
        monkeypatch.setattr(config, escape, outside)
        monkeypatch.setenv(envs[escape], str(outside))
    loaded = AsyncMock(side_effect=RuntimeError("loader reached after guard"))
    monkeypatch.setitem(sys.modules, "scripts.diag_common", SimpleNamespace(load_companion_no_loops=loaded))
    if escape:
        with pytest.raises((ValueError, RuntimeError, SystemExit)):
            asyncio.run(run())
        loaded.assert_not_awaited()
    else:
        assert require_isolated_paths() is config
        with pytest.raises(RuntimeError, match="loader reached after guard"):
            asyncio.run(run())
        loaded.assert_awaited_once()
    output = capsys.readouterr().out
    for name in ("DATA_DIR", "MIDTERM_DB_PATH", "CHROMA_DB_PATH", "EMBEDDING_CACHE_DIR"):
        assert name in output


def test_purge_dry_run_apply_and_idempotence(monkeypatch, tmp_path, capsys):
    import importlib
    semantic_mod = importlib.import_module("memory.semantic")
    import scripts.repair_common as repair_common
    from scripts.purge_synthesis_flood import run

    state = tmp_path / "state"
    state.mkdir()
    (state / "goals.json").write_text(json.dumps({"instrumental": [{"action_log": [
        {"action_type": "memory_synthesis", "artifact_refs": ["semantic:a"]}
    ]}], "completed": [], "abandoned": []}))
    fake_config = SimpleNamespace(data_dir=tmp_path, allowed_user_ids={1}, semantic_dedup_sim=0.9)
    monkeypatch.setattr(repair_common, "require_isolated_paths", lambda: fake_config)
    loaded = AsyncMock(return_value=SimpleNamespace())
    monkeypatch.setitem(sys.modules, "scripts.diag_common", SimpleNamespace(load_companion_no_loops=loaded))
    col = FakeCollection()
    col.upsert(ids=["a", "b", "other", "evidenced"],
               embeddings=[[1, 0], [1, 0.01], [0, 1], [1, 0]],
               documents=["seed", "paraphrase", "unrelated", "evidenced"],
               metadatas=[{"source_episode_ids": "[]", "invalidated": "false"},
                          {"source_episode_ids": "[]", "invalidated": "false"},
                          {"source_episode_ids": "[]", "invalidated": "false"},
                          {"source_episode_ids": '["ep-1"]', "invalidated": "false"}])
    store = SemanticMemory()
    monkeypatch.setattr(store, "_collection", lambda uid: col)
    monkeypatch.setattr(semantic_mod, "semantic", store)
    invalidate = AsyncMock(wraps=store.invalidate)
    monkeypatch.setattr(store, "invalidate", invalidate)
    asyncio.run(run())
    invalidate.assert_not_awaited()
    output = capsys.readouterr().out
    assert "|A|=1 |B|=2 |A ∪ B|=2" in output
    assert "other source-less docs left untouched: 1" in output
    asyncio.run(run(apply=True))
    assert invalidate.await_count == 2
    assert {i for i, d in col.docs.items() if d["metadata"]["invalidated"] == "true"} == {"a", "b"}
    assert all(call.kwargs["reason"] == "SPEC-0004: unevidenced autonomous synthesis"
               for call in invalidate.await_args_list)
    asyncio.run(run(apply=True))
    assert invalidate.await_count == 2
    assert "invalidated 0 document(s)" in capsys.readouterr().out


# Stage 2: all embeddings, search, loaders and episode writes are mocked.
from core.opinions import Opinion, OpinionOrigin, OpinionRegistry
from goals.research import ResearchEngine
from memory.episodic import EpisodicMemory
from scripts.compact_opinions import compact_snapshot


def make_opinion(oid, origin=OpinionOrigin.EXTERNAL, conviction=0.7, **kwargs):
    return Opinion(id=oid, domain=kwargs.pop("domain", oid),
                   position=kwargs.pop("position", f"position {oid}"),
                   reasoning="evidence", origin=origin, conviction=conviction, **kwargs)


@pytest.mark.parametrize("similar,enabled,external", [
    (True, True, True), (False, True, True),
    (True, False, True), (True, True, False),
])
def test_v6_research_merge_or_novel(monkeypatch, similar, enabled, external, caplog):
    import goals.research as research_mod
    import importlib
    search_mod = importlib.import_module("tools.web_search")
    import core.jev as jev

    monkeypatch.setattr(config, "research_merge_enabled", enabled)
    monkeypatch.setattr(config, "opinion_dup_sim", 0.9)
    registry = OpinionRegistry()
    existing = make_opinion("old", origin=OpinionOrigin.EXTERNAL if external else OpinionOrigin.COLLABORATIVE,
                            related_evidence=["source-0", "source-1"])
    registry.opinions[existing.id] = existing
    memory = SimpleNamespace(store_episode=AsyncMock(return_value="research-episode"))
    engine = ResearchEngine(registry, memory)
    engine._generate_queries = AsyncMock(return_value=["query"])
    engine._synthesize = AsyncMock(return_value={
        "position": "new position", "reasoning": "new evidence", "conviction": 0.99,
        "key_sources": ["source-1"] + [f"source-{i}" for i in range(2, 15)],
    })
    monkeypatch.setattr(search_mod, "run_search", AsyncMock(return_value=[{"content": "hit"}]))
    monkeypatch.setattr(jev, "jev_should_run", AsyncMock(return_value=True))
    embed = AsyncMock(return_value=[[1, 0], [1, 0] if similar else [0, 1]])
    monkeypatch.setattr(research_mod, "embed_texts", embed)
    from memory.episodic import episodic
    monkeypatch.setattr(episodic, "add_episode_ref", AsyncMock())
    with caplog.at_level("INFO"):
        result = asyncio.run(engine.research_topic("topic", user_id=1))
    if similar and enabled and external:
        assert result["merged_into"] == "old"
        assert "opinion_id" not in result and "episode_id" not in result
        assert len(registry.opinions) == 1
        assert existing.corroborated_count == 1
        assert existing.last_corroborated_at
        assert existing.conviction == 0.7
        assert existing.related_evidence == [f"source-{i}" for i in range(10)]
        memory.store_episode.assert_not_awaited()
        assert "RESEARCH_MERGE" in caplog.text
    else:
        assert result["opinion_id"] != "old"
        assert "merged_into" not in result
        assert len(registry.opinions) == 2
        memory.store_episode.assert_awaited_once()
    if not enabled or not external:
        embed.assert_not_awaited()


def test_v7_prune_external_first_and_protect(monkeypatch):
    monkeypatch.setattr(config, "max_tracked_opinions", 100)
    registry = OpinionRegistry()
    registry.opinions = {str(i): make_opinion(str(i), conviction=0.9) for i in range(98)}
    for i in ("collab-a", "collab-b"):
        registry.opinions[i] = make_opinion(i, OpinionOrigin.COLLABORATIVE, 0.1)
    added = asyncio.run(registry.form_opinion("new", "new", "why", OpinionOrigin.EXTERNAL, 0.01))
    assert len(registry.opinions) == 100
    assert {"collab-a", "collab-b", added.id} <= registry.opinions.keys()
    assert len([o for o in registry.opinions.values() if o.origin == OpinionOrigin.EXTERNAL]) == 98

    monkeypatch.setattr(config, "max_tracked_opinions", 2)
    registry.opinions = {i: make_opinion(i, OpinionOrigin.COLLABORATIVE, v)
                         for i, v in (("low", 0.1), ("mid", 0.5), ("protected", 0.01))}
    registry._prune(protect="protected")
    assert set(registry.opinions) == {"mid", "protected"}


def test_v8_unit_alert_requires_real_comparable_set(monkeypatch):
    monkeypatch.setattr(config, "drift_min_comparable", 3)
    registry = OpinionRegistry()
    registry._user_known_positions = {f"d{i}": "evidence needs verification" for i in range(5)}

    def alerts():
        return [a for a in registry.detect_drift() if "don't appear to disagree" in a]

    registry.opinions = {str(i): make_opinion(str(i), domain=f"other{i}") for i in range(5)}
    assert not alerts()
    registry.opinions = {str(i): make_opinion(str(i), domain=f"d{i}", position="evidence needs verification")
                         for i in range(3)}
    assert alerts()
    registry.opinions["2"].position = "hiking creates happiness"
    assert not alerts()
    registry.opinions["2"].position = "evidence needs verification"
    registry.opinions["2"].conviction = 0.5
    assert not alerts()
    registry.opinions["2"].conviction = 0.9
    registry.opinions["2"].origin = OpinionOrigin.COLLABORATIVE
    assert not alerts()
    registry.opinions["2"].origin = OpinionOrigin.INDEPENDENT
    assert alerts()
    registry._user_known_positions.pop("d4")  # Preserve existing outer >=5 gate.
    assert not alerts()


def test_v9_new_fields_round_trip_and_legacy():
    registry = OpinionRegistry()
    registry.opinions["a"] = make_opinion("a", corroborated_count=4, last_corroborated_at=T.isoformat())
    snapshot = registry.snapshot()
    restored = OpinionRegistry()
    restored.restore(json.loads(json.dumps(snapshot)))
    assert restored.snapshot() == snapshot
    del snapshot["opinions"]["a"]["corroborated_count"]
    del snapshot["opinions"]["a"]["last_corroborated_at"]
    restored.restore(snapshot)
    assert restored.opinions["a"].corroborated_count == 0
    assert restored.opinions["a"].last_corroborated_at is None


@pytest.mark.parametrize("cap", [0, 1])
def test_v10_format_autonomous_cap_before_ranking_budget(monkeypatch, cap):
    monkeypatch.setattr(config, "memory_block_autonomous_max", cap)
    monkeypatch.setattr(config, "FORGET_PROMPT_CHAR_BUDGET", 10000)
    store = EpisodicMemory()
    rows = [episode(1, source="autonomous_artifact", type="opinion", content="research_completed: alpha"),
            episode(2, source="autonomous_artifact", type="research", content="research_completed: beta"),
            episode(3, content="I enjoy visiting mountains.")]
    store.recall = AsyncMock(return_value=rows)
    block, ids, stats = asyncio.run(store.format_for_prompt(1, now=T))
    assert "I enjoy visiting mountains." in block
    assert sum(text in block for text in ("research_completed: alpha", "research_completed: beta")) == cap
    assert "ep-3" in ids and stats["ep_kept"] == 1 + cap


def compact_fixture():
    raw = {"opinions": {
        "a": make_opinion("a", conviction=0.9, corroborated_count=4, related_evidence=["first"]).to_dict(),
        "b": make_opinion("b", conviction=0.8, corroborated_count=2, related_evidence=["first", "second"]).to_dict(),
        "c": make_opinion("c", conviction=0.7).to_dict(),
        "conversation": make_opinion("conversation", OpinionOrigin.COLLABORATIVE).to_dict(),
    }, "user_known_positions": {"topic": "user stance"}}
    # Legacy non-external shape must not acquire new fields through restore/save.
    del raw["opinions"]["conversation"]["corroborated_count"]
    return raw


def test_compact_mass_idempotence_and_raw_nonexternals():
    raw = compact_fixture()
    before = json.dumps(raw["opinions"]["conversation"])
    compacted, clusters = compact_snapshot(raw, [[1, 0], [1, 0], [0, 1]], 0.9)
    assert clusters == [["a", "b"], ["c"]]
    assert sorted(i for cluster in clusters for i in cluster) == ["a", "b", "c"]
    assert len(compacted["opinions"]) == 3
    canonical = compacted["opinions"]["a"]
    assert canonical["corroborated_count"] == 7  # 4 + (2 + removed member unit)
    assert canonical["related_evidence"] == ["first", "second"]
    for key in ("id", "position", "domain", "conviction"):
        assert canonical[key] == raw["opinions"]["a"][key]
    assert json.dumps(compacted["opinions"]["conversation"]) == before
    assert compacted["user_known_positions"] == raw["user_known_positions"]
    again, _ = compact_snapshot(compacted, [[1, 0], [0, 1]], 0.9)
    assert again == compacted
    assert raw["opinions"]["a"]["corroborated_count"] == 4


@pytest.mark.parametrize("apply", [False, True])
@pytest.mark.parametrize("escape", ["midterm_db_path", "chroma_db_path", "embedding_cache_dir", None])
def test_v17_compact_guard_before_loader(monkeypatch, capsys, escape, apply):
    from scripts.compact_opinions import run
    if escape:
        monkeypatch.setattr(config, escape, COPY.parent / "escape")
    loaded = AsyncMock(side_effect=RuntimeError("loader reached after guard"))
    monkeypatch.setitem(sys.modules, "scripts.diag_common", SimpleNamespace(load_companion_no_loops=loaded))
    if escape:
        with pytest.raises(ValueError):
            asyncio.run(run(apply=apply))
        loaded.assert_not_awaited()
    else:
        with pytest.raises(RuntimeError, match="loader reached after guard"):
            asyncio.run(run(apply=apply))
        loaded.assert_awaited_once()
    output = capsys.readouterr().out
    for name in ("DATA_DIR", "MIDTERM_DB_PATH", "CHROMA_DB_PATH", "EMBEDDING_CACHE_DIR"):
        assert name in output


def test_compact_dry_apply_embedded_consistency_and_second_apply(monkeypatch, tmp_path, capsys):
    import scripts.compact_opinions as compact_mod
    import core.atomic as atomic
    state = tmp_path / "state"
    state.mkdir()
    snapshot = compact_fixture()
    opinion_path, model_path = state / "opinions.json", state / "self_model.json"
    opinion_path.write_text(json.dumps(snapshot))
    model_path.write_text(json.dumps({"version": 13, "_opinions": snapshot}))
    for attr, value in (("data_dir", tmp_path), ("midterm_db_path", tmp_path / "db"),
                        ("chroma_db_path", tmp_path / "chroma"), ("embedding_cache_dir", tmp_path / "cache")):
        monkeypatch.setattr(config, attr, value)
    monkeypatch.setattr(config, "opinion_dup_sim", 0.9)
    monkeypatch.setitem(sys.modules, "scripts.diag_common", SimpleNamespace(load_companion_no_loops=AsyncMock()))
    monkeypatch.setattr(compact_mod, "embed_texts", AsyncMock(side_effect=lambda texts: [
        [0, 1] if text == "position c" else [1, 0] for text in texts]))
    writes = []
    real_write = atomic.atomic_write_text

    def write(path, text):
        writes.append(path)
        real_write(path, text)

    monkeypatch.setattr(atomic, "atomic_write_text", write)
    original = opinion_path.read_bytes(), model_path.read_bytes()
    asyncio.run(compact_mod.run())
    assert (opinion_path.read_bytes(), model_path.read_bytes()) == original
    assert not writes
    assert "canonical=a members=2" in capsys.readouterr().out
    asyncio.run(compact_mod.run(apply=True))
    result = json.loads(opinion_path.read_text())
    assert json.loads(model_path.read_text())["_opinions"] == result
    assert len(result["opinions"]) == 3
    assert result["opinions"]["a"]["corroborated_count"] == 7
    assert json.dumps(result["opinions"]["conversation"]) == json.dumps(snapshot["opinions"]["conversation"])
    restored = OpinionRegistry()
    restored.load(opinion_path)
    assert len(restored.opinions) == 3
    assert writes == [opinion_path, model_path]
    applied = opinion_path.read_bytes(), model_path.read_bytes()
    asyncio.run(compact_mod.run(apply=True))
    assert (opinion_path.read_bytes(), model_path.read_bytes()) == applied
    assert writes == [opinion_path, model_path]


def test_compact_preserves_literal_nonexternal_json_tokens():
    from scripts.compact_opinions import _updated_json
    raw_record = '{ "origin" : "collaborative", "position": "caf\\u00e9", "conviction": 0.500 }'
    original = '{"opinions": {"user": ' + raw_record + ', "a": {"origin":"external"}, "b": {"origin":"external"}}, "user_known_positions": { "x" : "y" }}'
    updated = json.loads(original)
    del updated["opinions"]["b"]
    updated["opinions"]["a"]["corroborated_count"] = 1
    encoded = _updated_json(original, updated)
    assert json.loads(encoded) == updated
    assert raw_record in encoded
    assert '{ "x" : "y" }' in encoded
    assert _updated_json(encoded, updated) == encoded


def test_research_merged_summary_does_not_claim_new_artifacts(monkeypatch):
    import goals.action_rating
    engine, goal, _, _, _ = harness(monkeypatch, [])
    engine.research = SimpleNamespace(research_topic=AsyncMock(return_value={
        "merged_into": "canonical", "position": "existing position"}))
    engine.self_model = SimpleNamespace(observe=AsyncMock())
    monkeypatch.setattr(goals.action_rating, "select_action_type", lambda *a, **k: "web_research")
    result = asyncio.run(engine.take_action(1))
    assert result["action"] == "Research corroborated opinion canonical"
    assert result["artifact_refs"] == []
    # Stage 2 deliberately preserves observe; its gate belongs to stage 3.
    engine.self_model.observe.assert_awaited_once()
