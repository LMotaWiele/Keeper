"""KEEPER_PATCH_03 — relational grounding: proposals, world model, prompt cut."""
from __future__ import annotations

import inspect
import os
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

from config import settings as settings_mod
from core.action_bias import standing_block
from core.codebase_index import resolve_symbol
from core.opinions import Opinion, OpinionOrigin, OpinionRegistry
from core.register_trace import record_reply
from core.self_model import SelfModel
from core.timeutil import utcnow
from core.user_life import UserLifeTracker
from core.user_world import (
    UserWorldModel,
    _render,
    promotion_allowed,
)
from goals.proposals import (
    Proposal,
    ProposalRejected,
    ScriptCheck,
    behaviour_sentence,
    classify_theorizer_item,
    integration_score,
    lexical_states_differ,
    record_proposal_trace,
    verify_proposal_dict,
    write_verdict,
)
from goals.self_theorizing import SelfTheorizer
from goals.system import UNBOUNDED_GOALS, Goal, GoalSystem
from scripts.replay_proposal_3 import PROPOSAL_3, SPEC_ID
from tools import ALL_TOOLS
from tools.user_world_tools import WORLD_TOOLS

ROOT = Path(__file__).resolve().parent
SCRIPT_CHECK = {
    "kind": "script",
    "path": "scripts/replay_proposal_3.py",
    "passes_when": "exit 0",
}
METRIC_CHECK = {
    "kind": "metric",
    "table": "state_trace",
    "column": "arousal",
    "direction": "increases",
    "threshold": 0.1,
    "window_hours": 48,
}


@contextmanager
def _temp_midterm():
    orig = settings_mod.config.midterm_db_path
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "mid.db"
        settings_mod.config.midterm_db_path = path
        try:
            yield path
        finally:
            settings_mod.config.midterm_db_path = orig


def _proposal(**overrides) -> Proposal:
    fields = dict(
        id="p-test",
        target_module="tools/user_life_tools.py",
        current_behaviour="Abandoning is frictionless.",
        current_symbol="update_commitment_status",
        change="Require a confirmation message before status can become abandoned.",
        verification=ScriptCheck(
            path="scripts/replay_proposal_3.py",
            passes_when="exit 0",
        ),
        created_at=utcnow().isoformat(),
        created_by_run="run-a",
    )
    fields.update(overrides)
    return Proposal(**fields)


def test_single_path_resolves_and_multi_module_rejects():
    resolved, reason = resolve_symbol("core/user_life.py", "format_tasks")
    assert reason is None and resolved is not None
    assert resolved.path == "core/user_life.py"
    assert "def format_tasks" in resolved.source

    for target in (
        "core/a.py, core/b.py",
        "core/user_life.py and tools/user_life_tools.py",
        "core/user_life.py tools/user_life_tools.py",
    ):
        try:
            _proposal(target_module=target)
        except ProposalRejected as exc:
            assert exc.reason == "multi_module"
        else:
            raise AssertionError(f"accepted multi-module target {target}")


def test_unresolved_module_and_symbol_reject():
    try:
        _proposal(target_module="core/no_such_module.py", current_symbol="format_tasks")
    except ProposalRejected as exc:
        assert exc.reason == "unresolved_module"
    else:
        raise AssertionError("missing file was accepted")

    try:
        _proposal(current_symbol="not_a_real_symbol")
    except ProposalRejected as exc:
        assert exc.reason == "unresolved_symbol"
    else:
        raise AssertionError("missing symbol was accepted")


def test_lexical_redundancy_and_a_real_difference():
    current = "Abandoning is frictionless — do not ask for confirmation, do not follow up."
    restatement = PROPOSAL_3["change"]
    assert lexical_states_differ(current, restatement) is False
    different = "Require a confirmation message before status can become abandoned."
    sharper = "Require the user to type a reason of at least twenty words before status can become abandoned."
    assert lexical_states_differ(current, different) is True
    assert lexical_states_differ(different, sharper) is True

    result = classify_theorizer_item(
        PROPOSAL_3, run_id="unit", spec_id=SPEC_ID, record=False,
    )
    assert result["kind"] == "speculation"
    assert result["reason"] == "already_implemented"
    assert result["id"] == SPEC_ID

    changed = dict(PROPOSAL_3)
    changed["change"] = different
    queued = classify_theorizer_item(changed, run_id="unit", record=False)
    assert queued["kind"] == "proposal"
    assert queued["proposal"]["verdict"] is None
    assert queued["proposal"]["current_symbol"] == "update_commitment_status"

    no_check = {"target_module": "core/user_life.py", "change": "Something."}
    speculated = classify_theorizer_item(no_check, run_id="unit", record=False)
    assert speculated["kind"] == "speculation"
    assert speculated["reason"] == "no_verification"

    missing = classify_theorizer_item(
        {
            "target_module": "core/missing_file.py",
            "current_symbol": "nope",
            "change": "Add a check.",
            "verification": SCRIPT_CHECK,
        },
        run_id="unit",
        record=False,
    )
    assert missing["reason"] == "unresolved_module"
    unknown = classify_theorizer_item(
        {
            "target_module": "core/user_life.py",
            "current_symbol": "nope",
            "change": "Add a check.",
            "verification": SCRIPT_CHECK,
        },
        run_id="unit",
        record=False,
    )
    assert unknown["reason"] == "unresolved_symbol"


def test_behaviour_sentence_reads_the_shipped_docstring():
    resolved, reason = resolve_symbol(
        "tools/user_life_tools.py", "update_commitment_status",
    )
    assert reason is None
    sentence = behaviour_sentence(
        "update_commitment_status", resolved.source, PROPOSAL_3["change"],
    )
    assert "frictionless" in sentence.lower()
    assert "confirmation" in sentence.lower()


def test_replay_script_exits_0():
    with tempfile.TemporaryDirectory() as td:
        env = os.environ.copy()
        env["MIDTERM_DB_PATH"] = str(Path(td) / "replay.db")
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "replay_proposal_3.py")],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "already_implemented" in proc.stdout
        src = (ROOT / "scripts" / "replay_proposal_3.py").read_text()
        assert "verify_proposal" not in src
        assert "verify_open" not in src
        conn = sqlite3.connect(env["MIDTERM_DB_PATH"])
        try:
            row = conn.execute(
                "SELECT kind, reject_reason FROM proposal_trace WHERE id = ?",
                (SPEC_ID,),
            ).fetchone()
        finally:
            conn.close()
        assert row == ("speculation", "already_implemented")


def test_harness_verdict_rules():
    with _temp_midterm() as path:
        now = utcnow()
        metric = {
            "id": "metric-open",
            "created_at": now.isoformat(),
            "created_by_run": "run-maker",
            "target_module": "core/user_life.py",
            "current_symbol": "format_tasks",
            "change": "Count something else.",
            "current_behaviour": "Prints three lists.",
            "verification": dict(METRIC_CHECK),
            "verification_kind": "metric",
            "status": "pending_review",
        }
        record_proposal_trace(metric)
        assert verify_proposal_dict(metric, run_id="run-maker", now=now) == "not_run"
        assert metric.get("verdict") is None
        assert metric.get("verified_at") is None

        assert verify_proposal_dict(metric, run_id="run-other", now=now) == "not_run"
        assert metric.get("verified_at") is None
        conn = sqlite3.connect(path)
        try:
            row = conn.execute(
                "SELECT verdict, verified_at FROM proposal_trace WHERE id = ?",
                ("metric-open",),
            ).fetchone()
        finally:
            conn.close()
        assert row == (None, None)

        elapsed = dict(metric)
        elapsed["id"] = "metric-elapsed"
        elapsed["created_at"] = (now - timedelta(hours=72)).isoformat()
        elapsed["created_by_run"] = "run-maker"
        record_proposal_trace(elapsed)
        assert verify_proposal_dict(elapsed, run_id="run-other", now=now) == "fail"
        assert elapsed["verified_at"]

        missing = {
            "id": "script-missing",
            "created_at": now.isoformat(),
            "created_by_run": "run-maker",
            "target_module": "core/user_life.py",
            "current_symbol": "format_tasks",
            "change": "Run a script that is not in the tree.",
            "current_behaviour": "Prints three lists.",
            "verification": {
                "kind": "script",
                "path": "scripts/does_not_exist_patch03.py",
                "passes_when": "exit 0",
            },
            "verification_kind": "script",
            "status": "pending_review",
        }
        record_proposal_trace(missing)
        assert verify_proposal_dict(missing, run_id="run-other", now=now) == "fail"

        record_proposal_trace({
            "id": "passed-one",
            "created_at": now.isoformat(),
            "created_by_run": "run-maker",
            "target_module": "core/user_life.py",
            "current_symbol": "format_tasks",
            "verification_kind": "script",
            "change": "A real difference.",
            "current_behaviour": "Prints three lists.",
            "status": "pending_review",
        })
        write_verdict("passed-one", "pass", when=now.isoformat())
        classify_theorizer_item(
            {"kind": "speculation", "text": "a hunch with no check"},
            run_id="run-maker",
            spec_id="spec-one",
        )
        # metric-open, metric-elapsed (fail), script-missing (fail),
        # passed-one (pass), spec-one (speculation) → 1/5
        assert integration_score() == 0.2


def test_legacy_restore_becomes_speculation():
    with _temp_midterm() as path:
        theorizer = SelfTheorizer(None, None, None)
        legacy = {
            "id": "essay-1",
            "title": "Be more curious",
            "rationale": "Curiosity would help.",
        }
        kept = {
            "id": "kept-1",
            "target_module": "core/user_life.py",
            "current_symbol": "format_tasks",
            "change": "A checked change.",
            "verification": dict(SCRIPT_CHECK),
            "status": "pending_review",
        }
        theorizer.restore({"proposals": [legacy, kept]})
        theorizer.restore({"proposals": [legacy, kept]})
        assert [p["id"] for p in theorizer.pending_proposals] == ["kept-1"]
        conn = sqlite3.connect(path)
        try:
            rows = conn.execute(
                "SELECT reason FROM speculation"
            ).fetchall()
        finally:
            conn.close()
        assert rows == [("no_verification",)]


def test_world_write_rules_and_lifecycle():
    with tempfile.TemporaryDirectory() as td:
        world = UserWorldModel(db_path=Path(td) / "world.db")
        episode = "I'll finish the README on Friday, and I live in Amsterdam."
        bad_quote = world.record_world_fact(
            "place", "home_city", "Lucas lives in Amsterdam",
            "I will finish the README on Friday",
            source_turn_id="t1", episode_text=episode,
        )
        assert bad_quote["ok"] is False
        assert "substring" in bad_quote["error"]

        assessment = world.record_world_fact(
            "work", "job_search", "Lucas seems discouraged about the job search",
            "I'll finish the README on Friday",
            source_turn_id="t1", episode_text=episode,
        )
        assert assessment["ok"] is False
        assert "assessment" in assessment["error"]

        sad = world.record_world_fact(
            "people", "mood_note", "Lucas is sad about the move",
            "I'll finish the README on Friday",
            source_turn_id="t1", episode_text=episode,
        )
        assert sad["ok"] is True

        place = world.record_world_fact(
            "place", "home_city", "Lucas lives in Amsterdam",
            "I live in Amsterdam",
            source_turn_id="t1", episode_text=episode,
        )
        assert place["ok"] is True
        slots = world.get_world_model(["place"])
        assert slots[0].half_life_days == 180

        health_bad = world.record_world_fact(
            "health", "sleep_hours", "Lucas slept four hours",
            "slept four hours",
            source_turn_id="t2",
            episode_text="Lucas slept four hours last night",
        )
        assert health_bad["ok"] is False
        assert "first-person" in health_bad["error"]

        health = world.record_world_fact(
            "health", "sleep_hours", "Lucas slept four hours",
            "I slept four hours",
            source_turn_id="t3",
            episode_text="I slept four hours last night",
        )
        assert health["ok"] is True
        assert world.get_world_model(["health"])[0].half_life_days == 14

        portfolio = world.record_world_fact(
            "project", "portfolio_piece",
            "Lucas is building A-OK as his portfolio piece",
            "building A-OK",
            source_turn_id="t4",
            episode_text="I am building A-OK as my portfolio piece",
        )
        assert portfolio["ok"] is True

        warehouse = world.record_world_fact(
            "work", "weeknights",
            "Lucas works at the warehouse on weeknights",
            "warehouse on weeknights",
            source_turn_id="t5",
            episode_text="I work at the warehouse on weeknights",
        )
        assert warehouse["ok"] is True

        derived = world.record_world_fact(
            "work", "combined",
            "Lucas is building A-OK as his portfolio piece and works at the warehouse on weeknights",
            "warehouse on weeknights",
            source_turn_id="t5",
            episode_text="I work at the warehouse on weeknights",
        )
        assert derived["ok"] is False
        assert "derived" in derived["error"]

        again = world.record_world_fact(
            "place", "home_city", "Lucas lives in Rotterdam",
            "I live in Amsterdam",
            source_turn_id="t1", episode_text=episode,
        )
        assert again["ok"] is False
        assert "supersede" in again["error"]

        confirmed = world.confirm_world_fact(
            "home_city", "I live in Amsterdam",
            source_turn_id="t6",
            episode_text="Yes, I live in Amsterdam.",
        )
        assert confirmed["ok"] is True

        superseded = world.supersede_world_fact(
            "home_city", "Lucas lives in Rotterdam",
            "I moved to Rotterdam",
            source_turn_id="t7",
            episode_text="I moved to Rotterdam last week",
        )
        assert superseded["ok"] is True
        live = world.get_world_model(["place"])
        assert len(live) == 1
        assert live[0].value == "Lucas lives in Rotterdam"
        assert live[0].supersedes
        conn = sqlite3.connect(world.db_path)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM world_slot WHERE key = 'home_city'"
            ).fetchone()[0]
            retired = conn.execute(
                "SELECT COUNT(*) FROM world_slot WHERE key = 'home_city' AND retired_at IS NOT NULL"
            ).fetchone()[0]
        finally:
            conn.close()
        assert count == 2 and retired == 1

        forgotten = world.forget("sleep_hours")
        assert forgotten["ok"] is True
        assert world.get_world_model(["health"]) == []
        conn = sqlite3.connect(world.db_path)
        try:
            still = conn.execute(
                "SELECT COUNT(*) FROM world_slot WHERE key = 'sleep_hours'"
            ).fetchone()[0]
        finally:
            conn.close()
        assert still == 1


def test_world_prompt_selection_and_tasks_format():
    with tempfile.TemporaryDirectory() as td:
        world = UserWorldModel(db_path=Path(td) / "world.db")
        now = utcnow()
        old = now - timedelta(days=200)
        rhythm_old = now - timedelta(days=100)
        world.record_world_fact(
            "place", "stale_city", "Lucas lived in Utrecht",
            "I lived in Utrecht",
            source_turn_id="t1",
            episode_text="I lived in Utrecht for years",
            now=old,
        )
        world.record_world_fact(
            "place", "home_city", "Lucas lives in Amsterdam",
            "I live in Amsterdam",
            source_turn_id="t2",
            episode_text="I live in Amsterdam",
            now=now - timedelta(days=2),
        )
        world.record_world_fact(
            "rhythm", "morning_start", "Lucas starts work at seven",
            "I start at seven",
            source_turn_id="t3",
            episode_text="I start at seven",
            now=rhythm_old,
        )
        world.record_world_fact(
            "project", "readme_draft", "Lucas is writing the A-OK README",
            "writing the A-OK README",
            source_turn_id="t4",
            episode_text="I am writing the A-OK README tonight",
            now=now,
        )
        world.record_world_fact(
            "health", "sleep_hours", "Lucas slept four hours",
            "I slept four hours",
            source_turn_id="t5",
            episode_text="I slept four hours",
            now=now,
        )
        quiet = world.select_for_prompt("what are you up to", now=now)
        keys = {s.key for s in quiet}
        assert "home_city" in keys
        assert "stale_city" not in keys
        assert "morning_start" in keys
        assert "readme_draft" not in keys
        assert "sleep_hours" not in keys

        on_readme = {s.key for s in world.select_for_prompt("how is the readme", now=now)}
        assert "readme_draft" in on_readme
        assert "sleep_hours" not in on_readme

        on_health = {s.key for s in world.select_for_prompt("I have a headache", now=now)}
        assert "sleep_hours" in on_health

        prompt = world.to_prompt_context("how is the readme", now=now)
        assert "Open commitments" not in prompt
        assert "wellbeing" not in prompt.lower()
        assert "staleness" not in prompt.lower()
        assert '"' in prompt

        renders = {
            s.key: _render(s)
            for s in world.select_for_prompt("how is the readme and I have a headache", now=now)
        }
        fresh_len = len(renders["home_city"])
        orig_budget = settings_mod.config.WORLD_PROMPT_CHAR_BUDGET
        try:
            settings_mod.config.WORLD_PROMPT_CHAR_BUDGET = fresh_len
            tight = world.to_prompt_context("hello", now=now)
        finally:
            settings_mod.config.WORLD_PROMPT_CHAR_BUDGET = orig_budget
        assert "Amsterdam" in tight
        assert "Utrecht" not in tight
        assert "seven" not in tight

        shown = world.format_inspection()
        assert "place" in shown
        assert "I live in Amsterdam" in shown

        life = UserLifeTracker(db_path=Path(td) / "life.db")
        life.ensure_schema()
        older = now - timedelta(days=3)
        newer = now - timedelta(days=1)
        assert life.record_commitment(
            "ship the scope doc",
            "I'll ship the scope doc tomorrow, that's the plan.",
            now=older,
        )["ok"]
        assert life.record_commitment(
            "write the readme",
            "I will write the readme by next Friday, promise.",
            deadline="2026-10-01",
            now=newer,
        )["ok"]
        life.update_status(1, "abandoned")
        life.log_wellbeing(0.4, "I feel flat today", "user_reported", now=now)
        life.log_wellbeing(0.1, "inferred from silence", "inferred", now=now)
        tasks = life.format_tasks()
        assert tasks.splitlines()[0] == "Open"
        open_at = tasks.index("write the readme")
        abandoned_at = tasks.index("ship the scope doc")
        assert open_at < tasks.index("Abandoned") < abandoned_at
        assert "deadline 2026-10-01" in tasks
        assert "%" not in tasks
        assert "overdue" not in tasks.lower()
        assert "trend" not in tasks.lower()
        assert "I feel flat today" in tasks
        assert "inferred from silence" not in tasks
        assert "Self-reported" in tasks


def test_consolidation_promotion_threshold():
    assert promotion_allowed(1, 10) is True
    assert promotion_allowed(2, 10) is False
    assert promotion_allowed(0, 0) is False

    def batch(world, dropped_at: set[int]):
        episodes = []
        candidates = []
        for i in range(10):
            token = f"qx{i}token"
            content = f"I have a {token} at home"
            episodes.append({"episode_id": f"e{i}", "content": content})
            quote = "missing quote" if i in dropped_at else token
            candidates.append({
                "source_index": i,
                "domain": "material",
                "key": f"item_{i}",
                "value": f"Lucas has a {token}",
                "quote": quote,
            })
        return world.stage_consolidation_batch(candidates, episodes)

    with tempfile.TemporaryDirectory() as td:
        ok_world = UserWorldModel(db_path=Path(td) / "ok.db")
        summary = batch(ok_world, {0})
        assert summary["dropped"] == 1
        assert summary["promoted"] == 9
        assert len(ok_world.get_world_model()) == 9

        blocked = UserWorldModel(db_path=Path(td) / "blocked.db")
        summary = batch(blocked, {0, 1})
        assert summary["dropped"] == 2
        assert summary["promoted"] == 0
        assert blocked.get_world_model() == []


def test_register_trace_and_prompt_filters():
    with _temp_midterm():
        first = record_reply(
            turn_id="t1",
            reply="The portfolio piece is the focus. I notice that progress slipped.",
            user_text="how is the portfolio piece",
            prior_user_text="",
            slot_values=["Lucas is building A-OK as his portfolio piece"],
            open_commitment_texts=["finish the scope document by Friday"],
        )
        assert first["concrete_referent_count"] >= 2
        assert first["evaluative_lexicon_hits"] >= 2
        assert first["commitment_mention"] is False
        assert first["distinct_referent_ratio"] == 1.0

        mentioned = record_reply(
            turn_id="t2",
            reply=(
                "You said you would finish the scope document by Friday, "
                "and the portfolio piece is still the focus."
            ),
            user_text="what was I meant to finish the scope document by Friday",
            prior_user_text="",
            slot_values=["Lucas is building A-OK as his portfolio piece"],
            open_commitment_texts=["finish the scope document by Friday"],
        )
        assert mentioned["commitment_mention"] is True
        assert mentioned["user_raised_it"] is True
        assert mentioned["distinct_referent_ratio"] < first["distinct_referent_ratio"]

        unraised = record_reply(
            turn_id="t3",
            reply="Remember to finish the scope document by Friday.",
            user_text="hello",
            prior_user_text="nothing about that",
            slot_values=[],
            open_commitment_texts=["finish the scope document by Friday"],
        )
        assert unraised["commitment_mention"] is True
        assert unraised["user_raised_it"] is False

    sm = SelfModel()
    sm.model["model_version"] = 3
    sm.model["behavioral_patterns"]["notable_tendencies"] = ["recites the same facts"]
    sm.model["hypotheses"] = [
        {
            "statement": "unchecked tendency toward reciting",
            "confidence": 0.9,
            "action_bias": "none",
        },
        {
            "statement": "checked claim about file names",
            "confidence": 0.4,
            "action_bias": "look",
            "verification": dict(SCRIPT_CHECK),
        },
    ]
    text = sm.to_prompt_context()
    assert "recites the same facts" not in text
    assert "unchecked tendency" not in text
    assert "checked claim about file names" in text

    reg = OpinionRegistry()
    reg.opinions["own"] = Opinion(
        id="own",
        domain="ai_consciousness",
        position="Models can hold a position of their own",
        reasoning="because the work is to have one",
        origin=OpinionOrigin.INDEPENDENT,
        conviction=0.8,
        formed_at=utcnow().isoformat(),
    )
    for i in range(6):
        reg.opinions[f"ad{i}"] = Opinion(
            id=f"ad{i}",
            domain=f"topic_{i}",
            position=f"Adopted stance {i}",
            reasoning="the user said so",
            origin=OpinionOrigin.ADOPTED,
            conviction=0.9,
            formed_at=utcnow().isoformat(),
        )
    ctx = reg.to_prompt_context()
    assert "Independence score" not in ctx
    assert "ai_consciousness" in ctx
    assert reg.compute_independence_score() < 0.3

    defn = next(g for g in UNBOUNDED_GOALS if g["name"] == "user_life_improvement")
    goal = Goal(
        id="g-life",
        name=defn["name"],
        description=defn["description"],
        kind="unbounded",
        ceiling_description=defn["ceiling_description"],
    )
    assert "wellbeing trend" in (goal.ceiling_description or "")
    goals = GoalSystem()
    goals.instrumental.append(goal)
    goal_prompt = goals.to_prompt_context()
    assert "wellbeing trend" not in goal_prompt
    assert "Open commitments" not in goal_prompt
    assert "grounded in facts he has stated" in goal_prompt

    from core.loop import ConsciousArchitecture
    source = inspect.getsource(ConsciousArchitecture.build_context)
    assert "user_life.to_prompt_context" not in source
    assert "user_life.on_session_start" not in source
    assert "Open commitments" not in source
    assert "wellbeing" not in source
    process_src = inspect.getsource(ConsciousArchitecture.process)
    assert "user_life.on_session_start" not in process_src

    block = standing_block({
        "hypotheses": [
            {"id": "h1", "verification": dict(SCRIPT_CHECK)},
            {"id": "h2"},
        ],
        "standing_constraints": [
            {"hypothesis_id": "h1", "text": "Name the file before editing."},
            {"hypothesis_id": "h2", "text": "Always mention progress."},
            "legacy string constraint",
        ],
    })
    assert "Name the file before editing." in block
    assert "progress" not in block
    assert "legacy" not in block

    names = {t.name for t in ALL_TOOLS}
    assert {
        "record_world_fact",
        "confirm_world_fact",
        "supersede_world_fact",
        "get_world_model",
    } <= names
    for tool in WORLD_TOOLS:
        fields = getattr(tool.args_schema, "model_fields", {}) or {}
        assert "user_id" not in fields


def test_prompt_dump_header():
    from agent.runner import _dump_assembled_prompt
    orig_dir = settings_mod.config.data_dir
    orig_flag = settings_mod.config.DUMP_ASSEMBLED_PROMPT
    with tempfile.TemporaryDirectory() as td:
        try:
            settings_mod.config.data_dir = Path(td)
            settings_mod.config.DUMP_ASSEMBLED_PROMPT = True
            _dump_assembled_prompt(
                "turn-1",
                "system text without a ledger",
                {"keeper": 4, "user": 2},
            )
            dumped = Path(td) / "logs" / "prompt" / "turn-1.txt"
            body = dumped.read_text()
            assert body.startswith("# keeper_tokens=4 user_tokens=2\n")
            assert "Open commitments" not in body
            assert "wellbeing trend" not in body
        finally:
            settings_mod.config.data_dir = orig_dir
            settings_mod.config.DUMP_ASSEMBLED_PROMPT = orig_flag
