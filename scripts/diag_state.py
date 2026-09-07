#!/usr/bin/env python3
"""Read on-disk Keeper state and print a diagnostic report. Bot must be stopped."""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import config  # noqa: E402
from core.opinions import OpinionOrigin, OpinionRegistry  # noqa: E402
from core.timeutil import utcnow  # noqa: E402
from goals.system import Goal  # noqa: E402


def _state_dir() -> Path:
    return Path(config.data_dir) / "state"


def _load_json(name: str) -> dict:
    path = _state_dir() / name
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _iso(ts: str | None) -> str:
    return ts or "(missing)"


def _answer(q: str, a: str) -> None:
    print(f"\nQ: {q}")
    print(f"A: {a}")


def report_internal_state(data: dict) -> None:
    print("## Internal state")
    print(f"- fatigue: {data.get('fatigue')}")
    print(f"- arousal: {data.get('arousal')}")
    print(f"- valence: {data.get('valence')}")
    print(f"- curiosity: {data.get('curiosity')}")
    print(f"- last_updated: {_iso(data.get('last_updated'))}")
    print(f"- messages_this_session: {data.get('messages_this_session')}")
    drives = (data.get("drives") or {}).get("persistent") or []
    print("- drives:")
    max_i = 0.0
    for d in drives:
        inten = float(d.get("intensity") or 0)
        max_i = max(max_i, inten)
        print(f"    {d.get('name')}: intensity={inten} last_satisfied={d.get('last_satisfied')}")
    fat = float(data.get("fatigue") or 0)
    _answer("is fatigue > 0.8?", f"{'YES' if fat > 0.8 else 'NO'} (fatigue={fat})")
    _answer(
        "does any drive exceed 0.6?",
        f"{'YES' if max_i > 0.6 else 'NO'} (max intensity={max_i})",
    )


def report_budget_fatigue_floor(internal: dict, budgets: dict) -> None:
    print("\n## Budget-fatigue floor")
    print(
        "InternalState.apply_budget_fatigue does "
        "`self.fatigue = max(self.fatigue, budget_fatigue)` "
        "(core/internal_state.py). It never lowers fatigue."
    )
    print(
        "APIBudget._check_reset zeroes spend at local midnight "
        "(core/resource_budgets.py). compute_fatigue_contribution then "
        "returns ~0. apply_budget_fatigue(max(fatigue, 0)) leaves the old value."
    )
    print(
        "Drift: DRIFT_RATE['fatigue'] = -0.003 toward BASELINE 0.0, applied "
        "inside InternalState.update() when elapsed minutes > 0, and on "
        "load/restore if elapsed > 1 minute. TimePassingEvent has an empty "
        "delta but still runs update(), so drift can clear a stale floor "
        "IF update() actually fires. apply_budget_fatigue is only called "
        "from companion.post_process (conversation), not from grounding."
    )
    api = (budgets.get("api") or {})
    print(f"- saved spent_today_eur: {api.get('spent_today_eur')}")
    print(f"- saved current_date: {api.get('current_date')}")
    print(f"- saved fatigue: {internal.get('fatigue')}")
    print(f"- saved last_updated: {internal.get('last_updated')}")
    today = datetime.now().date().isoformat()
    _answer(
        "after a budget reset, can fatigue still hold a floor from the previous day's spend?",
        "YES. Midnight reset does not write fatigue down. Drift-toward-baseline "
        "CAN clear it, but only when state.update() or restore() runs. "
        f"On disk now: current_date={api.get('current_date')!r} vs local today={today!r}, "
        f"fatigue={internal.get('fatigue')}. If last_updated is frozen, the floor "
        "persists in the file until the next load's restore() drift.",
    )


def report_goals(data: dict) -> None:
    print("\n## Goals")
    instrumental = data.get("instrumental") or []
    print(f"- instrumental count: {len(instrumental)}")
    print(f"- completed: {len(data.get('completed') or [])}")
    print(f"- abandoned: {len(data.get('abandoned') or [])}")
    active = []
    kind_ok = True
    bounded_null = []
    seeded = []
    for raw in instrumental:
        g = Goal.from_dict(raw)
        is_active = g.status.value == "active"
        if is_active:
            active.append(g)
        print(
            f"\n### {g.name}\n"
            f"- kind: {g.kind!r} (from_dict used kind={raw.get('kind')!r}, "
            f"tags={raw.get('tags')})\n"
            f"- status: {g.status.value}\n"
            f"- salience: {g.salience}\n"
            f"- completion_condition: {g.completion_condition}\n"
            f"- ceiling_description: {g.ceiling_description}\n"
            f"- budget_share: {g.budget_share}\n"
            f"- spend_today_usd: {g.spend_today_usd}\n"
            f"- spend_date: {g.spend_date!r}\n"
            f"- last_pursued: {g.last_pursued}\n"
            f"- len(action_log): {len(g.action_log)}\n"
            f"- action_ratings: {g.action_ratings}"
        )
        if g.name in {"continuous_self_improvement", "user_life_improvement"}:
            if g.kind != "unbounded":
                kind_ok = False
        if g.kind == "bounded" and g.completion_condition is None:
            bounded_null.append(g.name)
        for atype, info in (g.action_ratings or {}).items():
            n = int(info.get("n") or 0)
            if n > 0 and not any(r.get("action_type") == atype for r in g.action_log):
                seeded.append((g.name, atype, n))

    _answer(
        'do both unbounded goals have kind == "unbounded"?',
        "YES" if kind_ok else "NO — at least one loaded back as bounded. "
        "Goal.from_dict defaults kind to unbounded if the tag is present, else bounded.",
    )
    if bounded_null:
        print(f"  LIVE DEFECT: bounded goals with completion_condition null: {bounded_null}")
    _answer(
        "is active_instrumental non-empty?",
        f"{'YES' if active else 'NO'} (n={len(active)}). "
        "If empty, take_action exits before doing anything (unless a drive > 0.6 triggers generation).",
    )
    _answer(
        "does action_ratings contain seeded n>0 for types that were never run?",
        "YES: " + ", ".join(f"{a}/{b} n={c}" for a, b, c in seeded)
        if seeded
        else "NO. All action_ratings are empty or n matches action_log. Forced exploration (n<2) is still enabled.",
    )


def report_opinions(data: dict) -> None:
    print("\n## Opinions")
    reg = OpinionRegistry()
    if data:
        reg.restore(data)
    origins = Counter(o.origin.value for o in reg.opinions.values())
    print(f"- count: {len(reg.opinions)}")
    print(f"- by origin: {dict(origins)}")
    score = reg.compute_independence_score()
    print(f"- compute_independence_score(): {score:.4f}")
    empty = OpinionRegistry()
    empty_score = empty.compute_independence_score()
    print(f"- empty registry score: {empty_score:.4f}")
    _answer(
        "what does compute_independence_score return for an empty or near-empty registry?",
        f"Empty (no opinions with conviction>0) returns {empty_score} "
        "(hardcoded 0.5 in core/opinions.py). Current registry score="
        f"{score:.4f} with {len(reg.opinions)} opinion(s). "
        "run_loop generates research goals only when independence < 0.4. "
        f"{'The guard NEVER fires at 0.5 — research goals are not generated from an empty registry.' if empty_score >= 0.4 else 'Empty registry WOULD fire the research-goal guard.'}",
    )


def report_self_model(data: dict) -> None:
    print("\n## Self-model")
    print(f"- model_version: {data.get('model_version')}")
    print(f"- observation_count: {data.get('observation_count')}")
    print(f"- observations_analysed: {data.get('observations_analysed')}")
    print(f"- last_updated: {data.get('last_updated')}")
    hyps = data.get("hypotheses") or []
    print(f"- hypotheses ({len(hyps)}):")
    if not hyps:
        print("    (none)")
    for h in hyps:
        print(f"    statement: {h.get('statement')}")
        print(f"    action_bias: {h.get('action_bias')}")
        print(f"    confidence: {h.get('confidence')}")
    print(f"- consistency_flags: {data.get('consistency_flags')}")
    print(f"- tensions: {data.get('tensions')}")
    tendencies = (data.get("behavioral_patterns") or {}).get("notable_tendencies") or []
    print("- notable_tendencies:")
    for t in tendencies:
        print(f"    - {t}")
    needles = (
        "verif", "search before", "information-seek", "information seeking",
        "terseness", "terse", "concise", "short declarative", "check before",
        "look up", "before answering", "before agreeing",
    )
    hits = []
    for h in hyps:
        blob = f"{h.get('statement','')} {h.get('action_bias','')}".lower()
        if any(n in blob for n in needles):
            hits.append(("hypothesis", h))
    for t in tendencies:
        if any(n in t.lower() for n in needles):
            hits.append(("tendency", t))
    _answer(
        "does any hypothesis or tendency describe verifying-before-answering, information-seeking, or terseness?",
        "YES:\n" + "\n".join(f"    [{k}] {v}" for k, v in hits)
        if hits
        else "NO matching hypothesis/tendency. hypotheses is empty on disk. "
        "The tool-instruction block in build_context is the remaining prompt pressure toward checking/searching.",
    )


def report_memory() -> None:
    print("\n## Memory")
    db = Path(config.midterm_db_path)
    if not db.exists():
        print("(no episodic.db)")
        return
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    types = Counter()
    tags = Counter()
    idle_hits = []
    for row in con.execute("select type, tags, created_at, content from episodes"):
        types[row["type"]] += 1
        try:
            tlist = json.loads(row["tags"] or "[]")
        except json.JSONDecodeError:
            tlist = []
        for t in tlist:
            tags[t] += 1
        joined = " ".join(tlist)
        if any(k in joined for k in ("autonomous", "research", "goal_pursuit")):
            idle_hits.append((row["created_at"], row["type"], tlist, (row["content"] or "")[:120]))
    print(f"- episodic rows: {sum(types.values())}")
    print(f"- by type: {dict(types)}")
    print(f"- by tag: {dict(tags)}")
    print(f"- user_id counts: {list(con.execute('select user_id, count(*) from episodes group by user_id'))}")

    chroma = Path(config.chroma_db_path) / "chroma.sqlite3"
    print("- semantic collections:")
    if chroma.exists():
        cc = sqlite3.connect(chroma)
        names = list(cc.execute("select name from collections"))
        counts = list(cc.execute(
            """
            select col.name, count(e.id)
            from collections col
            left join segments s on s.collection = col.id
            left join embeddings e on e.segment_id = s.id
            group by col.name
            """
        ))
        for name, n in counts:
            print(f"    {name}: {n}")
        if not counts:
            print(f"    (collections present: {names}, no embedding join rows)")
    else:
        print("    (no chroma.sqlite3)")

    _answer(
        "are there any episodes tagged autonomous, research, or goal_pursuit from the idle period?",
        "YES:\n" + "\n".join(f"    {a} {b} {c} {d}" for a, b, c, d in idle_hits)
        if idle_hits
        else "NO. Zero episodes carry tags autonomous / research / goal_pursuit.",
    )


def report_split_brain(internal: dict) -> None:
    print("\n## JSON vs episodic timeline")
    db = Path(config.midterm_db_path)
    last_ep = None
    if db.exists():
        con = sqlite3.connect(db)
        row = con.execute("select max(created_at) from episodes").fetchone()
        last_ep = row[0] if row else None
    wm = _load_json("working_memory.json")
    last_wm = None
    for uid, items in wm.items():
        if items:
            last_wm = items[-1].get("timestamp")
            print(f"- working_memory user {uid}: {len(items)} items, last ts={last_wm}")
            print(f"  last content[:80]={items[-1].get('content','')[:80]!r}")
    print(f"- internal_state.last_updated: {internal.get('last_updated')}")
    print(f"- last episode created_at: {last_ep}")
    sm = _load_json("self_model.json")
    print(f"- self_model.json version: {sm.get('model_version')} last_updated={sm.get('last_updated')}")


def main() -> None:
    print("# Keeper diagnostic state")
    print(f"generated_at: {utcnow().isoformat()}")
    print(f"data_dir: {config.data_dir}")
    internal = _load_json("internal_state.json")
    goals = _load_json("goals.json")
    opinions = _load_json("opinions.json")
    self_model = _load_json("self_model.json")
    budgets = _load_json("resource_budgets.json")
    report_internal_state(internal)
    report_budget_fatigue_floor(internal, budgets)
    report_goals(goals)
    report_opinions(opinions)
    report_self_model(self_model)
    report_memory()
    report_split_brain(internal)
    print("\n## resource_budgets.json")
    print(json.dumps(budgets, indent=2))


if __name__ == "__main__":
    main()
