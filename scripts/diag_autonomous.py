#!/usr/bin/env python3
"""Call take_action and research_topic against real state, bypassing run_loop gates."""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import config  # noqa: E402
from core.llm import Tier  # noqa: E402
from core.timeutil import utcnow  # noqa: E402
from scripts.diag_common import load_companion_no_loops  # noqa: E402

log = logging.getLogger("diag_autonomous")


def _primary_user_id() -> int:
    ids = list(config.allowed_user_ids)
    return ids[0] if ids else 0


def report_gates(companion, min_idle_minutes: float = 2.0) -> None:
    print("\n===== run_loop gates vs current in-memory state =====")
    session = companion.any_session_active()
    print(f"session_active (any_session_active): {session}  -> {'FAIL skip' if session else 'PASS'}")
    print(f"  _session_active map: {companion._session_active}")

    low_ok = companion.api_budget.allows(Tier.LOW, background=True)
    mid_ok = companion.api_budget.allows(Tier.MID, background=True)
    frac = companion.api_budget.budget_fraction_used
    print(
        f"budget_gate LOW background: allows={low_ok} fraction={frac:.3f} "
        f"spent=€{companion.api_budget.spent_today_eur:.4f} "
        f"-> {'FAIL skip' if not low_ok else 'PASS'}"
    )
    print(f"budget MID background (not a hard skip, marks types unavailable): allows={mid_ok}")

    idle = (utcnow() - companion.state.last_updated).total_seconds() / 60
    print(
        f"idle_below_min: last_updated={companion.state.last_updated.isoformat()} "
        f"now={utcnow().isoformat()} idle={idle:.2f}m min={min_idle_minutes} "
        f"-> {'FAIL skip' if idle < min_idle_minutes else 'PASS'}"
    )

    fat = companion.state.fatigue
    print(f"fatigue_gate: fatigue={fat:.4f} threshold=0.8 -> {'FAIL skip' if fat > 0.8 else 'PASS'}")

    active = companion.goals.active_instrumental
    print(f"active_instrumental: {len(active)} {[g.name for g in active]}")
    drives = list(companion.state.drives.all_drives)
    dmax = max((d.intensity for d in drives), default=0.0)
    print(f"drives_max: {dmax:.3f} (generation gated at >0.6)")
    independence = companion.self_model.opinions.compute_independence_score()
    print(
        f"independence: {independence:.3f} (research-goal gen if < 0.4) "
        f"n_opinions={len(companion.self_model.opinions.opinions)}"
    )


def _print_goal_spend(companion) -> None:
    daily_limit_usd = (
        companion.api_budget.daily_limit_eur / companion.api_budget.USD_TO_EUR
        if companion.api_budget.USD_TO_EUR
        else companion.api_budget.daily_limit_eur
    )
    print(f"daily_limit_usd={daily_limit_usd:.4f} spent_today_eur=€{companion.api_budget.spent_today_eur:.4f}")
    print(f"spend_by_task={companion.api_budget.spend_by_task}")
    for g in companion.goals.active_instrumental:
        over = g.over_budget_share(daily_limit_usd)
        print(
            f"  goal={g.name} spend_today_usd={g.spend_today_usd:.6f} "
            f"budget_share={g.budget_share} over_budget_share={over} "
            f"n_log={len(g.action_log)} ratings={g.action_ratings}"
        )


async def run_take_action_loop(companion, user_id: int, n: int) -> None:
    print(f"\n===== take_action loop n={n} (gates bypassed) =====")
    _print_goal_spend(companion)
    results = []
    skip_over_budget = 0
    for i in range(1, n + 1):
        print(f"\n----- cycle {i}/{n} -----")
        result = await companion.autonomous.take_action(user_id)
        skip = companion.autonomous.last_skip_reason
        print(f"take_action returned: {result}")
        print(f"last_skip_reason: {skip}")
        if skip == "all_goals_over_budget_share":
            skip_over_budget += 1
        results.append({"i": i, "result": result, "skip": skip})
        print(f"after cycle {i}:")
        _print_goal_spend(companion)

    print("\n===== loop summary =====")
    print(f"cycles={n} over_budget_share_skips={skip_over_budget}")
    by_type: dict[str, list] = {}
    empty_arts: dict[str, int] = {}
    for item in results:
        r = item["result"] or {}
        at = r.get("action_type") or item["skip"] or "none"
        by_type.setdefault(at, []).append(r)
        if r:
            arts = r.get("artifact_refs") or []
            if not arts:
                empty_arts[at] = empty_arts.get(at, 0) + 1
    print("action types:", {k: len(v) for k, v in by_type.items()})
    print("unproductive (empty artifact_refs) by type:", empty_arts)
    for g in companion.goals.active_instrumental:
        print(f"ratings table {g.name}: {g.action_ratings}")
        print(f"spend_today_usd series {g.name}: {g.spend_today_usd}")
        elos = [(a.get("action_type"), a.get("elo"), a.get("artifact_refs")) for a in g.action_log]
        print(f"action_log {g.name}: {elos}")
    total_eur = companion.api_budget.spent_today_eur
    print(f"total spent_today_eur after {n} cycles: €{total_eur:.4f}")
    print(f"extrapolate ×{96 / n:.2f} to 96 cycles: €{total_eur * (96 / n):.4f}")
    print(f"exceeds €10? {total_eur * (96 / n) > 10.0}")


async def amain() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", type=int, default=0, help="Call take_action N times (Phase D). 0 = once + research")
    parser.add_argument("--skip-research", action="store_true")
    args = parser.parse_args()

    companion = await load_companion_no_loops()
    user_id = _primary_user_id()
    print(f"loaded companion, user_id={user_id} data_dir={config.data_dir}")
    report_gates(companion)

    if args.loop > 0:
        await run_take_action_loop(companion, user_id, args.loop)
        print("\nNOTE: loop mutated in-memory + episodic/chroma under DATA_DIR. "
              "JSON is not written unless _save_state is called.")
        return

    print("\n===== take_action (gates bypassed) =====")
    result = await companion.autonomous.take_action(user_id)
    print(f"take_action returned: {result}")
    print(f"last_skip_reason: {companion.autonomous.last_skip_reason}")
    if result:
        print(
            "Q: does it produce an action when gates are bypassed? YES. "
            f"action_type={result.get('action_type')} goal={result.get('goal_name')}"
        )
        print("S2 is a run_loop gating problem, not a take_action dispatch problem.")
    else:
        print(
            "Q: does it produce an action when gates are bypassed? NO. "
            f"skip={companion.autonomous.last_skip_reason}. "
            "S2 is a take_action / dispatch problem."
        )

    if args.skip_research:
        return

    print("\n===== ResearchEngine.research_topic standalone =====")
    try:
        research = await companion.research.research_topic(
            topic="test topic",
            context="diagnostic",
            user_id=user_id,
        )
    except Exception as exc:
        log.exception("research_topic raised")
        research = {"error": repr(exc)}
    print(f"research_topic returned: {research}")
    from tools.web_search import HEALTH
    print(f"SearchHealth: {HEALTH.snapshot()}")
    if research and research.get("opinion_id"):
        print("Q: research works standalone: YES (SearXNG + extract + synthesis + opinion). S3 is that nothing calls it.")
    elif research is None:
        print("Q: research failed standalone (None). S3 may be inside ResearchEngine/SearXNG, not only the goal system.")
    else:
        print(f"Q: research returned a non-opinion payload: {list(research) if isinstance(research, dict) else type(research)}")

    print("\nNOTE: take_action / research_topic mutated in-memory state. "
          "This process does not call _save_state; disk JSON is unchanged unless "
          "a store_episode wrote to SQLite/Chroma.")


if __name__ == "__main__":
    asyncio.run(amain())
