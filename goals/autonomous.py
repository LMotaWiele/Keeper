"""Autonomous action — internal goal pursuit between user turns.

Does not message the user. Session-gated so it never runs mid-conversation.
See docs/goals-and-self.md.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from core.timeutil import utcnow, parse_iso
from typing import Any, Callable
from goals.research import ResearchEngine
from langchain_core.messages import HumanMessage

from core.events import (
    GoalProgressEvent,
    GoalCompletedEvent,
    GoalFrustratedEvent,
)

log = logging.getLogger(__name__)


class AutonomousEngine:
    """
    Runs goal-directed behavior in the background between user interactions.

    Designed to be started as an asyncio task. Checks periodically whether
    there are goals worth pursuing and takes small steps toward them.
    """

    def __init__(
        self,
        goal_system: Any,
        state: Any,
        memory_system: Any,
        self_model: Any,
        research_engine: ResearchEngine | None = None,
    ):
        self.goals = goal_system
        self.state = state
        self.memory = memory_system
        self.self_model = self_model
        self.research = research_engine
        self._running = False
        self._last_action: datetime | None = None
        self.theorizer: Any | None = None    # SelfTheorizer, set by ConsciousArchitecture
        self.simulator: Any | None = None    # FutureSimulator, set by ConsciousArchitecture

    async def take_research_action(self, user_id: int, goal) -> dict | None:
        """Run web research + opinion formation. Does not complete the goal."""
        if not self.research:
            log.warning("Research engine not available")
            return None

        topic = goal.description
        context = f"Pursuing goal '{goal.name}' (serves: {goal.parent_goal})"
        try:
            result = await self.research.research_topic(
                topic=topic,
                context=context,
                user_id=user_id,
            )
        except Exception as e:
            log.warning("Research action failed: %s", e)
            return None
        return result

    # ── Single action ─────────────────────────────────────────────────────

    async def take_action(self, user_id: int) -> dict | None:
        """Take one rated autonomous step toward the highest-salience eligible goal."""
        from core.llm import Tier, get_llm
        from core.loop import companion
        from goals.action_rating import ActionRecord, rate_action, select_action_type
        from goals.completion import check as check_condition, validate_condition
        from goals.system import GoalStatus

        active = self.goals.active_instrumental
        if not active:
            if self.state and self.state.drives and any(
                d.intensity > 0.6 for d in self.state.drives.all_drives
            ):
                await self.goals.generate_instrumental(self.state, self.memory, user_id)
                active = self.goals.active_instrumental
            if not active:
                return None

        daily_limit_usd = (
            companion.api_budget.daily_limit_eur / companion.api_budget.USD_TO_EUR
            if companion.api_budget.USD_TO_EUR
            else companion.api_budget.daily_limit_eur
        )
        eligible = [
            g for g in active
            if not (g.is_unbounded and g.over_budget_share(daily_limit_usd))
        ]
        if not eligible:
            return None

        ranked = sorted(eligible, key=lambda g: g.salience, reverse=True)
        goal = ranked[0]

        mid_ok = companion.api_budget.allows(Tier.MID, background=True)
        unavailable: set[str] = set()
        if not mid_ok:
            unavailable.update({"web_research", "self_theorize", "future_simulate"})
        if not self.research:
            unavailable.add("web_research")
        if not self.theorizer:
            unavailable.add("self_theorize")
        if not self.simulator:
            unavailable.add("future_simulate")
        if not getattr(companion, "codebase", None):
            unavailable.add("codebase_read")

        action_type = select_action_type(goal, unavailable=unavailable)
        spend_before = companion.api_budget.spent_today_eur

        summary = ""
        artifact_refs: list[str] = []

        try:
            if action_type == "web_research":
                result = await self.take_research_action(user_id, goal)
                if result:
                    summary = (
                        f"Researched and formed opinion: "
                        f"{result.get('position', '')[:150]}"
                    )
                    if result.get("opinion_id"):
                        artifact_refs.append(f"opinion:{result['opinion_id']}")
                    if result.get("episode_id"):
                        artifact_refs.append(f"episode:{result['episode_id']}")
                    if self.self_model:
                        await self.self_model.observe(
                            user_id=user_id,
                            action=summary,
                            context=f"Goal: {goal.description}",
                        )
                else:
                    summary = "Web research produced nothing"

            elif action_type == "self_theorize":
                result = await self.theorizer.theorize(user_id, self.state)
                proposals = (result or {}).get("proposals") or []
                summary = f"Theorizing cycle: {len(proposals)} proposals"
                artifact_refs = [f"proposal:{i}" for i, _ in enumerate(proposals)]

            elif action_type == "future_simulate":
                action_desc = f"[Autonomous] Goal '{goal.name}': {goal.description[:200]}"
                sim = await self.simulator.simulate(
                    action_description=action_desc,
                    conversation_context="No active session",
                    internal_state=self.state,
                    memory_context="",
                    goals_context=f"Active goals: {[g.name for g in self.goals.active_instrumental]}",
                )
                if sim:
                    summary = (
                        f"Simulated next step: {sim.get('recommendation')} — "
                        f"{(sim.get('reasoning') or '')[:180]}"
                    )
                    if self.memory:
                        ep_id = await self.memory.store_episode(
                            user_id=user_id,
                            content=f"[Simulation] {goal.name}: {summary}",
                            internal_state=self.state,
                            salience=0.5,
                            type="event",
                            tags=["autonomous", "simulation", goal.name],
                        )
                        artifact_refs = [f"episode:{ep_id}"]
                else:
                    summary = "Simulation produced no result"

            elif action_type == "codebase_read":
                modules = list(companion.codebase.index.keys()) if companion.codebase else []
                chosen = modules[0] if modules else ""
                if modules:
                    pick_prompt = (
                        f"Goal: {goal.name} — {goal.description}\n"
                        f"Modules:\n" + "\n".join(f"- {m}" for m in modules[:40]) +
                        "\nPick ONE module path most relevant to this goal. "
                        "Output ONLY the path."
                    )
                    try:
                        pick = await get_llm("autonomous_pursuit").ainvoke(
                            [HumanMessage(content=pick_prompt)]
                        )
                        text = (pick.content or "").strip().strip("`")
                        for m in modules:
                            if m in text:
                                chosen = m
                                break
                    except Exception:
                        pass
                detail = companion.codebase.get_module_detail(chosen) if chosen else ""
                analysis_prompt = (
                    f"Goal: {goal.name} — {goal.description}\n"
                    f"Module detail:\n{detail[:4000]}\n"
                    "Write a short analysis of how this module relates to the goal "
                    "and one concrete observation."
                )
                analysis = await get_llm("autonomous_pursuit").ainvoke(
                    [HumanMessage(content=analysis_prompt)]
                )
                summary = f"Read {chosen}: {(analysis.content or '')[:200]}"
                if self.memory:
                    ep_id = await self.memory.store_episode(
                        user_id=user_id,
                        content=f"[Codebase] {chosen}: {analysis.content}",
                        internal_state=self.state,
                        salience=0.5,
                        type="event",
                        tags=["autonomous", "codebase", goal.name],
                    )
                    artifact_refs = [f"episode:{ep_id}"]

            elif action_type == "memory_synthesis":
                recalled = {"episodic": [], "semantic": []}
                if self.memory:
                    recalled = await self.memory.recall(
                        user_id, goal.description, current_state=self.state,
                    )
                blob = "\n".join(
                    f"- {ep.get('content', '')[:200]}"
                    for ep in (recalled.get("episodic") or [])[:8]
                )
                prompt = (
                    f"Goal: {goal.name} — {goal.description}\n"
                    f"Recalled episodes:\n{blob or '(none)'}\n"
                    "Extract one durable pattern as JSON: "
                    '{"content": "...", "type": "preference|trait|knowledge|behavioral"}'
                )
                from core.json_utils import parse_json_lenient
                result = await get_llm("pattern_extraction", json_mode=True).ainvoke(
                    [HumanMessage(content=prompt)]
                )
                parsed = parse_json_lenient(result.content)
                raw = (result.content or "")
                if isinstance(parsed, list):
                    parsed = parsed[0] if parsed else {}
                content = (parsed or {}).get("content") or raw[:300]
                from memory.semantic import semantic
                doc_id = await semantic.store(
                    user_id=user_id,
                    content=content,
                    pattern_type=(parsed or {}).get("type", "general"),
                    confidence=0.6,
                )
                summary = f"Synthesised pattern: {content[:180]}"
                artifact_refs = [f"semantic:{doc_id}"]

            elif action_type == "consolidation_review":
                from memory.semantic import semantic
                candidates = await semantic.list_review_candidates(user_id)
                if not candidates:
                    summary = "No weak/invalidated patterns to review"
                else:
                    blob = "\n".join(
                        f"[{i}] ({c['confidence']:.2f}{' invalidated' if c.get('invalidated') else ''}) "
                        f"{c['content'][:200]}"
                        for i, c in enumerate(candidates[:8])
                    )
                    prompt = (
                        f"Goal: {goal.name}\nPatterns:\n{blob}\n"
                        "For each, decide keep, revise, or drop. "
                        'JSON array: [{"index": 0, "action": "keep|revise|drop", '
                        '"content": "revised text if revise"}]'
                    )
                    from core.json_utils import parse_json_lenient
                    result = await get_llm("contradiction_check", json_mode=True).ainvoke(
                        [HumanMessage(content=prompt)]
                    )
                    parsed = parse_json_lenient(result.content)
                    if isinstance(parsed, dict):
                        parsed = parsed.get("decisions") or parsed.get("results") or []
                    touched = []
                    for decision in parsed or []:
                        try:
                            idx = int(decision.get("index", -1))
                        except (TypeError, ValueError):
                            continue
                        if idx < 0 or idx >= len(candidates):
                            continue
                        cand = candidates[idx]
                        act = str(decision.get("action") or "keep").lower()
                        if act == "drop":
                            await semantic.invalidate(
                                user_id, cand["id"], reason="consolidation_review drop",
                            )
                            touched.append(cand["id"])
                        elif act == "revise" and decision.get("content"):
                            doc_id = await semantic.store(
                                user_id=user_id,
                                content=decision["content"],
                                pattern_type=cand.get("metadata", {}).get("pattern_type", "general"),
                                confidence=max(0.5, cand.get("confidence", 0.4)),
                            )
                            touched.append(doc_id)
                        else:
                            touched.append(cand["id"])
                    summary = f"Reviewed {len(candidates[:8])} patterns, touched {len(touched)}"
                    artifact_refs = [f"semantic:{d}" for d in touched]

            else:
                summary = f"Unknown action type {action_type}"

        except Exception as e:
            log.warning("Autonomous action failed for goal %s (%s): %s", goal.name, action_type, e)
            if self.state:
                self.state.update(GoalFrustratedEvent(goal_id=goal.id, reason=str(e)))
            return None

        cost_eur = max(0.0, companion.api_budget.spent_today_eur - spend_before)
        cost_usd = (
            cost_eur / companion.api_budget.USD_TO_EUR
            if companion.api_budget.USD_TO_EUR
            else cost_eur
        )
        goal.record_spend(cost_usd)
        goal.last_pursued = utcnow()
        if summary:
            goal.progress_notes.append(f"[{utcnow().isoformat()[:16]}] {summary[:200]}")

        import uuid as _uuid
        record = ActionRecord(
            id=_uuid.uuid4().hex[:10],
            goal_id=goal.id,
            action_type=action_type,
            summary=summary or action_type,
            artifact_refs=artifact_refs,
            cost_usd=cost_usd,
            elo=0.0,
            timestamp=utcnow().isoformat(),
        )
        await rate_action(goal, record, companion)

        completed = False
        if goal.is_bounded and goal.completion_condition:
            cond = validate_condition(goal.completion_condition)
            if cond is not None:
                satisfied, evidence = await check_condition(
                    cond, goal=goal, companion=companion, user_id=user_id,
                )
                if satisfied:
                    goal.status = GoalStatus.COMPLETED
                    goal.completed_at = utcnow()
                    goal.progress_notes.append(f"[completed] {evidence}")
                    self.goals._archive_completed(goal)
                    if self.state:
                        self.state.update(GoalCompletedEvent(
                            goal_id=goal.id, goal_description=goal.description,
                        ))
                    completed = True
                elif self.state:
                    self.state.update(GoalProgressEvent(
                        goal_id=goal.id,
                        goal_description=goal.description,
                        progress_delta=0.0,
                    ))
        elif goal.is_unbounded:
            if self.state:
                self.state.update(GoalProgressEvent(
                    goal_id=goal.id,
                    goal_description=goal.description,
                    progress_delta=0.05,
                ))

        self._last_action = utcnow()
        action_result = {
            "goal_id": goal.id,
            "goal_name": goal.name,
            "action": summary[:200],
            "action_type": action_type,
            "elo": record.elo,
            "artifact_refs": artifact_refs,
            "cost_usd": cost_usd,
            "completed": completed,
        }
        log.info(
            "Autonomous action: %s type=%s elo=%.0f artifacts=%s",
            goal.name, action_type, record.elo, artifact_refs,
        )
        return action_result

    async def _generate_research_goals(self, user_id: int) -> None:
        """Generate research goals when independence score is low."""
        if not self.research:
            return
        
        active_goals = [g.to_dict() for g in self.goals.active_instrumental]
        topics = await self.research.suggest_research_topics(
            user_id=user_id,
            state=self.state,
            active_goals=active_goals,
        )
        
        if not topics:
            return
        
        import uuid
        from goals.system import Goal

        for t in topics[:2]:  # cap at 2 research goals at a time
            domain = t["topic"][:40].replace(" ", "_").lower()
            goal = Goal(
                id=f"ig_{uuid.uuid4().hex[:8]}",
                name=f"research_{t['topic'][:30].replace(' ', '_').lower()}",
                description=t["topic"],
                parent_goal=t.get("parent_goal", "understand"),
                salience=0.7,
                tags=["research", "external_grounding"],
                context=t.get("context", ""),
                kind="bounded",
                completion_condition={
                    "type": "opinion_registered",
                    "params": {"domain": domain},
                    "description": (
                        f"An independent or external opinion is registered on {t['topic']}"
                    ),
                    "created_at": utcnow().isoformat(),
                },
            )
            self.goals.instrumental.append(goal)
            log.info(
                "Generated research goal: %s (independence=%.2f)",
                goal.name,
                self.self_model.opinions.compute_independence_score(),
            )
    
    # ── Background loop ───────────────────────────────────────────────────

    async def run_loop(
        self,
        user_id: int,
        interval_minutes: int = 5,
        min_idle_minutes: float = 2,
        session_check: Callable[[], bool] | None = None,
    ) -> None:
        """Run one autonomous step on a schedule. Skip while a session is active."""
        self._running = True
        log.info(
            "Autonomous loop started (interval=%dm, idle_min=%.1fm)",
            interval_minutes, min_idle_minutes,
        )

        while self._running:
            await asyncio.sleep(interval_minutes * 60)

            if not self._running:
                break

            if session_check and session_check():
                log.debug("Autonomous action deferred — active session")
                continue

            from core.llm import Tier
            from core.loop import companion
            if not companion.api_budget.allows(Tier.LOW, background=True):
                log.debug("Autonomous action skipped — background LOW budget gate")
                continue

            if self.state:
                idle = (utcnow() - self.state.last_updated).total_seconds() / 60
                if idle < min_idle_minutes:
                    continue

            if self.state and self.state.fatigue > 0.8:
                continue

            abandoned = self.goals.abandon_stale()
            if abandoned:
                log.info("Abandoned stale goals: %s", [g.name for g in abandoned])

            if self.research and self.self_model:
                independence = self.self_model.opinions.compute_independence_score()
                has_research_goals = any(
                    "research" in (g.tags or [])
                    for g in self.goals.active_instrumental
                )
                if independence < 0.4 and not has_research_goals:
                    await self._generate_research_goals(user_id)

            await self.take_action(user_id)

    def stop(self) -> None:
        self._running = False
