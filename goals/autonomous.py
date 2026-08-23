"""Autonomous action — internal goal pursuit between user turns.

Does not message the user. Session-gated so it never runs mid-conversation.
See docs/goals-and-self.md.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Callable
from goals.research import ResearchEngine
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from config.settings import config
from core.events import (
    GoalProgressEvent,
    GoalCompletedEvent,
    GoalFrustratedEvent,
)

log = logging.getLogger(__name__)


PURSUIT_PROMPT = """\
You are an autonomous agent pursuing a goal during downtime (no user present).

Goal: {goal_name}
Description: {goal_description}
Current progress: {progress}

Your internal state:
{state_context}

Relevant memories:
{memory_context}

Self-model:
{self_model_context}

Take one concrete step toward this goal. You can:
- Synthesise what you know about the topic into a clear understanding
- Identify specific questions that need answering
- Connect ideas from different conversations or memories
- Prepare a useful framing or explanation you could share later
- Note contradictions or gaps in your understanding
- Reflect on why this goal matters given what you know about the user

Write your step as a brief internal note (2-5 sentences). Be specific
and substantive — this gets stored in memory and informs future behavior.

After your note, output a JSON block on its own line:
{{"progress_delta": 0.0-0.3, "summary": "one-line summary of what you did"}}
"""


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
        self._llm: ChatAnthropic | None = None
        self._running = False
        self._last_action: datetime | None = None
        self.theorizer: Any | None = None    # SelfTheorizer, set by ConsciousArchitecture
        self.simulator: Any | None = None    # FutureSimulator, set by ConsciousArchitecture

    @property
    def llm(self) -> ChatAnthropic:
        if self._llm is None:
            self._llm = ChatAnthropic(
                model=config.llm_model,
                anthropic_api_key=config.anthropic_api_key,
                temperature=0.7,
                max_tokens=1024,
            )
        return self._llm

    async def take_research_action(self, user_id: int, goal) -> dict | None:
        """
        Execute a research goal — web search + opinion formation.
        
        Research goals have tag "research" and their description contains
        the topic to investigate.
        """
        if not self.research:
            log.warning("Research engine not available, falling back to standard pursuit")
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
        
        if not result:
            return None
        
        # One successful research pass completes the goal.
        goal.advance(1.0, f"Formed opinion: {result['position'][:80]}")
        if goal.status.value == "completed":
            self.goals._archive_completed(goal)
        
        # Self-model observes the research
        if self.self_model:
            await self.self_model.observe(
                user_id=user_id,
                action=(
                    f"Autonomously researched '{result['topic']}' "
                    f"and formed opinion: {result['position'][:100]}"
                ),
                context=context,
            )
        
        from core.events import GoalCompletedEvent
        if self.state:
            self.state.update(GoalCompletedEvent(
                goal_id=goal.id,
                goal_description=goal.description,
            ))
        
        self._last_action = datetime.utcnow()
        
        log.info(
            "Research action complete: %s → opinion [%s]",
            goal.name, result['domain'],
        )
        
        return {
            "goal_id": goal.id,
            "goal_name": goal.name,
            "action": f"Researched and formed opinion: {result['position'][:150]}",
            "progress_delta": 1.0,
            "new_progress": 1.0,
            "completed": True,
            "research_result": result,
        }

    # ── Single action ─────────────────────────────────────────────────────

    async def take_action(self, user_id: int) -> dict | None:
        """
        Take one autonomous step toward the top goal.
        Returns a dict with the action taken, or None if nothing to do.
        """
        active = self.goals.active_instrumental
        if not active:
            if self.state and self.state.drives and any(
                d.intensity > 0.6 for d in self.state.drives.all_drives
            ):
                await self.goals.generate_instrumental(self.state, self.memory, user_id)
                active = self.goals.active_instrumental
            if not active:
                return None

        # Tie-break: don't let continuous_self_improvement block every other 0.9 goal.
        ranked = sorted(
            active,
            key=lambda g: (g.salience, 0 if g.name == "continuous_self_improvement" else 1),
            reverse=True,
        )

        goal = None
        for candidate in ranked:
            if candidate.name == "continuous_self_improvement" and self.theorizer:
                result = await self.theorizer.theorize(user_id, self.state)
                if result:
                    candidate.advance(
                        0.1,
                        f"Theorizing cycle: {len(result.get('proposals', []))} proposals",
                    )
                    return result
                continue
            goal = candidate
            break
        if goal is None:
            return None

        # Build context for the pursuit
        state_context = self.state.to_prompt_context() if self.state else "No state available"

        # Get relevant memories
        memory_context = "No memories available"
        if self.memory:
            try:
                results = await self.memory.recall(
                    user_id, goal.description, current_state=self.state
                )
                episodic_items = results.get("episodic", [])
                semantic_items = results.get("semantic", [])
                parts = []
                if episodic_items:
                    parts.append("Recent episodes:\n" + "\n".join(
                        f"- {ep.get('content', '')[:200]}" for ep in episodic_items[:3]
                    ))
                if semantic_items:
                    parts.append("Known patterns:\n" + "\n".join(
                        f"- {s.get('content', '')[:200]}" for s in semantic_items[:3]
                    ))
                if parts:
                    memory_context = "\n".join(parts)
            except Exception as e:
                log.debug("Memory recall failed for autonomous action: %s", e)

        # Self-model context
        self_model_context = "No self-model yet"
        if self.self_model:
            self_model_context = self.self_model.to_prompt_context() or self_model_context

        pursuit_description = goal.description
        if self.simulator:
            action_desc = f"[Autonomous] Goal '{goal.name}': {goal.description[:200]}"
            goals_ctx = f"Active goals: {[g.name for g in self.goals.active_instrumental]}"
            if await self.simulator.should_simulate(action_desc, self.state):
                sim = await self.simulator.simulate(
                    action_description=action_desc,
                    conversation_context="No active session",
                    internal_state=self.state,
                    memory_context=memory_context,
                    goals_context=goals_ctx,
                )
                if sim and sim.get("recommendation") == "abandon":
                    goal.progress_notes.append(
                        f"Action abandoned after simulation: {sim.get('reasoning', '')[:200]}"
                    )
                    return None
                if sim and sim.get("recommendation") == "modify":
                    modified_desc = sim.get("suggested_modification") or ""
                    if modified_desc:
                        pursuit_description = modified_desc

        if "research" in (goal.tags or []) and self.research:
            return await self.take_research_action(user_id, goal)

        prompt = PURSUIT_PROMPT.format(
            goal_name=goal.name,
            goal_description=pursuit_description,
            progress=f"{goal.progress:.0%}",
            state_context=state_context,
            memory_context=memory_context,
            self_model_context=self_model_context,
        )

        try:
            response = await self.llm.ainvoke([
                HumanMessage(content=prompt),
            ])
            text = response.content

            # Parse progress delta from the JSON block
            progress_delta = 0.05
            note = text
            try:
                # Find the last JSON block in the response
                json_start = text.rfind("{")
                json_end = text.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_block = json.loads(text[json_start:json_end])
                    progress_delta = json_block.get("progress_delta", 0.05)
                    note = text[:json_start].strip()
            except json.JSONDecodeError:
                pass

            # Update goal progress
            goal.advance(progress_delta, note[:200])

            # Store the action as an episodic memory
            if self.memory:
                await self.memory.store_episode(
                    user_id=user_id,
                    content=f"[Autonomous] Goal '{goal.name}': {note[:300]}",
                    internal_state=self.state,
                    salience=0.4,
                    type="event",
                    tags=["autonomous", "goal_pursuit", goal.name],
                )

            # Self-observe
            if self.self_model:
                await self.self_model.observe(
                    user_id=user_id,
                    action=f"Autonomous pursuit of '{goal.name}': {note[:200]}",
                    context=f"Goal: {goal.description}",
                    internal_state=self.state,
                )

            # Fire appropriate event into internal state
            if goal.status.value == "completed":
                event = GoalCompletedEvent(
                    goal_id=goal.id, goal_description=goal.description
                )
            else:
                event = GoalProgressEvent(
                    goal_id=goal.id,
                    goal_description=goal.description,
                    progress_delta=progress_delta,
                )
            if self.state:
                self.state.update(event)

            self._last_action = datetime.utcnow()

            action_result = {
                "goal_id": goal.id,
                "goal_name": goal.name,
                "action": note[:200],
                "progress_delta": progress_delta,
                "new_progress": goal.progress,
                "completed": goal.status.value == "completed",
            }

            log.info(
                "Autonomous action: %s (+%.0f%% → %.0f%%)",
                goal.name, progress_delta * 100, goal.progress * 100,
            )

            return action_result

        except Exception as e:
            log.warning("Autonomous action failed for goal %s: %s", goal.name, e)

            # Fire frustration event
            if self.state:
                self.state.update(GoalFrustratedEvent(
                    goal_id=goal.id, reason=str(e)
                ))

            return None

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
            goal = Goal(
                id=f"ig_{uuid.uuid4().hex[:8]}",
                name=f"research_{t['topic'][:30].replace(' ', '_').lower()}",
                description=t["topic"],
                parent_goal=t.get("parent_goal", "understand"),
                salience=0.7,
                tags=["research", "external_grounding"],
                context=t.get("context", ""),
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

            if self.state:
                idle = (datetime.utcnow() - self.state.last_updated).total_seconds() / 60
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
