"""
Autonomous action — self-directed behavior between user inputs.

This is what makes the system more than a reactive tool. Between
conversations, the autonomous loop:
  1. Checks if there are active goals to pursue
  2. Generates new goals if drives are high but goal queue is empty
  3. Picks the highest-salience goal and takes a step toward it
  4. Records the action as a self-observation
  5. Updates internal state and memory

Autonomous actions are internal — they don't send messages to the user
unprompted (that would be invasive). Instead they:
  - Research topics (web search, memory search)
  - Consolidate understanding (synthesise what's been learned)
  - Prepare for likely future conversations
  - Reflect on tensions or open questions
  - Update the self-model

The results surface naturally in future conversations through richer
context, better recall, and more informed engagement.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

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
    ):
        self.goals = goal_system
        self.state = state
        self.memory = memory_system
        self.self_model = self_model
        self._llm: ChatAnthropic | None = None
        self._running = False
        self._last_action: datetime | None = None

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

    # ── Single action ─────────────────────────────────────────────────────

    async def take_action(self, user_id: int) -> dict | None:
        """
        Take one autonomous step toward the top goal.
        Returns a dict with the action taken, or None if nothing to do.
        """
        # Generate goals if queue is empty but drives are active
        if not self.goals.has_active:
            if self.state and self.state.drives.active_drives:
                new_goals = await self.goals.generate_instrumental(
                    self.state, self.memory, user_id
                )
                if not new_goals:
                    return None
            else:
                return None

        goal = self.goals.top_goal
        if goal is None:
            return None

        # Build context for the pursuit
        state_context = self.state.to_prompt_context() if self.state else "No state"

        memory_context = "No relevant memories"
        if self.memory:
            try:
                results = await self.memory.recall(
                    user_id, goal.description, self.state, n_episodic=5, n_semantic=3
                )
                episodes = results.get("episodic", [])
                semantic = results.get("semantic", [])
                mem_lines = []
                for ep in episodes[:5]:
                    mem_lines.append(f"- [episodic] {ep['content'][:120]}")
                for s in semantic[:3]:
                    mem_lines.append(f"- [semantic] {s['content'][:120]}")
                if mem_lines:
                    memory_context = "\n".join(mem_lines)
            except Exception:
                pass

        self_model_context = ""
        if self.self_model:
            self_model_context = self.self_model.to_prompt_context()

        prompt = PURSUIT_PROMPT.format(
            goal_name=goal.name,
            goal_description=goal.description,
            progress=f"{goal.progress:.0%}",
            state_context=state_context,
            memory_context=memory_context,
            self_model_context=self_model_context,
        )

        try:
            result = await self.llm.ainvoke([HumanMessage(content=prompt)])
            raw = result.content.strip()

            # Parse the structured part (JSON at the end)
            note = raw
            progress_delta = 0.1
            summary = ""

            # Try to extract JSON from the end of the response
            if "{" in raw:
                json_start = raw.rfind("{")
                json_end = raw.rfind("}") + 1
                if json_end > json_start:
                    try:
                        meta = json.loads(raw[json_start:json_end])
                        progress_delta = min(0.3, max(0.0, meta.get("progress_delta", 0.1)))
                        summary = meta.get("summary", "")
                        note = raw[:json_start].strip()
                    except json.JSONDecodeError:
                        pass

            # Apply progress
            goal.advance(progress_delta, summary or note[:80])

            # Store as memory
            if self.memory:
                await self.memory.store_episode(
                    user_id=user_id,
                    content=f"autonomous_action [{goal.name}]: {note}",
                    internal_state=self.state,
                    salience=0.7,
                    type="event",
                    tags=["autonomous", goal.name],
                )

            # Self-model observes what we did
            if self.self_model:
                await self.self_model.observe(
                    user_id=user_id,
                    action=f"Autonomously pursued goal '{goal.name}': {summary or note[:100]}",
                    context=f"Goal: {goal.description}, Progress: {goal.progress:.0%}",
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

    # ── Background loop ───────────────────────────────────────────────────

    async def run_loop(
        self,
        user_id: int,
        interval_minutes: int = 5,
        min_idle_minutes: float = 2,
    ) -> None:
        """
        Run autonomous actions on a schedule.

        Args:
            user_id: which user context to operate in
            interval_minutes: how often to attempt an action
            min_idle_minutes: minimum time since last user message
                              before acting (avoids interrupting active conversations)
        """
        self._running = True
        log.info(
            "Autonomous loop started (interval=%dm, idle_min=%.1fm)",
            interval_minutes, min_idle_minutes,
        )

        while self._running:
            await asyncio.sleep(interval_minutes * 60)

            if not self._running:
                break

            # Don't act if we're in an active conversation
            # (check by looking at how recently the state was updated)
            if self.state:
                idle = (datetime.utcnow() - self.state.last_updated).total_seconds() / 60
                if idle < min_idle_minutes:
                    continue

            # Don't act if fatigue is too high
            if self.state and self.state.fatigue > 0.8:
                continue

            # Maintain goals: abandon stale ones
            abandoned = self.goals.abandon_stale()
            if abandoned:
                log.info("Abandoned stale goals: %s", [g.name for g in abandoned])

            # Take an action
            await self.take_action(user_id)

    def stop(self) -> None:
        self._running = False


# Need json import for parsing
import json
