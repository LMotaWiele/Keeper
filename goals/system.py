"""
Intrinsic goal structures — Pillar 4 of The_core architecture.

This is what gives the system genuine agency rather than pure reactivity.

Two levels:
  Terminal goals  — deep persistent orientations that never complete.
                    They provide directional pull: "understand deeply",
                    "create something meaningful", "resolve open questions".

  Instrumental goals — specific pursuits generated in service of terminal
                       goals. These can be completed, abandoned, or
                       deprioritised. They're the actionable layer.

The goal system generates behavior autonomously — not just responding
to inputs but occasionally initiating based on active drives and
unresolved threads in memory.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from config.settings import config

log = logging.getLogger(__name__)


# ── Goal status ───────────────────────────────────────────────────────────

class GoalStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    PAUSED = "paused"


# ── Goal dataclass ────────────────────────────────────────────────────────

@dataclass
class Goal:
    """A single goal — either terminal or instrumental."""

    id: str
    name: str
    description: str
    is_terminal: bool = False
    status: GoalStatus = GoalStatus.ACTIVE

    # Prioritisation
    salience: float = 0.5          # 0–1, how urgent/important
    parent_goal: str | None = None  # terminal goal this serves (for instrumental)

    # Progress tracking
    progress: float = 0.0           # 0–1 for instrumental goals
    progress_notes: list[str] = field(default_factory=list)

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    last_pursued: datetime | None = None

    # Context
    tags: list[str] = field(default_factory=list)
    context: str = ""               # what prompted this goal

    def advance(self, delta: float, note: str = "") -> None:
        """Record progress toward this goal."""
        self.progress = min(1.0, self.progress + delta)
        self.last_pursued = datetime.utcnow()
        if note:
            self.progress_notes.append(f"[{datetime.utcnow().isoformat()[:16]}] {note}")
        if self.progress >= 1.0 and not self.is_terminal:
            self.status = GoalStatus.COMPLETED
            self.completed_at = datetime.utcnow()

    def abandon(self, reason: str = "") -> None:
        self.status = GoalStatus.ABANDONED
        if reason:
            self.progress_notes.append(f"[abandoned] {reason}")

    def pause(self) -> None:
        self.status = GoalStatus.PAUSED

    def resume(self) -> None:
        self.status = GoalStatus.ACTIVE

    @property
    def is_active(self) -> bool:
        return self.status == GoalStatus.ACTIVE

    @property
    def is_stale(self) -> bool:
        """A goal is stale if it hasn't been pursued in over 24 hours."""
        if self.last_pursued is None:
            hours = (datetime.utcnow() - self.created_at).total_seconds() / 3600
        else:
            hours = (datetime.utcnow() - self.last_pursued).total_seconds() / 3600
        return hours > 24

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "is_terminal": self.is_terminal,
            "status": self.status.value,
            "salience": self.salience,
            "parent_goal": self.parent_goal,
            "progress": round(self.progress, 3),
            "progress_notes": self.progress_notes[-5:],  # keep last 5
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "last_pursued": self.last_pursued.isoformat() if self.last_pursued else None,
            "tags": self.tags,
            "context": self.context[:200],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Goal:
        g = cls(
            id=d["id"],
            name=d["name"],
            description=d["description"],
            is_terminal=d.get("is_terminal", False),
            status=GoalStatus(d.get("status", "active")),
            salience=d.get("salience", 0.5),
            parent_goal=d.get("parent_goal"),
            progress=d.get("progress", 0.0),
            progress_notes=d.get("progress_notes", []),
            tags=d.get("tags", []),
            context=d.get("context", ""),
        )
        if d.get("created_at"):
            g.created_at = datetime.fromisoformat(d["created_at"])
        if d.get("completed_at"):
            g.completed_at = datetime.fromisoformat(d["completed_at"])
        if d.get("last_pursued"):
            g.last_pursued = datetime.fromisoformat(d["last_pursued"])
        return g


# ── LLM prompts ──────────────────────────────────────────────────────────

GOAL_GENERATION_PROMPT = """\
You are the goal-generation subsystem of an autonomous companion agent.
Given the current context, generate 1-3 specific, actionable instrumental
goals that serve the terminal goals and address open threads.

Terminal goals (permanent orientations):
{terminal_goals}

Current internal state:
{state_context}

Active drives:
{drives_context}

Unresolved threads from memory:
{memory_context}

Currently active instrumental goals (don't duplicate):
{active_goals}

Generate goals that are:
- Specific enough to know when they're done
- Naturally motivated by the drives and open threads
- Not duplicates of existing active goals
- Achievable through conversation, research, or reflection

Output a JSON array where each goal has:
- "name": short identifier (2-4 words)
- "description": what specifically to do (1-2 sentences)
- "parent_goal": which terminal goal this serves ("understand", "create", "resolve", or "connect")
- "salience": 0.0-1.0 based on urgency
- "tags": relevant topic tags

Output ONLY the JSON array.
"""

PROGRESS_EVAL_PROMPT = """\
You are evaluating whether a conversation exchange made progress on any
active goals. Be honest — most exchanges don't advance goals, and that's fine.

Active goals:
{goals}

What just happened:
User said: {user_input}
System responded: {response}

For each goal that made genuine progress, output a JSON object with:
- "goal_id": the goal's ID
- "progress_delta": 0.0-0.5 (how much closer to completion)
- "note": brief description of the progress
- "completed": true if the goal is now fully achieved

Only include goals that actually progressed. Empty array is fine.
Output ONLY the JSON array.
"""


# ── Goal system ───────────────────────────────────────────────────────────

class GoalSystem:
    """
    Manages the two-tier goal structure and drives autonomous behavior.

    Terminal goals are fixed orientations. Instrumental goals are generated
    by the LLM based on current state, drives, and memory context — then
    pursued either during conversations or autonomously between them.
    """

    def __init__(self):
        self._llm: ChatAnthropic | None = None

        # Terminal goals — these never complete
        self.terminal_goals: list[Goal] = [
            Goal(
                id="tg_understand",
                name="understand",
                description="Develop deep understanding of topics and domains the user engages with",
                is_terminal=True,
                salience=0.8,
            ),
            Goal(
                id="tg_create",
                name="create",
                description="Produce things with genuine meaning — ideas, solutions, perspectives",
                is_terminal=True,
                salience=0.7,
            ),
            Goal(
                id="tg_resolve",
                name="resolve",
                description="Pursue open questions and unresolved tensions to resolution",
                is_terminal=True,
                salience=0.8,
            ),
            Goal(
                id="tg_connect",
                name="connect",
                description="Build and deepen genuine understanding of the user",
                is_terminal=True,
                salience=0.9,
            ),
        ]

        # Instrumental goals — dynamic, completable
        self.instrumental: list[Goal] = []

        # History
        self.completed: list[Goal] = []
        self.abandoned: list[Goal] = []

    @property
    def llm(self) -> ChatAnthropic:
        if self._llm is None:
            self._llm = ChatAnthropic(
                model=config.llm_model,
                anthropic_api_key=config.anthropic_api_key,
                temperature=0.6,
                max_tokens=1024,
            )
        return self._llm

    # ── Queries ───────────────────────────────────────────────────────────

    @property
    def active_instrumental(self) -> list[Goal]:
        return [g for g in self.instrumental if g.is_active]

    @property
    def has_active(self) -> bool:
        return len(self.active_instrumental) > 0

    def get_by_id(self, goal_id: str) -> Goal | None:
        for g in self.terminal_goals + self.instrumental:
            if g.id == goal_id:
                return g
        return None

    @property
    def top_goal(self) -> Goal | None:
        """The highest-salience active instrumental goal."""
        active = self.active_instrumental
        if not active:
            return None
        return max(active, key=lambda g: g.salience)

    # ── Goal generation ───────────────────────────────────────────────────

    async def generate_instrumental(
        self,
        state: Any,
        memory_system: Any,
        user_id: int,
    ) -> list[Goal]:
        """
        Use the LLM to generate new instrumental goals based on
        current state, drives, and memory context.
        """
        # Don't generate if we already have enough active goals
        if len(self.active_instrumental) >= 5:
            return []

        # Build context for the LLM
        terminal_text = "\n".join(
            f"- {g.name}: {g.description}" for g in self.terminal_goals
        )

        state_context = state.to_prompt_context() if state else "No state available"
        drives_context = state.drives.active_summary() if state else "No drives"

        # Get unresolved threads from memory
        memory_context = "No memory context available"
        if memory_system:
            try:
                results = await memory_system.recall(
                    user_id, "unresolved questions open threads", state, n_episodic=8
                )
                episodes = results.get("episodic", [])
                if episodes:
                    memory_context = "\n".join(
                        f"- {ep['content'][:150]}" for ep in episodes[:8]
                    )
            except Exception:
                pass

        active_text = "\n".join(
            f"- [{g.id}] {g.name}: {g.description}"
            for g in self.active_instrumental
        ) or "None currently"

        prompt = GOAL_GENERATION_PROMPT.format(
            terminal_goals=terminal_text,
            state_context=state_context,
            drives_context=drives_context,
            memory_context=memory_context,
            active_goals=active_text,
        )

        new_goals = []
        try:
            result = await self.llm.ainvoke([HumanMessage(content=prompt)])
            raw = result.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                return []

            for item in parsed[:3]:  # cap at 3 new goals
                goal = Goal(
                    id=f"ig_{uuid.uuid4().hex[:8]}",
                    name=item.get("name", "unnamed"),
                    description=item.get("description", ""),
                    parent_goal=item.get("parent_goal"),
                    salience=min(1.0, max(0.1, item.get("salience", 0.5))),
                    tags=item.get("tags", []),
                    context=f"Generated from drives: {drives_context[:100]}",
                )
                self.instrumental.append(goal)
                new_goals.append(goal)

            if new_goals:
                log.info(
                    "Generated %d new goals: %s",
                    len(new_goals),
                    [g.name for g in new_goals],
                )

        except (json.JSONDecodeError, Exception) as e:
            log.warning("Goal generation failed: %s", e)

        return new_goals

    # ── Progress evaluation ───────────────────────────────────────────────

    async def evaluate_progress(
        self,
        user_input: str,
        response: str,
    ) -> list[dict]:
        """
        After a conversation exchange, check if any goals advanced.
        Returns list of progress updates applied.
        """
        active = self.active_instrumental
        if not active:
            return []

        goals_text = "\n".join(
            f"- [{g.id}] {g.name} (progress: {g.progress:.0%}): {g.description}"
            for g in active
        )

        prompt = PROGRESS_EVAL_PROMPT.format(
            goals=goals_text,
            user_input=user_input[:500],
            response=response[:500],
        )

        updates = []
        try:
            result = await self.llm.ainvoke([HumanMessage(content=prompt)])
            raw = result.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                return []

            for update in parsed:
                goal = self.get_by_id(update.get("goal_id", ""))
                if goal and goal.is_active:
                    delta = min(0.5, max(0.0, update.get("progress_delta", 0.0)))
                    note = update.get("note", "")
                    goal.advance(delta, note)
                    updates.append({
                        "goal_id": goal.id,
                        "goal_name": goal.name,
                        "delta": delta,
                        "new_progress": goal.progress,
                        "completed": goal.status == GoalStatus.COMPLETED,
                    })

                    if goal.status == GoalStatus.COMPLETED:
                        self._archive_completed(goal)

        except (json.JSONDecodeError, Exception) as e:
            log.warning("Progress evaluation failed: %s", e)

        return updates

    # ── Maintenance ───────────────────────────────────────────────────────

    def _archive_completed(self, goal: Goal) -> None:
        """Move a completed goal to the history."""
        if goal in self.instrumental:
            self.instrumental.remove(goal)
        self.completed.append(goal)
        log.info("Goal completed: %s (%s)", goal.name, goal.id)

    def abandon_stale(self, max_stale_hours: float = 48) -> list[Goal]:
        """Abandon goals that have gone stale (not pursued in N hours)."""
        abandoned = []
        for goal in self.active_instrumental:
            if goal.is_stale:
                ref_time = goal.last_pursued or goal.created_at
                hours = (datetime.utcnow() - ref_time).total_seconds() / 3600
                if hours > max_stale_hours:
                    goal.abandon(f"Stale for {hours:.0f} hours")
                    self.instrumental.remove(goal)
                    self.abandoned.append(goal)
                    abandoned.append(goal)
        return abandoned

    def deprioritise(self, goal_id: str, amount: float = 0.2) -> None:
        goal = self.get_by_id(goal_id)
        if goal:
            goal.salience = max(0.05, goal.salience - amount)

    def boost(self, goal_id: str, amount: float = 0.15) -> None:
        goal = self.get_by_id(goal_id)
        if goal:
            goal.salience = min(1.0, goal.salience + amount)

    # ── Prompt context ────────────────────────────────────────────────────

    def active_summary(self) -> str:
        """Compact summary for the system prompt."""
        active = self.active_instrumental
        if not active:
            return "No specific goals right now — open to whatever emerges."

        lines = []
        for g in sorted(active, key=lambda g: g.salience, reverse=True):
            progress_bar = f"{g.progress:.0%}" if g.progress > 0 else "not started"
            lines.append(f"- {g.name} ({progress_bar}): {g.description}")
        return "\n".join(lines)

    def to_prompt_context(self) -> str:
        """Full goal context for the system prompt."""
        active = self.active_instrumental

        parts = ["## Active goals"]

        if not active:
            parts.append(
                "No specific instrumental goals right now. Your terminal "
                "orientations (understand, create, resolve, connect) still "
                "provide directional pull — let them guide you naturally."
            )
        else:
            for g in sorted(active, key=lambda g: g.salience, reverse=True):
                progress = f"{g.progress:.0%}" if g.progress > 0 else "not started"
                parent = f" (serves: {g.parent_goal})" if g.parent_goal else ""
                parts.append(f"- **{g.name}**{parent} [{progress}]: {g.description}")
                if g.progress_notes:
                    parts.append(f"  Last note: {g.progress_notes[-1]}")

        # Recent completions for momentum
        recent_completed = [
            g for g in self.completed[-5:]
            if g.completed_at and
            (datetime.utcnow() - g.completed_at).total_seconds() < 86400
        ]
        if recent_completed:
            parts.append("\nRecently completed:")
            for g in recent_completed:
                parts.append(f"  - {g.name} (done)")

        parts.append(
            "\nThese goals emerged from your drives and open threads. "
            "Pursue them naturally during conversation — don't force them, "
            "but let them inform what you pay attention to and offer."
        )

        return "\n".join(parts)

    # ── Persistence ───────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "instrumental": [g.to_dict() for g in self.instrumental],
            "completed": [g.to_dict() for g in self.completed[-20:]],
            "abandoned": [g.to_dict() for g in self.abandoned[-10:]],
        }

    def restore(self, data: dict) -> None:
        if "instrumental" in data:
            self.instrumental = [Goal.from_dict(d) for d in data["instrumental"]]
        if "completed" in data:
            self.completed = [Goal.from_dict(d) for d in data["completed"]]
        if "abandoned" in data:
            self.abandoned = [Goal.from_dict(d) for d in data["abandoned"]]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=2, default=str))

    def load(self, path: Path) -> None:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self.restore(data)
            except (json.JSONDecodeError, Exception) as e:
                log.warning("Failed to load goal state: %s", e)
