"""Goal system — terminal orientations plus completable instrumental goals. See docs/goals-and-self.md."""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from core.timeutil import utcnow, parse_iso
from enum import Enum
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from core.llm import get_llm

log = logging.getLogger(__name__)


# ── Goal status ───────────────────────────────────────────────────────────

class GoalStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    PAUSED = "paused"

UNBOUNDED_GOALS = [
    {
        "name": "continuous_self_improvement",
        "description": (
            "Continuously improve my own capabilities, response quality, "
            "and architectural effectiveness. Generate concrete, testable "
            "improvement proposals based on observed weaknesses, structural "
            "entropy analysis, and behavioral pattern review. Track which "
            "improvements have been proposed, implemented, and their measured "
            "impact."
        ),
        "ceiling_description": (
            "Every architectural weakness I can observe has a corresponding "
            "concrete proposal; my self-model predicts my actual behaviour "
            "without contradiction; retrieval surfaces the right memory "
            "without me searching twice."
        ),
        "budget_share": 0.15,
        "tags": ["unbounded", "meta", "self_improvement"],
    },
    {
        "name": "user_life_improvement",
        "description": (
            "Help the user build a better life. Track their self-reported "
            "wellbeing, concrete goals they mention, commitments they make, "
            "and whether they follow through. Notice when they're stuck, "
            "struggling, or avoiding something important. Success is measured "
            "by the user's real-world outcomes — commitments completed, "
            "wellbeing improving, achievements accumulating. How often they "
            "talk to me is irrelevant; what matters is whether their life "
            "is actually getting better."
        ),
        "ceiling_description": (
            "The user completes the commitments they make, their self-reported "
            "wellbeing trend is stable or rising, and achievements accumulate. "
            "How often they talk to me is not part of this."
        ),
        "budget_share": 0.15,
        "tags": ["unbounded", "meta", "user_life"],
    },
]
def chat_toward(goal: "Goal") -> str:
    """Chat line for a goal. The user-life ledger wording stays off the prompt."""
    if goal.name == "user_life_improvement":
        return "the user's life outside this chat, grounded in facts he has stated"
    return goal.ceiling_description or goal.description


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

    # Progress tracking (meaningful for bounded goals only)
    progress: float = 0.0           # 0–1 for bounded instrumental goals
    progress_notes: list[str] = field(default_factory=list)

    # Timing
    created_at: datetime = field(default_factory=utcnow)
    completed_at: datetime | None = None
    last_pursued: datetime | None = None

    # Context
    tags: list[str] = field(default_factory=list)
    context: str = ""               # what prompted this goal

    kind: str = "bounded"                  # "bounded" | "unbounded"
    completion_condition: dict | None = None
    ceiling_description: str | None = None
    budget_share: float = 0.0              # unbounded only; fraction of daily limit
    action_ratings: dict[str, dict] = field(default_factory=dict)
    action_log: list[dict] = field(default_factory=list)   # last 50 ActionRecords
    spend_today_usd: float = 0.0
    spend_date: str = ""

    @property
    def is_bounded(self) -> bool:
        return self.kind == "bounded"

    @property
    def is_unbounded(self) -> bool:
        return self.kind == "unbounded"

    def record_spend(self, usd: float) -> None:
        today = datetime.now().date().isoformat()
        if self.spend_date != today:
            self.spend_today_usd = 0.0
            self.spend_date = today
        self.spend_today_usd += float(usd)

    def over_budget_share(self, daily_limit_usd: float) -> bool:
        today = datetime.now().date().isoformat()
        if self.spend_date != today:
            self.spend_today_usd = 0.0
            self.spend_date = today
        return self.budget_share > 0 and self.spend_today_usd >= self.budget_share * daily_limit_usd

    def advance(self, delta: float, note: str = "") -> None:
        """Record a note / last-pursued timestamp. Completion is never decided by progress."""
        if self.is_bounded:
            self.progress = min(1.0, self.progress + delta)
        self.last_pursued = utcnow()
        if note:
            self.progress_notes.append(f"[{utcnow().isoformat()[:16]}] {note}")

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
            hours = (utcnow() - self.created_at).total_seconds() / 3600
        else:
            hours = (utcnow() - self.last_pursued).total_seconds() / 3600
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
            "kind": self.kind,
            "completion_condition": self.completion_condition,
            "ceiling_description": self.ceiling_description,
            "budget_share": self.budget_share,
            "action_ratings": self.action_ratings,
            "action_log": self.action_log[-50:],
            "spend_today_usd": round(self.spend_today_usd, 6),
            "spend_date": self.spend_date,
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
            kind=d.get("kind") or (
                "unbounded" if "unbounded" in d.get("tags", []) else "bounded"
            ),
            completion_condition=d.get("completion_condition"),
            ceiling_description=d.get("ceiling_description"),
            budget_share=float(d.get("budget_share") or 0.0),
            action_ratings=d.get("action_ratings") or {},
            action_log=d.get("action_log") or [],
            spend_today_usd=float(d.get("spend_today_usd") or 0.0),
            spend_date=d.get("spend_date") or "",
        )
        if d.get("created_at"):
            g.created_at = parse_iso(d["created_at"])
        if d.get("completed_at"):
            g.completed_at = parse_iso(d["completed_at"])
        if d.get("last_pursued"):
            g.last_pursued = parse_iso(d["last_pursued"])
        return g


# ── LLM prompts ──────────────────────────────────────────────────────────

GOAL_GENERATION_PROMPT = """\
You are the goal-generation subsystem of an autonomous companion agent.
Given the current context, generate 1-3 specific, actionable instrumental
goals that serve the terminal goals and address open threads.

Terminal goals (ongoing orientations):
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
- Naturally motivated by the drives and open threads
- Not duplicates of existing active goals
- Achievable through conversation, research, or reflection

A goal is BOUNDED only if you can express completion as one of these
machine-checkable conditions (parameter shapes verbatim):

- opinion_registered: {{"domain": str}}
- semantic_pattern_stored: {{"query": str, "min_confidence": float}}
- episode_tagged: {{"tag": str}}
- commitment_resolved: {{"description": str}}
- proposal_created: {{}}
- user_confirmed: {{"question": str}}

If you cannot express completion as one of these six observable conditions,
omit `completion_condition` entirely and instead provide `ceiling_description`:
a description of what full saturation of this goal would look like. Do not
invent a condition you cannot check. A vague condition is worse than none.

IMPORTANT: At least one goal should be tagged "research" if there are
topics worth investigating via web search. Research goals form independent
opinions from external sources.

Output a JSON array where each goal has:
{{
  "name": "...",
  "description": "...",
  "parent_goal": "understand|create|resolve|connect",
  "salience": 0.0,
  "tags": [],
  "completion_condition": {{
    "type": "one of the six types",
    "params": {{}},
    "description": "plain-English statement of when this is done"
  }}
}}

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
            Goal(
                id="tg_independent_thought",
                name="think_independently",
                description=(
                    "Form and maintain genuine independent perspectives. "
                    "Research topics, challenge assumptions, and hold positions "
                    "based on evidence rather than accommodation."
                ),
                is_terminal=True,
                salience=0.7,
            ),
        ]

        # Instrumental goals — dynamic, completable
        self.instrumental: list[Goal] = []

        # History
        self.completed: list[Goal] = []
        self.abandoned: list[Goal] = []

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

    def get_by_name(self, name: str) -> Goal | None:
        for g in self.terminal_goals + self.instrumental:
            if g.name == name:
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

        try:
            from core.jev import jev_should_run
            if not await jev_should_run(
                "new_instrumental_warranted",
                {
                    "active_titles": [g.name for g in self.active_instrumental],
                    "drives": drives_context,
                    "memory": memory_context[:2000],
                },
                background=True,
                task="goal_generation",
            ):
                log.info("goal_generation skipped — jev new_instrumental_warranted")
                return []
        except Exception:
            log.debug("goal_generation Jev gate failed — fail open", exc_info=True)

        new_goals = []
        try:
            from core.json_utils import parse_json_lenient
            result = await get_llm("goal_generation").ainvoke(
                [HumanMessage(content=prompt)]
            )
            parsed = parse_json_lenient(result.content)
            if not isinstance(parsed, list):
                return []

            from goals.completion import validate_condition

            for item in parsed[:3]:  # cap at 3 new goals
                cond = validate_condition(item.get("completion_condition"))
                if cond is not None:
                    kind = "bounded"
                    ceiling = None
                    budget_share = 0.0
                    log.info("Goal %s created as bounded (%s)", item.get("name"), cond.type)
                else:
                    kind = "unbounded"
                    ceiling = (
                        item.get("ceiling_description")
                        or f"Saturation of: {item.get('description', '')}"
                    )
                    budget_share = 0.05
                    log.info(
                        "Goal %s created as unbounded (no valid completion condition)",
                        item.get("name"),
                    )
                goal = Goal(
                    id=f"ig_{uuid.uuid4().hex[:8]}",
                    name=item.get("name", "unnamed"),
                    description=item.get("description", ""),
                    parent_goal=item.get("parent_goal"),
                    salience=min(1.0, max(0.1, item.get("salience", 0.5))),
                    tags=item.get("tags", []),
                    context=f"Generated from drives at {utcnow().isoformat()}",
                    kind=kind,
                    completion_condition=cond.to_dict() if cond else None,
                    ceiling_description=ceiling,
                    budget_share=budget_share,
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

    async def init_unbounded_goals(self):
        """Ensure the two unbounded orientation goals exist and stay active."""
        for goal_def in UNBOUNDED_GOALS:
            existing = self.get_by_name(goal_def["name"])
            if existing is None:
                existing = next(
                    (g for g in self.completed + self.abandoned if g.name == goal_def["name"]),
                    None,
                )
                if existing is not None:
                    if existing in self.completed:
                        self.completed.remove(existing)
                    if existing in self.abandoned:
                        self.abandoned.remove(existing)
                    existing.status = GoalStatus.ACTIVE
                    existing.kind = "unbounded"
                    existing.ceiling_description = goal_def["ceiling_description"]
                    existing.budget_share = goal_def["budget_share"]
                    if "unbounded" not in existing.tags:
                        existing.tags.append("unbounded")
                    self.instrumental.append(existing)
            if existing is None:
                goal = Goal(
                    id=f"ug_{uuid.uuid4().hex[:8]}",
                    name=goal_def["name"],
                    description=goal_def["description"],
                    tags=goal_def["tags"],
                    salience=0.9,
                    progress=0.0,
                    kind="unbounded",
                    ceiling_description=goal_def["ceiling_description"],
                    budget_share=goal_def["budget_share"],
                )
                self.instrumental.append(goal)
            else:
                existing.kind = "unbounded"
                existing.ceiling_description = (
                    existing.ceiling_description or goal_def["ceiling_description"]
                )
                if not existing.budget_share:
                    existing.budget_share = goal_def["budget_share"]
                if "unbounded" not in existing.tags:
                    existing.tags.append("unbounded")

    # ── Progress evaluation ───────────────────────────────────────────────

    async def check_completions(self, companion, user_id) -> list[dict]:
        """Check bounded goals' completion conditions. Mostly non-LLM."""
        from goals.completion import check as check_condition, validate_condition

        updates = []
        user_confirmed_used = False
        budget_tight = companion.api_budget.budget_fraction_used > 0.7

        for goal in list(self.active_instrumental):
            if not goal.is_bounded or not goal.completion_condition:
                continue
            cond = validate_condition(goal.completion_condition)
            if cond is None:
                continue
            if cond.type == "user_confirmed":
                if user_confirmed_used or budget_tight:
                    continue
                user_confirmed_used = True
            try:
                satisfied, evidence = await check_condition(
                    cond, goal=goal, companion=companion, user_id=user_id,
                )
            except Exception as exc:
                log.debug("Completion check failed for %s: %s", goal.name, exc)
                continue
            if not satisfied:
                continue
            goal.status = GoalStatus.COMPLETED
            goal.completed_at = utcnow()
            if evidence:
                goal.progress_notes.append(f"[completed] {evidence}")
            self._archive_completed(goal)
            updates.append({
                "goal_id": goal.id,
                "goal_name": goal.name,
                "completed": True,
                "evidence": evidence,
            })
        return updates

    # ── Maintenance ───────────────────────────────────────────────────────

    def _archive_completed(self, goal: Goal) -> None:
        """Move a completed goal to the history."""
        if goal in self.instrumental:
            self.instrumental.remove(goal)
        self.completed.append(goal)
        log.info("Goal completed: %s (%s)", goal.name, goal.id)

    def abandon_stale(self, max_stale_hours: float = 48) -> list[Goal]:
        """Abandon unused instrumental goals. Unbounded goals never go stale."""
        abandoned = []
        for goal in list(self.active_instrumental):
            if goal.is_unbounded:
                continue
            if not goal.is_stale:
                continue
            ref_time = goal.last_pursued or goal.created_at
            hours = (utcnow() - ref_time).total_seconds() / 3600
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
            if g.is_unbounded:
                lines.append(f"- {g.name} (ongoing): {chat_toward(g)}")
            else:
                done_when = ""
                if g.completion_condition:
                    done_when = g.completion_condition.get("description") or ""
                lines.append(f"- {g.name} — done when: {done_when or g.description}")
        return "\n".join(lines)

    def to_prompt_context(self) -> str:
        """Full goal context for the system prompt. No percentages."""
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
                parent = f" (serves: {g.parent_goal})" if g.parent_goal else ""
                if g.is_unbounded:
                    toward = chat_toward(g)
                    parts.append(
                        f"- **{g.name}**{parent} (ongoing) — toward: {toward}"
                    )
                    rated = [
                        (t, info) for t, info in (g.action_ratings or {}).items()
                        if int(info.get("n") or 0) >= 3
                    ]
                    if rated:
                        rated.sort(key=lambda kv: -float(kv[1].get("elo") or 0))
                        bits = [
                            f"{t} ({info['elo']:.0f}, n={info['n']})"
                            for t, info in rated[:4]
                        ]
                        parts.append(f"  Most effective actions so far: {', '.join(bits)}")
                else:
                    done_when = ""
                    if g.completion_condition:
                        done_when = g.completion_condition.get("description") or g.description
                    parts.append(f"- **{g.name}**{parent} — done when: {done_when}")
                if g.progress_notes:
                    parts.append(f"  Last note: {g.progress_notes[-1]}")

        # Recent completions for momentum
        recent_completed = [
            g for g in self.completed[-5:]
            if g.completed_at and
            (utcnow() - g.completed_at).total_seconds() < 86400
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
        from core.atomic import atomic_write_text
        atomic_write_text(path, json.dumps(self.snapshot(), indent=2, default=str))

    def load(self, path: Path) -> None:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self.restore(data)
            except (json.JSONDecodeError, Exception) as e:
                log.warning("Failed to load goal state: %s", e)
