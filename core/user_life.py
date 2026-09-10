"""User-life tracker — commitments, wellbeing, achievements. Success is their life, not chat volume."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from core.timeutil import utcnow, parse_iso

log = logging.getLogger(__name__)


@dataclass
class UserCommitment:
    """Something the user said they'd do."""
    description: str
    mentioned_at: str  # ISO timestamp
    deadline: str | None = None
    status: str = "open"  # open | completed | abandoned | unknown
    followup_count: int = 0
    resolved_at: str | None = None
    source_episode_ids: list[str] = field(default_factory=list)


@dataclass
class WellbeingSnapshot:
    """A point-in-time reading of user-reported state."""
    timestamp: str
    mood: str  # user's own words
    context: str
    inferred_valence: float  # -1.0 to 1.0


@dataclass
class LifeAchievement:
    """Something concrete the user accomplished."""
    description: str
    timestamp: str
    category: str  # career, health, creative, social, financial, personal
    keeper_contributed: bool


@dataclass
class UserLifeTracker:
    """
    Tracks the user's actual life trajectory.
    This is what the user_life_improvement unbounded goal optimizes for.
    """

    commitments: list[UserCommitment] = field(default_factory=list)
    wellbeing_history: list[WellbeingSnapshot] = field(default_factory=list)
    achievements: list[LifeAchievement] = field(default_factory=list)

    def record_commitment(
        self,
        description: str,
        deadline: str | None = None,
        source_episode_ids: list[str] | None = None,
    ):
        """User said they'd do something — track it."""
        ids = [str(s) for s in (source_episode_ids or [])]
        self.commitments.append(UserCommitment(
            description=description,
            mentioned_at=utcnow().isoformat(),
            deadline=deadline,
            source_episode_ids=ids,
        ))
        if ids:
            try:
                import asyncio
                from memory.episodic import episodic
                ref_id = f"commitment:{description[:40]}"
                loop = asyncio.get_running_loop()
                for eid in ids:
                    loop.create_task(episodic.add_episode_ref(eid, "user_life", ref_id))
            except Exception:
                log.warning("user_life episode ref failed", exc_info=True)

    def resolve_commitment(self, index: int, status: str):
        """Mark a commitment as completed, abandoned, or unknown."""
        if 0 <= index < len(self.commitments):
            self.commitments[index].status = status
            self.commitments[index].resolved_at = utcnow().isoformat()

    def record_wellbeing(self, mood: str, context: str, valence: float):
        """Snapshot of how the user seems to be doing."""
        self.wellbeing_history.append(WellbeingSnapshot(
            timestamp=utcnow().isoformat(),
            mood=mood,
            context=context[:200],
            inferred_valence=max(-1.0, min(1.0, valence)),
        ))
        if len(self.wellbeing_history) > 100:
            self.wellbeing_history = self.wellbeing_history[-100:]

    def record_achievement(self, description: str, category: str,
                           keeper_contributed: bool = False):
        """User accomplished something real."""
        self.achievements.append(LifeAchievement(
            description=description,
            timestamp=utcnow().isoformat(),
            category=category,
            keeper_contributed=keeper_contributed,
        ))

    # ── Derived metrics ───────────────────────────────────────────────────

    @property
    def commitment_followthrough_rate(self) -> float:
        """What fraction of commitments did the user actually complete?"""
        resolved = [c for c in self.commitments if c.status != "open"]
        if not resolved:
            return 0.5  # insufficient data
        completed = sum(1 for c in resolved if c.status == "completed")
        return completed / len(resolved)

    @property
    def wellbeing_trend(self) -> str:
        """Is the user's reported state improving over time?"""
        if len(self.wellbeing_history) < 5:
            return "insufficient_data"
        recent = self.wellbeing_history[-5:]
        older = (
            self.wellbeing_history[-10:-5]
            if len(self.wellbeing_history) >= 10
            else recent
        )
        r_avg = sum(w.inferred_valence for w in recent) / len(recent)
        o_avg = sum(w.inferred_valence for w in older) / len(older)
        if r_avg > o_avg + 0.15:
            return "improving"
        elif r_avg < o_avg - 0.15:
            return "declining"
        return "stable"

    @property
    def open_commitments(self) -> list[UserCommitment]:
        return [c for c in self.commitments if c.status == "open"]

    @property
    def stale_commitments(self) -> list[UserCommitment]:
        """Commitments open for >7 days with no followup."""
        cutoff = (utcnow() - timedelta(days=7)).isoformat()
        return [
            c for c in self.commitments
            if c.status == "open" and c.mentioned_at < cutoff
        ]

    def to_metrics(self) -> dict:
        return {
            "commitment_followthrough": self.commitment_followthrough_rate,
            "open_commitments": len(self.open_commitments),
            "stale_commitments": len(self.stale_commitments),
            "wellbeing_trend": self.wellbeing_trend,
            "achievements_count": len(self.achievements),
        }

    def to_prompt_context(self) -> str:
        """Injected when pursuing the user_life_improvement goal."""
        parts = []

        if self.open_commitments:
            parts.append("## User's open commitments")
            for c in self.open_commitments:
                stale = " (stale — consider checking in)" if c in self.stale_commitments else ""
                parts.append(f"- {c.description} (since {c.mentioned_at[:10]}){stale}")

        if self.wellbeing_trend != "insufficient_data":
            parts.append(f"\nWellbeing trend: {self.wellbeing_trend}")

        if self.achievements:
            recent = self.achievements[-3:]
            parts.append("\nRecent achievements:")
            for a in recent:
                parts.append(f"- [{a.category}] {a.description}")

        return "\n".join(parts) if parts else ""

    # ── Persistence ───────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "commitments": [
                {
                    "description": c.description,
                    "mentioned_at": c.mentioned_at,
                    "deadline": c.deadline,
                    "status": c.status,
                    "followup_count": c.followup_count,
                    "resolved_at": c.resolved_at,
                    "source_episode_ids": c.source_episode_ids,
                }
                for c in self.commitments
            ],
            "wellbeing_history": [
                {
                    "timestamp": w.timestamp,
                    "mood": w.mood,
                    "context": w.context,
                    "inferred_valence": w.inferred_valence,
                }
                for w in self.wellbeing_history
            ],
            "achievements": [
                {
                    "description": a.description,
                    "timestamp": a.timestamp,
                    "category": a.category,
                    "keeper_contributed": a.keeper_contributed,
                }
                for a in self.achievements
            ],
        }

    def restore(self, data: dict):
        self.commitments = [
            UserCommitment(**c) for c in data.get("commitments", [])
        ]
        self.wellbeing_history = [
            WellbeingSnapshot(**w) for w in data.get("wellbeing_history", [])
        ]
        self.achievements = [
            LifeAchievement(**a) for a in data.get("achievements", [])
        ]
