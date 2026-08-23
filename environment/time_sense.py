"""Time sense — session rhythm and subjective duration, not just clock time."""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass
class SessionRecord:
    """A record of a past session."""
    started_at: datetime
    ended_at: datetime | None = None
    message_count: int = 0
    duration_minutes: float = 0.0

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "message_count": self.message_count,
            "duration_minutes": round(self.duration_minutes, 1),
        }

    @classmethod
    def from_dict(cls, d: dict) -> SessionRecord:
        return cls(
            started_at=datetime.fromisoformat(d["started_at"]),
            ended_at=datetime.fromisoformat(d["ended_at"]) if d.get("ended_at") else None,
            message_count=d.get("message_count", 0),
            duration_minutes=d.get("duration_minutes", 0.0),
        )


@dataclass
class TemporalLandmark:
    """A notable event pinned to a point in time."""
    description: str
    timestamp: datetime
    tags: list[str] = field(default_factory=list)

    @property
    def age_hours(self) -> float:
        return (datetime.utcnow() - self.timestamp).total_seconds() / 3600

    @property
    def age_description(self) -> str:
        hours = self.age_hours
        if hours < 1:
            return f"{hours * 60:.0f} minutes ago"
        elif hours < 24:
            return f"{hours:.1f} hours ago"
        elif hours < 168:
            return f"{hours / 24:.1f} days ago"
        else:
            return f"{hours / 168:.1f} weeks ago"

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TemporalLandmark:
        return cls(
            description=d["description"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            tags=d.get("tags", []),
        )


class TimeSense:
    """
    Subjective temporal awareness. Maintains a felt sense of time
    that goes beyond raw clock values.
    """

    def __init__(self):
        # Session history (last N sessions)
        self.sessions: deque[SessionRecord] = deque(maxlen=50)
        self._current_session: SessionRecord | None = None

        # Message timestamps within current session (for tempo tracking)
        self._message_times: deque[datetime] = deque(maxlen=100)

        # Temporal landmarks
        self.landmarks: deque[TemporalLandmark] = deque(maxlen=100)

        # Tracking
        self._first_interaction: datetime | None = None

    # ── Session tracking ──────────────────────────────────────────────────

    def start_session(self) -> None:
        """Mark the beginning of a new interaction session."""
        now = datetime.utcnow()
        self._current_session = SessionRecord(started_at=now)
        if self._first_interaction is None:
            self._first_interaction = now

    def end_session(self, message_count: int = 0) -> None:
        """Mark the end of a session and archive it."""
        if self._current_session is None:
            return

        now = datetime.utcnow()
        self._current_session.ended_at = now
        self._current_session.message_count = message_count
        self._current_session.duration_minutes = (
            (now - self._current_session.started_at).total_seconds() / 60
        )
        self.sessions.append(self._current_session)
        self._current_session = None

    def record_message(self) -> None:
        """Record a message timestamp for tempo tracking."""
        self._message_times.append(datetime.utcnow())
        if self._current_session:
            self._current_session.message_count += 1

    # ── Landmarks ─────────────────────────────────────────────────────────

    def add_landmark(self, description: str, tags: list[str] | None = None) -> None:
        """Pin a notable event to the current moment."""
        self.landmarks.append(TemporalLandmark(
            description=description,
            timestamp=datetime.utcnow(),
            tags=tags or [],
        ))

    def recent_landmarks(self, hours: float = 48, tags: list[str] | None = None) -> list[TemporalLandmark]:
        """Get landmarks from the last N hours, optionally filtered by tags."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        results = [lm for lm in self.landmarks if lm.timestamp > cutoff]
        if tags:
            results = [lm for lm in results if any(t in lm.tags for t in tags)]
        return sorted(results, key=lambda lm: lm.timestamp, reverse=True)

    # ── Derived temporal qualities ────────────────────────────────────────

    @property
    def conversation_tempo(self) -> str:
        """How fast messages are flowing in the current session."""
        if len(self._message_times) < 2:
            return "just started"

        # Average gap between last 5 messages
        recent = list(self._message_times)[-5:]
        gaps = [
            (recent[i+1] - recent[i]).total_seconds()
            for i in range(len(recent) - 1)
        ]
        avg_gap = sum(gaps) / len(gaps)

        if avg_gap < 15:
            return "rapid-fire"
        elif avg_gap < 60:
            return "flowing"
        elif avg_gap < 180:
            return "relaxed"
        elif avg_gap < 600:
            return "slow"
        else:
            return "sporadic"

    @property
    def session_rhythm(self) -> dict:
        """Patterns in how often and how long the user interacts."""
        if len(self.sessions) < 2:
            return {"pattern": "not enough data", "avg_gap_hours": None, "avg_duration_minutes": None}

        # Gaps between sessions
        gaps = []
        for i in range(1, len(self.sessions)):
            prev_end = self.sessions[i-1].ended_at or self.sessions[i-1].started_at
            gap = (self.sessions[i].started_at - prev_end).total_seconds() / 3600
            gaps.append(gap)

        durations = [s.duration_minutes for s in self.sessions if s.duration_minutes > 0]

        avg_gap = sum(gaps) / len(gaps) if gaps else 0
        avg_duration = sum(durations) / len(durations) if durations else 0

        # Characterise the rhythm
        if avg_gap < 2:
            pattern = "frequent — checks in multiple times a day"
        elif avg_gap < 8:
            pattern = "regular — a few sessions per day"
        elif avg_gap < 24:
            pattern = "daily — about once a day"
        elif avg_gap < 72:
            pattern = "periodic — every few days"
        else:
            pattern = "sporadic — comes and goes"

        return {
            "pattern": pattern,
            "avg_gap_hours": round(avg_gap, 1),
            "avg_duration_minutes": round(avg_duration, 1),
            "total_sessions": len(self.sessions),
        }

    @property
    def hours_since_last_session(self) -> float | None:
        """How long since the last completed session ended."""
        if not self.sessions:
            return None
        last = self.sessions[-1]
        ref = last.ended_at or last.started_at
        return (datetime.utcnow() - ref).total_seconds() / 3600

    @property
    def relationship_age_days(self) -> float | None:
        """How long since the very first interaction."""
        if self._first_interaction is None:
            return None
        return (datetime.utcnow() - self._first_interaction).total_seconds() / 86400

    @property
    def subjective_time_since_last(self) -> str:
        """
        A felt description of how long it's been, not just the number.
        Accounts for the rhythm — if the user usually checks in daily,
        two days feels long. If they usually come weekly, two days is soon.
        """
        hours = self.hours_since_last_session
        if hours is None:
            return "this is our first session"

        rhythm = self.session_rhythm
        avg_gap = rhythm.get("avg_gap_hours")

        if hours < 0.5:
            return "you just left and came right back"
        elif hours < 1:
            return "it's been a little while"
        elif hours < 4:
            return "a few hours since we talked"

        # Compare against the user's normal rhythm
        if avg_gap and avg_gap > 0:
            ratio = hours / avg_gap
            if ratio < 0.5:
                return f"you're back sooner than usual ({hours:.0f}h vs typical {avg_gap:.0f}h)"
            elif ratio < 1.5:
                return f"about the usual gap ({hours:.0f}h)"
            elif ratio < 3:
                return f"it's been longer than usual ({hours:.0f}h vs typical {avg_gap:.0f}h)"
            else:
                return f"it's been a while — {hours:.0f}h since we last talked"

        # Fallback without rhythm data
        if hours < 24:
            return f"{hours:.0f} hours since we talked"
        elif hours < 72:
            return f"about {hours/24:.0f} days since we talked"
        else:
            return f"it's been {hours/24:.0f} days — good to hear from you"

    # ── Prompt context ────────────────────────────────────────────────────

    def to_prompt_context(self) -> str:
        """Temporal awareness block for the system prompt."""
        parts = []

        # Time of day
        now = datetime.utcnow()
        parts.append(f"It's {now.strftime('%A')} {now.strftime('%H:%M')} UTC, {self._time_of_day()}")

        # Session gap
        subjective = self.subjective_time_since_last
        if subjective:
            parts.append(subjective.capitalize())

        # Current session duration
        if self._current_session:
            dur = (datetime.utcnow() - self._current_session.started_at).total_seconds() / 60
            msg_count = self._current_session.message_count
            if dur > 5:
                parts.append(f"This session: {dur:.0f}m, {msg_count} messages ({self.conversation_tempo} pace)")

        # Relationship age
        age = self.relationship_age_days
        if age is not None and age > 1:
            parts.append(f"Known each other: {age:.0f} days")

        # Recent landmarks
        recent = self.recent_landmarks(hours=48)
        if recent:
            lm_parts = [f"{lm.description} ({lm.age_description})" for lm in recent[:3]]
            parts.append("Recent: " + "; ".join(lm_parts))

        return " | ".join(parts)

    def _time_of_day(self) -> str:
        hour = datetime.utcnow().hour
        if hour < 6:
            return "late night"
        elif hour < 12:
            return "morning"
        elif hour < 17:
            return "afternoon"
        elif hour < 21:
            return "evening"
        else:
            return "night"

    # ── Persistence ───────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "sessions": [s.to_dict() for s in self.sessions],
            "landmarks": [lm.to_dict() for lm in self.landmarks],
            "first_interaction": self._first_interaction.isoformat() if self._first_interaction else None,
        }

    def restore(self, data: dict) -> None:
        if "sessions" in data:
            self.sessions = deque(
                [SessionRecord.from_dict(s) for s in data["sessions"]],
                maxlen=50,
            )
        if "landmarks" in data:
            self.landmarks = deque(
                [TemporalLandmark.from_dict(lm) for lm in data["landmarks"]],
                maxlen=100,
            )
        if data.get("first_interaction"):
            self._first_interaction = datetime.fromisoformat(data["first_interaction"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=2, default=str))

    def load(self, path: Path) -> None:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self.restore(data)
            except (json.JSONDecodeError, Exception) as e:
                import logging
                logging.getLogger(__name__).warning("Failed to load time sense: %s", e)
