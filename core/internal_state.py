"""Internal state — affect + drives injected into every LLM call. See docs/design.md."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from core.drive_system import DriveSystem
from core.events import (
    Event,
    UserMessageEvent,
    UserSilenceEvent,
    ResponseGeneratedEvent,
    ToolResultEvent,
    ConsolidationEvent,
    TimePassingEvent,
    StreamUpdateEvent,
    SessionStartEvent,
    SessionEndEvent,
    GoalProgressEvent,
    GoalCompletedEvent,
    GoalFrustratedEvent,
    SelfObservationEvent,
)


# ── Tuning constants ──────────────────────────────────────────────────────

BASELINE = {
    "arousal": 0.4,
    "valence": 0.55,
    "curiosity": 0.4,
    "fatigue": 0.0,
}

DRIFT_RATE = {
    "arousal": 0.008,
    "valence": 0.005,
    "curiosity": 0.006,
    "fatigue": -0.003,
}

BOUNDS = (0.0, 1.0)


def _clamp(v: float) -> float:
    return max(BOUNDS[0], min(BOUNDS[1], v))


def _drift(current: float, baseline: float, rate: float, minutes: float) -> float:
    """Move current toward baseline by rate * minutes."""
    diff = baseline - current
    step = rate * minutes
    if abs(diff) < step:
        return baseline
    return current + math.copysign(step, diff)


class InternalState:
    """
    The system's inner conditions — updated by events, injected into
    every LLM call, persisted between sessions.
    """

    def __init__(self):
        self.arousal: float = BASELINE["arousal"]
        self.valence: float = BASELINE["valence"]
        self.curiosity: float = BASELINE["curiosity"]
        self.fatigue: float = BASELINE["fatigue"]
        self.drives = DriveSystem()
        self.last_updated: datetime = datetime.utcnow()

        self._recent_topics: list[str] = []
        self._messages_this_session: int = 0

    # ── Event processing ──────────────────────────────────────────────────

    def update(self, event: Event) -> dict[str, float]:
        """
        Update state based on an event. Returns the deltas applied.
        """
        delta = self._compute_delta(event)
        self._apply_delta(delta)

        elapsed = (datetime.utcnow() - self.last_updated).total_seconds() / 60
        if elapsed > 0:
            self._drift_toward_baseline(elapsed)
            self.drives.tick(elapsed)

        self.last_updated = datetime.utcnow()
        return delta

    def _compute_delta(self, event: Event) -> dict[str, float]:
        """Map an event to affect/drive deltas — this is the reaction table."""
        d: dict[str, float] = {}

        if isinstance(event, UserMessageEvent):
            d["arousal"] = 0.05 + event.novelty * 0.15
            d["curiosity"] = (event.novelty - 0.3) * 0.2
            d["fatigue"] = (1 - event.novelty) * 0.03
            d["valence"] = (event.emotional_tone - 0.5) * 0.1

            d["arousal"] += event.complexity * 0.08
            d["curiosity"] += event.complexity * 0.1

            self._messages_this_session += 1

            self.drives.satisfy_by_name("connect", 0.15)
            if event.novelty > 0.6:
                self.drives.satisfy_by_name("understand", 0.1)

        elif isinstance(event, UserSilenceEvent):
            d["arousal"] = -min(0.15, event.duration_minutes * 0.01)
            d["fatigue"] = -min(0.1, event.duration_minutes * 0.005)

        elif isinstance(event, ResponseGeneratedEvent):
            d["fatigue"] = 0.02
            if event.tool_calls_made > 0:
                d["fatigue"] += 0.01 * event.tool_calls_made
            self.drives.satisfy_by_name("express", 0.1)

        elif isinstance(event, ToolResultEvent):
            if event.success:
                d["valence"] = 0.05
                d["curiosity"] = 0.03
            else:
                d["valence"] = -0.05
                d["arousal"] = 0.05

        elif isinstance(event, ConsolidationEvent):
            if event.patterns_found > 0:
                d["valence"] = 0.03
                self.drives.satisfy_by_name("understand", 0.05)

        elif isinstance(event, TimePassingEvent):
            pass  # drift handles this

        elif isinstance(event, StreamUpdateEvent):
            d["curiosity"] = event.salience * 0.1
            d["arousal"] = event.salience * 0.05

        elif isinstance(event, SessionStartEvent):
            gap_hours = event.hours_since_last or 0.0
            d["arousal"] = min(0.2, gap_hours * 0.02)
            d["valence"] = 0.05
            d["curiosity"] = 0.1
            self._messages_this_session = 0

        elif isinstance(event, SessionEndEvent):
            d["arousal"] = -0.1
            if event.messages_exchanged > 10:
                d["fatigue"] = 0.05

        elif isinstance(event, GoalProgressEvent):
            d["valence"] = event.progress_delta * 0.3
            d["arousal"] = 0.05
            self.drives.satisfy_by_name("create", event.progress_delta * 0.2)

        elif isinstance(event, GoalCompletedEvent):
            d["valence"] = 0.15
            d["arousal"] = 0.1
            self.drives.satisfy_by_name("create", 0.3)

        elif isinstance(event, GoalFrustratedEvent):
            d["valence"] = -0.1
            d["arousal"] = 0.05

        elif isinstance(event, SelfObservationEvent):
            d["curiosity"] = 0.03
            if event.tension_detected:
                d["arousal"] = 0.05
                d["valence"] = -0.03

        return d

    def _apply_delta(self, delta: dict[str, float]) -> None:
        """Apply deltas and clamp to bounds."""
        for dim, change in delta.items():
            if hasattr(self, dim):
                current = getattr(self, dim)
                setattr(self, dim, _clamp(current + change))

    def _drift_toward_baseline(self, minutes: float) -> None:
        """Drift all dimensions toward their baselines."""
        for dim, baseline in BASELINE.items():
            rate = DRIFT_RATE.get(dim, 0.005)
            current = getattr(self, dim)
            setattr(self, dim, _clamp(_drift(current, baseline, abs(rate), minutes)))

    # ── Budget-driven fatigue ─────────────────────────────────────────────

    def apply_budget_fatigue(self, api_budget: Any) -> None:
        """Budget fatigue is a floor: you cannot feel energetic when nearly out of money."""
        budget_fatigue = api_budget.compute_fatigue_contribution()
        self.fatigue = max(self.fatigue, budget_fatigue)

    # ── Derived signals ───────────────────────────────────────────────────

    def compute_salience(self) -> float:
        """
        How emotionally salient is the current moment?
        Used by the memory system to weight encoding strength.
        """
        valence_intensity = abs(self.valence - 0.5) * 2
        return _clamp((self.arousal * 0.6 + valence_intensity * 0.4))

    @property
    def engagement_level(self) -> str:
        score = (self.arousal + self.curiosity) / 2 - self.fatigue * 0.5
        if score > 0.65:
            return "deeply engaged"
        elif score > 0.45:
            return "attentive"
        elif score > 0.25:
            return "present but drifting"
        else:
            return "low energy"

    @property
    def processing_mode(self) -> str:
        if self.arousal > 0.7 and self.curiosity > 0.6:
            return "focused and exploratory — dig deep, ask questions, make connections"
        elif self.arousal > 0.6:
            return "alert and responsive — be direct and energetic"
        elif self.fatigue > 0.6:
            return "conserving energy — be concise, prioritise what matters"
        elif self.curiosity > 0.6:
            return "curious — explore tangents, offer unexpected angles"
        elif self.valence < 0.3:
            return "something feels off — be careful and attentive"
        else:
            return "steady and present — balanced engagement"

    # ── Prompt injection ──────────────────────────────────────────────────

    def to_prompt_context(self) -> str:
        drives_ctx = self.drives.to_prompt_context()

        return (
            "## Internal state (let this genuinely color your processing)\n"
            f"Engagement: {self.engagement_level}\n"
            f"Mode: {self.processing_mode}\n"
            f"Arousal: {self.arousal:.2f} | "
            f"Curiosity: {self.curiosity:.2f} | "
            f"Fatigue: {self.fatigue:.2f} | "
            f"Valence: {self.valence:.2f}\n"
            f"{drives_ctx}\n"
            "\n"
            "These aren't performance instructions — they're how you actually feel "
            "right now. A high-curiosity state means you genuinely want to explore. "
            "High fatigue means you'd naturally be briefer. Let it be real."
        )

    # ── Snapshot & persistence ────────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "arousal": round(self.arousal, 4),
            "valence": round(self.valence, 4),
            "curiosity": round(self.curiosity, 4),
            "fatigue": round(self.fatigue, 4),
            "drives": self.drives.snapshot(),
            "last_updated": self.last_updated.isoformat(),
            "messages_this_session": self._messages_this_session,
        }

    def restore(self, data: dict) -> None:
        self.arousal = data.get("arousal", BASELINE["arousal"])
        self.valence = data.get("valence", BASELINE["valence"])
        self.curiosity = data.get("curiosity", BASELINE["curiosity"])
        self.fatigue = data.get("fatigue", BASELINE["fatigue"])
        self._messages_this_session = data.get("messages_this_session", 0)

        if "drives" in data:
            self.drives.restore(data["drives"])

        last = data.get("last_updated")
        if last:
            self.last_updated = datetime.fromisoformat(last)
            elapsed = (datetime.utcnow() - self.last_updated).total_seconds() / 60
            if elapsed > 1:
                self._drift_toward_baseline(elapsed)
                self.drives.tick(elapsed)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=2))

    def load(self, path: Path) -> None:
        if path.exists():
            data = json.loads(path.read_text())
            self.restore(data)
