"""
Internal state — the system's continuously maintained inner conditions.

This is Pillar 2 of The_core architecture, and arguably the most important
thing that standard agent frameworks completely ignore.

Four primary dimensions:
  - Arousal:   general activation level (sharpens focus vs diffuse processing)
  - Valence:   positive/negative affect (colors interpretation of ambiguous input)
  - Curiosity: interest in current topic (drives engagement depth)
  - Fatigue:   processing degradation over extended use (recovers during rest)

Plus a DriveSystem for motivational states.

The crucial property: this state *actually influences* every LLM call.
It's not decorative metadata — it gets injected into the system prompt
and shapes how the model processes input. And it updates based on what
happens, creating a genuine feedback loop.

State persists to disk between sessions so the system "wakes up"
in roughly the condition it was in when it last ran.
"""
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

# How fast each dimension drifts back to baseline when nothing happens
BASELINE = {
    "arousal": 0.4,
    "valence": 0.55,
    "curiosity": 0.4,
    "fatigue": 0.0,
}

# Per-minute drift rate toward baseline
DRIFT_RATE = {
    "arousal": 0.008,
    "valence": 0.005,
    "curiosity": 0.006,
    "fatigue": -0.003,    # fatigue recovers (decreases) during inactivity
}

# Clamp all dimensions to these bounds
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

        # Running stats for novelty detection
        self._recent_topics: list[str] = []
        self._messages_this_session: int = 0

    # ── Event processing ──────────────────────────────────────────────────

    def update(self, event: Event) -> dict[str, float]:
        """
        Update state based on an event. Returns the deltas applied.
        This is the core feedback mechanism — what happens in the world
        changes how the system feels, which changes how it processes
        the next thing.
        """
        delta = self._compute_delta(event)
        self._apply_delta(delta)

        # Time-based drift toward baseline
        elapsed = (datetime.utcnow() - self.last_updated).total_seconds() / 60
        if elapsed > 0:
            self._drift_toward_baseline(elapsed)
            self.drives.tick(elapsed)

        self.last_updated = datetime.utcnow()
        return delta

    def _compute_delta(self, event: Event) -> dict[str, float]:
        """
        Map an event to state dimension changes.
        This is where the personality lives — how the system reacts
        to different kinds of stimuli.
        """
        d: dict[str, float] = {}

        if isinstance(event, UserMessageEvent):
            # Novel input → arousal + curiosity spike
            # Repetitive input → fatigue builds
            d["arousal"] = 0.05 + event.novelty * 0.15
            d["curiosity"] = (event.novelty - 0.3) * 0.2   # novel = curious, repetitive = bored
            d["fatigue"] = (1 - event.novelty) * 0.03       # repetition is tiring
            d["valence"] = (event.emotional_tone - 0.5) * 0.1  # user mood bleeds through

            # Complex questions are energising
            d["arousal"] += event.complexity * 0.08
            d["curiosity"] += event.complexity * 0.1

            self._messages_this_session += 1

            # Satisfy the "connect" drive — user is engaging
            self.drives.satisfy_by_name("connect", 0.15)

            # High-novelty input partially satisfies "understand"
            if event.novelty > 0.6:
                self.drives.satisfy_by_name("understand", 0.1)

        elif isinstance(event, UserSilenceEvent):
            # Silence → arousal drops, connect drive builds
            d["arousal"] = -min(0.15, event.duration_minutes * 0.01)
            d["fatigue"] = -min(0.1, event.duration_minutes * 0.005)  # rest recovers fatigue

        elif isinstance(event, ResponseGeneratedEvent):
            # Generating a response costs effort
            d["fatigue"] = 0.02
            if event.tool_calls_made > 0:
                d["fatigue"] += 0.01 * event.tool_calls_made
                d["arousal"] += 0.03  # tool use is slightly stimulating

            # Producing output partially satisfies "create" drive
            self.drives.satisfy_by_name("create", 0.05)

        elif isinstance(event, ToolResultEvent):
            if event.success:
                d["valence"] = 0.05
                d["curiosity"] = 0.03  # successful tool use often reveals something
            else:
                d["valence"] = -0.08
                d["arousal"] = 0.05   # errors are alerting

        elif isinstance(event, TimePassingEvent):
            # Just time going by — handled mostly by drift,
            # but extended time builds the connect drive
            pass  # drift handles this

        elif isinstance(event, StreamUpdateEvent):
            # Something interesting from the environment
            d["arousal"] = event.salience * 0.1
            d["curiosity"] = event.salience * 0.15

        elif isinstance(event, SessionStartEvent):
            # Waking up after time away
            d["arousal"] = 0.15
            d["curiosity"] = 0.1
            # Long gaps increase the connect drive retroactively
            if event.hours_since_last > 12:
                self.drives.satisfy_by_name("connect", -0.2)  # it built up

            self._messages_this_session = 0

        elif isinstance(event, SessionEndEvent):
            d["arousal"] = -0.1
            d["fatigue"] = -0.05  # session end is slightly restful

        elif isinstance(event, GoalProgressEvent):
            d["valence"] = 0.1 + event.progress_delta * 0.15
            d["arousal"] = 0.05
            self.drives.satisfy_by_name("resolve", event.progress_delta * 0.3)

        elif isinstance(event, GoalCompletedEvent):
            d["valence"] = 0.2
            d["arousal"] = 0.1
            self.drives.satisfy_by_name("resolve", 0.4)

        elif isinstance(event, GoalFrustratedEvent):
            d["valence"] = -0.15
            d["arousal"] = 0.1   # frustration is activating
            self.drives.satisfy_by_name("resolve", -0.1)  # drives up the urge

        elif isinstance(event, ConsolidationEvent):
            # Learning happened in the background
            if event.patterns_found > 0:
                d["valence"] = 0.05
                self.drives.satisfy_by_name("understand", 0.15)

        elif isinstance(event, SelfObservationEvent):
            if event.tension_detected:
                d["curiosity"] = 0.1   # tensions are interesting
                d["arousal"] = 0.05

        return d

    def _apply_delta(self, delta: dict[str, float]) -> None:
        for key, value in delta.items():
            if hasattr(self, key):
                current = getattr(self, key)
                setattr(self, key, _clamp(current + value))

    def _drift_toward_baseline(self, minutes: float) -> None:
        """Gently pull all dimensions back toward their resting values."""
        for dim, baseline in BASELINE.items():
            rate = DRIFT_RATE.get(dim, 0.005)
            current = getattr(self, dim)
            setattr(self, dim, _clamp(_drift(current, baseline, abs(rate), minutes)))

    # ── Derived signals ───────────────────────────────────────────────────

    def compute_salience(self) -> float:
        """
        How emotionally salient is the current moment?
        Used by the memory system to weight encoding strength.
        High arousal + strong valence (positive or negative) = high salience.
        """
        valence_intensity = abs(self.valence - 0.5) * 2  # 0 at neutral, 1 at extremes
        return _clamp((self.arousal * 0.6 + valence_intensity * 0.4))

    @property
    def engagement_level(self) -> str:
        """Human-readable engagement level."""
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
        """Suggests how the LLM should approach its response."""
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
        """
        Injected into every LLM call. This is what makes internal state
        actually matter — it's not metadata, it shapes processing.
        """
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
        """Full state snapshot for memory storage and persistence."""
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
        """Restore from a snapshot (e.g. on startup)."""
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
            # Apply drift for time that passed while the system was off
            elapsed = (datetime.utcnow() - self.last_updated).total_seconds() / 60
            if elapsed > 1:
                self._drift_toward_baseline(elapsed)
                self.drives.tick(elapsed)

    def save(self, path: Path) -> None:
        """Persist full state to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=2))

    def load(self, path: Path) -> None:
        """Restore state from disk."""
        if path.exists():
            data = json.loads(path.read_text())
            self.restore(data)
