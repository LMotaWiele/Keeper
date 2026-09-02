"""Typed events — the shared vocabulary for state, memory, and self-model updates."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from core.timeutil import utcnow, parse_iso
from typing import Any, Literal


EventSource = Literal[
    "user",          # external human input
    "system",        # internal system action
    "environment",   # background stream or time
    "goal",          # goal-related
    "self",          # self-observation
]


@dataclass
class Event:
    """Base event. All events carry a source, timestamp, and optional payload."""

    source: EventSource
    type: str
    timestamp: datetime = field(default_factory=utcnow)
    payload: dict = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        return (utcnow() - self.timestamp).total_seconds()


# ── User events ───────────────────────────────────────────────────────────

@dataclass
class UserMessageEvent(Event):
    """The user sent a message."""
    source: EventSource = "user"
    type: str = "message"
    content: str = ""
    novelty: float = 0.5       # 0 = repetitive, 1 = completely new topic
    complexity: float = 0.5    # 0 = simple greeting, 1 = deep multi-part question
    emotional_tone: float = 0.5  # 0 = negative, 0.5 = neutral, 1 = positive


@dataclass
class UserSilenceEvent(Event):
    """The user hasn't said anything for a while."""
    source: EventSource = "user"
    type: str = "silence"
    duration_minutes: float = 0.0


# ── System events ─────────────────────────────────────────────────────────

@dataclass
class ResponseGeneratedEvent(Event):
    """The system produced a response."""
    source: EventSource = "system"
    type: str = "response_generated"
    content: str = ""
    tool_calls_made: int = 0
    processing_time_ms: float = 0.0


@dataclass
class ToolResultEvent(Event):
    """A tool call returned a result."""
    source: EventSource = "system"
    type: str = "tool_result"
    tool_name: str = ""
    success: bool = True
    content: str = ""


@dataclass
class ConsolidationEvent(Event):
    """Memory consolidation completed."""
    source: EventSource = "system"
    type: str = "consolidation"
    patterns_found: int = 0
    episodes_forgotten: int = 0


# ── Environment events ────────────────────────────────────────────────────

@dataclass
class TimePassingEvent(Event):
    """Time has passed — the most basic environmental signal."""
    source: EventSource = "environment"
    type: str = "time_passing"
    minutes_elapsed: float = 1.0


@dataclass
class StreamUpdateEvent(Event):
    """Something arrived from an environmental stream (RSS, news, etc.)."""
    source: EventSource = "environment"
    type: str = "stream_update"
    stream_name: str = ""
    content: str = ""
    salience: float = 0.5


@dataclass
class SessionStartEvent(Event):
    """A new interaction session has begun."""
    source: EventSource = "environment"
    type: str = "session_start"
    hours_since_last: float = 0.0
    time_of_day: str = ""


@dataclass
class SessionEndEvent(Event):
    """An interaction session has ended (user went quiet)."""
    source: EventSource = "environment"
    type: str = "session_end"
    messages_exchanged: int = 0
    duration_minutes: float = 0.0


# ── Goal events ───────────────────────────────────────────────────────────

@dataclass
class GoalProgressEvent(Event):
    """Progress was made toward a goal."""
    source: EventSource = "goal"
    type: str = "goal_progress"
    goal_id: str = ""
    goal_description: str = ""
    progress_delta: float = 0.0  # how much closer (0–1)


@dataclass
class GoalCompletedEvent(Event):
    """A goal was achieved."""
    source: EventSource = "goal"
    type: str = "goal_completed"
    goal_id: str = ""
    goal_description: str = ""


@dataclass
class GoalFrustratedEvent(Event):
    """A goal is blocked or failing."""
    source: EventSource = "goal"
    type: str = "goal_frustrated"
    goal_id: str = ""
    reason: str = ""


# ── Self events ───────────────────────────────────────────────────────────

@dataclass
class SelfObservationEvent(Event):
    """The self-model noticed something about the system's own behavior."""
    source: EventSource = "self"
    type: str = "self_observation"
    observation: str = ""
    tension_detected: bool = False
