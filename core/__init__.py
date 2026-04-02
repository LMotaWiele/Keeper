"""
Core — the system's inner life.

Exports:
  - InternalState: the continuously maintained state vector
  - DriveSystem: motivational hunger states
  - Event types: the signals that drive state changes
"""
from core.internal_state import InternalState
from core.drive_system import DriveSystem, Drive
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

__all__ = [
    "InternalState",
    "DriveSystem",
    "Drive",
    "Event",
    "UserMessageEvent",
    "UserSilenceEvent",
    "ResponseGeneratedEvent",
    "ToolResultEvent",
    "ConsolidationEvent",
    "TimePassingEvent",
    "StreamUpdateEvent",
    "SessionStartEvent",
    "SessionEndEvent",
    "GoalProgressEvent",
    "GoalCompletedEvent",
    "GoalFrustratedEvent",
    "SelfObservationEvent",
]
