"""
Goals — intrinsic motivation and autonomous behavior.

Exports:
  - GoalSystem: terminal + instrumental goal management
  - AutonomousEngine: background loop pursuing goals between user inputs
  - Goal: individual goal dataclass
  - GoalStatus: active/completed/abandoned/paused
"""
from goals.system import GoalSystem, Goal, GoalStatus
from goals.autonomous import AutonomousEngine

__all__ = [
    "GoalSystem",
    "AutonomousEngine",
    "Goal",
    "GoalStatus",
]
