"""
Environment — grounding the system in something beyond conversations.

Exports:
  - EnvironmentalGrounding: the main polling loop and orchestrator
  - TimeSense: subjective temporal awareness
  - Streams: BaseStream, TimeStream, RSSStream, OutputHistoryStream, SystemHealthStream
"""
from environment.grounding import EnvironmentalGrounding
from environment.time_sense import TimeSense
from environment.streams import (
    BaseStream,
    TimeStream,
    RSSStream,
    OutputHistoryStream,
    SystemHealthStream,
)

__all__ = [
    "EnvironmentalGrounding",
    "TimeSense",
    "BaseStream",
    "TimeStream",
    "RSSStream",
    "OutputHistoryStream",
    "SystemHealthStream",
]
