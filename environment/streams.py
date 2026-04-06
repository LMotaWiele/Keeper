"""
Environmental streams — sources of grounding beyond conversation.

Each stream polls an external or internal source and yields events
when something noteworthy happens. The grounding loop consumes these
events and routes them through state + memory.

Streams included:
  - TimeStream:          clock time, session gaps, time-of-day awareness
  - RSSStream:           external information feeds
  - OutputHistoryStream: the system's own recent behavior
  - SystemHealthStream:  computational self-awareness (disk, memory, uptime)

Adding a new stream: subclass BaseStream, implement poll(), done.

PATCHED:
  - TimeStream session_timeout_minutes increased from 15 to 60 minutes.
    15 minutes was far too aggressive — natural conversation pauses
    (thinking, doing something else, getting coffee) were triggering
    false session ends that wiped conversational context.
  - SessionEndEvent is now only fired once per actual session end
    (added _end_event_fired guard to prevent repeated firing)
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from core.events import (
    Event,
    TimePassingEvent,
    StreamUpdateEvent,
    SessionStartEvent,
    SessionEndEvent,
)

log = logging.getLogger(__name__)


# ── Base class ────────────────────────────────────────────────────────────

class BaseStream(ABC):
    """
    A source of environmental events. Subclass and implement poll().
    """

    def __init__(self, name: str, poll_interval_seconds: float = 60):
        self.name = name
        self.poll_interval = poll_interval_seconds
        self._last_poll: datetime | None = None
        self.enabled = True

    @abstractmethod
    async def poll(self) -> list[Event]:
        """
        Check for new events. Returns an empty list if nothing happened.
        Called by the grounding loop at the configured interval.
        """
        ...

    @property
    def is_due(self) -> bool:
        if self._last_poll is None:
            return True
        elapsed = (datetime.utcnow() - self._last_poll).total_seconds()
        return elapsed >= self.poll_interval

    def mark_polled(self) -> None:
        self._last_poll = datetime.utcnow()


# ── Time stream ───────────────────────────────────────────────────────────

class TimeStream(BaseStream):
    """
    The most basic grounding source — awareness of time passing.

    Emits:
      - TimePassingEvent on every poll (the system always feels time)
      - SessionStartEvent when the user returns after absence
      - SessionEndEvent when silence exceeds a threshold

    Also provides contextual time awareness: time of day, day of week,
    how long since the last interaction.

    PATCHED: session_timeout_minutes default changed from 15 to 60.
    Added _end_event_fired guard so SessionEndEvent only fires once
    per session end, not on every poll after timeout.
    """

    def __init__(
        self,
        poll_interval_seconds: float = 60,
        session_timeout_minutes: float = 60,  # PATCHED: was 15, now 60
    ):
        super().__init__("time", poll_interval_seconds)
        self._session_timeout = session_timeout_minutes
        self._last_user_activity: datetime | None = None
        self._session_active = False
        self._session_start: datetime | None = None
        self._session_message_count = 0
        self._end_event_fired = False  # PATCH: prevent repeated SessionEndEvents

    def record_user_activity(self) -> None:
        """Called by the bot when the user sends a message."""
        now = datetime.utcnow()

        if not self._session_active:
            # Session is starting
            self._session_active = True
            self._session_start = now
            self._session_message_count = 0
            self._end_event_fired = False  # PATCH: reset guard

        self._last_user_activity = now
        self._session_message_count += 1

    async def poll(self) -> list[Event]:
        events: list[Event] = []
        now = datetime.utcnow()

        # Time passing is always an event
        minutes_since_poll = 1.0
        if self._last_poll:
            minutes_since_poll = (now - self._last_poll).total_seconds() / 60

        events.append(TimePassingEvent(minutes_elapsed=minutes_since_poll))

        # Check for session transitions
        if self._session_active and self._last_user_activity:
            silence = (now - self._last_user_activity).total_seconds() / 60
            if silence > self._session_timeout and not self._end_event_fired:
                # Session ended — fire event ONCE
                duration = 0.0
                if self._session_start:
                    duration = (self._last_user_activity - self._session_start).total_seconds() / 60

                events.append(SessionEndEvent(
                    messages_exchanged=self._session_message_count,
                    duration_minutes=duration,
                ))
                self._session_active = False
                self._end_event_fired = True  # PATCH: don't fire again

                log.info(
                    "Session ended (timeout after %.0fm silence, "
                    "session was %.0fm with %d messages)",
                    silence, duration, self._session_message_count,
                )

        self.mark_polled()
        return events

    def create_session_start_event(self) -> SessionStartEvent:
        """Generate a session start event (called when user first messages)."""
        gap_hours = None
        if self._last_user_activity:
            gap_hours = (datetime.utcnow() - self._last_user_activity).total_seconds() / 3600

        return SessionStartEvent(
            hours_since_last=gap_hours or 0.0,
            time_of_day=self._time_of_day(),
        )

    def _time_of_day(self) -> str:
        """Human-readable time of day."""
        hour = datetime.utcnow().hour  # TODO: adjust for user timezone
        if hour < 6:
            return "late_night"
        elif hour < 12:
            return "morning"
        elif hour < 17:
            return "afternoon"
        elif hour < 21:
            return "evening"
        else:
            return "night"

    def to_context(self) -> str:
        """Time awareness context for the prompt."""
        parts = []
        now = datetime.utcnow()
        parts.append(f"Current time: {now.strftime('%A %H:%M UTC')}")

        if self._session_active and self._session_start:
            duration = (now - self._session_start).total_seconds() / 60
            parts.append(f"Session active for {duration:.0f} minutes")
            parts.append(f"Messages this session: {self._session_message_count}")

        if self._last_user_activity:
            silence = (now - self._last_user_activity).total_seconds() / 60
            if silence > 2:
                parts.append(f"Silence: {silence:.0f} minutes")

        return "\n".join(parts) if parts else ""


# ── RSS stream ────────────────────────────────────────────────────────────

class RSSStream(BaseStream):
    """
    External information feeds. Emits StreamUpdateEvents
    when something appears that might be salient.

    Requires: feedparser (pip install feedparser)
    Falls back gracefully if not installed.
    """

    def __init__(
        self,
        feeds: list[dict] | None = None,
        poll_interval_seconds: float = 900,  # 15 min default
    ):
        super().__init__("rss", poll_interval_seconds)
        # feeds: [{"url": "...", "name": "...", "tags": [...]}]
        self.feeds = feeds or []
        self._seen_ids: set[str] = set()
        self._feedparser = None

    def _get_feedparser(self):
        if self._feedparser is None:
            try:
                import feedparser
                self._feedparser = feedparser
            except ImportError:
                log.warning("feedparser not installed — RSS stream disabled")
                self.enabled = False
                return None
        return self._feedparser

    def add_feed(self, url: str, name: str = "", tags: list[str] | None = None) -> None:
        self.feeds.append({"url": url, "name": name or url, "tags": tags or []})

    async def poll(self) -> list[Event]:
        fp = self._get_feedparser()
        if not fp or not self.feeds:
            self.mark_polled()
            return []

        events: list[Event] = []

        for feed_config in self.feeds:
            try:
                # feedparser is sync — run in executor to not block
                loop = asyncio.get_event_loop()
                parsed = await loop.run_in_executor(
                    None, fp.parse, feed_config["url"]
                )

                for entry in parsed.entries[:10]:
                    entry_id = entry.get("id") or entry.get("link") or entry.get("title", "")
                    if entry_id in self._seen_ids:
                        continue

                    self._seen_ids.add(entry_id)
                    title = entry.get("title", "untitled")
                    summary = entry.get("summary", "")[:200]

                    events.append(StreamUpdateEvent(
                        stream_name=feed_config["name"],
                        content=f"{title}: {summary}",
                        salience=0.3,  # base salience — grounding loop may adjust
                    ))

            except Exception as e:
                log.debug("RSS poll failed for %s: %s", feed_config.get("name"), e)

        # Cap seen IDs to prevent unbounded growth
        if len(self._seen_ids) > 5000:
            self._seen_ids = set(list(self._seen_ids)[-2000:])

        self.mark_polled()
        return events


# ── Output history stream ─────────────────────────────────────────────────

class OutputHistoryStream(BaseStream):
    """
    Reviews the system's own recent outputs — a relationship with
    its own history. Surfaces patterns, unresolved threads, and
    things worth revisiting.

    Doesn't generate events on every poll — only when it detects
    something worth noting (e.g. a topic came up repeatedly,
    a question was left unanswered, a promise was made).
    """

    def __init__(
        self,
        memory_system: Any = None,
        poll_interval_seconds: float = 600,  # 10 min
    ):
        super().__init__("output_history", poll_interval_seconds)
        self.memory = memory_system
        self._last_review_count: int = 0

    async def poll(self) -> list[Event]:
        if self.memory is None:
            self.mark_polled()
            return []

        events: list[Event] = []

        try:
            # Check working memory for unresolved threads
            # This is a lightweight scan — the heavy analysis
            # happens in the consolidator and self-model
            items = self.memory.working.get_items(user_id=0)  # system-level
            current_count = len(items)

            if current_count > self._last_review_count + 5:
                # Significant new activity — note it
                events.append(StreamUpdateEvent(
                    stream_name="output_history",
                    content=f"Activity spike: {current_count - self._last_review_count} new working memory items since last check",
                    salience=0.2,
                ))
                self._last_review_count = current_count

        except Exception as e:
            log.debug("Output history poll failed: %s", e)

        self.mark_polled()
        return events


# ── System health stream ──────────────────────────────────────────────────

class SystemHealthStream(BaseStream):
    """
    Computational self-awareness — monitors the system's own resources.
    Emits events when resource usage is noteworthy (high disk, low memory,
    long uptime).

    This gives the system a grounded sense of its own physical substrate.
    """

    def __init__(self, poll_interval_seconds: float = 300):  # 5 min
        super().__init__("system_health", poll_interval_seconds)
        self._start_time = datetime.utcnow()
        self._last_disk_warning: datetime | None = None

    async def poll(self) -> list[Event]:
        events: list[Event] = []

        try:
            import shutil
            import pathlib

            # Disk usage
            total, used, free = shutil.disk_usage(pathlib.Path.home())
            pct_used = used / total * 100

            if pct_used > 90 and self._should_warn_disk():
                events.append(StreamUpdateEvent(
                    stream_name="system_health",
                    content=f"Disk usage high: {pct_used:.0f}% ({free // (1024**3)}GB free)",
                    salience=0.6,
                ))
                self._last_disk_warning = datetime.utcnow()

            # Uptime awareness
            uptime_hours = (datetime.utcnow() - self._start_time).total_seconds() / 3600
            if uptime_hours > 0 and uptime_hours % 24 < (self.poll_interval / 3600):
                # Just crossed a 24-hour boundary
                events.append(StreamUpdateEvent(
                    stream_name="system_health",
                    content=f"Uptime milestone: running for {uptime_hours:.0f} hours",
                    salience=0.15,
                ))

        except Exception as e:
            log.debug("System health poll failed: %s", e)

        self.mark_polled()
        return events

    def _should_warn_disk(self) -> bool:
        if self._last_disk_warning is None:
            return True
        elapsed = (datetime.utcnow() - self._last_disk_warning).total_seconds() / 3600
        return elapsed > 6  # warn at most every 6 hours

    @property
    def uptime_hours(self) -> float:
        return (datetime.utcnow() - self._start_time).total_seconds() / 3600

    def to_context(self) -> str:
        """System awareness context for the prompt."""
        return f"Uptime: {self.uptime_hours:.1f}h"
