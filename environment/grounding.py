"""Environmental grounding — background streams so the system exists between turns."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from config.settings import config

from core.events import Event, SessionEndEvent, StreamUpdateEvent, TimePassingEvent
from environment.streams import (
    BaseStream,
    TimeStream,
    RSSStream,
    OutputHistoryStream,
    SystemHealthStream,
)
from environment.time_sense import TimeSense

log = logging.getLogger(__name__)


class EnvironmentalGrounding:
    """Poll streams, route salient events into state and memory."""

    def __init__(
        self,
        state: Any = None,
        memory_system: Any = None,
        salience_threshold: float = 0.2,
    ):
        self.state = state
        self.memory = memory_system
        self.salience_threshold = salience_threshold

        self.time_sense = TimeSense()
        self.time_stream = TimeStream(
            session_timeout_minutes=config.session_timeout_minutes,
        )
        self.health_stream = SystemHealthStream()
        self.on_session_end_cb: Callable[[int, int], None] | None = None

        # All registered streams
        self.streams: list[BaseStream] = [
            self.time_stream,
            self.health_stream,
        ]

        # Loop state
        self._running = False
        self._task: asyncio.Task | None = None
        self._events_processed: int = 0
        self._events_stored: int = 0

    # ── Stream management ─────────────────────────────────────────────────

    def add_stream(self, stream: BaseStream) -> None:
        """Register a new environmental stream."""
        self.streams.append(stream)
        log.info("Added stream: %s (interval=%ds)", stream.name, stream.poll_interval)

    def add_rss_feed(self, url: str, name: str = "", tags: list[str] | None = None) -> None:
        """Convenience: add an RSS feed. Creates an RSSStream if none exists."""
        rss = self._get_rss_stream()
        rss.add_feed(url, name, tags)

    def add_output_history(self, memory_system: Any) -> None:
        """Add the output history stream (needs memory system reference)."""
        stream = OutputHistoryStream(memory_system=memory_system)
        self.add_stream(stream)

    def _get_rss_stream(self) -> RSSStream:
        for s in self.streams:
            if isinstance(s, RSSStream):
                return s
        rss = RSSStream()
        self.add_stream(rss)
        return rss

    # ── Session hooks (called by the bot) ─────────────────────────────────

    def on_user_message(self, user_id: int) -> None:
        """Called when the user sends a message. Updates time tracking."""
        self.time_stream.record_user_activity()
        self.time_sense.record_message()

    def on_session_start(self, user_id: int) -> None:
        """Called when a new session begins (first message after silence)."""
        self.time_sense.start_session()
        # Generate and process a session start event
        event = self.time_stream.create_session_start_event()
        if self.state:
            self.state.update(event)

    def on_session_end(self, user_id: int, message_count: int = 0) -> None:
        """Called when a session ends (user goes quiet)."""
        self.time_sense.end_session(message_count)

    # ── Salience assessment ───────────────────────────────────────────────

    def assess_salience(self, event: Event) -> float:
        """
        Determine how noteworthy an event is given the current context.
        Uses the event's own salience as a base, then adjusts based on
        internal state — high curiosity makes everything more salient,
        high fatigue dampens things.
        """
        # Start with the event's intrinsic salience
        if isinstance(event, StreamUpdateEvent):
            base = event.salience
        elif isinstance(event, TimePassingEvent):
            # Time passing has very low base salience — it's background
            base = 0.05
        else:
            base = 0.3

        # State-based modifiers
        if self.state:
            curiosity_boost = (self.state.curiosity - 0.5) * 0.2
            fatigue_dampen = self.state.fatigue * -0.15
            base += curiosity_boost + fatigue_dampen

        return max(0.0, min(1.0, base))

    # ── Event processing ──────────────────────────────────────────────────

    async def _process_event(self, event: Event, user_id: int) -> None:
        """Route a single event through state and memory."""
        salience = self.assess_salience(event)
        self._events_processed += 1

        if self.state:
            self.state.update(event)

        if isinstance(event, SessionEndEvent):
            self.time_sense.end_session(event.messages_exchanged)
            if self.on_session_end_cb:
                self.on_session_end_cb(user_id, event.messages_exchanged)

        # Store only above threshold so idle time-passing does not flood memory.
        if salience >= self.salience_threshold and self.memory:
            content = self._event_to_content(event)
            if content:
                await self.memory.store_episode(
                    user_id=user_id,
                    content=content,
                    internal_state=self.state,
                    salience=salience,
                    type="event",
                    tags=["environment", event.type],
                )
                self._events_stored += 1

    def _event_to_content(self, event: Event) -> str | None:
        """Convert an event to a storable content string."""
        if isinstance(event, StreamUpdateEvent):
            return f"[{event.stream_name}] {event.content}"
        elif isinstance(event, TimePassingEvent):
            # Don't store routine time passing
            return None
        else:
            return f"[{event.type}] {event.payload}" if event.payload else None

    # ── Main loop ─────────────────────────────────────────────────────────

    async def run_loop(
        self,
        user_id: int,
        base_interval_seconds: float = 30,
    ) -> None:
        """
        The continuous grounding loop. Polls all streams at their
        individual intervals and processes events.

        Args:
            user_id: which user context to operate in
            base_interval_seconds: how often the loop wakes up to check
                                   (individual streams have their own schedules)
        """
        self._running = True
        log.info(
            "Grounding loop started (%d streams, base_interval=%ds)",
            len(self.streams), base_interval_seconds,
        )

        while self._running:
            for stream in self.streams:
                if not stream.enabled or not stream.is_due:
                    continue

                try:
                    events = await stream.poll()
                    for event in events:
                        await self._process_event(event, user_id)
                except Exception as e:
                    log.debug("Stream %s poll error: %s", stream.name, e)

            await asyncio.sleep(base_interval_seconds)

    def start(self, user_id: int) -> None:
        """Start the grounding loop as a background task."""
        if self._task is not None and not self._task.done():
            return  # already running

        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self.run_loop(user_id))
        log.info("Grounding loop task started")

    def stop(self) -> None:
        """Stop the grounding loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        log.info(
            "Grounding loop stopped (processed=%d, stored=%d)",
            self._events_processed, self._events_stored,
        )

    # ── Prompt context ────────────────────────────────────────────────────

    def to_prompt_context(self) -> str:
        """
        Environmental awareness block for the system prompt.
        Combines time sense, system health, and any recent notable events.
        """
        parts = ["## Environmental awareness"]

        # Time sense
        time_ctx = self.time_sense.to_prompt_context()
        if time_ctx:
            parts.append(time_ctx)

        # System health
        parts.append(self.health_stream.to_context())

        # Time stream context
        time_stream_ctx = self.time_stream.to_context()
        if time_stream_ctx:
            parts.append(time_stream_ctx)

        parts.append(
            "\nYou exist in time. Between conversations, time passes, "
            "things happen in the world, and your own processes continue. "
            "Let this temporal grounding inform your sense of continuity."
        )

        return "\n".join(parts)

    # ── Stats ─────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "streams": len(self.streams),
            "events_processed": self._events_processed,
            "events_stored": self._events_stored,
            "running": self._running,
        }

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Persist time sense and stream state."""
        import json
        from pathlib import Path as P

        path = P(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "time_sense": self.time_sense.snapshot(),
            "stats": self.stats,
        }
        path.write_text(json.dumps(data, indent=2, default=str))

    def load(self, path: Path) -> None:
        """Restore time sense and stream state."""
        import json
        from pathlib import Path as P

        path = P(path)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                if "time_sense" in data:
                    self.time_sense.restore(data["time_sense"])
            except (json.JSONDecodeError, Exception) as e:
                log.warning("Failed to load grounding state: %s", e)



