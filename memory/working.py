"""Working memory — dialogue ring plus dated pins. Persisted so restarts keep the thread."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from config.settings import config
from core.atomic import atomic_write_text
from core.timeutil import format_local_stamp, parse_iso, utcnow
from memory.timebind import token_overlap


MessageRole = Literal["user", "assistant", "system"]


@dataclass(order=False)
class WorkingItem:
    """A single item held in working memory."""

    role: MessageRole
    content: str
    salience: float = 0.5
    timestamp: datetime = field(default_factory=utcnow)
    last_used_at: datetime = field(default_factory=utcnow)
    metadata: dict = field(default_factory=dict)

    def __lt__(self, other: WorkingItem) -> bool:
        return self.salience < other.salience


class WorkingMemory:
    """
    Per-user context window: a chronological dialogue ring that is never
    salience-evicted, plus a small pin ring of older high-salience user turns.
    """

    DEFAULT_SALIENCE = {
        "user": 0.7,
        "assistant": 0.5,
        "system": 0.9,
    }

    def __init__(
        self,
        capacity: int | None = None,
        dialogue_window: int | None = None,
        pin_capacity: int | None = None,
    ):
        self._capacity = capacity
        self._dialogue_window = dialogue_window
        self._pin_capacity = pin_capacity
        self._buffers: dict[int, list[WorkingItem]] = {}

    @property
    def dialogue_window(self) -> int:
        if self._dialogue_window is not None:
            return self._dialogue_window
        return int(getattr(config, "dialogue_window", 12))

    @property
    def pin_capacity(self) -> int:
        if self._pin_capacity is not None:
            return self._pin_capacity
        return int(getattr(config, "pin_capacity", 8))

    @property
    def pin_max_age_days(self) -> int:
        return int(getattr(config, "pin_max_age_days", 14))

    def _buf(self, user_id: int) -> list[WorkingItem]:
        if user_id not in self._buffers:
            self._buffers[user_id] = []
        return self._buffers[user_id]

    # ── Write ─────────────────────────────────────────────────────────────

    def add(
        self,
        user_id: int,
        role: MessageRole,
        content: str,
        salience: float | None = None,
        metadata: dict | None = None,
        timestamp: datetime | None = None,
    ) -> WorkingItem | None:
        """
        Add an item, then rebalance into dialogue + pins.
        Returns one evicted item if any were dropped, else None.
        """
        if salience is None:
            salience = self.DEFAULT_SALIENCE.get(role, 0.5)
        ts = timestamp or utcnow()
        item = WorkingItem(
            role=role,
            content=content,
            salience=salience,
            timestamp=ts,
            last_used_at=ts,
            metadata=metadata or {},
        )
        self._buf(user_id).append(item)
        evicted = self._rebalance(user_id)
        return evicted[0] if evicted else None

    def boost(self, user_id: int, index: int, amount: float = 0.1) -> None:
        """Boost the salience of an item (e.g. when it's referenced again)."""
        buf = self._buf(user_id)
        if 0 <= index < len(buf):
            buf[index].salience = min(1.0, buf[index].salience + amount)

    def split_window(self, user_id: int) -> tuple[list[WorkingItem], list[WorkingItem]]:
        """Return (dialogue, pins). Does not mutate the buffer."""
        items = sorted(self._buf(user_id), key=lambda i: i.timestamp)
        dw = max(1, self.dialogue_window)
        if len(items) <= dw:
            return items, []
        dialogue = items[-dw:]
        older = items[:-dw]
        pins: list[WorkingItem] = []
        for item in sorted(older, key=lambda i: i.salience, reverse=True):
            if item.role != "user":
                continue
            if any(token_overlap(item.content, d.content) for d in dialogue):
                continue
            pins.append(item)
            if len(pins) >= self.pin_capacity:
                break
        pins.sort(key=lambda i: i.timestamp)
        return dialogue, pins

    def _rebalance(self, user_id: int) -> list[WorkingItem]:
        buf = self._buf(user_id)
        dialogue, pins = self.split_window(user_id)
        keep = {id(x) for x in dialogue} | {id(x) for x in pins}
        kept = [x for x in sorted(buf, key=lambda i: i.timestamp) if id(x) in keep]
        evicted = [x for x in buf if id(x) not in keep]
        self._buffers[user_id] = kept
        return evicted

    def prune_stale_pins(self, user_id: int, now: datetime | None = None) -> list[WorkingItem]:
        """Drop pins older than pin_max_age_days that were not used recently."""
        now = now or utcnow()
        dialogue, pins = self.split_window(user_id)
        max_age = timedelta(days=self.pin_max_age_days)
        kept_pins = []
        dropped = []
        for item in pins:
            aged = (now - item.timestamp) > max_age
            unused = (now - item.last_used_at) > max_age
            if aged and unused:
                dropped.append(item)
            else:
                kept_pins.append(item)
        keep = {id(x) for x in dialogue} | {id(x) for x in kept_pins}
        self._buffers[user_id] = [
            x for x in sorted(self._buf(user_id), key=lambda i: i.timestamp) if id(x) in keep
        ]
        return dropped

    def on_session_end(self, user_id: int, now: datetime | None = None) -> None:
        self.decay(user_id)
        self.prune_stale_pins(user_id, now=now)
        self._rebalance(user_id)

    def idle_tick(self, user_id: int, now: datetime | None = None) -> None:
        self.on_session_end(user_id, now=now)

    # ── Read ──────────────────────────────────────────────────────────────

    def get_items(self, user_id: int) -> list[WorkingItem]:
        """Return dialogue + pins in chronological order."""
        dialogue, pins = self.split_window(user_id)
        return sorted(pins + dialogue, key=lambda i: i.timestamp)

    def get_recent(self, user_id: int, n: int = 5) -> list[WorkingItem]:
        items = self.get_items(user_id)
        return items[-n:]

    def get_salient(self, user_id: int, n: int = 5) -> list[WorkingItem]:
        buf = self._buf(user_id)
        return sorted(buf, key=lambda i: i.salience, reverse=True)[:n]

    @property
    def capacity(self) -> int:
        if self._capacity is not None:
            return self._capacity
        return self.dialogue_window + self.pin_capacity

    def size(self, user_id: int) -> int:
        return len(self._buf(user_id))

    def span_hours(self, user_id: int) -> float:
        """Hours between oldest and newest item in the dialogue ring."""
        dialogue, _ = self.split_window(user_id)
        if len(dialogue) < 2:
            return 0.0
        delta = dialogue[-1].timestamp - dialogue[0].timestamp
        return max(0.0, delta.total_seconds() / 3600)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def clear(self, user_id: int) -> None:
        self._buffers.pop(user_id, None)

    def decay(self, user_id: int, rate: float = 0.02) -> None:
        """Slightly reduce salience of all items over time."""
        for item in self._buf(user_id):
            item.salience = max(0.05, item.salience - rate)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {}
        for user_id, buf in self._buffers.items():
            data[str(user_id)] = [
                {
                    "role": item.role,
                    "content": item.content,
                    "salience": item.salience,
                    "timestamp": item.timestamp.isoformat(),
                    "last_used_at": item.last_used_at.isoformat(),
                    "metadata": item.metadata,
                }
                for item in buf
            ]
        atomic_write_text(path, json.dumps(data, indent=2))

    def load(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            return

        try:
            data = json.loads(path.read_text())
            for user_id_str, items in data.items():
                user_id = int(user_id_str)
                for item_data in items:
                    ts = parse_iso(item_data["timestamp"])
                    used_raw = item_data.get("last_used_at")
                    wi = WorkingItem(
                        role=item_data["role"],
                        content=item_data["content"],
                        salience=item_data.get("salience", 0.5),
                        timestamp=ts,
                        last_used_at=parse_iso(used_raw) if used_raw else ts,
                        metadata=item_data.get("metadata", {}),
                    )
                    self._buf(user_id).append(wi)
                self._rebalance(user_id)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass  # corrupted — start fresh

    def has_recent_activity(self, user_id: int, minutes: float = 15) -> bool:
        buf = self._buf(user_id)
        if not buf:
            return False
        latest = max(item.timestamp for item in buf)
        elapsed = (utcnow() - latest).total_seconds() / 60
        return elapsed < minutes

    # ── Export ────────────────────────────────────────────────────────────

    def to_langchain_messages(
        self,
        user_id: int,
        now: datetime | None = None,
    ) -> list[dict]:
        """Dialogue + pins as LangChain dicts, each prefixed with user-local time."""
        now = now or utcnow()
        dialogue, pins = self.split_window(user_id)
        out: list[dict] = []
        for item in pins:
            stamp = format_local_stamp(item.timestamp, date_only=True)
            prefix = f"[earlier · {stamp}] "
            item.last_used_at = now
            out.append({"role": item.role, "content": prefix + item.content})
        for item in dialogue:
            stamp = format_local_stamp(item.timestamp)
            prefix = f"[{stamp}] "
            item.last_used_at = now
            out.append({"role": item.role, "content": prefix + item.content})
        return out

    def to_prompt_context(self, user_id: int, now: datetime | None = None) -> str:
        """Compact summary for injection into the system prompt."""
        items = self.to_langchain_messages(user_id, now=now)
        if not items:
            return ""
        lines = []
        for item in items:
            prefix = "You" if item["role"] == "assistant" else item["role"].capitalize()
            lines.append(f"[{prefix}] {item['content'][:200]}")
        return "\n".join(lines)


working_memory = WorkingMemory()
