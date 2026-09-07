"""Working memory — salience-limited conversation window. Persisted so restarts keep the thread."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from core.timeutil import utcnow, parse_iso
from pathlib import Path
from typing import Literal

from config.settings import config


MessageRole = Literal["user", "assistant", "system"]


@dataclass(order=False)
class WorkingItem:
    """A single item held in working memory."""

    role: MessageRole
    content: str
    salience: float = 0.5              # 0–1, drives eviction priority
    timestamp: datetime = field(default_factory=utcnow)
    metadata: dict = field(default_factory=dict)

    def __lt__(self, other: WorkingItem) -> bool:
        return self.salience < other.salience


class WorkingMemory:
    """
    Per-user active context window.

    Fixed capacity — when full, the lowest-salience item is evicted.
    User messages default to higher salience than assistant messages
    because the human's words usually matter more for context.
    """

    DEFAULT_SALIENCE = {
        "user": 0.7,
        "assistant": 0.5,
        "system": 0.9,
    }

    def __init__(self, capacity: int = 20):
        self._capacity = capacity
        self._buffers: dict[int, list[WorkingItem]] = {}

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
    ) -> WorkingItem | None:
        """
        Add an item. Returns the evicted item if capacity was exceeded,
        or None if there was room.
        """
        if salience is None:
            salience = self.DEFAULT_SALIENCE.get(role, 0.5)

        item = WorkingItem(
            role=role,
            content=content,
            salience=salience,
            metadata=metadata or {},
        )

        buf = self._buf(user_id)
        evicted = None

        if len(buf) >= self._capacity:
            # Evict the least-salient item
            min_idx = min(range(len(buf)), key=lambda i: buf[i].salience)
            evicted = buf.pop(min_idx)

        buf.append(item)
        return evicted

    def boost(self, user_id: int, index: int, amount: float = 0.1) -> None:
        """Boost the salience of an item (e.g. when it's referenced again)."""
        buf = self._buf(user_id)
        if 0 <= index < len(buf):
            buf[index].salience = min(1.0, buf[index].salience + amount)

    # ── Read ──────────────────────────────────────────────────────────────

    def get_items(self, user_id: int) -> list[WorkingItem]:
        """Return all items in chronological order."""
        return sorted(self._buf(user_id), key=lambda i: i.timestamp)

    def get_recent(self, user_id: int, n: int = 5) -> list[WorkingItem]:
        """Return the N most recent items."""
        items = self.get_items(user_id)
        return items[-n:]

    def get_salient(self, user_id: int, n: int = 5) -> list[WorkingItem]:
        """Return the N most salient items (regardless of recency)."""
        buf = self._buf(user_id)
        return sorted(buf, key=lambda i: i.salience, reverse=True)[:n]

    @property
    def capacity(self) -> int:
        return self._capacity

    def size(self, user_id: int) -> int:
        return len(self._buf(user_id))

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def clear(self, user_id: int) -> None:
        self._buffers.pop(user_id, None)

    def decay(self, user_id: int, rate: float = 0.02) -> None:
        """
        Slightly reduce salience of all items over time.
        Called periodically so older context naturally fades
        unless it's been boosted by relevance.
        """
        for item in self._buf(user_id):
            item.salience = max(0.05, item.salience - rate)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """
        Persist working memory to disk alongside other state saves.
        This is what prevents conversation loss on restart.
        """
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
                    "metadata": item.metadata,
                }
                for item in buf
            ]

        from core.atomic import atomic_write_text
        atomic_write_text(path, json.dumps(data, indent=2))

    def load(self, path: Path) -> None:
        """
        Restore working memory from disk on startup.
        If the file is missing or corrupt, start fresh — never crash.
        """
        path = Path(path)
        if not path.exists():
            return

        try:
            data = json.loads(path.read_text())
            for user_id_str, items in data.items():
                user_id = int(user_id_str)
                for item_data in items:
                    wi = WorkingItem(
                        role=item_data["role"],
                        content=item_data["content"],
                        salience=item_data.get("salience", 0.5),
                        timestamp=parse_iso(item_data["timestamp"]),
                        metadata=item_data.get("metadata", {}),
                    )
                    self._buf(user_id).append(wi)
                buf = self._buf(user_id)
                while len(buf) > self._capacity:
                    min_idx = min(range(len(buf)), key=lambda i: buf[i].salience)
                    buf.pop(min_idx)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass  # corrupted — start fresh

    def has_recent_activity(self, user_id: int, minutes: float = 15) -> bool:
        """
        Check if there's been recent activity for this user.
        Used by background processes to avoid disrupting active conversations.
        """
        buf = self._buf(user_id)
        if not buf:
            return False
        latest = max(item.timestamp for item in buf)
        elapsed = (utcnow() - latest).total_seconds() / 60
        return elapsed < minutes

    # ── Export ────────────────────────────────────────────────────────────

    def to_langchain_messages(self, user_id: int) -> list[dict]:
        """Return items as LangChain-compatible message dicts, chronological."""
        return [
            {"role": item.role, "content": item.content}
            for item in self.get_items(user_id)
        ]

    def to_prompt_context(self, user_id: int) -> str:
        """Compact summary for injection into the system prompt."""
        items = self.get_items(user_id)
        if not items:
            return ""
        lines = []
        for item in items:
            prefix = "You" if item.role == "assistant" else item.role.capitalize()
            lines.append(f"[{prefix}] {item.content[:200]}")
        return "\n".join(lines)


# Singleton
working_memory = WorkingMemory(capacity=config.working_memory_capacity)
