"""
Short-term memory — the current conversation window.

Holds the last N messages in RAM. Cleared when the bot restarts.
This is what gives the companion context within a single conversation.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


MessageRole = Literal["user", "assistant", "system"]


@dataclass
class Message:
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)


class ShortTermMemory:
    """Sliding window of recent messages, per user."""

    def __init__(self, max_messages: int = 20):
        self._max = max_messages
        self._windows: dict[int, deque[Message]] = {}

    def _window(self, user_id: int) -> deque[Message]:
        if user_id not in self._windows:
            self._windows[user_id] = deque(maxlen=self._max)
        return self._windows[user_id]

    def add(self, user_id: int, role: MessageRole, content: str, metadata: dict | None = None) -> None:
        self._window(user_id).append(
            Message(role=role, content=content, metadata=metadata or {})
        )

    def get_messages(self, user_id: int) -> list[Message]:
        return list(self._window(user_id))

    def clear(self, user_id: int) -> None:
        self._windows.pop(user_id, None)

    def to_langchain_messages(self, user_id: int) -> list[dict]:
        """Return messages in LangChain dict format."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self._window(user_id)
        ]


# Singleton
short_term = ShortTermMemory()
