"""
Mid-term memory — episodic SQLite store.

Persists conversation summaries, facts learned about the user, and notable
events across sessions. Lives for weeks/months. The companion reads relevant
episodes at the start of each conversation to feel "familiar."
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import aiosqlite

from config.settings import config

EpisodeType = Literal["summary", "fact", "event", "preference", "note"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    type        TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    tags        TEXT    DEFAULT '[]',
    importance  REAL    DEFAULT 0.5,
    created_at  TEXT    NOT NULL,
    expires_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id);
CREATE INDEX IF NOT EXISTS idx_episodes_type ON episodes(type);
"""


class MidTermMemory:
    def __init__(self, db_path: Path | None = None):
        self.db_path = str(db_path or config.midterm_db_path)

    async def init(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(SCHEMA)
            await db.commit()

    async def store(
        self,
        user_id: int,
        type: EpisodeType,
        content: str,
        tags: list[str] | None = None,
        importance: float = 0.5,
        expires_at: datetime | None = None,
    ) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """INSERT INTO episodes (user_id, type, content, tags, importance, created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_id,
                    type,
                    content,
                    json.dumps(tags or []),
                    importance,
                    datetime.utcnow().isoformat(),
                    expires_at.isoformat() if expires_at else None,
                ),
            )
            await db.commit()
            return cursor.lastrowid

    async def get_recent(
        self,
        user_id: int,
        limit: int = 10,
        type: EpisodeType | None = None,
    ) -> list[dict]:
        query = "SELECT * FROM episodes WHERE user_id = ?"
        params: list = [user_id]

        if type:
            query += " AND type = ?"
            params.append(type)

        # Filter out expired entries
        query += " AND (expires_at IS NULL OR expires_at > ?)"
        params.append(datetime.utcnow().isoformat())

        query += " ORDER BY importance DESC, created_at DESC LIMIT ?"
        params.append(limit)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def get_facts(self, user_id: int) -> list[dict]:
        return await self.get_recent(user_id, limit=20, type="fact")

    async def get_preferences(self, user_id: int) -> list[dict]:
        return await self.get_recent(user_id, limit=10, type="preference")

    async def delete(self, episode_id: int) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
            await db.commit()

    async def format_for_prompt(self, user_id: int) -> str:
        """Return a compact text block for injection into the system prompt."""
        episodes = await self.get_recent(user_id, limit=15)
        if not episodes:
            return ""

        lines = ["## What I remember about you"]
        for ep in episodes:
            tags = json.loads(ep["tags"])
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            lines.append(f"- [{ep['type']}]{tag_str} {ep['content']}")
        return "\n".join(lines)


# Singleton
mid_term = MidTermMemory()
