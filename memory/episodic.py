"""Episodic memory — SQLite experiences with decay. Forgetting is intentional; see docs/memory.md."""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from core.timeutil import utcnow, parse_iso
from pathlib import Path
from typing import Any, Literal

import aiosqlite

from config.settings import config

EpisodeType = Literal[
    "summary", "fact", "event", "preference", "note",
    "self_observation", "opinion", "research",
]


SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          INTEGER NOT NULL,
    type             TEXT    NOT NULL,
    content          TEXT    NOT NULL,
    tags             TEXT    DEFAULT '[]',

    -- salience & emotion
    importance        REAL    DEFAULT 0.5,
    emotional_weight  REAL    DEFAULT 0.5,
    state_snapshot    TEXT    DEFAULT '{}',

    -- decay & recall
    decay_rate        REAL    DEFAULT 0.01,
    recall_count      INTEGER DEFAULT 0,
    last_recalled_at  TEXT,
    effective_strength REAL   DEFAULT 1.0,

    -- associations
    associations      TEXT    DEFAULT '[]',

    -- timestamps
    created_at        TEXT    NOT NULL,
    expires_at        TEXT
);

CREATE INDEX IF NOT EXISTS idx_ep_user       ON episodes(user_id);
CREATE INDEX IF NOT EXISTS idx_ep_type       ON episodes(type);
CREATE INDEX IF NOT EXISTS idx_ep_strength   ON episodes(effective_strength);
CREATE INDEX IF NOT EXISTS idx_ep_importance ON episodes(importance DESC);
"""

MIGRATION_CHECK = """
SELECT COUNT(*) FROM pragma_table_info('episodes') WHERE name = 'emotional_weight';
"""

MIGRATION = """
ALTER TABLE episodes ADD COLUMN emotional_weight   REAL DEFAULT 0.5;
ALTER TABLE episodes ADD COLUMN state_snapshot      TEXT DEFAULT '{}';
ALTER TABLE episodes ADD COLUMN decay_rate          REAL DEFAULT 0.01;
ALTER TABLE episodes ADD COLUMN recall_count        INTEGER DEFAULT 0;
ALTER TABLE episodes ADD COLUMN last_recalled_at    TEXT;
ALTER TABLE episodes ADD COLUMN effective_strength  REAL DEFAULT 1.0;
ALTER TABLE episodes ADD COLUMN associations        TEXT DEFAULT '[]';
"""


def _now_iso() -> str:
    return utcnow().isoformat()


class EpisodicMemory:
    """
    Persistent episodic store backed by SQLite.

    Each episode decays over time unless reinforced by recall or
    high emotional weight. The effective_strength field is the
    single number used for retrieval ranking — it blends importance,
    emotional weight, recency, and recall frequency.
    """

    def __init__(self, db_path: Path | None = None):
        self.db_path = str(db_path or config.midterm_db_path)
        self._new_since_consolidation: dict[int, int] = {}  # user_id → count

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def init(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(SCHEMA)
            # Migrate old mid_term tables if needed
            async with db.execute(MIGRATION_CHECK) as cur:
                (has_col,) = await cur.fetchone()
            if not has_col:
                for stmt in MIGRATION.strip().split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        try:
                            await db.execute(stmt)
                        except Exception:
                            pass  # column may already exist
            await db.commit()

    # ── Store ─────────────────────────────────────────────────────────────

    async def store(
        self,
        user_id: int,
        type: EpisodeType,
        content: str,
        tags: list[str] | None = None,
        importance: float = 0.5,
        emotional_weight: float = 0.5,
        state_snapshot: dict | None = None,
        associations: list[int] | None = None,
        expires_at: datetime | None = None,
    ) -> int:
        """
        Store a new episode. Emotional weight influences decay rate —
        high-emotion memories decay 5× slower than neutral ones.
        """
        # High emotional weight → slow decay, low → fast decay
        decay_rate = 0.02 * (1.0 - emotional_weight * 0.8)

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """INSERT INTO episodes
                   (user_id, type, content, tags, importance,
                    emotional_weight, state_snapshot, decay_rate,
                    recall_count, effective_strength, associations,
                    created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 1.0, ?, ?, ?)""",
                (
                    user_id,
                    type,
                    content,
                    json.dumps(tags or []),
                    importance,
                    emotional_weight,
                    json.dumps(state_snapshot or {}),
                    decay_rate,
                    json.dumps(associations or []),
                    _now_iso(),
                    expires_at.isoformat() if expires_at else None,
                ),
            )

            uid = user_id
            self._new_since_consolidation[uid] = (
                self._new_since_consolidation.get(uid, 0) + 1
            )
            await db.commit()
            return cursor.lastrowid

    # ── Recall ────────────────────────────────────────────────────────────

    async def recall(
        self,
        user_id: int,
        limit: int = 10,
        type: EpisodeType | None = None,
        min_strength: float = 0.1,
        state_bias: dict | None = None,
    ) -> list[dict]:
        """
        Retrieve episodes ranked by effective_strength.

        Optionally biased by the current internal state — high-arousal
        states tend to surface high-emotion memories, for example.

        Updates recall_count and last_recalled_at on every retrieved episode
        (the act of remembering reinforces the memory).
        """
        query = """
            SELECT * FROM episodes
            WHERE user_id = ?
              AND effective_strength >= ?
              AND (expires_at IS NULL OR expires_at > ?)
              AND (tags IS NULL OR tags NOT LIKE '%"degraded"%')
        """
        params: list[Any] = [user_id, min_strength, _now_iso()]

        if type:
            query += " AND type = ?"
            params.append(type)

        query += " ORDER BY effective_strength DESC, importance DESC LIMIT ?"
        params.append(limit)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                episodes = [dict(row) for row in rows]

            # Reinforce recalled memories
            if episodes:
                ids = [ep["id"] for ep in episodes]
                placeholders = ",".join("?" for _ in ids)
                await db.execute(
                    f"""UPDATE episodes
                        SET recall_count = recall_count + 1,
                            last_recalled_at = ?
                        WHERE id IN ({placeholders})""",
                    [_now_iso()] + ids,
                )
                await db.commit()

        return episodes

    async def get_recent(
        self,
        user_id: int,
        limit: int = 10,
        type: EpisodeType | None = None,
    ) -> list[dict]:
        """Get most recent episodes (by creation time, ignoring strength)."""
        query = "SELECT * FROM episodes WHERE user_id = ?"
        params: list[Any] = [user_id]

        if type:
            query += " AND type = ?"
            params.append(type)

        query += " AND (expires_at IS NULL OR expires_at > ?)"
        params.append(_now_iso())

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def get_facts(self, user_id: int) -> list[dict]:
        return await self.recall(user_id, limit=20, type="fact")

    async def get_preferences(self, user_id: int) -> list[dict]:
        return await self.recall(user_id, limit=10, type="preference")

    async def get_for_consolidation(
        self,
        user_id: int,
        since_hours: int = 24,
        limit: int = 50,
    ) -> list[dict]:
        """Get recent high-strength episodes for the consolidator."""
        cutoff = (utcnow() - timedelta(hours=since_hours)).isoformat()
        query = """
            SELECT * FROM episodes
            WHERE user_id = ?
              AND created_at > ?
              AND effective_strength > 0.3
            ORDER BY importance DESC, emotional_weight DESC
            LIMIT ?
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, (user_id, cutoff, limit)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    # ── Decay ─────────────────────────────────────────────────────────────

    async def apply_decay(self, user_id: int | None = None) -> int:
        """
        Apply time-based decay to all episodes. Called periodically.

        Strength formula:
            base = importance × (1 + emotional_weight)
            recency_factor = exp(-decay_rate × hours_since_creation)
            recall_bonus = log(1 + recall_count) × 0.15
            effective_strength = base × recency_factor + recall_bonus

        Returns the number of episodes updated.
        """
        now = utcnow()

        where = "WHERE 1=1"
        params: list[Any] = []
        if user_id is not None:
            where += " AND user_id = ?"
            params.append(user_id)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(f"SELECT * FROM episodes {where}", params) as cursor:
                rows = await cursor.fetchall()

            updated = 0
            for row in rows:
                ep = dict(row)
                created = parse_iso(ep["created_at"])
                hours_elapsed = max((now - created).total_seconds() / 3600, 0.01)

                base = ep["importance"] * (1.0 + ep.get("emotional_weight", 0.5))
                recency = math.exp(-ep.get("decay_rate", 0.01) * hours_elapsed)
                recall_bonus = math.log(1 + ep.get("recall_count", 0)) * 0.15

                new_strength = round(base * recency + recall_bonus, 4)
                new_strength = max(0.0, min(2.0, new_strength))

                await db.execute(
                    "UPDATE episodes SET effective_strength = ? WHERE id = ?",
                    (new_strength, ep["id"]),
                )
                updated += 1

            await db.commit()
            return updated

    async def forget(self, user_id: int | None = None, threshold: float = 0.05) -> int:
        """
        Remove episodes whose strength has decayed below threshold.
        This is actual forgetting — they're gone.
        Returns count of forgotten episodes.
        """
        where = "WHERE effective_strength < ?"
        params: list[Any] = [threshold]
        if user_id is not None:
            where += " AND user_id = ?"
            params.append(user_id)

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                f"DELETE FROM episodes {where}", params
            )
            await db.commit()
            return cursor.rowcount

    # ── Associations ──────────────────────────────────────────────────────

    async def link(self, episode_id: int, related_id: int) -> None:
        """Create a bidirectional association between two episodes."""
        async with aiosqlite.connect(self.db_path) as db:
            for eid, rid in [(episode_id, related_id), (related_id, episode_id)]:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT associations FROM episodes WHERE id = ?", (eid,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if not row:
                        continue
                    assocs = json.loads(row["associations"])
                    if rid not in assocs:
                        assocs.append(rid)
                        await db.execute(
                            "UPDATE episodes SET associations = ? WHERE id = ?",
                            (json.dumps(assocs), eid),
                        )
            await db.commit()

    # ── Delete ────────────────────────────────────────────────────────────

    async def delete(self, episode_id: int) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
            await db.commit()

    # ── Prompt formatting ─────────────────────────────────────────────────

    async def format_for_prompt(self, user_id: int) -> str:
        """Return a compact block for injection into the system prompt."""
        episodes = await self.recall(user_id, limit=15)
        if not episodes:
            return ""

        lines = ["## What I remember about you"]
        for ep in episodes:
            tags = json.loads(ep["tags"] or "[]")
            if "degraded" in tags:
                continue
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            strength = ep.get("effective_strength", 1.0)
            # Dim fading memories with a visual cue
            fade = "" if strength > 0.5 else " (fading)"
            lines.append(f"- [{ep['type']}]{tag_str} {ep['content']}{fade}")
        return "\n".join(lines)

    # ── Stats ─────────────────────────────────────────────────────────────

    async def count(self, user_id: int) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT COUNT(*) FROM episodes WHERE user_id = ?", (user_id,)
            ) as cursor:
                (n,) = await cursor.fetchone()
                return n
    
    def new_episodes_since_consolidation(self, user_id: int) -> int:
            return self._new_since_consolidation.get(user_id, 0)
     
    def mark_consolidated(self, user_id: int):
            self._new_since_consolidation[user_id] = 0


# Singleton
episodic = EpisodicMemory()
