"""Episodic memory — SQLite experiences with decay. Forgetting is intentional; see docs/memory.md."""
from __future__ import annotations

import json
import logging
import math
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import aiosqlite

from config.settings import config
from core.timeutil import utcnow, parse_iso

log = logging.getLogger(__name__)

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
    expires_at        TEXT,

    -- evidence-based salience (KEEPER_PATCH_01)
    episode_id        TEXT,
    source            TEXT    DEFAULT 'user_turn',
    novelty           REAL    DEFAULT 1.0,
    used_count        INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_ep_user       ON episodes(user_id);
CREATE INDEX IF NOT EXISTS idx_ep_type       ON episodes(type);
CREATE INDEX IF NOT EXISTS idx_ep_strength   ON episodes(effective_strength);
CREATE INDEX IF NOT EXISTS idx_ep_importance ON episodes(importance DESC);

CREATE TABLE IF NOT EXISTS episode_refs (
    episode_id TEXT NOT NULL,
    kind       TEXT NOT NULL,
    ref_id     TEXT NOT NULL,
    PRIMARY KEY (episode_id, kind, ref_id)
);
CREATE INDEX IF NOT EXISTS idx_ep_refs_episode ON episode_refs(episode_id);

CREATE TABLE IF NOT EXISTS state_trace (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT    NOT NULL,
    event_type   TEXT    NOT NULL,
    event_detail TEXT,
    session_id   TEXT,
    arousal      REAL    NOT NULL,
    valence      REAL    NOT NULL,
    curiosity    REAL    NOT NULL,
    fatigue      REAL    NOT NULL,
    drives_json  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_state_trace_ts ON state_trace(ts);

CREATE TABLE IF NOT EXISTS action_bias_trial (
    trial_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    turn_id       TEXT NOT NULL,
    hypothesis_id TEXT NOT NULL,
    bias_text     TEXT NOT NULL,
    verdict       TEXT,
    reason        TEXT,
    evaluated_at  TEXT
);
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
ALTER TABLE episodes ADD COLUMN episode_id          TEXT;
ALTER TABLE episodes ADD COLUMN source              TEXT DEFAULT 'user_turn';
ALTER TABLE episodes ADD COLUMN novelty             REAL DEFAULT 1.0;
ALTER TABLE episodes ADD COLUMN used_count          INTEGER DEFAULT 0;
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
            for stmt in (
                "ALTER TABLE episodes ADD COLUMN episode_id TEXT",
                "ALTER TABLE episodes ADD COLUMN source TEXT DEFAULT 'user_turn'",
                "ALTER TABLE episodes ADD COLUMN novelty REAL DEFAULT 1.0",
                "ALTER TABLE episodes ADD COLUMN used_count INTEGER DEFAULT 0",
            ):
                try:
                    await db.execute(stmt)
                except Exception:
                    pass
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_ep_episode_id ON episodes(episode_id)"
            )
            # Backfill episode_id on legacy rows (no chroma dual-write).
            async with db.execute(
                "SELECT id FROM episodes WHERE episode_id IS NULL OR episode_id = ''"
            ) as cur:
                missing = await cur.fetchall()
            for (row_id,) in missing:
                await db.execute(
                    "UPDATE episodes SET episode_id = ? WHERE id = ?",
                    (str(uuid.uuid4()), row_id),
                )
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
        source: str = "user_turn",
        episode_id: str | None = None,
    ) -> str:
        """
        Store a new episode. Returns the stable UUID episode_id.
        Novelty is embedding distance to the nearest existing episode.
        """
        decay_rate = 0.02 * (1.0 - emotional_weight * 0.8)
        eid = episode_id or str(uuid.uuid4())
        created_at = _now_iso()
        novelty = await self._compute_novelty(user_id, content)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO episodes
                   (user_id, type, content, tags, importance,
                    emotional_weight, state_snapshot, decay_rate,
                    recall_count, used_count, effective_strength, associations,
                    created_at, expires_at, episode_id, source, novelty)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 1.0, ?, ?, ?, ?, ?, ?)""",
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
                    created_at,
                    expires_at.isoformat() if expires_at else None,
                    eid,
                    source,
                    novelty,
                ),
            )

            uid = user_id
            self._new_since_consolidation[uid] = (
                self._new_since_consolidation.get(uid, 0) + 1
            )
            await db.commit()

        await self._chroma_upsert(
            user_id=user_id,
            episode_id=eid,
            content=content,
            created_at=created_at,
            source=source,
            novelty=novelty,
        )
        return eid

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

        Does not increment recall_count — that happens in a batch after
        prompt assembly (KEEPER_PATCH_01 B5).
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

    async def format_for_prompt(self, user_id: int) -> tuple[str, list[str], list[str]]:
        """Return (block, recalled_ids, used_ids). Truncates to the char budget."""
        episodes = await self.recall(user_id, limit=15)
        if not episodes:
            return "", [], []

        recalled = [ep.get("episode_id") or str(ep["id"]) for ep in episodes]
        lines = ["## What I remember about you"]
        used: list[str] = []
        budget = config.FORGET_PROMPT_CHAR_BUDGET
        for ep in episodes:
            tags = json.loads(ep["tags"] or "[]")
            if "degraded" in tags:
                continue
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            strength = ep.get("effective_strength", 1.0)
            fade = "" if strength > 0.5 else " (fading)"
            line = f"- [{ep['type']}]{tag_str} {ep['content']}{fade}"
            candidate = "\n".join(lines + [line])
            if used and len(candidate) > budget:
                break
            lines.append(line)
            used.append(ep.get("episode_id") or str(ep["id"]))
        return "\n".join(lines), recalled, used

    async def _compute_novelty(self, user_id: int, content: str) -> float:
        """Cosine distance to nearest existing episode. 1.0 if the store is empty."""
        try:
            from memory.salience import clamp
            from memory.semantic import semantic

            col = semantic._get_client().get_or_create_collection(
                name=f"episodic_v1_user_{user_id}",
                metadata={"hnsw:space": "cosine"},
            )
            if col.count() == 0:
                return 1.0
            embedding = await semantic._embed(content)
            results = col.query(
                query_embeddings=[embedding],
                n_results=1,
                include=["distances"],
            )
            dists = (results.get("distances") or [[]])[0]
            if not dists:
                return 1.0
            return clamp(float(dists[0]), 0.0, 1.0)
        except Exception:
            log.warning("novelty compute failed; defaulting to 1.0", exc_info=True)
            return 1.0

    async def _chroma_upsert(
        self,
        user_id: int,
        episode_id: str,
        content: str,
        created_at: str,
        source: str,
        novelty: float,
        recall_count: int = 0,
        used_count: int = 0,
    ) -> None:
        try:
            from memory.semantic import semantic
            col = semantic._get_client().get_or_create_collection(
                name=f"episodic_v1_user_{user_id}",
                metadata={"hnsw:space": "cosine"},
            )
            embedding = await semantic._embed(content)
            col.upsert(
                ids=[episode_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[{
                    "episode_id": episode_id,
                    "created_at": created_at,
                    "source": source,
                    "novelty": float(novelty),
                    "recall_count": int(recall_count),
                    "used_count": int(used_count),
                    "user_id": int(user_id),
                }],
            )
        except Exception:
            log.warning("episodic chroma upsert failed for %s", episode_id, exc_info=True)

    def _chroma_update_counts(self, rows: list) -> None:
        if not rows:
            return
        try:
            from collections import defaultdict
            from memory.semantic import semantic
            by_user: dict[int, list] = defaultdict(list)
            for row in rows:
                by_user[int(row["user_id"])].append(row)
            for uid, items in by_user.items():
                col = semantic._get_client().get_or_create_collection(
                    name=f"episodic_v1_user_{uid}",
                    metadata={"hnsw:space": "cosine"},
                )
                col.update(
                    ids=[r["episode_id"] for r in items],
                    metadatas=[{
                        "episode_id": r["episode_id"],
                        "created_at": r["created_at"] or "",
                        "source": r["source"] or "user_turn",
                        "novelty": float(r["novelty"] if r["novelty"] is not None else 1.0),
                        "recall_count": int(r["recall_count"] or 0),
                        "used_count": int(r["used_count"] or 0),
                        "user_id": uid,
                    } for r in items],
                )
        except Exception:
            log.warning("episodic chroma count update failed", exc_info=True)

    def _chroma_delete(self, user_id: int, episode_ids: list[str]) -> None:
        if not episode_ids:
            return
        try:
            from memory.semantic import semantic
            col = semantic._get_client().get_or_create_collection(
                name=f"episodic_v1_user_{user_id}",
                metadata={"hnsw:space": "cosine"},
            )
            col.delete(ids=episode_ids)
        except Exception:
            log.warning("episodic chroma delete failed", exc_info=True)

    async def increment_counters(
        self,
        recall_ids: list[str],
        used_ids: list[str],
    ) -> None:
        """Batched recall_count / used_count update after prompt assembly."""
        if not recall_ids and not used_ids:
            return
        now = _now_iso()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            if recall_ids:
                placeholders = ",".join("?" for _ in recall_ids)
                await db.execute(
                    f"""UPDATE episodes
                        SET recall_count = recall_count + 1,
                            last_recalled_at = ?
                        WHERE episode_id IN ({placeholders})""",
                    [now, *recall_ids],
                )
            if used_ids:
                placeholders = ",".join("?" for _ in used_ids)
                await db.execute(
                    f"""UPDATE episodes
                        SET used_count = used_count + 1
                        WHERE episode_id IN ({placeholders})""",
                    used_ids,
                )
            await db.commit()
            ids = list(dict.fromkeys([*recall_ids, *used_ids]))
            ph = ",".join("?" for _ in ids)
            async with db.execute(
                f"SELECT user_id, episode_id, created_at, source, "
                f"novelty, recall_count, used_count FROM episodes "
                f"WHERE episode_id IN ({ph})",
                ids,
            ) as cur:
                rows = await cur.fetchall()
        self._chroma_update_counts(rows)

    async def add_episode_ref(self, episode_id: str, kind: str, ref_id: str) -> None:
        if not episode_id:
            return
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR IGNORE INTO episode_refs (episode_id, kind, ref_id)
                   VALUES (?, ?, ?)""",
                (str(episode_id), kind, str(ref_id)),
            )
            await db.commit()

    async def add_episode_refs(self, episode_ids: list[str], kind: str, ref_id: str) -> None:
        for eid in episode_ids or []:
            await self.add_episode_ref(str(eid), kind, ref_id)

    async def reference_count(self, episode_id: str) -> int:
        """Number of semantic / opinion / user_life records citing this episode."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT COUNT(*) FROM episode_refs WHERE episode_id = ?",
                (str(episode_id),),
            ) as cur:
                (n,) = await cur.fetchone()
                return int(n or 0)

    async def forget_by_evidence(self, user_id: int | None = None) -> int:
        """Prune aged low-salience episodes. Dry-run unless FORGET_DRY_RUN is false."""
        from memory.salience import episode_salience, salience_terms

        now = utcnow()
        where = "WHERE 1=1"
        params: list[Any] = []
        if user_id is not None:
            where += " AND user_id = ?"
            params.append(user_id)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(f"SELECT * FROM episodes {where}", params) as cur:
                rows = [dict(r) for r in await cur.fetchall()]

        scored: list[tuple[float, dict, dict]] = []
        for ep in rows:
            created_raw = ep.get("created_at") or ""
            try:
                created = parse_iso(str(created_raw))
                age_days = (now - created).total_seconds() / 86400.0
            except Exception:
                age_days = 0.0
            if age_days < config.FORGET_MIN_AGE_DAYS:
                continue
            eid = ep.get("episode_id") or str(ep["id"])
            refs = await self.reference_count(eid)
            meta = {
                "episode_id": eid,
                "created_at": ep.get("created_at"),
                "source": ep.get("source") or "user_turn",
                "novelty": ep.get("novelty") if ep.get("novelty") is not None else 1.0,
                "used_count": ep.get("used_count") or 0,
            }
            sal = episode_salience(meta, now=now, ref_count=refs)
            if sal < config.FORGET_THRESHOLD:
                terms = salience_terms(meta, now=now, ref_count=refs)
                scored.append((sal, ep, terms))

        scored.sort(key=lambda t: t[0])
        victims = scored[: config.FORGET_MAX_PER_PASS]
        if not victims:
            return 0

        for sal, ep, terms in victims:
            log.info(
                "forget candidate episode_id=%s salience=%.3f source=%s age_days=%.1f "
                "dry_run=%s terms=%s",
                ep.get("episode_id") or ep["id"],
                sal,
                ep.get("source"),
                terms.get("age_days"),
                config.FORGET_DRY_RUN,
                {k: round(v, 3) if isinstance(v, float) else v for k, v in terms.items()},
            )

        if config.FORGET_DRY_RUN:
            return 0

        by_user: dict[int, list[str]] = {}
        ids = []
        for _, ep, _ in victims:
            eid = ep.get("episode_id") or str(ep["id"])
            ids.append(ep["id"])
            by_user.setdefault(int(ep["user_id"]), []).append(eid)

        async with aiosqlite.connect(self.db_path) as db:
            ph = ",".join("?" for _ in ids)
            await db.execute(f"DELETE FROM episodes WHERE id IN ({ph})", ids)
            await db.commit()
        for uid, eids in by_user.items():
            self._chroma_delete(uid, eids)
        return len(ids)

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
