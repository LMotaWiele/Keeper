"""User-life tracker — commitments, wellbeing, achievements. Success is their life, not chat volume."""
from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from config.settings import config
from core.timeutil import parse_iso, user_tz, utcnow

log = logging.getLogger(__name__)

VALID_STATUSES = ("active", "done", "abandoned", "missed")
TERMINAL_STATUSES = frozenset({"done", "abandoned", "missed"})
WELLBEING_SOURCES = ("user_reported", "inferred")

SCHEMA = """
CREATE TABLE IF NOT EXISTS user_commitment (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    text              TEXT    NOT NULL,
    evidence          TEXT    NOT NULL,
    created_at        TEXT    NOT NULL,
    deadline          TEXT,
    deadline_raw      TEXT,
    status            TEXT    NOT NULL DEFAULT 'active',
    closed_at         TEXT,
    close_note        TEXT,
    source_episode_id TEXT,
    last_surfaced_at  TEXT
);

CREATE TABLE IF NOT EXISTS wellbeing_snapshot (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT  NOT NULL,
    score        REAL  NOT NULL,
    source       TEXT  NOT NULL,
    evidence     TEXT  NOT NULL,
    note         TEXT
);
"""


def ensure_schema(db_path: str | Path | None = None) -> None:
    path = str(db_path or config.midterm_db_path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _tz() -> ZoneInfo:
    return user_tz()


def parse_deadline(raw: str, now: datetime | None = None) -> datetime:
    """Resolve a deadline in USER_TIMEZONE. Raises ValueError if unparseable."""
    import dateparser

    text = (raw or "").strip()
    if not text:
        raise ValueError("empty deadline")
    now = now or datetime.now(_tz())
    if now.tzinfo is None:
        now = now.replace(tzinfo=_tz())
    else:
        now = now.astimezone(_tz())
    parsed = dateparser.parse(
        text,
        settings={
            "PREFER_DATES_FROM": "future",
            "TIMEZONE": config.USER_TIMEZONE,
            "TO_TIMEZONE": config.USER_TIMEZONE,
            "RETURN_AS_TIMEZONE_AWARE": True,
            "RELATIVE_BASE": now.replace(tzinfo=None),
        },
    )
    if parsed is None:
        raise ValueError(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_tz())
    else:
        parsed = parsed.astimezone(_tz())
    # Date-only parses land at midnight; treat as end of that local day.
    if parsed.hour == 0 and parsed.minute == 0 and parsed.second == 0 and parsed.microsecond == 0:
        parsed = parsed.replace(hour=23, minute=59, second=0, microsecond=0)
    return parsed


def format_deadline_display(dt: datetime) -> str:
    local = dt.astimezone(_tz())
    return local.strftime("%A %-d %b")


def _norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]+", " ", (s or "").casefold()).strip()


def evidence_is_paraphrase(commitment: str, evidence: str) -> bool:
    """True when evidence is empty, identical, or a near-duplicate of commitment."""
    b = _norm_text(evidence)
    if not b:
        return True
    a = _norm_text(commitment)
    if not a:
        return False
    if a == b:
        return True
    longer, shorter = (a, b) if len(a) >= len(b) else (b, a)
    if shorter in longer and len(shorter) / max(len(longer), 1) >= 0.8:
        return True
    ta, tb = set(a.split()), set(b.split())
    if ta and tb:
        jaccard = len(ta & tb) / len(ta | tb)
        if jaccard >= 0.85:
            return True
    return False


@dataclass
class UserCommitment:
    """Something the user said they'd do."""
    text: str
    evidence: str
    created_at: str
    id: int | None = None
    deadline: str | None = None
    deadline_raw: str | None = None
    status: str = "active"
    closed_at: str | None = None
    close_note: str | None = None
    source_episode_id: str | None = None
    last_surfaced_at: str | None = None

    # Legacy aliases so older snapshots/callers keep working.
    @property
    def description(self) -> str:
        return self.text

    @property
    def mentioned_at(self) -> str:
        return self.created_at

    @property
    def resolved_at(self) -> str | None:
        return self.closed_at


@dataclass
class WellbeingSnapshot:
    """A point-in-time reading of user-reported or inferred state."""
    ts: str
    score: float
    source: str
    evidence: str
    id: int | None = None
    note: str | None = None

    @property
    def timestamp(self) -> str:
        return self.ts

    @property
    def mood(self) -> str:
        return self.evidence

    @property
    def context(self) -> str:
        return self.note or ""

    @property
    def inferred_valence(self) -> float:
        # Map [0, 1] score onto the legacy [-1, 1] valence scale.
        return max(-1.0, min(1.0, (self.score * 2.0) - 1.0))


@dataclass
class LifeAchievement:
    """Something concrete the user accomplished."""
    description: str
    timestamp: str
    category: str  # career, health, creative, social, financial, personal
    keeper_contributed: bool


def _row_commitment(row: sqlite3.Row) -> UserCommitment:
    return UserCommitment(
        id=int(row["id"]),
        text=row["text"],
        evidence=row["evidence"],
        created_at=row["created_at"],
        deadline=row["deadline"],
        deadline_raw=row["deadline_raw"],
        status=row["status"],
        closed_at=row["closed_at"],
        close_note=row["close_note"],
        source_episode_id=row["source_episode_id"],
        last_surfaced_at=row["last_surfaced_at"],
    )


def _row_wellbeing(row: sqlite3.Row) -> WellbeingSnapshot:
    return WellbeingSnapshot(
        id=int(row["id"]),
        ts=row["ts"],
        score=float(row["score"]),
        source=row["source"],
        evidence=row["evidence"],
        note=row["note"],
    )


class UserLifeTracker:
    """
    Tracks the user's actual life trajectory.
    Commitments and wellbeing live in SQLite; achievements remain JSON-backed.
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = str(db_path or config.midterm_db_path)
        self.achievements: list[LifeAchievement] = []
        self._session_block: str = ""
        self._session_cached: bool = False
        self._schema_ready = False

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        ensure_schema(self.db_path)
        self._schema_ready = True

    def _conn(self) -> sqlite3.Connection:
        self.ensure_schema()
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Commitments ───────────────────────────────────────────────────────

    def record_commitment(
        self,
        text: str,
        evidence: str,
        deadline: str | None = None,
        source_episode_id: str | None = None,
        now: datetime | None = None,
    ) -> dict:
        """Insert an active commitment. Returns a structured ok/error payload."""
        now = now or utcnow()
        commitment = (text or "").strip()
        ev = (evidence or "").strip()
        if not commitment:
            return {"ok": False, "error": "commitment text is required"}
        if evidence_is_paraphrase(commitment, ev):
            return {
                "ok": False,
                "error": (
                    "evidence must be the user's own words justifying the entry; "
                    "empty, whitespace-only, or a paraphrase of the commitment is rejected"
                ),
            }

        deadline_iso = None
        deadline_raw = None
        deadline_display = None
        raw = (deadline or "").strip() or None
        if raw is not None:
            try:
                parsed = parse_deadline(raw, now=now.astimezone(_tz()))
            except ValueError:
                return {
                    "ok": False,
                    "error": (
                        f"Could not parse deadline {raw!r}. "
                        "Use an ISO date or a relative expression like "
                        "'tomorrow' or 'end of week'."
                    ),
                }
            deadline_iso = parsed.isoformat()
            deadline_raw = raw
            deadline_display = format_deadline_display(parsed)

        created = now.isoformat()
        conn = self._conn()
        try:
            cur = conn.execute(
                """INSERT INTO user_commitment
                   (text, evidence, created_at, deadline, deadline_raw, status,
                    source_episode_id)
                   VALUES (?, ?, ?, ?, ?, 'active', ?)""",
                (commitment, ev, created, deadline_iso, deadline_raw, source_episode_id),
            )
            conn.commit()
            cid = int(cur.lastrowid)
        finally:
            conn.close()

        payload = {
            "ok": True,
            "id": cid,
            "text": commitment,
            "deadline": deadline_iso,
            "deadline_display": deadline_display,
            "deadline_raw": deadline_raw,
            "source_episode_id": source_episode_id,
        }
        return payload

    async def register_episode_ref(self, episode_id: str | None, commitment_id: int) -> None:
        if not episode_id:
            return
        try:
            from memory.episodic import episodic
            await episodic.add_episode_ref(
                str(episode_id), "user_life", f"commitment:{commitment_id}"
            )
        except Exception:
            log.warning("user_life episode ref failed", exc_info=True)

    def update_status(
        self,
        commitment_id: int,
        status: str,
        note: str | None = None,
        now: datetime | None = None,
    ) -> dict:
        now = now or utcnow()
        wanted = (status or "").strip().lower()
        if wanted not in VALID_STATUSES:
            return {
                "ok": False,
                "error": (
                    f"Invalid status {status!r}. "
                    f"Valid statuses: {', '.join(VALID_STATUSES)}"
                ),
            }

        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT * FROM user_commitment WHERE id = ?", (int(commitment_id),)
            ).fetchone()
            if row is None:
                open_ids = [
                    int(r["id"])
                    for r in conn.execute(
                        "SELECT id FROM user_commitment WHERE status = 'active' ORDER BY id"
                    )
                ]
                if open_ids:
                    listed = ", ".join(str(i) for i in open_ids)
                    err = (
                        f"Unknown commitment_id {commitment_id}. "
                        f"Currently open ids: {listed}"
                    )
                else:
                    err = (
                        f"Unknown commitment_id {commitment_id}. "
                        "No open commitments."
                    )
                return {"ok": False, "error": err}

            current = row["status"]
            if current in TERMINAL_STATUSES:
                return {
                    "ok": False,
                    "error": (
                        f"Commitment {commitment_id} is already {current}; "
                        "cannot change a terminal status."
                    ),
                }

            closed_at = now.isoformat() if wanted in TERMINAL_STATUSES else None
            conn.execute(
                """UPDATE user_commitment
                   SET status = ?, closed_at = ?, close_note = ?
                   WHERE id = ?""",
                (wanted, closed_at, note, int(commitment_id)),
            )
            conn.commit()
        finally:
            conn.close()
        return {
            "ok": True,
            "id": int(commitment_id),
            "status": wanted,
            "closed_at": closed_at,
        }

    def get_active_commitments(self, now: datetime | None = None) -> list[UserCommitment]:
        """Active items plus missed that have not yet been surfaced, nearest deadline first."""
        now = now or utcnow()
        self.mark_overdue_missed(now=now)
        conn = self._conn()
        try:
            rows = conn.execute(
                """SELECT * FROM user_commitment
                   WHERE status = 'active'
                      OR (status = 'missed' AND last_surfaced_at IS NULL)"""
            ).fetchall()
        finally:
            conn.close()
        items = [_row_commitment(r) for r in rows]
        items.sort(key=lambda c: self._deadline_sort_key(c))
        return items

    def open_ids(self) -> list[int]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT id FROM user_commitment WHERE status = 'active' ORDER BY id"
            ).fetchall()
        finally:
            conn.close()
        return [int(r["id"]) for r in rows]

    def mark_overdue_missed(self, now: datetime | None = None) -> int:
        """active → missed once deadline + grace has elapsed. Returns rows updated."""
        now = now or utcnow()
        grace = timedelta(hours=config.COMMITMENT_GRACE_HOURS)
        conn = self._conn()
        n = 0
        try:
            rows = conn.execute(
                "SELECT id, deadline FROM user_commitment WHERE status = 'active' AND deadline IS NOT NULL"
            ).fetchall()
            for row in rows:
                try:
                    dl = parse_iso(row["deadline"])
                except Exception:
                    continue
                if now > dl + grace:
                    conn.execute(
                        """UPDATE user_commitment
                           SET status = 'missed', closed_at = ?, close_note = ?
                           WHERE id = ? AND status = 'active'""",
                        (now.isoformat(), "deadline + grace elapsed", int(row["id"])),
                    )
                    n += 1
            if n:
                conn.commit()
        finally:
            conn.close()
        return n

    def _deadline_sort_key(self, c: UserCommitment):
        if not c.deadline:
            return (1, datetime.max.replace(tzinfo=timezone.utc), c.id or 0)
        try:
            return (0, parse_iso(c.deadline), c.id or 0)
        except Exception:
            return (1, datetime.max.replace(tzinfo=timezone.utc), c.id or 0)

    def _is_overdue(self, c: UserCommitment, now: datetime) -> bool:
        if c.status != "active" or not c.deadline:
            return False
        try:
            return parse_iso(c.deadline) < now
        except Exception:
            return False

    def _cooldown_elapsed(self, last_surfaced_at: str | None, now: datetime) -> bool:
        if not last_surfaced_at:
            return True
        try:
            last = parse_iso(last_surfaced_at)
        except Exception:
            return True
        return (now - last) >= timedelta(hours=config.OVERDUE_SURFACE_COOLDOWN_H)

    def _injection_items(self, now: datetime | None = None) -> list[UserCommitment]:
        now = now or utcnow()
        self.mark_overdue_missed(now=now)
        conn = self._conn()
        try:
            rows = conn.execute(
                """SELECT * FROM user_commitment
                   WHERE status IN ('active', 'missed')"""
            ).fetchall()
        finally:
            conn.close()

        chosen: list[UserCommitment] = []
        for row in rows:
            c = _row_commitment(row)
            if c.status == "abandoned":
                continue
            if c.status == "missed":
                if c.last_surfaced_at is None:
                    chosen.append(c)
                continue
            if self._is_overdue(c, now):
                if self._cooldown_elapsed(c.last_surfaced_at, now):
                    chosen.append(c)
            else:
                chosen.append(c)

        chosen.sort(key=lambda c: self._deadline_sort_key(c))
        return chosen[: config.COMMITMENTS_INJECTED_MAX]

    def _stamp_surfaced(self, items: list[UserCommitment], now: datetime) -> None:
        stamp = now.isoformat()
        ids = [
            c.id for c in items
            if c.id is not None and (c.status == "missed" or self._is_overdue(c, now))
        ]
        if not ids:
            return
        conn = self._conn()
        try:
            conn.executemany(
                "UPDATE user_commitment SET last_surfaced_at = ? WHERE id = ?",
                [(stamp, i) for i in ids],
            )
            conn.commit()
        finally:
            conn.close()
        for c in items:
            if c.id in ids:
                c.last_surfaced_at = stamp

    def _due_phrase(self, c: UserCommitment, now: datetime) -> str:
        if not c.deadline:
            return "missed" if c.status == "missed" else ""
        try:
            dl = parse_iso(c.deadline).astimezone(_tz())
        except Exception:
            return ""
        local_now = now.astimezone(_tz())
        day = dl.strftime("%a %-d %b")
        delta = (dl.date() - local_now.date()).days
        if c.status == "missed":
            return f"missed, was due {day}"
        if delta == 0:
            return "due today"
        if delta > 0:
            unit = "day" if delta == 1 else "days"
            return f"due {day} ({delta} {unit})"
        overdue = abs(delta)
        unit = "day" if overdue == 1 else "days"
        return f"due {day} ({overdue} {unit} overdue)"

    def _format_block(self, items: list[UserCommitment], now: datetime) -> str:
        if not items:
            return ""
        lines = ["Open commitments:"]
        for c in items:
            due = self._due_phrase(c, now)
            extra = f" — {due}" if due else ""
            lines.append(f"  #{c.id}  {c.text}{extra}")
        return "\n".join(lines)

    def on_session_start(self, session_id: str | None = None, now: datetime | None = None) -> str:
        now = now or utcnow()
        items = self._injection_items(now=now)
        self._stamp_surfaced(items, now)
        self._session_block = self._format_block(items, now)
        self._session_cached = True
        return self._session_block

    def on_session_end(self) -> None:
        self._session_cached = False
        self._session_block = ""

    def to_prompt_context(self) -> str:
        """Session-cached commitments block. Omitted entirely when empty."""
        if not self._session_cached:
            self.on_session_start()
        return self._session_block or ""

    # ── Wellbeing ─────────────────────────────────────────────────────────

    def log_wellbeing(
        self,
        score: float,
        evidence: str,
        source: str,
        note: str | None = None,
        now: datetime | None = None,
    ) -> dict:
        now = now or utcnow()
        try:
            val = float(score)
        except (TypeError, ValueError):
            return {"ok": False, "error": f"score must be a number in [0.0, 1.0], got {score!r}"}
        if val < 0.0 or val > 1.0:
            return {"ok": False, "error": f"score must be in [0.0, 1.0], got {val}"}
        src = (source or "").strip()
        if src not in WELLBEING_SOURCES:
            return {
                "ok": False,
                "error": (
                    f"source must be 'user_reported' or 'inferred', got {source!r}"
                ),
            }
        ev = (evidence or "").strip()
        if not ev:
            return {
                "ok": False,
                "error": "evidence is required for both user_reported and inferred snapshots",
            }
        conn = self._conn()
        try:
            cur = conn.execute(
                """INSERT INTO wellbeing_snapshot (ts, score, source, evidence, note)
                   VALUES (?, ?, ?, ?, ?)""",
                (now.isoformat(), val, src, ev, note),
            )
            conn.commit()
            wid = int(cur.lastrowid)
        finally:
            conn.close()
        return {"ok": True, "id": wid, "score": val, "source": src}

    def wellbeing_trend_rows(self) -> list[WellbeingSnapshot]:
        """Query layer: inferred snapshots are excluded from trends unless flipped."""
        conn = self._conn()
        try:
            if config.WELLBEING_INFERRED_IN_TRENDS:
                rows = conn.execute(
                    "SELECT * FROM wellbeing_snapshot ORDER BY ts ASC"
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM wellbeing_snapshot
                       WHERE source = 'user_reported'
                       ORDER BY ts ASC"""
                ).fetchall()
        finally:
            conn.close()
        return [_row_wellbeing(r) for r in rows]

    def record_wellbeing(self, mood: str, context: str, valence: float):
        """Legacy helper: map [-1, 1] valence onto a user_reported snapshot."""
        score = max(0.0, min(1.0, (float(valence) + 1.0) / 2.0))
        self.log_wellbeing(
            score=score,
            evidence=mood,
            source="user_reported",
            note=context[:200] if context else None,
        )

    # ── Achievements (JSON-backed, unchanged) ─────────────────────────────

    def record_achievement(self, description: str, category: str,
                           keeper_contributed: bool = False):
        """User accomplished something real."""
        self.achievements.append(LifeAchievement(
            description=description,
            timestamp=utcnow().isoformat(),
            category=category,
            keeper_contributed=keeper_contributed,
        ))

    # ── Derived metrics ───────────────────────────────────────────────────

    def _all_commitments(self) -> list[UserCommitment]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT * FROM user_commitment ORDER BY id"
            ).fetchall()
        finally:
            conn.close()
        return [_row_commitment(r) for r in rows]

    @property
    def commitments(self) -> list[UserCommitment]:
        return self._all_commitments()

    @property
    def wellbeing_history(self) -> list[WellbeingSnapshot]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT * FROM wellbeing_snapshot ORDER BY ts ASC"
            ).fetchall()
        finally:
            conn.close()
        return [_row_wellbeing(r) for r in rows]

    @property
    def commitment_followthrough_rate(self) -> float:
        resolved = [c for c in self.commitments if c.status in TERMINAL_STATUSES]
        if not resolved:
            return 0.5
        completed = sum(1 for c in resolved if c.status == "done")
        return completed / len(resolved)

    @property
    def wellbeing_trend(self) -> str:
        history = self.wellbeing_trend_rows()
        if len(history) < 5:
            return "insufficient_data"
        recent = history[-5:]
        older = history[-10:-5] if len(history) >= 10 else recent
        r_avg = sum(w.score for w in recent) / len(recent)
        o_avg = sum(w.score for w in older) / len(older)
        if r_avg > o_avg + 0.15:
            return "improving"
        elif r_avg < o_avg - 0.15:
            return "declining"
        return "stable"

    @property
    def open_commitments(self) -> list[UserCommitment]:
        return [c for c in self.commitments if c.status == "active"]

    @property
    def stale_commitments(self) -> list[UserCommitment]:
        cutoff = (utcnow() - timedelta(days=7)).isoformat()
        return [
            c for c in self.commitments
            if c.status == "active" and c.created_at < cutoff
        ]

    def to_metrics(self) -> dict:
        return {
            "commitment_followthrough": self.commitment_followthrough_rate,
            "open_commitments": len(self.open_commitments),
            "stale_commitments": len(self.stale_commitments),
            "wellbeing_trend": self.wellbeing_trend,
            "achievements_count": len(self.achievements),
        }

    # ── Persistence ───────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "commitments": [
                {
                    "id": c.id,
                    "text": c.text,
                    "evidence": c.evidence,
                    "created_at": c.created_at,
                    "deadline": c.deadline,
                    "deadline_raw": c.deadline_raw,
                    "status": c.status,
                    "closed_at": c.closed_at,
                    "close_note": c.close_note,
                    "source_episode_id": c.source_episode_id,
                    "last_surfaced_at": c.last_surfaced_at,
                }
                for c in self.commitments
            ],
            "wellbeing_history": [
                {
                    "id": w.id,
                    "ts": w.ts,
                    "score": w.score,
                    "source": w.source,
                    "evidence": w.evidence,
                    "note": w.note,
                }
                for w in self.wellbeing_history
            ],
            "achievements": [
                {
                    "description": a.description,
                    "timestamp": a.timestamp,
                    "category": a.category,
                    "keeper_contributed": a.keeper_contributed,
                }
                for a in self.achievements
            ],
        }

    def restore(self, data: dict):
        self.ensure_schema()
        self.achievements = [
            LifeAchievement(**a) for a in data.get("achievements", [])
        ]
        conn = self._conn()
        try:
            existing = conn.execute("SELECT COUNT(*) FROM user_commitment").fetchone()[0]
            if existing:
                return
            for raw in data.get("commitments", []):
                text = raw.get("text") or raw.get("description") or ""
                if not text:
                    continue
                status = raw.get("status") or "active"
                if status == "open":
                    status = "active"
                elif status == "completed":
                    status = "done"
                elif status == "unknown":
                    status = "active"
                if status not in VALID_STATUSES:
                    status = "active"
                evidence = raw.get("evidence") or text
                source_ids = raw.get("source_episode_ids") or []
                source_id = raw.get("source_episode_id")
                if not source_id and source_ids:
                    source_id = str(source_ids[0])
                conn.execute(
                    """INSERT INTO user_commitment
                       (text, evidence, created_at, deadline, deadline_raw, status,
                        closed_at, close_note, source_episode_id, last_surfaced_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        text,
                        evidence,
                        raw.get("created_at") or raw.get("mentioned_at") or utcnow().isoformat(),
                        raw.get("deadline"),
                        raw.get("deadline_raw"),
                        status,
                        raw.get("closed_at") or raw.get("resolved_at"),
                        raw.get("close_note"),
                        source_id,
                        raw.get("last_surfaced_at"),
                    ),
                )
            for raw in data.get("wellbeing_history", []):
                if "score" in raw:
                    score = float(raw["score"])
                    source = raw.get("source") or "user_reported"
                    evidence = raw.get("evidence") or raw.get("mood") or ""
                    ts = raw.get("ts") or raw.get("timestamp") or utcnow().isoformat()
                    note = raw.get("note") or raw.get("context")
                else:
                    valence = float(raw.get("inferred_valence") or 0.0)
                    score = max(0.0, min(1.0, (valence + 1.0) / 2.0))
                    source = "inferred"
                    evidence = raw.get("mood") or ""
                    ts = raw.get("timestamp") or utcnow().isoformat()
                    note = raw.get("context")
                conn.execute(
                    """INSERT INTO wellbeing_snapshot (ts, score, source, evidence, note)
                       VALUES (?, ?, ?, ?, ?)""",
                    (ts, score, source, evidence, note),
                )
            conn.commit()
        finally:
            conn.close()
