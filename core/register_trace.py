"""One deterministic row per conversational reply. No model call."""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path

from config.settings import config
from core.timeutil import utcnow

log = logging.getLogger(__name__)

EVALUATIVE_PHRASES = (
    "i notice that",
    "progress",
    "status",
    "commitment",
    "accountability",
    "tendency",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS register_trace (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                       TEXT NOT NULL,
    turn_id                  TEXT NOT NULL,
    concrete_referent_count  INTEGER NOT NULL,
    distinct_referent_ratio  REAL,
    commitment_mention       INTEGER NOT NULL,
    user_raised_it           INTEGER NOT NULL,
    evaluative_lexicon_hits  INTEGER NOT NULL,
    reply_token_count        INTEGER NOT NULL,
    referents_json           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_register_trace_ts ON register_trace(ts);
"""

_TOKEN = re.compile(r"[a-z0-9][a-z0-9'+-]*", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9']+", re.IGNORECASE)


def _conn() -> sqlite3.Connection:
    path = Path(config.midterm_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def reply_token_count(text: str) -> int:
    return len((text or "").split())


def lexicon_hits(reply: str) -> int:
    low = (reply or "").lower()
    hits = low.count("i notice that")
    for word in ("progress", "status", "commitment", "accountability", "tendency"):
        hits += len(re.findall(rf"\b{word}\b", low))
    return hits


def _significant(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text or "") if len(t) >= 4}


def referent_tokens(reply: str, slot_values: list[str]) -> list[str]:
    """Distinct reply tokens that also occur in a live world-slot value."""
    reply_toks = _significant(reply)
    slot_toks: set[str] = set()
    for value in slot_values:
        slot_toks |= _significant(value)
    return sorted(reply_toks & slot_toks)


def _windows(text: str, size: int = 4) -> list[str]:
    words = [w.lower() for w in _WORD.findall(text or "")]
    if len(words) < size:
        if len(words) >= 2 and len(" ".join(words)) >= 8:
            return [" ".join(words)]
        return []
    return [" ".join(words[i:i + size]) for i in range(len(words) - size + 1)]


def commitment_mentioned(reply: str, commitment_texts: list[str]) -> str | None:
    """Return the commitment text the reply quotes, or None."""
    reply_l = (reply or "").lower()
    for text in commitment_texts:
        raw = (text or "").strip()
        if len(raw) >= 8 and raw.lower() in reply_l:
            return raw
        for window in _windows(raw):
            if window in reply_l:
                return raw
    return None


def user_raised(subject: str, user_text: str, prior_user_text: str) -> bool:
    if not subject:
        return False
    blob = f"{user_text or ''}\n{prior_user_text or ''}".lower()
    if subject.lower() in blob:
        return True
    for window in _windows(subject):
        if window in blob:
            return True
    return False


def _rolling_ratio(conn: sqlite3.Connection) -> float | None:
    rows = conn.execute(
        """SELECT referents_json FROM register_trace
           ORDER BY id DESC LIMIT 20"""
    ).fetchall()
    distinct: set[str] = set()
    total = 0
    for row in rows:
        try:
            tokens = json.loads(row[0] or "[]")
        except json.JSONDecodeError:
            tokens = []
        if not isinstance(tokens, list):
            continue
        distinct.update(str(t) for t in tokens)
        total += len(tokens)
    if total == 0:
        return None
    return len(distinct) / total


def record_reply(
    *,
    turn_id: str,
    reply: str,
    user_text: str,
    prior_user_text: str,
    slot_values: list[str],
    open_commitment_texts: list[str],
) -> dict:
    tokens = referent_tokens(reply, slot_values)
    mentioned = commitment_mentioned(reply, open_commitment_texts)
    raised = user_raised(mentioned, user_text, prior_user_text) if mentioned else False
    conn = _conn()
    try:
        conn.execute(
            """INSERT INTO register_trace
               (ts, turn_id, concrete_referent_count, distinct_referent_ratio,
                commitment_mention, user_raised_it, evaluative_lexicon_hits,
                reply_token_count, referents_json)
               VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?)""",
            (
                utcnow().isoformat(),
                turn_id,
                len(tokens),
                1 if mentioned else 0,
                1 if raised else 0,
                lexicon_hits(reply),
                reply_token_count(reply),
                json.dumps(tokens),
            ),
        )
        conn.commit()
        ratio = _rolling_ratio(conn)
        conn.execute(
            """UPDATE register_trace
               SET distinct_referent_ratio = ?
               WHERE id = (SELECT MAX(id) FROM register_trace)""",
            (ratio,),
        )
        conn.commit()
    finally:
        conn.close()
    return {
        "turn_id": turn_id,
        "concrete_referent_count": len(tokens),
        "distinct_referent_ratio": ratio,
        "commitment_mention": bool(mentioned),
        "user_raised_it": raised,
        "evaluative_lexicon_hits": lexicon_hits(reply),
        "reply_token_count": reply_token_count(reply),
    }
