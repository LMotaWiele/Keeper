"""User world model — quote-anchored facts, no assessments, no status.

One row per slot. A slot with no verbatim quote cannot be written. The
quote must be a substring of the episode it cites. Staleness changes
selection order and is never rendered.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config.settings import config
from core.timeutil import parse_iso, utcnow

log = logging.getLogger(__name__)

DOMAINS = (
    "work", "project", "people", "place", "rhythm", "material", "health",
)

HALF_LIFE_DAYS = {
    "place": 180,
    "people": 180,
    "work": 180,
    "project": 30,
    "material": 30,
    "rhythm": 21,
    "health": 14,
}

# Evaluative predicates. A stated fact can pass; an assessment cannot.
EVALUATIVE_PREDICATES = (
    "seems", "seemed", "appear", "appears", "appeared", "apparently",
    "discouraged", "anxious", "overwhelmed", "struggling", "thriving",
    "frustrated", "worried", "stressed", "depressed", "motivated",
    "unmotivated", "tendency", "tendencies", "hopeless",
)

_EVAL_RE = re.compile(
    r"\b(" + "|".join(EVALUATIVE_PREDICATES) + r")\b",
    re.IGNORECASE,
)
_FIRST_PERSON = re.compile(
    r"\b(i|i'm|i've|i'd|i'll|me|my|mine)\b",
    re.IGNORECASE,
)
_TOKEN = re.compile(r"[a-z0-9][a-z0-9'+-]*", re.IGNORECASE)
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "that", "this", "it",
    "and", "or", "but", "not", "he", "she", "they", "his", "her", "their",
    "lucas", "user",
}

HEALTH_SUBJECT = {
    "sleep", "slept", "tired", "sick", "health", "pain", "headache",
    "insomnia", "unwell", "ill", "nausea", "migraine", "energy", "rest",
}

DOMAIN_ORDER = list(DOMAINS)

SCHEMA = """
CREATE TABLE IF NOT EXISTS world_slot (
    id              TEXT PRIMARY KEY,
    domain          TEXT NOT NULL,
    key             TEXT NOT NULL,
    value           TEXT NOT NULL,
    quote           TEXT NOT NULL,
    source_turn_id  TEXT NOT NULL,
    stated_at       TEXT NOT NULL,
    confirmed_at    TEXT,
    supersedes      TEXT,
    half_life_days  INTEGER NOT NULL,
    retired_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_world_slot_key ON world_slot(key);

CREATE TABLE IF NOT EXISTS world_slot_staging (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    domain          TEXT,
    key             TEXT,
    value           TEXT,
    quote           TEXT,
    source_turn_id  TEXT,
    created_at      TEXT NOT NULL,
    status          TEXT NOT NULL,
    drop_reason     TEXT
);
"""


@dataclass
class WorldSlot:
    id: str
    domain: str
    key: str
    value: str
    quote: str
    source_turn_id: str
    stated_at: str
    confirmed_at: str | None
    supersedes: str | None
    half_life_days: int
    retired_at: str | None = None

    @property
    def live(self) -> bool:
        return self.retired_at is None


def _row(row: sqlite3.Row) -> WorldSlot:
    return WorldSlot(
        id=row["id"],
        domain=row["domain"],
        key=row["key"],
        value=row["value"],
        quote=row["quote"],
        source_turn_id=row["source_turn_id"],
        stated_at=row["stated_at"],
        confirmed_at=row["confirmed_at"],
        supersedes=row["supersedes"],
        half_life_days=int(row["half_life_days"]),
        retired_at=row["retired_at"],
    )


def staleness(slot: WorldSlot, now: datetime | None = None) -> float:
    now = now or utcnow()
    anchor = slot.confirmed_at or slot.stated_at
    try:
        then = parse_iso(anchor)
    except Exception:
        return 999.0
    days = max(0.0, (now - then).total_seconds() / 86400)
    life = slot.half_life_days or HALF_LIFE_DAYS.get(slot.domain, 30)
    if life <= 0:
        return 999.0
    return days / life


def is_stale(slot: WorldSlot, now: datetime | None = None) -> bool:
    return staleness(slot, now) >= 1.0


def evaluative(value: str) -> bool:
    return _EVAL_RE.search(value or "") is not None


def first_person(text: str) -> bool:
    return _FIRST_PERSON.search(text or "") is not None


def _distinctive(text: str) -> set[str]:
    return {
        t.lower() for t in _TOKEN.findall(text or "")
        if len(t) >= 4 and t.lower() not in _STOP
    }


def _mentions_slot(value: str, slot: WorldSlot) -> bool:
    lowered = (value or "").lower()
    if len(slot.value) >= 12 and slot.value.lower() in lowered:
        return True
    signature = _distinctive(slot.value)
    if not signature:
        return False
    found = signature & _distinctive(value)
    if not found:
        return False
    longest = max(signature, key=len)
    if longest in found and len(longest) >= 5:
        return True
    return len(found) >= max(1, (len(signature) + 1) // 2)


def is_derived(value: str, others: list[WorldSlot]) -> bool:
    """True when value stitches two existing slots together."""
    hits = sum(1 for slot in others if slot.live and _mentions_slot(value, slot))
    return hits >= 2


def promotion_allowed(dropped: int, total: int, max_rate: float | None = None) -> bool:
    """False when staged quotes fail above the configured rate."""
    if total <= 0:
        return False
    rate = max_rate if max_rate is not None else config.WORLD_CONSOLIDATION_MAX_DROP_RATE
    return (dropped / total) <= rate


def fetch_episode_content(episode_id: str, db_path: str | None = None) -> str | None:
    if not episode_id:
        return None
    path = str(db_path or config.midterm_db_path)
    try:
        conn = sqlite3.connect(path, timeout=10)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute(
            "SELECT content FROM episodes WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()
    if row is None:
        return None
    return row[0]


class UserWorldModel:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = str(db_path or config.midterm_db_path)
        self._schema_ready = False

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=10)
        try:
            conn.executescript(SCHEMA)
            conn.commit()
        finally:
            conn.close()
        self._schema_ready = True

    def _conn(self) -> sqlite3.Connection:
        self.ensure_schema()
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _live(self, key: str | None = None) -> list[WorldSlot]:
        conn = self._conn()
        try:
            if key is None:
                rows = conn.execute(
                    "SELECT * FROM world_slot WHERE retired_at IS NULL ORDER BY stated_at"
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM world_slot
                       WHERE retired_at IS NULL AND key = ?
                       ORDER BY stated_at DESC""",
                    (key,),
                ).fetchall()
        finally:
            conn.close()
        return [_row(r) for r in rows]

    def _episode_text(self, source_turn_id: str, episode_text: str | None) -> str | None:
        if episode_text is not None:
            return episode_text
        return fetch_episode_content(source_turn_id, self.db_path)

    def _reject_value(
        self,
        *,
        domain: str,
        key: str,
        value: str,
        quote: str,
        episode: str | None,
        others: list[WorldSlot],
    ) -> str | None:
        if domain not in DOMAINS:
            return f"domain must be one of {', '.join(DOMAINS)}"
        if not (key or "").strip():
            return "key is required"
        if not (value or "").strip():
            return "value is required"
        if not (quote or "").strip():
            return "quote is required and must be the user's verbatim words"
        if episode is None:
            return "source episode not found; quote cannot be checked"
        if quote not in episode:
            log.info(
                "world slot rejected paraphrase key=%s quote=%r",
                key, quote[:180],
            )
            return "quote is not a substring of the source episode"
        if evaluative(value):
            return "value must be a fact, not an assessment"
        if domain == "health" and not first_person(episode):
            return "health slots require a first-person statement in the source episode"
        if is_derived(value, others):
            return "derived slots are refused; record one fact at a time"
        return None

    def record_world_fact(
        self,
        domain: str,
        key: str,
        value: str,
        quote: str,
        *,
        source_turn_id: str | None = None,
        episode_text: str | None = None,
        now: datetime | None = None,
    ) -> dict:
        domain = (domain or "").strip().lower()
        key = (key or "").strip()
        value = (value or "").strip()
        quote = (quote or "").strip()
        turn = (source_turn_id or "").strip()
        if not turn:
            return {"ok": False, "error": "source_turn_id is required"}
        existing = self._live(key)
        if existing:
            return {
                "ok": False,
                "error": f"key {key!r} already exists; use supersede_world_fact",
            }
        episode = self._episode_text(turn, episode_text)
        others = [s for s in self._live() if s.key != key]
        reason = self._reject_value(
            domain=domain, key=key, value=value, quote=quote,
            episode=episode, others=others,
        )
        if reason:
            return {"ok": False, "error": reason}
        now = now or utcnow()
        slot_id = str(uuid.uuid4())
        half = HALF_LIFE_DAYS[domain]
        conn = self._conn()
        try:
            conn.execute(
                """INSERT INTO world_slot
                   (id, domain, key, value, quote, source_turn_id, stated_at,
                    confirmed_at, supersedes, half_life_days, retired_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, NULL)""",
                (slot_id, domain, key, value, quote, turn, now.isoformat(), half),
            )
            conn.commit()
        finally:
            conn.close()
        return {"ok": True, "id": slot_id, "key": key, "domain": domain}

    def confirm_world_fact(
        self,
        key: str,
        quote: str,
        *,
        source_turn_id: str | None = None,
        episode_text: str | None = None,
        now: datetime | None = None,
    ) -> dict:
        key = (key or "").strip()
        quote = (quote or "").strip()
        turn = (source_turn_id or "").strip()
        if not quote:
            return {"ok": False, "error": "quote is required"}
        if not turn:
            return {"ok": False, "error": "source_turn_id is required"}
        live = self._live(key)
        if not live:
            return {"ok": False, "error": f"no live slot for key {key!r}"}
        episode = self._episode_text(turn, episode_text)
        if episode is None or quote not in episode:
            log.info("world confirm rejected paraphrase key=%s", key)
            return {"ok": False, "error": "quote is not a substring of the source episode"}
        now = now or utcnow()
        slot = live[0]
        conn = self._conn()
        try:
            conn.execute(
                "UPDATE world_slot SET confirmed_at = ? WHERE id = ?",
                (now.isoformat(), slot.id),
            )
            conn.commit()
        finally:
            conn.close()
        return {"ok": True, "id": slot.id, "key": key, "confirmed_at": now.isoformat()}

    def supersede_world_fact(
        self,
        key: str,
        new_value: str,
        quote: str,
        *,
        source_turn_id: str | None = None,
        episode_text: str | None = None,
        now: datetime | None = None,
    ) -> dict:
        key = (key or "").strip()
        quote = (quote or "").strip()
        new_value = (new_value or "").strip()
        turn = (source_turn_id or "").strip()
        live = self._live(key)
        if not live:
            return {"ok": False, "error": f"no live slot for key {key!r}"}
        old = live[0]
        if not turn:
            return {"ok": False, "error": "source_turn_id is required"}
        episode = self._episode_text(turn, episode_text)
        others = [s for s in self._live() if s.id != old.id]
        reason = self._reject_value(
            domain=old.domain, key=key, value=new_value, quote=quote,
            episode=episode, others=others,
        )
        if reason:
            return {"ok": False, "error": reason}
        now = now or utcnow()
        slot_id = str(uuid.uuid4())
        conn = self._conn()
        try:
            conn.execute(
                "UPDATE world_slot SET retired_at = ? WHERE id = ?",
                (now.isoformat(), old.id),
            )
            conn.execute(
                """INSERT INTO world_slot
                   (id, domain, key, value, quote, source_turn_id, stated_at,
                    confirmed_at, supersedes, half_life_days, retired_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, NULL)""",
                (
                    slot_id, old.domain, key, new_value, quote, turn,
                    now.isoformat(), old.id, HALF_LIFE_DAYS[old.domain],
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return {"ok": True, "id": slot_id, "key": key, "supersedes": old.id}

    def forget(self, key: str, now: datetime | None = None) -> dict:
        """Retire every live slot with this key. No confirmation step."""
        key = (key or "").strip()
        live = self._live(key)
        if not live:
            return {"ok": False, "error": f"no live slot for key {key!r}"}
        now = now or utcnow()
        stamp = now.isoformat()
        conn = self._conn()
        try:
            conn.executemany(
                "UPDATE world_slot SET retired_at = ? WHERE id = ?",
                [(stamp, s.id) for s in live],
            )
            conn.commit()
        finally:
            conn.close()
        return {"ok": True, "key": key, "retired": len(live)}

    def get_world_model(self, domains: list[str] | None = None) -> list[WorldSlot]:
        wanted = None
        if domains:
            wanted = {d.strip().lower() for d in domains if d and d.strip()}
        slots = self._live()
        if wanted:
            slots = [s for s in slots if s.domain in wanted]
        return slots

    def quote_check_failures(self) -> list[WorldSlot]:
        """Live slots whose quote is not a substring of the source episode."""
        bad = []
        for slot in self._live():
            episode = fetch_episode_content(slot.source_turn_id, self.db_path)
            if episode is None or slot.quote not in episode:
                bad.append(slot)
        return bad

    # ── Consolidation staging ─────────────────────────────────────────────

    def stage_consolidation_batch(
        self,
        candidates: list[dict],
        episodes: list[dict],
        *,
        now: datetime | None = None,
    ) -> dict:
        """Stage candidates. Promote quote-valid rows only if the drop rate holds."""
        now = now or utcnow()
        staged = 0
        dropped = 0
        passed: list[dict] = []
        for cand in candidates or []:
            idx = cand.get("source_index")
            episode = None
            turn = ""
            if isinstance(idx, int) and 0 <= idx < len(episodes):
                episode = episodes[idx]
                turn = str(episode.get("episode_id") or episode.get("id") or "")
            content = (episode or {}).get("content") or ""
            quote = cand.get("quote") or ""
            domain = (cand.get("domain") or "").strip().lower()
            key = (cand.get("key") or "").strip()
            value = (cand.get("value") or "").strip()
            reason = None
            if not content or not turn:
                reason = "missing episode"
            elif quote not in content:
                reason = "quote not in episode"
            else:
                others = self._live()
                reason = self._reject_value(
                    domain=domain, key=key, value=value, quote=quote,
                    episode=content, others=others,
                )
            status = "dropped" if reason else "passed"
            if reason:
                dropped += 1
                log.info(
                    "world slot candidate dropped key=%s reason=%s",
                    key, reason,
                )
            else:
                passed.append({
                    "domain": domain, "key": key, "value": value,
                    "quote": quote, "source_turn_id": turn,
                    "episode_text": content,
                })
            staged += 1
            conn = self._conn()
            try:
                conn.execute(
                    """INSERT INTO world_slot_staging
                       (domain, key, value, quote, source_turn_id, created_at, status, drop_reason)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (domain, key, value, quote, turn, now.isoformat(), status, reason),
                )
                conn.commit()
            finally:
                conn.close()

        promoted = 0
        if promotion_allowed(dropped, staged):
            for item in passed:
                if self._live(item["key"]):
                    result = self.supersede_world_fact(
                        item["key"], item["value"], item["quote"],
                        source_turn_id=item["source_turn_id"],
                        episode_text=item["episode_text"],
                        now=now,
                    )
                else:
                    result = self.record_world_fact(
                        item["domain"], item["key"], item["value"], item["quote"],
                        source_turn_id=item["source_turn_id"],
                        episode_text=item["episode_text"],
                        now=now,
                    )
                if result.get("ok"):
                    promoted += 1
        return {
            "staged": staged,
            "dropped": dropped,
            "promoted": promoted,
        }

    # ── Prompt ────────────────────────────────────────────────────────────

    def select_for_prompt(
        self, user_text: str, now: datetime | None = None,
    ) -> list[WorldSlot]:
        now = now or utcnow()
        text = user_text or ""
        chosen: dict[str, WorldSlot] = {}
        for slot in self._live():
            include = False
            if slot.domain in {"place", "people", "work"} and not is_stale(slot, now):
                include = True
            elif slot.domain == "rhythm":
                include = True
            elif slot.domain in {"project", "material"} and _key_in_text(slot.key, text):
                include = True
            elif slot.domain == "health" and _health_subject(slot, text):
                include = True
            if include:
                chosen[slot.id] = slot
        return list(chosen.values())

    def to_prompt_context(self, user_text: str = "", now: datetime | None = None) -> str:
        """Prose, third person. No status labels, no ages, no ratios."""
        now = now or utcnow()
        slots = self.select_for_prompt(user_text, now=now)
        if not slots:
            return ""
        budget = int(config.WORLD_PROMPT_CHAR_BUDGET)
        ranked = sorted(slots, key=lambda s: staleness(s, now), reverse=True)
        kept: list[WorldSlot] = list(ranked)
        while kept and len("\n".join(_render(s) for s in kept)) > budget:
            kept.pop(0)
        kept.sort(key=lambda s: (DOMAIN_ORDER.index(s.domain) if s.domain in DOMAIN_ORDER else 99, s.key))
        lines = [_render(s) for s in kept]
        if not lines:
            return ""
        return "\n".join(lines)

    def format_inspection(self) -> str:
        slots = self._live()
        if not slots:
            return "No world facts yet."
        by_domain: dict[str, list[WorldSlot]] = {}
        for slot in slots:
            by_domain.setdefault(slot.domain, []).append(slot)
        lines: list[str] = []
        for domain in DOMAIN_ORDER:
            group = by_domain.get(domain) or []
            if not group:
                continue
            lines.append(domain)
            for slot in sorted(group, key=lambda s: s.stated_at, reverse=True):
                when = (slot.confirmed_at or slot.stated_at)[:10]
                lines.append(f"- {slot.key}: {slot.value}")
                lines.append(f"  \"{slot.quote}\" — {when}")
            lines.append("")
        return "\n".join(lines).rstrip()


def _key_in_text(key: str, text: str) -> bool:
    words = set(_TOKEN.findall((text or "").lower()))
    parts = [p for p in re.split(r"[_\s]+", (key or "").lower()) if len(p) >= 3]
    if any(p in words for p in parts):
        return True
    return (key or "").lower() in (text or "").lower()


def _health_subject(slot: WorldSlot, text: str) -> bool:
    words = set(t.lower() for t in _TOKEN.findall(text or ""))
    if words & HEALTH_SUBJECT:
        return True
    if words & _distinctive(slot.key.replace("_", " ")):
        return True
    if words & _distinctive(slot.value):
        return True
    return False


def _render(slot: WorldSlot) -> str:
    value = slot.value.strip().rstrip(".")
    sentence = value[0].upper() + value[1:] if value else value
    if sentence and not sentence.endswith("."):
        sentence += "."
    quote = (slot.quote or "").strip()
    if quote and len(quote) <= 80 and quote.lower() not in sentence.lower():
        return f"{sentence[:-1]} (\"{quote}\")." if sentence.endswith(".") else f"{sentence} (\"{quote}\")"
    return sentence


_model: UserWorldModel | None = None


def get_world() -> UserWorldModel:
    global _model
    if _model is None:
        _model = UserWorldModel()
    return _model
