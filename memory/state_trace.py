"""Change A — behaviour-neutral affect/drive trace. Never raises into the chat path."""
from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
from typing import Any

from config.settings import config
from core.timeutil import utcnow

log = logging.getLogger(__name__)

SCHEMA = """
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
"""

_queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=4096)
_writer_started = False
_writer_lock = threading.Lock()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def _write_row(row: dict[str, Any]) -> None:
    conn = sqlite3.connect(str(config.midterm_db_path), timeout=10)
    try:
        _ensure_schema(conn)
        conn.execute(
            """INSERT INTO state_trace
               (ts, event_type, event_detail, session_id,
                arousal, valence, curiosity, fatigue, drives_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row["ts"],
                row["event_type"],
                row.get("event_detail"),
                row.get("session_id"),
                row["arousal"],
                row["valence"],
                row["curiosity"],
                row["fatigue"],
                row["drives_json"],
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _writer_loop() -> None:
    while True:
        row = _queue.get()
        if row is None:
            break
        try:
            _write_row(row)
        except Exception:
            log.warning("state_trace write failed", exc_info=True)


def _ensure_writer() -> None:
    global _writer_started
    with _writer_lock:
        if _writer_started:
            return
        config.midterm_db_path.parent.mkdir(parents=True, exist_ok=True)
        thread = threading.Thread(target=_writer_loop, name="state-trace-writer", daemon=True)
        thread.start()
        _writer_started = True


def record(state: Any, event: Any) -> None:
    """Enqueue one trace row. Safe to call from the sync update path."""
    try:
        _ensure_writer()
        drives = {}
        drive_sys = getattr(state, "drives", None)
        if drive_sys is not None:
            for d in getattr(drive_sys, "all_drives", []) or []:
                drives[d.name] = round(float(d.intensity), 4)
        detail = (
            getattr(event, "tool_name", None)
            or getattr(event, "type", None)
            or None
        )
        row = {
            "ts": utcnow().isoformat(),
            "event_type": type(event).__name__,
            "event_detail": str(detail) if detail is not None else None,
            "session_id": getattr(state, "session_id", None),
            "arousal": float(state.arousal),
            "valence": float(state.valence),
            "curiosity": float(state.curiosity),
            "fatigue": float(getattr(state, "effective_fatigue", state.fatigue)),
            "drives_json": json.dumps(drives),
        }
        _queue.put_nowait(row)
    except Exception:
        log.warning("state_trace enqueue failed", exc_info=True)


def flush(timeout: float = 2.0) -> None:
    """Block until the queue is empty (tests)."""
    sentinel = object()
    try:
        _queue.join  # type: ignore[attr-defined]
    except Exception:
        pass
    deadline = timeout
    import time
    start = time.monotonic()
    while not _queue.empty() and (time.monotonic() - start) < deadline:
        time.sleep(0.02)
