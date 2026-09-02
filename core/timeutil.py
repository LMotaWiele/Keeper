"""UTC helpers. All timestamps in the system are timezone-aware UTC."""
from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_iso(s: str) -> datetime:
    """Parse an ISO timestamp, attaching UTC when the value is naive."""
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt
