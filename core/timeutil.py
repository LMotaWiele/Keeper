"""UTC stores plus user-local display. Persisted timestamps stay timezone-aware UTC."""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_iso(s: str) -> datetime:
    """Parse an ISO timestamp, attaching UTC when the value is naive."""
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def user_tz() -> ZoneInfo:
    from config.settings import config

    name = (config.USER_TIMEZONE or "Europe/Amsterdam").strip()
    try:
        return ZoneInfo(name)
    except Exception:
        return ZoneInfo("UTC")


def to_user_local(dt: datetime | None = None) -> datetime:
    return as_utc(dt or utcnow()).astimezone(user_tz())


def local_now(now: datetime | None = None) -> datetime:
    return to_user_local(now or utcnow())


def format_local_stamp(dt: datetime, date_only: bool = False) -> str:
    """`2026-09-07 16:39 CEST` or `2026-09-07 CEST`."""
    local = to_user_local(dt)
    tz = local.tzname() or "UTC"
    if date_only:
        return f"{local.strftime('%Y-%m-%d')} {tz}"
    return f"{local.strftime('%Y-%m-%d %H:%M')} {tz}"


def age_phrase(event_at: datetime, now: datetime | None = None) -> str:
    """Relative age in the user timezone, recomputed against `now`."""
    now_local = local_now(now)
    event_local = to_user_local(event_at)
    days = (now_local.date() - event_local.date()).days
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    return f"{days}d ago"


def memory_time_prefix(event_at: datetime, now: datetime | None = None) -> str:
    """`[2026-09-06 CEST · 12d ago]`."""
    stamp = format_local_stamp(event_at, date_only=True)
    return f"[{stamp} · {age_phrase(event_at, now)}]"
