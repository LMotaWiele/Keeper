"""Deictic time resolution and prompt overlap helpers."""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Callable

from core.timeutil import as_utc, to_user_local

OVERLAP_JACCARD = 0.45

_STOP = {
    "user", "said", "responded", "self", "observation", "given",
    "the", "and", "for", "that", "this", "with", "from", "have",
}

_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

# Longer phrases first so "last night" wins over "night".
_DAY_PHRASES: list[tuple[str, int | None]] = [
    ("last night", None),  # special-cased
    ("this morning", 0),
    ("this afternoon", 0),
    ("this evening", 0),
    ("last week", -7),
    ("this week", None),  # special-cased → Monday of current week
    ("yesterday", -1),
    ("tomorrow", 1),
    ("tonight", 0),
    ("today", 0),
]


_ECHO_PREFIXES = ("User said:", "Responded:")
_GIVEN_RE = re.compile(r"Given '([^']+)'")


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if len(w) > 2 and w not in _STOP}


def echo_core(content: str) -> str:
    """Inner utterance of a User said / Responded / Given '…' wrapper."""
    c = (content or "").strip()
    for prefix in _ECHO_PREFIXES:
        if c.startswith(prefix):
            return c[len(prefix):].strip()
    m = _GIVEN_RE.search(c)
    if m:
        return m.group(1)
    return c


def token_overlap(a: str, b: str, threshold: float = OVERLAP_JACCARD) -> bool:
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= threshold


def _pair_overlap(a: str, b: str, threshold: float) -> bool:
    if not a or not b:
        return False
    al, bl = a.lower(), b.lower()
    if len(al) >= 20 and (al in bl or bl in al):
        return True
    return token_overlap(a, b, threshold)


def overlaps_any(text: str, others: list[str], threshold: float = OVERLAP_JACCARD) -> bool:
    cores_a = [text, echo_core(text)]
    for other in others:
        if not other:
            continue
        cores_b = [other, echo_core(other)]
        for a in cores_a:
            for b in cores_b:
                if _pair_overlap(a, b, threshold):
                    return True
    return False


def is_transcript_echo(content: str, ep_type: str | None = None) -> bool:
    if ep_type == "self_observation":
        return True
    c = (content or "").lstrip()
    return c.startswith("User said:") or c.startswith("Responded:")


def _shift_local(local: datetime, days: int) -> datetime:
    return local + timedelta(days=days)


def _last_night(local: datetime) -> datetime:
    return local + timedelta(days=-1)


def _this_week_monday(local: datetime) -> datetime:
    return local - timedelta(days=local.weekday())


def _weekday_occurrence(local: datetime, target: int, *, future: bool) -> datetime:
    delta = (target - local.weekday()) % 7
    if future:
        if delta == 0:
            return local
        return local + timedelta(days=delta)
    if delta == 0:
        return local
    return local - timedelta(days=(7 - delta) % 7)


_FUTURE_WEEKDAY = re.compile(
    r"\b(?:by|until|before|this coming)\s+"
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.I,
)
_PAST_WEEKDAY = re.compile(
    r"\b(?:on|last|this past)\s+"
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.I,
)


def resolve_event_time(
    content: str,
    created_at: datetime,
) -> tuple[datetime, str]:
    """
    Bind deictic tokens in `content` to an absolute instant.

    Returns (event_at UTC, event_local_date YYYY-MM-DD).
    Unmatched or ambiguous weekday → event_at = created_at.
    Does not rewrite `content`.
    """
    created_at = as_utc(created_at)
    local = to_user_local(created_at)
    text = content or ""

    hits: list[tuple[int, Callable[[datetime], datetime]]] = []

    for phrase, days in _DAY_PHRASES:
        m = re.search(r"\b" + re.escape(phrase) + r"\b", text, re.I)
        if not m:
            continue
        if phrase == "last night":
            handler: Callable[[datetime], datetime] = _last_night
        elif phrase == "this week":
            handler = _this_week_monday
        else:
            shift = 0 if days is None else days
            handler = (lambda loc, d=shift: _shift_local(loc, d))
        hits.append((m.start(), handler))

    for m in _FUTURE_WEEKDAY.finditer(text):
        target = _WEEKDAYS[m.group(1).lower()]
        hits.append((
            m.start(),
            lambda loc, t=target: _weekday_occurrence(loc, t, future=True),
        ))
    for m in _PAST_WEEKDAY.finditer(text):
        target = _WEEKDAYS[m.group(1).lower()]
        hits.append((
            m.start(),
            lambda loc, t=target: _weekday_occurrence(loc, t, future=False),
        ))

    if hits:
        hits.sort(key=lambda h: h[0])
        local_event = hits[0][1](local)
    else:
        local_event = local

    event_utc = as_utc(local_event)
    return event_utc, local_event.date().isoformat()
