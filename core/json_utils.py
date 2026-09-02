"""Lenient JSON parsing for LLM output — strip fences, recover truncated payloads."""
from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)


def parse_json_lenient(raw: str) -> Any | None:
    """Strip fences, try strict parse, then recover truncated arrays/objects."""
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None

    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.lower().startswith("json"):
            text = text[4:].lstrip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    if text.startswith("["):
        recovered, n = _recover_array(text)
        if recovered is not None:
            log.warning(
                "JSON recovery used on truncated array — salvaged %d elements",
                n,
            )
            return recovered
    if text.startswith("{"):
        recovered = _recover_object(text)
        if recovered is not None:
            log.warning("JSON recovery used on truncated object")
            return recovered
    return None


def _recover_array(s: str) -> tuple[list | None, int]:
    in_str = False
    esc = False
    depth = 0
    last_good = None
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 1:
                last_good = i
    if last_good is None:
        return None, 0
    candidate = s[: last_good + 1] + "]"
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None, 0
    if isinstance(parsed, list):
        return parsed, len(parsed)
    return None, 0


def _recover_object(s: str) -> dict | None:
    in_str = False
    esc = False
    brace = 0
    bracket = 0
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            brace += 1
        elif ch == "}":
            brace -= 1
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket -= 1
    candidate = s
    if in_str:
        candidate += '"'
    candidate += "]" * max(0, bracket)
    candidate += "}" * max(0, brace)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
