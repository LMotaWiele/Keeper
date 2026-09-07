#!/usr/bin/env python3
"""Tag historical tool-call preambles in episodic memory as degraded.

Does not delete rows. Dry-run by default; pass --apply to write tags.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import config  # noqa: E402

PREAMBLE_MARKERS = (
    "I need ",
    "I'll check",
    "I'll look",
    "Reading the ",
    "The summaries ",
    "Next is ",
    "before answering",
)


def _response_body(content: str) -> str:
    if content.startswith("Responded: "):
        return content[len("Responded: "):]
    return content


def _is_preamble(content: str) -> bool:
    body = _response_body(content)
    if len(body) >= 200:
        return False
    return any(m in body for m in PREAMBLE_MARKERS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--db", default=str(config.midterm_db_path))
    args = parser.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT id, created_at, content, tags, importance
        FROM episodes
        WHERE type = 'event'
          AND tags LIKE '%response%'
        ORDER BY id
        """
    ).fetchall()

    matches = []
    for r in rows:
        tags = r["tags"] or "[]"
        try:
            tag_list = json.loads(tags) if isinstance(tags, str) else list(tags)
        except json.JSONDecodeError:
            tag_list = []
        if "response" not in tag_list:
            continue
        if not _is_preamble(r["content"] or ""):
            continue
        matches.append(r)

    print(f"matched {len(matches)} preamble-shaped response episode(s)")
    for r in matches:
        body = _response_body(r["content"] or "")
        print(
            f"  id={r['id']} {str(r['created_at'])[:19]} "
            f"imp={r['importance']} len={len(body)} {body[:160]!r}"
        )

    if not args.apply:
        print("dry-run — pass --apply to tag degraded and scale importance * 0.1")
        con.close()
        return

    updated = 0
    for r in matches:
        try:
            tag_list = json.loads(r["tags"] or "[]")
        except json.JSONDecodeError:
            tag_list = []
        if "degraded" not in tag_list:
            tag_list.append("degraded")
        importance = float(r["importance"] or 0.5) * 0.1
        con.execute(
            "UPDATE episodes SET tags = ?, importance = ? WHERE id = ?",
            (json.dumps(tag_list), importance, r["id"]),
        )
        updated += 1
    con.commit()
    con.close()
    print(f"applied: tagged {updated} row(s) degraded, importance *= 0.1")


if __name__ == "__main__":
    main()
