#!/usr/bin/env python3
"""Degrade autonomous-derived observations on an isolated copy; dry-run by default.

--rebuild requires candidates to have been degraded (by an earlier --apply or
this invocation). Only that explicit option can invoke model analysis.
"""
from __future__ import annotations

import argparse
import asyncio
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.timeutil import parse_iso


def _tags(value) -> list:
    if isinstance(value, str):
        value = json.loads(value or "[]")
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Episode tags must be a JSON array")
    return list(value)


def classify_observations(episodes: list[dict]) -> list[dict]:
    """Pure classifier: nearest preceding non-observation, same user, <=120s.

    Source is authoritative; unknown/missing sources are unclassified, not
    inferred from prose or session history. ID breaks equal-timestamp ties.
    Return each observation with classification, predecessor_id and rule_a.
    """
    preceding = {}
    observations = []
    for index, episode in enumerate(episodes):
        row = dict(episode)
        try:
            key = (parse_iso(row["created_at"]), int(row["id"]))
        except (KeyError, TypeError, ValueError):
            key = None
        if row.get("type") == "self_observation":
            observations.append((row, key))
        elif key is not None:
            preceding.setdefault(row.get("user_id"), []).append((key, index, row))
    for rows in preceding.values():
        rows.sort(key=lambda entry: entry[0])
    keys = {uid: [entry[0] for entry in rows] for uid, rows in preceding.items()}
    result = []
    for row, key in observations:
        classification, predecessor_id = "unclassified", None
        uid = row.get("user_id")
        if key is not None and uid is not None:
            rows = preceding.get(uid, [])
            position = bisect_right(keys.get(uid, []), key) - 1
            if position >= 0:
                prev_key, _, prev = rows[position]
                if 0 <= (key[0] - prev_key[0]).total_seconds() <= 120:
                    predecessor_id = prev.get("id")
                    if prev.get("source") == "autonomous_artifact":
                        classification = "autonomous"
                    elif prev.get("source") in ("user_turn", "keeper_response"):
                        classification = "conversation"
        result.append({**row, "classification": classification,
                       "predecessor_id": predecessor_id,
                       "rule_a": "Given 'Goal: " in (row.get("content") or "")})
    return result


def validate_classifications(rows: list[dict]) -> None:
    """Safety stops apply before repair writes AND before any rebuild."""
    mismatches = sum(row["rule_a"] and row["classification"] != "autonomous" for row in rows)
    unknown = sum(row["classification"] == "unclassified" for row in rows)
    if mismatches or (rows and unknown / len(rows) > 0.10):
        raise ValueError(
            f"Refusing repair/rebuild: rule-(a) mismatches={mismatches}; "
            f"unclassified={unknown}/{len(rows)} (must be <=10%)"
        )


async def run(*, apply: bool = False, rebuild: bool = False) -> None:
    from scripts.repair_common import require_isolated_paths
    settings = require_isolated_paths()  # All four paths, every mode, BEFORE loader.
    from scripts.diag_common import load_companion_no_loops
    companion = await load_companion_no_loops()

    with sqlite3.connect(str(settings.midterm_db_path)) as db:
        db.row_factory = sqlite3.Row
        episodes = [dict(row) for row in db.execute("SELECT * FROM episodes")]
        rows = classify_observations(episodes)
        counts = Counter(row["classification"] for row in rows)
        rule_a = sum(row["rule_a"] for row in rows)
        mismatch = sum(row["rule_a"] and row["classification"] != "autonomous" for row in rows)
        percentage = 100 * counts["unclassified"] / len(rows) if rows else 0
        print(f"observations={len(rows)} rule-(a)={rule_a} rule-(b)={counts['autonomous']}")
        print(f"autonomous={counts['autonomous']} conversation={counts['conversation']} "
              f"unclassified={counts['unclassified']} ({percentage:.2f}%)")
        print(f"rule-(a) validation: mismatches={mismatch}")
        validate_classifications(rows)
        candidates = [row for row in rows if row["classification"] == "autonomous"]
        pending = [(row, _tags(row.get("tags"))) for row in candidates
                   if "degraded" not in _tags(row.get("tags"))]
        print(f"candidates={len(candidates)} not yet degraded={len(pending)}")
        if rebuild and pending and not apply:
            raise ValueError("--rebuild requires --apply already completed on this data")
        if apply:
            for row, tags in pending:
                tags.append("degraded")
                importance = row.get("importance")
                db.execute("UPDATE episodes SET tags = ?, importance = ? WHERE id = ?",
                           (json.dumps(tags), (0.5 if importance is None else float(importance)) * 0.1,
                            row["id"]))
            db.commit()
            print(f"applied: degraded {len(pending)} observation(s)")
        else:
            print("no degradation writes (--apply not requested)" if rebuild
                  else "dry-run: no degradation writes")

    if not rebuild:
        return
    model = companion.self_model
    model.goals = companion.goals
    if model.model.get("_spec0004_rebuild"):
        print("rebuild skipped: SPEC-0004 clean rebuild already completed")
        return
    # One global model: choose the user with the largest surviving corpus;
    # never pool different users' observations into one analysis window.
    windows = []
    for uid in sorted({row["user_id"] for row in rows}, key=str):
        clean = await model._clean_observations(uid, limit=50)
        windows.append((len(clean), uid))
    count, uid = max(windows, key=lambda item: item[0]) if windows else (0, None)
    if count < 5:
        print(f"rebuild skipped: {count} clean observations (need >=5); model unchanged")
        return
    previous_version = model.model.get("model_version", 0)
    changed = await model._layer1_observation_update(
        uid, bypass_jev=True, previous_model_context="",
    )
    if changed:
        await model.generate_hypotheses(uid)
        model.model["_spec0004_rebuild"] = {
            "from_version": previous_version,
            "to_version": model.model["model_version"],
        }
        # Same atomic state writer as the live process, without unrelated saves.
        model.save(Path(settings.data_dir) / "state" / "self_model.json")
        print(f"rebuilt self-model v{model.model['model_version']}")
    else:
        print("rebuild did not update model")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    try:
        asyncio.run(run(apply=args.apply, rebuild=args.rebuild))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
