#!/usr/bin/env python3
"""Print per-field variance of state_trace over a window (default 24h)."""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import config  # noqa: E402
from core.timeutil import parse_iso, utcnow  # noqa: E402

URGENT = 0.7
FIELDS = ("arousal", "valence", "curiosity", "fatigue")


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0, "std": 0, "min": 0, "max": 0, "p05": 0, "p95": 0}
    xs = sorted(values)
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    def pct(p: float) -> float:
        if n == 1:
            return xs[0]
        i = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
        return xs[i]
    return {
        "n": n,
        "mean": mean,
        "std": math.sqrt(var),
        "min": xs[0],
        "max": xs[-1],
        "p05": pct(5),
        "p95": pct(95),
    }


def _longest_urgent(series: list[tuple[str, float]]) -> float:
    """Longest continuous span (hours) above URGENT."""
    best = 0.0
    run_start = None
    prev_ts = None
    for ts, val in series:
        dt = parse_iso(ts)
        if val > URGENT:
            if run_start is None:
                run_start = dt
        else:
            if run_start is not None and prev_ts is not None:
                best = max(best, (prev_ts - run_start).total_seconds() / 3600.0)
            run_start = None
        prev_ts = dt
    if run_start is not None and prev_ts is not None:
        best = max(best, (prev_ts - run_start).total_seconds() / 3600.0)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="state_trace variance report")
    parser.add_argument("--hours", type=float, default=24.0)
    args = parser.parse_args()

    cutoff = (utcnow() - timedelta(hours=args.hours)).isoformat()
    db = str(config.midterm_db_path)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM state_trace WHERE ts >= ? ORDER BY ts ASC",
            (cutoff,),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        print(f"state_trace unavailable: {exc}")
        sys.exit(1)
    finally:
        conn.close()

    print(f"state_trace window={args.hours}h rows={len(rows)} db={db}")
    if not rows:
        print("suggested STATE_FIELDS_INJECTED=[]")
        print("suggested DRIVES_INJECTED=[]")
        return

    print(f"{'field':<16} {'n':>6} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} {'p05':>8} {'p95':>8}")
    suggested: list[str] = []
    for name in FIELDS:
        vals = [float(r[name]) for r in rows]
        s = _stats(vals)
        print(
            f"{name:<16} {s['n']:6.0f} {s['mean']:8.4f} {s['std']:8.4f} "
            f"{s['min']:8.4f} {s['max']:8.4f} {s['p05']:8.4f} {s['p95']:8.4f}"
        )
        if s["std"] >= 0.05:
            suggested.append(name)

    drive_names: set[str] = set()
    per_drive: dict[str, list[tuple[str, float]]] = {}
    for r in rows:
        try:
            drives = json.loads(r["drives_json"] or "{}")
        except json.JSONDecodeError:
            continue
        for name, val in drives.items():
            drive_names.add(name)
            per_drive.setdefault(name, []).append((r["ts"], float(val)))

    print()
    print(f"{'drive':<16} {'n':>6} {'mean':>8} {'std':>8} {'urgent_h':>8}")
    suggested_drives: list[str] = []
    for name in sorted(drive_names):
        series = per_drive[name]
        s = _stats([v for _, v in series])
        urgent_h = _longest_urgent(series)
        print(
            f"{name:<16} {s['n']:6.0f} {s['mean']:8.4f} {s['std']:8.4f} {urgent_h:8.2f}"
        )
        if s["std"] >= 0.05:
            suggested_drives.append(name)

    print()
    print("suggested STATE_FIELDS_INJECTED=" + ",".join(suggested))
    print("suggested DRIVES_INJECTED=" + ",".join(suggested_drives))


if __name__ == "__main__":
    main()
