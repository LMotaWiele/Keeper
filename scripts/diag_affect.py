#!/usr/bin/env python3
"""Synthetic 30-turn affect run. Acceptance criteria in docs/fix/03-affect.md."""
from __future__ import annotations

import random
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.events import UserMessageEvent  # noqa: E402
from core.internal_state import InternalState  # noqa: E402

OUT_PATH = ROOT / "docs" / "fix" / "03-affect.md"
PIN_THRESHOLD = 0.999


def main() -> int:
    rng = random.Random(42)
    clock = {"t": datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)}

    def fake_utcnow():
        return clock["t"]

    rows = []
    gaps_s: list[float] = []

    # 30 user turns, 30s–4min apart, plus two long gaps for drift.
    with patch("core.internal_state.utcnow", fake_utcnow):
        state = InternalState()
        for i in range(30):
            if i == 10:
                gap = 12 * 60
            elif i == 20:
                gap = 10 * 60
            elif i < 4:
                gap = rng.uniform(180, 240)
            elif i < 10 or (14 <= i < 20):
                gap = rng.uniform(30, 50)
            else:
                gap = rng.uniform(90, 240)
            gaps_s.append(gap)
            clock["t"] = clock["t"] + timedelta(seconds=gap)
            novelty = rng.uniform(0.1, 0.9)
            complexity = rng.uniform(0.1, 0.9)
            state.update(UserMessageEvent(
                content=f"synthetic turn {i+1}",
                novelty=novelty,
                complexity=complexity,
                emotional_tone=0.5,
            ))
            rows.append({
                "i": i + 1,
                "gap_s": gap,
                "novelty": novelty,
                "complexity": complexity,
                "arousal": state.arousal,
                "curiosity": state.curiosity,
                "fatigue": state.effective_fatigue,
                "valence": state.valence,
                "salience": state.compute_salience(),
                "mode": state.processing_mode,
            })

    saliences = [r["salience"] for r in rows]
    stdev = statistics.pstdev(saliences) if len(saliences) > 1 else 0.0
    modes = {r["mode"] for r in rows}

    def pinned_run(dim: str) -> int:
        longest = cur = 0
        for r in rows:
            if r[dim] >= PIN_THRESHOLD:
                cur += 1
                longest = max(longest, cur)
            else:
                cur = 0
        return longest

    pin_runs = {
        dim: pinned_run(dim)
        for dim in ("arousal", "curiosity", "fatigue", "valence")
    }

    drop_ok = False
    drop_details = []
    for i in range(1, len(rows)):
        if rows[i]["gap_s"] >= 10 * 60:
            drop = rows[i - 1]["arousal"] - rows[i]["arousal"]
            drop_details.append((rows[i]["i"], rows[i]["gap_s"] / 60, drop))
            if drop >= 0.15:
                drop_ok = True

    c1 = stdev > 0.08
    c2 = all(v <= 3 for v in pin_runs.values())
    c3 = len(modes) >= 3
    c4 = drop_ok
    passed = c1 and c2 and c3 and c4

    lines = [
        "# Phase 3 — Affect range",
        "",
        "Harness: `scripts/diag_affect.py`. Fresh `InternalState`, 30 synthetic "
        "`UserMessageEvent`s, novelty/complexity ~ U[0.1, 0.9], gaps 30s–4min "
        "plus 10m and 12m pauses. Seed=42.",
        "",
        f"| turn | gap | nov | cx | arousal | curiosity | fatigue | valence | salience | mode |",
        f"|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        mode_short = r["mode"].split("—")[0].strip()
        lines.append(
            f"| {r['i']} | {r['gap_s']:.0f}s | {r['novelty']:.2f} | {r['complexity']:.2f} "
            f"| {r['arousal']:.3f} | {r['curiosity']:.3f} | {r['fatigue']:.3f} "
            f"| {r['valence']:.3f} | {r['salience']:.3f} | {mode_short} |"
        )
    lines += [
        "",
        "## Acceptance",
        "",
        f"1. stdev(compute_salience) = **{stdev:.4f}** (need > 0.08) — "
        f"{'PASS' if c1 else 'FAIL'}",
        f"2. longest pin at 1.0: {pin_runs} (need ≤ 3 consecutive) — "
        f"{'PASS' if c2 else 'FAIL'}",
        f"3. distinct processing_mode branches = **{len(modes)}** "
        f"({sorted(m.split('—')[0].strip() for m in modes)}) — "
        f"{'PASS' if c3 else 'FAIL'}",
        f"4. arousal drop across ≥10min gaps: {drop_details} (need ≥ 0.15) — "
        f"{'PASS' if c4 else 'FAIL'}",
        "",
        f"**Overall: {'PASS' if passed else 'FAIL'}**",
        "",
    ]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
