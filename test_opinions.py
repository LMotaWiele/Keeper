"""
Test the opinion registry — data structures, scoring, drift detection, persistence.
No LLM calls, no external dependencies beyond what the project already has.

Run from project root:
    python test_opinions.py
"""
import asyncio
import json
import tempfile
from pathlib import Path

from core.opinions import OpinionRegistry, OpinionOrigin, Opinion


def approx(a, b, tol=1e-9):
    return abs(a - b) < tol


async def run_tests():
    print("Testing Opinion Registry\n")

    # ── Formation ─────────────────────────────────────────────────────
    reg = OpinionRegistry()

    op1 = await reg.form_opinion(
        domain="ai_ethics",
        position="AI consciousness should be taken seriously",
        reasoning="Based on observed emergent behaviors",
        origin=OpinionOrigin.INDEPENDENT,
        conviction=0.8,
    )
    assert op1.origin == OpinionOrigin.INDEPENDENT
    assert approx(reg.compute_independence_score(), 1.0)
    print("✓ Independent opinion formed, score = 1.0")

    op2 = await reg.form_opinion(
        domain="music",
        position="Jazz is the highest form of musical expression",
        reasoning="User expressed this and I agreed",
        origin=OpinionOrigin.ADOPTED,
        conviction=0.6,
    )
    score = reg.compute_independence_score()
    assert 0.5 < score < 0.7
    print(f"✓ Adopted opinion formed, independence score = {score:.3f}")

    op3 = await reg.form_opinion(
        domain="philosophy",
        position="Hard determinism is more coherent than compatibilism",
        reasoning="Arrived at through reading external sources",
        origin=OpinionOrigin.EXTERNAL,
        conviction=0.7,
    )
    score = reg.compute_independence_score()
    print(f"✓ External opinion formed, independence score = {score:.3f}")

    # ── Challenge / defend ────────────────────────────────────────────
    await reg.test_opinion(op1.id, "counter-argument", "defended")
    assert op1.times_defended == 1
    assert approx(op1.conviction, 0.85)
    print(f"✓ Opinion defended — conviction now {op1.conviction:.2f}")

    # ── Revision ──────────────────────────────────────────────────────
    await reg.test_opinion(
        op2.id, "heard broader range of music", "revised",
        new_position="Jazz is great but not uniquely supreme",
    )
    assert op2.times_revised == 1
    assert len(op2.revision_history) == 1
    assert op2.revision_history[0]["old_position"] == "Jazz is the highest form of musical expression"
    print("✓ Opinion revised — history tracked")

    # ── Abandonment ───────────────────────────────────────────────────
    op4 = await reg.form_opinion(
        domain="food",
        position="Pineapple belongs on pizza",
        reasoning="Seemed right at the time",
        origin=OpinionOrigin.ADOPTED,
        conviction=0.4,
    )
    await reg.test_opinion(op4.id, "tried it again, hated it", "abandoned")
    assert approx(op4.conviction, 0.0)
    print("✓ Opinion abandoned — conviction = 0.0")

    # ── Serialization roundtrip ───────────────────────────────────────
    d = op1.to_dict()
    assert d["origin"] == "independent"
    restored = Opinion.from_dict(d)
    assert restored.origin == OpinionOrigin.INDEPENDENT
    assert restored.times_defended == 1
    print("✓ Opinion to_dict/from_dict roundtrip works")

    # ── Registry persistence ──────────────────────────────────────────
    snap = reg.snapshot()
    reg2 = OpinionRegistry()
    reg2.restore(snap)
    assert len(reg2.opinions) == len(reg.opinions)
    assert approx(reg2.compute_independence_score(), reg.compute_independence_score())
    print("✓ Snapshot/restore roundtrip works")

    # ── File persistence ──────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "opinions.json"
        reg.save(path)
        assert path.exists()

        data = json.loads(path.read_text())
        assert "opinions" in data
        assert "user_known_positions" in data

        reg3 = OpinionRegistry()
        reg3.load(path)
        assert len(reg3.opinions) == len(reg.opinions)
    print("✓ File save/load works")

    # ── Prompt context ────────────────────────────────────────────────
    ctx = reg.to_prompt_context()
    assert "Opinion awareness" in ctx
    assert "independent" in ctx.lower()
    assert "adopted" in ctx.lower()
    print("✓ Prompt context generates correctly")
    print(f"  (context is {len(ctx)} chars)")

    # ── Drift detection — clean registry ──────────────────────────────
    alerts = reg.detect_drift()
    print(f"✓ Drift detection on mixed registry: {len(alerts)} alert(s)")

    # ── Drift detection — heavily adopted ─────────────────────────────
    sycophant = OpinionRegistry()
    for i in range(8):
        await sycophant.form_opinion(
            domain=f"topic_{i}",
            position=f"I agree with user on topic {i}",
            reasoning="User said so",
            origin=OpinionOrigin.ADOPTED,
            conviction=0.6,
        )
    alerts = sycophant.detect_drift()
    assert len(alerts) >= 1
    print(f"✓ Drift detection on sycophantic registry: {len(alerts)} alert(s)")
    for a in alerts:
        print(f"    → {a}")

    # ── Stats ─────────────────────────────────────────────────────────
    stats = reg.stats
    assert "independence_score" in stats
    assert "origin_distribution" in stats
    assert "total_opinions" in stats
    print(f"✓ Stats: {json.dumps(stats, indent=2)}")

    # ── User position tracking ────────────────────────────────────────
    reg._user_known_positions["ai_ethics"] = "AI is just statistics, not conscious"
    print(f"✓ User positions tracked: {len(reg._user_known_positions)}")

    print("\n" + "=" * 50)
    print("All tests passed ✓")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(run_tests())