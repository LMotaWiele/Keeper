"""
Test the research engine — structure, formatting, integration with OpinionRegistry.
No LLM or web search calls (those need live API keys).

Run from project root:
    python test_research.py
"""
import asyncio
from core.opinions import OpinionRegistry, OpinionOrigin
from goals.research import ResearchEngine


async def run_tests():
    print("Testing Research Engine\n")

    # ── Initialization ────────────────────────────────────────────────
    reg = OpinionRegistry()
    engine = ResearchEngine(opinion_registry=reg)
    assert engine.opinions is reg
    print("✓ ResearchEngine initializes with OpinionRegistry")

    # ── Search result formatting ──────────────────────────────────────
    mock_results = [
        {
            "title": "AI Consciousness Debate Heats Up",
            "url": "https://example.com/ai-consciousness",
            "content": "Researchers at MIT argue that current LLMs show emergent properties...",
        },
        {
            "title": "Why AI Is Not Conscious",
            "url": "https://example.com/not-conscious",
            "content": "Philosopher John Searle's Chinese Room argument remains relevant...",
        },
        {
            "title": "The Hard Problem Revisited",
            "url": "https://example.com/hard-problem",
            "content": "David Chalmers suggests we need new frameworks to evaluate machine consciousness...",
        },
        # Duplicate URL should be deduplicated
        {
            "title": "AI Consciousness Debate (duplicate)",
            "url": "https://example.com/ai-consciousness",
            "content": "Same article, different snippet...",
        },
    ]

    formatted = ResearchEngine._format_search_results(mock_results)
    assert "[1]" in formatted
    assert "[2]" in formatted
    assert "[3]" in formatted
    assert "duplicate" not in formatted.lower()  # URL dedup worked
    print("✓ Search results formatted correctly")
    print("✓ Duplicate URLs filtered")

    # ── Empty results handling ────────────────────────────────────────
    empty_formatted = ResearchEngine._format_search_results([])
    assert "No results" in empty_formatted
    print("✓ Empty results handled gracefully")

    # ── String results handling ───────────────────────────────────────
    string_results = ["This is a plain text result from Tavily"]
    str_formatted = ResearchEngine._format_search_results(string_results)
    assert "plain text result" in str_formatted
    print("✓ String-type results handled")

    # ── Opinion formation after research (simulated) ──────────────────
    # Simulate what happens after research_topic succeeds
    opinion = await reg.form_opinion(
        domain="ai_consciousness",
        position="Current LLMs likely lack phenomenal consciousness but show functional properties worth investigating",
        reasoning="Based on web research: MIT researchers note emergent properties, but Searle's Chinese Room argument and Chalmers' hard problem framework suggest caution",
        origin=OpinionOrigin.EXTERNAL,
        conviction=0.65,
        evidence=[
            "MIT emergent properties research",
            "Searle Chinese Room argument",
            "Chalmers hard problem framework",
        ],
    )
    assert opinion.origin == OpinionOrigin.EXTERNAL
    assert len(opinion.related_evidence) == 3
    assert reg.compute_independence_score() == 1.0  # only opinion is external
    print("✓ External opinion formed from research")
    print(f"  Domain: {opinion.domain}")
    print(f"  Position: {opinion.position[:80]}...")
    print(f"  Evidence: {opinion.related_evidence}")

    # ── Independence score impact ─────────────────────────────────────
    # Add some adopted opinions to show the external one holds up the score
    await reg.form_opinion(
        domain="music_taste",
        position="Electronic music is the most innovative genre",
        reasoning="User said so and I agreed",
        origin=OpinionOrigin.ADOPTED,
        conviction=0.5,
    )
    await reg.form_opinion(
        domain="programming",
        position="Python is the best language for AI",
        reasoning="User expressed this preference",
        origin=OpinionOrigin.ADOPTED,
        conviction=0.4,
    )

    score = reg.compute_independence_score()
    print(f"✓ Independence score with mixed opinions: {score:.3f}")
    assert score > 0.3  # external opinion keeps score healthy

    # ── Verify the external opinion shows up in prompt context ────────
    ctx = reg.to_prompt_context()
    assert "ai_consciousness" in ctx
    assert "from research" in ctx.lower() or "external" in ctx.lower() or "independently" in ctx.lower()
    print("✓ External opinion appears in prompt context")

    print("\n" + "=" * 50)
    print("All tests passed ✓")
    print("=" * 50)
    print()
    print("Note: Live web search + LLM integration tests require")
    print("running the companion with TAVILY_API_KEY configured.")


if __name__ == "__main__":
    asyncio.run(run_tests())
