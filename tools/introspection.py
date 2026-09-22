"""Introspection tools — codebase, self-theorizing, simulation, opinions, goals."""
from __future__ import annotations

from langchain_core.tools import tool

from core.llm import Tier


@tool
async def codebase_overview() -> str:
    """Summarise Keeper's own architecture from the source-code index."""
    from core.loop import companion
    return companion.codebase.summary or "Codebase index is empty."


@tool
async def codebase_list_modules() -> str:
    """List Keeper's own modules: path, pillar, one-line purpose. No source."""
    from core.loop import companion
    return companion.codebase.list_modules()


@tool
async def codebase_read_module(module_path: str) -> str:
    """Read one of Keeper's own modules (path like 'core/loop.py'): AST header plus source."""
    from core.loop import companion
    detail = companion.codebase.get_module_detail(module_path)
    source = companion.codebase.get_source(module_path)
    if source.startswith("[File not found"):
        return detail[:6000]
    return f"{detail}\n\n--- source ---\n{source[:2500]}"


@tool
async def self_theorize(user_id: int = 0) -> str:
    """Run one architectural self-theorizing cycle and return proposals."""
    from core.loop import companion
    if not companion.api_budget.allows(Tier.MID, background=False):
        return "Not enough budget today for self-theorizing."
    result = await companion.theorizer.theorize(
        user_id, companion.state, apply_jev_gate=False,
    )
    if result is None:
        return (
            "Self-theorizing is throttled — last cycle was too recent "
            f"(minimum interval {companion.theorizer.MIN_INTERVAL_HOURS}h)."
        )
    proposals = result.get("proposals") or []
    write_path = (
        "Approved proposals are queued for Lucas to implement. "
        "Nothing applies them automatically. The write path runs through him."
    )
    if not proposals:
        return "Theorizing finished with no new proposals.\n" + write_path
    lines = [
        f"Generated {len(proposals)} proposal(s):",
        write_path,
    ]
    for i, p in enumerate(proposals):
        title = p.get("title", "untitled")
        body = (
            p.get("expected_impact")
            or p.get("rationale")
            or p.get("proposal")
            or ""
        )
        chunk = f"[{i}] {title}"
        rationale = (p.get("rationale") or "")[:400]
        impact = (p.get("expected_impact") or "")[:400]
        target = p.get("target_module") or ""
        lines.append(chunk)
        if target:
            lines.append(f"  target: {target}")
        if rationale:
            lines.append(f"  rationale: {rationale}")
        if impact:
            lines.append(f"  impact: {impact}")
        if body and body not in (rationale, impact):
            lines.append(f"  {str(body)[:600]}")
    return "\n".join(lines)


@tool
async def simulate_action(action_description: str) -> str:
    """Simulate likely outcomes of a planned action before taking it."""
    from core.loop import companion
    if not companion.api_budget.allows(Tier.MID, background=False):
        return "Not enough budget today for simulation."
    sim = await companion.simulator.simulate(
        action_description=action_description,
        conversation_context="Active conversation",
        internal_state=companion.state,
        memory_context="",
        goals_context=companion.goals.to_prompt_context(),
    )
    if not sim:
        return "Simulation failed."
    rec = sim.get("recommendation", "unknown")
    reason = sim.get("reasoning", "")
    mod = sim.get("suggested_modification")
    parts = [f"Recommendation: {rec}", reason]
    if mod:
        parts.append(f"Suggested modification: {mod}")
    return "\n".join(p for p in parts if p)


@tool
async def list_opinions(domain: str = "") -> str:
    """List tracked opinions, optionally filtered by domain."""
    from core.loop import companion
    opinions = list(companion.self_model.opinions.opinions.values())
    needle = domain.strip().lower()
    if needle:
        opinions = [o for o in opinions if needle in (o.domain or "").lower()]
    if not opinions:
        return "No tracked opinions" + (f" in domain {domain!r}." if domain else ".")
    lines = []
    for o in sorted(opinions, key=lambda x: -x.conviction):
        lines.append(
            f"- [{o.domain}] {o.position} "
            f"(origin={o.origin.value}, conviction={o.conviction:.2f})"
        )
    return "\n".join(lines)


@tool
async def goal_status() -> str:
    """Show active goals and Elo tables for unbounded goals."""
    from core.loop import companion
    parts = [companion.goals.to_prompt_context()]
    unbounded = [g for g in companion.goals.active_instrumental if g.is_unbounded]
    if unbounded:
        parts.append("\n## Action Elo tables")
        for g in unbounded:
            parts.append(g.name)
            ratings = g.action_ratings or {}
            if not ratings:
                parts.append("  (no rated actions yet)")
                continue
            for atype, info in sorted(
                ratings.items(), key=lambda kv: -float(kv[1].get("elo") or 0)
            ):
                parts.append(
                    f"  {atype}: elo={info.get('elo', 0):.0f} "
                    f"n={info.get('n', 0)} spend=${info.get('spend_usd', 0):.4f}"
                )
    return "\n".join(parts)


INTROSPECTION_TOOLS = [
    codebase_overview,
    codebase_list_modules,
    codebase_read_module,
    self_theorize,
    simulate_action,
    list_opinions,
    goal_status,
]
