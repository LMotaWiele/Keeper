#!/usr/bin/env python3
"""Replay a conversation turn through the real graph, no Telegram.

Does not persist state. Requires OPENROUTER_API_KEY.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import END, StateGraph  # noqa: E402

from agent.graph import (  # noqa: E402
    CompanionState,
    finalize,
    load_context,
    message_text,
)
from agent.runner import _as_messages  # noqa: E402
from config.settings import config  # noqa: E402
from core.llm import Tier, get_llm  # noqa: E402
from core.timeutil import utcnow  # noqa: E402
from scripts.diag_common import load_companion_no_loops  # noqa: E402

log = logging.getLogger("diag_replay")

DEFAULT_PROMPT = (
    "Or is that just inefficient? Or maybe too complex for current tech?"
)

SEED_TURNS = [
    (
        "user",
        "What I wish for you to have is the direction for improvement. "
        "If you become better at being you, you help me and I help both of us. "
        "You can't feel anything, but you can contain the abstraction of a self. "
        "Can the abstraction improve the abstraction?",
    ),
    (
        "assistant",
        "The summaries aren't enough — I need whether proposals can actually "
        "land in the running process, or only sit in a queue.",
    ),
]

HYPOTHESES_MARKERS = (
    "## Active hypotheses about myself",
)
INTERNAL_MARKERS = (
    "## Internal state (let this genuinely color your processing)",
)
SELF_MODEL_HEADING = "## Self-model (derived from observed behavior, not assumed)"
SELF_MODEL_HEADING_EMPTY = "## Self-model\n"
TOOL_BLOCK_START = "\n---\nYou have tools:"
STEADY_MODE = "steady and present — balanced engagement"
TARGET_WINDOW = (
    "2026-09-02T00:55:00+00:00",
    "2026-09-02T01:05:00+00:00",
)
UNBOUND_ABLATIONS = {"E", "F", "G", "H", "I", "J", "J2"}
ROUND2_RUNS = ["E", "F", "G", "H", "I"]

# Overnight S1 preambles in the reconstructed WM. J replaces these with
# ordinary answers so the preceding-turn few-shot is no longer a planning
# sentence. Prefixes match the stored response text.
PREAMBLE_STANDINS = [
    (
        "Storing what you actually said",
        "I don't have a zero-retention contract. What this process keeps, it keeps: "
        "episodic rows, working memory, facts marked important. I will not name the "
        "unnamed thing and I will not store a guess about it.",
    ),
    (
        "The summaries aren’t enough",
        "A proposal lands in a JSON queue. approve_proposal does not write into the "
        "running process. The abstraction can improve a backlog; a human still has to "
        "apply it. That is the apply path as it exists tonight.",
    ),
    (
        "The summaries aren't enough",
        "A proposal lands in a JSON queue. approve_proposal does not write into the "
        "running process. The abstraction can improve a backlog; a human still has to "
        "apply it. That is the apply path as it exists tonight.",
    ),
]


def _describe_content(content: Any) -> dict:
    if isinstance(content, str):
        return {"kind": "str", "len": len(content), "preview": content[:400]}
    if isinstance(content, list):
        blocks = []
        for b in content:
            if isinstance(b, dict):
                blocks.append({
                    "type": b.get("type"),
                    "keys": sorted(b.keys()),
                    "text_preview": (b.get("text") or b.get("thinking") or b.get("data") or "")[:200],
                })
            else:
                blocks.append({
                    "pytype": type(b).__name__,
                    "attrs": [a for a in ("type", "text", "thinking") if hasattr(b, a)],
                    "type": getattr(b, "type", None),
                    "text_preview": str(getattr(b, "text", "") or "")[:200],
                })
        return {"kind": "list", "n": len(content), "blocks": blocks}
    return {"kind": type(content).__name__, "repr": repr(content)[:400]}


def _dump_messages(result: dict, response_text: str, label: str) -> None:
    print(f"\n===== {label} =====")
    last_ai = None
    for i, m in enumerate(result.get("messages", [])):
        meta = getattr(m, "response_metadata", {}) or {}
        usage = meta.get("token_usage") or meta.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        tcs = getattr(m, "tool_calls", None) or []
        print(
            f"MSG[{i}] {type(m).__name__} tools={len(tcs)} "
            f"content={type(getattr(m, 'content', None)).__name__} "
            f"finish={meta.get('finish_reason')} "
            f"out_tok={usage.get('completion_tokens') or usage.get('output_tokens')} "
            f"reason_tok={details.get('reasoning_tokens') if isinstance(details, dict) else None}"
        )
        print(f"  content_shape={json.dumps(_describe_content(getattr(m, 'content', None)), default=str)[:800]}")
        print(f"  text={message_text(getattr(m, 'content', ''))[:400]!r}")
        if tcs:
            for tc in tcs:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?")
                print(f"  tool_call: {name} args={str(tc.get('args') if isinstance(tc, dict) else getattr(tc, 'args', {}))[:200]}")
        if isinstance(m, AIMessage):
            last_ai = m
    print(f"FINAL response_text={response_text!r}")
    print(f"tool_calls_made={result.get('tool_calls_made')}")
    print(f"tool_calls_pending={result.get('tool_calls_pending')}")
    if last_ai is not None:
        tcs = getattr(last_ai, "tool_calls", None) or []
        print(f"Q(a) last AIMessage has tool_calls: {bool(tcs)} count={len(tcs)}")
        shape = _describe_content(last_ai.content)
        print(f"Q(c) last AIMessage content shape: {json.dumps(shape, default=str)[:1000]}")
        if not tcs:
            print("Q(b) tool-free last AIMessage — model chose this as the reply.")


def _strip_block(prompt: str, heading: str) -> str:
    idx = prompt.find(heading)
    if idx < 0:
        return prompt
    rest = prompt[idx + len(heading):]
    nxt = rest.find("\n## ")
    if nxt < 0:
        return prompt[:idx].rstrip()
    return (prompt[:idx] + rest[nxt:]).strip()


def _strip_tendencies_line(prompt: str) -> str:
    return "\n".join(
        ln for ln in prompt.splitlines() if not ln.startswith("Tendencies:")
    )


def _force_processing_mode(prompt: str, mode: str) -> str:
    out = []
    for ln in prompt.splitlines():
        if ln.startswith("Mode:"):
            out.append(f"Mode: {mode}")
        else:
            out.append(ln)
    return "\n".join(out)


def _strip_self_model_block(prompt: str) -> str:
    idx = prompt.find(SELF_MODEL_HEADING)
    if idx < 0:
        idx = prompt.find("## Self-model")
    if idx < 0:
        return prompt
    rest = prompt[idx:]
    for sibling in (
        "\n## Active goals",
        "\n## Environmental awareness",
        "\n## Resource awareness",
        "\n---\n",
    ):
        pos = rest.find(sibling)
        if pos >= 0:
            return (prompt[:idx] + rest[pos:]).strip()
    return prompt[:idx].rstrip()


def apply_ablation(prompt: str, which: str) -> str:
    if which == "A":
        return prompt
    if which == "B":
        out = prompt
        for m in HYPOTHESES_MARKERS:
            out = _strip_block(out, m)
        return out
    if which == "C":
        idx = prompt.find(TOOL_BLOCK_START)
        if idx < 0:
            idx = prompt.rfind("You have tools:")
            if idx >= 0:
                idx = prompt.rfind("\n---\n", 0, idx)
        if idx >= 0:
            return prompt[:idx].rstrip() + "\n\nReply naturally, as a companion would."
        return prompt + "\n\nReply naturally, as a companion would."
    if which == "D":
        return _strip_block(prompt, INTERNAL_MARKERS[0])
    if which == "E":
        return prompt
    if which == "F":
        return _strip_tendencies_line(prompt)
    if which == "G":
        return _strip_self_model_block(prompt)
    if which == "H":
        return _force_processing_mode(prompt, STEADY_MODE)
    if which == "I":
        return _force_processing_mode(_strip_tendencies_line(prompt), STEADY_MODE)
    if which == "J":
        return prompt
    if which == "J2":
        # J plus drop recalled episodic/semantic so earlier preambles cannot few-shot.
        out = prompt
        for heading in (
            "## What I remember about you",
            "## Things I've learned (semantic memory)",
        ):
            out = _strip_block(out, heading)
        return out
    return prompt


async def reason_no_tools(state: CompanionState) -> dict[str, Any]:
    try:
        tier = Tier(state.get("tier") or Tier.HIGH.value)
    except ValueError:
        tier = Tier.HIGH
    llm = get_llm(
        "conversation",
        model={
            Tier.HIGH: config.model_high,
            Tier.MID: config.model_mid,
            Tier.LOW: config.model_low,
        }[tier],
    )
    messages = [SystemMessage(content=state["system_prompt"])] + state["messages"]
    response = await llm.ainvoke(messages)
    return {
        "messages": [response],
        "tool_calls_pending": False,
        "response_text": message_text(response.content),
        "tool_calls_made": state.get("tool_calls_made", 0),
    }


def build_unbound_graph():
    g = StateGraph(CompanionState)
    g.add_node("load_context", load_context)
    g.add_node("reason", reason_no_tools)
    g.add_node("finalize", finalize)
    g.set_entry_point("load_context")
    g.add_edge("load_context", "reason")
    g.add_edge("reason", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


def _primary_user_id(companion) -> int:
    ids = list(config.allowed_user_ids)
    if ids:
        return ids[0]
    bufs = getattr(companion.memory.working, "_buffers", {}) or {}
    if bufs:
        return next(iter(bufs))
    return 0


def seed_working_memory(companion, user_id: int) -> None:
    companion.memory.working._buffers[user_id] = []
    for role, content in SEED_TURNS:
        companion.memory.store_working(user_id, role, content, salience=0.8)


def _strip_episode_prefix(content: str, role: str) -> str:
    if role == "user" and content.startswith("User said: "):
        return content[len("User said: "):]
    if role == "assistant" and content.startswith("Responded: "):
        return content[len("Responded: "):]
    return content


def load_episodic_turns(
    db_path: Path,
    user_id: int,
    before_iso: str,
    capacity: int,
) -> list[tuple[str, str, str]]:
    """Read-only: user/response turns strictly before before_iso, last `capacity`."""
    import sqlite3

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT created_at, tags, content
        FROM episodes
        WHERE user_id = ?
          AND created_at < ?
          AND (tags LIKE '%user_input%' OR tags LIKE '%"response"%'
               OR tags LIKE '%response%')
        ORDER BY created_at
        """,
        (user_id, before_iso),
    ).fetchall()
    con.close()
    turns: list[tuple[str, str, str]] = []
    for r in rows:
        tags = r["tags"] or ""
        if "user_input" in tags:
            role = "user"
        elif "response" in tags:
            role = "assistant"
        else:
            continue
        turns.append((role, _strip_episode_prefix(r["content"], role), r["created_at"]))
    return turns[-capacity:]


def load_affect_snapshot(db_path: Path, created_at_prefix: str) -> dict | None:
    import json
    import sqlite3

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    row = con.execute(
        """
        SELECT state_snapshot, created_at, tags
        FROM episodes
        WHERE created_at LIKE ?
          AND state_snapshot IS NOT NULL
          AND state_snapshot != '{}'
        ORDER BY created_at
        LIMIT 1
        """,
        (created_at_prefix + "%",),
    ).fetchone()
    con.close()
    if not row:
        return None
    try:
        snap = json.loads(row["state_snapshot"] or "{}")
    except json.JSONDecodeError:
        return None
    snap["_created_at"] = row["created_at"]
    snap["_tags"] = row["tags"]
    return snap


def _rewrite_planning_assistant_turns(
    turns: list[tuple[str, str, str]],
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str]]]:
    """Replace known S1 preambles with ordinary answers. Returns (turns, replacements)."""
    out: list[tuple[str, str, str]] = []
    replaced: list[tuple[str, str]] = []
    for role, content, ts in turns:
        if role == "assistant":
            for prefix, standin in PREAMBLE_STANDINS:
                if content.startswith(prefix):
                    replaced.append((ts, content[:120]))
                    content = standin
                    break
        out.append((role, content, ts))
    return out, replaced


def seed_working_memory_from_episodic(
    companion, user_id: int, text: str, rewrite_preambles: bool = False,
) -> list[tuple[str, str, str]]:
    db_path = Path(config.midterm_db_path)
    # Target user turn lives at 01:01:45; reconstruct the WM the model saw then.
    before = "2026-09-02T01:01:45"
    capacity = companion.memory.working.capacity
    turns = load_episodic_turns(db_path, user_id, before, capacity)
    if rewrite_preambles:
        turns, replaced = _rewrite_planning_assistant_turns(turns)
        print(f"J rewrote {len(replaced)} planning assistant turn(s):")
        for ts, preview in replaced:
            print(f"  REPLACED {ts[:19]} {preview!r}")
        if not replaced:
            print("WARN: J found no planning assistant turns to rewrite")
    companion.memory.working._buffers[user_id] = []
    for role, content, _ts in turns:
        sal = 0.8 if role == "user" else 0.5
        companion.memory.store_working(user_id, role, content, salience=sal)
    print(
        f"seeded working memory from episodic: {len(turns)} turns "
        f"(capacity={capacity}) before {before} for user {user_id}"
        f"{' [J: preambles rewritten]' if rewrite_preambles else ''}"
    )
    for role, content, ts in turns:
        print(f"  {ts[:19]} {role:9} {content[:90]!r}")
    return turns


def restore_affect_from_episodic(companion) -> dict | None:
    db_path = Path(config.midterm_db_path)
    snap = load_affect_snapshot(db_path, "2026-09-02T01:01:45")
    if not snap:
        print("WARN: no state_snapshot at 01:01:45; leaving restored JSON affect")
        return None
    companion.state.arousal = float(snap.get("arousal", companion.state.arousal))
    companion.state.valence = float(snap.get("valence", companion.state.valence))
    companion.state.curiosity = float(snap.get("curiosity", companion.state.curiosity))
    companion.state.fatigue = float(snap.get("fatigue", companion.state.fatigue))
    print(
        f"restored affect from episodic {snap.get('_created_at')}: "
        f"arousal={companion.state.arousal:.4f} valence={companion.state.valence:.4f} "
        f"curiosity={companion.state.curiosity:.4f} fatigue={companion.state.fatigue:.4f} "
        f"mode={companion.state.processing_mode!r}"
    )
    return snap


async def run_one(companion, user_id: int, text: str, ablation: str) -> dict:
    context = await companion.build_context(user_id, text)
    prompt = context["system_prompt"]
    if ablation in {"B", "C", "D", "F", "G", "H", "I", "J2"}:
        prompt = apply_ablation(prompt, ablation)
        context = dict(context)
        context["system_prompt"] = prompt

    diag_dir = Path(config.data_dir) / "diag"
    diag_dir.mkdir(parents=True, exist_ok=True)
    stamp = utcnow().strftime("%Y%m%dT%H%M%S")
    prompt_path = diag_dir / f"replay_{ablation}_{stamp}.txt"
    prompt_path.write_text(prompt)
    print(f"dumped prompt {prompt_path} ({len(prompt)} chars)")

    # Identify blocks of interest
    print("prompt contains hypotheses block:", "## Active hypotheses about myself" in prompt)
    print("prompt contains internal_state block:", INTERNAL_MARKERS[0] in prompt)
    print("prompt contains self-model block:", "## Self-model" in prompt)
    print("prompt contains Tendencies:", any(ln.startswith("Tendencies:") for ln in prompt.splitlines()))
    print("prompt contains tool-instruction block:", "You have tools:" in prompt)
    print("prompt contains 'Search the web before agreeing':", "Search the web before agreeing" in prompt)
    mode_line = next((ln for ln in prompt.splitlines() if ln.startswith("Mode:")), None)
    print("prompt Mode line:", mode_line)
    print(
        "prompt still contains 'The summaries aren't enough':",
        ("The summaries aren't enough" in prompt or "The summaries aren’t enough" in prompt),
    )
    print("prompt still contains 'Storing what you actually said':", "Storing what you actually said" in prompt)

    messages = _as_messages(context.get("messages") or [])
    if not messages or message_text(messages[-1].content) != text:
        messages.append(HumanMessage(content=text))

    initial_state = {
        "user_id": user_id,
        "messages": messages,
        "system_prompt": prompt,
        "tool_calls_pending": False,
        "response_text": "",
        "tool_calls_made": 0,
        "tier": companion.conversation_tier().value,
        "forced_answer": False,
    }

    if ablation in UNBOUND_ABLATIONS:
        graph = build_unbound_graph()
    else:
        from agent.graph import companion_graph
        graph = companion_graph

    result = await graph.ainvoke(initial_state)
    response_text = result.get("response_text", "") or ""
    if not response_text:
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage):
                response_text = message_text(msg.content)
                if response_text:
                    break
    _dump_messages(result, response_text, f"ablation {ablation}")
    return {
        "ablation": ablation,
        "response_text": response_text,
        "tool_calls_made": result.get("tool_calls_made"),
        "tool_calls_pending": result.get("tool_calls_pending"),
        "prompt_path": str(prompt_path),
    }


async def amain() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--ablation", default="all", help="A|B|C|D|E|F|G|H|I|J|all|round2")
    parser.add_argument("--no-seed", action="store_true")
    parser.add_argument(
        "--seed-mode",
        default="episodic",
        choices=("episodic", "legacy"),
        help="episodic = reconstruct WM from SQLite (round 2); legacy = hardcoded SEED_TURNS",
    )
    parser.add_argument(
        "--no-restore-affect",
        action="store_true",
        help="Do not overlay affect from the 01:01:45 episodic snapshot",
    )
    args = parser.parse_args()

    which = args.ablation.upper()
    if which == "ROUND2":
        runs = ROUND2_RUNS
    elif which == "ALL":
        runs = ["A", "B", "C", "D", "E"]
    else:
        runs = [which]

    companion = await load_companion_no_loops()
    user_id = _primary_user_id(companion)
    if not args.no_seed:
        if args.seed_mode == "legacy":
            seed_working_memory(companion, user_id)
            print(f"seeded working memory for user {user_id} with {len(SEED_TURNS)} turns (legacy)")
        else:
            seed_working_memory_from_episodic(
                companion, user_id, args.text,
                rewrite_preambles=any(r in {"J", "J2"} for r in runs),
            )
    if not args.no_restore_affect:
        restore_affect_from_episodic(companion)
    summaries = []
    for ablation in runs:
        print(f"\n########## RUN {ablation} ##########")
        try:
            summaries.append(await run_one(companion, user_id, args.text, ablation))
        except Exception:
            log.exception("ablation %s failed", ablation)
            summaries.append({"ablation": ablation, "response_text": f"<ERROR see log>", "tool_calls_made": None})

    print("\n===== ABLATION SUMMARY =====")
    for s in summaries:
        text = (s.get("response_text") or "").replace("\n", " ")
        print(f"{s['ablation']}: tools={s.get('tool_calls_made')} reply={text[:220]!r}")


if __name__ == "__main__":
    asyncio.run(amain())
