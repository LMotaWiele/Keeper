"""Agent runner — process → graph → post_process. Single entry for user turns."""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agent.graph import MAX_TOOL_CALLS, companion_graph, message_text
from config.settings import config
from core.loop import companion
from core.timeutil import utcnow

log = logging.getLogger(__name__)


def _as_messages(raw: list) -> list[BaseMessage]:
    """Coerce working-memory dicts into LangChain messages."""
    out: list[BaseMessage] = []
    for m in raw or []:
        if isinstance(m, BaseMessage):
            out.append(m)
            continue
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "assistant":
            out.append(AIMessage(content=content))
        elif role == "system":
            out.append(SystemMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out


def _usage_from_messages(messages: list) -> tuple[int, int]:
    """Sum input/output tokens from AI message metadata."""
    inp = out = 0
    for msg in messages or []:
        usage = getattr(msg, "usage_metadata", None) or {}
        if not usage:
            meta = getattr(msg, "response_metadata", None) or {}
            usage = meta.get("usage") or meta.get("token_usage") or {}
        inp += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        out += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    return inp, out


async def run_agent(user_id: int, user_text: str) -> str:
    """Run one user message through the architecture. Returns reply text."""
    start = time.monotonic()

    context = await companion.process(user_id, user_text)

    messages = _as_messages(context.get("messages") or [])
    if not messages or message_text(messages[-1].content) != user_text:
        messages.append(HumanMessage(content=user_text))

    initial_state = {
        "user_id": user_id,
        "messages": messages,
        "system_prompt": context["system_prompt"],
        "tool_calls_pending": False,
        "response_text": "",
        "tool_calls_made": 0,
        "tier": companion.conversation_tier().value,
        "forced_answer": False,
    }

    turn_id = str(uuid.uuid4())
    _dump_assembled_prompt(
        turn_id,
        context.get("system_prompt") or "",
        context.get("prompt_token_split"),
    )

    if config.diag_mode:
        diag_dir = Path(config.data_dir) / "diag"
        diag_dir.mkdir(parents=True, exist_ok=True)
        stamp = utcnow().strftime("%Y%m%dT%H%M%S")
        prompt_path = diag_dir / f"prompt_{stamp}.txt"
        prompt_path.write_text(context.get("system_prompt") or "")
        log.info("DIAG dumped system prompt to %s (%d chars)", prompt_path, prompt_path.stat().st_size)

    trial_id = None
    bias_text = context.get("action_bias_text")
    hypothesis_id = context.get("action_bias_hypothesis_id")
    if config.ACTION_BIAS_ENABLED and bias_text and hypothesis_id:
        try:
            from core.action_bias import insert_trial
            trial_id = insert_trial(turn_id, str(hypothesis_id), str(bias_text))
        except Exception:
            log.warning("action_bias trial insert failed", exc_info=True)

    result = await companion_graph.ainvoke(initial_state)

    last_ai = next(
        (m for m in reversed(result.get("messages", [])) if isinstance(m, AIMessage)),
        None,
    )

    if config.diag_mode:
        for i, m in enumerate(result.get("messages", [])):
            meta = getattr(m, "response_metadata", {}) or {}
            usage = meta.get("token_usage") or meta.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
            log.info(
                "MSG[%d] %s tools=%d content=%s finish=%s out_tok=%s reason_tok=%s text=%r",
                i, type(m).__name__,
                len(getattr(m, "tool_calls", []) or []),
                type(getattr(m, "content", None)).__name__,
                meta.get("finish_reason"),
                usage.get("completion_tokens") or usage.get("output_tokens"),
                details.get("reasoning_tokens") if isinstance(details, dict) else None,
                message_text(getattr(m, "content", ""))[:300],
            )

    response_text = result.get("response_text", "") or ""
    if (
        not response_text
        and last_ai is not None
        and not getattr(last_ai, "tool_calls", None)
    ):
        response_text = message_text(last_ai.content)

    if not response_text:
        response_text = "I'm not sure what to say right now."

    forced_answer = bool(result.get("forced_answer"))

    if config.diag_mode:
        log.info(
            "FINAL response_text=%r tool_calls_made=%s",
            response_text, result.get("tool_calls_made"),
        )

    elapsed_ms = (time.monotonic() - start) * 1000
    input_tokens, output_tokens = _usage_from_messages(result.get("messages", []))

    await companion.post_process(
        user_id=user_id,
        user_text=user_text,
        response_text=response_text,
        tool_calls_made=result.get("tool_calls_made", 0),
        processing_time_ms=elapsed_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        forced_answer=forced_answer,
    )

    if trial_id is not None and bias_text and hypothesis_id:
        try:
            from core.action_bias import (
                schedule_evaluation,
                trial_eligible_after_reply,
                write_trial_verdict,
            )
            tool_only = bool(getattr(last_ai, "tool_calls", None)) and not (
                response_text or ""
            ).strip()
            if trial_eligible_after_reply(
                response_text=response_text,
                tool_calls_only=tool_only,
                error=False,
            ):
                schedule_evaluation(
                    trial_id, str(bias_text), response_text, str(hypothesis_id)
                )
            else:
                write_trial_verdict(trial_id, "INVALID", "ineligible reply")
        except Exception:
            log.warning("action_bias eval schedule failed", exc_info=True)

    last_meta = getattr(last_ai, "response_metadata", None) or {} if last_ai else {}
    last_finish_reason = last_meta.get("finish_reason")
    shipped_preamble = bool(getattr(last_ai, "tool_calls", None))
    tool_calls_made = result.get("tool_calls_made", 0)
    log.info(
        "Agent completed — user=%s tools=%d cap=%d finish=%s shipped_preamble=%s len=%d time=%.0fms",
        user_id, tool_calls_made, MAX_TOOL_CALLS, last_finish_reason,
        shipped_preamble, len(response_text), elapsed_ms,
    )

    try:
        _record_register(user_id, turn_id, user_text, response_text)
    except Exception:
        log.warning("register_trace failed", exc_info=True)

    return response_text


def _dump_assembled_prompt(turn_id: str, system_prompt: str, split: dict | None) -> None:
    """Write the assembled system prompt before generation."""
    if not config.DUMP_ASSEMBLED_PROMPT:
        return
    directory = Path(config.data_dir) / "logs" / "prompt"
    directory.mkdir(parents=True, exist_ok=True)
    keeper = (split or {}).get("keeper", 0)
    user = (split or {}).get("user", 0)
    header = f"# keeper_tokens={keeper} user_tokens={user}\n"
    path = directory / f"{turn_id}.txt"
    path.write_text(header + system_prompt, encoding="utf-8")
    log.info("PROMPT dumped %s keeper_tokens=%s user_tokens=%s", path, keeper, user)


def _record_register(user_id: int, turn_id: str, user_text: str, reply: str) -> None:
    from core.register_trace import record_reply
    items = companion.memory.working.get_items(user_id)
    humans = [
        getattr(i, "content", "")
        for i in items
        if getattr(i, "role", "") in ("user", "human")
    ]
    prior = humans[-2] if len(humans) >= 2 else ""
    slots = [s.value for s in companion.user_world.get_world_model()]
    open_texts = [
        c.text for c in companion.user_life.commitments if c.status == "active"
    ]
    record_reply(
        turn_id=turn_id,
        reply=reply,
        user_text=user_text,
        prior_user_text=prior,
        slot_values=slots,
        open_commitment_texts=open_texts,
    )
