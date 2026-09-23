"""Self-theorizer — offline architecture proposals, min 6h between cycles."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from core.timeutil import utcnow, parse_iso
from typing import Any

from langchain_core.messages import HumanMessage

from core.llm import get_llm

log = logging.getLogger(__name__)

THEORIZE_PROMPT = """\
You are examining your own architecture. Use the structure below. Look at
what the code does now before saying what should differ.

## Your Architecture
{architecture_summary}

## Your Current Self-Model
{self_model}

## Your Current Internal State
{internal_state}

## Recent Behavioral Observations
{recent_observations}

Emit a JSON list. Each item is either a proposal or a speculation.

A proposal requires all of:
  target_module    one file path
  current_symbol   the function or class as it exists now
  current_behaviour  one sentence on what it does today
  change           one sentence on what differs afterwards
  verification     {{"kind":"script", "path":"scripts/<name>.py", "passes_when":"exit 0"}} or
                   {{"kind":"metric", "table":..., "column":...,
                    "direction":"increases"|"decreases"|"varies",
                    "threshold":..., "window_hours":...}}

If you cannot supply a verification that could come back false,
emit {{"kind":"speculation", "text": ...}} instead. This is the
correct output for most observations. Do not invent a check to
qualify something as a proposal.

One module per proposal. A change spanning several modules is not a
proposal — emit it as speculation, or split it.
"""


class SelfTheorizer:
    """Generates architectural improvement proposals during downtime."""

    MIN_INTERVAL_HOURS = 6

    def __init__(self, codebase: Any, self_model: Any, memory_system: Any):
        self.codebase = codebase
        self.self_model = self_model
        self.memory = memory_system
        self._last_run: datetime | None = None
        self._proposals: list[dict] = []

    async def theorize(
        self,
        user_id: int,
        internal_state: Any,
        *,
        apply_jev_gate: bool = True,
    ) -> dict | None:
        """Run one theorizing cycle. Returns the analysis or None."""
        if self._last_run and (
            utcnow() - self._last_run
        ).total_seconds() < self.MIN_INTERVAL_HOURS * 3600:
            return None

        arch_summary = self.codebase.summary if self.codebase else "Architecture index unavailable"
        model_ctx = json.dumps(self.self_model.model, indent=2, default=str)
        state_ctx = internal_state.to_prompt_context() if internal_state else "No state"

        observations = []
        if self.memory:
            obs = await self.memory.episodic.get_recent(
                user_id, limit=20, type="self_observation"
            )
            observations = [o["content"] for o in obs]

        obs_text = "\n".join(f"- {o}" for o in observations[-15:]) or "No observations yet"

        if apply_jev_gate:
            try:
                from core.jev import jev_should_run
                if not await jev_should_run(
                    "architecture_question_open",
                    {
                        "architecture_summary": arch_summary[:3000],
                        "tensions": (self.self_model.model.get("tensions") if self.self_model else None),
                        "recent_observations": obs_text[:3000],
                    },
                    background=True,
                    task="self_theorize",
                ):
                    log.info("self_theorize skipped — jev architecture_question_open")
                    return None
            except Exception:
                log.debug("self_theorize Jev gate failed — fail open", exc_info=True)

        prompt = THEORIZE_PROMPT.format(
            architecture_summary=arch_summary,
            self_model=model_ctx,
            internal_state=state_ctx,
            recent_observations=obs_text,
        )

        try:
            from core.json_utils import parse_json_lenient
            from goals.proposals import (
                coerce_theorizer_items,
                current_run_id,
                integration_score,
            )
            result = await get_llm("self_theorize", json_mode=True).ainvoke(
                [HumanMessage(content=prompt)]
            )
            parsed = parse_json_lenient(result.content)
            items = coerce_theorizer_items(parsed)
            if parsed is None:
                log.warning("Self-theorizing failed to parse")
                return None
            self._last_run = utcnow()
            run_id = current_run_id()

            queued: list[dict] = []
            speculations: list[dict] = []
            for item in items:
                classified = await self._classify(item, run_id)
                if classified.get("kind") == "proposal":
                    stored = classified["proposal"]
                    self._proposals.append(stored)
                    queued.append(stored)
                else:
                    speculations.append(classified)

            score = integration_score()
            score_txt = f"{score:.2f}" if score is not None else "n/a"
            if self.memory:
                await self.memory.store_episode(
                    user_id=user_id,
                    content=(
                        f"[Self-theorizing] {len(queued)} proposals queued, "
                        f"{len(speculations)} speculations stored. "
                        f"Integration: {score_txt}"
                    ),
                    internal_state=internal_state,
                    salience=0.7,
                    type="event",
                    tags=["self_theorizing", "meta", "self_improvement"],
                    source="autonomous_artifact",
                )

            log.info(
                "Self-theorizing complete: %d proposals, %d speculations, integration=%s",
                len(queued), len(speculations), score_txt,
            )
            return {
                "proposals": queued,
                "speculations": speculations,
                "integration": score,
            }

        except Exception as e:
            log.warning("Self-theorizing failed: %s", e)
            return None

    async def _classify(self, item: Any, run_id: str) -> dict:
        """Ground current_behaviour in the symbol source, then test redundancy."""
        from goals.proposals import (
            behaviour_sentence,
            classify_theorizer_item_async,
            lexical_states_differ,
        )

        async def describe(symbol: str, source: str, change: str) -> str:
            try:
                prompt = (
                    f"Source of {symbol}:\n\n{source[:6000]}\n\n"
                    "In one sentence, what does this code do now? "
                    "Describe only what is in the source. Output the sentence only."
                )
                result = await get_llm("proposal_current_behaviour").ainvoke(
                    [HumanMessage(content=prompt)]
                )
                text = (getattr(result, "content", None) or "").strip().splitlines()
                if text and text[0].strip():
                    return text[0].strip()[:400]
            except Exception:
                log.debug("proposal current_behaviour call failed", exc_info=True)
            return behaviour_sentence(symbol, source, change)

        async def differs(current: str, change: str) -> bool:
            try:
                prompt = (
                    f'Current: "{current}"\n'
                    f'Proposed: "{change}"\n\n'
                    "Does the proposed state differ from the current state? yes or no."
                )
                result = await get_llm(
                    "proposal_redundancy", temperature=0, max_tokens=8,
                ).ainvoke([HumanMessage(content=prompt)])
                answer = (getattr(result, "content", None) or "").strip().lower()
                if answer.startswith("no"):
                    return False
                if answer.startswith("yes"):
                    return True
            except Exception:
                log.debug("proposal redundancy call failed", exc_info=True)
            return lexical_states_differ(current, change)

        return await classify_theorizer_item_async(
            item,
            run_id=run_id,
            describe=describe,
            differs=differs,
        )

    @property
    def pending_proposals(self) -> list[dict]:
        return [
            p for p in self._proposals
            if p.get("status") == "pending_review" and p.get("verification")
        ]

    def approve_proposal(self, index: int):
        if 0 <= index < len(self._proposals):
            self._proposals[index]["status"] = "approved"

    def reject_proposal(self, index: int):
        if 0 <= index < len(self._proposals):
            self._proposals[index]["status"] = "rejected"

    def snapshot(self) -> dict:
        return {
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "proposals": self._proposals,
        }

    def restore(self, data: dict):
        from goals.proposals import (
            current_run_id,
            has_executable_check,
            legacy_speculation_id,
            record_speculation,
        )
        lr = data.get("last_run")
        self._last_run = parse_iso(lr) if lr else None
        kept: list[dict] = []
        run_id = current_run_id()
        for raw in data.get("proposals", []) or []:
            if not isinstance(raw, dict):
                continue
            if raw.get("verification") and has_executable_check(raw) and raw.get("target_module"):
                raw.setdefault("status", "pending_review")
                raw.setdefault("generated_at", raw.get("created_at"))
                kept.append(raw)
                continue
            text = str(
                raw.get("change")
                or raw.get("rationale")
                or raw.get("title")
                or raw.get("expected_impact")
                or raw
            )
            try:
                record_speculation(
                    text=text,
                    reason="no_verification",
                    created_by_run=str(raw.get("created_by_run") or run_id),
                    target_module=str(raw.get("target_module") or ""),
                    current_symbol=str(raw.get("current_symbol") or ""),
                    spec_id=legacy_speculation_id(raw),
                    created_at=str(raw.get("created_at") or raw.get("generated_at") or utcnow().isoformat()),
                )
            except Exception:
                log.debug("legacy proposal speculation migrate failed", exc_info=True)
        self._proposals = kept
