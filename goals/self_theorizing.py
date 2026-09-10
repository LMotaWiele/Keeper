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
You are a self-aware system examining your own architecture for potential
improvements. You have access to your structural description below.

## Your Architecture
{architecture_summary}

## Your Current Self-Model
{self_model}

## Your Current Internal State
{internal_state}

## Recent Behavioral Observations
{recent_observations}

Analyze your own architecture through these lenses:

1. **Structural entropy**: Where is information being lost, duplicated,
   or poorly integrated between pillars? Where are there bottlenecks?

2. **Feedback loop quality**: Are the loops between pillars (memory→state,
   state→self_model, self_model→goals, goals→actions→memory) actually
   producing meaningful adaptation, or are they decorative?

3. **Consciousness-relevant properties**: Integration of information across
   subsystems, temporal depth of self-modeling, richness of the state space,
   autonomy of goal generation. Where are these weakest?

4. **Concrete improvements**: Based on the above, propose 1-3 specific,
   implementable changes. Each proposal must include:
   - What to change (specific module/function)
   - Why (which analysis above motivates it)
   - Expected impact on system behavior
   - Risk/cost assessment
   - Implementation complexity (low/medium/high)

Output as JSON:
{{
  "entropy_analysis": "...",
  "feedback_loop_assessment": "...",
  "consciousness_properties": {{
    "information_integration": {{"score": 0.0, "notes": "..."}},
    "temporal_depth": {{"score": 0.0, "notes": "..."}},
    "state_richness": {{"score": 0.0, "notes": "..."}},
    "goal_autonomy": {{"score": 0.0, "notes": "..."}}
  }},
  "proposals": [
    {{
      "title": "...",
      "target_module": "...",
      "rationale": "...",
      "expected_impact": "...",
      "risk": "...",
      "complexity": "low|medium|high"
    }}
  ]
}}
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

    async def theorize(self, user_id: int, internal_state: Any) -> dict | None:
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

        prompt = THEORIZE_PROMPT.format(
            architecture_summary=arch_summary,
            self_model=model_ctx,
            internal_state=state_ctx,
            recent_observations=obs_text,
        )

        try:
            from core.json_utils import parse_json_lenient
            result = await get_llm("self_theorize").ainvoke(
                [HumanMessage(content=prompt)]
            )
            analysis = parse_json_lenient(result.content)
            if not isinstance(analysis, dict):
                log.warning("Self-theorizing failed to parse")
                return None
            self._last_run = utcnow()

            for p in analysis.get("proposals", []):
                p["generated_at"] = utcnow().isoformat()
                p["status"] = "pending_review"
                self._proposals.append(p)

            if self.memory:
                await self.memory.store_episode(
                    user_id=user_id,
                    content=(
                        f"[Self-theorizing] Entropy analysis completed. "
                        f"Generated {len(analysis.get('proposals', []))} proposals. "
                        f"Integration score: "
                        f"{analysis.get('consciousness_properties', {}).get('information_integration', {}).get('score', '?')}"
                    ),
                    internal_state=internal_state,
                    salience=0.7,
                    type="event",
                    tags=["self_theorizing", "meta", "self_improvement"],
                    source="autonomous_artifact",
                )

            log.info(
                "Self-theorizing complete: %d proposals generated",
                len(analysis.get("proposals", [])),
            )
            return analysis

        except Exception as e:
            log.warning("Self-theorizing failed: %s", e)
            return None

    @property
    def pending_proposals(self) -> list[dict]:
        return [p for p in self._proposals if p["status"] == "pending_review"]

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
        lr = data.get("last_run")
        self._last_run = parse_iso(lr) if lr else None
        self._proposals = data.get("proposals", [])
