"""Future simulation — gated 'what if' before high-stakes autonomous actions."""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage

from core.llm import get_llm

log = logging.getLogger(__name__)

SIMULATE_PROMPT = """\
You are simulating the likely outcomes of a planned action before executing it.

## Planned Action
{action_description}

## Current Context
- Conversation state: {conversation_context}
- Internal state: {internal_state}
- Relevant memory: {memory_context}
- Active goals: {goals_context}

Simulate 2-3 likely outcomes of this action:

For each outcome, assess:
1. Probability (0.0-1.0)
2. Impact on user relationship (positive/neutral/negative)
3. Impact on active goals (advances/neutral/hinders)
4. Emotional trajectory (how internal state would shift)
5. Risk level (low/medium/high)

Then recommend: proceed, modify, or abandon — with reasoning.

Output as JSON:
{{
  "outcomes": [
    {{
      "description": "...",
      "probability": 0.0,
      "user_impact": "positive|neutral|negative",
      "goal_impact": "advances|neutral|hinders",
      "emotional_shift": "...",
      "risk": "low|medium|high"
    }}
  ],
  "recommendation": "proceed|modify|abandon",
  "reasoning": "...",
  "suggested_modification": null
}}
"""


class FutureSimulator:
    """
    Simulates action outcomes before execution.

    Only invoked when action importance exceeds threshold — most
    conversational responses skip this entirely.
    """

    IMPORTANCE_THRESHOLD = 0.6

    def __init__(self):
        pass

    async def should_simulate(
        self, action_description: str, internal_state: Any
    ) -> bool:
        """
        Is this action important enough to simulate?

        Jev noul replaces the keyword/affect threshold when available.
        Fail open to the heuristic if Jev is down.
        """
        importance = 0.3

        if "[autonomous]" in action_description.lower():
            importance += 0.3

        if internal_state and internal_state.valence < 0.3:
            importance += 0.2

        if internal_state and internal_state.arousal > 0.8:
            importance += 0.15

        heuristic = importance >= self.IMPORTANCE_THRESHOLD
        try:
            from core.jev import jev_decide, noul_allows, spec
            answers = await jev_decide(
                {
                    "action": (action_description or "")[:1500],
                    "valence": getattr(internal_state, "valence", None),
                    "arousal": getattr(internal_state, "arousal", None),
                    "autonomous": "[autonomous]" in (action_description or "").lower(),
                },
                {"high_stakes_enough": spec("high_stakes_enough")},
                background=True,
                task="future_simulate_gate",
            )
            if not answers:
                return heuristic
            return noul_allows(answers.get("high_stakes_enough"), 0.50)
        except Exception:
            log.debug("should_simulate Jev failed — using heuristic", exc_info=True)
            return heuristic

    async def simulate(
        self,
        action_description: str,
        conversation_context: str = "",
        internal_state: Any = None,
        memory_context: str = "",
        goals_context: str = "",
    ) -> dict | None:
        """Run a simulation. Returns recommendation dict or None on failure."""
        prompt = SIMULATE_PROMPT.format(
            action_description=action_description,
            conversation_context=conversation_context[:500],
            internal_state=(
                internal_state.to_prompt_context() if internal_state else "N/A"
            ),
            memory_context=memory_context[:500],
            goals_context=goals_context[:300],
        )

        try:
            from core.json_utils import parse_json_lenient
            result = await get_llm("future_simulate").ainvoke(
                [HumanMessage(content=prompt)]
            )
            parsed = parse_json_lenient(result.content)
            return parsed if isinstance(parsed, dict) else None
        except Exception as e:
            log.warning("Future simulation failed: %s", e)
            return None
