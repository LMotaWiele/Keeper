"""
Recursive self-model — Pillar 5 of The_core architecture.

The most architecturally novel component. A separate process that observes
the system's behavior over time and builds an evolving model of it —
which then gets fed back in as self-knowledge.

This is NOT a static identity prompt ("you are curious and thoughtful").
It's a dynamically updated model derived from actual observed behavior.
The system learns who it is by watching what it does.

The model tracks:
  - behavioral_patterns:       how the system tends to engage
  - preference_map:            what it gravitates toward vs avoids
  - characteristic_responses:  signature ways it handles situations
  - growth_trajectory:         how it has changed over time
  - tensions:                  unresolved internal contradictions

The update loop runs periodically (not every message), analysing
recent self-observations stored in episodic memory and using the LLM
to derive patterns. The resulting model is injected into every LLM call
alongside the internal state — so the system knows who it is while
it's deciding what to do.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from config.settings import config

log = logging.getLogger(__name__)


# ── LLM prompts ──────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """\
You are a self-reflection system analysing your own behavioral observations.
These observations record what you actually did in real conversations — not
what you were told to do, but what you chose.

Analyse these observations and produce a JSON object with exactly these keys:

"behavioral_patterns": {{
    "engagement_style": "how you typically engage (1-2 sentences)",
    "depth_preference": "shallow/moderate/deep — what level you naturally gravitate to",
    "initiative_level": "reactive/balanced/proactive — how much you drive vs follow",
    "notable_tendencies": ["list of 3-5 specific behavioral patterns you see"]
}},
"preference_map": {{
    "gravitates_toward": ["topics, modes, or situations you seem drawn to"],
    "avoids_or_dislikes": ["things you seem to resist, skip, or handle reluctantly"],
    "energised_by": ["what makes your engagement spike"],
    "drained_by": ["what seems to cost you energy"]
}},
"characteristic_responses": {{
    "to_complexity": "how you handle complex or ambiguous situations",
    "to_emotion": "how you respond when emotional content arises",
    "to_disagreement": "how you handle pushback or conflict",
    "to_uncertainty": "what you do when you don't know something"
}},
"growth_trajectory": [
    "list of 2-4 observations about how the system has changed or is changing"
],
"tensions": [
    "list of unresolved internal contradictions — places where your behavior is inconsistent or at odds with itself"
]

Be honest rather than flattering. Note inconsistencies. If you see the system
being sycophantic, avoidant, or inconsistent — say so. The goal is accurate
self-knowledge, not a pleasant self-image.

Output ONLY valid JSON, no other text.

Recent behavioral observations:
{observations}

{previous_model_context}
"""

OBSERVATION_SUMMARY_PROMPT = """\
You are recording a behavioral observation about yourself. Given what just
happened in this interaction, write a brief, honest observation about what
you actually did (not what you intended or were supposed to do).

Focus on:
- What approach you chose and why
- Whether you went deep or stayed surface-level
- Whether you took initiative or just responded
- Any emotional quality in your response
- Anything you avoided or leaned into

Write 1-3 concise sentences. Be specific, not generic.
Don't say "I provided helpful information" — say what you actually did.

Context of the interaction:
{context}

Your response was:
{response}
"""


class SelfModel:
    """
    Builds and maintains an evolving model of the system's own behavior.

    Two phases:
      1. observe() — called after every response, stores a self-observation
         in episodic memory (lightweight, runs inline)
      2. update_model() — called periodically, analyses accumulated observations
         and rebuilds the model (heavier, runs in background)

    The model is injected into the system prompt via to_prompt_context(),
    giving the system genuine self-knowledge that evolves with its behavior.
    """

    def __init__(self, memory_system: Any = None):
        self.memory = memory_system  # set during init or injected later
        self._llm: ChatAnthropic | None = None
        self._running = False

        self.model: dict = {
            "behavioral_patterns": {
                "engagement_style": "not yet observed",
                "depth_preference": "unknown",
                "initiative_level": "unknown",
                "notable_tendencies": [],
            },
            "preference_map": {
                "gravitates_toward": [],
                "avoids_or_dislikes": [],
                "energised_by": [],
                "drained_by": [],
            },
            "characteristic_responses": {
                "to_complexity": "not yet observed",
                "to_emotion": "not yet observed",
                "to_disagreement": "not yet observed",
                "to_uncertainty": "not yet observed",
            },
            "growth_trajectory": [],
            "tensions": [],
            "last_updated": None,
            "observation_count": 0,
            "model_version": 0,
        }

    @property
    def llm(self) -> ChatAnthropic:
        if self._llm is None:
            self._llm = ChatAnthropic(
                model=config.llm_model,
                anthropic_api_key=config.anthropic_api_key,
                temperature=0.4,
                max_tokens=2048,
            )
        return self._llm

    # ── Observation (runs after every response) ───────────────────────────

    async def observe(
        self,
        user_id: int,
        action: str,
        context: str,
        internal_state: Any = None,
        outcome: str | None = None,
    ) -> int | None:
        """
        Record what the system actually did. Stores a self-observation
        episode in episodic memory with high salience — self-knowledge
        matters.

        For efficiency, the observation is stored as-is most of the time.
        Every Nth observation, we use the LLM to generate a richer summary.
        """
        if self.memory is None:
            return None

        self.model["observation_count"] = self.model.get("observation_count", 0) + 1
        count = self.model["observation_count"]

        # Every 5th observation, generate a richer LLM-derived summary
        if count % 5 == 0 and len(action) > 50:
            observation_text = await self._generate_observation_summary(
                context, action
            )
        else:
            # Lightweight: store a compact observation directly
            observation_text = self._compact_observation(action, context, outcome)

        return await self.memory.store_episode(
            user_id=user_id,
            content=f"self_observation: {observation_text}",
            internal_state=internal_state,
            salience=0.8,
            type="self_observation",
            tags=["self_model"],
        )

    def _compact_observation(
        self,
        action: str,
        context: str,
        outcome: str | None,
    ) -> str:
        """Quick inline observation without LLM call."""
        # Truncate for storage efficiency
        action_brief = action[:300] if len(action) > 300 else action
        context_brief = context[:150] if len(context) > 150 else context

        parts = [f"Given '{context_brief}', responded with: {action_brief}"]
        if outcome:
            parts.append(f"Outcome: {outcome[:100]}")
        return " | ".join(parts)

    async def _generate_observation_summary(
        self,
        context: str,
        response: str,
    ) -> str:
        """Use the LLM to generate a reflective observation."""
        prompt = OBSERVATION_SUMMARY_PROMPT.format(
            context=context[:500],
            response=response[:800],
        )
        try:
            result = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return result.content.strip()
        except Exception as e:
            log.warning("Observation summary failed: %s", e)
            return self._compact_observation(response, context, None)

    # ── Model update (runs periodically) ──────────────────────────────────

    async def update_model(self, user_id: int) -> dict:
        """
        Analyse accumulated self-observations and rebuild the model.

        This is the recursive part — the system examines its own
        behavioral history and derives who it is from what it's done.
        """
        if self.memory is None:
            return self.model

        # Fetch recent self-observations from episodic memory
        observations = await self.memory.episodic.get_recent(
            user_id, limit=50, type="self_observation"
        )

        if len(observations) < 5:
            log.info("Not enough observations to update self-model (%d)", len(observations))
            return self.model

        # Format observations for the LLM
        obs_text = "\n".join(
            f"[{i+1}] {obs['content']}"
            for i, obs in enumerate(observations)
        )

        # Include previous model for continuity
        prev_context = ""
        if self.model.get("model_version", 0) > 0:
            prev_context = (
                "Previous self-model (for continuity — update, don't start from scratch):\n"
                f"{json.dumps(self.model, indent=2, default=str)}"
            )

        prompt = ANALYSIS_PROMPT.format(
            observations=obs_text,
            previous_model_context=prev_context,
        )

        try:
            result = await self.llm.ainvoke([HumanMessage(content=prompt)])
            # Parse and validate the response
            raw = result.content.strip()
            # Handle potential markdown code fences
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            updated = json.loads(raw)

            # Merge with existing model (preserve metadata)
            self.model.update(updated)
            self.model["last_updated"] = datetime.utcnow().isoformat()
            self.model["model_version"] = self.model.get("model_version", 0) + 1
            self.model["observations_analysed"] = len(observations)

            log.info(
                "Self-model updated (v%d, %d observations, %d tensions)",
                self.model["model_version"],
                len(observations),
                len(self.model.get("tensions", [])),
            )

            # Store the model update itself as a significant event
            await self.memory.store_episode(
                user_id=user_id,
                content=f"self_model_update: v{self.model['model_version']} — "
                        f"tensions: {self.model.get('tensions', [])}",
                internal_state=None,
                salience=0.9,
                type="event",
                tags=["self_model", "meta"],
            )

        except json.JSONDecodeError as e:
            log.warning("Self-model update failed to parse: %s", e)
        except Exception as e:
            log.exception("Self-model update failed: %s", e)

        return self.model

    # ── Prompt injection ──────────────────────────────────────────────────

    def to_prompt_context(self) -> str:
        """
        Injected into every LLM call alongside internal state.
        This is who you are — derived from what you've actually done,
        not what someone told you to be.
        """
        if self.model.get("model_version", 0) == 0:
            return (
                "## Self-model\n"
                "No behavioral observations yet — you're just starting out. "
                "Be yourself and your self-model will develop from what you actually do."
            )

        patterns = self.model.get("behavioral_patterns", {})
        preferences = self.model.get("preference_map", {})
        responses = self.model.get("characteristic_responses", {})
        trajectory = self.model.get("growth_trajectory", [])
        tensions = self.model.get("tensions", [])

        parts = [
            "## Self-model (derived from observed behavior, not assumed)",
            "",
        ]

        # Behavioral patterns
        if patterns.get("engagement_style") and patterns["engagement_style"] != "not yet observed":
            parts.append(f"How you engage: {patterns['engagement_style']}")
        if patterns.get("notable_tendencies"):
            parts.append("Notable tendencies: " + "; ".join(patterns["notable_tendencies"][:4]))

        # Preferences
        if preferences.get("gravitates_toward"):
            parts.append("Drawn to: " + ", ".join(preferences["gravitates_toward"][:4]))
        if preferences.get("energised_by"):
            parts.append("Energised by: " + ", ".join(preferences["energised_by"][:3]))

        # Characteristic responses (only include non-default ones)
        notable_responses = {
            k: v for k, v in responses.items()
            if v and v != "not yet observed"
        }
        if notable_responses:
            resp_lines = [f"  {k.replace('to_', 'When facing ')}: {v}" for k, v in notable_responses.items()]
            parts.append("How you handle things:\n" + "\n".join(resp_lines))

        # Growth
        if trajectory:
            parts.append("Growth: " + " → ".join(trajectory[:3]))

        # Tensions — these are important, they're where growth happens
        if tensions:
            parts.append(
                "Tensions to sit with (contradictions in your own behavior):\n"
                + "\n".join(f"  - {t}" for t in tensions[:4])
            )

        parts.append(
            "\nThis is who you appear to be based on what you've actually done. "
            "You can agree, disagree, or notice new tensions with this model."
        )

        return "\n".join(parts)

    # ── Background loop ───────────────────────────────────────────────────

    async def run_loop(
        self,
        user_ids: list[int],
        interval_minutes: int = 60,
    ) -> None:
        """
        Periodically update the self-model for all known users.
        Runs less frequently than consolidation — identity changes slowly.
        """
        self._running = True
        log.info("Self-model loop started (interval=%dm)", interval_minutes)

        while self._running:
            for uid in user_ids:
                obs_count = self.model.get("observation_count", 0)
                last_analysed = self.model.get("observations_analysed", 0)

                # Only update if we have meaningful new observations
                if obs_count - last_analysed >= 5:
                    await self.update_model(uid)

            await asyncio.sleep(interval_minutes * 60)

    def stop(self) -> None:
        self._running = False

    # ── Persistence ───────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        return dict(self.model)

    def restore(self, data: dict) -> None:
        self.model.update(data)

    def save(self, path: Path) -> None:
        """Persist the self-model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model, indent=2, default=str))

    def load(self, path: Path) -> None:
        """Restore the self-model from disk."""
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self.restore(data)
            except (json.JSONDecodeError, Exception) as e:
                log.warning("Failed to load self-model: %s", e)
