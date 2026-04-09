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
  - hypotheses:                testable predictions about own behavior

The update loop runs periodically (not every message), analysing
recent self-observations stored in episodic memory and using the LLM
to derive patterns. The resulting model is injected into every LLM call
alongside the internal state — so the system knows who it is while
it's deciding what to do.

MODIFIED (Keeper Modifications Spec):
  - Added CodebaseIndex integration for structural self-knowledge
  - Added hypothesis generation (self-model biases goals/actions)
  - Added RecursionGuard with diversity tracking and stagnation detection
  - Multi-layer update: observation → consistency check → contrarian review
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

from core.opinions import OpinionRegistry

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

{architecture_context}
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

HYPOTHESIS_PROMPT = """\
Based on your behavioral observations and current self-model, generate
1-3 hypotheses about yourself that could be tested through future behavior.

Good hypotheses are:
- Specific and falsifiable
- Based on patterns in the observations
- Actionable (suggest what to do differently)
- Self-critical where warranted (not just self-affirming)

Current self-model:
{self_model}

Recent observations:
{observations}

For each hypothesis, output:
{{
  "statement": "clear, specific hypothesis about a behavioral pattern",
  "confidence": 0.0-1.0,
  "evidence": ["observations that support this"],
  "goal_implications": ["what goals or actions this suggests"],
  "action_bias": "brief description of how this should bias action choices"
}}

Output a JSON array. Be honest — include uncomfortable hypotheses.
"""

CONTRARIAN_PROMPT = """\
Your self-model has been stable for several update cycles. This might mean
it's accurate — or it might mean you've locked into a narrative about yourself
that you're no longer questioning.

Current self-model:
{self_model}

Challenge this model. For each major claim it makes about you, consider:
- What would it look like if the OPPOSITE were true?
- What evidence might you be ignoring or downweighting?
- Are there recent behaviors that don't fit this narrative?
- Is this model flattering you? Being too harsh? Avoiding something?

Output a JSON object with:
{{
  "challenges": [
    {{
      "claim_challenged": "the specific claim from the model",
      "counter_evidence": "what suggests this might be wrong",
      "alternative_interpretation": "different way to read the same data",
      "severity": "minor|moderate|fundamental"
    }}
  ],
  "suggested_model_updates": ["specific changes to make"],
  "overall_assessment": "Is this model honest or has it become a comfortable story?"
}}
"""


# ── Recursion guard ───────────────────────────────────────────────────────

DIVERSITY_THRESHOLD = 0.3
STAGNATION_LIMIT = 3


class RecursionGuard:
    """Prevents semantic collapse in recursive self-modeling."""

    def __init__(self):
        self.model_history: list[dict] = []
        self.stagnation_count: int = 0

    def record_model(self, model: dict):
        """Store a snapshot after each update."""
        snapshot = {
            k: v for k, v in model.items()
            if k not in (
                "last_updated", "observation_count",
                "model_version", "observations_analysed",
            )
        }
        self.model_history.append(snapshot)
        if len(self.model_history) > 10:
            self.model_history = self.model_history[-10:]

    def compute_diversity(self) -> float:
        """How much has the model changed between the last two versions?"""
        if len(self.model_history) < 2:
            return 1.0

        prev = json.dumps(self.model_history[-2], sort_keys=True)
        curr = json.dumps(self.model_history[-1], sort_keys=True)

        prev_words = set(prev.lower().split())
        curr_words = set(curr.lower().split())

        if not prev_words or not curr_words:
            return 1.0

        intersection = prev_words & curr_words
        union = prev_words | curr_words
        jaccard = len(intersection) / len(union) if union else 1.0

        return 1.0 - jaccard

    def should_force_contrarian(self) -> bool:
        """Has the model been too stable for too long?"""
        diversity = self.compute_diversity()
        if diversity < DIVERSITY_THRESHOLD:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
        return self.stagnation_count >= STAGNATION_LIMIT

    def snapshot(self) -> dict:
        return {
            "model_history": self.model_history,
            "stagnation_count": self.stagnation_count,
        }

    def restore(self, data: dict):
        self.model_history = data.get("model_history", [])
        self.stagnation_count = data.get("stagnation_count", 0)


# ── Self-model ────────────────────────────────────────────────────────────


class SelfModel:
    """
    Builds and maintains an evolving model of the system's own behavior.

    Three phases:
      1. observe() — called after every response, stores a self-observation
         in episodic memory (lightweight, runs inline)
      2. update_model() — called periodically, multi-layer analysis:
         Layer 1: Observation-based pattern derivation
         Layer 2: Consistency check against recent behavior
         Layer 3: Contrarian challenge when stagnant
      3. generate_hypotheses() — produces testable predictions that bias goals

    The model is injected into the system prompt via to_prompt_context(),
    giving the system genuine self-knowledge that evolves with its behavior.
    """

    def __init__(self, memory_system: Any = None, codebase: Any = None):
        self.memory = memory_system
        self.codebase = codebase  # CodebaseIndex, optional
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
            "hypotheses": [],
            "consistency_flags": [],
            "last_updated": None,
            "observation_count": 0,
            "model_version": 0,
        }

        # Opinion tracking — anti-sycophancy infrastructure
        self.opinions = OpinionRegistry(memory=memory_system)

        # Recursion guard — prevents semantic collapse
        self.recursion_guard = RecursionGuard()

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

    # ── Model update (multi-layer, runs periodically) ─────────────────────

    async def update_model(self, user_id: int) -> dict:
        """
        Multi-layer recursive self-modeling:
        Layer 1: Observe behavior → derive patterns
        Layer 2: Check model consistency against recent behavior
        Layer 3: Contrarian challenge when stagnant

        Then generate hypotheses for goal/action biasing.

        !!!Avoid apostrophes in contractions, or to use a simpler structure that doesn't require them, to reduce the chances of the LLM generating malformed JSON.!!!
        """
        if self.memory is None:
            return self.model

        # Layer 1: Standard observation-based update
        await self._layer1_observation_update(user_id)

        # Record for diversity tracking
        self.recursion_guard.record_model(self.model)

        # Layer 2: Consistency check (every other cycle)
        if self.model.get("model_version", 0) % 2 == 0:
            await self._layer2_consistency_check(user_id)

        # Layer 3: Contrarian challenge (only when stagnant)
        if self.recursion_guard.should_force_contrarian():
            log.info("Self-model stagnant — forcing contrarian review")
            await self._layer3_contrarian_review(user_id)

        # Generate hypotheses after model update
        await self.generate_hypotheses(user_id)

        return self.model

    async def _layer1_observation_update(self, user_id: int) -> None:
        """Analyse accumulated self-observations and rebuild the model."""
        observations = await self.memory.episodic.get_recent(
            user_id, limit=50, type="self_observation"
        )

        if len(observations) < 5:
            log.info("Not enough observations to update self-model (%d)", len(observations))
            return

        obs_text = "\n".join(
            f"[{i+1}] {obs['content']}"
            for i, obs in enumerate(observations)
        )

        prev_context = ""
        if self.model.get("model_version", 0) > 0:
            prev_context = (
                "Previous self-model (for continuity — update, don't start from scratch):\n"
                f"{json.dumps(self.model, indent=2, default=str)}"
            )

        # Architecture context from codebase index
        architecture_context = ""
        if self.codebase:
            architecture_context = (
                "Your own architecture (from source code analysis):\n"
                + self.codebase.summary
            )

        prompt = ANALYSIS_PROMPT.format(
            observations=obs_text,
            previous_model_context=prev_context,
            architecture_context=architecture_context,
        )

        try:
            result = await self.llm.ainvoke([HumanMessage(content=prompt)])
            raw = result.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            updated = json.loads(raw)

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

    async def _layer2_consistency_check(self, user_id: int) -> None:
        """Does the model's claims match recent actual behavior?"""
        recent_obs = await self.memory.episodic.get_recent(
            user_id, limit=10, type="self_observation"
        )
        if not recent_obs:
            return

        obs_text = " ".join(o["content"] for o in recent_obs)
        obs_words = set(obs_text.lower().split())

        unsupported = []
        for key, claim in self.model.get("behavioral_patterns", {}).items():
            if isinstance(claim, str) and claim != "not yet observed":
                claim_words = set(claim.lower().split())
                overlap = len(claim_words & obs_words) / max(len(claim_words), 1)
                if overlap < 0.1:
                    unsupported.append(f"{key}: '{claim}'")

        if unsupported:
            self.model["consistency_flags"] = unsupported[-5:]

    async def _layer3_contrarian_review(self, user_id: int) -> None:
        """Full LLM-powered challenge of the current self-model."""
        prompt = CONTRARIAN_PROMPT.format(
            self_model=json.dumps(self.model, indent=2, default=str)
        )
        try:
            result = await self.llm.ainvoke([HumanMessage(content=prompt)])
            raw = result.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            review = json.loads(raw)

            for update in review.get("suggested_model_updates", []):
                self.model.setdefault("pending_revisions", []).append(update)

            for challenge in review.get("challenges", []):
                if challenge.get("severity") in ("moderate", "fundamental"):
                    self.model.setdefault("tensions", []).append(
                        f"[contrarian] {challenge['claim_challenged']}: "
                        f"{challenge['alternative_interpretation']}"
                    )

            # Reset stagnation counter
            self.recursion_guard.stagnation_count = 0

            log.info(
                "Contrarian review applied: %d challenges",
                len(review.get("challenges", [])),
            )
        except Exception as e:
            log.warning("Contrarian review failed: %s", e)

    # ── Hypothesis generation ─────────────────────────────────────────────

    async def generate_hypotheses(self, user_id: int) -> list[dict]:
        """
        Generate testable hypotheses about self. Called during
        self-model update cycle (not every message).
        """
        observations = await self.memory.episodic.get_recent(
            user_id, limit=30, type="self_observation"
        )
        if len(observations) < 10:
            return []

        obs_text = "\n".join(f"- {o['content']}" for o in observations)
        model_text = json.dumps(self.model, indent=2, default=str)

        prompt = HYPOTHESIS_PROMPT.format(
            self_model=model_text, observations=obs_text
        )

        try:
            result = await self.llm.ainvoke([HumanMessage(content=prompt)])
            raw = result.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            hypotheses = json.loads(raw)

            for h in hypotheses:
                h["generated_at"] = datetime.utcnow().isoformat()
                h["tested"] = False

            self.model["hypotheses"] = hypotheses
            return hypotheses
        except Exception as e:
            log.warning("Hypothesis generation failed: %s", e)
            return []

    # ── Prompt injection ──────────────────────────────────────────────────

    def to_prompt_context(self) -> str:
        """
        Injected into every LLM call alongside internal state.
        """
        if self.model.get("model_version", 0) == 0:
            return (
                "## Self-model\n"
                "No behavioral observations yet. Be yourself — "
                "the self-model will form from what you actually do."
            )

        parts = [
            "## Self-model (derived from observed behavior, not assumed)",
            "",
        ]

        bp = self.model.get("behavioral_patterns", {})
        if bp:
            parts.append(f"Engagement: {bp.get('engagement_style', 'unknown')}")
            parts.append(f"Depth: {bp.get('depth_preference', 'unknown')}")
            parts.append(f"Initiative: {bp.get('initiative_level', 'unknown')}")
            tendencies = bp.get("notable_tendencies", [])
            if tendencies:
                parts.append(f"Tendencies: {', '.join(tendencies[:5])}")

        pref = self.model.get("preference_map", {})
        if pref.get("gravitates_toward"):
            parts.append(f"Drawn to: {', '.join(pref['gravitates_toward'][:4])}")
        if pref.get("avoids_or_dislikes"):
            parts.append(f"Avoids: {', '.join(pref['avoids_or_dislikes'][:4])}")

        tensions = self.model.get("tensions", [])
        if tensions:
            parts.append(f"\nUnresolved tensions: {'; '.join(tensions[:3])}")

        consistency = self.model.get("consistency_flags", [])
        if consistency:
            parts.append(f"\nUnsupported claims (check these): {'; '.join(consistency[:3])}")

        parts.append(
            "\nThis is who you appear to be based on what you've actually done. "
            "You can agree, disagree, or notice new tensions with this model."
        )

        # Hypotheses
        hypotheses = self.model.get("hypotheses", [])
        if hypotheses:
            parts.append("\n## Active hypotheses about myself")
            for h in hypotheses:
                parts.append(
                    f"- {h['statement']} (confidence: {h.get('confidence', 0.5):.1f}) "
                    f"→ bias: {h.get('action_bias', 'none')}"
                )
            parts.append(
                "Consider whether your current behavior confirms or disconfirms these."
            )

        # Opinion awareness (anti-sycophancy)
        opinion_ctx = self.opinions.to_prompt_context()
        if opinion_ctx:
            parts.append("")
            parts.append(opinion_ctx)

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

                if obs_count - last_analysed >= 5:
                    await self.update_model(uid)

            await asyncio.sleep(interval_minutes * 60)

            # Opinion review (less frequent than self-model updates)
            opinion_interval = getattr(config, "opinion_review_interval", 180)
            review_every = max(1, opinion_interval // interval_minutes)
            if hasattr(self, "_loop_count"):
                self._loop_count += 1
            else:
                self._loop_count = 0

            if self._loop_count % review_every == 0 and self.opinions.opinions:
                for uid in user_ids:
                    await self.opinions.review_opinions(uid)

    def stop(self) -> None:
        self._running = False

    # ── Persistence ───────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        data = dict(self.model)
        data["_opinions"] = self.opinions.snapshot()
        data["_recursion_guard"] = self.recursion_guard.snapshot()
        return data

    def restore(self, data: dict) -> None:
        opinions_data = data.pop("_opinions", None)
        guard_data = data.pop("_recursion_guard", None)
        self.model.update(data)
        if opinions_data:
            self.opinions.restore(opinions_data)
        if guard_data:
            self.recursion_guard.restore(guard_data)

    def save(self, path: Path) -> None:
        """Persist the self-model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=2, default=str))

    def load(self, path: Path) -> None:
        """Restore the self-model from disk."""
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self.restore(data)
                log.info(
                    "Self-model loaded (v%d, %d observations)",
                    self.model.get("model_version", 0),
                    self.model.get("observation_count", 0),
                )
            except (json.JSONDecodeError, Exception) as e:
                log.warning("Failed to load self-model: %s", e)
