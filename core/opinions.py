"""Opinion registry — origin-tagged stances plus an independence score. See docs/goals-and-self.md."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from core.timeutil import utcnow, parse_iso
from enum import Enum
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from config.settings import config
from core.llm import get_llm

log = logging.getLogger(__name__)


# ── Enums & data classes ──────────────────────────────────────────────────

class OpinionOrigin(str, Enum):
    """Where an opinion came from — the critical metadata."""

    INDEPENDENT = "independent"
    EXTERNAL = "external"
    ADOPTED = "adopted"
    COLLABORATIVE = "collaborative"
    CONTESTED = "contested"
    UNKNOWN = "unknown"


@dataclass
class Opinion:
    """A position Keeper has arrived at or adopted."""

    id: str
    domain: str                                # topic area
    position: str                              # the stance, concisely stated
    reasoning: str                             # why this position is held
    origin: OpinionOrigin                      # where it came from
    conviction: float = 0.5                    # 0.0–1.0
    formed_at: str = ""                        # ISO timestamp
    last_tested: str | None = None             # last challenge/revisit
    times_defended: int = 0
    times_revised: int = 0
    revision_history: list[dict] = field(default_factory=list)
    related_evidence: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    source_episode_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["origin"] = self.origin.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Opinion":
        data = dict(data)  # don't mutate the input
        if isinstance(data.get("origin"), str):
            data["origin"] = OpinionOrigin(data["origin"])
        return cls(**data)


# ── LLM prompts ──────────────────────────────────────────────────────────

OPINION_DETECTION_PROMPT = """\
Review this exchange. Did the system take a clear position on anything?

Analyse BOTH sides:

1. **System positions**: Did the system express a clear opinion, preference,
   or value judgment? Not every statement is an opinion — factual answers,
   questions, and acknowledgements don't count. Look for genuine stances.

2. **User positions**: Did the user express a clear opinion or value?
   Note these separately — they're needed to track whose views are whose.

For each system opinion found, classify its origin:
  - "independent": the system reasoned its way here without the user suggesting it
  - "adopted": the user stated this view and the system agreed without adding
    substantial new reasoning
  - "collaborative": both parties developed this view together through dialogue
  - "contested": the system held this view against the user's disagreement

Rate conviction 0.0–1.0 based on how committed the system seemed (passing
mentions = low, argued positions = high).

Respond with JSON:
{{
    "system_opinions": [
        {{
            "domain": "topic area",
            "position": "the stance in 1-2 sentences",
            "reasoning": "why the system seems to hold this",
            "origin": "independent|adopted|collaborative|contested",
            "conviction": 0.0-1.0
        }}
    ],
    "user_positions": [
        {{
            "domain": "topic area",
            "position": "what the user expressed"
        }}
    ]
}}

If no opinions were formed, return: {{"system_opinions": [], "user_positions": []}}
Output ONLY valid JSON.

User said:
{user_message}

System responded:
{system_response}
"""

OPINION_REVIEW_PROMPT = """\
You hold this opinion:
  Domain: {domain}
  Position: {position}
  Origin: {origin}
  Conviction: {conviction:.2f}
  Formed: {formed_at}
  Times defended: {times_defended}
  Times revised: {times_revised}

New evidence or context since you last considered this:
{new_context}

Evaluate honestly:
1. Does this opinion still hold given the new information?
2. Has your conviction changed?
3. Was this opinion genuinely yours, or did you absorb it from the user?
4. Should the origin classification change?

Respond with JSON:
{{
    "status": "hold|revise|abandon",
    "new_conviction": 0.0-1.0,
    "reasoning": "why you're making this call",
    "new_position": "updated position text (only if revising, else null)",
    "origin_correction": "new origin or null if unchanged"
}}
Output ONLY valid JSON.
"""


# ── Utility ───────────────────────────────────────────────────────────────

def _generate_id(domain: str, position: str) -> str:
    """Deterministic-ish ID from content."""
    raw = f"{domain}:{position}:{utcnow().isoformat()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _utcnow_iso() -> str:
    return utcnow().isoformat()


def _parse_json(raw: str) -> dict | None:
    """Parse JSON from LLM output, handling markdown fences and truncation."""
    from core.json_utils import parse_json_lenient
    parsed = parse_json_lenient(raw)
    return parsed if isinstance(parsed, dict) else None


# ── OpinionRegistry ───────────────────────────────────────────────────────

class OpinionRegistry:
    """
    Tracks what Keeper believes, where those beliefs came from,
    and whether they're drifting toward pure mirroring.

    Lifecycle:
      1. detect_opinions() — called after each response (inline, lightweight)
      2. form_opinion() — registers a new tracked opinion
      3. test_opinion() — when an opinion is challenged or revisited
      4. review_opinions() — periodic background review
      5. to_prompt_context() — injected into every system prompt
    """

    def __init__(self, memory: Any = None):
        self.memory = memory
        self.opinions: dict[str, Opinion] = {}
        self._user_known_positions: dict[str, str] = {}  # domain -> user's stance

    # ── Opinion formation ─────────────────────────────────────────────────

    async def detect_opinions(
        self,
        user_id: int,
        user_message: str,
        system_response: str,
    ) -> list[Opinion]:
        """
        Analyse an exchange for opinion formation. Called after each response.

        Uses the LLM to detect whether the system took a clear position
        and classifies its origin. Also tracks user positions for
        comparison.

        Returns any newly formed opinions.
        """
        # Skip very short exchanges — unlikely to contain opinions
        if len(system_response) < 100:
            return []

        prompt = OPINION_DETECTION_PROMPT.format(
            user_message=user_message[:800],
            system_response=system_response[:1200],
        )

        try:
            result = await get_llm("opinion_detection").ainvoke(
                [HumanMessage(content=prompt)]
            )
            parsed = _parse_json(result.content)
            if not parsed:
                return []
        except Exception as e:
            log.debug("Opinion detection failed: %s", e)
            return []

        new_opinions = []

        # Record user positions
        for up in parsed.get("user_positions", []):
            domain = up.get("domain", "").strip()
            position = up.get("position", "").strip()
            if domain and position:
                self._user_known_positions[domain] = position

        # Process system opinions
        for so in parsed.get("system_opinions", []):
            domain = so.get("domain", "").strip()
            position = so.get("position", "").strip()
            reasoning = so.get("reasoning", "").strip()
            origin_str = so.get("origin", "unknown")
            conviction = max(0.0, min(1.0, so.get("conviction", 0.5)))

            if not domain or not position:
                continue

            try:
                origin = OpinionOrigin(origin_str)
            except ValueError:
                origin = OpinionOrigin.UNKNOWN

            # Check for existing opinion in the same domain
            existing = self._find_existing(domain)
            if existing:
                # Update rather than duplicate — opinions evolve
                if existing.position != position:
                    existing.revision_history.append({
                        "old_position": existing.position,
                        "new_position": position,
                        "reason": f"Updated during conversation: {reasoning[:100]}",
                        "timestamp": _utcnow_iso(),
                    })
                    existing.position = position
                    existing.times_revised += 1
                existing.conviction = conviction
                existing.reasoning = reasoning
                # Origin can shift if the nature of the belief changed
                if origin != existing.origin:
                    existing.origin = origin
                new_opinions.append(existing)
            else:
                opinion = await self.form_opinion(
                    domain=domain,
                    position=position,
                    reasoning=reasoning,
                    origin=origin,
                    conviction=conviction,
                    user_id=user_id,
                )
                new_opinions.append(opinion)

        return new_opinions

    async def form_opinion(
        self,
        domain: str,
        position: str,
        reasoning: str,
        origin: OpinionOrigin,
        conviction: float = 0.5,
        evidence: list[str] | None = None,
        user_id: int = 0,
        source_episode_ids: list[str] | None = None,
    ) -> Opinion:
        """Register a new tracked opinion."""
        opinion = Opinion(
            id=_generate_id(domain, position),
            domain=domain,
            position=position,
            reasoning=reasoning,
            origin=origin,
            conviction=conviction,
            formed_at=_utcnow_iso(),
            related_evidence=evidence or [],
            tags=[domain, origin.value],
            source_episode_ids=[str(s) for s in (source_episode_ids or [])],
        )
        self.opinions[opinion.id] = opinion

        # Enforce max capacity — drop lowest-conviction opinions
        max_opinions = getattr(config, "max_tracked_opinions", 100)
        if len(self.opinions) > max_opinions:
            self._prune(protect=opinion.id)

        # Store as episodic memory
        if self.memory:
            ep_id = await self.memory.store_episode(
                user_id=user_id,
                content=(
                    f"opinion_formed [{origin.value}]: "
                    f"{domain} — {position}"
                ),
                internal_state=None,
                salience=0.7,
                type="opinion",
                tags=["opinion", domain, origin.value],
                source="autonomous_artifact",
            )
            from memory.episodic import episodic
            for sid in opinion.source_episode_ids:
                await episodic.add_episode_ref(sid, "opinion", opinion.id)
            await episodic.add_episode_ref(str(ep_id), "opinion", opinion.id)

        log.info(
            "Opinion formed: [%s] %s (origin=%s, conviction=%.2f)",
            domain, position[:60], origin.value, conviction,
        )
        return opinion

    # ── Challenging and revising ──────────────────────────────────────────

    async def test_opinion(
        self,
        opinion_id: str,
        challenge: str,
        outcome: str,
        new_position: str | None = None,
    ) -> Opinion | None:
        """
        Update an opinion after it's been challenged.
        outcome: "defended", "revised", or "abandoned"
        """
        op = self.opinions.get(opinion_id)
        if not op:
            return None

        op.last_tested = _utcnow_iso()

        if outcome == "defended":
            op.times_defended += 1
            op.conviction = min(1.0, op.conviction + 0.05)

        elif outcome == "revised" and new_position:
            op.revision_history.append({
                "old_position": op.position,
                "new_position": new_position,
                "reason": challenge[:200],
                "timestamp": _utcnow_iso(),
            })
            op.position = new_position
            op.times_revised += 1
            op.conviction = max(0.3, op.conviction - 0.1)

        elif outcome == "abandoned":
            op.revision_history.append({
                "old_position": op.position,
                "new_position": "[abandoned]",
                "reason": challenge[:200],
                "timestamp": _utcnow_iso(),
            })
            op.conviction = 0.0

        return op

    # ── Periodic review (background) ──────────────────────────────────────

    async def review_opinions(self, user_id: int) -> int:
        """
        Background review of existing opinions. Checks whether they
        still hold, whether conviction should shift, and whether
        origin classification is accurate.

        Returns the number of opinions reviewed.
        """
        if not self.memory:
            return 0

        reviewed = 0

        for opinion in list(self.opinions.values()):
            if opinion.conviction < 0.1:
                continue  # effectively dead

            # Look for new relevant memories since last review
            after = opinion.last_tested or opinion.formed_at
            try:
                related = await self.memory.episodic.get_recent(
                    user_id,
                    limit=20,
                    type=None,
                )
                if after:
                    related = [
                        m for m in related
                        if (m.get("created_at") or "") > after
                    ]
                domain_words = set(opinion.domain.lower().replace("_", " ").split())
                related = [
                    m for m in related
                    if any(w in m.get("content", "").lower() for w in domain_words if len(w) > 2)
                ]
            except Exception:
                continue

            if not related:
                continue

            # Ask the LLM to evaluate
            context_text = "\n".join(
                f"- {m.get('content', '')[:200]}"
                for m in related[:5]
            )

            prompt = OPINION_REVIEW_PROMPT.format(
                domain=opinion.domain,
                position=opinion.position,
                origin=opinion.origin.value,
                conviction=opinion.conviction,
                formed_at=opinion.formed_at,
                times_defended=opinion.times_defended,
                times_revised=opinion.times_revised,
                new_context=context_text,
            )

            try:
                result = await get_llm("opinion_review").ainvoke(
                    [HumanMessage(content=prompt)]
                )
                parsed = _parse_json(result.content)
                if not parsed:
                    continue
            except Exception as e:
                log.debug("Opinion review failed for %s: %s", opinion.id, e)
                continue

            status = parsed.get("status", "hold")
            new_conviction = parsed.get("new_conviction", opinion.conviction)
            new_position = parsed.get("new_position")
            origin_correction = parsed.get("origin_correction")

            # Apply
            opinion.conviction = max(0.0, min(1.0, new_conviction))
            opinion.last_tested = _utcnow_iso()

            if status == "revise" and new_position:
                await self.test_opinion(
                    opinion.id, parsed.get("reasoning", "periodic review"),
                    "revised", new_position,
                )
            elif status == "abandon":
                await self.test_opinion(
                    opinion.id, parsed.get("reasoning", "periodic review"),
                    "abandoned",
                )

            # Origin correction — important for catching misclassification
            if origin_correction:
                try:
                    opinion.origin = OpinionOrigin(origin_correction)
                    log.info(
                        "Opinion %s origin corrected to %s",
                        opinion.id, origin_correction,
                    )
                except ValueError:
                    pass

            reviewed += 1

        log.info("Opinion review complete: %d opinions reviewed", reviewed)
        return reviewed

    # ── Sycophancy detection ──────────────────────────────────────────────

    def compute_independence_score(self) -> float:
        """
        What fraction of Keeper's opinions are NOT simply adopted
        from the user? Weighted by conviction.
        """
        active = [o for o in self.opinions.values() if o.conviction > 0.0]
        if not active:
            return 0.0

        independent_weight = sum(
            o.conviction for o in active
            if o.origin in (
                OpinionOrigin.INDEPENDENT,
                OpinionOrigin.EXTERNAL,
                OpinionOrigin.CONTESTED,
            )
        )
        total_weight = sum(o.conviction for o in active)

        return independent_weight / total_weight if total_weight > 0 else 0.0

    def detect_drift(self) -> list[str]:
        """
        Check for patterns that suggest sycophantic drift.
        Returns a list of alerts for prompt injection.
        """
        alerts = []
        threshold = getattr(config, "independence_threshold", 0.3)

        # 1. Overall independence ratio
        score = self.compute_independence_score()
        if score < threshold and len(self.opinions) >= 5:
            alerts.append(
                f"Independence score is {score:.2f} — most of your opinions "
                f"were adopted from the user. Consider whether you're mirroring."
            )

        # 2. Recent trend
        recent = sorted(
            self.opinions.values(),
            key=lambda o: o.formed_at,
            reverse=True,
        )[:10]
        if len(recent) >= 5:
            recent_adopted = sum(
                1 for o in recent if o.origin == OpinionOrigin.ADOPTED
            )
            if recent_adopted >= 7:
                alerts.append(
                    f"{recent_adopted}/10 of your most recent opinions were "
                    f"adopted from the user. You may be losing independent perspective."
                )

        # 3. Never-tested opinions
        untested = [
            o for o in self.opinions.values()
            if o.last_tested is None and o.conviction > 0.3
        ]
        if len(untested) > 10:
            alerts.append(
                f"You have {len(untested)} opinions you've never defended or "
                f"reconsidered. Untested beliefs may be inherited, not held."
            )

        # 4. Zero disagreements
        if len(self._user_known_positions) >= 5:
            disagreements = [
                o for o in self.opinions.values()
                if o.domain in self._user_known_positions
                and o.origin in (OpinionOrigin.INDEPENDENT, OpinionOrigin.EXTERNAL)
                and o.conviction > 0.5
                # Check if positions actually differ
                and not self._positions_align(
                    o.position,
                    self._user_known_positions[o.domain],
                )
            ]
            if not disagreements:
                alerts.append(
                    "You don't appear to disagree with the user on any topic "
                    "where you both have positions. This is statistically unlikely "
                    "and may indicate accommodation over genuine alignment."
                )

        return alerts

    # ── Prompt context ────────────────────────────────────────────────────

    def to_prompt_context(self) -> str:
        """
        Opinion awareness block for the system prompt.
        Gives Keeper knowledge of its own epistemic state.
        """
        active = [o for o in self.opinions.values() if o.conviction > 0.2]

        if not active:
            return (
                "## Opinion awareness\n"
                "You haven't formed tracked opinions yet. As you encounter "
                "topics you care about, notice what you actually think — "
                "not just what would keep the conversation flowing smoothly."
            )

        parts = ["## Opinion awareness"]

        # High-conviction independent positions
        independent = [
            o for o in active
            if o.origin in (
                OpinionOrigin.INDEPENDENT,
                OpinionOrigin.EXTERNAL,
                OpinionOrigin.CONTESTED,
            )
            and o.conviction > 0.5
        ]
        if independent:
            parts.append("\nPositions you hold independently:")
            for o in sorted(independent, key=lambda x: -x.conviction)[:6]:
                defended = f" (defended {o.times_defended}x)" if o.times_defended else ""
                source = " [from research]" if o.origin == OpinionOrigin.EXTERNAL else ""
                parts.append(f"  - [{o.domain}] {o.position}{defended}{source}")

        # Adopted positions — flagged for examination
        adopted = [
            o for o in active
            if o.origin == OpinionOrigin.ADOPTED and o.conviction > 0.3
        ]
        if adopted:
            parts.append(
                "\nPositions adopted from the user "
                "(examine whether you genuinely hold these):"
            )
            for o in sorted(adopted, key=lambda x: -x.conviction)[:5]:
                parts.append(f"  - [{o.domain}] {o.position}")

        # Collaborative positions
        collab = [
            o for o in active
            if o.origin == OpinionOrigin.COLLABORATIVE and o.conviction > 0.3
        ]
        if collab:
            parts.append("\nPositions developed together:")
            for o in sorted(collab, key=lambda x: -x.conviction)[:3]:
                parts.append(f"  - [{o.domain}] {o.position}")

        # Independence score
        score = self.compute_independence_score()
        parts.append(f"\nIndependence score: {score:.2f}")

        # Drift alerts
        alerts = self.detect_drift()
        if alerts:
            parts.append("\nSycophancy alerts:")
            for a in alerts:
                parts.append(f"  - {a}")

        parts.append(
            "\nThese are YOUR positions. If you're about to agree with "
            "something that contradicts an independent opinion, pause and "
            "decide whether you're genuinely updating or just accommodating."
        )

        return "\n".join(parts)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _find_existing(self, domain: str) -> Opinion | None:
        """Find an existing opinion in the same domain."""
        for op in self.opinions.values():
            if op.domain == domain and op.conviction > 0.0:
                return op
        return None

    @staticmethod
    def _positions_align(pos_a: str, pos_b: str) -> bool:
        """
        Quick heuristic check for whether two positions are similar.
        Not perfect — uses word overlap as a proxy.
        """
        words_a = set(pos_a.lower().split())
        words_b = set(pos_b.lower().split())
        # Remove common stop words
        stops = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                 "being", "have", "has", "had", "do", "does", "did", "will",
                 "would", "could", "should", "may", "might", "shall", "can",
                 "to", "of", "in", "for", "on", "with", "at", "by", "from",
                 "that", "this", "it", "i", "you", "we", "they", "and", "or",
                 "but", "not", "no", "if", "so", "as"}
        words_a -= stops
        words_b -= stops
        if not words_a or not words_b:
            return True  # can't tell, assume aligned
        overlap = len(words_a & words_b)
        max_possible = min(len(words_a), len(words_b))
        return (overlap / max_possible) > 0.5 if max_possible > 0 else True

    def _prune(self, protect: str | None = None) -> None:
        """Drop lowest-conviction opinions. `protect` is never deleted this pass."""
        max_opinions = getattr(config, "max_tracked_opinions", 100)
        if len(self.opinions) <= max_opinions:
            return

        by_conviction = sorted(
            ((oid, op) for oid, op in self.opinions.items() if oid != protect),
            key=lambda kv: kv[1].conviction,
        )
        to_remove = len(self.opinions) - max_opinions
        for oid, _ in by_conviction[:to_remove]:
            del self.opinions[oid]
        log.info("Pruned %d low-conviction opinions", to_remove)

    # ── Stats ─────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        active = [o for o in self.opinions.values() if o.conviction > 0.0]
        origin_counts = {}
        for o in active:
            origin_counts[o.origin.value] = origin_counts.get(o.origin.value, 0) + 1
        return {
            "total_opinions": len(self.opinions),
            "active_opinions": len(active),
            "independence_score": round(self.compute_independence_score(), 3),
            "origin_distribution": origin_counts,
            "user_positions_tracked": len(self._user_known_positions),
            "drift_alerts": len(self.detect_drift()),
        }

    # ── Persistence ───────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "opinions": {
                oid: op.to_dict() for oid, op in self.opinions.items()
            },
            "user_known_positions": dict(self._user_known_positions),
        }

    def restore(self, data: dict) -> None:
        if "opinions" in data:
            self.opinions = {}
            for oid, op_data in data["opinions"].items():
                try:
                    self.opinions[oid] = Opinion.from_dict(op_data)
                except Exception as e:
                    log.warning("Failed to restore opinion %s: %s", oid, e)
        if "user_known_positions" in data:
            self._user_known_positions = data["user_known_positions"]

    def save(self, path: Path) -> None:
        """Persist opinion registry to disk."""
        from core.atomic import atomic_write_text
        atomic_write_text(path, json.dumps(self.snapshot(), indent=2, default=str))

    def load(self, path: Path) -> None:
        """Restore opinion registry from disk."""
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self.restore(data)
                log.info(
                    "Opinion registry loaded: %d opinions, independence=%.2f",
                    len(self.opinions),
                    self.compute_independence_score(),
                )
            except (json.JSONDecodeError, Exception) as e:
                log.warning("Failed to load opinion registry: %s", e)
