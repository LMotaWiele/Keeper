"""Consolidator — episodic → semantic patterns, plus decay. Skipped while a session is active."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Callable

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import config
from memory.episodic import episodic, EpisodeType
from memory.semantic import semantic

log = logging.getLogger(__name__)


def _strip_fences(text: str) -> str:
    """Strip markdown code fences that LLMs sometimes wrap JSON in."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]  # drop opening fence line
        text = text.rsplit("```", 1)[0]  # drop closing fence
    return text.strip()


PATTERN_EXTRACTION_PROMPT = """\
You are a memory consolidation system. Given a batch of recent episodic memories,
extract the underlying *patterns* — things that are true in general, not tied to
a single moment.

Categories to look for:
- "trait": stable characteristics about the user (personality, skills, background)
- "preference": recurring likes, dislikes, or tendencies
- "relationship": how the user relates to people, topics, or the system
- "knowledge": domain knowledge the user has demonstrated
- "behavioral": patterns in how the user communicates or makes decisions
- "contradiction": something that contradicts a previously held pattern

For each pattern, provide:
- "content": a clear, concise statement of the pattern
- "type": one of the categories above
- "confidence": 0.0–1.0 based on how much evidence supports it
- "source_indices": which episode numbers (0-indexed) support this pattern

Output ONLY a JSON array of pattern objects. No other text.

Example output:
[
  {"content": "User is a software engineer working on AI projects", "type": "trait", "confidence": 0.9, "source_indices": [0, 3, 7]},
  {"content": "User prefers concise responses over long explanations", "type": "preference", "confidence": 0.7, "source_indices": [2, 5]}
]

Episodes to analyse:
"""

CONTRADICTION_CHECK_PROMPT = """\
You are checking new patterns against existing knowledge for contradictions.

Existing patterns:
{existing}

New patterns:
{new}

For each new pattern, determine if it contradicts any existing pattern.
Output a JSON array where each item has:
- "new_index": index of the new pattern
- "contradicts": null if no contradiction, or the content of the contradicted existing pattern
- "resolution": "keep_new", "keep_old", or "merge" with explanation

Output ONLY the JSON array.
"""


class MemoryConsolidator:
    """
    Orchestrates the episodic → semantic consolidation cycle.

    Call `run_cycle()` periodically (e.g. every 30 minutes or after
    N messages). It's designed to be idempotent — running it twice
    on the same data won't create duplicate patterns because
    SemanticMemory.store() deduplicates by content hash.
    """

    def __init__(self):
        self._llm = None
        self._running = False
        self._last_run: dict[int, datetime] = {}

    @property
    def llm(self) -> ChatAnthropic:
        if self._llm is None:
            self._llm = ChatAnthropic(
                model=config.llm_model,
                anthropic_api_key=config.anthropic_api_key,
                temperature=0.3,
                max_tokens=2048,
            )
        return self._llm

    # ── Main cycle ────────────────────────────────────────────────────────

    async def run_cycle(self, user_id: int, since_hours: int = 24) -> dict:
        """
        Run one consolidation cycle for a user.

        Returns a summary dict with counts of patterns extracted,
        episodes decayed, and episodes forgotten.
        """
        result = {
            "patterns_extracted": 0,
            "patterns_reinforced": 0,
            "episodes_decayed": 0,
            "episodes_forgotten": 0,
            "contradictions_found": 0,
            "skipped": False,
        }

        try:
            new_count = episodic.new_episodes_since_consolidation(user_id)
            tracked = user_id in episodic._new_since_consolidation
            # Skip LLM extract only when we have a live counter and it is still small.
            # After restart the counter is empty — still consolidate leftover episodes.
            if tracked and new_count < 3:
                log.debug(
                    "Skipping consolidation for user %s: only %d new episodes",
                    user_id, new_count,
                )
                result["skipped"] = True
                # Still apply decay/forgetting (cheap, no LLM call)
                result["episodes_decayed"] = await episodic.apply_decay(user_id)
                result["episodes_forgotten"] = await episodic.forget(user_id)
                return result
            # 1. Get recent episodes
            episodes = await episodic.get_for_consolidation(
                user_id, since_hours=since_hours
            )

            if len(episodes) >= 3:
                # 2. Extract patterns (need at least a few episodes)
                patterns = await self._extract_patterns(episodes)
                result["patterns_extracted"] = len(patterns)

                if patterns:
                    # 3. Check for contradictions against existing knowledge
                    contradictions = await self._check_contradictions(
                        user_id, patterns
                    )
                    result["contradictions_found"] = len(
                        [c for c in contradictions if c.get("contradicts")]
                    )

                    # 4. Store new patterns in semantic memory
                    for i, pattern in enumerate(patterns):
                        source_ids = [
                            episodes[idx]["id"]
                            for idx in pattern.get("source_indices", [])
                            if idx < len(episodes)
                        ]
                        doc_id = await semantic.store(
                            user_id=user_id,
                            content=pattern["content"],
                            pattern_type=pattern.get("type", "general"),
                            confidence=pattern.get("confidence", 0.5),
                            source_episode_ids=source_ids,
                        )

            # 5. Apply decay to all episodes
            result["episodes_decayed"] = await episodic.apply_decay(user_id)

            # 6. Forget very weak memories
            result["episodes_forgotten"] = await episodic.forget(user_id)

            self._last_run[user_id] = datetime.utcnow()

            # Mark episodes as consolidated
            episodic.mark_consolidated(user_id)            

        except Exception:
            log.exception("Consolidation cycle failed for user %s", user_id)

        return result

    # ── Pattern extraction ────────────────────────────────────────────────

    async def _extract_patterns(self, episodes: list[dict]) -> list[dict]:
        """Use the LLM to extract patterns from a batch of episodes."""
        episode_text = "\n".join(
            f"[{i}] ({ep['type']}) {ep['content']}"
            for i, ep in enumerate(episodes)
        )

        prompt = PATTERN_EXTRACTION_PROMPT + episode_text

        try:
            response = await self.llm.ainvoke([
                HumanMessage(content=prompt),
            ])
            patterns = json.loads(_strip_fences(response.content))
            if not isinstance(patterns, list):
                return []
            return patterns
        except (json.JSONDecodeError, Exception) as e:
            log.warning("Pattern extraction failed: %s", e)
            return []

    # ── Contradiction checking ────────────────────────────────────────────

    async def _check_contradictions(
        self,
        user_id: int,
        new_patterns: list[dict],
    ) -> list[dict]:
        """
        Check new patterns against existing semantic memory for contradictions.
        Handles resolution by invalidating old patterns when new evidence wins.
        """
        # Get existing patterns that might be related
        existing_patterns = []
        for pattern in new_patterns:
            related = await semantic.search(
                user_id, pattern["content"], n_results=3
            )
            for r in related:
                if r["content"] not in [e["content"] for e in existing_patterns]:
                    existing_patterns.append(r)

        if not existing_patterns:
            return []

        existing_text = "\n".join(
            f"- {p['content']} (confidence: {p['confidence']})"
            for p in existing_patterns
        )
        new_text = "\n".join(
            f"[{i}] {p['content']} (confidence: {p.get('confidence', 0.5)})"
            for i, p in enumerate(new_patterns)
        )

        prompt = CONTRADICTION_CHECK_PROMPT.format(
            existing=existing_text, new=new_text
        )

        try:
            response = await self.llm.ainvoke([
                HumanMessage(content=prompt),
            ])
            contradictions = json.loads(_strip_fences(response.content))
            if not isinstance(contradictions, list):
                return []

            # Handle resolutions
            for c in contradictions:
                if c.get("contradicts") and c.get("resolution") == "keep_new":
                    # Find and invalidate the old pattern
                    for existing in existing_patterns:
                        if existing["content"] == c["contradicts"]:
                            from memory.semantic import _content_id
                            doc_id = _content_id(existing["content"])
                            await semantic.invalidate(
                                user_id, doc_id,
                                reason=f"Contradicted by: {new_patterns[c['new_index']]['content']}"
                            )
                            break

            return contradictions
        except (json.JSONDecodeError, Exception) as e:
            log.warning("Contradiction check failed: %s", e)
            return []

    # ── Background loop ───────────────────────────────────────────────────

    async def run_loop(
        self,
        user_ids: list[int],
        interval_minutes: int = 30,
        session_check: Callable[[], bool] | None = None,
    ) -> None:
        """Periodic consolidation. `session_check` defers work during live conversation."""
        self._running = True
        log.info(
            "Consolidation loop started (interval=%dm, users=%s)",
            interval_minutes, user_ids,
        )

        while self._running:
            if session_check and session_check():
                log.debug("Consolidation deferred — active session detected")
                await asyncio.sleep(interval_minutes * 60)
                continue

            for uid in user_ids:
                # Skip if we ran recently
                last = self._last_run.get(uid)
                if last:
                    elapsed = (datetime.utcnow() - last).total_seconds() / 60
                    if elapsed < interval_minutes * 0.8:
                        continue

                result = await self.run_cycle(uid)
                if any(v > 0 for v in result.values()):
                    log.info("Consolidation for user %s: %s", uid, result)

            await asyncio.sleep(interval_minutes * 60)

    def stop(self) -> None:
        self._running = False


# Singleton
consolidator = MemoryConsolidator()
