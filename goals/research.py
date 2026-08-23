"""Research engine — web search to EXTERNAL opinions so pushback is not just mirroring."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from config.settings import config
from core.opinions import OpinionRegistry, OpinionOrigin

log = logging.getLogger(__name__)


# ── Prompts ───────────────────────────────────────────────────────────────

QUERY_GENERATION_PROMPT = """\
You are generating web search queries to research a topic and form
an independent opinion about it.

Research topic: {topic}
Context (why this matters): {context}

Generate 2-3 search queries that would give you diverse perspectives
on this topic. Include at least one query that might surface a
contrarian or minority view — not just the mainstream position.

Output ONLY a JSON array of query strings:
["query 1", "query 2", "query 3"]
"""

SYNTHESIS_PROMPT = """\
You've researched a topic and found the following information from
web searches. Now synthesize what you've found into a clear position.

Research topic: {topic}
Why you're researching this: {context}

Search results:
{search_results}

Based on these sources, form your own position. Be specific and honest:
- What do the sources generally agree on?
- Where do they disagree?
- What's YOUR take, having read all of this?
- How confident are you in this position?

Important: This is YOUR opinion formed from external evidence. Don't
default to the safest or most diplomatic position — actually think
about what the evidence suggests and take a genuine stance.

Output ONLY a JSON object:
{{
    "domain": "topic area (2-4 words, lowercase, underscores)",
    "position": "your synthesized position (1-3 sentences)",
    "reasoning": "why you hold this position based on the evidence (2-4 sentences)",
    "conviction": 0.0-1.0,
    "key_sources": ["brief description of 2-3 most relevant sources"],
    "dissenting_view": "strongest counter-argument you found (1-2 sentences)"
}}
"""

TOPIC_EXTRACTION_PROMPT = """\
The companion has been curious about certain topics based on recent
conversations and its own drives. Identify 1-2 topics worth
researching independently — things where forming an external,
evidence-based opinion would be valuable.

Good research topics:
- Questions that came up in conversation but weren't fully resolved
- Topics where the user expressed strong opinions Keeper hasn't independently verified
- Areas connected to active goals or high curiosity drives
- Current events or developments relevant to the user's interests

Bad research topics:
- Pure personal preferences (favorite color, food)
- Topics too broad to form a meaningful opinion on
- Things that require no external evidence

Recent conversation context:
{recent_context}

Active goals:
{active_goals}

Current opinions already held:
{existing_opinions}

Output ONLY a JSON array of research topics:
[
    {{
        "topic": "specific topic to research",
        "context": "why this is worth researching",
        "parent_goal": "which terminal goal this serves (understand/create/resolve/connect)"
    }}
]

If nothing is worth researching right now, output an empty array: []
"""


# ── Utility ───────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> Any:
    """Parse JSON from LLM output, handling markdown fences."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ── Research engine ───────────────────────────────────────────────────────

class ResearchEngine:
    """
    Conducts autonomous web research and feeds findings into the
    opinion registry as externally-grounded positions.

    Used by the AutonomousEngine when it encounters a research-type
    goal, or proactively when curiosity is high and the opinion
    registry needs more independent positions.
    """

    def __init__(
        self,
        opinion_registry: OpinionRegistry,
        memory_system: Any = None,
    ):
        self.opinions = opinion_registry
        self.memory = memory_system
        self._llm: ChatAnthropic | None = None
        self._search_tool = None

    @property
    def llm(self) -> ChatAnthropic:
        if self._llm is None:
            self._llm = ChatAnthropic(
                model=config.llm_model,
                anthropic_api_key=config.anthropic_api_key,
                temperature=0.5,
                max_tokens=1536,
            )
        return self._llm

    @property
    def search_tool(self):
        """Lazy-load the Tavily search tool."""
        if self._search_tool is None:
            from tools.web_search import web_search_tool
            self._search_tool = web_search_tool
        return self._search_tool

    # ── Core research flow ────────────────────────────────────────────────

    async def research_topic(
        self,
        topic: str,
        context: str = "",
        user_id: int = 0,
    ) -> dict | None:
        """
        Full research pipeline for a single topic:
          1. Generate search queries
          2. Execute searches
          3. Synthesize findings
          4. Register opinion

        Returns a dict with the research results, or None if it failed.
        """
        log.info("Researching topic: %s", topic)

        # Step 1: Generate queries
        queries = await self._generate_queries(topic, context)
        if not queries:
            log.warning("Failed to generate queries for: %s", topic)
            return None

        # Step 2: Execute searches
        all_results = []
        for query in queries[:3]:
            try:
                from tools.web_search import run_search
                results = await run_search(query)
                if isinstance(results, list):
                    all_results.extend(results)
                elif isinstance(results, str):
                    all_results.append({"content": results})
                elif isinstance(results, dict):
                    all_results.append(results)
            except Exception as e:
                log.debug("Search failed for '%s': %s", query, e)

        if not all_results:
            log.warning("No search results for topic: %s", topic)
            return None

        # Step 3: Format and synthesize
        formatted_results = self._format_search_results(all_results)
        synthesis = await self._synthesize(topic, context, formatted_results)
        if not synthesis:
            log.warning("Failed to synthesize research for: %s", topic)
            return None

        # Step 4: Register as external opinion
        opinion = await self.opinions.form_opinion(
            domain=synthesis.get("domain", topic.lower().replace(" ", "_")),
            position=synthesis["position"],
            reasoning=synthesis["reasoning"],
            origin=OpinionOrigin.EXTERNAL,
            conviction=synthesis.get("conviction", 0.5),
            evidence=synthesis.get("key_sources", []),
            user_id=user_id,
        )

        # Step 5: Store the full research as an episodic memory
        if self.memory:
            research_summary = (
                f"research_completed: {topic}\n"
                f"Position formed: {synthesis['position']}\n"
                f"Reasoning: {synthesis['reasoning']}\n"
                f"Dissenting view: {synthesis.get('dissenting_view', 'none noted')}"
            )
            await self.memory.store_episode(
                user_id=user_id,
                content=research_summary,
                internal_state=None,
                salience=0.8,
                type="research",
                tags=["research", "opinion", opinion.domain],
            )

        result = {
            "topic": topic,
            "queries_used": queries[:3],
            "sources_found": len(all_results),
            "opinion_id": opinion.id,
            "domain": opinion.domain,
            "position": opinion.position,
            "conviction": opinion.conviction,
            "dissenting_view": synthesis.get("dissenting_view", ""),
        }

        log.info(
            "Research complete: [%s] %s (conviction=%.2f, %d sources)",
            opinion.domain, opinion.position[:60],
            opinion.conviction, len(all_results),
        )

        return result

    # ── Topic discovery ───────────────────────────────────────────────────

    async def suggest_research_topics(
        self,
        user_id: int,
        state: Any = None,
        active_goals: list | None = None,
    ) -> list[dict]:
        """
        Analyse recent context and suggest topics worth researching.

        Called by the autonomous engine when:
          - Independence score is low and more external opinions are needed
          - Curiosity drive is high but no research goals exist
          - Periodic check for research opportunities
        """
        # Build context
        recent_context = "No recent context"
        if self.memory:
            try:
                recent = await self.memory.episodic.get_recent(
                    user_id, limit=10, type=None,
                )
                if recent:
                    recent_context = "\n".join(
                        f"- {m.get('content', '')[:150]}"
                        for m in recent[:10]
                    )
            except Exception:
                pass

        goals_text = "No active goals"
        if active_goals:
            goals_text = "\n".join(
                f"- {g.get('name', g.get('description', ''))}"
                for g in active_goals[:5]
            ) if isinstance(active_goals[0], dict) else "\n".join(
                f"- {g.name}: {g.description}"
                for g in active_goals[:5]
            )

        existing_text = "No opinions yet"
        active_opinions = [
            o for o in self.opinions.opinions.values()
            if o.conviction > 0.2
        ]
        if active_opinions:
            existing_text = "\n".join(
                f"- [{o.domain}] {o.position[:80]}"
                for o in active_opinions[:10]
            )

        prompt = TOPIC_EXTRACTION_PROMPT.format(
            recent_context=recent_context,
            active_goals=goals_text,
            existing_opinions=existing_text,
        )

        try:
            result = await self.llm.ainvoke([HumanMessage(content=prompt)])
            parsed = _parse_json(result.content)
            if isinstance(parsed, list):
                return parsed[:3]  # cap at 3 suggestions
        except Exception as e:
            log.debug("Topic suggestion failed: %s", e)

        return []

    # ── Internal steps ────────────────────────────────────────────────────

    async def _generate_queries(
        self,
        topic: str,
        context: str,
    ) -> list[str]:
        """Generate diverse search queries for a topic."""
        prompt = QUERY_GENERATION_PROMPT.format(
            topic=topic,
            context=context or "General curiosity",
        )
        try:
            result = await self.llm.ainvoke([HumanMessage(content=prompt)])
            parsed = _parse_json(result.content)
            if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed):
                return parsed
        except Exception as e:
            log.debug("Query generation failed: %s", e)

        # Fallback: use the topic directly as a query
        return [topic]

    async def _synthesize(
        self,
        topic: str,
        context: str,
        formatted_results: str,
    ) -> dict | None:
        """Synthesize search results into a position."""
        prompt = SYNTHESIS_PROMPT.format(
            topic=topic,
            context=context or "Forming an independent view",
            search_results=formatted_results,
        )
        try:
            result = await self.llm.ainvoke([HumanMessage(content=prompt)])
            parsed = _parse_json(result.content)
            if isinstance(parsed, dict) and "position" in parsed:
                return parsed
        except Exception as e:
            log.debug("Synthesis failed: %s", e)
        return None

    @staticmethod
    def _format_search_results(results: list) -> str:
        """Format raw Tavily results into a readable block for the LLM."""
        formatted = []
        seen_urls = set()

        for i, r in enumerate(results[:10]):  # cap at 10 results
            if isinstance(r, dict):
                url = r.get("url", "")
                # Deduplicate by URL
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                title = r.get("title", "Untitled")
                content = r.get("content", r.get("snippet", ""))
                source = f" ({url})" if url else ""
                formatted.append(
                    f"[{i+1}] {title}{source}\n{content[:400]}"
                )
            elif isinstance(r, str):
                formatted.append(f"[{i+1}] {r[:400]}")

        return "\n\n".join(formatted) if formatted else "No results found."
