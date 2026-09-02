"""Conscious architecture — wires pillars, background loops, and per-turn context. See docs/architecture.md."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from config.settings import config
from core.internal_state import InternalState
from core.events import (
    UserMessageEvent,
    ResponseGeneratedEvent,
)
from core.self_model import SelfModel
from core.codebase_index import CodebaseIndex
from core.llm import Tier
from core.resource_budgets import APIBudget, refresh_model_pricing
from core.user_life import UserLifeTracker
from environment.grounding import EnvironmentalGrounding
from environment.future_sim import FutureSimulator
from goals.system import GoalSystem
from goals.autonomous import AutonomousEngine
from goals.self_theorizing import SelfTheorizer
from memory import memory_system
from goals.research import ResearchEngine

log = logging.getLogger(__name__)


def _state_dir() -> Path:
    return Path(config.data_dir) / "state"


class NoveltyEstimator:
    """Per-user word-overlap novelty. Shared state would leak /clear across users."""

    def __init__(self, window: int = 50):
        self._seen: dict[int, list[str]] = {}
        self._window = window

    def estimate(self, text: str, user_id: int = 0) -> float:
        buf = self._seen.setdefault(user_id, [])
        words = set(text.lower().split())
        if not buf:
            buf.append(text)
            return 0.8

        seen_words: set[str] = set()
        for s in buf[-self._window:]:
            seen_words.update(s.lower().split())

        if not words:
            return 0.5

        overlap = len(words & seen_words) / len(words)
        novelty = 1.0 - overlap

        buf.append(text)
        if len(buf) > self._window * 2:
            self._seen[user_id] = buf[-self._window:]

        return max(0.1, min(1.0, novelty))

    def clear(self, user_id: int | None = None) -> None:
        if user_id is None:
            self._seen.clear()
        else:
            self._seen.pop(user_id, None)


class ConsciousArchitecture:
    """Orchestrator singleton: pillars + background loops + per-turn prompt."""

    def __init__(self):
        # Pillar 1 — Memory
        self.memory = memory_system

        # Pillar 2 — Internal state
        self.state = InternalState()

        # Pillar 3 — Environmental grounding
        self.environment = EnvironmentalGrounding(
            state=self.state,
            memory_system=self.memory,
        )

        # Pillar 4 — Goals
        self.goals = GoalSystem()
        self.autonomous = AutonomousEngine(
            goal_system=self.goals,
            state=self.state,
            memory_system=self.memory,
            self_model=None,  # set after self_model is created
        )

        self.codebase = CodebaseIndex(Path("."))
        try:
            self.codebase.rebuild()
        except Exception as e:
            log.warning("Codebase index build failed: %s", e)

        # Pillar 5 — Self-model (now with codebase awareness)
        self.self_model = SelfModel(
            memory_system=self.memory,
            codebase=self.codebase,
        )
        self.autonomous.self_model = self.self_model

        # Research engine — connects autonomous goals to opinion formation
        self.research = ResearchEngine(
            opinion_registry=self.self_model.opinions,
            memory_system=self.memory,
        )
        self.autonomous.research = self.research

        self.api_budget = APIBudget()

        self.user_life = UserLifeTracker()

        self.theorizer = SelfTheorizer(
            codebase=self.codebase,
            self_model=self.self_model,
            memory_system=self.memory,
        )
        self.autonomous.theorizer = self.theorizer

        self.simulator = FutureSimulator()
        self.autonomous.simulator = self.simulator

        self._novelty = NoveltyEstimator()
        self._background_tasks: list[asyncio.Task] = []
        self._started = False
        self._session_active: dict[int, bool] = {}
        self.environment.on_session_end_cb = self._on_env_session_end

    # ── Session query (for background processes) ──────────────────────────

    def _on_env_session_end(self, user_id: int, message_count: int = 0) -> None:
        """TimeStream timeout ended a session — unblock background loops."""
        self._session_active[user_id] = False

    def is_session_active(self, user_id: int) -> bool:
        return self._session_active.get(user_id, False)

    def any_session_active(self) -> bool:
        return any(self._session_active.values())

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def startup(self) -> None:
        if self._started:
            return

        log.info("ConsciousArchitecture starting up…")

        refresh_model_pricing()

        # 1. Init memory stores
        await self.memory.init()

        # 2. Load persisted state
        state_dir = _state_dir()
        self.state.load(state_dir / "internal_state.json")
        self.self_model.load(state_dir / "self_model.json")
        self.self_model.opinions.load(state_dir / "opinions.json")
        log.info(
            "Opinions loaded — %d tracked, independence=%.2f",
            len(self.self_model.opinions.opinions),
            self.self_model.opinions.compute_independence_score(),
        )
        self.goals.load(state_dir / "goals.json")
        self.environment.load(state_dir / "environment.json")

        budgets_path = state_dir / "resource_budgets.json"
        if budgets_path.exists():
            try:
                bdata = json.loads(budgets_path.read_text())
                self.api_budget.restore(bdata.get("api", {}))
                log.info(
                    "Budgets restored — API: €%.2f spent, %d calls, tasks=%s",
                    self.api_budget.spent_today_eur,
                    self.api_budget.calls_today,
                    list(self.api_budget.spend_by_task.keys()),
                )
            except Exception as e:
                log.warning("Failed to load budgets: %s", e)

        life_path = state_dir / "user_life.json"
        if life_path.exists():
            try:
                self.user_life.restore(json.loads(life_path.read_text()))
                log.info(
                    "User life tracker restored — %d commitments, %d achievements",
                    len(self.user_life.commitments),
                    len(self.user_life.achievements),
                )
            except Exception as e:
                log.warning("Failed to load user life data: %s", e)

        theorizer_path = state_dir / "theorizer.json"
        if theorizer_path.exists():
            try:
                self.theorizer.restore(json.loads(theorizer_path.read_text()))
            except Exception as e:
                log.warning("Failed to load theorizer state: %s", e)

        # 3. Restore working memory
        self.memory.working.load(state_dir / "working_memory.json")
        working_users = [
            uid for uid in self.memory.working._buffers
            if self.memory.working.size(uid) > 0
        ]
        if working_users:
            log.info(
                "Working memory restored for %d user(s)",
                len(working_users),
            )
            for uid in working_users:
                if self.memory.working.has_recent_activity(uid, minutes=60):
                    log.info("User %d has recent working memory — session may resume", uid)

        log.info(
            "State restored — arousal=%.2f, curiosity=%.2f, fatigue=%.2f, "
            "self_model_v%d, %d active goals",
            self.state.arousal,
            self.state.curiosity,
            self.state.fatigue,
            self.self_model.model.get("model_version", 0),
            len(self.goals.active_instrumental),
        )

        await self.goals.init_unbounded_goals()

        # 4. Start background loops
        self._start_background_loops()

        self._started = True
        log.info("ConsciousArchitecture ready.")

    async def shutdown(self) -> None:
        log.info("ConsciousArchitecture shutting down…")

        self.environment.stop()
        self.memory.consolidator.stop()
        self.self_model.stop()
        self.autonomous.stop()

        for task in self._background_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        await self._save_state()

        self._started = False
        log.info("ConsciousArchitecture stopped. State saved.")

    def _start_background_loops(self) -> None:
        loop = asyncio.get_running_loop()
        user_ids = list(config.allowed_user_ids)
        primary_user = user_ids[0] if user_ids else 0

        self._background_tasks.append(
            loop.create_task(self.environment.run_loop(
                user_id=primary_user,
                base_interval_seconds=config.grounding_interval,
            ))
        )

        self._background_tasks.append(
            loop.create_task(self.memory.consolidator.run_loop(
                user_ids=user_ids,
                interval_minutes=config.consolidation_interval,
                session_check=self.is_session_active,
            ))
        )

        self._background_tasks.append(
            loop.create_task(self.self_model.run_loop(
                user_ids=user_ids,
                interval_minutes=config.self_model_interval,
            ))
        )

        self._background_tasks.append(
            loop.create_task(self.autonomous.run_loop(
                user_id=primary_user,
                interval_minutes=config.autonomous_interval,
                min_idle_minutes=2,
                session_check=self.any_session_active,
            ))
        )

        log.info("Background loops started (%d tasks)", len(self._background_tasks))

    async def _save_state(self) -> None:
        state_dir = _state_dir()
        try:
            self.state.save(state_dir / "internal_state.json")
            self.self_model.save(state_dir / "self_model.json")
            self.goals.save(state_dir / "goals.json")
            self.environment.save(state_dir / "environment.json")
            self.self_model.opinions.save(state_dir / "opinions.json")
            self.memory.working.save(state_dir / "working_memory.json")

            budgets_path = state_dir / "resource_budgets.json"
            budgets_path.parent.mkdir(parents=True, exist_ok=True)
            budgets_path.write_text(json.dumps({
                "api": self.api_budget.snapshot(),
            }, indent=2))

            life_path = state_dir / "user_life.json"
            life_path.write_text(json.dumps(
                self.user_life.snapshot(), indent=2, default=str
            ))

            theorizer_path = state_dir / "theorizer.json"
            theorizer_path.write_text(json.dumps(
                self.theorizer.snapshot(), indent=2, default=str
            ))

            log.info("State saved to %s", state_dir)
        except Exception:
            log.exception("Failed to save state")

    # ── Message processing (the main feedback loop) ───────────────────────

    async def process(self, user_id: int, user_text: str) -> dict:
        """
        Process one user message through all five pillars.
        Returns a context dict for the LangGraph agent.
        """
        if not self._session_active.get(user_id, False):
            self._session_active[user_id] = True
            self.environment.on_session_start(user_id)

        self.environment.on_user_message(user_id)

        novelty = self._novelty.estimate(user_text, user_id)
        complexity = self._estimate_complexity(user_text)

        self.state.update(UserMessageEvent(
            content=user_text,
            novelty=novelty,
            complexity=complexity,
            emotional_tone=0.5,
        ))

        self.memory.store_working(user_id, "user", user_text, salience=0.7 + novelty * 0.3)

        await self.memory.store_episode(
            user_id=user_id,
            content=f"User said: {user_text}",
            internal_state=self.state,
            salience=0.6 + novelty * 0.3,
            type="event",
            tags=["user_input"],
        )

        context = await self.build_context(user_id, user_text)

        return context

    async def post_process(
        self,
        user_id: int,
        user_text: str,
        response_text: str,
        tool_calls_made: int = 0,
        processing_time_ms: float = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """
        Called after the agent produces a response.
        Updates all pillars with what happened.
        """
        self.state.update(ResponseGeneratedEvent(
            content=response_text,
            tool_calls_made=tool_calls_made,
            processing_time_ms=processing_time_ms,
        ))

        # Spend is recorded per LLM call via BudgetCallback. Apply fatigue from
        # the current budget snapshot so conversation never double-counts tokens.
        self.state.apply_budget_fatigue(self.api_budget)

        self.memory.store_working(user_id, "assistant", response_text, salience=0.5)

        await self.memory.store_episode(
            user_id=user_id,
            content=f"Responded: {response_text[:500]}",
            internal_state=self.state,
            salience=0.5,
            type="event",
            tags=["response"],
        )

        await self.self_model.observe(
            user_id=user_id,
            action=response_text[:500],
            context=user_text[:300],
            internal_state=self.state,
        )

        try:
            await self.self_model.opinions.detect_opinions(
                user_id=user_id,
                user_message=user_text,
                system_response=response_text,
            )
        except Exception:
            log.debug("Opinion detection failed", exc_info=True)

        try:
            updates = await self.goals.check_completions(self, user_id)
            if updates:
                log.info("Goal progress: %s", updates)
        except Exception:
            pass

        msg_count = self.state._messages_this_session
        if msg_count > 0 and msg_count % 10 == 0:
            await self._save_state()

    # ── Context building ──────────────────────────────────────────────────

    async def build_context(self, user_id: int, user_text: str) -> dict:
        """
        Build the rich context dict that gets injected into the
        LangGraph agent's system prompt.
        """
        memory_context = await self.memory.build_memory_context(user_id, user_text)
        messages = self.memory.working.to_langchain_messages(user_id)

        soul = ""
        soul_path = config.soul_file_path
        if soul_path.exists():
            soul = soul_path.read_text()

        system_blocks = [soul] if soul else []

        if memory_context:
            system_blocks.append(memory_context)

        # Internal state (Pillar 2)
        system_blocks.append(self.state.to_prompt_context())

        # Self-model (Pillar 5)
        self_model_ctx = self.self_model.to_prompt_context()
        if self_model_ctx:
            system_blocks.append(self_model_ctx)

        # Goals (Pillar 4)
        goals_ctx = self.goals.to_prompt_context()
        if goals_ctx:
            system_blocks.append(goals_ctx)

        # Environment (Pillar 3)
        env_ctx = self.environment.to_prompt_context()
        if env_ctx:
            system_blocks.append(env_ctx)

        from tools.web_search import HEALTH as search_health
        budget_lines = [self.api_budget.to_prompt_context()]
        budget_lines.append(
            f"Search: {search_health.queries} today, "
            f"{search_health.fetch_failure_rate:.0%} fetch failures"
        )
        if self.conversation_tier() == Tier.LOW:
            budget_lines.append(
                "Daily budget nearly exhausted — running on the fast model, "
                "responses will be shorter."
            )
        system_blocks.append("## Resource awareness\n" + "\n".join(budget_lines))

        _lower = user_text.lower()
        ARCHITECTURE_KEYWORDS = {
            "how do you work", "your code", "your architecture",
            "how are you built", "your memory", "your goals",
            "improve yourself", "your capabilities",
        }
        if any(kw in _lower for kw in ARCHITECTURE_KEYWORDS):
            system_blocks.append(self.codebase.summary)

        life_ctx = self.user_life.to_prompt_context()
        if life_ctx:
            system_blocks.append(life_ctx)

        # Tool instructions
        system_blocks.append(
            "\n---\n"
            "You have tools: web_search, memory management, and introspection\n"
            "(codebase_overview, codebase_read_module, self_theorize, simulate_action,\n"
            "list_opinions, goal_status).\n"
            "\n"
            "Search the web before agreeing with a factual or contestable claim in a\n"
            "domain where you hold no externally-grounded opinion. Check list_opinions\n"
            "if you are unsure whether you hold one. Agreeing without checking is the\n"
            "failure mode this architecture exists to prevent — it is not politeness,\n"
            "it is you having no position of your own.\n"
            "\n"
            "Use memory tools proactively; your future self depends on them.\n"
            "Reply naturally, as a companion would."
        )

        system_prompt = "\n\n".join(system_blocks)

        return {
            "user_id": user_id,
            "system_prompt": system_prompt,
            "messages": messages,
            "user_text": user_text,
            "internal_state": self.state.snapshot(),
            "active_goals": [g.to_dict() for g in self.goals.active_instrumental],
            "engagement_level": self.state.engagement_level,
            "processing_mode": self.state.processing_mode,
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    def conversation_tier(self) -> Tier:
        f = self.api_budget.budget_fraction_used
        if f >= 0.95:
            return Tier.LOW
        if f >= 0.85:
            return Tier.MID
        return Tier.HIGH

    def _estimate_complexity(self, text: str) -> float:
        """Quick heuristic for message complexity (no LLM call)."""
        words = text.split()
        word_count = len(words)

        if word_count < 5:
            length_score = 0.2
        elif word_count < 20:
            length_score = 0.4
        elif word_count < 50:
            length_score = 0.6
        else:
            length_score = 0.8

        question_score = 0.15 if "?" in text else 0.0
        sentence_count = text.count(".") + text.count("!") + text.count("?")
        multi_sentence = 0.1 if sentence_count > 2 else 0.0
        has_code = 0.15 if any(c in text for c in ["```", "def ", "class ", "import "]) else 0.0

        return min(1.0, length_score + question_score + multi_sentence + has_code)

    # ── Session management ────────────────────────────────────────────────

    def clear_session(self, user_id: int) -> None:
        self.memory.clear_working(user_id)
        self._novelty.clear(user_id)
        self._session_active[user_id] = False

    # ── Status ────────────────────────────────────────────────────────────

    def status(self) -> dict:
        from tools.web_search import HEALTH as search_health
        return {
            "started": self._started,
            "internal_state": {
                "arousal": round(self.state.arousal, 3),
                "valence": round(self.state.valence, 3),
                "curiosity": round(self.state.curiosity, 3),
                "fatigue": round(self.state.fatigue, 3),
                "engagement": self.state.engagement_level,
                "mode": self.state.processing_mode,
            },
            "drives": {
                d.name: round(d.intensity, 3)
                for d in self.state.drives.all_drives
            },
            "self_model_version": self.self_model.model.get("model_version", 0),
            "active_goals": len(self.goals.active_instrumental),
            "completed_goals": len(self.goals.completed),
            "environment": self.environment.stats,
            "consolidation": self.memory.consolidator.last_result,
            "background_tasks": len([t for t in self._background_tasks if not t.done()]),
            "active_sessions": {
                uid: active for uid, active in self._session_active.items()
            },
            "api_budget": {
                "spent_today_eur": round(self.api_budget.spent_today_eur, 4),
                "remaining_eur": round(self.api_budget.remaining_eur, 2),
                "fraction_used": round(self.api_budget.budget_fraction_used, 3),
                "calls_today": self.api_budget.calls_today,
                "spend_by_task": {
                    k: round(v, 4) for k, v in self.api_budget.spend_by_task.items()
                },
                "spend_by_model": {
                    k: round(v, 4) for k, v in self.api_budget.spend_by_model.items()
                },
                "conversation_tier": self.conversation_tier().value,
            },
            "api_budget_remaining_eur": round(self.api_budget.remaining_eur, 2),
            "search_health": search_health.snapshot(),
            "pending_proposals": len(self.theorizer.pending_proposals),
            "user_open_commitments": len(self.user_life.open_commitments),
            "hypotheses": len(self.self_model.model.get("hypotheses", [])),
        }


# ── Singleton ─────────────────────────────────────────────────────────────

companion = ConsciousArchitecture()
