"""
Conscious architecture — the central orchestrator.

MODIFIED (Keeper Modifications Spec):
  - Added CodebaseIndex for structural self-knowledge (mod 1)
  - Added permanent goals initialization (mod 2)
  - Added UserLifeTracker (mod 2)
  - Added SelfTheorizer for offline improvement proposals (mod 3)
  - Added SearchBudget and APIBudget (mods 4+5)
  - Added FutureSimulator for action outcome prediction (mod 6)
  - Budget-driven fatigue in post_process (mod 5)
  - All new state persisted/restored

    Background processes (started on startup):
      - Environmental grounding loop
      - Memory consolidation loop (now conditional — mod 9)
      - Self-model update loop (now multi-layer — mod 8)
      - Autonomous goal pursuit loop

    Foreground process (called per user message):
      - process() builds context from all pillars, returns it for the
        LangGraph agent, then updates all pillars with what happened.
"""
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
from core.resource_budgets import SearchBudget, APIBudget
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
    """Track what's been seen to estimate novelty of new inputs."""

    def __init__(self, window: int = 50):
        self._seen: list[str] = []
        self._window = window

    def estimate(self, text: str) -> float:
        words = set(text.lower().split())
        if not self._seen:
            self._seen.append(text)
            return 0.8

        seen_words = set()
        for s in self._seen[-self._window:]:
            seen_words.update(s.lower().split())

        if not words:
            return 0.5

        overlap = len(words & seen_words) / len(words)
        novelty = 1.0 - overlap

        self._seen.append(text)
        if len(self._seen) > self._window * 2:
            self._seen = self._seen[-self._window:]

        return max(0.1, min(1.0, novelty))

    def clear(self) -> None:
        self._seen.clear()


class ConsciousArchitecture:
    """
    The central loop wiring all five pillars together.

    Background processes (started on startup):
      - Environmental grounding loop (polls streams, updates state)
      - Memory consolidation loop (episodic → semantic, decay, forgetting)
      - Self-model update loop (analyzes behavioral history)
      - Autonomous goal pursuit loop (acts between user inputs)

    Foreground process (called per user message):
      - process() builds context from all pillars, returns it for the
        LangGraph agent, then updates all pillars with what happened.
    """

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

        # ── NEW: Codebase index for structural self-knowledge ─────────────
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
        self.autonomous.self_model = self.self_model  # wire the circular dep

        # Research engine — connects autonomous goals to opinion formation
        self.research = ResearchEngine(
            opinion_registry=self.self_model.opinions,
            memory_system=self.memory,
        )
        self.autonomous.research = self.research

        # ── NEW: Resource budgets ─────────────────────────────────────────
        self.search_budget = SearchBudget()
        self.api_budget = APIBudget()

        # ── NEW: User life tracker ────────────────────────────────────────
        self.user_life = UserLifeTracker()

        # ── NEW: Self-theorizer (offline improvement proposals) ───────────
        self.theorizer = SelfTheorizer(
            codebase=self.codebase,
            self_model=self.self_model,
            memory_system=self.memory,
        )
        self.autonomous.theorizer = self.theorizer

        # ── NEW: Future simulator ─────────────────────────────────────────
        self.simulator = FutureSimulator()
        self.autonomous.simulator = self.simulator

        # Utilities
        self._novelty = NoveltyEstimator()
        self._background_tasks: list[asyncio.Task] = []
        self._started = False
        self._session_active: dict[int, bool] = {}

    # ── Session query (for background processes) ──────────────────────────

    def is_session_active(self, user_id: int) -> bool:
        return self._session_active.get(user_id, False)

    def any_session_active(self) -> bool:
        return any(self._session_active.values())

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def startup(self) -> None:
        if self._started:
            return

        log.info("ConsciousArchitecture starting up…")

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

        # NEW: Load resource budgets
        budgets_path = state_dir / "resource_budgets.json"
        if budgets_path.exists():
            try:
                bdata = json.loads(budgets_path.read_text())
                self.search_budget.restore(bdata.get("search", {}))
                self.api_budget.restore(bdata.get("api", {}))
                log.info(
                    "Budgets restored — API: €%.2f spent, Search: %d used",
                    self.api_budget.spent_today_eur,
                    self.search_budget.searches_today,
                )
            except Exception as e:
                log.warning("Failed to load budgets: %s", e)

        # NEW: Load user life tracker
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

        # NEW: Load theorizer state
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

        # NEW: Initialize permanent goals
        await self.goals.init_permanent_goals()

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
        loop = asyncio.get_event_loop()
        user_ids = list(config.allowed_user_ids)
        primary_user = user_ids[0] if user_ids else 0

        self._background_tasks.append(
            loop.create_task(self.environment.run_loop(
                user_id=primary_user,
                base_interval_seconds=30,
            ))
        )

        self._background_tasks.append(
            loop.create_task(self.memory.consolidator.run_loop(
                user_ids=user_ids,
                interval_minutes=30,
                session_check=self.any_session_active,
            ))
        )

        self._background_tasks.append(
            loop.create_task(self.self_model.run_loop(
                user_ids=user_ids,
                interval_minutes=60,
            ))
        )

        self._background_tasks.append(
            loop.create_task(self.autonomous.run_loop(
                user_id=primary_user,
                interval_minutes=5,
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

            # NEW: Save resource budgets
            budgets_path = state_dir / "resource_budgets.json"
            budgets_path.parent.mkdir(parents=True, exist_ok=True)
            budgets_path.write_text(json.dumps({
                "search": self.search_budget.snapshot(),
                "api": self.api_budget.snapshot(),
            }, indent=2))

            # NEW: Save user life tracker
            life_path = state_dir / "user_life.json"
            life_path.write_text(json.dumps(
                self.user_life.snapshot(), indent=2, default=str
            ))

            # NEW: Save theorizer state
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

        novelty = self._novelty.estimate(user_text)
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

        # NEW: Record API usage and apply budget fatigue
        if input_tokens > 0 or output_tokens > 0:
            self.api_budget.record_usage(input_tokens, output_tokens)
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
            updates = await self.goals.evaluate_progress(user_text, response_text)
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

        # NEW: Resource budget awareness
        budget_lines = []
        budget_lines.append(self.search_budget.to_prompt_context(self.state.curiosity))
        budget_lines.append(self.api_budget.to_prompt_context())
        system_blocks.append("## Resource awareness\n" + "\n".join(budget_lines))

        # NEW: Architecture awareness (only when relevant)
        _lower = user_text.lower()
        ARCHITECTURE_KEYWORDS = {
            "how do you work", "your code", "your architecture",
            "how are you built", "your memory", "your goals",
            "improve yourself", "your capabilities",
        }
        if any(kw in _lower for kw in ARCHITECTURE_KEYWORDS):
            system_blocks.append(self.codebase.summary)

        # NEW: User life context (if there are open commitments)
        life_ctx = self.user_life.to_prompt_context()
        if life_ctx:
            system_blocks.append(life_ctx)

        # Tool instructions
        system_blocks.append(
            "\n---\n"
            "You have tools available: web search and memory management. "
            "Use web_search when you need current information. "
            "Use memory tools to save important facts/preferences/summaries "
            "proactively — your future self depends on them.\n"
            "Reply naturally, as a companion would. Never break character."
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
        self._novelty.clear()
        self._session_active[user_id] = False

    # ── Status ────────────────────────────────────────────────────────────

    def status(self) -> dict:
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
            "background_tasks": len([t for t in self._background_tasks if not t.done()]),
            "active_sessions": {
                uid: active for uid, active in self._session_active.items()
            },
            # NEW
            "api_budget_remaining_eur": round(self.api_budget.remaining_eur, 2),
            "search_budget_remaining": self.search_budget.remaining,
            "pending_proposals": len(self.theorizer.pending_proposals),
            "user_open_commitments": len(self.user_life.open_commitments),
            "hypotheses": len(self.self_model.model.get("hypotheses", [])),
        }


# ── Singleton ─────────────────────────────────────────────────────────────

companion = ConsciousArchitecture()
