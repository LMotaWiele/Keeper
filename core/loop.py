"""
Conscious architecture — the central orchestrator.

PATCHED:
  - Working memory now persisted in _save_state() and restored in startup()
  - Background loops pass session_active reference so they can check before acting
  - _save_state() is now safe to call anytime without side effects on working memory
  - Added is_session_active() helper for background processes

    Background processes (started on startup):
      - Environmental grounding loop (polls streams, updates state)
      - Memory consolidation loop (episodic → semantic, decay, forgetting)
      - Self-model update loop (analyzes behavioral history)
      - Autonomous goal pursuit loop (acts between user inputs)

    Foreground process (called per user message):
      - process() builds context from all pillars, returns it for the
        LangGraph agent, then updates all pillars with what happened.
"""
from __future__ import annotations

import asyncio
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
from environment.grounding import EnvironmentalGrounding
from goals.system import GoalSystem
from goals.autonomous import AutonomousEngine
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

        # Pillar 5 — Self-model
        self.self_model = SelfModel(memory_system=self.memory)
        self.autonomous.self_model = self.self_model  # wire the circular dep

        # Research engine — connects autonomous goals to opinion formation
        self.research = ResearchEngine(
            opinion_registry=self.self_model.opinions,
            memory_system=self.memory,
        )
        self.autonomous.research = self.research

        # Utilities
        self._novelty = NoveltyEstimator()
        self._background_tasks: list[asyncio.Task] = []
        self._started = False
        self._session_active: dict[int, bool] = {}

    # ── Session query (for background processes) ──────────────────────────

    def is_session_active(self, user_id: int) -> bool:
        """
        Check if a user has an active conversation session.
        Used by background processes to avoid disrupting active conversations.
        """
        return self._session_active.get(user_id, False)

    def any_session_active(self) -> bool:
        """Check if ANY user has an active session."""
        return any(self._session_active.values())

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def startup(self) -> None:
        """
        Initialize everything, load persisted state, start background loops.
        Call this once at application start.
        """
        if self._started:
            return

        log.info("ConsciousArchitecture starting up…")

        # 1. Init memory stores (creates SQLite tables etc.)
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

        # 3. Restore working memory (conversation thread)
        self.memory.working.load(state_dir / "working_memory.json")
        working_users = [
            uid for uid in self.memory.working._buffers
            if self.memory.working.size(uid) > 0
        ]
        if working_users:
            log.info(
                "Working memory restored for %d user(s) — conversation continuity preserved",
                len(working_users),
            )
            # If there's recent working memory, mark session as potentially active
            # (the next message will confirm it properly)
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

        # 4. Start background loops
        self._start_background_loops()

        self._started = True
        log.info("ConsciousArchitecture ready.")

    async def shutdown(self) -> None:
        """
        Save all state and stop background loops.
        Call this on application exit.
        """
        log.info("ConsciousArchitecture shutting down…")

        # Stop background loops
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

        # Save state (includes working memory now)
        await self._save_state()

        self._started = False
        log.info("ConsciousArchitecture stopped. State saved.")

    def _start_background_loops(self) -> None:
        """Spin up all background async tasks."""
        loop = asyncio.get_event_loop()

        # Get user IDs from config for background processes
        user_ids = list(config.allowed_user_ids)
        primary_user = user_ids[0] if user_ids else 0

        # Environment grounding (polls every 30s)
        self._background_tasks.append(
            loop.create_task(self.environment.run_loop(
                user_id=primary_user,
                base_interval_seconds=30,
            ))
        )

        # Memory consolidation (every 30 min, session-aware)
        self._background_tasks.append(
            loop.create_task(self.memory.consolidator.run_loop(
                user_ids=user_ids,
                interval_minutes=30,
                session_check=self.any_session_active,
            ))
        )

        # Self-model updates (every 60 min)
        self._background_tasks.append(
            loop.create_task(self.self_model.run_loop(
                user_ids=user_ids,
                interval_minutes=60,
            ))
        )

        # Autonomous goal pursuit (every 5 min, only when idle)
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
        """
        Persist all pillar state to disk.
        PATCHED: Now includes working memory so conversations survive restarts.
        """
        state_dir = _state_dir()
        try:
            self.state.save(state_dir / "internal_state.json")
            self.self_model.save(state_dir / "self_model.json")
            self.goals.save(state_dir / "goals.json")
            self.environment.save(state_dir / "environment.json")
            self.self_model.opinions.save(state_dir / "opinions.json")

            # PATCH: persist the conversation thread
            self.memory.working.save(state_dir / "working_memory.json")

            log.info("State saved to %s", state_dir)
        except Exception:
            log.exception("Failed to save state")

    # ── Message processing (the main feedback loop) ───────────────────────

    async def process(self, user_id: int, user_text: str) -> dict:
        """
        Process one user message through all five pillars.

        Returns a context dict that the LangGraph agent uses to build
        its system prompt and message history. After the agent responds,
        call post_process() with the response.

        Flow:
          1. Detect session start
          2. Estimate novelty and complexity
          3. Fire UserMessageEvent into internal state
          4. Store input in working + episodic memory
          5. Build rich context from all pillars
          6. Return context for the agent
        """
        # Session management
        if not self._session_active.get(user_id, False):
            self._session_active[user_id] = True
            self.environment.on_session_start(user_id)

        self.environment.on_user_message(user_id)

        # Estimate input properties
        novelty = self._novelty.estimate(user_text)
        complexity = self._estimate_complexity(user_text)

        # Fire event into internal state
        self.state.update(UserMessageEvent(
            content=user_text,
            novelty=novelty,
            complexity=complexity,
            emotional_tone=0.5,  # neutral default; could add sentiment analysis
        ))

        # Store in working memory
        self.memory.store_working(user_id, "user", user_text, salience=0.7 + novelty * 0.3)

        # Store in episodic memory
        await self.memory.store_episode(
            user_id=user_id,
            content=f"User said: {user_text}",
            internal_state=self.state,
            salience=0.6 + novelty * 0.3,
            type="event",
            tags=["user_input"],
        )

        # Build context from all pillars
        context = await self.build_context(user_id, user_text)

        return context

    async def post_process(
        self,
        user_id: int,
        user_text: str,
        response_text: str,
        tool_calls_made: int = 0,
        processing_time_ms: float = 0,
    ) -> None:
        """
        Called after the agent produces a response. Updates all pillars
        with what happened.

        Flow:
          1. Fire ResponseGeneratedEvent into internal state
          2. Store response in working + episodic memory
          3. Self-model observes the response
          4. Goals evaluate progress
          5. Periodic state save
        """
        # Fire event
        self.state.update(ResponseGeneratedEvent(
            content=response_text,
            tool_calls_made=tool_calls_made,
            processing_time_ms=processing_time_ms,
        ))

        # Store response in working memory
        self.memory.store_working(user_id, "assistant", response_text, salience=0.5)

        # Store in episodic memory
        await self.memory.store_episode(
            user_id=user_id,
            content=f"Responded: {response_text[:500]}",
            internal_state=self.state,
            salience=0.5,
            type="event",
            tags=["response"],
        )

        # Self-model observes
        await self.self_model.observe(
            user_id=user_id,
            action=response_text[:500],
            context=user_text[:300],
            internal_state=self.state,
        )

        # Opinion detection — runs after each response
        try:
            await self.self_model.opinions.detect_opinions(
                user_id=user_id,
                user_message=user_text,
                system_response=response_text,
            )
        except Exception:
            log.debug("Opinion detection failed", exc_info=True)

        # Goals evaluate progress
        try:
            updates = await self.goals.evaluate_progress(user_text, response_text)
            if updates:
                log.info("Goal progress: %s", updates)
        except Exception:
            pass  # goal eval is best-effort

        # Periodic save (every 10 messages) — now includes working memory
        msg_count = self.state._messages_this_session
        if msg_count > 0 and msg_count % 10 == 0:
            await self._save_state()

    # ── Context building ──────────────────────────────────────────────────

    async def build_context(self, user_id: int, user_text: str) -> dict:
        """
        Build the rich context dict that gets injected into the
        LangGraph agent's system prompt.

        Combines all five pillars into a coherent prompt block,
        plus returns structured data the agent nodes can use.
        """
        # Memory context (episodic + semantic)
        memory_context = await self.memory.build_memory_context(user_id, user_text)

        # Working memory as message history
        messages = self.memory.working.to_langchain_messages(user_id)

        # Soul file
        soul = ""
        soul_path = config.soul_file_path
        if soul_path.exists():
            soul = soul_path.read_text()

        # Assemble system prompt blocks
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
            # Structured data for agent nodes
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

        # Length factor
        if word_count < 5:
            length_score = 0.2
        elif word_count < 20:
            length_score = 0.4
        elif word_count < 50:
            length_score = 0.6
        else:
            length_score = 0.8

        # Question marks suggest questions (slightly more complex)
        question_score = 0.15 if "?" in text else 0.0

        # Multiple sentences suggest more complex input
        sentence_count = text.count(".") + text.count("!") + text.count("?")
        multi_sentence = 0.1 if sentence_count > 2 else 0.0

        # Technical indicators
        has_code = 0.15 if any(c in text for c in ["```", "def ", "class ", "import "]) else 0.0

        return min(1.0, length_score + question_score + multi_sentence + has_code)

    # ── Session management ────────────────────────────────────────────────

    def clear_session(self, user_id: int) -> None:
        """Clear working memory and reset session state for a user."""
        self.memory.clear_working(user_id)
        self._novelty.clear()
        self._session_active[user_id] = False

    # ── Status ────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Current status of all pillars — useful for debugging."""
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
        }


# ── Singleton ─────────────────────────────────────────────────────────────

companion = ConsciousArchitecture()
