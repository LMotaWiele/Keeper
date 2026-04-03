"""
The conscious architecture — the main feedback loop.

This is where all five pillars connect into a coherent system.
Not a pipeline (input → output) but a continuous loop:

    Internal state → influences what gets remembered
    Memory → shapes the self-model
    Self-model → influences goal structures
    Goals → direct environmental attention
    Environment → updates internal state

The system that responds tomorrow is different from the one today
because things happened in between — not because instructions changed,
but because it experienced something, consolidated memories, updated
its self-model, and pursued goals while nobody was watching.

Usage:
    from core.loop import companion

    await companion.startup()        # init + load state + start background loops
    response = await companion.process(user_id, "hello")
    await companion.shutdown()       # save state + stop background loops
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import config
from core.internal_state import InternalState
from core.self_model import SelfModel
from core.events import (
    UserMessageEvent,
    ResponseGeneratedEvent,
    SessionStartEvent,
)
from memory import MemorySystem, memory_system
from environment.grounding import EnvironmentalGrounding
from goals.system import GoalSystem
from goals.autonomous import AutonomousEngine

log = logging.getLogger(__name__)


# ── Novelty estimation (lightweight, no LLM call) ────────────────────────

class NoveltyEstimator:
    """
    Tracks recent topics to estimate how novel a new message is.
    Uses simple token overlap — not perfect, but fast and inline.
    """

    def __init__(self, window: int = 30):
        self._recent_tokens: list[set[str]] = []
        self._window = window

    def estimate(self, text: str) -> float:
        """Return 0.0 (repetitive) to 1.0 (completely new)."""
        tokens = set(text.lower().split())
        if not tokens:
            return 0.5

        if not self._recent_tokens:
            self._recent_tokens.append(tokens)
            return 0.8  # first message is fairly novel

        # Compute overlap with recent messages
        all_recent = set()
        for t in self._recent_tokens:
            all_recent |= t

        if not all_recent:
            return 0.8

        overlap = len(tokens & all_recent) / len(tokens)
        novelty = 1.0 - overlap

        self._recent_tokens.append(tokens)
        if len(self._recent_tokens) > self._window:
            self._recent_tokens.pop(0)

        # Scale: 0.2 (very repetitive) to 0.9 (very novel)
        return 0.2 + novelty * 0.7

    def clear(self) -> None:
        self._recent_tokens.clear()


# ── State persistence paths ───────────────────────────────────────────────

def _state_dir() -> Path:
    d = Path(config.data_dir) / "state"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── The main architecture ────────────────────────────────────────────────

class ConsciousArchitecture:
    """
    The central nervous system. Owns all five pillars and orchestrates
    the continuous feedback loop between them.

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

        # Utilities
        self._novelty = NoveltyEstimator()
        self._background_tasks: list[asyncio.Task] = []
        self._started = False
        self._session_active: dict[int, bool] = {}

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
        self.goals.load(state_dir / "goals.json")
        self.environment.load(state_dir / "environment.json")

        log.info(
            "State restored — arousal=%.2f, curiosity=%.2f, fatigue=%.2f, "
            "self_model_v%d, %d active goals",
            self.state.arousal,
            self.state.curiosity,
            self.state.fatigue,
            self.self_model.model.get("model_version", 0),
            len(self.goals.active_instrumental),
        )

        # 3. Start background loops
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

        # Save state
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

        # Memory consolidation (every 30 min)
        self._background_tasks.append(
            loop.create_task(self.memory.consolidator.run_loop(
                user_ids=user_ids,
                interval_minutes=30,
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
            ))
        )

        log.info("Background loops started (%d tasks)", len(self._background_tasks))

    async def _save_state(self) -> None:
        """Persist all pillar state to disk."""
        state_dir = _state_dir()
        try:
            self.state.save(state_dir / "internal_state.json")
            self.self_model.save(state_dir / "self_model.json")
            self.goals.save(state_dir / "goals.json")
            self.environment.save(state_dir / "environment.json")
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

        # Goals evaluate progress
        try:
            updates = await self.goals.evaluate_progress(user_text, response_text)
            if updates:
                log.info("Goal progress: %s", updates)
        except Exception:
            pass  # goal eval is best-effort

        # Periodic save (every 10 messages)
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
        }


# ── Singleton ─────────────────────────────────────────────────────────────

companion = ConsciousArchitecture()
