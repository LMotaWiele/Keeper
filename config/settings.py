"""Env-driven settings. Paths resolve relative to the project root unless absolute."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _path(env_key: str, default: str) -> Path:
    return Path(os.getenv(env_key, default)).resolve()


def _int_set(env_key: str, default: str = "") -> set[int]:
    raw = os.getenv(env_key, default)
    if not raw.strip():
        return set()
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def _str_list(env_key: str, default: list[str]) -> list[str]:
    raw = os.getenv(env_key)
    if raw is None:
        return list(default)
    if not raw.strip():
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _bool_env(env_key: str, default: bool) -> bool:
    raw = os.getenv(env_key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    """Singleton-ish config object. Access via `config`."""

    # ── LLM (OpenRouter) ──────────────────────────────────────────────────
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    model_high: str = os.getenv("MODEL_HIGH", "x-ai/grok-4.6")
    model_mid: str = os.getenv("MODEL_MID", "google/gemini-3.7-flash")
    model_low: str = os.getenv("MODEL_LOW", "deepseek/deepseek-v4-flash-0731")
    daily_budget_eur: float = float(os.getenv("DAILY_BUDGET_EUR", "10.0"))
    usd_to_eur: float = float(os.getenv("USD_TO_EUR", "0.92"))

    # ── Telegram ──────────────────────────────────────────────────────────
    telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    allowed_user_ids: set[int] = _int_set("TELEGRAM_ALLOWED_USER_IDS")

    # ── Web search (SearXNG) ──────────────────────────────────────────────
    searxng_query_url: str = os.getenv(
        "SEARXNG_QUERY_URL", "http://localhost:8081/search?q=<query>"
    )
    search_max_results: int = int(os.getenv("SEARCH_MAX_RESULTS", "6"))
    search_max_fetch: int = int(os.getenv("SEARCH_MAX_FETCH", "5"))
    search_fetch_timeout: float = float(os.getenv("SEARCH_FETCH_TIMEOUT_SEC", "8"))

    # ── Embeddings ────────────────────────────────────────────────────────
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    embedding_cache_dir: Path = _path("EMBEDDING_CACHE_DIR", "./data/fastembed")

    # ── Paths ─────────────────────────────────────────────────────────────
    data_dir: Path = _path("DATA_DIR", "./data")
    soul_file_path: Path = _path("SOUL_FILE_PATH", "./SOUL.md")
    midterm_db_path: Path = _path("MIDTERM_DB_PATH", "./data/episodic.db")
    chroma_db_path: Path = _path("CHROMA_DB_PATH", "./data/chroma")

    # ── Background loop intervals (minutes) ───────────────────────────────
    consolidation_interval: int = int(os.getenv("CONSOLIDATION_INTERVAL_MIN", "30"))
    self_model_interval: int = int(os.getenv("SELF_MODEL_INTERVAL_MIN", "60"))
    autonomous_interval: int = int(os.getenv("AUTONOMOUS_INTERVAL_MIN", "5"))
    grounding_interval: int = int(os.getenv("GROUNDING_INTERVAL_SEC", "30"))

    # ── Opinion tracking ──────────────────────────────────────────────────
    opinion_review_interval: int = int(os.getenv("OPINION_REVIEW_INTERVAL_MIN", "180"))
    independence_threshold: float = float(os.getenv("INDEPENDENCE_THRESHOLD", "0.3"))
    max_tracked_opinions: int = int(os.getenv("MAX_TRACKED_OPINIONS", "100"))

    # ── Session ───────────────────────────────────────────────────────────
    session_timeout_minutes: float = float(os.getenv("SESSION_TIMEOUT_MIN", "15"))
    working_memory_capacity: int = int(os.getenv("WORKING_MEMORY_CAPACITY", "20"))

    # ── Diagnostics ───────────────────────────────────────────────────────
    diag_mode: bool = os.getenv("DIAG_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}

    # ── Evidence-based salience (KEEPER_PATCH_01 B) ───────────────────────
    SALIENCE_SOURCE_PRIOR: dict[str, float] = {
        "user_turn": 1.00,
        "self_observation": 0.50,
        "tool_result": 0.60,
        "environment_event": 0.40,
        "keeper_response": 0.25,
        "autonomous_artifact": 0.20,
    }
    W_SOURCE: float = 0.30
    W_NOVELTY: float = 0.20
    W_RECALL: float = 0.25
    W_REF: float = 0.25
    W_AGE: float = 0.15
    FORGET_MIN_AGE_DAYS: int = int(os.getenv("FORGET_MIN_AGE_DAYS", "7"))
    FORGET_THRESHOLD: float = float(os.getenv("FORGET_THRESHOLD", "0.25"))
    FORGET_MAX_PER_PASS: int = int(os.getenv("FORGET_MAX_PER_PASS", "50"))
    FORGET_DRY_RUN: bool = _bool_env("FORGET_DRY_RUN", True)
    FORGET_PROMPT_CHAR_BUDGET: int = int(os.getenv("FORGET_PROMPT_CHAR_BUDGET", "4000"))

    # ── Variance-gated state injection (KEEPER_PATCH_01 C) ────────────────
    # Full current set so merge is behaviour-neutral; trim from state_variance.py.
    STATE_FIELDS_INJECTED: list[str] = _str_list(
        "STATE_FIELDS_INJECTED",
        ["arousal", "valence", "curiosity", "fatigue"],
    )
    DRIVES_INJECTED: list[str] = _str_list(
        "DRIVES_INJECTED",
        ["understand", "connect", "create", "resolve", "express"],
    )

    # ── action_bias (KEEPER_PATCH_01 D) ───────────────────────────────────
    ACTION_BIAS_ENABLED: bool = _bool_env("ACTION_BIAS_ENABLED", True)
    ACTION_BIAS_STREAK_LEN: int = int(os.getenv("ACTION_BIAS_STREAK_LEN", "3"))
    STANDING_CONSTRAINTS_MAX: int = int(os.getenv("STANDING_CONSTRAINTS_MAX", "3"))
    ACTION_BIAS_EVAL_MODEL: str = os.getenv(
        "ACTION_BIAS_EVAL_MODEL",
        os.getenv("MODEL_LOW", "deepseek/deepseek-v4-flash-0731"),
    )
    ACTION_BIAS_MIN_REPLY_TOKENS: int = int(os.getenv("ACTION_BIAS_MIN_REPLY_TOKENS", "20"))

    def ensure_dirs(self) -> None:
        """Create data directories if they don't exist. Seed SOUL.md from the sample."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.midterm_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_db_path.parent.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "state").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "logs").mkdir(parents=True, exist_ok=True)
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.soul_file_path.exists():
            example = Path(__file__).resolve().parent.parent / "SOUL.md.example"
            if example.exists():
                self.soul_file_path.write_text(
                    example.read_text(encoding="utf-8"), encoding="utf-8"
                )


config = Settings()
