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
    model_high: str = os.getenv("MODEL_HIGH", "google/gemini-3.8-flash")
    model_mid: str = os.getenv("MODEL_MID", "google/gemini-3.8-flash")
    model_low: str = os.getenv("MODEL_LOW", "z-ai/glm-5.3-flash")
    model_high_fallback: str = os.getenv("MODEL_HIGH_FALLBACK", "google/gemini-3.7-flash")
    model_mid_fallback: str = os.getenv("MODEL_MID_FALLBACK", "google/gemini-3.7-flash")
    model_low_fallback: str = os.getenv("MODEL_LOW_FALLBACK", "~deepseek/deepseek-flash-latest")
    jev_model: str = os.getenv("JEV_MODEL", "~typesafe/jev-latest")
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
    dialogue_window: int = int(os.getenv("DIALOGUE_WINDOW", "12"))
    pin_capacity: int = int(os.getenv("PIN_CAPACITY", "8"))
    pin_max_age_days: int = int(os.getenv("PIN_MAX_AGE_DAYS", "14"))

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
    FORGET_DRY_RUN: bool = _bool_env("FORGET_DRY_RUN", False)
    FORGET_PROMPT_CHAR_BUDGET: int = int(os.getenv("FORGET_PROMPT_CHAR_BUDGET", "4000"))

    # ── Variance-gated state injection (KEEPER_PATCH_01 C / PATCH_02 A2) ──
    # 72h state_trace: all affect fields std < 0.05; all five drives std > 0.15.
    # Empty affect list omits the state block. Do not restore dropped fields
    # if the autonomous loop goes quiet — that means it was firing on a constant.
    STATE_FIELDS_INJECTED: list[str] = _str_list("STATE_FIELDS_INJECTED", [])
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
        os.getenv("MODEL_LOW", "z-ai/glm-5.3-flash"),
    )
    ACTION_BIAS_MIN_REPLY_TOKENS: int = int(os.getenv("ACTION_BIAS_MIN_REPLY_TOKENS", "20"))

    # ── User-life tracker (KEEPER_PATCH_02 B) ─────────────────────────────
    COMMITMENTS_INJECTED_MAX: int = int(os.getenv("COMMITMENTS_INJECTED_MAX", "5"))
    COMMITMENT_GRACE_HOURS: int = int(os.getenv("COMMITMENT_GRACE_HOURS", "24"))
    OVERDUE_SURFACE_COOLDOWN_H: int = int(os.getenv("OVERDUE_SURFACE_COOLDOWN_H", "24"))
    WELLBEING_INFERRED_IN_TRENDS: bool = False  # do not flip; inferred is audit-only
    USER_TIMEZONE: str = os.getenv("USER_TIMEZONE", "Europe/Amsterdam")

    # ── Relational grounding (KEEPER_PATCH_03) ────────────────────────────
    # Prompt dump lands in data/logs/prompt/<turn_id>.txt (the logs/prompt tree).
    DUMP_ASSEMBLED_PROMPT: bool = _bool_env("DUMP_ASSEMBLED_PROMPT", True)
    # Roughly the character budget the commitment block used. Substitution, not growth.
    WORLD_PROMPT_CHAR_BUDGET: int = int(os.getenv("WORLD_PROMPT_CHAR_BUDGET", "800"))
    # Opt out of consolidation as a world-slot write path. The forgetting
    # predicate is untouched and still gated by FORGET_DRY_RUN.
    WORLD_CONSOLIDATION_ENABLED: bool = _bool_env("WORLD_CONSOLIDATION_ENABLED", True)
    WORLD_CONSOLIDATION_MAX_DROP_RATE: float = float(
        os.getenv("WORLD_CONSOLIDATION_MAX_DROP_RATE", "0.10")
    )

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
