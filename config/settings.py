"""
Configuration — env-driven settings for the entire companion.
 
Loads from .env via python-dotenv. All paths are resolved relative to
the project root unless absolute paths are given.
"""
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
 
 
class Settings:
    """Singleton-ish config object. Access via `config`."""
 
    # ── LLM ───────────────────────────────────────────────────────────────
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "claude-sonnet-4-5")
 
    # ── Telegram ──────────────────────────────────────────────────────────
    telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    allowed_user_ids: set[int] = _int_set("TELEGRAM_ALLOWED_USER_IDS")
 
    # ── Web search ────────────────────────────────────────────────────────
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
 
    # ── Embeddings ────────────────────────────────────────────────────────
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
 
    # ── Paths ─────────────────────────────────────────────────────────────
    data_dir: str = os.getenv("DATA_DIR", "./data")
    soul_file_path: Path = _path("SOUL_FILE_PATH", "./SOUL.md")
    midterm_db_path: Path = _path("MIDTERM_DB_PATH", "./data/episodic.db")
    chroma_db_path: Path = _path("CHROMA_DB_PATH", "./data/chroma")
 
    # ── Background loop intervals (minutes) ───────────────────────────────
    consolidation_interval: int = int(os.getenv("CONSOLIDATION_INTERVAL_MIN", "30"))
    self_model_interval: int = int(os.getenv("SELF_MODEL_INTERVAL_MIN", "60"))
    autonomous_interval: int = int(os.getenv("AUTONOMOUS_INTERVAL_MIN", "5"))
    grounding_interval: int = int(os.getenv("GROUNDING_INTERVAL_SEC", "30"))
 
    # ── Session ───────────────────────────────────────────────────────────
    session_timeout_minutes: float = float(os.getenv("SESSION_TIMEOUT_MIN", "15"))
    working_memory_capacity: int = int(os.getenv("WORKING_MEMORY_CAPACITY", "20"))
 
    def ensure_dirs(self) -> None:
        """Create data directories if they don't exist."""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        self.midterm_db_path.parent.mkdir(parents=True, exist_ok=True)
        Path(self.chroma_db_path).parent.mkdir(parents=True, exist_ok=True)
        (Path(self.data_dir) / "state").mkdir(parents=True, exist_ok=True)
 
 
config = Settings()