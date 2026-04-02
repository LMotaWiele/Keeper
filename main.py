"""
Main entrypoint — initialises everything and starts the Telegram bot.
"""
from __future__ import annotations

import asyncio
import logging
import sys

import structlog

from config.settings import config
from memory.mid_term import mid_term
from telegram.bot import build_application


def setup_logging() -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )
    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


async def startup() -> None:
    log = structlog.get_logger()
    log.info("Companion starting up…")

    config.ensure_dirs()
    await mid_term.init()

    log.info("Memory layers ready", db=str(config.midterm_db_path), chroma=str(config.chroma_db_path))
    log.info("Allowed users", ids=config.allowed_user_ids)
    log.info("LLM model", model=config.llm_model)


def main() -> None:
    setup_logging()
    asyncio.run(startup())

    app = build_application()

    log = structlog.get_logger()
    log.info("Telegram bot polling… (Ctrl+C to stop)")

    app.run_polling(
        allowed_updates=["message"],
        drop_pending_updates=True,   # ignore messages sent while offline
    )


if __name__ == "__main__":
    main()
