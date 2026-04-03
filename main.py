"""
Main entrypoint — initialises the conscious architecture and starts the Telegram bot.

Startup order:
  1. Logging
  2. Config (ensure dirs)
  3. Companion startup (init memory, load state, start background loops)
  4. Telegram bot polling

Shutdown:
  - Companion saves all state to disk
  - Background loops stop gracefully
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys

import structlog

from config.settings import config
from core.loop import companion
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

    # Start the conscious architecture (loads state, starts background loops)
    await companion.startup()

    log.info("Architecture ready",
             db=str(config.midterm_db_path),
             chroma=str(config.chroma_db_path),
             users=config.allowed_user_ids,
             model=config.llm_model)


async def shutdown() -> None:
    log = structlog.get_logger()
    log.info("Shutting down…")
    await companion.shutdown()
    log.info("Goodbye.")


def main() -> None:
    setup_logging()
    log = structlog.get_logger()

    # Run startup
    asyncio.run(startup())

    # Build and start the Telegram bot
    app = build_application()

    log.info("Telegram bot polling… (Ctrl+C to stop)")

    try:
        app.run_polling(
            allowed_updates=["message"],
            drop_pending_updates=True,
        )
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        # Graceful shutdown — save all state
        asyncio.run(shutdown())


if __name__ == "__main__":
    main()