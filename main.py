"""
Main entrypoint — initialises the conscious architecture and starts the Telegram bot.

Fixed for Windows compatibility:
  - Everything runs in a single event loop (background tasks survive)
  - Uses python-telegram-bot's post_init/post_shutdown hooks instead
    of separate asyncio.run() calls that each create/destroy their own loop

Startup order:
  1. Logging
  2. Config (ensure dirs)
  3. Companion startup via post_init (init memory, load state, start background loops)
  4. Telegram bot polling (same event loop)

Shutdown:
  - Companion saves all state to disk via post_shutdown
  - Background loops stop gracefully
"""
from __future__ import annotations

import asyncio
import logging
import sys

import structlog

from config.settings import config
from core.loop import companion
from tg.bot import build_application


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


async def on_bot_startup(application) -> None:
    """
    Called by python-telegram-bot's post_init hook — runs inside
    the bot's own event loop, so background tasks stay alive.
    """
    log = structlog.get_logger()
    log.info("Companion starting up…")

    config.ensure_dirs()

    await companion.startup()

    log.info(
        "Architecture ready",
        db=str(config.midterm_db_path),
        chroma=str(config.chroma_db_path),
        users=config.allowed_user_ids,
        model=config.llm_model,
    )


async def on_bot_shutdown(application) -> None:
    """
    Called by python-telegram-bot's post_shutdown hook — same loop,
    background tasks are still reachable for graceful cancellation.
    """
    log = structlog.get_logger()
    log.info("Shutting down…")
    await companion.shutdown()
    log.info("Goodbye.")


def main() -> None:
    setup_logging()
    log = structlog.get_logger()

    # Build the Telegram application with lifecycle hooks
    app = build_application()
    app.post_init = on_bot_startup
    app.post_shutdown = on_bot_shutdown

    log.info("Telegram bot polling… (Ctrl+C to stop)")

    # run_polling() creates ONE event loop and runs everything inside it:
    #   post_init (companion startup + background loops)
    #   → polling (message handling)
    #   → post_shutdown (companion save + cleanup)
    app.run_polling(
        allowed_updates=["message"],
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()