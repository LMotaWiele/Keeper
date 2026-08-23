"""Entrypoint — one event loop for companion background tasks and Telegram polling."""
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
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


async def on_bot_startup(application) -> None:
    """post_init: start companion on the bot's event loop so background tasks survive."""
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
    """post_shutdown: persist state and stop background loops on the same loop."""
    log = structlog.get_logger()
    log.info("Shutting down…")
    await companion.shutdown()
    log.info("Goodbye.")


def main() -> None:
    setup_logging()
    log = structlog.get_logger()

    app = build_application()
    app.post_init = on_bot_startup
    app.post_shutdown = on_bot_shutdown

    log.info("Telegram bot polling… (Ctrl+C to stop)")

    app.run_polling(
        allowed_updates=["message"],
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()