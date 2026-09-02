"""Entrypoint — one event loop for companion background tasks and Telegram polling."""
from __future__ import annotations

import argparse
import logging
import shutil
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
        model_high=config.model_high,
        model_mid=config.model_mid,
        model_low=config.model_low,
    )


async def on_bot_shutdown(application) -> None:
    """post_shutdown: persist state and stop background loops on the same loop."""
    log = structlog.get_logger()
    log.info("Shutting down…")
    await companion.shutdown()
    log.info("Goodbye.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keeper companion")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete ./data/ before starting (no migration; all state is disposable)",
    )
    args = parser.parse_args()

    setup_logging()
    log = structlog.get_logger()

    if args.reset:
        data = config.data_dir
        if data.exists():
            shutil.rmtree(data)
            log.info("Wiped data directory", path=str(data))
        config.ensure_dirs()

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