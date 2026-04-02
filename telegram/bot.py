"""
Telegram bot — the companion's face.

Handles incoming messages, enforces the allowlist, shows typing indicators,
and routes everything through the LangGraph agent.
"""
from __future__ import annotations

import asyncio
import logging

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from agent.runner import run_agent
from config.settings import config
from memory.mid_term import mid_term
from memory.short_term import short_term

log = logging.getLogger(__name__)


# ── Guards ────────────────────────────────────────────────────────────────────

def allowed(update: Update) -> bool:
    """Only respond to allowlisted user IDs (SOUL.md: "Remember you're a guest")."""
    uid = update.effective_user.id if update.effective_user else None
    return uid in config.allowed_user_ids


# ── Handlers ──────────────────────────────────────────────────────────────────

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return
    await update.message.reply_text(
        "Hey. I'm here.\n\nTalk to me — I'll remember what matters."
    )


async def clear_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear short-term (in-session) memory."""
    if not allowed(update):
        return
    uid = update.effective_user.id
    short_term.clear(uid)
    await update.message.reply_text("Short-term memory cleared. Fresh start.")


async def memory_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Show what's stored in mid-term memory."""
    if not allowed(update):
        return
    uid = update.effective_user.id
    block = await mid_term.format_for_prompt(uid)
    if block:
        await update.message.reply_text(block)
    else:
        await update.message.reply_text("Nothing stored in mid-term memory yet.")


async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update) or not update.message or not update.message.text:
        return

    uid = update.effective_user.id
    user_text = update.message.text.strip()

    # Show typing indicator
    await ctx.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING,
    )

    try:
        reply = await run_agent(uid, user_text)
    except Exception as exc:
        log.exception("Agent error for user %s", uid)
        reply = f"Something went wrong on my end. ({type(exc).__name__})"

    # Split long replies (Telegram max ~4096 chars per message)
    chunk_size = 4000
    for i in range(0, len(reply), chunk_size):
        chunk = reply[i : i + chunk_size]
        await update.message.reply_text(chunk)

        if i + chunk_size < len(reply):
            # Brief pause between chunks + re-show typing
            await asyncio.sleep(0.5)
            await ctx.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action=ChatAction.TYPING,
            )


# ── Bot setup ─────────────────────────────────────────────────────────────────

def build_application() -> Application:
    app = (
        Application.builder()
        .token(config.telegram_token)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear", clear_cmd))
    app.add_handler(CommandHandler("memory", memory_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    return app
