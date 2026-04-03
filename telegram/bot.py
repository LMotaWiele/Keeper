"""
Telegram bot — the companion's face.

Updated to integrate with the ConsciousArchitecture:
  - Signals session start/end to the environment layer
  - Uses companion.clear_session() for /clear
  - Adds /status command to inspect pillar states
  - Adds /goals command to see active goals
"""
from __future__ import annotations

import asyncio
import json
import logging

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from agent.runner import run_agent
from config.settings import config
from core.loop import companion

log = logging.getLogger(__name__)


# ── Guards ────────────────────────────────────────────────────────────────

def allowed(update: Update) -> bool:
    uid = update.effective_user.id if update.effective_user else None
    return uid in config.allowed_user_ids


# ── Handlers ──────────────────────────────────────────────────────────────

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return
    await update.message.reply_text(
        "Hey. I'm here.\n\nTalk to me — I'll remember what matters."
    )


async def clear_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear working memory and reset session state."""
    if not allowed(update):
        return
    uid = update.effective_user.id
    companion.clear_session(uid)
    await update.message.reply_text("Working memory cleared. Fresh start.")


async def memory_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Show what's stored in episodic memory."""
    if not allowed(update):
        return
    uid = update.effective_user.id
    from memory.episodic import episodic
    block = await episodic.format_for_prompt(uid)
    if block:
        await update.message.reply_text(block)
    else:
        await update.message.reply_text("Nothing stored in episodic memory yet.")


async def status_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the current state of all pillars."""
    if not allowed(update):
        return
    status = companion.status()
    state = status["internal_state"]
    drives = status["drives"]

    lines = [
        "📊 **Companion Status**",
        "",
        f"**Engagement:** {state['engagement']}"  ,
        f"**Mode:** {state['mode']}",
        f"**Arousal:** {state['arousal']:.2f}  "
        f"**Curiosity:** {state['curiosity']:.2f}  "
        f"**Fatigue:** {state['fatigue']:.2f}",
        "",
        f"**Self-model:** v{status['self_model_version']}",
        f"**Active goals:** {status['active_goals']}",
        f"**Completed goals:** {status['completed_goals']}",
        f"**Background tasks:** {status['background_tasks']}",
        "",
        "**Drives:**",
    ]
    for name, intensity in sorted(drives.items(), key=lambda x: -x[1]):
        bar = "█" * int(intensity * 10) + "░" * (10 - int(intensity * 10))
        lines.append(f"  {name}: {bar} {intensity:.2f}")

    await update.message.reply_text("\n".join(lines))


async def goals_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Show active goals."""
    if not allowed(update):
        return
    active = companion.goals.active_instrumental
    if not active:
        await update.message.reply_text(
            "No specific goals right now — just the terminal orientations "
            "(understand, create, resolve, connect)."
        )
        return

    lines = ["🎯 **Active Goals**", ""]
    for g in sorted(active, key=lambda g: g.salience, reverse=True):
        progress = f"{g.progress:.0%}"
        parent = f" → {g.parent_goal}" if g.parent_goal else ""
        lines.append(f"**{g.name}**{parent} [{progress}]")
        lines.append(f"  {g.description}")
        if g.progress_notes:
            lines.append(f"  _Last: {g.progress_notes[-1]}_")
        lines.append("")

    await update.message.reply_text("\n".join(lines))


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

    # Split long replies (Telegram max ~4096 chars)
    chunk_size = 4000
    for i in range(0, len(reply), chunk_size):
        chunk = reply[i : i + chunk_size]
        await update.message.reply_text(chunk)

        if i + chunk_size < len(reply):
            await asyncio.sleep(0.5)
            await ctx.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action=ChatAction.TYPING,
            )


# ── Bot setup ─────────────────────────────────────────────────────────────

def build_application() -> Application:
    app = (
        Application.builder()
        .token(config.telegram_token)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear", clear_cmd))
    app.add_handler(CommandHandler("memory", memory_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("goals", goals_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    return app