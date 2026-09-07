"""Telegram interface — allowlisted chat, commands, typing indicator."""
from __future__ import annotations

import asyncio
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


CHUNK = 4000  # Telegram hard limit is 4096.


def allowed(update: Update) -> bool:
    uid = update.effective_user.id if update.effective_user else None
    return uid in config.allowed_user_ids


async def _reply(update: Update, text: str) -> None:
    """Send text in Telegram-safe chunks."""
    for i in range(0, max(len(text), 1), CHUNK):
        chunk = text[i : i + CHUNK] or text
        await update.message.reply_text(chunk)
        if i + CHUNK < len(text):
            await asyncio.sleep(0.5)


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
    await _reply(update, block or "Nothing stored in episodic memory yet.")


async def status_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the current state of all pillars."""
    if not allowed(update):
        return
    status = companion.status()
    state = status["internal_state"]
    drives = status["drives"]

    lines = [
        "Companion status",
        "",
        f"Engagement: {state['engagement']}",
        f"Mode: {state['mode']}",
        f"Arousal: {state['arousal']:.2f}  "
        f"Curiosity: {state['curiosity']:.2f}  "
        f"Fatigue: {state['fatigue']:.2f}",
        "",
        f"Self-model: v{status['self_model_version']}",
        f"Active goals: {status['active_goals']}",
        f"Completed goals: {status['completed_goals']}",
        f"Background tasks: {status['background_tasks']}",
        "",
        "Drives:",
    ]
    for name, intensity in sorted(drives.items(), key=lambda x: -x[1]):
        bar = "█" * int(intensity * 10) + "░" * (10 - int(intensity * 10))
        lines.append(f"  {name}: {bar} {intensity:.2f}")

    budget = status.get("api_budget") or {}
    lines.append("")
    lines.append(
        f"API budget: €{budget.get('spent_today_eur', 0):.4f} spent, "
        f"€{budget.get('remaining_eur', 0):.2f} remaining "
        f"({budget.get('fraction_used', 0):.0%})"
    )
    lines.append(
        f"Calls today: {budget.get('calls_today', 0)}  "
        f"tier: {budget.get('conversation_tier', '?')}"
    )
    spend_by_task = budget.get("spend_by_task") or {}
    if spend_by_task:
        lines.append("Spend by task:")
        for task, amt in sorted(spend_by_task.items(), key=lambda x: -x[1]):
            lines.append(f"  {task}: €{amt:.4f}")

    health = status.get("search_health") or {}
    if health:
        lines.append(
            f"Search: {health.get('queries', 0)} queries, "
            f"{health.get('zero_result_queries', 0)} empty, "
            f"{health.get('fetch_failure_rate', 0):.0%} fetch failures"
        )

    await _reply(update, "\n".join(lines))


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

    lines = ["Active goals", ""]
    for g in sorted(active, key=lambda g: g.salience, reverse=True):
        parent = f" -> {g.parent_goal}" if g.parent_goal else ""
        if g.is_unbounded:
            lines.append(f"{g.name}{parent} (ongoing)")
            lines.append(f"  toward: {g.ceiling_description or g.description}")
            rated = [
                (t, info) for t, info in (g.action_ratings or {}).items()
                if int(info.get("n") or 0) >= 3
            ]
            if rated:
                rated.sort(key=lambda kv: -float(kv[1].get("elo") or 0))
                bits = [f"{t} ({info['elo']:.0f}, n={info['n']})" for t, info in rated[:4]]
                lines.append(f"  most effective: {', '.join(bits)}")
        else:
            done_when = ""
            if g.completion_condition:
                done_when = g.completion_condition.get("description") or g.description
            lines.append(f"{g.name}{parent}")
            lines.append(f"  done when: {done_when}")
        if g.progress_notes:
            lines.append(f"  Last: {g.progress_notes[-1]}")
        lines.append("")

    await _reply(update, "\n".join(lines))


async def diag_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Read-only diagnostic dump."""
    if not allowed(update):
        return
    await _reply(update, companion.diag_report())


async def actions_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Show Elo tables for unbounded goals."""
    if not allowed(update):
        return
    unbounded = [g for g in companion.goals.active_instrumental if g.is_unbounded]
    if not unbounded:
        await update.message.reply_text("No unbounded goals yet.")
        return
    lines = ["Action ratings", ""]
    for g in unbounded:
        lines.append(g.name)
        ratings = g.action_ratings or {}
        if not ratings:
            lines.append("  (no rated actions yet)")
        else:
            for atype, info in sorted(
                ratings.items(), key=lambda kv: -float(kv[1].get("elo") or 0)
            ):
                lines.append(
                    f"  {atype}: elo={info.get('elo', 0):.0f}  "
                    f"n={info.get('n', 0)}  spend=${info.get('spend_usd', 0):.4f}"
                )
        lines.append("")
    await _reply(update, "\n".join(lines))


async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update) or not update.message or not update.message.text:
        return

    uid = update.effective_user.id
    user_text = update.message.text.strip()

    await ctx.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING,
    )

    try:
        reply = await run_agent(uid, user_text)
    except Exception as exc:
        log.exception("Agent error for user %s", uid)
        reply = f"Something went wrong on my end. ({type(exc).__name__})"

    await _reply(update, reply)


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
    app.add_handler(CommandHandler("actions", actions_cmd))
    app.add_handler(CommandHandler("diag", diag_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    return app