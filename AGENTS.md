<!-- agent-memory-schema: 1 -->
# Project rules

Agent-agnostic layout and conventions. Stacks on `~/.pi/agent/AGENTS.md` (global routing + spawn policy). Prefer **AGENTS.md** — no `CLAUDE.md` / `.claude/`.

## Overview

- **Purpose**: Personal AI companion. Each turn rebuilds a prompt from five pillars (memory, affect, environment, goals, self-model), then a LangGraph loop reasons and uses tools. Telegram is the client.
- **Stack**: Python 3.11+ · `requirements.txt` · `venv/` · LangGraph · OpenRouter · ChromaDB · SQLite · python-telegram-bot. No `pyproject.toml`.
- **Package**: none. Production packages live at the repo root: `agent/`, `core/`, `memory/`, `goals/`, `environment/`, `tg/`, `tools/`, `config/`. Entry point is `main.py`.
- **Build / test**: `python -m venv venv` · `source venv/bin/activate` · `pip install -r requirements.txt` · `pytest`

## Layout

```
Keeper/
├── AGENTS.md              # this file
├── README.md              # human: what this is + how to run
├── requirements.txt
├── main.py                # entry point
├── companion.service
├── agent/                 # LangGraph runner
├── core/                  # llm, affect, events, self-model
├── memory/                # working, episodic, semantic
├── goals/
├── environment/
├── tg/                    # Telegram bot
├── tools/
├── config/
├── test_*.py              # flat tests at the repo root
├── scripts/               # diagnostics and one-offs
├── docs/                  # public design
├── docs/specs/
├── data/                  # runtime state — off-limits (gitignored)
├── .agent/skills/         # shared SKILL.md (harness-agnostic)
├── .agent/notes/          # durable findings — INDEX.md first (see rule 7)
├── .agent/explain/        # walkthroughs the harness writes for the human — never written by hand
├── .pi/settings.json      # harness wiring: project overrides only
├── .pi/agents/            # subagent definitions (tier → pinned model)
├── .pi/extensions/        # project-local extensions, if any
└── .mcp.json              # optional extra MCP
```

**Exceptions vs the vanilla template:** there is no `src/` and no `pyproject.toml`. Do not move the root packages. Tests stay flat `test_*.py` at the repo root. `data/` is runtime memory and logs (gitignored). `.env` and `SOUL.md` are secrets and identity (gitignored; samples are `.env.example` and `SOUL.md.example`). `docs/diag/`, `docs/diag2/`, and `docs/fix/` quote live sessions and are off-limits. Public design stays in the other `docs/` files.

## Hard rules

1. Production code stays in the existing root packages (`agent/`, `core/`, `memory/`, `goals/`, `environment/`, `tg/`, `tools/`, `config/`) and `main.py`. Do not introduce `src/` or relocate them.
2. Tests are flat `test_*.py` at the repo root. Do not introduce a second tree style.
3. `scripts/` = diagnostics and one-offs. Do not import them from the packages. Promote shared code into the package that already owns it.
4. Public design stays in `docs/` (`architecture.md`, `design.md`, `memory.md`, and the specs). README stays short.
5. Before a **new top-level directory**, stop and ask.
6. `.agent/` = shared knowledge/skills, portable across harnesses. `.pi/` / `.mcp.json` = harness wiring — do not merge, do not move knowledge into wiring.
7. **Notes (cheap path):** before re-deriving a diagnosis, opening a design debate, or continuing prior investigation → read `.agent/notes/INDEX.md` only, then the **one** linked file. After a finding → update that note’s front-matter, and if a pre-committed condition fired, append the resulting item to INDEX `## Active next`. Do not bulk-read notes/ or dump chat. Shape: `.agent/notes/README.md` (on demand). Prefer `docs/design/` for sealed/spec authority over notes.
8. **INDEX is the queue.** `.agent/notes/INDEX.md § Active next` is the single authority for what happens next and for standing constraints. Notes hold *conditions* (`If → Then`), never a queue. When a condition fires, the Then writes a row into INDEX. Never read a note to find out what to work on.
9. Small focused edits; match neighbors; no unsolicited docs; verify against source not README alone.
10. Commit your own work, with messages naming the task. Never push, force-push, or rewrite commits you didn't make. No production logic in scripts/notebooks/notes.
11. **Explanations are not notes.** `.agent/explain/` is written by the harness only. Do not write there. Record progress on a spec in its run record; for other work, in the commit message. Promote a finding to `.agent/notes/*-gotchas.md` only when the trap will recur across tasks in this repo.
12. Do not read or commit `.env`, `SOUL.md`, `data/`, `docs/diag/`, `docs/diag2/`, or `docs/fix/`. They hold secrets or live session traces. Samples are `.env.example` and `SOUL.md.example`.

## New module

Add a module inside the package that already owns that concern, plus a flat `test_<mod>.py` at the repo root. Architecture change → `docs/`. Durable finding → `.agent/notes/` (via INDEX + front-matter). Next action → INDEX `## Active next`.
