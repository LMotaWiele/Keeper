# Architecture

Keeper is a five-pillar companion that keeps an inner state between turns, then injects that state into every LLM call. The LangGraph graph is only the reasoning step. Continuity lives in memory, goals, self-model, and background loops.

## Runtime shape

```
Telegram (allowlisted user)
        │
        ▼
   tg/bot.py                 message + commands
        │
        ▼
   agent/runner.py
        │
        ├─ companion.process()      build prompt from all pillars
        ├─ LangGraph ainvoke()      reason ⇄ tools → reply
        └─ companion.post_process() update state, memory, opinions, goals
```

Startup uses a **single event loop**. python-telegram-bot's `post_init` / `post_shutdown` start and stop `ConsciousArchitecture` on that same loop so background tasks survive. Separate `asyncio.run()` calls would tear those tasks down.

## Five pillars

| Pillar | Package | Role |
|--------|---------|------|
| 1 Memory | `memory/` | Working window, episodic SQLite, semantic Chroma, consolidator |
| 2 Internal state | `core/internal_state.py`, `core/drive_system.py` | Affect vector + drives; event-driven; injected into the prompt |
| 3 Environment | `environment/` | Time, optional RSS, system health; time sense between sessions |
| 4 Goals | `goals/` | Terminal orientations, instrumental goals, autonomous pursuit |
| 5 Self-model | `core/self_model.py`, `core/opinions.py` | Observed identity + opinion origins (anti-sycophancy) |

Keeper-spec extras on the same singleton (`core/loop.py`):

- `CodebaseIndex` — AST scan of this repo, used for self-reflection
- `SearchBudget` / `APIBudget` — daily quotas that raise fatigue as they drain
- `UserLifeTracker` — user commitments and wellbeing, not chat engagement
- `SelfTheorizer` — offline improvement proposals
- `FutureSimulator` — gated “what if” before some autonomous actions
- `ResearchEngine` — web search → EXTERNAL opinions

## Message path

1. Allowlist check. Typing indicator.
2. `process()` marks the session active, estimates novelty/complexity, updates internal state, stores the turn in working + episodic memory, then builds a system prompt.
3. The graph receives that prompt plus the working-memory transcript. `reason` may call tools (web search, memory read/write) and loops until it produces a reply.
4. `post_process()` records the reply, API usage, a self-observation, opinion detection, and goal progress. Full state is snapshotted every 10 messages and on shutdown.

`SOUL.md` is re-read every turn. Identity edits take effect without a restart.

## Graph

```
load_context → reason ⇄ tool_executor → finalize → END
```

`load_context` is a passthrough: the companion loop already built the prompt. Consolidation is **not** a graph node; it runs in the background so replies stay unblocked.

## Background loops

Started in `ConsciousArchitecture.startup()`:

| Loop | Default | Gate |
|------|---------|------|
| Environmental grounding | 30s wake | none |
| Memory consolidation | 30 min | skipped while a session is active |
| Self-model update | 60 min | needs ≥5 new self-observations |
| Autonomous goal pursuit | 5 min | skipped while a session is active, or if fatigue is high |

Intervals come from env (`CONSOLIDATION_INTERVAL_MIN`, etc.). Session end is driven by `TimeStream` silence timeout (`SESSION_TIMEOUT_MIN`, default 60). That timeout also clears the companion’s session flag so background work can resume.

Autonomous actions stay **internal** (research, notes, self-theorizing). They do not message the user unprompted.

## Persistence

Under `data/` (created on first run):

| Store | Contents |
|-------|----------|
| `data/episodic.db` | SQLite episodes |
| `data/chroma/` | Semantic patterns (vectors) |
| `data/state/*.json` | Internal state, self-model, opinions, goals, environment, budgets, user-life, theorizer, working memory |

## Prompt assembly

Each turn concatenates, in order:

1. `SOUL.md`
2. Episodic recall + semantic search
3. Internal state and drives
4. Self-model, hypotheses, opinion awareness
5. Active goals
6. Time / environment
7. Search and API budget remaining
8. Architecture summary (only if the user asks about how Keeper is built)
9. Open user-life commitments
10. Tool-use reminder

## Package map

| Path | Responsibility |
|------|----------------|
| `main.py` | Logging, Telegram polling, lifecycle hooks |
| `config/settings.py` | Env-driven settings |
| `agent/` | LangGraph + runner |
| `core/loop.py` | Orchestrator singleton |
| `memory/` | Three memory layers + consolidator |
| `environment/` | Grounding streams, time sense, future sim |
| `goals/` | Goal system, autonomy, research, self-theorizing |
| `tg/` | Telegram interface |
| `tools/` | Web search + explicit memory tools |
| `SOUL.md` | Identity prompt |
