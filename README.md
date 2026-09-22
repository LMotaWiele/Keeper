# Keeper

A personal AI companion with an inner life — memory that forgets, opinions it can defend, and goals it pursues while you are away.

Keeper is not a chatbot wrapper. Each turn rebuilds a system prompt from five pillars (memory, affect, environment, goals, self-model), then a small LangGraph loop reasons and uses tools. Between conversations, background processes consolidate memories, update a behavioral self-model, and research topics so the next session is not a blank slate.

Talk to it from your phone. It runs on your machine.

## Why this exists

Most agents are a prompt plus tools. They have no state between turns except a transcript. Keeper treats that as the interesting problem:

- **Affect is real in the prompt.** Arousal, curiosity, fatigue, and drives are not labels — they change how the model is instructed to respond, and they update from what just happened.
- **Forgetting is a feature.** Episodes decay. Recalling them reinforces them. What survives is part of identity.
- **Opinions have origins.** Independent, adopted, or researched. An independence score flags sycophantic drift.
- **Autonomy stays internal.** It does not ping you. Research notes and self-theorizing show up later as better context. Architecture proposals queue for you; nothing applies them automatically.

Identity lives in local `SOUL.md` (gitignored) and is re-read every turn. Start from [`SOUL.md.example`](SOUL.md.example). First run copies the sample if `SOUL.md` is missing.

## Architecture

```
Telegram (allowlisted)
        │
        ▼
   tg/bot.py
        │
        ▼
   agent/runner.py
        │
        ├─ companion.process()       assemble prompt from all pillars
        ├─ LangGraph                 reason ⇄ tools → reply
        │                            (force_answer if the tool cap is hit)
        └─ companion.post_process()  update state, memory, opinions, goals
```

| Pillar | What it does |
|--------|----------------|
| **Memory** | Working window (salience eviction), episodic SQLite with decay, semantic Chroma + local fastembed |
| **Internal state** | Affect vector + drives, event-driven, persisted across restarts |
| **Environment** | Time passing, session rhythm, optional RSS, system health |
| **Goals** | Terminal orientations + bounded/unbounded instrumental goals + idle autonomous pursuit |
| **Self-model** | Identity learned from observed behavior; opinion registry against mirroring |

All LLM traffic goes through OpenRouter (`core/llm.py`) with three cost tiers. Spend is recorded on every call. Daily ceiling: €10 (`DAILY_BUDGET_EUR`). Conversation degrades HIGH → MID → LOW as the budget fills; it never hard-stops. Design reasoning, memory details, and goal/self-model notes: [`docs/`](docs/).

## Stack

Python 3.11+ · LangGraph · OpenRouter (`ChatOpenAI`) · ChromaDB · SQLite · SearXNG · fastembed · python-telegram-bot

## Prerequisites

1. **OpenRouter API key** (`OPENROUTER_API_KEY`). All chat models are reached through `https://openrouter.ai/api/v1`. Defaults: `MODEL_HIGH=google/gemini-3.8-flash`, `MODEL_MID=google/gemini-3.8-flash`, `MODEL_LOW=z-ai/glm-5.3-flash` (LOW fallback `~deepseek/deepseek-flash-latest`). Typed skip-gates use the Decisions API (`JEV_MODEL=~typesafe/jev-latest`).
2. **Telegram bot token** and your user id (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_ALLOWED_USER_IDS`).
3. **A local SearXNG instance** for web search. Default query URL: `http://localhost:8081/search?q=<query>`.

   SearXNG must accept JSON and must not rate-limit local callers. In its `settings.yml`:

   ```yaml
   search:
     formats:
       - html
       - json
   server:
     limiter: false
   ```

   Without `json` in `search.formats` every request returns HTML. With the limiter on, local burst traffic gets 429s and search silently degrades.

4. **Local embeddings** via `fastembed` (`BAAI/bge-small-en-v1.5` by default). First run downloads ~130 MB into `data/fastembed`.

No Anthropic, Tavily, or OpenAI keys are required.

## Quick start

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then set OPENROUTER_API_KEY, TELEGRAM_BOT_TOKEN,
                                  # TELEGRAM_ALLOWED_USER_IDS
cp SOUL.md.example SOUL.md        # then edit the identity; first run copies this if missing
python main.py
```

Wipe stored state (no migration — all data is disposable; does not touch `SOUL.md`):

```bash
python main.py --reset
```

First run creates `./data/` (SQLite, Chroma, JSON state, embedding cache, rotating logs). That directory and live `SOUL.md` are gitignored.

Idle sessions time out after 15 minutes (`SESSION_TIMEOUT_MIN`) so background consolidation and autonomy can run.

### Telegram

| Command | What it does |
|---------|----------------|
| `/start` | Wake the companion |
| `/memory` | Show episodic memory |
| `/status` | Affect, drives, goals, self-model version, API spend by task, search health |
| `/goals` | Active goals (ceilings for unbounded, done-when for bounded) |
| `/actions` | Elo tables for unbounded-goal action types |
| `/diag` | Read-only dump of instance, sessions, goals, and loops |
| `/clear` | Wipe the working-memory window |

Everyone else is silently ignored (`TELEGRAM_ALLOWED_USER_IDS`).

### Always-on (Linux)

Edit paths in `companion.service`, then:

```bash
sudo cp companion.service /etc/systemd/system/companion@$USER.service
sudo systemctl enable --now companion@$USER
journalctl -u companion@$USER -f
```

## Layout

```
Keeper/
├── SOUL.md.example         sample identity (copy to gitignored SOUL.md)
├── .env.example            OpenRouter + Telegram + optional knobs
├── main.py                 process entry
├── companion.service       systemd unit template
├── config/                 env-driven settings
├── agent/                  LangGraph + runner
├── core/                   orchestrator, state, self-model, opinions, LLM factory
├── memory/                 working / episodic / semantic / consolidator
├── environment/            time, streams, future simulation
├── goals/                  goals, autonomy, research, self-theorizing, ratings
├── tg/                     Telegram bot
├── tools/                  web search, memory, introspection
├── scripts/                optional diagnostic harnesses
├── docs/                   architecture and design notes
└── test_*.py               unit tests
```

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
