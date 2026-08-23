# Keeper

A personal AI companion with an inner life — memory that forgets, opinions it can defend, and goals it pursues while you are away.

Keeper is not a chatbot wrapper. Each turn rebuilds a system prompt from five pillars (memory, affect, environment, goals, self-model), then a small LangGraph loop reasons and uses tools. Between conversations, background processes consolidate memories, update a behavioral self-model, and research topics so the next session is not a blank slate.

Talk to it from your phone. It runs on your machine.

## Why this exists

Most agents are a prompt plus tools. They have no state between turns except a transcript. Keeper treats that as the interesting problem:

- **Affect is real in the prompt.** Arousal, curiosity, fatigue, and drives are not labels — they change how the model is instructed to respond, and they update from what just happened.
- **Forgetting is a feature.** Episodes decay. Recalling them reinforces them. What survives is part of identity.
- **Opinions have origins.** Independent, adopted, or researched. An independence score flags sycophantic drift.
- **Autonomy stays internal.** It does not ping you. Research notes and self-theorizing show up later as better context.

Identity lives in [`SOUL.md`](SOUL.md) and is re-read every turn.

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
        ├─ process()       assemble prompt from all pillars
        ├─ LangGraph       reason ⇄ tools → reply
        └─ post_process()  update state, memory, opinions, goals
```

| Pillar | What it does |
|--------|----------------|
| **Memory** | Working window (salience eviction), episodic SQLite with decay, semantic Chroma patterns |
| **Internal state** | Affect vector + drives, event-driven, persisted across restarts |
| **Environment** | Time passing, session rhythm, optional RSS, system health |
| **Goals** | Terminal orientations + instrumental goals + idle autonomous pursuit |
| **Self-model** | Identity learned from observed behavior; opinion registry against mirroring |

Design reasoning, memory details, and goal/self-model notes: [`docs/`](docs/).

## Stack

Python 3.11+ · LangGraph · Claude · ChromaDB · SQLite · Tavily · python-telegram-bot

## Quick start

**You need:** a [Telegram bot token](https://t.me/BotFather), your Telegram user id, an Anthropic API key, a Tavily key, and an OpenAI key (embeddings only).

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then fill in keys
python main.py
```

First run creates `./data/` (SQLite, Chroma, JSON state).

### Telegram

| Command | What it does |
|---------|----------------|
| `/start` | Wake the companion |
| `/memory` | Show episodic memory |
| `/status` | Affect, drives, goals, self-model version |
| `/goals` | Active instrumental goals |
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
├── SOUL.md                 identity prompt (edit freely)
├── main.py                 process entry
├── agent/                  LangGraph + runner
├── core/                   orchestrator, state, self-model, opinions
├── memory/                 working / episodic / semantic / consolidator
├── environment/            time, streams, future simulation
├── goals/                  goals, autonomy, research, self-theorizing
├── tg/                     Telegram bot
├── tools/                  web search + memory tools
└── docs/                   architecture and design notes
```

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
