# Virtual Companion Agent

A soul-driven LangGraph agent with tiered memory, web search, and a Telegram interface. Runs on your PC, talks to you through your phone.

---

## Architecture

```
Telegram (phone)
      │
      ▼
 telegram/bot.py          ← message handler, allowlist, typing indicator
      │
      ▼
 agent/runner.py          ← wraps graph invocation, manages short-term window
      │
      ▼
 agent/graph.py           ← LangGraph state machine
      │
  ┌───┴────────────────────────┐
  │                            │
load_context               tools/
  │  • reads SOUL.md        web_search.py     (Tavily)
  │  • loads mid-term ctx   memory_tools.py   (save/recall facts)
  │  • semantic LTM search      │
  ▼                            │
reason (LLM) ◄────────────────┘
  │  • decides: reply or call tool
  │  • loops until done
  ▼
consolidate
  • extracts new facts/preferences
  • saves session summary to long-term memory
```

### Memory layers

| Layer | Storage | Lifetime | What goes there |
|-------|---------|----------|-----------------|
| **Short-term** | RAM (deque) | Session | Last 20 messages — conversation context |
| **Mid-term** | SQLite | Weeks–months | Facts, preferences, notable events |
| **Long-term** | ChromaDB (vectors) | Indefinite | Session summaries, important discussions, semantic search |

---

## Setup

### 1. Prerequisites

- Python 3.11+
- A Telegram bot token (from [@BotFather](https://t.me/BotFather))
- Your Telegram user ID (send a message to [@userinfobot](https://t.me/userinfobot))
- Anthropic API key
- Tavily API key (free tier at [tavily.com](https://tavily.com))
- OpenAI API key (used only for cheap text embeddings)

### 2. Clone & install

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```env
ANTHROPIC_API_KEY=sk-ant-...
TELEGRAM_BOT_TOKEN=123456:ABC-...
TELEGRAM_ALLOWED_USER_IDS=123456789   # your Telegram user ID
TAVILY_API_KEY=tvly-...
OPENAI_API_KEY=sk-...                 # for embeddings only
```

Everything else has sensible defaults.

### 4. Run

```bash
python main.py
```

On first run, it will create `./data/` with the SQLite and ChromaDB stores.

---

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Wake the companion |
| `/memory` | Show what's stored in mid-term memory |
| `/clear` | Clear the current session's short-term memory |

Everything else is just a message — talk naturally.

---

## Running as a background service (Linux)

So the companion is always on when your PC is running:

```bash
# Edit the paths in companion.service first
sudo cp companion.service /etc/systemd/system/companion@$USER.service
sudo systemctl daemon-reload
sudo systemctl enable companion@$USER
sudo systemctl start companion@$USER

# View logs
journalctl -u companion@$USER -f
```

---

## Project structure

```
companion/
├── SOUL.md                  ← The companion's identity (edit freely)
├── main.py                  ← Entry point
├── requirements.txt
├── .env.example
├── companion.service        ← systemd unit file
│
├── config/
│   └── settings.py          ← Env-driven config
│
├── agent/
│   ├── graph.py             ← LangGraph state machine
│   └── runner.py            ← Graph invocation wrapper
│
├── memory/
│   ├── short_term.py        ← In-process message window
│   ├── mid_term.py          ← SQLite episodic store
│   └── long_term.py         ← ChromaDB vector store
│
├── tools/
│   ├── web_search.py        ← Tavily web search
│   └── memory_tools.py      ← Memory read/write tools for the agent
│
├── telegram/
│   └── bot.py               ← python-telegram-bot handler
│
└── data/                    ← Auto-created on first run
    ├── midterm.db
    └── chroma/
```

---

## Extending the companion

### Add a new tool

1. Create a `@tool` function in `tools/` (or add it to `memory_tools.py`)
2. Import it in `tools/__init__.py` and add it to `ALL_TOOLS`

The agent will automatically have access to it on the next run.

### Add a new Telegram command

In `telegram/bot.py`, add a handler function and register it:

```python
async def my_cmd(update, ctx):
    ...

app.add_handler(CommandHandler("mycommand", my_cmd))
```

### Change the LLM

Edit `LLM_MODEL` in `.env`. Any model supported by `langchain-anthropic` works.

### Edit the soul

Just edit `SOUL.md`. It's re-read at the start of every conversation turn — changes take effect immediately without restart. Per SOUL.md itself: if the agent edits this file, it will tell you.

---

## Tips

- **Privacy**: `TELEGRAM_ALLOWED_USER_IDS` is your hard allowlist. The bot will silently ignore everyone else.
- **Cost**: Most turns use ~1k–3k tokens. Embeddings (OpenAI) are very cheap — a year of daily use costs a few dollars.
- **Offline tolerance**: The bot uses long-polling, so it handles your PC sleeping/waking gracefully. Set `drop_pending_updates=True` (already set) if you don't want it processing a backlog on wake.
- **Memory growth**: ChromaDB is append-only by default. Old summaries stay forever, which is the point. If you ever want to prune, use `long_term.delete(user_id, doc_id)`.
