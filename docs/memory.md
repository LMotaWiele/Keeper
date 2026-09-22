# Memory

Three stores plus a consolidator. The agent talks to `memory.memory_system`; it should not reach into a layer unless it is a tool.

## Working (`memory/working.py`)

Two rings, default total 20 (`WORKING_MEMORY_CAPACITY` = `DIALOGUE_WINDOW` 12 + `PIN_CAPACITY` 8).

- **Dialogue ring** — last 12 messages, chronological. Never salience-evicted. Assistant replies stay here even at salience 0.5.
- **Pins** — up to 8 older high-salience *user* turns that do not overlap the dialogue ring. `decay()` runs on session end and on the consolidator idle tick. Pins older than `PIN_MAX_AGE_DAYS` (14) with no recent use are dropped. Session timeout does **not** wipe the dialogue ring.

Each LangChain message is prefixed with user-local time (`[2026-09-07 16:39 CEST]`; pins: `[earlier · 2026-09-07 CEST]`).

Persisted to `data/state/working_memory.json`. Episodic/semantic content is de-duped against this window before it enters the system prompt. See `docs/context-time-spec.md`.

## Episodic (`memory/episodic.py`)

SQLite (`MIDTERM_DB_PATH`, default `data/episodic.db`). One row per experience: user input, reply, self-observation, fact, preference, opinion, research note.

Encoding stores:

- `importance` — caller-supplied salience
- `emotional_weight` — from internal state at write time
- `decay_rate` — slower when emotional weight is high
- `recall_count` / `last_recalled_at` — remembering reinforces (explicit `recall_facts` / tools, **not** automatic prompt injection)
- `effective_strength` — retrieval rank
- `event_at` / `event_local_date` — absolute time of the referred event (`USER_TIMEZONE`). Relative phrases in `content` are left as written; display prefixes `[2026-09-06 CEST · 12d ago]`.

Decay formula (applied on a schedule):

```
base = importance × (1 + emotional_weight)
recency = exp(−decay_rate × hours_since_creation)
recall_bonus = log(1 + recall_count) × 0.15
effective_strength = clamp(base × recency + recall_bonus)
```

`forget()` deletes rows below 0.05. That is intentional identity shaping, not a vacuum-cleaner bug.

`recall_facts` increments `recall_count`, so explicitly retrieved facts resist decay. Prompt assembly only bumps `used_count`.

`format_for_prompt` prefers facts/preferences, drops `User said:` / `Responded:` / `self_observation` lines that overlap the working-memory window, and caps self-observations at 2.

Types: `summary`, `fact`, `event`, `preference`, `note`, `self_observation`, `opinion`, `research`.

## Semantic (`memory/semantic.py`)

ChromaDB + local `fastembed` embeddings (`BAAI/bge-small-en-v1.5`, 384-d), collection `semantic_v2_user_{id}` so old 1536-d directories cannot collide. Documents are **patterns**, not transcripts. Embedding runs in a worker thread so Telegram polling is not blocked.

Same text hashes to the same id. A repeat **reinforces** (confidence +0.1, merged source episode ids) instead of duplicating. Contradictions `invalidate()` a pattern; cleanup can delete invalidated rows later.

Search over-fetches, drops invalidated/low-confidence hits, then ranks `0.6 * relevance + 0.4 * confidence`.

## Consolidator (`memory/consolidator.py`)

Offline job, not a per-reply graph node.

1. If fewer than 1 new episode since last run, unless it has been >6 hours: decay + forget only (no LLM).
2. Else: LLM extracts traits / preferences / relationships / knowledge / behavioral patterns / contradictions.
3. New patterns are checked against existing semantic hits. `keep_new` invalidates the old one.
4. Decay + forget. Reset the “new since consolidation” counter.

The loop skips **that user** while their session is active. Background LLM work is also skipped when `APIBudget.allows(LOW, background=True)` is false; decay still runs. `run_cycle` reports `reason`: `ran` / `skipped_no_new` / `skipped_budget`.

## Tools

The model can also write and read memory explicitly:

- `save_fact` / `save_preference` → episodic, high importance
- `save_long_term_memory` / `search_long_term_memory` → semantic
- `recall_facts` → facts + preferences

`user_id` is injected by the graph if the tool schema includes it.
