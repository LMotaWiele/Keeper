# Memory

Three stores plus a consolidator. The agent talks to `memory.memory_system`; it should not reach into a layer unless it is a tool.

## Working (`memory/working.py`)

Per-user list, default capacity 20 (`WORKING_MEMORY_CAPACITY`). Each item has a role, text, salience, and timestamp.

When the buffer is full, the **lowest-salience** item is evicted. User turns default higher than assistant turns. `boost()` raises salience if something is referenced again; `decay()` slowly lowers everything so unused context fades.

Persisted to `data/state/working_memory.json` so a process restart keeps the thread.

The runner sends this window to LangGraph as the conversation transcript. Episodic/semantic content is *not* duplicated there; it lives in the system prompt.

## Episodic (`memory/episodic.py`)

SQLite (`MIDTERM_DB_PATH`, default `data/episodic.db`). One row per experience: user input, reply, self-observation, fact, preference, opinion, research note.

Encoding stores:

- `importance` — caller-supplied salience
- `emotional_weight` — from internal state at write time
- `decay_rate` — slower when emotional weight is high
- `recall_count` / `last_recalled_at` — remembering reinforces
- `effective_strength` — retrieval rank

Decay formula (applied on a schedule):

```
base = importance × (1 + emotional_weight)
recency = exp(−decay_rate × hours_since_creation)
recall_bonus = log(1 + recall_count) × 0.15
effective_strength = clamp(base × recency + recall_bonus)
```

`forget()` deletes rows below 0.05. That is intentional identity shaping, not a vacuum-cleaner bug.

Recall of facts/preferences also increments `recall_count`, so frequently used facts resist decay.

Types: `summary`, `fact`, `event`, `preference`, `note`, `self_observation`, `opinion`, `research`.

## Semantic (`memory/semantic.py`)

ChromaDB + OpenAI embeddings, one collection per user. Documents are **patterns**, not transcripts.

Same text hashes to the same id. A repeat **reinforces** (confidence +0.1, merged source episode ids) instead of duplicating. Contradictions `invalidate()` a pattern; cleanup can delete invalidated rows later.

Search over-fetches, drops invalidated/low-confidence hits, then ranks `0.6 * relevance + 0.4 * confidence`.

## Consolidator (`memory/consolidator.py`)

Offline job, not a per-reply graph node.

1. If fewer than 3 new episodes since last run: decay + forget only (no LLM).
2. Else: LLM extracts traits / preferences / relationships / knowledge / behavioral patterns / contradictions.
3. New patterns are checked against existing semantic hits. `keep_new` invalidates the old one.
4. Decay + forget. Reset the “new since consolidation” counter.

The loop skips entirely while any user session is active.

## Tools

The model can also write and read memory explicitly:

- `save_fact` / `save_preference` → episodic, high importance
- `save_long_term_memory` / `search_long_term_memory` → semantic
- `recall_facts` → facts + preferences

`user_id` is injected by the graph if the tool schema includes it.
