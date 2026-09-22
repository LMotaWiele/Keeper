# Prompt reinsertion and temporal binding

**Status:** accepted — implemented (PRs 1–3).

Two live bugs, same assembly path:

1. **Repeat-across-turns.** Keeper restates the same facts (October housing, commission status, identity claims) on consecutive replies.
2. **Frozen relative time.** A memory that says “yesterday X happened” is treated as yesterday no matter how many days have passed.

Neither is a model-quirk. Both are how `build_context` and the three memory layers feed the LLM.

---

## 1. What this does not do

| Dropped | Why |
|---|---|
| Incremental / delta system prompts across API calls | Chat Completions is stateless. Every turn is a fresh `SystemMessage` + transcript. Identity *must* be re-sent. The fix is what gets re-sent, not a client-side “already injected” flag. |
| Mutating historical `episodes.content` / working-memory `content` in place | Keep the original utterance. Add fields and a display transform. |
| Jev replacing consolidator extract / `save_fact` / opinion detection | Those generators stay. Jev is not required for this cut. |
| Wiping working memory on every session end | Continuity across a 15-minute timeout is useful. The window is wrong, not the persistence. |
| Changing LangGraph topology, tool binding, or routing v2 | Assembly-only. |
| Making `DIAG_MODE` the default | Keep it opt-in; add cheap hash logs instead. |
| Rewriting SOUL.md, self-model generators, or goal prose | Stable identity blocks may stay verbatim. Anti-echo is an instruction plus a real dialogue window, not a soul edit. |

---

## 2. Call path (every user turn)

1. `agent/runner.py` → `companion.process`
2. `core/loop.py` `process` stores the user text in **working memory** and as an episodic `"User said: …"` row, then `build_context`
3. `build_context` rebuilds `system_prompt` from current stores and copies working memory as LangChain messages
4. `agent/graph.py` `reason` / `force_answer`: `[SystemMessage(system_prompt)] + messages` on **every** LLM step
5. `post_process` stores the reply in working memory, an episodic `"Responded: …"` row, and usually a `self_observation` that quotes both sides

There is no delta path, no overlap filter against the transcript, and no date rewrite.

Production logs (`data/logs/keeper.log`, `DIAG_MODE=0`) only record block **names**, not bodies:

```
PROMPT affect: … blocks=['soul', 'memory', 'internal_state', 'self_model',
  'goals', 'environment', 'resource_awareness', 'tool_instructions', 'action_bias']
```

Same set on essentially every turn. Conversation calls on 2026-09-18 were ~5.4k–5.7k input tokens for a short Telegram exchange.

---

## 3. Issue 1 — context reinserted the same way

### 3.1 What is injected

| Block | Cadence | Volatile? |
|---|---|---|
| soul (`SOUL.md`) | Every turn | No |
| memory (episodic top-15 + semantic hits) | Every turn, full rebuild | Should be |
| internal_state / drives | Every turn | Mildly |
| self-model + opinions | Every turn | Slow |
| goals | Every turn | Slow |
| environment / time | Every turn | Clock string only |
| resource_awareness | Every turn | Mildly |
| architecture | Keyword-gated | — |
| user_life commitments | Cached at session start, re-sent every turn | Should be |
| tool_instructions | Every turn, static | No |
| action_bias tail | Every turn when sticky hyp present | Per-turn |
| working memory | Separate channel: transcript, not system prompt | Should be the dialogue |

`WorkingMemory.to_prompt_context` exists and is unused. The runner sends `to_langchain_messages`, which strips timestamps:

```python
{"role": item.role, "content": item.content}  # no timestamp
```

### 3.2 Triple encoding of the same utterance

On each turn (`loop.py` `process` / `post_process`):

- User text → working (`role=user`, salience `0.7 + novelty*0.3`) **and** episode `User said: {text}`
- Reply → working (`role=assistant`, salience `0.5`) **and** episode `Responded: {text[:500]}`
- `self_model.observe` often writes a third copy: `self_observation: Given '…', responded with: …`

Docs (`docs/memory.md`) say episodic/semantic live in the system prompt and must **not** duplicate the transcript. The write path does the opposite.

Live top-15 by `effective_strength` on 2026-09-18 was 9 `event` + 6 `self_observation` — the same afternoon's user turns, quoted again as observations. That block is headed `## What I remember about you` and sits next to the transcript that already contains those turns.

### 3.3 Working memory is not a conversation window

Documented as a salience-limited conversation window, capacity 20. Live `data/state/working_memory.json` for user `8094442922` on 2026-09-18 15:00:

- **19 user messages** dated 2026-09-07 through 2026-09-18
- **1 assistant message** (2026-09-18 12:43)
- User salience 0.92–1.0; assistant 0.5

`decay()` is defined and **never called**. `boost()` is defined and **never called**. Session timeout (`_on_env_session_end`) does **not** call `clear_working` / `clear_session`. Eviction is lowest-salience, so assistant replies leave first and 11-day-old user lines stay.

The model therefore cannot see that it already recapped October housing last turn. It *can* see the user's housing sentences (from 09-10 and 09-18) as if they were the current dialogue, plus the same facts again in episodic / facts / opinions.

### 3.4 Prompt assembly treats injection as recall

`format_for_prompt` takes the top 15 by `effective_strength`, then `increment_counters(recalled, used)` bumps `recall_count` on every turn.

Decay uses `recall_bonus = log(1 + recall_count) * 0.15`. Injection therefore **locks** the same rows in the prompt.

Live:

| id | created | recall_count | content (truncated) |
|---|---|---|---|
| 8 | 2026-09-02 00:00Z | **209** | `User said: I wouldn't call this insomnia. I just didn't want to sleep tonight.` |
| 12 | 2026-09-02 00:01Z | **207** | self_observation quoting the same “tonight” line |

Those are sixteen days old and still in the injection set because they have been injected.

### 3.5 User-visible repetition (same day)

Episode 1051, 2026-09-18 12:13Z, user:

> You seem to be repeating some things in between responses. For example, you stated 3 times a resume of the October housing situation.

Keeper had already recapped October housing in replies at 11:39, 11:45, and 11:52 (episodes 1034 / 1040 / 1047). Parallel copies in the stores:

- Working-memory user line 2026-09-18 11:39 (“I have a place for October…”)
- Fact 1039 / 1046 (“Secured a place to stay for October…”)
- Opinion 1044 (“Securing the place for October was the most immediate logistical bottleneck”)
- Matching self_observations

Keeper’s own reply (1053) named the missing de-dupe. That diagnosis is correct; it is not implemented.

---

## 4. Issue 2 — relative time is not bound

### 4.1 Where time lives, and where it is ignored

| Store | Timestamp | Used at prompt / tool recall? |
|---|---|---|
| Episodic | `created_at` (UTC ISO) | Decay and ranking only. `format_for_prompt` emits raw `content`. |
| Semantic | `stored_at` / `reinforced_at` | Metadata only. Prompt is raw content + confidence. |
| Working | `timestamp` | Sort/persist only. Stripped in `to_langchain_messages`. |
| Facts via `save_fact` | `created_at` | `recall_facts` prints `[fact] {content}` with no date. |
| User-life commitments | `deadline` ISO + `deadline_raw` | **Does** recompute “due today / N days / missed, was due {day}” via `USER_TIMEZONE`. This is the pattern to copy. |
| User-life wellbeing | `ts` | Evidence text is raw (“yesterday's story”, “today”). Not in the default prompt. |
| Environment clock | `utcnow()` | `"It's {weekday} {HH:MM} UTC"` — not user-local. |

`USER_TIMEZONE` defaults to `Europe/Amsterdam` (`config/settings.py`). It is **only** consulted in `core/user_life.py`. `core/timeutil.py` is UTC-only. There is no deictic resolver anywhere else.

### 4.2 Frozen phrases still in the live window

Working memory, still injected as current user turns on 2026-09-18:

| timestamp (UTC) | relative phrase |
|---|---|
| 2026-09-07 14:39 | “I had to work on a commissioned project **yesterday**” → event was 2026-09-06 |
| 2026-09-09 11:29 | “I think I can finish **today or tomorrow**” → 2026-09-09/10 |
| 2026-09-10 22:45 | “I renounced god **today**” → 2026-09-10 |

Facts still stored at `effective_strength = 1.0` with unresolved deixis:

| id | created | content |
|---|---|---|
| 230 | 2026-09-06 | “Has a paid commission … that needs to be finished **tonight**.” |
| 643 | 2026-09-09 | “Committed to shipping … cleanly **today or tomorrow** …” |
| 138 | 2026-09-02 | “User intends to implement … **today** …” |

Wellbeing evidence 2026-09-10: “continuation of **yesterday's** story … dark conclusion **today**.”

Diag dumps from 2026-09-02 (`data/diag/replay_*.txt`) already show the memory block with “didn’t want to sleep **tonight**” / “only **tonight**” and no `created_at` prefix. That shape never changed.

### 4.3 User-visible temporal failure (same day)

Episode 1070, 2026-09-18 12:34Z, user:

> The "yesterday" was 2 weeks ago. We must adjust the schema so that dates are always recorded formally in your memories.

That is this spec’s issue 2, reported from the running unit, against the Sept 7 working-memory line.

---

## 5. Design

Keep the three stores and the LangGraph shape. Change **windowing**, **what the memory block contains**, and **how dates are stored and shown**.

Copy the user-life rule: store an absolute instant, show a phrase computed against **now** in `USER_TIMEZONE`.

### 5.1 Working memory becomes a real dialogue window (PR 1)

Split the buffer:

| Ring | Default | Eviction |
|---|---|---|
| Dialogue | last 12 messages, chronological | Never salience-evicted. Drop oldest when over cap. |
| Pins | up to 8 older high-salience user items not already in the dialogue ring | Salience + `decay()`. Skip if a normalized overlap with the dialogue ring or with the episodic block. |

`WORKING_MEMORY_CAPACITY` stays 20 (= 12 + 8).

Must:

1. Call `decay(user_id)` on session end and on the idle tick (the method already exists).
2. `to_langchain_messages` prefixes each item with user-local time, e.g. `[2026-09-07 16:39 CEST]` (pins: `[earlier · 2026-09-07 CEST]`).
3. Session timeout still does **not** wipe the dialogue ring. It does run decay and drops pins older than 14 days unless they were used this session.
4. Do not change the write of user/assistant turns into working memory.

### 5.2 Memory-block de-dupe (PR 2)

`format_for_prompt` (episodic) and `semantic.format_for_prompt`:

1. **Prefer** `fact` / `preference` / consolidator patterns over raw `User said:` / `Responded:` / `self_observation` when building the system-prompt block.
2. Drop a candidate line when normalized overlap with the current working-memory window exceeds a threshold (token Jaccard on lowercased alnum, or the existing episodic embedding if cheap). Same for overlap with another line already accepted this turn.
3. Cap `self_observation` in the prompt at 2. They remain in SQLite for consolidation.
4. Stop treating prompt assembly as recall: `increment_counters` must **not** bump `recall_count` for automatic injection. `used_count` may still rise. `recall_count` stays for explicit `recall_facts` / search tools / consolidator use. This breaks the 209-injection lock.
5. One anti-echo line in `tool_instructions` (or a tiny new block): do not recap standing user-life facts (housing, commission, income) unless the user asked or the fact **changed** this turn. Do not restated numbered plans from your previous reply.

Do not strip opinions from the prompt. Anti-sycophancy needs them. The dialogue window plus the anti-echo line is what stops opinions from being spoken as a recap.

### 5.3 Temporal binding (PR 3)

Schema (episodic, additive, nullable):

- `event_at` TEXT UTC ISO — when the **referred event** happened
- `event_local_date` TEXT `YYYY-MM-DD` in `USER_TIMEZONE`

Working-memory items already have `timestamp` = utterance time; that *is* `event_at` for dialogue. Semantic metadata gets the same two keys when a pattern is stored.

**At write** (`episodic.store`, `save_fact`, `save_preference`, consolidator pattern store, `save_long_term_memory`):

1. Leave `content` unchanged.
2. Run a deterministic deictic resolver against `created_at` + `USER_TIMEZONE`.
3. Fill `event_at` / `event_local_date`. If no deictic token, `event_at = created_at`.

Resolver tokens (word-boundary, case-insensitive):

`yesterday`, `today`, `tonight`, `tomorrow`, `this morning`, `this afternoon`, `this evening`, `last night`, `this week`, `last week`, weekday names (`Monday`…`Sunday`) when used as a when-clause.

Resolution is calendar-date in `USER_TIMEZONE` from the utterance instant (the user-life `_due_phrase` / `_tz()` helper, lifted to `core/timeutil.py` as `local_now` / `to_user_local`). Ambiguous weekday → the nearest occurrence consistent with the utterance (past for “on Monday” in past tense, upcoming for deadline-like “by Monday”). Fail open: if unsure, still set `event_at = created_at` and skip a guessed weekday.

Do **not** rewrite “today” out of the stored sentence. Display does the binding.

**At recall** (prompt, `recall_facts`, `search_long_term_memory`):

```
- [event] [2026-09-07 CEST · 11d ago] User said: I had to work on a commissioned project yesterday …
```

Age phrase from `event_at` vs now, same buckets as time-sense (`Nd ago` / `today` / `yesterday` **recomputed**). If `event_at.date != created_at.date` (deixis), the prefix uses `event_at`.

**Environment block:** show user-local clock first, UTC in parentheses:

`It's Friday 14:39 CEST (12:39 UTC), afternoon`

**Backfill:** one offline pass over existing episodes (and optional semantic metadata) that fills the two columns from `created_at` + the resolver. No content rewrite. Working memory needs no backfill; timestamps already exist.

**User-life:** already correct for commitments. Wellbeing evidence is out of the default prompt; when it is shown, prefix `ts` the same way. No change to deadline parsing.

---

## 6. Files

| Area | Files |
|---|---|
| Dialogue window | `memory/working.py`, `core/loop.py` (`_on_env_session_end`), `config/settings.py` (`DIALOGUE_WINDOW`, `PIN_CAPACITY`) |
| De-dupe / ranking | `memory/episodic.py` `format_for_prompt`, `memory/semantic.py` `format_for_prompt`, `memory/__init__.py` `build_memory_context`, `core/loop.py` tool_instructions |
| Recall counters | `memory/episodic.py` `increment_counters` callers |
| Time helpers | `core/timeutil.py` (keep UTC stores; add user-local) |
| Schema + resolver | `memory/episodic.py` store/migrate, `tools/memory_tools.py`, consolidator store path |
| Clock | `environment/time_sense.py` `to_prompt_context` |
| Docs | `docs/memory.md`, this spec, `docs/README.md` |

---

## 7. Tests (no live LLM)

Clock-frozen (`USER_TIMEZONE=Europe/Amsterdam`).

1. Working memory over capacity keeps the last 12 chronological messages even when they are assistant-role and low salience; old high-salience user turns land in pins, not the dialogue ring.
2. `decay` is invoked from session-end; a pin older than 14 days with no use this session is gone.
3. `to_langchain_messages` includes a CEST prefix; a 2026-09-07 14:39Z user line is not prefix-dated 2026-09-18.
4. Episodic `format_for_prompt` drops `User said:` / `Responded:` / `self_observation` lines that overlap the current working window; a fact with the same content can still appear, once, with a date prefix.
5. Building the memory block 20 times does **not** raise `recall_count` on those rows; `used_count` may.
6. Resolver: utterance 2026-09-07 14:39Z containing “yesterday” → `event_local_date=2026-09-06`. Prompt on 2026-09-18 shows `11d ago` (or `12d` depending on local calendar), never an unbound “yesterday” as the system’s now.
7. “tonight” on 2026-09-06 16:37Z → local date 2026-09-06, not the current local night.
8. Environment prompt contains `CEST` or `CET` and the Amsterdam weekday, not only UTC.
9. Backfill dry-run on a fixture row matches (6).
10. Existing user-life `_due_phrase` tests still pass; no commitment schema change.

---

## 8. Instrumentation

Keep `DIAG_MODE` dumps. Add one cheap log line per turn (no body dump):

```
PROMPT overlap: wm=20 dialogue=12 pins=8 ep_in=15 ep_kept=6 dropped_overlap=7
  mem_hash=… wm_span_h=264
```

`wm_span_h` is hours between oldest and newest working item. After PR 1 it should track the dialogue ring, not 11 days of user-only history.

---

## 9. Acceptance

A new session after this ships, with the current DB (no content wipe):

- Consecutive replies do not recap October housing / commission unless the user asked or the fact changed.
- The Sept 7 “yesterday” line, if still visible at all, is labeled as 2026-09-06 (or “Nd ago”), not as yesterday.
- Facts 230 / 643 / 138 cannot read as due tonight / today.
- Episode 8 (“sleep tonight”, 2026-09-02) is not in the default memory block after a handful of later turns (no injection→recall lock).
- Input tokens on a short follow-up drop versus the 5.4k–5.7k seen on 2026-09-18, mainly from a smaller memory block and a shorter transcript.

---

## 10. PR order

1. **Working-memory window + timestamps** — stops the model forgetting its own last replies; binds dialogue deixis at display time. Highest leverage for issue 1.
2. **Memory-block de-dupe + recall-counter fix** — stops the second and third copies, and the 209-lock.
3. **Schema + resolver + env clock + backfill** — issue 2 as recorded dates, not only display prefixes.

No wait required between PRs if 1 lands first (2 and 3 depend on a timestamped window and `timeutil` helpers respectively).
