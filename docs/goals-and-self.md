# Goals, autonomy, and self-model

## Goal kinds (`goals/system.py`)

**Terminal** — never complete. They are orientations: understand, create, resolve, connect, think independently.

Kind is decided at creation, not later:

- **Bounded** — has a machine-checkable `completion_condition` (one of six types: opinion registered, semantic pattern stored, episode tagged, commitment resolved, proposal created, user confirmed). It completes when the condition is satisfied. No LLM judgement about “how done” it is. `progress >= 1.0` never completes a goal.
- **Unbounded** — cannot have a checkable condition. Has a `ceiling_description` (saturation referent for scoring) and a `budget_share` cap on daily spend. **No progress meter.** Trace is a rated action history (Elo over *action types*, pairwise, not absolute scores).

The two orientation programs are unbounded:

- `continuous_self_improvement` — architecture and response quality (`budget_share` 0.15)
- `user_life_improvement` — the user’s real-world outcomes (`budget_share` 0.15)

Unbounded goals are never abandoned for staleness.

**Dynamic instrumental** — LLM-generated, max 5 active. If the model cannot name a valid condition, the goal is created unbounded (`budget_share` 0.05). After each user turn, `check_completions()` runs the non-LLM checks (at most one `user_confirmed` LLM call, skipped above 70% budget).

Action types: `web_research`, `self_theorize`, `future_simulate`, `codebase_read`, `memory_synthesis`, `consolidation_review`. An action with no artifacts is scored 1000 (unproductive) and skips the LLM rater.

## Autonomy (`goals/autonomous.py`)

Every `AUTONOMOUS_INTERVAL_MIN` minutes, if no session is active and fatigue is not high:

1. Abandon stale *bounded* goals.
2. If independence score is low and no research goal exists, suggest research topics.
3. Take one step on the highest-salience active goal that is not over its `budget_share`.

`select_action_type` is epsilon-greedy with forced exploration (`n < 2`). MID-tier actions (`web_research`, `self_theorize`, `future_simulate`) fall through when the background MID gate refuses.

## Research (`goals/research.py`)

1. Generate 2–3 search queries, including a contrarian one.
2. Query local SearXNG and extract full pages (`trafilatura`).
3. Synthesize a position.
4. `form_opinion(..., origin=EXTERNAL)`.
5. Store a research episode.

This is how Keeper gets something to push back *from* that is not the user’s own view reflected back. There is no search quota — token spend is the constraint.

## Self-model (`core/self_model.py`)

After every reply, `observe()` writes a self-observation. Every 5th non-trivial one is an LLM summary; the rest are compact.

Periodically `update_model()`:

1. Derive patterns from recent observations (and the codebase summary).
2. Every other version, flag claims that have no lexical support in recent behavior.
3. If the last few versions barely changed, force a contrarian review.
4. Generate hypotheses that bias later goals/actions.

The prompt block is honest about tensions and unsupported claims.

## Opinions (`core/opinions.py`)

Origins: independent, external, adopted, collaborative, contested, unknown.

Independence score = conviction-weighted share of independent + external + contested. Below `INDEPENDENCE_THRESHOLD` (default 0.3) with enough opinions, the prompt gets a sycophancy alert.

Periodic review (`OPINION_REVIEW_INTERVAL_MIN`) only considers memories newer than `last_tested` / `formed_at`, so the same old episodes cannot restress-test a stance forever.

Registry cap: `MAX_TRACKED_OPINIONS` (default 100). Lowest conviction is pruned.

## User life (`core/user_life.py`)

Tracks commitments, wellbeing snapshots, and achievements. Prompt injection lists open and stale commitments. Metrics feed the `user_life_improvement` goal. Talking more is not a success signal.
