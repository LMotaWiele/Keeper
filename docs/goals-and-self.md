# Goals, autonomy, and self-model

## Goal tiers (`goals/system.py`)

**Terminal** — never complete. They are orientations: understand, create, resolve, connect, think independently.

**Permanent instrumental** — long-running programs that cycle rather than finish:

- `continuous_self_improvement` — architecture and response quality
- `user_life_improvement` — the user’s real-world outcomes

Progress on permanent goals is capped at 0.95. At that point they reset toward 0.3 and start a new cycle. They must not be abandoned for staleness.

**Dynamic instrumental** — LLM-generated, max 5 active, completable. Research-tagged goals go through `ResearchEngine`. Stale non-permanent goals can be abandoned after 48 hours unused.

After each user turn, `evaluate_progress()` asks whether the exchange actually moved a goal. Most turns move nothing; that is expected.

## Autonomy (`goals/autonomous.py`)

Every `AUTONOMOUS_INTERVAL_MIN` minutes, if no session is active and fatigue is not high:

1. Abandon stale non-permanent goals.
2. If independence score is low and no research goal exists, suggest research topics.
3. Take one step on the highest-salience active goal.

Routing:

- `continuous_self_improvement` → `SelfTheorizer`
- tag `research` → web search + EXTERNAL opinion, then complete
- otherwise → an internal note stored as an episode

Optional future simulation can modify or abandon the planned step.

## Research (`goals/research.py`)

1. Generate 2–3 search queries, including a contrarian one.
2. Run Tavily (counts against `SearchBudget`).
3. Synthesize a position.
4. `form_opinion(..., origin=EXTERNAL)`.
5. Store a research episode.

This is how Keeper gets something to push back *from* that is not the user’s own view reflected back.

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
