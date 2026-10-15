File: /docs/specs/SPEC-0004-autonomous-loop-hygiene.md
# SPEC-0004: Stop the autonomous loop from writing unevidenced, duplicate state (Keeper Patch 04)

| Field | Value |
|---|---|
| Status | ready |
| Size | staged (3 stages) |
| Kind | fix |
| Parent | SPEC-0003 (`docs/specs/SPEC-0003-relational-grounding.md`) |
| Date | 2026-10-15 |
| Checkpoint | continuous |
| Run record | /docs/specs/SPEC-0004-run.md |
| Branch | `spec-0004-loop-hygiene` (local only) |

Read `/.agents/notes/index.md` if present. Record the revision you start from and any pre-existing uncommitted changes; don't fold unrelated work into this change.

**Git rule:** commit locally on `spec-0004-loop-hygiene`. Never `git push`, never create or delete remote branches, tags, PRs or releases, never change remotes. Work ends at a local commit; Lucas pushes.

**Live data rule:** never write to the live data directory (`DATA_DIR`, default `./data`: `episodic.db`, `chroma/`, `state/*.json`). Every script in this spec runs against a copy you make (`DATA_DIR=<copy>`). Scripts default to dry-run; `--apply` against live data is run by Lucas with Keeper stopped.

## 1. Intent

Keeper's autonomous loop writes state that the conversation path reads back. Three writers currently produce unevidenced or duplicate state, and each defeats a principle already in the codebase: every injected claim needs a traceable external referent; Keeper does not certify itself; inferred claims about the user don't get injected as fact (Patch 03, "the one line I am still holding").

Observed on the live instance (state snapshot 2026-10-14 23:18 UTC, HEAD `21a9136`):

1. **Synthesis flood.** One user message ("I'm pushing a little past my bedtime today, but it's worth it.") became ~70 semantic documents in ~9 hours via `memory_synthesis` (35 + 38 runs across both goals in `state/goals.json`), escalating to "regularly", "a recurring pattern", "rationalizing", "pushing past healthy limits". It resumed immediately after restart. Three exact-text collisions bumped confidence +0.1 each with no new evidence.
2. **Opinion clones.** `opinions.json` is at the 100 cap: 97 `external`, collapsing to two positions (≈66 "self-improvement needs external verification", ≈31 "behaviour change needs action planning"), 0 defended, 0 revised. Pruning by lowest conviction evicted the conversation-formed opinions. The "you never disagree with the user" alert fires every turn because research domains never string-match `user_known_positions` keys. Research episodes (salience 0.8) fill the "What I remember about you" block.
3. **Self-model built from harness strings, then frozen.** `self_model.json` v13 (last updated 2026-09-26) has as its top hypothesis (0.95) "I reflexively emit 'Researched and formed opinion:'" — an f-string in `goals/autonomous.py`. "Truncating mid-sentence" traces to `[:150]`/`[:180]` slices in the same summaries. Autonomous summaries reach the corpus framed as "Given '…', responded with: …". The model has not rebuilt since; `observations_analysed` is set to `len(observations)` (≤50) while `observation_count` is cumulative (397), so an update is attempted hourly and (hypothesis) refused by the Jev `observations_warrant_rewrite` gate.

Agreed decisions:
- All three in one merge, staged by dependency then impact (Lucas, 2026-10-15).
- New gates are restrictive, so they default **on** (they stop writes; they don't add behaviour). Each has a config flag to disable and a greppable trace line.
- Data repair is by scripts with dry-run default, modelled on `scripts/purge_preambles.py`. Repairs invalidate or tag; they don't hard-delete semantic or episodic rows. Opinion compaction does remove merged duplicates from the registry (evidence is kept on the canonical).
- Autonomous synthesis may still write claims about the user, but only with ≥2 cited user-turn episodes on ≥2 distinct local dates (Europe/Amsterdam). *Alternative if Lucas prefers:* autonomous synthesis never writes user claims (user facts only via the quote-anchored world model).
- The one-time self-model rebuild derives from cleaned observations **without** the previous model as context, so v13's harness-derived claims don't carry forward. v13 stays in `_recursion_guard.model_history`. *Alternative:* rebuild with v13 as context.

Required outcomes:
- O1: Autonomous `memory_synthesis` does not run when no user-turn episode has arrived since its last run for that goal.
- O2: An autonomous synthesis is stored only if it cites ≥2 user-turn episodes on ≥2 distinct local dates; stored docs carry those `source_episode_ids` and `origin=autonomous_synthesis`.
- O3: A semantic write that is a near-duplicate (cosine ≥ calibrated threshold) of an existing doc reinforces that doc instead of creating one; confidence rises only when the write adds a source episode the doc didn't have (all callers, including the exact-hash path).
- O4: A script identifies and invalidates the flood's documents in a copy of live data, with the candidate rule fixed in §6.1.
- O5: Research whose position is a near-duplicate of an existing `external` opinion corroborates it (evidence merged, counter incremented) instead of creating an opinion or a research episode.
- O6: Pruning at the cap removes `external` opinions before any other origin.
- O7: The zero-disagreement alert appears only when ≥3 comparable (domain-matched, independent/external, conviction > 0.5) opinions exist and none disagree.
- O8: Episodes with `source=autonomous_artifact` do not render in "What I remember about you".
- O9: A script compacts existing `external` opinion clusters to one canonical each in a copy of live data, leaving non-external opinions untouched.
- O10: Autonomous actions no longer enter the self-observation corpus as prose; the analysis prompt receives a deterministic autonomous-activity count block instead.
- O11: `observations_analysed` tracks `observation_count`, so update attempts happen only after ≥5 new observations, and a Jev skip also advances it.
- O12: A script degrades existing autonomous-derived self-observations and can force one rebuild from the cleaned corpus (Jev gate bypassed once, no previous-model context).

## 2. Priority

Stage 1 first: it is live, writes inferred health-adjacent claims about the user, and its output is read back by the conversation path. Stage 2 next: the opinion registry is the sycophancy countermeasure and is currently blind. Stage 3 last: the stale self-model is injected every turn but cannot drive action (no hypothesis has an executable check, so `action_bias` is idle by design).

Deferred (one line each):
- CONTESTED misclassification of flattery-as-pushback ("not alchemy, steering") — needs a judgment design, not a gate.
- Research topic is always `goal.description`; topic rotation belongs to `TODO(KEEPER-P2)`. After O5, repeat research writes nothing.
- `web_research` "produced nothing" since 2026-10-14 02:52 UTC (four `return None` paths in `research_topic`) — diagnose separately.
- The 2026-10-14 00:45:12 reply truncated at "2." — check `finish_reason`.
- Drive saturation (4/5 at 1.0, "[URGENT]" ×5 every turn).
- Duplicate world facts in the prompt ("moved houses" ×3).
- Tool block names world-fact tools but not `record_user_commitment` and friends.
- Restoring opinions already pruned before this patch.
- Self-model staleness indicator in the prompt.

## 3. Prior art

- **Generative Agents** (Park et al. 2023, arXiv:2304.03442): reflection is triggered by accumulated importance of new observations, not a timer, and each insight cites the memories it rests on. Adopted: event trigger (O1) and mandatory evidence citation (O2). The ≥2-episodes / ≥2-days rule is ours, from the referent principle: one event cannot be a pattern.
- **Mem0** (docs.mem0.ai "How it works"): checks related existing memories and deduplicates before storing; updates and deletes are explicit operations. Adopted: dedup-before-write (O3, O5). Rejected: LLM-decided UPDATE/DELETE — an extra call, and a model judging its own memory edits.
- **In repo:** `memory/consolidator.py` already has a `skipped_no_new` gate and the Jev `episodes_contain_new_pattern` gate; the autonomous synthesis path has neither. `scripts/purge_preambles.py` is the repair-script pattern (dry-run, `--apply`, tag `degraded`, scale importance). `scripts/diag_common.load_companion_no_loops()` is the bootstrap for scripts.
- **Thresholds:** no reliable published paraphrase threshold for `BAAI/bge-small-en-v1.5`; it is measured on the live clusters (V3, V8), not assumed.

## 4. Grounding

**Pre-landing planner revision:** isolation requires four independently resolved environment variables. Before any data-touching script or test, export absolute `DATA_DIR=<COPY>`, `MIDTERM_DB_PATH=<COPY>/episodic.db`, `CHROMA_DB_PATH=<COPY>/chroma`, and `EMBEDDING_CACHE_DIR=<COPY>/fastembed`. Report all four resolved paths in step 0; `DATA_DIR` alone does not isolate SQLite, Chroma, or the writable embedding cache. The supplied copy is `/home/george-contis/keeper-data-copy-20261015`.

**Pre-landing planner revision — resolved step-0 stops:** session-window degradation is removed in favour of a preceding-episode classifier with a 120-second window and the safety stops in §6.3/§9. One `_clean_observations` helper supplies layer 1, layer 2, and `generate_hypotheses`. These revisions resolve the three previously recorded isolation, degradation, and hypothesis-reread stops; they are not open design conflicts.

| Fact | Expected | Tag | Source |
|---|---|---|---|
| Start revision | `main` at `21a9136` or later | measured | `git log -1` |
| Autonomous synthesis path | `goals/autonomous.py` ~301–333: recalls on `goal.description`, asks for one pattern, `semantic.store(...)` with **no** `source_episode_ids`, no novelty gate | measured | code |
| Action selection seam | `take_action` builds `unavailable: set[str]` (~139–149) before `select_action_type(goal, unavailable=...)` | measured | code |
| Semantic dedup | `memory/semantic.py` `store()`: `_content_id` = sha256[:16] of text; on hit, confidence += 0.1 unconditionally | measured | code |
| Semantic collection metric | Chroma `hnsw:space = cosine`; `relevance = 1 - dist` | measured | code |
| Research → opinion | `goals/research.py` `research_topic`: `form_opinion(origin=EXTERNAL)` with no similarity check; stores `research_completed:` episode, salience 0.8, `source=autonomous_artifact` | measured | code |
| Opinion IDs | `_generate_id` hashes domain + position + **timestamp** — never collides | measured | `core/opinions.py` |
| Prune | `_prune` drops lowest conviction regardless of origin | measured | `core/opinions.py` |
| Alert 4 | `detect_drift` emits zero-disagreement alert when the comparable set is empty | measured | `core/opinions.py` |
| Memory block | `episodic.format_for_prompt` ranks `opinion`/`research` above `event`; char budget `FORGET_PROMPT_CHAR_BUDGET` | measured | `memory/episodic.py` |
| Recall rows include `source` | yes | agent-verify | `memory/episodic.py` recall |
| Autonomous self-observation | `goals/autonomous.py` ~177: `self_model.observe(action=summary, context="Goal: …")` where `summary = "Researched and formed opinion: " + position[:150]`; only autonomous caller | measured | code |
| Compact observation shape | `"Given '<context[:150]>', responded with: <action[:300]>"` | measured | `core/self_model.py` |
| Bookkeeping bug | `observations_analysed = len(observations)` (≤50); `run_loop` updates when `observation_count - observations_analysed >= 5` | measured | `core/self_model.py` ~458, ~713 |
| Freeze cause | Jev `observations_warrant_rewrite` returns skip each hour | agent-verify (hypothesis) | grep runtime logs for `self_model_analysis skipped — jev observations_warrant_rewrite` |
| Observation inputs | Existing layer-1, layer-2, and hypothesis readers lack a shared degraded filter; route all three through `_clean_observations` | agent-verify | `core/self_model.py`; pre-landing planner revision |
| Degradation provenance | Classify self-observations from preceding episodes within 120 seconds, not session windows; stop if any rule-(a) match is not classified autonomous or if unclassified observations exceed 10% | agent-verify | episodic copy; pre-landing planner revision |
| Authoritative opinion store | `opinions.json` loaded after `self_model.json` (which embeds `_opinions`) | agent-verify | `scripts/diag_common.py`, `core/loop.py` startup, `SelfModel.save` |
| Episode `source` reliability | user turns labelled `user_turn`; unknowns also default to `user_turn` | agent-verify (report distribution by type) | `memory/__init__.py` `_infer_source` |
| Data paths | Independently resolved `DATA_DIR`, `MIDTERM_DB_PATH`, `CHROMA_DB_PATH`, `EMBEDDING_CACHE_DIR`; export absolute paths under the copy and report all four; state lives under `DATA_DIR/state/` | agent-verify | `config/settings.py`; pre-landing planner revision |
| Test runner | `python test_patchNN.py` or pytest | agent-verify | existing `test_*.py` |
| Live flood size | ~70 synthesis doc IDs in `goals.json` `action_log[].artifact_refs` (log capped at 50/goal) | measured | live state |

Step 0: before editing, verify every row that describes repo or environment state (whatever its tag) and report expected vs actual. Ask Lucas for a copy of the live `DATA_DIR` if you don't have one; stages 1–3 need it for V3, V5, V8, V11, V14 and final acceptance. If a contradiction changes the design, outcomes, scope or verification, stop and report. Otherwise record it and continue.

## 5. Scope

In: the three stages in §6, two config sections, three repair scripts, a new `test_patch04.py`, doc updates to `docs/memory.md` and `docs/goals-and-self.md` where behaviour changes, `/docs/specs/index.md` (create with this spec's row), the run record.

Out: everything in §2 Deferred; consolidator novelty gates (unchanged); the world model; Jev thresholds; action Elo/rating logic; SOUL.md; any change to how conversation replies are observed.

## 6. Changes

### 6.0 Architecture

Three writers, three readers, one shared helper.

- **Helper** — `memory/similarity.py`: `embed_texts(list[str]) -> list[vec]` via the existing semantic embedder; `cosine(a, b)`; `nearest(vec, candidates) -> (index, sim)`. No new dependency. Used by stages 1 and 2.
- **Stage 1** writer `AutonomousEngine` (`memory_synthesis`) → `SemanticMemory.store` → reader `semantic.format_for_prompt`.
- **Stage 2** writer `ResearchEngine.research_topic` → `OpinionRegistry` (+ research episode) → readers `OpinionRegistry.to_prompt_context` and `episodic.format_for_prompt`.
- **Stage 3** writer `AutonomousEngine` → `SelfModel.observe` (removed) → reader `SelfModel._layer1_observation_update`; new deterministic activity block from `goals.json` action logs.
- **Config** (`config/settings.py`, env-driven like existing flags): `SYNTH_GATE_ENABLED=true`, `SYNTH_MIN_EVIDENCE_EPISODES=2`, `SYNTH_MIN_EVIDENCE_DAYS=2`, `SEMANTIC_DEDUP_SIM` (default set from V3), `RESEARCH_MERGE_ENABLED=true`, `OPINION_DUP_SIM` (default set from V8), `DRIFT_MIN_COMPARABLE=3`, `MEMORY_BLOCK_AUTONOMOUS_MAX=0`, `SELF_OBS_AUTONOMOUS=false`.
- **Trace:** one `log.info` line per decision with fixed prefixes: `SYNTH_GATE`, `SYNTH_REJECT`, `SEM_MERGE`, `RESEARCH_MERGE`, `OPINION_PRUNE`, `SELFMODEL_SKIP`.
- **Pre-landing planner revision — script isolation:** every new script prints all four resolved paths first and refuses to run unless `MIDTERM_DB_PATH`, `CHROMA_DB_PATH`, and `EMBEDDING_CACHE_DIR` are under `DATA_DIR`. Before any data-touching script or test, export absolute paths for all four variables. Thresholds remain config values until calibrated; missing-copy calibration is not a pass.
- **Pre-landing planner revision — observation provenance:** one `_clean_observations` helper excludes degraded episodes for layer 1, layer 2, and `generate_hypotheses`. The repair script uses the preceding-episode classifier in §6.3, never session windows.
- No walking skeleton: this is a fix to existing paths; each stage is exercised end to end by its own checks.
- Update `docs/memory.md` and `docs/goals-and-self.md` where described behaviour changes; preserve unrelated content.

### 6.1 Stage 1: Autonomous synthesis is evidenced and deduplicated

Outcomes: O1–O4. Where: `goals/autonomous.py`, `memory/semantic.py`, `memory/similarity.py`, `config/settings.py`, new `scripts/purge_synthesis_flood.py`.

Changes:
- **Novelty gate (O1).** Per goal, persist `last_synthesis_at` (and the newest input episode's `created_at`) with the goal state. In `take_action`, before selection, add `memory_synthesis` to `unavailable` when no episode with `source == "user_turn"` has `created_at` later than that watermark. A goal with no watermark is eligible. Trace `SYNTH_GATE goal=… new_user_turns=0`.
- **Evidence requirement (O2).** Synthesis input is user-turn episodes only (exclude `autonomous_artifact`, `self_observation`, `keeper_response`, `environment_event`, `tool_result`). Number them in the prompt and require the model to return `"evidence": [indices]`. Code validates: indices map to supplied episodes; ≥`SYNTH_MIN_EVIDENCE_EPISODES` distinct episodes; ≥`SYNTH_MIN_EVIDENCE_DAYS` distinct Europe/Amsterdam dates. On failure: no write, summary `"Synthesis rejected: insufficient evidence (n=…, days=…)"`, trace `SYNTH_REJECT`. On success: `semantic.store(..., source_episode_ids=[…], metadata={"origin": "autonomous_synthesis", "goal": goal.name})` and advance the watermark. Advance the watermark on rejection too (the input was considered).
- **Near-duplicate merge (O3).** `SemanticMemory.store` gains `dedup_sim: float | None = None` (default from config `SEMANTIC_DEDUP_SIM`; `None` keeps exact-hash-only behaviour). After the exact-hash check, query the nearest existing non-invalidated doc; if cosine ≥ threshold, reinforce that doc and return its ID (trace `SEM_MERGE sim=…`). Reinforcement on either path: merge source IDs, `reinforcement_count += 1`, `reinforced_at`; **confidence += 0.1 only if the merge added ≥1 new source episode ID**.
- **Repair script (O4).** `scripts/purge_synthesis_flood.py`, dry-run default, `--apply` invalidates via `semantic.invalidate(reason="SPEC-0004: unevidenced autonomous synthesis")`. Candidate set = **A ∪ B**: A = every `semantic:<id>` in `goals.json` `action_log` entries with `action_type == "memory_synthesis"`; B = docs with empty `source_episode_ids` whose cosine to any doc in A is ≥ `SEMANTIC_DEDUP_SIM`. Dry-run prints |A|, |B|, |A ∪ B|, each candidate's text (truncated for display only), and the count of other source-less docs left untouched.

Preserve: consolidator behaviour except the confidence rule; all non-autonomous `semantic.store` callers behave as before when no near-duplicate exists; action selection and Elo untouched apart from the `unavailable` entry.

Seams:
- `take_action` ↔ watermark: watermark lives on the goal object and must survive `GoalSystem` save/load; missing field = eligible.
- `memory_synthesis` ↔ LLM: malformed or missing `evidence` = rejection, not exception; parse with `parse_json_lenient` as now.
- `semantic.store` ↔ Chroma: nearest-neighbour query must exclude `invalidated == "true"`; empty collection = no merge. Embedding computed once and reused for the upsert.
- Script ↔ data: run via `load_companion_no_loops()` with `DATA_DIR` pointing at the copy.

Checks: V1–V5. Exit: V1–V5 pass; threshold recorded in config and run record.

### 6.2 Stage 2: Opinion registry stops cloning and the alert stops lying

Depends on: stage 1 (`memory/similarity.py`). Outcomes: O5–O9. Where: `goals/research.py`, `core/opinions.py`, `memory/episodic.py`, `goals/autonomous.py` (summary text only), new `scripts/compact_opinions.py`.

Changes:
- **Research merge (O5).** In `research_topic`, after synthesis and before `form_opinion`, embed the new position and find the nearest `external` opinion position. If cosine ≥ `OPINION_DUP_SIM`: append new `key_sources` to that opinion's `related_evidence` (dedupe, cap 10), `corroborated_count += 1`, `last_corroborated_at = now`; conviction unchanged; **no** new opinion, **no** research episode; return `{"merged_into": id, "position": existing.position, …}`. Trace `RESEARCH_MERGE`. The autonomous summary becomes `"Research corroborated opinion <id>"`. `Opinion` gains `corroborated_count: int = 0` and `last_corroborated_at: str | None = None`; `from_dict` tolerates their absence.
- **Origin-aware prune (O6).** `_prune` removes `external` opinions (lowest conviction first) before any other origin; other origins are pruned only if no `external` remains. `protect` behaves as now. Trace `OPINION_PRUNE origin=… conviction=…`.
- **Honest alert (O7).** In `detect_drift`, compute the comparable set as today; emit the zero-disagreement alert only if `len(comparable) >= DRIFT_MIN_COMPARABLE` and none disagree.
- **Memory block (O8).** In `episodic.format_for_prompt`, render at most `MEMORY_BLOCK_AUTONOMOUS_MAX` (default 0) episodes with `source == "autonomous_artifact"`. Opinions still reach the prompt via "Opinion awareness".
- **Compaction script (O9).** `scripts/compact_opinions.py`, dry-run default. Greedy clustering of `external` opinions: sort by conviction desc; each joins the first canonical with position cosine ≥ `OPINION_DUP_SIM`, else becomes a canonical. Canonical keeps its id, position, domain and conviction; gains merged `related_evidence` (dedupe, cap 10) and `corroborated_count = cluster size − 1`. Members are removed. Non-external opinions and `user_known_positions` untouched. `--apply` writes the authoritative store found in step 0 with `core/atomic.py`; if `self_model.json` embeds `_opinions` and is also loaded, keep both consistent. Dry-run prints each cluster (canonical + member count + one sample).

Preserve: `detect_opinions` conversation path (domain-match update); other alerts; opinion review loop; `form_opinion` behaviour for non-research callers.

Seams:
- `research_topic` → `form_opinion`: merge happens in `research_topic`, so `form_opinion` signature is unchanged.
- `research_topic` return: callers check `result.get("opinion_id")`; the merge result must not be mistaken for a new opinion. Confirm all callers (`goals/autonomous.py`, `scripts/diag_autonomous.py`) handle `merged_into`.
- Persistence: `snapshot`/`restore` round-trip with new fields and with old files lacking them.

Checks: V6–V11. Exit: V6–V11 pass; threshold recorded.

### 6.3 Stage 3: Self-model corpus excludes the harness, bookkeeping fixed, one clean rebuild

Depends on: none functionally; ordered last by impact. Outcomes: O10–O12. Where: `goals/autonomous.py`, `core/self_model.py`, `memory/episodic.py` (or filtering in the caller), new `scripts/degrade_autonomous_observations.py`.

Changes:
- **Corpus (O10).** When `SELF_OBS_AUTONOMOUS` is false (default), `take_action` does not call `self_model.observe`. `_layer1_observation_update` adds a deterministic block to the analysis prompt: autonomous action counts over the last 7 days by `action_type` × outcome (artifact produced / nothing), computed from goal `action_log`s, labelled as harness activity ("what the autonomous scheduler did, not what you said"). Add the `{autonomous_activity}` slot to `ANALYSIS_PROMPT`.
- **Bookkeeping (O11).** After a successful analysis **and** after a Jev skip, set `observations_analysed = observation_count`. Trace `SELFMODEL_SKIP reason=jev` on skip.
- **Degraded filter — pre-landing planner revision.** One `_clean_observations` helper excludes episodes tagged `degraded` and supplies layer 1, layer 2, and `generate_hypotheses` (over-fetch then filter, or a tag filter on `get_recent`; preserve `get_recent` default behaviour for other callers).
- **Repair + rebuild script (O12) — pre-landing planner revision.** `scripts/degrade_autonomous_observations.py`, dry-run default. Rule (a): `self_observation` episodes whose content contains `"Given 'Goal: "`. Rule (b): use a preceding-episode provenance classifier with a 120-second window to identify autonomous self-observations; session windows are not used. Validate the classifier before applying repairs: stop if any rule-(a) match is not classified autonomous, or if unclassified observations exceed 10% of observations evaluated. Unclassified observations are not automatically degraded. `--apply` tags autonomous-derived candidates `degraded` and scales importance ×0.1 (same as `purge_preambles.py`). `--rebuild` (requires `--apply` already done on that data): if ≥5 non-degraded observations remain, run layer 1 once with the Jev gate bypassed and `previous_model_context` empty, then `generate_hypotheses` using the same cleaned corpus helper; v13 remains in `_recursion_guard.model_history`. If <5 remain, do nothing and report the count. Dry-run prints counts per rule and classifier outcome, including unclassified count and percentage and rule-(a) validation.

Preserve: conversation-reply observation path (`core/loop.py` ~560); layer-2 behaviour except routing observations through `_clean_observations`; layer 3; hypothesis gating (`eligible_hypotheses`, speculation recording); `to_prompt_context` format.

Seams:
- `ANALYSIS_PROMPT.format` gets a new kwarg; every call site must pass it.
- Rebuild runs outside the live process via `load_companion_no_loops()` and must save state the same way the live process does.
- Preceding-episode classifier: use episode provenance within the preceding 120 seconds; no session-history inference. The rule-(a) validation and >10% unclassified stops apply before any repair writes.

Checks: V12–V16. Exit: V12–V16 pass.

## 7. Paths

| Path | Trigger | Planned check or justified exclusion |
|---|---|---|
| happy: new evidence on 2 days | 2 user-turn episodes, distinct dates | V2 |
| no new user input (the incident) | only autonomous episodes since watermark | V1, V4 |
| single-event pattern | evidence = 1 episode, or 2 on same date | V2 |
| malformed synthesis JSON / missing `evidence` | LLM returns no indices | V2 |
| paraphrase of existing doc | cosine ≥ threshold | V3 |
| unrelated doc | cosine < threshold | V3 |
| empty semantic collection | first write | V3 |
| first run, goal has no watermark | legacy `goals.json` | V1 |
| research paraphrase | near-dup external opinion | V6 |
| research novel position | below threshold | V6 |
| prune at cap with mixed origins | 101st opinion | V7 |
| alert with no comparable opinions | current live shape | V8 |
| old `opinions.json` without new fields | restore | V9 |
| autonomous episodes in memory block | research/opinion episodes present | V10 |
| autonomous observe disabled | web_research succeeds | V12 |
| Jev skip | gate returns skip | V13 |
| rebuild with <5 clean observations | sparse corpus | V15 |
| preceding-episode provenance and safety stops | 120-second boundary; rule-(a) mismatch; >10% unclassified | V15 |
| retry/cancel/timeouts | — | not exercised: no new network calls; embeddings are local |

## 8. Verification

| Outcome | Change | Check ID: falsifiable condition |
|---|---|---|
| Isolation | §6.0 planner revision | V17: each new script prints all four resolved paths and refuses before loading data if SQLite, Chroma, or embedding cache resolves outside `DATA_DIR`; test each escape and an accepted isolated configuration. |
| O1 | §6.1 gate | V1: unit — goal with watermark T and no `user_turn` episode after T → `memory_synthesis ∈ unavailable`; add one user-turn episode after T → absent; goal without watermark → absent. |
| O2 | §6.1 evidence | V2: unit (LLM mocked) — evidence [one episode] → no write, `SYNTH_REJECT`; two episodes same date → no write; two episodes on two dates → one write with exactly those `source_episode_ids` and `origin=autonomous_synthesis`; missing `evidence` → no write, no exception. |
| O3 | §6.1 merge | V3: threshold calibration + unit. On the data copy, embed the A set (O4). Record the distribution of each A doc's max cosine to the rest of A, and of max cosine from 200 random non-A docs (or all, if fewer) to A. Set `SEMANTIC_DEDUP_SIM` = highest value t such that ≥90% of A docs have a same-set neighbour ≥ t **and** ≤1% of non-A docs have any A neighbour ≥ t. If no t satisfies both, stop and report the distributions. Unit: two real paraphrases from A → second `store` returns first id, count unchanged, confidence unchanged with no new sources and +0.1 with one new source; an unrelated sentence → new doc. |
| O1–O3 | §6.1 | V4: replay of the incident — fixture with the single bedtime user-turn episode plus autonomous episodes; 10 `take_action` cycles, LLM mocked to return paraphrases with evidence [0] → 0 semantic writes. |
| O4 | §6.1 script | V5: dry-run on copy prints |A|, |B|; A ⊆ candidates; every candidate has empty `source_episode_ids`. `--apply` on copy, then `semantic.search(query="sleep bedtime late night work")` returns no candidate id. |
| O5 | §6.2 merge | V6: unit (synthesis mocked) — paraphrase of an existing external opinion → opinion count unchanged, `corroborated_count == 1`, evidence merged without duplicates, no `research_completed` episode stored; novel position → new opinion. |
| O6 | §6.2 prune | V7: unit — registry at cap with 98 external + 2 collaborative; add one → an external is removed, both collaborative remain; registry of only non-external at cap → lowest-conviction non-external removed. |
| O7 | §6.2 alert | V8: unit + calibration — ≥5 user positions with no domain overlap → alert absent; 3 comparable aligned → present; 3 comparable with one disagreeing → absent. Calibrate `OPINION_DUP_SIM` on the copy with the same rule as V3, using the two observed clusters as the positive sets (label by your own reading; record labels) and requiring the 3 non-external opinions have no external neighbour ≥ t. |
| O6–O9 | §6.2 persistence | V9: unit — snapshot/restore round-trip with new fields; restore of a pre-patch `opinions.json` works and defaults new fields. |
| O8 | §6.2 block | V10: unit — episodes with `source=autonomous_artifact` absent from "What I remember about you"; `user_turn` episodes present. |
| O9 | §6.2 script | V11: dry-run on copy prints clusters; every external opinion is in exactly one cluster; `--apply` on copy → count = clusters + non-external; non-external opinions byte-identical; registry loads via `load_companion_no_loops()`. |
| O10 | §6.3 corpus | V12: unit — successful `web_research` with `SELF_OBS_AUTONOMOUS=false` does not call `observe`; analysis prompt contains the activity block computed from a fixture action log. |
| O11 | §6.3 bookkeeping | V13: unit — after analysis and after Jev skip, `observations_analysed == observation_count`; `run_loop` does not call `update_model` until 5 more observations. |
| O12 | §6.3 filter | V14: unit — layer 1, layer 2, and `generate_hypotheses` all use `_clean_observations` and exclude `degraded` episodes. |
| O12 | §6.3 script | V15: dry-run on copy prints rule-(a) and preceding-episode classifier counts using the 120-second window, with unclassified percentage; stop before writes if any rule-(a) match is not autonomous or unclassified exceeds 10%; unclassified observations are not degraded. Exercise the window boundary and both stop conditions. `--apply` then `--rebuild` on copy produces v14, or reports <5 and changes nothing. |
| O10, O12 | §6.3 rebuild | V16: if V15 rebuilt, v14 JSON contains no `"Researched and formed opinion"` and no hypothesis mentions it; `model_history` contains v13. |

You write and run the commands. Protected (never edit, skip or weaken to pass): `test_budget.py`, `test_context_time.py`, `test_goals.py`, `test_opinions.py`, `test_patch01.py`, `test_patch02.py`, `test_patch03.py`, `test_research.py`, `test_routing_v2.py`, their copies under `docs/`, `SOUL.md`/`SOUL.md.example`, and the live `DATA_DIR`. New tests go in `test_patch04.py`.

Invariants:
- No semantic write without ≥1 `source_episode_id` from the autonomous path.
- Repeated identical or paraphrased writes never raise confidence without new source IDs.
- Opinion count never grows from research when a near-duplicate exists.
- Scripts are idempotent: a second `--apply` changes nothing.

Shared checks (every stage): full existing suite (protected files) plus `test_patch04.py`.

Final integrated acceptance (on a data copy with all three scripts applied in order 1→2→3, using `load_companion_no_loops()`): run 12 `take_action` cycles with LLM and search mocked to return paraphrases of existing content and no new user episodes → semantic count, opinion count and `self_observation` count unchanged; assemble a prompt for the user message "I stayed up late again working on the experiment" (via `scripts/diag_replay.py` or the same assembly path) → no `research_completed` lines under "What I remember about you", no flood candidate under "Things I've learned", no zero-disagreement alert, self-model block contains no `"Researched and formed opinion"`.

Resume baseline: run the existing suite before editing in any session; record pre-existing failures separately and don't fix them.

## 9. Stop conditions and repairs

Repairs: a repair round is a coherent diagnosis, a scoped fix and a rerun of affected checks. Up to three rounds per failed check group per stage. Don't reset counts by restarting sessions, renaming checks or cosmetically varying an attempt.

**Pre-landing planner revision:** the previously recorded embedding-cache, session-window degradation, and hypothesis-reread stops are resolved by four-variable isolation, preceding-episode classification, and the shared `_clean_observations` helper. Do not reopen them as unresolved design conflicts. Before data-touching scripts/tests, export all four isolated absolute paths; scripts must report and validate them (V17).

Stop and report instead when:
- the preceding-episode classifier labels any rule-(a) match other than autonomous, or leaves >10% of evaluated self-observations unclassified; do not apply degradation repairs in either case;
- a step-0 contradiction changes design, outcomes, scope or verification;
- a seam differs from §6;
- V3 or V8 finds no threshold satisfying its rule;
- scope would need to grow;
- a protected file looks wrong;
- **Pre-landing planner revision:** a missing live-data copy blocks only V3, V5, V8, V11, V15, V16 and integrated acceptance (mark blocked checks not-run, never passed). All other implementation and verification proceeds; thresholds stay config values until calibrated. This overrides copy-dependent stage-exit stops, but never permits marking a blocked stage accepted;
- the same failure repeats without a new testable diagnosis, or the budget is exhausted.

Continuation: maintain `/docs/specs/SPEC-0004-run.md` (status, spec revision, run base, current commit, per-stage status, checks with evidence pointers, repairs used, open work, resume action). Update it after each stage exit, each repair round, and before ending a session. Resume by reading this spec, the run record and git state, then running the resume baseline. Never advance past a failing required check. Under the pre-landing planner revision, missing-copy checks are blocked/not-run, not failed; proceed with other work while leaving stage acceptance pending. Continuous mode: proceed through stages without approval; one local commit per stage is fine.

Never mark an unrun check as passed. Never run `--apply` against the live `DATA_DIR`.

## 10. Report back

1. Start revision; step-0 expected vs actual for every §4 row, including the Jev-skip log evidence and the `source` distribution.
2. Per stage: each V-check with command, output, pass/fail/not-run, repair rounds used; stage exit status. Integrated acceptance separately. Run record path.
3. O-ID → evidence; which §7 paths actually ran.
4. Calibration tables for V3 and V8 and the chosen thresholds; dry-run outputs of all three scripts against the copy (counts plus a sample).
5. `git diff --stat` for the whole change; full diff of any protected file (should be empty); the complete changed functions with their callers.
6. Deviations from §6 with reasons; anything the spec didn't anticipate.
7. Exact commands for Lucas to run the three scripts on live data, in order, with Keeper stopped.
8. Create `/docs/specs/index.md` (if absent) with this spec's row: ID, title, status.

## 11. Landing

Filled from the review.
