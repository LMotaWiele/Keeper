# SPEC-0004 run record

## Status

Step 0 stopped on design/isolation contradictions under §9. Pre-landing planner revision recorded in §4, §6.0, §8 (V17), and §9. Implementation has not started; no stage is accepted.

- Spec: `SPEC-0004-autonomous-loop-hygiene.md`, including this session's user-requested pre-landing planner revision.
- Run base: `1c06f3c1735b6cabf5e25f2a9f4856ac5c2939e1` (descendant of `21a9136`, verified with `git merge-base --is-ancestor`).
- Branch: `spec-0004-loop-hygiene`.
- Current commit: this run record's local commit (resolve with `git log -1`); no push.
- Pre-existing uncommitted changes: untracked `.agent/` and `AGENTS.md`; excluded from this change.
- Supplied copy: literal `<COPY>` placeholder; no usable absolute copy path supplied. Missing-copy checks alone do not prevent other work under the planner revision.
- Repairs used: 0 in all groups. No implementation tests attempted.

## Step 0: expected versus actual

Read-only source inspection; no live state, secrets, session traces, or embedding model accessed.

| Grounding row | Expected versus actual | Evidence |
|---|---|---|
| Start revision | Later than `21a9136`: confirmed, base above | `git log -1`, ancestry check |
| Autonomous synthesis | Matches: description recall, no evidence IDs or novelty gate. Episodic recall is not query-ranked; description applies to semantic recall | `goals/autonomous.py:301–334`, `memory/__init__.py:144–149` |
| Selection seam | Matches: unavailable set before selection | `goals/autonomous.py:139–151` |
| Semantic dedup | Matches: exact SHA256[:16], unconditional confidence bump capped at 1.0; also clears invalidation | `memory/semantic.py:22–24,103–135` |
| Collection metric | Matches cosine and 1-distance relevance | `memory/semantic.py:69,211` |
| Research opinion/episode | Matches external opinion without similarity gate; autonomous research episode, salience 0.8 | `goals/research.py:213–241` |
| Opinion IDs | Timestamp included as expected; truncated hash is not literally collision-proof | `core/opinions.py:153–156` |
| Pruning | Matches lowest conviction, protected ID excluded, no origin ordering | `core/opinions.py:739–752` |
| Alert 4 | Matches empty comparable set alert, provided at least five known user positions exist | `core/opinions.py:607–625` |
| Memory block | Matches opinion/research prioritization and char budget | `memory/episodic.py:540,550–605` |
| Recall source | Confirmed: returned row dictionaries include source | `memory/episodic.py:303–325` |
| Autonomous observation | Matches research summary passed to observe; conversation call is separate | `goals/autonomous.py:168–181`, `core/loop.py:560–563` |
| Compact observation | Matches truncation and Given/responded string, with self_observation prefix | `core/self_model.py:308–331` |
| Bookkeeping | Confirmed bounded window length assigned; cumulative counter used for trigger | `core/self_model.py:458,712–716` |
| Freeze cause | Not verified: no permitted runtime-log evidence. Code supports Jev skips, not proof of historical recurrence | `core/self_model.py:434–441`, `config/settings.py:78` |
| Layer-1 input | Matches latest 50, no degraded filter. Hypothesis generation independently reads 30 unfiltered observations | `core/self_model.py:387–389,546–562` |
| Session windows | maxlen=50 confirmed; live count/date not verified. Current session is omitted from snapshot, contradicting repair assumptions | `environment/time_sense.py:87,102–120,322–341` |
| Authoritative opinions | Confirmed standalone loaded after embedded; both saved; embedded fallback remains | `core/loop.py:199–200,412–415`, `scripts/diag_common.py:22–23`, `core/self_model.py:740–763` |
| Source reliability | Unknown defaults to user_turn, explicit user-turn caller confirmed; distribution by type not-run (no copy) | `memory/__init__.py:14–28`, `memory/episodic.py:53,177,216`, `core/loop.py:473–474` |
| Data paths | Three paths independently resolved as planner stated; fourth independent embedding cache path also writes under live defaults | `config/settings.py:68,71–74`, `memory/semantic.py:51–57` |
| Test runner | Mixed pytest-style tests and direct async runners. Both needed; no tests executed yet | `test_research.py:123–124`, `test_opinions.py:161–162`, `test_patch01.py:37` |
| Flood size | Live size unverified; action-log cap 50 confirmed | `goals/system.py:190`, `goals/action_rating.py:189–190,268–269` |

### Path report

No isolated copy was selected. Shell/default paths, resolved without importing settings or loading `.env`:

```
DATA_DIR=/home/george-contis/Documents/Keeper/data
MIDTERM_DB_PATH=/home/george-contis/Documents/Keeper/data/episodic.db
CHROMA_DB_PATH=/home/george-contis/Documents/Keeper/data/chroma
EMBEDDING_CACHE_DIR=/home/george-contis/Documents/Keeper/data/fastembed
```

These are unsafe defaults, NOT an approved copy. Effective settings values after dotenv loading were not inspected. No script or test touching data was run. A synthetic temporary directory has not yet been created.

## Stop conditions / open design conflicts

1. **Isolation:** `SemanticMemory._get_embeddings` uses independently configured `EMBEDDING_CACHE_DIR`, default `./data/fastembed`. `load_companion_no_loops()` warms embeddings. Three-variable isolation and V17 as requested do not prevent cache writes to live data. Proposed repair: also export an absolute isolated `EMBEDDING_CACHE_DIR`, and extend script path validation/reporting to it (in addition to the required three).
2. **Session-based degradation:** `TimeSense.snapshot()` serializes only completed `sessions`, not `_current_session`. A missing window cannot establish autonomous origin. Rule (b) can degrade real user-session observations; bounded history also omits old sessions. Proposed repair: treat outside-window candidates as uncertain/report-only unless provenance establishes autonomous origin; retain rule (a) as an automatic degradation rule. This changes the prescribed repair candidate policy and needs a planner decision.
3. **Clean rebuild:** `generate_hypotheses()` independently reads unfiltered self-observations. Filtering only layer 1 does not ensure O12/V16. Proposed repair: explicitly authorize sharing the degraded filter with hypothesis generation during rebuild.
4. **Repair verification conflict:** A includes every synthesis artifact regardless of evidence; V5 requires every candidate source-less. A future evidenced artifact makes those requirements conflict. Proposed repair: decide whether V5's source-less assertion is copy-specific (expected historical incident data) or whether A should be restricted. Do not silently change A ∪ B.
5. **Activity window:** action logs retain at most 50 records per goal. Seven-day counts can only describe retained records, not guaranteed full activity. Proposed clarification: label the block retained-log activity counts.

These are independent of the missing-copy exception. No production changes were made to choose policies silently.

## Verification / stage status

- Resume baseline: **not-run**, stopped at source-only grounding before data-touching imports due isolation contradiction. No pre-existing test failures have been inferred.
- Stage 1: not started; V1–V5 not-run.
- Stage 2: not started; V6–V11 not-run.
- Stage 3: not started; V12–V16 not-run.
- V17: specified by planner revision; not implemented/not-run.
- Missing-copy dependent checks: V3, V5, V8, V11, V15, V16 and integrated acceptance **blocked/not-run**, never passed. Unit portions may proceed once independent step-0 contradictions are resolved.
- Calibration tables, chosen thresholds, script dry-runs, O1–O12 evidence: none yet.
- Protected files: untouched. No new scripts or production functions, so no changed-function/caller inventory or live repair commands yet.

## Resume action

Resolve the step-0 design conflicts through a planner revision, then resume baseline under fully isolated paths (including embedding cache), implement without the spec tool, and proceed with all non-copy-dependent checks. Supply an actual absolute copy path for calibration/data repair/integrated acceptance. Do not mark implemented or accepted while required checks remain unrun.
