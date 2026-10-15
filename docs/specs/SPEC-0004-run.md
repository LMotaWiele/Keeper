# SPEC-0004 run record

## Status

All three stages implemented and verified. V1–V17 and integrated acceptance passed, with the separately recorded pre-existing direct `test_goals.py` failure unchanged. Final pytest: 172 passed. Four-path isolation and all planner revisions applied; only the data copy was repaired.

- Spec: `SPEC-0004-autonomous-loop-hygiene.md`, including this session's user-requested pre-landing planner revision.
- Run base: `1c06f3c1735b6cabf5e25f2a9f4856ac5c2939e1` (descendant of `21a9136`, verified with `git merge-base --is-ancestor`).
- Branch: `spec-0004-loop-hygiene`.
- Current commit: this run record's local commit (resolve with `git log -1`); no push.
- Pre-existing uncommitted changes: untracked `.agent/` and `AGENTS.md`; excluded from this change.
- Supplied copy: `/home/george-contis/keeper-data-copy-20261015` (corrected by user and verified present with `episodic.db`, `chroma/`, and `state/`). The earlier `/home/keeper-data-copy-20261015` path was incorrect. No empty replacement was created.
- Repairs used: stage 1 zero; stage 2 V6 one test-namespace repair; stage 3 zero required-check repairs. Two review hardenings (one-time rebuild marker and user-memory refill) added with regression checks. Baseline and all checks below.

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

### Initial path report (historical; superseded below)

No isolated copy was selected. Shell/default paths, resolved without importing settings or loading `.env`:

```
DATA_DIR=/home/george-contis/Documents/Keeper/data
MIDTERM_DB_PATH=/home/george-contis/Documents/Keeper/data/episodic.db
CHROMA_DB_PATH=/home/george-contis/Documents/Keeper/data/chroma
EMBEDDING_CACHE_DIR=/home/george-contis/Documents/Keeper/data/fastembed
```

These are unsafe defaults, NOT an approved copy. Effective settings values after dotenv loading were not inspected. No script or test touching data was run. A synthetic temporary directory has not yet been created.

## Resolved step-0 stops / implementation observations

1. **Isolation — resolved by planner:** export `EMBEDDING_CACHE_DIR` as a fourth isolated absolute path. Every new script reports all four paths and refuses escapes before loading data; V17 covers SQLite, Chroma, and cache escapes.
2. **Degradation — resolved by planner:** session windows are removed. Rule (b) uses preceding-episode provenance within 120 seconds. Stop before applying repairs if any rule-(a) match is not classified autonomous, or unclassified observations exceed 10%. Unclassified observations are not automatically degraded.
3. **Clean rebuild — resolved by planner:** one `_clean_observations` helper feeds layer 1, layer 2, and `generate_hypotheses`.
4. **Candidate verification observation:** preserve A ∪ B exactly. Verify V5's source-less assertion on the supplied historical copy; do not infer failure from hypothetical future data.
5. **Activity observation:** action logs are capped at 50 per goal; compute the specified seven-day block from the available logs, without claiming unretained records were counted.

The three resolved items are not open design conflicts or reasons to stop implementation. Classifier safety checks and all other actual required checks remain enforceable.

## Verification / stage status

- Resume baseline: pytest **80 passed**; direct budget/opinions/research/routing passed. Direct `test_goals.py` has a pre-existing failure at `test_invalid_condition_creates_unbounded` (`assert len(new) == 1`); preserved unchanged. Logs: `/tmp/spec0004-baseline.log`, `/tmp/spec0004-baseline-direct-all.log`.
- Stage 1: V1–V5 passed; 110 pytest checks passed including 30 new mocked checks; direct runners unchanged from baseline. Exit accepted with the recorded pre-existing direct-run failure, not certified as a wholly green direct suite.
- Stage 2: V6–V11 passed; pytest 131 passed. One V6 test-namespace repair consumed; direct runners unchanged from baseline.
- Stage 3: V12–V16 passed; 172 pytest checks passed after review hardening; direct runners retain the baseline results.
- V17: passed for all three scripts, every escape and applicable dry/apply/rebuild mode.
- Copy-dependent checks V3, V5, V8, V11, V15, V16 and integrated acceptance: passed on the supplied copy, not on live data.
- O1–O12 evidence is mapped to checks in the stage tables.
- Protected files: untouched. Stage-1 changed-function inventory below; final live repair commands will be recorded after all scripts exist.

## Earlier resume at e047afd (historical; superseded by planner resolution below)

- Resume base: `b5c2c57ed8ed09ad9e00c46476682418b6c325d5`; same branch and same unrelated untracked files.
- Re-read spec and run record. The three requested planner changes are already recorded; corrected V17 table formatting without changing its condition.
- Checked each supplied copy path with `test -e`; all four are missing. Requested absolute isolation paths are:
  - `DATA_DIR=/home/keeper-data-copy-20261015`
  - `MIDTERM_DB_PATH=/home/keeper-data-copy-20261015/episodic.db`
  - `CHROMA_DB_PATH=/home/keeper-data-copy-20261015/chroma`
- These are requested path resolutions, not runtime settings verified through an import; no settings import, test, data script, or embedding operation was run. No paths were created or substituted for the supplied copy.
- At that earlier resume, the three design stops were recorded as unresolved. The subsequent explicit planner resolution below supersedes that status; none of the three remains open.
- Baseline and all V-check statuses remain not-run; no repair attempts consumed. No protected files changed, no push.

## Planner-resolution revision

Resume base: `e047afd249ffbaf5cc64233f4d86947e7a323877`, same branch and unrelated untracked files. User explicitly resolved all three stops. The spec at that revision still contained superseded requirements: §4 line 76 (three paths), §6.0 line 123 (three-path refusal), §6.3 lines 174–175 (layer-1-only filter and session windows), line 177 (unqualified layer-2 preservation), and line 182 (open-session seam). Corresponding §7/§8 checks also required updating. These have been replaced, not retained as alternatives. §9 explicitly records resolution and classifier safety stops. No production or protected files changed; no data-touching tests run for this documentation-only revision.

Required absolute paths for the next data-touching command:

```sh
export DATA_DIR=/home/george-contis/keeper-data-copy-20261015
export MIDTERM_DB_PATH=/home/george-contis/keeper-data-copy-20261015/episodic.db
export CHROMA_DB_PATH=/home/george-contis/keeper-data-copy-20261015/chroma
export EMBEDDING_CACHE_DIR=/home/george-contis/keeper-data-copy-20261015/fastembed
```

These are prescribed paths; effective runtime configuration has not yet been imported/verified. Repair counters remain zero.

## Implementation resume: stage 1

Start revision `64f6620`; unrelated untracked `.agent/` and `AGENTS.md` excluded. All four effective settings paths were printed and confirmed equal to the prescribed copy paths before baseline. CPU-only embeddings used the copy's cache; no GPU work or live state writes.

Grounding additions from the copy (earlier table is historical):
- Source distribution `(type, source, count)`: event/autonomous_artifact 48; event/keeper_response 10; event/user_turn 32; fact/user_turn 4; opinion/autonomous_artifact 115; preference/user_turn 3; research/autonomous_artifact 91; self_observation/self_observation 93.
- Goal logs: two active goals, 50 records each, respectively 36 and 38 synthesis entries; unique referenced A documents 71, all present. Approximate spec incident count is consistent.
- Jev-skip evidence now verified in copy `logs/keeper.log`, including `2026-10-14 12:46:22,907`, `2026-10-15 01:03:24,525`, `2026-10-15 01:33:25,855`. Repeated skips confirmed; not literally an hourly-only cadence.
- Standalone opinions and embedded persistence source expectations remain confirmed by source inspection. Classifier provenance validation is scheduled for stage 3.

### Stage-1 verification

Commands use the four exports above, `.venv/bin/python`, and repository cwd.

| Check | Result / evidence |
|---|---|
| V1 | PASS: watermark selection exclusion/new input/legacy eligibility; save/load; source query across >100 internal rows (`test_patch04.py`) |
| V2 | PASS: malformed JSON/shapes/indices, one episode, same Amsterdam date rejected; two-day sources and origin stored; rejection advances watermark |
| V3 | PASS: actual CPU embedding calibration below; real incident paraphrases merged in temporary Chroma, no confidence gain without sources, +0.1 with new source, unrelated sentence new; mocked exact/None/invalidated/empty cases |
| V4 | PASS: 10 cycles with single bedtime turn and autonomous artifacts, zero writes, one rejected synthesis call |
| V5 | PASS: dry-run A=71, B=77, union=77; all candidates source-less; apply invalidated 77; repeat invalidated 0; search returned no candidate IDs |
| V17 (purge) | PASS: each of three path escapes refuses before loader, all four paths printed; valid isolation reaches loader; remaining scripts checked in their stages |
| Shared | `python -m pytest -q`: **110 passed**; direct runners retain baseline results (only pre-existing direct goals failure) |

V3 calibration: fresh embeddings via existing BAAI/bge-small-en-v1.5 embedder; random seed 4; 71 A documents, 200 non-A documents. Highest admissible t = **0.8886954917228408**, positive neighbour recall 64/71 (90.1408%), negative matches 0/200. Config default set to t.

| Distribution | min | p10 | median | p90 | p99 | max |
|---|---|---|---|---|---|---|
| A nearest other A | 0.844429 | 0.888695 | 0.942765 | 0.981917 | — | 0.999456 |
| sampled non-A nearest A | 0.559458 | — | 0.677580 | 0.735855 | 0.769494 | 0.809616 |

Calibration arrays/IDs and real paraphrase pair: `/tmp/spec0004-calibration-v3.json`; output `/tmp/spec0004-calibration-v3.log`. Actual incident pair differs by “without pushing back” / “without pushback”. V5 output: `/tmp/spec0004-v5-dry.log`, `-apply.log`, `-repeat.log`. Sample candidate `07ecc48554137bbb` escalates the single bedtime message into a recurring sleep-for-intellectual-momentum pattern. 1161 other source-less documents untouched. Script safety refuses evidenced A candidates rather than silently weakening A ∪ B (none encountered).

Changed stage-1 functions and callers: `AutonomousEngine._synthesis_inputs`, `_synthesis_has_new`, `_synthesize` → `take_action` → `run_loop`; `Goal.to_dict/from_dict` via `GoalSystem.save/load`; `SemanticMemory.store/_reinforce/_register_refs` via autonomous, consolidator, memory facade and memory tools (signatures preserved apart from optional dedup control); similarity helper via purge and later research; purge `run/main/flood_candidates/source_ids`; shared `require_isolated_paths`. Complete functions are available in the local commit diff (`git show`), not copied into this record. Protected-file diff empty. No repair rounds consumed.

## Stage 2 verification

Stage-1 commit `7dc2270`. Stage 2 preserves the conversation opinion-formation path; research merge lives only in `ResearchEngine.research_topic`. Shared helper and guard unchanged. Threshold calibration used the unmodified pre-compaction opinion snapshot.

| Check | Result / evidence |
|---|---|
| V6 | PASS: research paraphrase corroborates external only; novel/disabled/non-external cases form new opinions; evidence capped/deduplicated, conviction unchanged, no new episode on merge |
| V7 | PASS: external-first pruning with mixed/only-nonexternal registries and protected IDs |
| V8 | PASS: no comparable opinions => no false alert, 3 aligned comparable => alert, disagreement => no alert; calibration below |
| V9 | PASS: new fields snapshot/restore and legacy defaults |
| V10 | PASS: autonomous memory default exclusion and configurable cap; user episodes retained |
| V11 | PASS: dry/apply/repeat on copy, 97 externals → 19 greedy clusters; total 22 with 3 nonexternals unchanged; raw nonexternal JSON value tokens byte-identical; user positions untouched; embedded store equals standalone; loader restores 22; second apply hashes identical |
| V17 (compact) | PASS: four paths printed, all three escapes rejected before loader for dry/apply, valid isolation accepted |
| Shared | `python -m pytest -q`: **131 passed**, direct runners unchanged from baseline |

V6 repair round 1: first run had 4 new test failures because `import tools.web_search as search_mod` resolved the package-exported StructuredTool, not the module. Diagnosis: wrong mock target, production not exercised. Fixed only the new tests with `importlib.import_module`; full suite passed on rerun. Logs `/tmp/spec0004-stage2-tests.log`, `/tmp/spec0004-stage2-repair1.log`. No other repairs consumed.

V8 manual labels (all 100 positions read): **external verification / bounded harness self-improvement** 66; **real-world action planning / low-friction progress monitoring** 31; 3 nonexternals. Per-ID labels and distributions in `/tmp/spec0004-calibration-v8.json`; reviewed positions `/tmp/spec0004-opinion-labels.txt`; output `/tmp/spec0004-calibration-v8.log`. Highest threshold meeting >=90% neighbour coverage in each positive cluster is **0.8590785323570129** (the 31-position cluster is binding). Cross-cluster max 0.828051, false matches 0%; three nonexternal max similarities to any external: 0.606805, 0.669913, 0.594830 (all below t). Config default set to t.

| Cluster nearest same-set neighbour | min | p10 | median | p90 | max |
|---|---|---|---|---|---|
| action planning (31) | 0.798639 | 0.859079 | 0.896160 | 0.907395 | 0.914347 |
| verification (66) | 0.867048 | 0.888252 | 0.917132 | 0.935590 | 0.943742 |

Compaction uses the specified greedy **canonical** similarity, not transitive connected components, so 19 canonicals rather than the two broad manually labelled research topics is expected. Largest canonical `c5fe7eaec3a7` had 44 members. Dry-run output `/tmp/spec0004-v11-dry.log`; apply/repeat `/tmp/spec0004-v11-apply.log`, `-repeat.log`; second-apply SHA files `/tmp/spec0004-v11-sha-before`, `-after`. Previous snapshot `/tmp/spec0004-opinions-before.json`. Every external was assigned exactly once.

Stage-2 changed functions/callers: `Opinion.to_dict/from_dict` (dataclass fields), registry `_prune/detect_drift`; `ResearchEngine.research_topic` via autonomous `take_research_action/take_action` and `scripts/diag_autonomous.py`; `EpisodicMemory.format_for_prompt` via memory facade; compaction `compact_snapshot/_updated_json/run/main` via CLI. Full functions in local commit diff. Compaction carries prior corroboration mass instead of resetting counters on rerun; initial copy had zero new counters, so first-run count equals cluster size−1. Atomicity is per file via `core.atomic`, consistent with existing persistence; retry reconciles embedded and authoritative stores. Protected diff empty.

## Stage 3 verification and final acceptance

Stage-2 commit `0534cf9`. All data commands used the four exports above. No live directory, identity file, or prohibited session-trace tree was read or edited. The actual self-model rebuild used the configured LLM; replay LLM/search seams were mocked.

| Check | Result / evidence |
|---|---|
| V12 | PASS: flag off prevents successful research observe; flag on preserves legacy behavior; actual analysis prompt fixture has deterministic seven-day action counts and no summary prose |
| V13 | PASS: analysis and Jev skip set analysed=397; background loop attempts only at 5 new observations |
| V14 | PASS: shared helper feeds layer 1, layer 2, hypotheses; clean window recovered behind 2000 degraded observations |
| V15 | PASS: 120-second boundary, same-user nearest predecessor, both refusal conditions and 10% boundary, dry/apply/repeat, unclassified preservation, sparse rebuild tested; actual copy counts below |
| V16 | PASS: actual v13→v14, previous-model content archived, active model and hypotheses contain no harness phrase; fresh rebuild omits prior context; mock sparse case leaves model unchanged |
| V17 (degrade) | PASS: all four paths print first; each DB/Chroma/cache escape refused before loader, in all four apply/rebuild combinations |
| Shared | Final `python -m pytest -q`: **172 passed**; direct runners unchanged from baseline |

Actual dry-run `/tmp/spec0004-v15-dry.log`: observations=93; rule-(a)=67; rule-(b)/autonomous=83; conversation=10; unclassified=0 (0%); rule-(a) mismatches=0. Apply degraded 83; repeat degraded 0. Ten conversational observations remained clean. Logs `/tmp/spec0004-v15-apply.log`, `-repeat.log`. Actual rebuild `/tmp/spec0004-v15-rebuild.log` produced **v14**, observation_count=observations_analysed=397, three regenerated hypotheses. Before snapshot `/tmp/spec0004-model-before-rebuild.json`. The complete old v13 derived content (excluding normal bookkeeping/embedded registries) is in `_recursion_guard.model_history`. Active v14 JSON (excluding historical archive and separate opinion registry) and the prompt block contain no `Researched and formed opinion`. The archive intentionally retains v13's old claim; deleting it would violate the spec.

Review hardening: persisted `_spec0004_rebuild` from/to-version marker prevents repeated explicit rebuild from creating v15; repeated `--apply --rebuild` preserved the model file hash (`/tmp/spec0004-v15-sha-before`, `-after`, `-rebuild-repeat.log`). Formatting refills past high-ranked autonomous artifacts so a flood cannot starve user facts below the initial 40-row window; regression fixture has 100 artifacts before a user fact. Unit paths were made temporary rather than machine-specific. No required-check failure preceded these hardenings.

One extra inspection compared embedded opinions to raw standalone JSON after a normal self-model save and found newly serialized dataclass defaults. Diagnosis: normal snapshot normalization, not changed opinions. Rechecking through `OpinionRegistry.restore/snapshot` confirmed equality. Standalone non-external JSON remains unchanged. No protected or production behavior was weakened to satisfy that inspection.

### Integrated acceptance

Command after scripts applied in order 1→2→3: four exports, then `.venv/bin/python /tmp/spec0004-integrated.py`. Harness artifact retained at that path; output `/tmp/spec0004-integrated.log`, result `/tmp/spec0004-integrated-result.json`. Uses `load_companion_no_loops`, real dispatch, gates, local embeddings/research merge, rating and `build_context`. Network LLM/search synthesis seams are mocked with real incident/research paraphrases; research cosine 0.928967. Only the off-limits identity block is omitted by pointing `soul_file_path` to a nonexistent copy-local test path; no identity contents were read. Budget was made permissive for the replay and selection targeted the affected action paths.

**PASS:** twelve `take_action` cycles (two legacy-eligible synthesis rejections, then ten research corroborations, rotating both goals) with no new user episodes. Semantic/opinion/self_observation counts **(1238, 22, 93) → (1238, 22, 93)**. No new artifacts. `build_context` for `I stayed up late again working on the experiment` retained the user-memory block, with no research_completed lines there, no invalidated flood content in semantic formatting, no zero-disagreement alert, and no harness phrase in the self-model block. This is acceptance on the copy only, not a live deployment assertion.

### Review decisions / deviations

- Four-path guard intentionally validates containment, as specified; it does not prohibit Lucas's explicit stopped-live repair commands below. Agent invocations used only the copy.
- Existing exact-hash invalidation/revalidation behavior is retained; nearest-neighbour matching excludes invalidated documents. No new revalidation policy was introduced.
- Research considers all external opinions as specified, including low-conviction ones; conviction stays unchanged on corroboration. No unscheduled abandonment/revalidation policy was added.
- Compaction preserves accumulated corroboration on retries instead of resetting to zero; first-run copy counters match cluster size−1. Per-file atomic writes match existing persistence, not a new multi-file transaction.
- Seven-day scheduler statistics explicitly cover retained logs, not evicted records.
- Rebuild archive retains content of v13 under the existing history format, whose normal `record_model` omits version metadata; marker records from_version=13 and to_version=14.
- The known direct-run goals baseline failure was not repaired or hidden. Pytest and the other direct runners remained green.

### Changed functions and callers

In addition to earlier inventories: `SelfModel.__init__` optional goals injection; `_clean_observations` via layer 1/layer 2/hypotheses; `_autonomous_activity` via `ANALYSIS_PROMPT`; `_layer1_observation_update` via `update_model` and repair script; existing autonomous research observe call gated by config; degradation `_tags/classify_observations/validate_classifications/run/main` via CLI. Full changed functions and their calling context are in `git diff 64f6620..HEAD -- core goals memory scripts config` (include the stage-3 commit). New tests reside only in `test_patch04.py`. Final whole-change stat and protected diff are recorded at landing; no protected files changed.

### Operator commands (NOT run on live data)

With Keeper stopped, from `/home/george-contis/Documents/Keeper`, Lucas can run these after reviewing dry-runs. Paths are deliberately explicit and all four must remain isolated together when using a copy. These particular commands intentionally target live data and are for Lucas only:

```sh
export DATA_DIR=/home/george-contis/Documents/Keeper/data
export MIDTERM_DB_PATH=/home/george-contis/Documents/Keeper/data/episodic.db
export CHROMA_DB_PATH=/home/george-contis/Documents/Keeper/data/chroma
export EMBEDDING_CACHE_DIR=/home/george-contis/Documents/Keeper/data/fastembed
.venv/bin/python scripts/purge_synthesis_flood.py
.venv/bin/python scripts/purge_synthesis_flood.py --apply
.venv/bin/python scripts/compact_opinions.py
.venv/bin/python scripts/compact_opinions.py --apply
.venv/bin/python scripts/degrade_autonomous_observations.py
.venv/bin/python scripts/degrade_autonomous_observations.py --apply
.venv/bin/python scripts/degrade_autonomous_observations.py --rebuild
```

Classifier safety stops must not be bypassed if the live dataset differs from the verified copy. Rebuild performs configured LLM calls; a second successful rebuild request is a no-op.

## Landing

Implemented by the coding assistant on `spec-0004-loop-hygiene`: stage 1 `7dc2270`, stage 2 `0534cf9`, stage 3 `23d2c72`; final documentation commit contains this landing record. Spec header and index marked `implemented` after the repaired-copy integrated acceptance passed. No pushes or remote operations.

O-ID evidence: O1→V1/V4; O2→V2/V4; O3→V3; O4→V5; O5→V6; O6→V7/V11; O7→V8; O8→V10/integrated; O9→V11; O10→V12/integrated; O11→V13; O12→V14/V15/V16. All §7 paths exercised except retry/cancel/timeouts (explicitly excluded by the spec); empty collections, legacy state, same-day/malformed evidence, unrelated text, novel research, mixed pruning, sparse rebuild and provenance safety boundaries have unit evidence in `test_patch04.py`.

Final evidence artifacts:
- `/tmp/spec0004-final-tests.log`: 172 passed.
- `/tmp/spec0004-final-test_*.py.log`: four direct runners passed; direct goals failure identical to baseline.
- `/tmp/spec0004-integrated.py`, `-integrated.log`, `-integrated-result.json`: copy acceptance harness/results.
- `/tmp/spec0004-complete-functions.diff`: complete changed functions with surrounding calling context (`git diff --function-context 64f6620 23d2c72 -- config core goals memory scripts`); inventories above name cross-module callers.
- `/tmp/spec0004-protected.diff`: empty; protected files and their docs copies unchanged.
- `/tmp/spec0004-final-stat.txt`: whole-change stat, generated with `git diff --stat 64f6620` before the final documentation-only landing additions. Implementation stage commits alone: 18 files, 1935 insertions, 113 deletions.

Artifacts under `/tmp` are local session evidence, not committed copies of incident data. The durable numeric results, diagnoses, commands, and outcome mapping are recorded here. The only remaining untracked paths are the pre-existing `.agent/` and `AGENTS.md`; neither was staged.

## Resume action

No implementation work queued. Review the local stage commits and run record; deployment/live repairs belong to Lucas. Never push. If resuming implementation changes, rerun baseline under four-path isolation first.
