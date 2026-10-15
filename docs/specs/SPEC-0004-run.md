# SPEC-0004 run record

## Status

Stage 1 implemented and verified (V1–V5 and stage-1 V17); stages 2–3 pending. Four-path isolation, preceding-episode classification, and shared observation cleaning are explicit in the spec. Final acceptance remains pending.

- Spec: `SPEC-0004-autonomous-loop-hygiene.md`, including this session's user-requested pre-landing planner revision.
- Run base: `1c06f3c1735b6cabf5e25f2a9f4856ac5c2939e1` (descendant of `21a9136`, verified with `git merge-base --is-ancestor`).
- Branch: `spec-0004-loop-hygiene`.
- Current commit: this run record's local commit (resolve with `git log -1`); no push.
- Pre-existing uncommitted changes: untracked `.agent/` and `AGENTS.md`; excluded from this change.
- Supplied copy: `/home/george-contis/keeper-data-copy-20261015` (corrected by user and verified present with `episodic.db`, `chroma/`, and `state/`). The earlier `/home/keeper-data-copy-20261015` path was incorrect. No empty replacement was created.
- Repairs used: 0 in all stage-1 groups. Baseline and stage-1 checks recorded below.

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
- Stage 2: not started; V6–V11 not-run.
- Stage 3: not started; V12–V16 not-run.
- V17: specified by planner revision; not implemented/not-run.
- Copy-dependent checks V3, V5, V8, V11, V15, V16 and integrated acceptance: **not-run**, never passed. Correct copy is now present; the former missing-copy blocker is cleared.
- Stage-1 calibration, dry-run and O1–O4 evidence below; O5–O12 pending.
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

## Resume action

Proceed to stage 2 after the stage-1 local commit; keep four-path isolation. Then stage 3 and integrated acceptance. Do not mark implemented or accepted while required checks remain unrun.
