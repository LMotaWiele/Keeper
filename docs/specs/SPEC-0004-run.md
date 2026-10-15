# SPEC-0004 run record

## Status

Pre-landing planner revision synchronized with the user's resolution of all three step-0 stops. Four-path isolation, preceding-episode classification, and shared observation cleaning are now explicit in the spec. Documentation revision committed before resuming implementation; no stage is accepted.

- Spec: `SPEC-0004-autonomous-loop-hygiene.md`, including this session's user-requested pre-landing planner revision.
- Run base: `1c06f3c1735b6cabf5e25f2a9f4856ac5c2939e1` (descendant of `21a9136`, verified with `git merge-base --is-ancestor`).
- Branch: `spec-0004-loop-hygiene`.
- Current commit: this run record's local commit (resolve with `git log -1`); no push.
- Pre-existing uncommitted changes: untracked `.agent/` and `AGENTS.md`; excluded from this change.
- Supplied copy: `/home/george-contis/keeper-data-copy-20261015` (corrected by user and verified present with `episodic.db`, `chroma/`, and `state/`). The earlier `/home/keeper-data-copy-20261015` path was incorrect. No empty replacement was created.
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

- Resume baseline: **not-run**, stopped at source-only grounding before data-touching imports due isolation contradiction. No pre-existing test failures have been inferred.
- Stage 1: not started; V1–V5 not-run.
- Stage 2: not started; V6–V11 not-run.
- Stage 3: not started; V12–V16 not-run.
- V17: specified by planner revision; not implemented/not-run.
- Copy-dependent checks V3, V5, V8, V11, V15, V16 and integrated acceptance: **not-run**, never passed. Correct copy is now present; the former missing-copy blocker is cleared.
- Calibration tables, chosen thresholds, script dry-runs, O1–O12 evidence: none yet.
- Protected files: untouched. No new scripts or production functions, so no changed-function/caller inventory or live repair commands yet.

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

## Resume action

Resume from step 0 under the four isolated absolute paths above, run the existing-suite baseline before production edits, then implement stages in order without the spec tool. Do not reopen the three resolved stops. Verify the classifier's 120-second provenance and safety conditions before repairs, and calibrate thresholds against the copy. Do not mark implemented or accepted while required checks remain unrun.
