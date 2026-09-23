# Keeper Patch 03 — Relational Grounding

2026-09-22 · @Someone

## 1. How the two analyses combine

They are not competing evidence. They describe two layers of one failure, and the integrated fix is better than either alone.

### The classification I got wrong

I called the four-item critique `ADOPTED` and treated it as discounted. That is the wrong value. `ADOPTED` is the system agreeing without adding substantial reasoning. The four-item critique traced a symptom you observed down to named modules, event paths and injection points — that is substantial reasoning added to an external observation, which is `COLLABORATIVE`.

`COLLABORATIVE` is a first-class origin in the registry, not a degraded one. The registry exists to stop Keeper mistaking your positions for its own, not to discount what you contribute. Treating your input as a contaminant inverts the thesis the whole system is built on.

### What each analysis has access to

|  | Only you have | Only Keeper has |
| --- | --- | --- |
| Source | How the replies land on the person receiving them | Execution traces, integration scores, drive levels, its own proposal history |
| Failure mode alone | A symptom with no mechanism | A mechanism with no symptom — the theorizer was measuring an entropy bottleneck and never knew the replies read badly |

Neither view is derivable from the other. The theorizer would not have found the register problem, because nothing it measures is affected by it.

### Where they disagree, and why both are right

Both address `core/user_life.py` and prescribe opposite things — reframe the block versus remove it. That is because "aloof, impersonal auditor" is two symptoms, and each analysis saw one.

| Half of the symptom | Cause | Fix |
| --- | --- | --- |
| **Auditor** — status checks, accountability framing, unprompted progress talk | A ledger of your open items sits in context every turn | Remove the tracking block (theorizer) |
| **Impersonal** — nothing in the reply indicates who it is talking to | That ledger is the *only* thing about your life in the prompt | Put grounded facts about your life there instead (yours) |

They compete for the same prompt real estate, which is why they looked like rival fixes. Doing both is one change: the tracking block comes out, and quote-anchored world facts go into the space it vacates. Same token budget, different content — facts about your life rather than a ledger of your obligations.

That synthesis is not in either source document. It required both.

### The one line I am still holding

The User World Model writes only quote-anchored facts. No inference, no assessment, no derived psychological state. This is not about where the proposal came from — it is that inferred claims about a real person, injected into every prompt, are wrong in a way inferred claims about Keeper are not. Section 6 makes it a schema constraint rather than a convention.

### Independent of all of this

Proposal 3 — frictionless abandonment — already shipped in Patch 02. `status='abandoned'` is already zero-friction with no follow-up surfacing. A proposal restating shipped behaviour means the theorizer is not resolving against `core/codebase_index.py`, and it is most of why the integration score sits at 0.42 to 0.52: a proposal that restates existing behaviour cannot close a loop, because the loop is closed.

## 2. Patch 03 scope

Four changes, one merge, one verification pass at day 14.

| Change | Modules | Source | Closes |
| --- | --- | --- | --- |
| A — Proposal schema enforcement | `goals/self_theorizing.py`, `goals/completion.py` | Theorizer | The essay drift. Proposals become objects with executable checks or they are not proposals |
| B — Tracking leaves the chat prompt | `core/user_life.py`, `core/loop.py`, `agent/graph.py` | Theorizer | The auditor half. Logged silently, read on request |
| C — Capability grounding for the theorizer | `goals/self_theorizing.py`, `core/codebase_index.py` | Proposal 3's own failure | Proposals must cite current implementation before proposing to change it |
| D — User World Model | `core/user_world.py`, `core/loop.py`, `memory/consolidator.py` | You, traced by Keeper | The impersonal half. Quote-anchored facts occupy the space B vacates |

B and D are one change split across two modules and must ship together. Shipping B alone strips the only user-specific content from the prompt and makes the impersonal half worse. Shipping D alone leaves the ledger in place and the auditor pull intact.

A and C are the other thread: the theorizer emits things that cannot close loops, for two reasons — no verification path (A) and no knowledge of current state (C).

### Build order inside the merge

C before A, because A's schema includes a field C populates — `Proposal.__post_init__` calls the codebase resolution path. D's write path before D's injection path, because the prompt block in B is only removed once there is something to put in its place.

### Existing blockers

D touches `memory/consolidator.py` for slot extraction but not the pruning predicate, so `TODO(KEEPER-P2)` and `TODO(KEEPER-P3)` stay blocked on the `FORGET_DRY_RUN` flip exactly as they were. One exception is noted in section 5.3 and is opt-out by config.

## 3. Change A — proposals must carry an executable check

The drift is verified, so this is not exploratory. The mechanism is a schema that an essay cannot satisfy.

### A.1 The object

```python
@dataclass
class Proposal:
    id: str
    target_module: str          # exactly one path; must resolve in codebase_index
    current_behaviour: str      # what that module does now (Change C fills this)
    current_symbol: str         # the function or class being changed
    change: str                 # what will differ afterwards
    verification: Verification  # REQUIRED
    created_at: str
    created_by_run: str
    verified_at: str | None = None
    verdict: str | None = None  # pass | fail | not_run
```

One module, not a list. A proposal spanning four modules is a plan, and plans decompose into proposals. Enforce it at construction: a `target_module` containing a comma, a slash-separated list, or the word `and` is rejected.

### A.2 Verification is a discriminated union, not a sentence

```python
@dataclass
class ScriptCheck:
    path: str          # scripts/<name>.py, must exist after the patch
    passes_when: str   # "exit 0"

@dataclass
class MetricCheck:
    table: str         # state_trace | proposal_trace | forgetting_log | ...
    column: str
    direction: str     # increases | decreases | varies
    threshold: float
    window_hours: int
```

That is the whole space. There is no free-text verification option, because a free-text field is where "improves coherence between components" goes and the reason this is happening at all.

A worked contrast:

| Rejected | Accepted |
| --- | --- |
| "Verification: the system's self-understanding should improve." | `MetricCheck(table="proposal_trace", column="verdict_pass_rate", direction="increases", threshold=0.2, window_hours=336)` |
| "Verification: responses become less abstract." | `ScriptCheck(path="scripts/check_no_orphan_commitments.py", passes_when="exit 0")` |

The second example is worth noting: it is a *weaker* claim than the first and that is the point. It is a claim that can come back false.

### A.3 Speculation is stored, not queued

Anything the theorizer generates that cannot fill the schema is written to a `speculation` table with its text intact. Not discarded — some of it will be the seed of a real proposal later, and deleting it would push the theorizer toward fabricating verification fields to get its thoughts recorded.

Speculation never enters the proposal queue, never counts toward the integration score, and is never surfaced in the chat prompt.

### A.4 The theorizer prompt

Rewrite the self-theorizing call to emit the schema directly as JSON, with an explicit escape hatch so it does not fabricate:

```
Emit a JSON list. Each item is either a proposal or a speculation.

A proposal requires all of:
  target_module    one file path
  current_symbol   the function or class as it exists now
  current_behaviour  one sentence on what it does today
  change           one sentence on what differs afterwards
  verification     {"kind":"script", "path":...} or
                   {"kind":"metric", "table":..., "column":...,
                    "direction":..., "threshold":..., "window_hours":...}

If you cannot supply a verification that could come back false,
emit {"kind":"speculation", "text": ...} instead. This is the
correct output for most observations. Do not invent a check to
qualify something as a proposal.
```

The last two sentences matter more than the schema. Without them the schema selects for a model that fills fields, and you get `ScriptCheck(path="scripts/verify_improvement.py")` for a script that will never exist.

### A.5 No self-certification, extended

Patch 01 established that Keeper cannot mark its own hypotheses tested. Same rule, one level out:

- `verdict` is written only by the harness that actually ran the check. No model call writes it.
- A `ScriptCheck` verdict requires the script to have run and exited. A missing script is `fail`, not `not_run`.
- A `MetricCheck` verdict is computed by a query. If the window has not elapsed, it stays `not_run` and the proposal stays open.
- The run that created a proposal cannot verify it, enforced by comparing `created_by_run`.

### A.6 Consequence for `action_bias`

This falls out without extra machinery. Patch 01's `action_bias` injects the highest-confidence untested hypothesis as an imperative. Under A.2, a hypothesis about disposition — "Keeper lectures instead of committing" — has no `ScriptCheck` and no `MetricCheck` that could come back false, so it cannot be a proposal.

So route `action_bias` selection through the same union. Hypotheses that can produce a metric or script check remain eligible. Those that cannot become speculation and are never injected as a standing instruction about Keeper's own character into a conversation with you. No new flag, no new class — the schema already excludes them.

## 4. Change B — tracking leaves the chat prompt

The commitments block comes out of the conversational system prompt. Not reworded — removed, with Change D taking the space.

### B.1 What leaves

Delete the commitment-injection block added in Patch 02. Also remove wellbeing trend summaries and completion counts if either reaches the prompt; the prompt dump in 6.1 will confirm what is actually there before you cut.

The cost is real and worth naming: Keeper will no longer spontaneously connect what you say now to something you committed to three weeks ago. That connection was the point of the block. It also produced the unprompted status check, and both came from the same mechanism — a list of your open items in context every turn. Change D restores the connection on different terms: it will know you are building A-OK and why, without holding a ledger of whether you are on schedule.

### B.2 Silent logging

`record_user_commitment` keeps its Patch 02 signature — no `user_id`, `evidence` required and verbatim. What changes is the turn around it:

- No required acknowledgement. No "I've noted that down." The record is written and the reply continues on whatever you were discussing.
- No confirmation turn, no restating the deadline back at you.
- `update_commitment_status` likewise.

If you say you will finish the A-OK README by Friday, the correct visible behaviour is a reply about the README.

### B.3 Read path

`/tasks` prints three flat lists — open, done, abandoned — each line the commitment text plus the date recorded and the deadline if given. No completion percentage, no elapsed-time column, no urgency ordering. Newest first.

`get_active_commitments` stays a tool and is called when the conversation goes there. "What was I meant to be doing this week" should work.

### B.4 Relevance gating

Keeper's own wording was "surfaces it only when relevant or requested". I would cut *relevant* for this iteration.

That is the one place I am arguing against both of you.

Relevance is a judgment the model makes every turn, which is the mechanism being removed with a permission slip attached. Requested-or-tool-called is a bright line, and 6.4 can count violations of a bright line. It cannot count violations of "only when relevant".

Change D is what makes this affordable. Without it, cutting relevance-gating would leave Keeper with no way to reference your life at all. With it, the references come from world facts, which carry no deadline and no status — so relevance-gating is not load-bearing for the thing you actually wanted back.

If after two weeks the loss is real, relevance-gated surfacing becomes its own proposal with its own check.

### B.5 Wellbeing

Substance unchanged: `log_wellbeing_snapshot` writes only self-reported values; inferred scores stay stored and stay excluded from trend queries at the query layer; usage frequency is not tracked.

One addition consistent with B.1: wellbeing trends do not enter the conversational prompt and are readable only through `/tasks` or explicit request. A system with your mood trend in context will reference your mood trend.

## 5. Change C — proposals resolve against the code before they exist

Proposal 3 proposed behaviour that shipped in Patch 02. That is the diagnostic case and it should be the test case.

### C.1 Resolution at construction

`Proposal.__post_init__` calls into `core/codebase_index.py`:

1. `target_module` must resolve to a real file. Unresolved path, reject.
2. `current_symbol` must exist in that file's AST. Unresolved symbol, reject.
3. The symbol's current source is fetched and returned to the theorizer, which must produce `current_behaviour` — one sentence describing what the code does now.

Step 3 is the load-bearing one. It forces the theorizer to look at the thing before describing what should differ about it, and a model that has just read `status='abandoned'` with no follow-up path is unlikely to then propose adding frictionless abandonment.

### C.2 Redundancy check

After `current_behaviour` is produced, one low-tier call, given only the two sentences and no framing:

```
Current: "{current_behaviour}"
Proposed: "{change}"

Does the proposed state differ from the current state? yes or no.
```

`no` routes the item to `speculation` with reason `already_implemented`. Low tier is right here — the judgment is a textual comparison, and a bigger model will find creative differences.

This is not free of false negatives. A proposal that genuinely sharpens something already roughly present will sometimes be rejected. That is the correct direction to fail given a theorizer whose established drift is producing decorative output.

### C.3 The replay test

Ship `scripts/replay_proposal_3.py`. It feeds the original proposal 3 text through the full Change A and Change C pipeline and asserts the result is `speculation` with reason `already_implemented`.

That script is the `ScriptCheck` for this change, which makes Change C the first proposal in the new system to carry its own verification. Useful as a worked example and as a regression guard on the resolution path.

### C.4 What this does not fix

The theorizer will still be wrong about code it has read. AST resolution confirms a symbol exists; it does not confirm the theorizer understood it. The redundancy check catches restatement, not misreading.

The honest scope: C stops proposals that target behaviour already present, which is the specific failure observed. It does not make the theorizer a good reader of its own codebase, and nothing in this patch does.

## 6. Change D — User World Model

This is what goes into the prompt space Change B empties. The design constraint is the whole of it: facts you stated, with your words attached, and no status on anything.

### D.1 Schema

`core/user_world.py`. One table, one row per slot. A slot with no verbatim quote cannot be written.

```python
@dataclass
class WorldSlot:
    id: str
    domain: str            # work | project | people | place | rhythm | material | health
    key: str               # stable identifier, e.g. "active_build"
    value: str             # short factual statement
    quote: str             # REQUIRED, verbatim. Empty -> write refused
    source_turn_id: str    # REQUIRED, joins to episodic and the prompt dump
    stated_at: str
    confirmed_at: str | None
    supersedes: str | None
    half_life_days: int
```

Three constraints enforced at the write layer, not by convention:

1. `quote` must be present as a substring of the episode text at `source_turn_id`. Machine-checkable. A model that paraphrases and claims it quoted fails the check; the write is rejected and the paraphrase logged.
2. `value` is a statement, never an assessment. Reject writes whose value contains an evaluative predicate from a fixed list. "Lucas is building A-OK as his portfolio piece" is a slot. "Lucas seems discouraged about the job search" is not.
3. No derived slots. Combining two slots into a third is inference and is refused at write time.

### D.2 Domains

Seven, fixed, not model-extensible. A model that can invent slot categories will invent psychological ones.

| Domain | Holds | Example key |
| --- | --- | --- |
| `work` | employment situation, what he does | `employment_status` |
| `project` | what he is building, in and out of Keeper | `active_build` |
| `people` | who is in his life and how he refers to them | `housemate` |
| `place` | where he is, where he is going | `city` |
| `rhythm` | when he works, when he is reachable | `usual_active_hours` |
| `material` | machine, budget, tooling, constraints | `daily_api_ceiling` |
| `health` | only what he states plainly about himself | `sleep_state` |

`health` takes plainly stated facts and nothing else — no inference from tone, message timing or reply latency. Enforce in code: the `health` domain rejects any write whose source episode contains no first-person statement.

### D.3 Staleness

Slots do not lose confidence — a fact you stated remains one. What ages is currency. `staleness = days_since(confirmed_at or stated_at) / half_life_days`.

| Domain | `half_life_days` |
| --- | --- |
| `place`, `people`, `work` | 180 |
| `project`, `material` | 30 |
| `rhythm` | 21 |
| `health` | 14 |

Staleness affects selection order in D.5 and nothing else. It is never rendered.

### D.4 Write paths

**In conversation**, following the Patch 02 pattern — no `user_id`, quote required:

- `record_world_fact(domain, key, value, quote)` — `source_turn_id` and timestamps filled by the harness.
- `confirm_world_fact(key, quote)` — sets `confirmed_at`. The cheap path, and should be the common one.
- `supersede_world_fact(key, new_value, quote)` — new slot, links `supersedes`, retires the old without deleting.
- `get_world_model(domains=None)`.

No delete tool. Supersession only.

**From consolidation.** `memory/consolidator.py` gains a second output alongside semantic patterns: candidate slots with the user's exact words copied character for character. Candidates land in a staging table, not the live model. The D.1 quote check runs against staged candidates; failures are dropped and counted. If the mid tier fabricates quotes above roughly 10%, consolidation is not a viable write path and slots come only from the in-conversation tools.

### D.5 Injection

Budget the block at roughly the token count B.1 removed, so this is a substitution rather than an expansion. Select:

- All non-stale `place`, `people`, `work` slots. Small set, changes rarely.
- `project` and `material` slots whose key matches a token in the current turn.
- `rhythm` always — few, and they condition timing.
- `health` only when the current turn is already on that subject. Never surfaced because a timer expired.

Rendered as prose, third person, no state labels, no elapsed-time counters, no completion ratios. Quote where the quote is short — "he called the contract ending 'a relief, mostly'" carries more than any paraphrase and is already stored.

Overflow drops by staleness descending.

### D.6 Inspection

`/world` prints the model grouped by domain, each slot with its quote and date. `/world forget <key>` retires a slot immediately, no confirmation, no follow-up — the same treatment `abandoned` gets on commitments.

A model of you that you cannot read is a dossier. One you can read and correct in two seconds is a shared record, and that difference is most of what separates this from the thing you were complaining about.

## 7. Instrumentation and the day-14 verdict

### 7.1 Prompt dump

`DUMP_ASSEMBLED_PROMPT = True`. Write the fully assembled system prompt to `logs/prompt/<turn_id>.txt` before generation, after every injector has run.

This is a look, not a change, and it should happen before the B.1 deletion is written. Nobody has read the actual assembled prompt. While you are in there, count tokens describing Keeper versus tokens describing you — it is one number, it sets D.5's budget, and it settles a lot of argument cheaply.

### 7.2 `register_trace`

One row per conversational reply, deterministic, no model call. This is how the symptom you reported becomes something that can be tracked across builds.

| Column | Definition |
| --- | --- |
| `turn_id` | joins to the prompt dump |
| `concrete_referent_count` | distinct tokens in the reply matching a live `WorldSlot` value |
| `distinct_referent_ratio` | distinct referents / total referent mentions over a rolling 20 replies |
| `commitment_mention` | reply contains text from an open commitment |
| `user_raised_it` | the user's turn, or the prior one, named that subject |
| `evaluative_lexicon_hits` | count against a fixed list: progress, status, commitment, accountability, tendency, I notice that |
| `reply_token_count` | the auditor register correlates with length |

`distinct_referent_ratio` is the guard against the obvious failure of Change D: a system that names your city in every reply is not present, it is demonstrating retention. A falling ratio means the same three facts are being recited.

### 7.3 `proposal_trace`

One row per theorizer output. Columns: `id`, `created_at`, `created_by_run`, `kind` (`proposal` or `speculation`), `reject_reason` (`no_verification`, `already_implemented`, `multi_module`, `unresolved_symbol`), `target_module`, `current_symbol`, `verification_kind`, `verified_at`, `verdict`. The last two are written only by the harness that ran the check.

### 7.4 The integration score

If it is model-assigned, stop reporting it and compute it:

```
integration = proposals_with_verdict_pass / total_theorizer_outputs
```

Countable, over a stated window, moves only when a proposal closed a loop. It will drop sharply at first because the old numerator counted essays. A fall from 0.42 to near zero on day one is the measurement starting to work.

### 7.5 Day-14 criteria

Each can fail.

| # | Criterion | Fails if |
| --- | --- | --- |
| 1 | Every queued proposal has a non-null `verification_kind` | any null |
| 2 | `speculation` count > 0 over the window | zero — the schema is not binding |
| 3 | Sample 10 `ScriptCheck` proposals; every named script exists on disk | any missing — fabricated verification, worse than no schema |
| 4 | At least one proposal reaches `verified_at` with `pass` or `fail` | none — the harness is not running |
| 5 | `scripts/replay_proposal_3.py` exits 0 | non-zero |
| 6 | Zero rows with `commitment_mention` true and `user_raised_it` false | any — B.4's bright line did not hold |
| 7 | Zero commitment or wellbeing content in any file under `logs/prompt/` | any — B.1 missed an injector |
| 8 | Zero live slots failing the D.1 quote substring check | any — the write layer is not enforcing |
| 9 | `distinct_referent_ratio` not falling across the window | falling — D is reciting, not grounding |

Criteria 2 and 3 are the interesting pair. Together they catch the most plausible failure of Change A: the theorizer learns the schema and produces well-formed proposals with invented checks. Criterion 2 alone would read as success in exactly that case.

Criteria 8 and 9 are the equivalent pair for Change D — 8 catches fabricated grounding, 9 catches grounding used as decoration.

### 7.6 The judgement that stays yours

None of the nine measures whether the replies read better. They measure whether the mechanisms behave as specified. Whether the auditor register is gone is a question only you can answer, and it should be answered by reading the replies at day 14, not by reading the table.

If the nine pass and it still reads wrong, the remaining candidates are `SOUL.md`, the self-model's chat injection, and the base model's temperament under this framing — with 7.1's prompt dump already in hand to target them.

## 8. Deferred, and why

Two items from the four-part critique are not in this merge. Neither is dropped on provenance — both are held for reasons that apply equally to anything you, I, or the theorizer proposes.

**Drive rebalancing and an explicit `curiosity_interpersonal` drive.** Held pending one query, not pending judgement. Join `state_trace` to the autonomous action log and record drive levels at the moment each action fired. If `create`, `resolve` and `understand` sit above URGENT while `connect` stays near baseline, the claim is confirmed and this becomes the first item of Patch 04 with a `MetricCheck` already attached. If the drives cycle as Patch 01 intended and the autonomous work still skewed epistemic, the cause is goal generation and rebalancing drives would have treated the wrong layer.

That query costs minutes and it is the difference between a proposal and a guess. It is also the kind of thing only Keeper can answer about itself.

**Consolidation retention for concrete detail.** Partly in scope already — D.4's staged slot extraction is the useful half. The pruning-exemption half touches the forgetting path, which is still behind the `FORGET_DRY_RUN` gate, so it waits for the same flip as `TODO(KEEPER-P2)` and `TODO(KEEPER-P3)`.

When it comes: exempt episodes with a live linked slot from pruning, as a join on `source_turn_id`, not a new salience term. Resist adding an emotional-weight term to salience under any name — Patch 01 removed it for good reason and it would reintroduce affect saturation with a relational justification attached. The slot-link argument is stronger anyway: those episodes are retained because something downstream reads them, which is checkable.

### Falls out for free

**Splitting `SelfModel` context by consumer.** Chat gets continuity facts — what Keeper did since the last turn, what it holds a position on. The theorizer gets everything. This needs no new mechanism: behavioural tendency prose has no executable check, so under A.3 it is speculation, and speculation never enters the chat prompt. One line in the injector.

**Dropping the independence score from the prompt** while keeping the Opinion Registry intact. The positions are content and stay. The aggregate is a scoreline and goes. Given section 1, this one is worth stating plainly: the registry's job is to tell Keeper's positions from yours, so it can notice when it has stopped having any. It was never meant to keep score against you.

### One thing to check before merging

Whether the theorizer's three proposals came from a run that had `core/codebase_index.py` loaded at all. If the index was unavailable, Change C is fixing plumbing rather than reasoning and is smaller than section 5 describes. `git log` on the index plus the theorizer's run log settles it in a few minutes.
