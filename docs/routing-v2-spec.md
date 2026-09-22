# Keeper routing v2 (non-conflicting)

**Status:** accepted — implemented (PRs 1–4 plus `should_simulate` / `any_module_relevant`).

This is the original routing spec with every change that would rewrite a live
consumer dropped. What remains is: pin the suggested models, a few factory
knobs that do not change task I/O, and Jev used as a **cheap skip-gate** in
front of existing generators.

Daily ceiling stays €10. Budget gates stay as they are (MID background off at
60%, all background LLM off at 85%, conversation degrades 85% / 95% and never
hard-stops).

---

## 1. What this does not do

These were in the source spec and are **out of scope** because they change
current outputs or control flow:

| Dropped | Why |
|---|---|
| Jev *replaces* `opinion_detection` / `opinion_review` | Those tasks write `domain`, `position`, `origin`, `conviction`. Anti-sycophancy needs that prose. |
| Jev *replaces* `action_rating` / Elo / `select_action_type` | Idle type pick is already code + Elo. The LLM writes pairwise `why`. |
| Jev *replaces* `completion_check` | Consumer requires a verbatim Human `quote`. |
| Jev *replaces* `contradiction_check` | Consumer needs `keep_new \| keep_old \| merge`. |
| Redefining `action_bias_eval` as an action-permission gate | Live job is post-reply `HELD \| VIOLATED` on a hypothesis trial. |
| Two-phase MID `consistency_check` | `get_llm("consistency_check")` is unused; layer 2 is a word-overlap heuristic. |
| `autonomous_pursuit_select` | Selection is not an LLM. |
| Strict `json_schema` + new MID shapes | Every parser uses `json_object` / `parse_json_lenient` and today’s keys. |
| Strip tools at 95% LOW conversation | Graph binds `ALL_TOOLS` at every conversation tier. |
| New caps (2 MID/hour, 1 self-model/6h) | Extra gates, not in the current machine. |
| OpenRouter `allow_fallbacks: true` | Factory currently pins the backend for reproducibility. |
| “Budget MID” → DeepSeek at 60% spend | MID background is already refused at 60%. |

---

## 2. Model map (PR 1)

Pin HIGH and MID. Alias allowed on LOW.

| Route | Slug | OpenRouter | List price / 1M | Role |
|---|---|---|---|---|
| HIGH | `google/gemini-3.8-flash` | Chat Completions | $0.75 / $3.75 | Conversation + tools |
| MID | `google/gemini-3.8-flash` | Chat Completions | $0.75 / $3.75 | Structured reasoning / JSON write-ups |
| LOW | `z-ai/glm-5.3-flash` | Chat Completions | $0.075 / $0.25 | Compression, query drafting, open-set extract |

`x-ai/grok-4.6` leaves the default map. It is not used by the running unit
today (`MODEL_HIGH` is already Gemini 3.7 in systemd).

Fallback slugs, used **only** by our code on 5xx / empty body after one retry,
not via OpenRouter provider fallbacks:

| Route | Fallback |
|---|---|
| HIGH | `google/gemini-3.7-flash` |
| MID | `google/gemini-3.7-flash` (same lab, tool schema matches) |
| LOW | `~deepseek/deepseek-flash-latest` |

Do not use `:free` endpoints for anything that writes the self-model, opinions,
or semantic memory.

Files: `config/settings.py`, `.env`, `.env.example`, `README.md`,
`core/resource_budgets.py` pricing table, user unit
`~/.config/systemd/user/companion.service` (`Environment=MODEL_HIGH=...`).

---

## 3. Non-conflicting factory changes (PR 1)

Keep `get_llm(task)` as the only chat entry point. Keep `TASK_TIERS`. Keep
`json_mode` → `response_format: json_object`. Keep `usage.include`. Keep
`provider.allow_fallbacks: False`.

Add:

1. **Reasoning off on every LOW call that accepts it.** Thinking tokens bill as
   output. GLM 5.3 Flash rejects disable (`enabled: false` / `effort: none` →
   400 "Reasoning is mandatory"). LOW GLM therefore uses `effort: "low"`; the
   DeepSeek fallback still sends `enabled: false, effort: none`. HIGH and MID
   stay on the provider default.
2. **Provider sort, still no auto-fallback.**
   - HIGH: `provider.sort = "exacto"` (tool-call accuracy).
   - MID / LOW: `provider.sort = "price"`.
3. **`action_bias_text` → LOW.** It is phrasing (one imperative sentence), not
   analysis. Params stay `temperature=0.3`, `max_tokens=128`.
4. **`force_answer` uses the turn’s degraded model.** Today it calls unbound
   `get_llm("conversation")` and jumps back to HIGH mid-turn. Pass the graph
   `tier` so one model owns the whole turn.

Do **not** cut MID `max_tokens` from 4096 → 1500 in this PR. That is a quality
risk on `self_model_analysis` / `hypothesis_generation`, not a routing fix.

Conversation degrade table (unchanged):

| Daily spend | Conversation route | Tools |
|---|---|---|
| < 85% | HIGH | full set |
| 85–95% | MID | full set (same Gemini tool schema) |
| ≥ 95% | LOW | full set; prompt already says replies will be shorter |

---

## 4. Jev: gates, not generators

Jev (`~typesafe/jev-latest`, OpenRouter Decisions API, $0.042/M in, **$0 out**)
is a fourth **transport**, not a fourth budget tier. For `APIBudget.allows()` it
counts as **LOW**. Conversation-triggered Jev uses `background=False` (exempt,
same as today’s post-turn classifiers). Idle-loop Jev uses `background=True`
and dies with the rest of background LLM at 85%.

**Rule:** Jev never writes text the rest of the system stores. If the consumer
needs a paragraph, JSON blob, quote, origin tag, or `why`, the existing LLM
task still runs. Jev only answers yes/no, a closed label, or a rubric score
that **code** already knows how to apply.

**Rule:** no Jev on the path *before* the first HIGH token. An 8s Decisions
timeout must not sit in front of Telegram. Post-turn and background only.

**Rule:** one HTTP call per event. Pack every question for that event into
`jev_decide`. Frozen question dicts in `JEV_SPECS[task]`. LOW JSON fallback
on 4xx/5xx/timeout after one retry, using the **current** generator (fail
open to today’s behavior, not fail closed to silence).

Adapter (new `core/jev.py`), as in the source spec: `JevAnswer` +
`jev_decide(state, questions, timeout_s=8)`, client-side validation, state
slicer (default 6K tokens, hard 24K), log input tokens at $0.042/M.

---

## 5. Where Jev pays for itself

Cost of one Jev call with ~2K input: **~$0.00008**. Cost of a typical LOW
GLM extract: **~$0.0002–0.0005**. Cost of a MID Gemini rewrite: **~$0.01–0.02**.
Gating MID is the money. Gating high-frequency LOW is the volume. Replacing a
60-token classifier is cleanliness, not savings.

### 5.1 Replace (same job, binary consumer)

Only one live LLM task ignores generated prose except a log string:

#### `action_bias_eval` (post-turn, `background=False`)

Today: 0-temp LOW, 60 tokens, parse first word `HELD|VIOLATED`.

Jev noul `followed_instruction` on state `{bias_text, response_text[:4k]}`.

| noul | verdict |
|---|---|
| ≥ 0.65 | `HELD` |
| ≤ 0.35 | `VIOLATED` |
| in between | `INVALID` (already a legal verdict; no streak update) |

`reason` becomes `jev noul=…` so the trial row still has a string. LOW fallback
keeps the current prompt if Decisions API fails.

This is the only **replacement**. It does not change streaks, standing
constraints, or VOID handling.

### 5.2 Gate (skip a generator that still owns the write)

Each row: Jev first; on miss, **do not call** the LLM; on hit, call it
unchanged.

| Event | Jev question | Skip when | Saves |
|---|---|---|---|
| After a non-degraded turn | noul `stance_present`: did user or Keeper take a stance (not a question, fact, or greeting)? | noul < 0.45 | `opinion_detection` (LOW, every turn, up to 1024 out) |
| After a turn, 5th-obs slot | noul `worth_rich_observation`: does this turn add a new behavioral observation vs the compact form? | noul < 0.50 | `observation_summary` (LOW). Also **spend** the summary on a 3rd or 7th turn when Jev says yes — drop the rigid `% 5` once the fixture says the noul is calibrated. Until then: Jev may only *skip* the 5th, not add extra calls. |
| Periodic opinion review | noul `new_evidence_for_opinion` per stored opinion, packed in one request | noul < 0.40 | `opinion_review` (LOW) on stale opinions |
| Consolidation extract | noul `episodes_contain_new_pattern` | noul < 0.40 | `pattern_extraction` (LOW). Decay/forget still run (already true on skip). |
| After extract, if any new patterns | noul `contradicts_existing` | noul < 0.55 | `contradiction_check` (LOW). On hit, existing LLM still emits `keep_new\|keep_old\|merge`. |
| Bounded `user_confirmed` completion | noul `human_confirmed` | noul < 0.40 | `completion_check` LLM. On hit, LLM still must return a Human `quote`; quote check stays. Never mark complete from Jev alone. |
| Self-model loop, ≥5 new obs | noul `observations_warrant_rewrite` vs a compact claim list | noul < 0.50 | `self_model_analysis` **and** the follow-on `hypothesis_generation` (MID). Biggest dollar skip. |
| Before `research_synthesis` | noul `hits_support_a_position` | noul < 0.45 or empty hits | MID/LOW synthesis on junk SERPs. Empty-hit short-circuit can stay code-only. |
| Before `goal_generation` | noul `new_instrumental_warranted` given active titles + drives | noul < 0.45 | MID-priced `goal_generation` that often returns nothing useful. |
| Before `self_theorize` | noul `architecture_question_open` | noul < 0.45 | A 4096-token MID cycle. 6h throttle stays. |
| Autonomous `codebase_read` pick | noul `any_module_relevant` | noul < 0.40 | The two `autonomous_pursuit` LOW calls. Type selection stays Elo. |

Pack post-turn questions in **one** `jev_decide`:
`stance_present`, `worth_rich_observation` (only if this is a summary slot),
`followed_instruction` (only if a bias trial is eligible).

### 5.3 Smarter existing heuristics (optional, PR 3)

These already skip work in code. Jev can replace a brittle threshold **without
removing the skip**. Fail open to the current heuristic if Jev is down.

| Heuristic today | Jev upgrade | Why |
|---|---|---|
| `FutureSimulator.should_simulate` — keyword `[autonomous]` + valence/arousal vs 0.6 | noul `high_stakes_enough` on `{action, valence, arousal, autonomous: bool}` | Catches high-stakes user turns the keyword misses; still skips most chats. |
| Architecture prompt inject — substring list (`how do you work`, …) | **Not on the pre-HIGH path.** Leave keywords. A post-hoc noul is useless for this turn’s prompt. | Latency. |

Do **not** Jev-score episodic salience, Elo, or drive updates in this round.
Those are control loops with persisted numbers; a new scorer is a behavior
change.

### 5.4 Explicitly not Jev

| Task | Reason |
|---|---|
| `conversation` | Generation + tools. |
| `action_bias_text` | Must write the sentence. Stays LOW. |
| `query_generation` / `topic_extraction` | Open-set strings. |
| `self_model_analysis` body | Must write identity JSON. Gate only. |
| `contrarian_review` | Rare (stagnation ≥ 3) and must write attacks. Not worth a gate. |
| `future_simulate` body | Must write outcomes. Gate = `should_simulate` only. |
| Idle action **type** | Elo in `select_action_type`. |

---

## 6. Suggested thresholds (starting points)

Lock after a fixture run. Until then, **bias toward running the generator**
(false skip is the failure mode).

| Question | Skip / negative | Run generator | Abstain (= run generator) |
|---|---|---|---|
| `followed_instruction` | ≤ 0.35 → VIOLATED | ≥ 0.65 → HELD | 0.35–0.65 → INVALID |
| `stance_present` | < 0.45 skip | ≥ 0.45 run | — |
| `observations_warrant_rewrite` | < 0.50 skip | ≥ 0.50 run | — |
| `contradicts_existing` | < 0.55 skip | ≥ 0.55 run | — |
| `human_confirmed` | < 0.40 skip | ≥ 0.40 run + quote | — |

If Jev confidence (choice) is ever used, confidence < 0.35 → treat as skip-fail-open
(run the LLM).

---

## 7. Budget sketch (same day-mix, after gates)

Assume the source spec’s chatty day, with 60% of opinion extracts skipped,
50% of empty MID rewrites skipped, reasoning off on LOW:

| Slice | Before (today-ish) | After |
|---|---|---|
| 30× HIGH conversation | ~$0.13 | ~$0.13 (3.8 at same list price; watch thinking tokens) |
| 30× opinion_detection | LOW every turn | ~12 LOW + 30 Jev ≈ **much cheaper** |
| 1× self-model + hyps | ~$0.02–0.03 | 0 or 1, plus one Jev noul |
| LOW summaries / patterns / ratings | small | smaller if gates fire |
| Jev overhead | 0 | tens of calls × $0.00008 ≈ **< $0.01** |
| **Typical day** | well under €10 | still **≪ $1**, HIGH still dominates |

What can still blow the cap: HIGH thinking + long tool traces. That is
unchanged. Do not “spend the headroom” by promoting LOW tasks onto Gemini.

Gemini 3.8 default thinking is **medium**. If HIGH/MID output tokens rise vs
3.7 at the same $3.75/M, spend rises. Log `reasoning_tokens` for a week after
PR 1; if HIGH cost/turn is >1.3× the 3.7 baseline, set HIGH reasoning
effort `low` (not off).

---

## 8. Reliability (minimum before each Jev cutover)

Contract tests (no network): payload validation, MID options not involved
(Elo still owns that), skip/run branching, `force_answer` tier, LOW reasoning
flag, fallback slug on simulated 503.

Live evals behind `RUN_MODEL_EVAL=1`, replay after first pin:

- `action_bias_eval`: 40 labeled HELD/VIOLATED turns. Jev vs current LOW
  within 3 points; INVALID rate not up >10 points.
- `stance_present` gate: 80 turns. False-skip (Jev skip, human says opinion
  present) ≤ 10%. False-run is acceptable.
- `observations_warrant_rewrite`: 15 packs. Must not skip a pack that contains
  a planted contradiction or a new tension.
- Fallback: force 503 → current LLM still runs, budget logs a LOW/MID call.

No 50-turn HIGH tool bake-off is required to *ship* 3.8 at the same price as
3.7 (OpenRouter lists it as a strict upgrade at the intro rate). Still log
tool-call parse errors for a week; revert HIGH slug to 3.7 if valid tool-call
rate drops.

---

## 9. Cutover

| PR | What | Behavior change? |
|---|---|---|
| **1** | Slugs + pricing + LOW reasoning off + provider sort + `action_bias_text` LOW + `force_answer` tier + systemd `MODEL_HIGH` | Models and factory only. Restart `companion.service`. |
| **2** | `core/jev.py` + contract tests + budget logging as LOW | None. Dead code until wired. |
| **3** | Wire `action_bias_eval` replacement + post-turn `stance_present` / `worth_rich_observation` skip-gates | Skip some LOW calls; HELD/VIOLATED still the verdict. |
| **4** | Background gates: opinion review, consolidation extract/contradict, completion quote, self-model rewrite, synthesis, goal gen, self_theorize | Skip some LOW/MID calls. Fail open. |
| **5** | Optional: Jev `should_simulate` | Only if PR 3–4 look calibrated. |

Do not ship PR 3 and PR 1’s HIGH slug on the same day if we also want a clean
attribution window. PR 1 can land first; Jev later.

---

## 10. Key decisions

1. **Keep every generator’s I/O.** Jev is a skip-gate (and one binary
   replacement whose parser already discards prose).
2. **Jev counts as LOW for the existing gate**, background vs conversation
   unchanged.
3. **No Jev in front of HIGH.** Latency.
4. **Fail open** to the current LLM on Jev errors so a Decisions outage does
   not silence opinions, completion, or self-model.
5. **Pin Gemini 3.8** on HIGH/MID at the same intro price as 3.7; GLM 5.3 Flash
   on LOW; DeepSeek Flash Latest only as LOW 5xx fallback.
6. **Do not turn on OpenRouter auto-fallbacks.** Explicit one-retry to a
   named slug instead.
)
