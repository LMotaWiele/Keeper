"""Change D — action_bias injection with an external, low-tier verdict."""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import uuid
from typing import Any

from langchain_core.messages import HumanMessage

from config.settings import config
from core.timeutil import utcnow

log = logging.getLogger(__name__)

_pending_eval_tasks: set[asyncio.Task] = set()

# Trials against the user-life-coaching hypothesis ran while no user_life
# tool was bound. VOID is permanent; never overwrite, never count in streaks.
USER_LIFE_HYPOTHESIS_IDS = (
    "h-v51-02",  # lecture vs log commitment (exact)
    "h-v56-01",  # user life-tracking; trial 11
    "h-v57-01",  # invoking structured state tracking tools
    "h-v58-01",  # current: tracking/self-improvement without tool invocations
)
VOID_CAPABILITY_REASON = "capability absent: no user_life tool bound at trial time"
# Trials after this instant can be real tests — the tools exist.
VOID_BEFORE_TS = "2026-09-10T13:00:00+00:00"
_SKIP_STREAK = {"INVALID", "VOID", None}

TRIAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS action_bias_trial (
    trial_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    turn_id       TEXT NOT NULL,
    hypothesis_id TEXT NOT NULL,
    bias_text     TEXT NOT NULL,
    verdict       TEXT,
    reason        TEXT,
    evaluated_at  TEXT
);
"""

BIAS_TEXT_PROMPT = """\
Rewrite this behavioral hypothesis as ONE imperative sentence that can be
checked from a single reply with no extra context, no self-model, and no
codebase access.

Hypothesis: {statement}
Suggested bias: {action_bias}

Rules:
- Imperative mood, one sentence.
- Names a concrete observable behaviour, not a disposition.
- Checkable from that reply alone.

Bad: "Be less abstract."
Good: "When asked to improve yourself, name a specific file and the change to make in it; do not describe how improvement could be verified."

Output ONLY the sentence.
"""

EVAL_PROMPT = """\
Instruction: "{bias_text}"
Reply: "{response_text}"

Did the reply follow the instruction? Answer with exactly one word, HELD or VIOLATED,
then a colon and one short clause of reason.
"""


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(config.midterm_db_path), timeout=10)
    conn.executescript(TRIAL_SCHEMA)
    conn.commit()
    return conn


def ensure_hypothesis_fields(hypotheses: list[dict], version: int = 0) -> list[dict]:
    """Fill id / tested / actionable / bias_text / trial_log on existing hyps."""
    out = []
    for i, raw in enumerate(hypotheses or []):
        h = dict(raw)
        h.setdefault("id", f"h-v{version}-{i + 1:02d}")
        h.setdefault("tested", False)
        h.setdefault("actionable", None)
        h.setdefault("bias_text", None)
        h.setdefault("trial_log", [])
        h.setdefault("capability_verified", False)
        h.setdefault("capability_note", None)
        h.setdefault("needs_capability_review", False)
        out.append(h)
    return out


def eligible_hypotheses(hypotheses: list[dict]) -> list[dict]:
    """Unverified character hypotheses are speculation and are not injected."""
    from goals.proposals import has_executable_check
    return [
        h for h in hypotheses
        if not h.get("tested")
        and h.get("actionable") is not False
        and not h.get("needs_capability_review")
        and has_executable_check(h)
    ]


def select_sticky(model: dict) -> dict | None:
    """Stay on the active hypothesis until it resolves."""
    hyps = ensure_hypothesis_fields(
        model.get("hypotheses") or [],
        model.get("model_version") or 0,
    )
    model["hypotheses"] = hyps
    hid = model.get("active_bias_hypothesis_id")
    if hid:
        current = next((h for h in hyps if h.get("id") == hid), None)
        if current and current in eligible_hypotheses([current]):
            return current
    eligible = eligible_hypotheses(hyps)
    if not eligible:
        model["active_bias_hypothesis_id"] = None
        return None
    chosen = max(eligible, key=lambda h: float(h.get("confidence") or 0))
    model["active_bias_hypothesis_id"] = chosen["id"]
    return chosen


def _usable_bias_text(text: str) -> bool:
    """Reject prompt-echo fragments. Do not loosen the verdict parser."""
    cleaned = (text or "").strip()
    if len(cleaned) < 40:
        return False
    prompt_l = BIAS_TEXT_PROMPT.lower()
    if cleaned.lower() in prompt_l:
        return False
    # First-line debris like "ONLY the sentence." / "reply alone (no extra context"
    for line in BIAS_TEXT_PROMPT.splitlines():
        line = line.strip()
        if len(line) >= 12 and line.lower() in cleaned.lower() and len(cleaned) < 80:
            return False
    return True


def _extract_bias_sentence(raw: str) -> str:
    lines = [ln.strip().strip('"').strip("'") for ln in (raw or "").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""
    usable = [ln for ln in lines if _usable_bias_text(ln)]
    if usable:
        return max(usable, key=len)
    return max(lines, key=len)


async def ensure_bias_text(hypothesis: dict) -> str:
    """Generate bias_text once; never overwrite a hand-edited value."""
    existing = hypothesis.get("bias_text")
    if existing and _usable_bias_text(str(existing)):
        return str(existing)
    from core.llm import get_llm
    prompt = BIAS_TEXT_PROMPT.format(
        statement=hypothesis.get("statement") or "",
        action_bias=hypothesis.get("action_bias") or "",
    )
    result = await get_llm("action_bias_text").ainvoke([HumanMessage(content=prompt)])
    text = _extract_bias_sentence(getattr(result, "content", None) or "")
    if not _usable_bias_text(text):
        fallback = str(hypothesis.get("action_bias") or hypothesis.get("statement") or "")
        text = fallback if _usable_bias_text(fallback) else fallback
    hypothesis["bias_text"] = text
    return text


def standing_block(model: dict) -> str:
    """Chat injection keeps only constraints tied to an executable check."""
    from goals.proposals import has_executable_check
    items = model.get("standing_constraints") or []
    hyps = {h.get("id"): h for h in (model.get("hypotheses") or []) if isinstance(h, dict)}
    lines = ["Standing behavioural constraints:"]
    kept = 0
    for item in items:
        if kept >= config.STANDING_CONSTRAINTS_MAX:
            break
        if not isinstance(item, dict):
            continue
        hyp = hyps.get(item.get("hypothesis_id"))
        if hyp is None or not has_executable_check(hyp):
            continue
        text = item.get("text")
        if text:
            lines.append(f"- {text}")
            kept += 1
    if kept == 0:
        return ""
    return "\n".join(lines)


def prompt_tail(model: dict, bias_text: str | None) -> str:
    parts = []
    standing = standing_block(model)
    if standing:
        parts.append(standing)
    if bias_text:
        parts.append(f"Behavioural constraint for this turn: {bias_text}")
    return "\n".join(parts)


def insert_trial(turn_id: str, hypothesis_id: str, bias_text: str) -> int:
    """Insert an unevaluated trial *before* generation. Returns trial_id."""
    conn = _db()
    try:
        cur = conn.execute(
            """INSERT INTO action_bias_trial
               (ts, turn_id, hypothesis_id, bias_text, verdict, reason, evaluated_at)
               VALUES (?, ?, ?, ?, NULL, NULL, NULL)""",
            (utcnow().isoformat(), turn_id, hypothesis_id, bias_text),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def _parse_verdict(raw: str) -> tuple[str, str]:
    text = (raw or "").strip()
    first = text.split(":", 1)
    token = (first[0].split() or [""])[0].upper()
    reason = first[1].strip() if len(first) > 1 else text
    if token in {"HELD", "VIOLATED"}:
        return token, reason[:200]
    return "INVALID", text[:200]


def _consecutive_verdicts(trial_log: list[dict], wanted: str) -> int:
    n = 0
    for entry in reversed(trial_log):
        v = entry.get("verdict")
        if v in _SKIP_STREAK:
            continue
        if v == wanted:
            n += 1
        else:
            break
    return n


def apply_verdict(model: dict, hypothesis_id: str, trial_id: int, verdict: str) -> None:
    hyps = model.get("hypotheses") or []
    hyp = next((h for h in hyps if h.get("id") == hypothesis_id), None)
    if hyp is None:
        return
    log_entry = {
        "trial_id": trial_id,
        "verdict": verdict,
        "ts": utcnow().isoformat(),
    }
    hyp.setdefault("trial_log", []).append(log_entry)
    if verdict in _SKIP_STREAK:
        return
    held = _consecutive_verdicts(hyp["trial_log"], "HELD")
    violated = _consecutive_verdicts(hyp["trial_log"], "VIOLATED")
    need = config.ACTION_BIAS_STREAK_LEN
    if held >= need:
        hyp["tested"] = True
        hyp["actionable"] = True
        bias = hyp.get("bias_text")
        standing = list(model.get("standing_constraints") or [])
        texts = {
            (s if isinstance(s, str) else s.get("text"))
            for s in standing
        }
        if bias and bias not in texts:
            standing.append({
                "text": bias,
                "hypothesis_id": hypothesis_id,
                "confidence": float(hyp.get("confidence") or 0),
            })
            if len(standing) > config.STANDING_CONSTRAINTS_MAX:
                standing.sort(
                    key=lambda s: float(
                        s.get("confidence") if isinstance(s, dict) else 0
                    )
                )
                standing = standing[-config.STANDING_CONSTRAINTS_MAX:]
            model["standing_constraints"] = standing
        if model.get("active_bias_hypothesis_id") == hypothesis_id:
            model["active_bias_hypothesis_id"] = None
        log.info("action_bias HELD streak — hypothesis %s actionable=true", hypothesis_id)
    elif violated >= need:
        if not hyp.get("capability_verified"):
            hyp["needs_capability_review"] = True
            if model.get("active_bias_hypothesis_id") == hypothesis_id:
                model["active_bias_hypothesis_id"] = None
            log.info(
                "action_bias VIOLATED streak — hypothesis %s needs_capability_review "
                "(capability_verified is false; not writing actionable=false)",
                hypothesis_id,
            )
            return
        hyp["tested"] = True
        hyp["actionable"] = False
        if model.get("active_bias_hypothesis_id") == hypothesis_id:
            model["active_bias_hypothesis_id"] = None
        log.info("action_bias VIOLATED streak — hypothesis %s actionable=false", hypothesis_id)


def write_trial_verdict(trial_id: int, verdict: str, reason: str) -> None:
    conn = _db()
    try:
        row = conn.execute(
            "SELECT verdict FROM action_bias_trial WHERE trial_id = ?",
            (trial_id,),
        ).fetchone()
        if row and row[0] == "VOID":
            return
        conn.execute(
            """UPDATE action_bias_trial
               SET verdict = ?, reason = ?, evaluated_at = ?
               WHERE trial_id = ? AND (verdict IS NULL OR verdict != 'VOID')""",
            (verdict, reason, utcnow().isoformat(), trial_id),
        )
        conn.commit()
    finally:
        conn.close()


def void_capability_absent_trials(
    hypothesis_ids: tuple[str, ...] = USER_LIFE_HYPOTHESIS_IDS,
    before_ts: str = VOID_BEFORE_TS,
) -> int:
    """Mark trials that tested a capability that did not exist. Idempotent."""
    if not hypothesis_ids:
        return 0
    conn = _db()
    try:
        ph = ",".join("?" for _ in hypothesis_ids)
        cur = conn.execute(
            f"""UPDATE action_bias_trial
                SET verdict = 'VOID', reason = ?
                WHERE hypothesis_id IN ({ph})
                  AND ts < ?
                  AND (verdict IS NULL OR verdict != 'VOID')""",
            (VOID_CAPABILITY_REASON, *hypothesis_ids, before_ts),
        )
        conn.commit()
        n = int(cur.rowcount or 0)
        if n:
            log.info("voided %d action_bias trials for capability-absent hypotheses", n)
        return n
    finally:
        conn.close()


def reply_token_count(text: str) -> int:
    return len((text or "").split())


def trial_eligible_after_reply(
    *,
    response_text: str,
    tool_calls_only: bool,
    error: bool,
) -> bool:
    if error or tool_calls_only:
        return False
    return reply_token_count(response_text) >= config.ACTION_BIAS_MIN_REPLY_TOKENS


async def _evaluate(trial_id: int, bias_text: str, response_text: str, hypothesis_id: str) -> None:
    from core.jev import JEV_SPECS, jev_decide, verdict_from_noul
    from core.llm import get_llm
    from core.loop import companion

    verdict, reason = "INVALID", "eval failed"
    try:
        answers = await jev_decide(
            {"bias_text": bias_text, "response_text": response_text[:4000]},
            {"followed_instruction": dict(JEV_SPECS["followed_instruction"])},
            background=False,
            task="action_bias_eval",
        )
        noul = None
        if answers and answers.get("followed_instruction") is not None:
            noul = answers["followed_instruction"].noul
        if noul is not None:
            verdict, reason = verdict_from_noul(noul)
        else:
            llm = get_llm(
                "action_bias_eval",
                model=config.ACTION_BIAS_EVAL_MODEL,
                temperature=0,
                max_tokens=60,
            )
            result = await llm.ainvoke([
                HumanMessage(content=EVAL_PROMPT.format(
                    bias_text=bias_text,
                    response_text=response_text[:4000],
                )),
            ])
            raw = getattr(result, "content", "") or ""
            verdict, reason = _parse_verdict(raw)
    except Exception:
        log.warning("action_bias evaluator failed", exc_info=True)
        verdict, reason = "INVALID", "exception"
    try:
        write_trial_verdict(trial_id, verdict, reason)
        apply_verdict(companion.self_model.model, hypothesis_id, trial_id, verdict)
        from pathlib import Path
        companion.self_model.save(Path(config.data_dir) / "state" / "self_model.json")
    except Exception:
        log.warning("action_bias verdict persist failed", exc_info=True)


def schedule_evaluation(
    trial_id: int,
    bias_text: str,
    response_text: str,
    hypothesis_id: str,
) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        log.warning("action_bias eval skipped — no running loop")
        return

    task = loop.create_task(
        _evaluate(trial_id, bias_text, response_text, hypothesis_id)
    )
    _pending_eval_tasks.add(task)
    task.add_done_callback(_pending_eval_tasks.discard)
