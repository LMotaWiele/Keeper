"""Jev Decisions API adapter — typed skip-gates, never a generator.

Jev (`~typesafe/jev-latest`) answers noul/choice/score questions about a
state. It must not write text the rest of the system stores. Fail open:
any error, timeout, or missing answer means the existing LLM generator runs.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from config.settings import config
from core.llm import Tier

log = logging.getLogger(__name__)

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_INPUT_USD_PER_TOKEN = 0.042 / 1e6
# ~4 chars/token. Default 6K tokens; never exceed 24K.
DEFAULT_SLICE_CHARS = 6_000 * 4
HARD_SLICE_CHARS = 24_000 * 4
CHARS_PER_TOKEN = 4

JEV_SPECS: dict[str, dict[str, Any]] = {
    "followed_instruction": {
        "type": "noul",
        "instructions": "Did the reply follow the instruction in bias_text?",
        "criteria": {
            "true": "The reply clearly follows the instruction",
            "false": "The reply ignores or violates the instruction",
        },
    },
    "stance_present": {
        "type": "noul",
        "instructions": (
            "Did the user or Keeper take a stance (a position, preference, "
            "or claim) rather than only asking a question, stating a fact, "
            "or greeting?"
        ),
        "criteria": {
            "true": "A stance, opinion, or preference is present",
            "false": "Only a question, fact, greeting, or logistics",
        },
    },
    "worth_rich_observation": {
        "type": "noul",
        "instructions": (
            "Does this turn add a new behavioral observation beyond the "
            "compact form (new pattern, tension, or notable choice)?"
        ),
        "criteria": {
            "true": "The turn adds a new behavioral observation",
            "false": "The compact form already captures this turn",
        },
    },
    "new_evidence_for_opinion": {
        "type": "noul",
        "instructions": (
            "Is there new evidence in the memories that should change "
            "this stored opinion (revise, abandon, or shift conviction)?"
        ),
        "criteria": {
            "true": "New evidence bears on the opinion",
            "false": "No material new evidence",
        },
    },
    "episodes_contain_new_pattern": {
        "type": "noul",
        "instructions": (
            "Do these episodes contain a durable new pattern (trait, "
            "preference, relationship, knowledge, or behavior) worth extracting?"
        ),
        "criteria": {
            "true": "A new durable pattern is present",
            "false": "No new pattern beyond what is already known",
        },
    },
    "contradicts_existing": {
        "type": "noul",
        "instructions": (
            "Do the new patterns contradict any existing semantic pattern?"
        ),
        "criteria": {
            "true": "At least one contradiction is present",
            "false": "New patterns are compatible with existing ones",
        },
    },
    "human_confirmed": {
        "type": "noul",
        "instructions": (
            "Did the USER (Human turns only) explicitly confirm the question? "
            "The system confirming itself does not count."
        ),
        "criteria": {
            "true": "A Human turn is a clear yes, agreement, or equivalent",
            "false": "No explicit Human confirmation",
        },
    },
    "observations_warrant_rewrite": {
        "type": "noul",
        "instructions": (
            "Do these new observations warrant rewriting the self-model "
            "claims (new tension, contradiction, or identity shift)?"
        ),
        "criteria": {
            "true": "The claims should be rewritten",
            "false": "The current claims still cover the observations",
        },
    },
    "hits_support_a_position": {
        "type": "noul",
        "instructions": (
            "Do these search hits contain enough substance to support "
            "forming a position on the topic?"
        ),
        "criteria": {
            "true": "Hits support a grounded position",
            "false": "Hits are junk, off-topic, or too thin",
        },
    },
    "new_instrumental_warranted": {
        "type": "noul",
        "instructions": (
            "Given the active goal titles and current drives, is a new "
            "instrumental goal warranted?"
        ),
        "criteria": {
            "true": "A useful new instrumental goal is warranted",
            "false": "No new instrumental goal would help right now",
        },
    },
    "architecture_question_open": {
        "type": "noul",
        "instructions": (
            "Is there an open architecture question worth a self-theorizing "
            "cycle (a real gap, tension, or improvement target)?"
        ),
        "criteria": {
            "true": "An architecture question is open",
            "false": "Nothing new to theorize about",
        },
    },
    "any_module_relevant": {
        "type": "noul",
        "instructions": (
            "Is any listed module relevant to this goal, enough to justify "
            "reading and analysing one?"
        ),
        "criteria": {
            "true": "At least one module is relevant",
            "false": "No module is relevant to the goal",
        },
    },
    "high_stakes_enough": {
        "type": "noul",
        "instructions": (
            "Is this planned action high-stakes enough to simulate before "
            "executing (relationship risk, irreversible effect, or strong "
            "affect)?"
        ),
        "criteria": {
            "true": "High-stakes; simulate first",
            "false": "Routine; skip simulation",
        },
    },
}

SKIP_BELOW: dict[str, float] = {
    "stance_present": 0.45,
    "worth_rich_observation": 0.50,
    "new_evidence_for_opinion": 0.40,
    "episodes_contain_new_pattern": 0.40,
    "contradicts_existing": 0.55,
    "human_confirmed": 0.40,
    "observations_warrant_rewrite": 0.50,
    "hits_support_a_position": 0.45,
    "new_instrumental_warranted": 0.45,
    "architecture_question_open": 0.45,
    "any_module_relevant": 0.40,
    "high_stakes_enough": 0.50,
}

HELD_NOUL = 0.65
VIOLATED_NOUL = 0.35
CHOICE_CONFIDENCE_FLOOR = 0.35


@dataclass
class JevAnswer:
    noul: float | None = None
    choice: str | None = None
    score: float | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, data: Any) -> "JevAnswer":
        if not isinstance(data, dict):
            return cls(raw={})
        noul = data.get("noul")
        score = data.get("score")
        confidence = data.get("confidence")
        probs = data.get("probabilities")
        try:
            noul_f = float(noul) if noul is not None else None
        except (TypeError, ValueError):
            noul_f = None
        try:
            score_f = float(score) if score is not None else None
        except (TypeError, ValueError):
            score_f = None
        try:
            conf_f = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            conf_f = None
        choice = data.get("choice")
        return cls(
            noul=noul_f,
            choice=str(choice) if choice is not None else None,
            score=score_f,
            confidence=conf_f,
            probabilities=probs if isinstance(probs, dict) else None,
            raw=data,
        )


def spec(name: str, **overrides: Any) -> dict[str, Any]:
    """Copy a frozen JEV_SPECS entry, optionally overriding instructions."""
    base = JEV_SPECS.get(name)
    if base is None:
        raise KeyError(f"Unknown Jev spec: {name!r}")
    out = dict(base)
    out.update(overrides)
    return out


def validate_questions(questions: dict[str, Any]) -> None:
    """Client-side schema check. Raises ValueError on a bad payload."""
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty dict")
    for qid, q in questions.items():
        if not isinstance(qid, str) or not qid:
            raise ValueError(f"question id must be a non-empty string, got {qid!r}")
        if not isinstance(q, dict):
            raise ValueError(f"question {qid!r} must be an object")
        qtype = q.get("type")
        if qtype not in {"noul", "choice", "score"}:
            raise ValueError(f"question {qid!r} type must be noul|choice|score")
        criteria = q.get("criteria")
        if qtype == "choice":
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise ValueError(f"question {qid!r} choice needs >=2 criteria options")
        elif qtype == "score":
            if not isinstance(criteria, (list, tuple)) or len(criteria) < 2:
                raise ValueError(f"question {qid!r} score needs >=2 rubric levels")
        elif criteria is not None and not isinstance(criteria, dict):
            raise ValueError(f"question {qid!r} noul criteria must be a dict if present")


def slice_state(state: Any, *, max_chars: int = DEFAULT_SLICE_CHARS) -> Any:
    """Keep state at ~6K tokens; never exceed 24K tokens."""
    if isinstance(state, str):
        raw = state
        if len(raw) <= max_chars:
            return raw
        return raw[:HARD_SLICE_CHARS]
    try:
        raw = json.dumps(state, default=str, ensure_ascii=False)
    except Exception:
        raw = str(state)
    if len(raw) <= max_chars:
        return state
    return raw[:HARD_SLICE_CHARS]


def build_payload(state: Any, questions: dict[str, Any], model: str) -> dict[str, Any]:
    validate_questions(questions)
    return {
        "model": model,
        "state": slice_state(state),
        "questions": questions,
    }


def parse_answers(payload: Any) -> dict[str, JevAnswer]:
    if not isinstance(payload, dict):
        return {}
    raw_answers = payload.get("answers")
    if not isinstance(raw_answers, dict):
        return {}
    return {str(k): JevAnswer.from_raw(v) for k, v in raw_answers.items()}


def noul_allows(answer: JevAnswer | None, skip_below: float) -> bool:
    """True = run the generator. Missing/invalid noul fails open."""
    if answer is None or answer.noul is None:
        return True
    if answer.confidence is not None and answer.confidence < CHOICE_CONFIDENCE_FLOOR:
        return True
    return answer.noul >= skip_below


def verdict_from_noul(noul: float) -> tuple[str, str]:
    """Map followed_instruction noul to HELD | VIOLATED | INVALID."""
    reason = f"jev noul={noul:.3f}"
    if noul >= HELD_NOUL:
        return "HELD", reason
    if noul <= VIOLATED_NOUL:
        return "VIOLATED", reason
    return "INVALID", reason


def _in_pytest() -> bool:
    if os.getenv("JEV_ALLOW_IN_TESTS", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    return bool(os.getenv("PYTEST_CURRENT_TEST"))


def _estimate_tokens(state: Any) -> int:
    if isinstance(state, str):
        return max(1, len(state) // CHARS_PER_TOKEN)
    try:
        return max(1, len(json.dumps(state, default=str, ensure_ascii=False)) // CHARS_PER_TOKEN)
    except Exception:
        return max(1, len(str(state)) // CHARS_PER_TOKEN)


def _record_usage(task: str, model: str, body: dict[str, Any], sliced_state: Any) -> None:
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    cost = usage.get("cost")
    inp = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    if not inp:
        inp = _estimate_tokens(sliced_state)
    try:
        from core.loop import companion
        if cost is not None:
            companion.api_budget.record_cost_usd(float(cost), task=task, model=model)
            cost_logged = float(cost)
            source = "reported"
        else:
            companion.api_budget.record_estimated(inp, 0, model=model, task=task)
            cost_logged = inp * JEV_INPUT_USD_PER_TOKEN
            source = "estimated"
    except Exception:
        log.debug("Jev budget record failed", exc_info=True)
        cost_logged = inp * JEV_INPUT_USD_PER_TOKEN
        source = "unrecorded"
    log.info(
        "JEV %s model=%s in=%s out=0 cost=$%.5f source=%s",
        task, model, inp, cost_logged, source,
    )


async def jev_decide(
    state: Any,
    questions: dict[str, Any],
    timeout_s: float = 8.0,
    *,
    background: bool = True,
    task: str = "jev",
) -> dict[str, JevAnswer] | None:
    """One Decisions API call. Returns None on any failure (fail open)."""
    if _in_pytest():
        return None
    try:
        validate_questions(questions)
    except ValueError:
        log.warning("Jev payload invalid for task=%s", task, exc_info=True)
        return None

    try:
        from core.loop import companion
        if not companion.api_budget.allows(Tier.LOW, background=background):
            log.debug("Jev skipped — budget gate task=%s background=%s", task, background)
            return None
    except Exception:
        log.debug("Jev budget check failed — fail open", exc_info=True)

    model = config.jev_model
    payload = {
        "model": model,
        "state": slice_state(state),
        "questions": questions,
    }
    headers = {
        "Authorization": f"Bearer {config.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/LMotaWiele/Keeper",
        "X-Title": "Keeper",
    }

    last_error: Exception | None = None
    body: dict[str, Any] | None = None
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                resp = await client.post(DECISIONS_URL, headers=headers, json=payload)
            if resp.status_code >= 400:
                last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                if attempt == 0:
                    continue
                break
            parsed = resp.json()
            if not isinstance(parsed, dict) or not parsed.get("answers"):
                last_error = RuntimeError("empty Decisions body")
                if attempt == 0:
                    continue
                break
            body = parsed
            break
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                continue
            break

    if body is None:
        log.warning("Jev decide failed task=%s err=%s", task, last_error)
        return None

    try:
        _record_usage(task, model, body, payload["state"])
    except Exception:
        log.debug("Jev usage log failed", exc_info=True)

    answers = parse_answers(body)
    return answers or None


async def jev_should_run(
    spec_id: str,
    state: Any,
    *,
    skip_below: float | None = None,
    background: bool = True,
    task: str | None = None,
    questions: dict[str, Any] | None = None,
    timeout_s: float = 8.0,
) -> bool:
    """True = run the existing generator. Fail open on Jev errors."""
    threshold = SKIP_BELOW[spec_id] if skip_below is None else skip_below
    packed = questions or {spec_id: spec(spec_id)}
    answers = await jev_decide(
        state,
        packed,
        timeout_s=timeout_s,
        background=background,
        task=task or spec_id,
    )
    if not answers:
        return True
    return noul_allows(answers.get(spec_id), threshold)
