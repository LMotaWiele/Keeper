"""Proposal schema, codebase grounding, and theorizer traces.

A proposal is an object with one module, one symbol, and a check that can
come back false. Anything else is speculation: stored, never queued, never
injected into the chat prompt.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import logging
import re
import sqlite3
import textwrap
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

from config.settings import config
from core.codebase_index import SymbolResolution, project_root, resolve_symbol
from core.json_utils import parse_json_lenient
from core.timeutil import parse_iso, utcnow

log = logging.getLogger(__name__)

TRACE_SCHEMA = """
CREATE TABLE IF NOT EXISTS speculation (
    id              TEXT PRIMARY KEY,
    text            TEXT NOT NULL,
    reason          TEXT,
    created_at      TEXT NOT NULL,
    created_by_run  TEXT,
    target_module   TEXT,
    current_symbol  TEXT
);

CREATE TABLE IF NOT EXISTS proposal_trace (
    id                TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    created_by_run    TEXT,
    kind              TEXT NOT NULL,
    reject_reason     TEXT,
    target_module     TEXT,
    current_symbol    TEXT,
    verification_kind TEXT,
    verified_at       TEXT,
    verdict           TEXT,
    change_text       TEXT,
    current_behaviour TEXT,
    status            TEXT
);
"""

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "that",
    "this", "it", "and", "or", "but", "not", "no", "if", "so", "as",
    "should", "would", "could", "can", "will", "do", "does", "did",
    "have", "has", "had", "its", "into", "than", "then", "also",
}


class ProposalRejected(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass
class ScriptCheck:
    path: str
    passes_when: str


@dataclass
class MetricCheck:
    table: str
    column: str
    direction: str
    threshold: float
    window_hours: int


@dataclass
class Proposal:
    id: str
    target_module: str
    current_behaviour: str
    current_symbol: str
    change: str
    verification: ScriptCheck | MetricCheck
    created_at: str
    created_by_run: str
    verified_at: str | None = None
    verdict: str | None = None
    status: str = "pending_review"
    symbol_source: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        if not (self.current_behaviour or "").strip():
            raise ProposalRejected("incomplete")
        if not (self.change or "").strip():
            raise ProposalRejected("incomplete")
        if not (self.current_symbol or "").strip():
            raise ProposalRejected("unresolved_symbol")
        resolved, reason = resolve_symbol(self.target_module, self.current_symbol)
        if resolved is None:
            raise ProposalRejected(reason or "unresolved_symbol")
        self.target_module = resolved.path
        self.symbol_source = resolved.source
        if self.verdict not in (None, "pass", "fail", "not_run"):
            raise ProposalRejected("incomplete")

    def to_dict(self) -> dict:
        ver = self.verification
        if isinstance(ver, ScriptCheck):
            verification = {
                "kind": "script",
                "path": ver.path,
                "passes_when": ver.passes_when,
            }
            kind = "script"
        else:
            verification = {
                "kind": "metric",
                "table": ver.table,
                "column": ver.column,
                "direction": ver.direction,
                "threshold": ver.threshold,
                "window_hours": ver.window_hours,
            }
            kind = "metric"
        return {
            "id": self.id,
            "target_module": self.target_module,
            "current_behaviour": self.current_behaviour,
            "current_symbol": self.current_symbol,
            "change": self.change,
            "title": self.change[:80],
            "verification": verification,
            "verification_kind": kind,
            "created_at": self.created_at,
            "generated_at": self.created_at,
            "created_by_run": self.created_by_run,
            "verified_at": self.verified_at,
            "verdict": self.verdict,
            "status": self.status,
        }


def has_executable_check(item: dict | None) -> bool:
    """True when a hypothesis or proposal carries a script or metric check."""
    if not isinstance(item, dict):
        return False
    return parse_verification(item.get("verification")) is not None


def parse_verification(raw: Any) -> ScriptCheck | MetricCheck | None:
    if not isinstance(raw, dict):
        return None
    kind = (raw.get("kind") or "").strip().lower()
    if kind == "script":
        path = (raw.get("path") or "").strip().replace("\\", "/")
        passes = (raw.get("passes_when") or "").strip()
        if passes != "exit 0":
            return None
        if not _script_path_shape(path):
            return None
        return ScriptCheck(path=path, passes_when=passes)
    if kind == "metric":
        direction = (raw.get("direction") or "").strip().lower()
        if direction not in {"increases", "decreases", "varies"}:
            return None
        table = (raw.get("table") or "").strip()
        column = (raw.get("column") or "").strip()
        if not _safe_ident(table) or not _safe_ident(column):
            return None
        try:
            threshold = float(raw["threshold"])
            window = int(raw["window_hours"])
        except (KeyError, TypeError, ValueError):
            return None
        if window <= 0:
            return None
        return MetricCheck(
            table=table,
            column=column,
            direction=direction,
            threshold=threshold,
            window_hours=window,
        )
    return None


def _safe_ident(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name or ""))


def _script_path_shape(path: str) -> bool:
    if not path.startswith("scripts/") or not path.endswith(".py"):
        return False
    parts = Path(path).parts
    return ".." not in parts and parts[0] == "scripts"


def _tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(t) > 2 and t not in _STOP
    }


def lexical_states_differ(current: str, proposed: str) -> bool:
    """False when the proposed sentence restates the current one.

    The low-tier redundancy call uses the same yes/no question. This
    comparator is the offline decision and the fallback when that call
    is unavailable. Restatement fails closed: high overlap is 'no'.
    """
    current_toks = _tokens(current)
    proposed_toks = _tokens(proposed)
    if not proposed_toks:
        return False
    covered = len(proposed_toks & current_toks) / len(proposed_toks)
    novel = proposed_toks - current_toks
    if covered >= 0.55 and len(novel) <= 3:
        return False
    return True


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def behaviour_sentence(symbol: str, source: str, change: str = "") -> str:
    """One sentence describing the symbol from its source, not from a guess."""
    doc = ""
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        tree = None
    if tree is not None:
        target = None
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                target = node
                break
        if target is not None:
            doc = ast.get_docstring(target) or ""
        if not doc:
            doc = ast.get_docstring(tree) or ""
    sentences = _sentences(doc)
    if not sentences:
        comments = [
            ln.strip().lstrip("#").strip()
            for ln in source.splitlines()
            if ln.strip().startswith("#")
        ]
        sentences = _sentences(" ".join(comments))
    if not sentences:
        literals = re.findall(r"['\"]([^'\"]{8,80})['\"]", source)
        if literals:
            return f"{symbol} currently contains: " + "; ".join(literals[:3]) + "."
        return f"{symbol} is implemented in this module."
    if not change:
        return sentences[0]

    def score(sentence: str) -> int:
        return len(_tokens(sentence) & _tokens(change))

    best = max(sentences, key=score)
    if score(best) == 0:
        return sentences[0]
    return best


def current_run_id() -> str:
    """Process identity. A proposal's creating run cannot verify it."""
    host_pid = f"{uuid.getnode():x}:{__import__('os').getpid()}"
    try:
        path = Path(config.data_dir) / "state" / "instance.json"
        if path.exists():
            import json
            data = json.loads(path.read_text())
            started = data.get("started_at") or ""
            pid = data.get("pid") or ""
            if started or pid:
                return f"{pid}:{started}"
    except Exception:
        pass
    return host_pid


def _connect() -> sqlite3.Connection:
    path = Path(config.midterm_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.executescript(TRACE_SCHEMA)
    conn.commit()
    return conn


def record_speculation(
    *,
    text: str,
    reason: str,
    created_by_run: str,
    target_module: str = "",
    current_symbol: str = "",
    spec_id: str | None = None,
    created_at: str | None = None,
) -> str:
    sid = spec_id or str(uuid.uuid4())
    now = created_at or utcnow().isoformat()
    conn = _connect()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO speculation
               (id, text, reason, created_at, created_by_run, target_module, current_symbol)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (sid, text or "", reason, now, created_by_run, target_module, current_symbol),
        )
        conn.execute(
            """INSERT OR REPLACE INTO proposal_trace
               (id, created_at, created_by_run, kind, reject_reason,
                target_module, current_symbol, verification_kind,
                verified_at, verdict, change_text, current_behaviour, status)
               VALUES (?, ?, ?, 'speculation', ?, ?, ?, NULL, NULL, NULL, ?, NULL, 'speculation')""",
            (sid, now, created_by_run, reason, target_module, current_symbol, (text or "")[:500]),
        )
        conn.commit()
    finally:
        conn.close()
    return sid


def record_proposal_trace(proposal: dict) -> None:
    """Insert a queued proposal. Verdict columns stay null until the harness runs."""
    conn = _connect()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO proposal_trace
               (id, created_at, created_by_run, kind, reject_reason,
                target_module, current_symbol, verification_kind,
                verified_at, verdict, change_text, current_behaviour, status)
               VALUES (?, ?, ?, 'proposal', NULL, ?, ?, ?, NULL, NULL, ?, ?, ?)""",
            (
                proposal["id"],
                proposal.get("created_at"),
                proposal.get("created_by_run"),
                proposal.get("target_module"),
                proposal.get("current_symbol"),
                proposal.get("verification_kind"),
                proposal.get("change"),
                proposal.get("current_behaviour"),
                proposal.get("status") or "pending_review",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def write_verdict(proposal_id: str, verdict: str, *, when: str | None = None) -> None:
    """The only writer of verdict / verified_at. No model call reaches this."""
    if verdict not in {"pass", "fail", "not_run"}:
        raise ValueError(f"bad verdict {verdict}")
    stamp = when or utcnow().isoformat()
    verified_at = None if verdict == "not_run" else stamp
    conn = _connect()
    try:
        conn.execute(
            """UPDATE proposal_trace
               SET verdict = ?, verified_at = ?
               WHERE id = ? AND kind = 'proposal'""",
            (verdict, verified_at, proposal_id),
        )
        conn.commit()
    finally:
        conn.close()


def integration_score() -> float | None:
    """proposals with verdict pass / every theorizer output. None when empty."""
    conn = _connect()
    try:
        row = conn.execute(
            """SELECT
                 SUM(CASE WHEN kind = 'proposal' AND verdict = 'pass' THEN 1 ELSE 0 END),
                 COUNT(*)
               FROM proposal_trace"""
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()
    total = int(row[1] or 0)
    if total == 0:
        return None
    return int(row[0] or 0) / total


def parse_theorizer_item(raw: Any) -> dict:
    """Accept a JSON object or the labeled block the replay script feeds."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {"kind": "speculation", "text": str(raw)}
    text = raw.strip()
    parsed = parse_json_lenient(text)
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        return parsed[0]
    fields: dict[str, Any] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        norm = key.strip().lower().replace(" ", "_")
        if norm in {
            "target_module", "current_symbol", "current_behaviour",
            "change", "verification",
        }:
            fields[norm] = value.strip()
    if "verification" in fields:
        ver = parse_json_lenient(fields["verification"])
        if isinstance(ver, dict):
            fields["verification"] = ver
    if fields.get("target_module"):
        return fields
    return {"kind": "speculation", "text": text}


def coerce_theorizer_items(parsed: Any) -> list:
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("items", "outputs", "proposals", "results"):
            if isinstance(parsed.get(key), list):
                return parsed[key]
        if any(k in parsed for k in ("kind", "target_module", "text", "verification")):
            return [parsed]
    if isinstance(parsed, str):
        return [parsed]
    return []


def classify_theorizer_item(
    raw: Any,
    *,
    run_id: str,
    describe: Callable[[str, str, str], str] | None = None,
    differs: Callable[[str, str], bool] | None = None,
    spec_id: str | None = None,
    record: bool = True,
) -> dict:
    """Run one theorizer item through schema enforcement and redundancy.

    Returns a dict with kind 'proposal' or 'speculation'. Verdict is never
    set here. `describe` turns symbol source into current_behaviour.
    `differs` answers whether the proposed state is different; False routes
    to speculation with reason already_implemented.
    """
    describe = describe or behaviour_sentence
    differs = differs or lexical_states_differ
    item = parse_theorizer_item(raw)
    now = utcnow().isoformat()

    def _speculation(reason: str, text: str, module: str = "", symbol: str = "") -> dict:
        sid = spec_id or str(uuid.uuid4())
        if record:
            record_speculation(
                text=text,
                reason=reason,
                created_by_run=run_id,
                target_module=module,
                current_symbol=symbol,
                spec_id=sid,
                created_at=now,
            )
        return {
            "kind": "speculation",
            "reason": reason,
            "id": sid,
            "text": text,
            "target_module": module,
            "current_symbol": symbol,
        }

    if (item.get("kind") or "").lower() == "speculation" or (
        item.get("text") and not item.get("target_module")
    ):
        return _speculation("no_verification", str(item.get("text") or ""))

    verification = parse_verification(item.get("verification"))
    if verification is None:
        text = str(item.get("change") or item.get("text") or item)
        return _speculation(
            "no_verification",
            text,
            str(item.get("target_module") or ""),
            str(item.get("current_symbol") or ""),
        )

    target = str(item.get("target_module") or "")
    symbol = str(item.get("current_symbol") or "")
    change = str(item.get("change") or "").strip()
    if not change:
        return _speculation("incomplete", str(item), target, symbol)

    resolved, reason = resolve_symbol(target, symbol)
    if resolved is None:
        return _speculation(reason or "unresolved_symbol", change, target, symbol)

    try:
        current = describe(symbol, resolved.source, change).strip()
    except Exception:
        log.debug("current_behaviour describe failed", exc_info=True)
        current = ""
    if not current:
        current = behaviour_sentence(symbol, resolved.source, change)

    try:
        different = bool(differs(current, change))
    except Exception:
        log.debug("redundancy check failed — lexical fallback", exc_info=True)
        different = lexical_states_differ(current, change)
    if not different:
        return _speculation("already_implemented", change, resolved.path, symbol)

    proposal = Proposal(
        id=str(item.get("id") or uuid.uuid4()),
        target_module=resolved.path,
        current_behaviour=current,
        current_symbol=symbol,
        change=change,
        verification=verification,
        created_at=str(item.get("created_at") or now),
        created_by_run=run_id,
    )
    stored = proposal.to_dict()
    if record:
        record_proposal_trace(stored)
    return {"kind": "proposal", "reason": None, "proposal": stored}


async def classify_theorizer_item_async(
    raw: Any,
    *,
    run_id: str,
    describe: Callable[..., Any] | None = None,
    differs: Callable[..., Any] | None = None,
    spec_id: str | None = None,
    record: bool = True,
) -> dict:
    """Same as classify_theorizer_item, awaiting describe and differs."""

    async def _call(fn: Callable[..., Any] | None, fallback: Callable[..., Any], *args):
        if fn is None:
            return fallback(*args)
        result = fn(*args)
        if inspect.isawaitable(result):
            return await result
        return result

    def describe_sync(symbol: str, source: str, change: str) -> str:
        # Placeholder; the real call happens before Proposal construction
        # via the async path below. This keeps the signature local.
        return behaviour_sentence(symbol, source, change)

    # Re-implement the tail of classify so the two calls can be awaited.
    # Schema rejection stays in the sync function by passing describe/differs
    # that close over already-computed values when resolution succeeds.
    item = parse_theorizer_item(raw)
    target = str(item.get("target_module") or "")
    symbol = str(item.get("current_symbol") or "")
    change = str(item.get("change") or "").strip()
    verification = parse_verification(item.get("verification"))
    pre_resolved = None
    if verification is not None and change and target and symbol:
        pre_resolved, _reason = resolve_symbol(target, symbol)

    current_text = ""
    if pre_resolved is not None:
        current_text = await _call(describe, behaviour_sentence, symbol, pre_resolved.source, change)
        if not (current_text or "").strip():
            current_text = behaviour_sentence(symbol, pre_resolved.source, change)

    decided: bool | None = None
    if pre_resolved is not None and current_text:
        decided = bool(await _call(differs, lexical_states_differ, current_text, change))

    def _describe(sym: str, source: str, ch: str) -> str:
        if current_text:
            return current_text
        return describe_sync(sym, source, ch)

    def _differs(current: str, proposed: str) -> bool:
        if decided is not None:
            return decided
        return lexical_states_differ(current, proposed)

    return classify_theorizer_item(
        raw,
        run_id=run_id,
        describe=_describe,
        differs=_differs,
        spec_id=spec_id,
        record=record,
    )


def legacy_speculation_id(raw: dict) -> str:
    blob = str(raw.get("id") or "") + str(raw.get("title") or "") + str(raw.get("rationale") or "")
    digest = hashlib.sha1(blob.encode()).hexdigest()[:16]
    return f"legacy-{digest}"


# ── Verification harness ──────────────────────────────────────────────────

_ALLOWED_TABLES = {
    "state_trace",
    "proposal_trace",
    "forgetting_log",
    "register_trace",
    "speculation",
    "world_slot",
}


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if table not in _ALLOWED_TABLES:
        return set()
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.OperationalError:
        return set()
    return {str(r[1]) for r in rows}


def _metric_series(
    conn: sqlite3.Connection, check: MetricCheck, start_iso: str, end_iso: str,
) -> list[float] | None:
    """Numeric samples in the window. None when the query cannot be run."""
    if check.column == "verdict_pass_rate" and check.table == "proposal_trace":
        row = conn.execute(
            """SELECT
                 SUM(CASE WHEN verdict = 'pass' THEN 1 ELSE 0 END),
                 SUM(CASE WHEN verdict IS NOT NULL THEN 1 ELSE 0 END)
               FROM proposal_trace
               WHERE kind = 'proposal' AND created_at >= ? AND created_at <= ?""",
            (start_iso, end_iso),
        ).fetchone()
        denom = int(row[1] or 0)
        if denom == 0:
            return []
        return [int(row[0] or 0) / denom]

    columns = _table_columns(conn, check.table)
    if check.column not in columns:
        return None
    time_col = "ts" if "ts" in columns else ("created_at" if "created_at" in columns else None)
    if time_col is None:
        return None
    rows = conn.execute(
        f"""SELECT {check.column} FROM {check.table}
            WHERE {time_col} >= ? AND {time_col} <= ?
            ORDER BY {time_col} ASC""",
        (start_iso, end_iso),
    ).fetchall()
    values = []
    for row in rows:
        try:
            if row[0] is not None:
                values.append(float(row[0]))
        except (TypeError, ValueError):
            continue
    return values


def evaluate_metric(check: MetricCheck, *, created_at: str, now=None) -> str:
    """pass, fail, or not_run. Window not elapsed stays not_run."""
    now = now or utcnow()
    try:
        opened = parse_iso(created_at)
    except Exception:
        return "fail"
    if now < opened + timedelta(hours=check.window_hours):
        return "not_run"
    end = opened + timedelta(hours=check.window_hours)
    conn = _connect()
    try:
        series = _metric_series(conn, check, created_at, end.isoformat())
    finally:
        conn.close()
    if series is None or not series:
        return "fail"
    if check.direction == "varies":
        if len(series) < 2:
            return "fail"
        spread = max(series) - min(series)
        return "pass" if spread >= check.threshold else "fail"
    if check.column == "verdict_pass_rate":
        delta = series[-1]
        if check.direction == "increases":
            return "pass" if delta >= check.threshold else "fail"
        if check.direction == "decreases":
            return "pass" if delta <= check.threshold else "fail"
        return "fail"
    mid = max(1, len(series) // 2)
    older = sum(series[:mid]) / mid
    recent_vals = series[mid:] or series[-1:]
    recent = sum(recent_vals) / len(recent_vals)
    delta = recent - older
    if check.direction == "increases":
        return "pass" if delta >= check.threshold else "fail"
    if check.direction == "decreases":
        return "pass" if (-delta) >= check.threshold else "fail"
    return "fail"


def script_file(check: ScriptCheck) -> Path | None:
    if not _script_path_shape(check.path):
        return None
    full = (project_root() / check.path).resolve()
    try:
        full.relative_to(project_root().resolve())
    except ValueError:
        return None
    return full


def run_script_check(check: ScriptCheck) -> str:
    """A missing script is fail. pass only when the process exits 0."""
    import subprocess
    import sys
    path = script_file(check)
    if path is None or not path.is_file():
        return "fail"
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(project_root()),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "fail"
    return "pass" if proc.returncode == 0 else "fail"


def verify_proposal_dict(proposal: dict, *, run_id: str, now=None) -> str:
    """Run the check unless this is the run that created the proposal.

    Writes verdict only for pass/fail. not_run leaves verified_at null.
    Returns the verdict without writing when the creating run matches.
    """
    if proposal.get("created_by_run") == run_id:
        return "not_run"
    if proposal.get("verdict") in {"pass", "fail"}:
        return str(proposal["verdict"])
    verification = parse_verification(proposal.get("verification"))
    if verification is None:
        return "not_run"
    now = now or utcnow()
    if isinstance(verification, ScriptCheck):
        verdict = run_script_check(verification)
    else:
        verdict = evaluate_metric(
            verification,
            created_at=str(proposal.get("created_at") or ""),
            now=now,
        )
    if verdict == "not_run":
        return "not_run"
    proposal["verdict"] = verdict
    proposal["verified_at"] = now.isoformat()
    if proposal.get("status") == "pending_review":
        proposal["status"] = "verified"
    write_verdict(str(proposal.get("id")), verdict, when=proposal["verified_at"])
    return verdict


def verify_open_proposals(proposals: list[dict], *, run_id: str, now=None) -> int:
    """Verify queued proposals whose creating run is not this one. Returns writes."""
    written = 0
    for proposal in proposals:
        if proposal.get("status") not in (None, "pending_review", "verified"):
            continue
        if proposal.get("verdict") in {"pass", "fail"}:
            continue
        if not proposal.get("verification"):
            continue
        before = proposal.get("verdict")
        verify_proposal_dict(proposal, run_id=run_id, now=now)
        if proposal.get("verdict") in {"pass", "fail"} and proposal.get("verdict") != before:
            written += 1
    return written


def symbol_resolution_for(module_path: str, symbol: str) -> SymbolResolution | None:
    resolved, _reason = resolve_symbol(module_path, symbol)
    return resolved
