#!/usr/bin/env python3
"""Replay proposal 3 through schema enforcement and codebase grounding.

Proposal 3 restated frictionless abandonment. Patch 02 already shipped
that on update_commitment_status: abandoning needs no confirmation and
no follow-up. This script runs that text through the proposal pipeline
and exits 0 only when the result is speculation with reason
already_implemented.

The redundancy decision is the offline lexical check, which is also the
fallback for the low-tier call. No model call, and no verification run.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from goals.proposals import classify_theorizer_item  # noqa: E402

SPEC_ID = "replay-proposal-3"

# The original proposal text is not stored in the repo. This is the
# diagnostic restatement: the shipped docstring, said again.
PROPOSAL_3 = {
    "target_module": "tools/user_life_tools.py",
    "current_symbol": "update_commitment_status",
    "change": (
        "Abandoning a commitment should be frictionless, with no "
        "confirmation and no follow-up surfacing."
    ),
    "verification": {
        "kind": "script",
        "path": "scripts/replay_proposal_3.py",
        "passes_when": "exit 0",
    },
}


def main() -> int:
    result = classify_theorizer_item(
        PROPOSAL_3,
        run_id=SPEC_ID,
        spec_id=SPEC_ID,
        record=True,
    )
    kind = result.get("kind")
    reason = result.get("reason")
    if kind == "speculation" and reason == "already_implemented" and result.get("id") == SPEC_ID:
        print("speculation already_implemented")
        return 0
    print(f"FAIL kind={kind} reason={reason} id={result.get('id')}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
