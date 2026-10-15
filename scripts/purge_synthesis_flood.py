#!/usr/bin/env python3
"""Invalidate SPEC-0004 synthesis flood candidates; dry-run unless --apply.

Uses stored vectors, not fresh model embeddings. All four storage paths must
be isolated before even importing the companion loader.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def source_ids(meta: dict) -> list:
    value = meta.get("source_episode_ids", [])
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("Malformed source_episode_ids; refusing repair")
    return value


def flood_candidates(goals: list[dict], docs: list[dict], threshold: float) -> tuple[set[str], set[str]]:
    """A is action-log references; B is source-less neighbours of A."""
    from memory.similarity import cosine

    a = {
        ref.split(":", 1)[1]
        for goal in goals
        for entry in goal.get("action_log", [])
        if entry.get("action_type") == "memory_synthesis"
        for ref in entry.get("artifact_refs", [])
        if isinstance(ref, str) and ref.startswith("semantic:")
    }
    seeds = [doc for doc in docs if doc["id"] in a]
    b = {
        doc["id"] for doc in docs
        if not source_ids(doc["metadata"])
        and any(cosine(doc["embedding"], seed["embedding"]) >= threshold for seed in seeds)
    }
    return a, b


async def run(*, apply: bool = False) -> None:
    from scripts.repair_common import require_isolated_paths

    config = require_isolated_paths()
    # Deliberately delayed until the isolation guard has succeeded.
    from scripts.diag_common import load_companion_no_loops
    from memory.semantic import semantic

    await load_companion_no_loops()
    # Read the full saved logs, including archived goals, without snapshot caps.
    goals_path = Path(config.data_dir) / "state" / "goals.json"
    saved = json.loads(goals_path.read_text()) if goals_path.exists() else {}
    goals = [goal for key in ("instrumental", "completed", "abandoned") for goal in saved.get(key, [])]
    users = set(config.allowed_user_ids)
    if not users:
        raise ValueError("No configured user IDs; refusing an incomplete repair")
    for user_id in sorted(users):
        col = semantic._collection(user_id)
        rows = col.get(include=["documents", "metadatas", "embeddings"])
        docs = [
            {"id": doc_id, "content": rows["documents"][i],
             "metadata": rows["metadatas"][i] or {}, "embedding": rows["embeddings"][i]}
            for i, doc_id in enumerate(rows["ids"])
        ]
        a, b = flood_candidates(goals, docs, config.semantic_dedup_sim)
        candidates = a | b
        by_id = {doc["id"]: doc for doc in docs}
        untouched = sum(not source_ids(doc["metadata"]) and doc["id"] not in candidates for doc in docs)
        print(f"user={user_id} |A|={len(a)} |B|={len(b)} |A ∪ B|={len(candidates)}")
        for doc_id in sorted(candidates):
            doc = by_id.get(doc_id)
            print(f"  {doc_id}: {doc['content'][:240] if doc else '(missing document)'}")
        print(f"other source-less docs left untouched: {untouched}")
        if apply:
            # Never silently erase newly evidenced post-patch synthesis. This
            # also makes a V5 mismatch explicit rather than weakening A ∪ B.
            if any(source_ids(by_id[i]["metadata"]) for i in candidates if i in by_id):
                raise ValueError("Candidates include evidenced documents; refusing repair")
            changed = 0
            for doc_id in sorted(candidates):
                doc = by_id.get(doc_id)
                if doc and doc["metadata"].get("invalidated") != "true":
                    await semantic.invalidate(
                        user_id, doc_id, reason="SPEC-0004: unevidenced autonomous synthesis",
                    )
                    changed += 1
            print(f"applied: invalidated {changed} document(s)")
        else:
            print("dry-run — pass --apply to invalidate candidates")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(apply=args.apply))


if __name__ == "__main__":
    main()
