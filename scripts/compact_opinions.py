#!/usr/bin/env python3
"""Compact external opinion clones on an isolated copy; dry-run by default."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory.similarity import cosine, embed_texts


def compact_snapshot(snapshot: dict, vectors: list, threshold: float) -> tuple[dict, list[list[str]]]:
    """Greedy conviction-first clusters, preserving raw non-external records.

    A removed member represents itself plus its prior corroborations. Keeping
    that mass on the canonical makes retries (including singleton clusters)
    idempotent, rather than resetting prior research corroboration to zero.
    Vectors follow external opinions in snapshot insertion order.
    """
    result = copy.deepcopy(snapshot)
    opinions = result.get("opinions", {})
    external = [(oid, op) for oid, op in opinions.items() if op.get("origin") == "external"]
    if len(vectors) != len(external):
        raise ValueError("Expected one embedding per external opinion")
    embedded = {oid: vec for (oid, _), vec in zip(external, vectors)}
    ordered = sorted(external, key=lambda item: -item[1].get("conviction", 0.5))
    clusters: list[list[str]] = []
    for oid, _ in ordered:
        for cluster in clusters:
            if cosine(embedded[oid], embedded[cluster[0]]) >= threshold:
                cluster.append(oid)
                break
        else:
            clusters.append([oid])

    for cluster in clusters:
        if len(cluster) == 1:
            continue  # Do not add fields or reset counts on a second apply.
        canonical = opinions[cluster[0]]
        evidence = list(canonical.get("related_evidence", []))
        count = canonical.get("corroborated_count", 0)
        timestamps = [canonical.get("last_corroborated_at")]
        for oid in cluster[1:]:
            member = opinions.pop(oid)
            evidence.extend(member.get("related_evidence", []))
            count += 1 + member.get("corroborated_count", 0)
            timestamps.append(member.get("last_corroborated_at"))
        canonical["related_evidence"] = list(dict.fromkeys(evidence))[:10]
        canonical["corroborated_count"] = count
        known_times = [t for t in timestamps if t]
        if known_times:
            canonical["last_corroborated_at"] = max(known_times)
    return result, clusters


def _updated_json(original: str, updated: dict) -> str:
    """Replace changed JSON values, retaining unchanged records byte-for-byte."""
    decoder = json.JSONDecoder()
    previous = json.loads(original)
    if previous == updated:
        return original
    # Preserve raw value tokens rather than round-tripping non-external records
    # through an encoder (which would change spacing/escapes or legacy fields).
    tokens = {}
    pos = original.index("{") + 1
    while True:
        while original[pos].isspace():
            pos += 1
        if original[pos] == "}":
            break
        key, pos = decoder.raw_decode(original, pos)
        while original[pos].isspace() or original[pos] == ":":
            pos += 1
        start = pos
        _, pos = decoder.raw_decode(original, pos)
        tokens[key] = original[start:pos]
        while original[pos].isspace():
            pos += 1
        if original[pos] == ",":
            pos += 1
    fields = []
    for key, value in updated.items():
        if key in tokens and previous[key] == value:
            encoded = tokens[key]
        elif key in tokens and isinstance(previous[key], dict) and isinstance(value, dict):
            encoded = _updated_json(tokens[key], value)
        else:
            encoded = json.dumps(value, ensure_ascii=False)
        fields.append(json.dumps(key) + ": " + encoded)
    return "{\n" + ",\n".join(fields) + "\n}"


async def run(apply: bool = False) -> None:
    from scripts.repair_common import require_isolated_paths
    config = require_isolated_paths()  # Every mode, before loader or file access.
    from scripts.diag_common import load_companion_no_loops
    from core.atomic import atomic_write_text

    await load_companion_no_loops()
    state = Path(config.data_dir) / "state"
    authoritative = state / "opinions.json"
    model_path = state / "self_model.json"
    model_text = model_path.read_text() if model_path.exists() else None
    model = json.loads(model_text) if model_text is not None else None
    authoritative_text = authoritative.read_text() if authoritative.exists() else None
    # Startup restores self_model first, then opinions.json if it exists.
    snapshot = (json.loads(authoritative_text) if authoritative_text is not None
                else (model or {}).get("_opinions"))
    if snapshot is None:
        raise ValueError("No authoritative opinion snapshot found")
    external = [op for op in snapshot.get("opinions", {}).values() if op.get("origin") == "external"]
    vectors = await embed_texts([op["position"] for op in external])
    compacted, clusters = compact_snapshot(snapshot, vectors, config.opinion_dup_sim)
    for cluster in clusters:
        sample = snapshot["opinions"][cluster[-1]]["position"]
        print(f"canonical={cluster[0]} members={len(cluster)} sample={sample[:160]}")
    print(f"external={len(external)} clusters={len(clusters)} removed={len(external) - len(clusters)}")
    if not apply:
        print("dry-run: no opinion stores written")
        return

    # Atomic replacement per store, matching core.atomic's supported contract.
    # Raw dictionaries avoid dataclass defaults changing non-external records.
    if authoritative.exists() and compacted != snapshot:
        atomic_write_text(authoritative, _updated_json(authoritative_text, compacted))
    if model is not None and "_opinions" in model and model["_opinions"] != compacted:
        model["_opinions"] = compacted
        atomic_write_text(model_path, _updated_json(model_text, model))
    print("applied: opinion stores consistent")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(apply=args.apply))


if __name__ == "__main__":
    main()
