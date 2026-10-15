# Specifications

Implementation documents use `SPEC-NNNN-<slug>.md`; new specs take the next unused number. Run records use `SPEC-NNNN-run.md`. Architecture and reference guides remain in `docs/`.

Implemented by is filled from the run record. Migrated historical specs retain their recorded implementation status and cite the original commit; this migration does not certify their acceptance checks. Numbers follow first-added history (routing precedes context-time within the same commit).

| ID | Title | Status | Implemented by | Date first added | Former path |
|---|---|---|---|---|---|
| [SPEC-0001](SPEC-0001-routing-v2.md) | Keeper routing v2 | implemented (historical) | `1f0370a` | 2026-09-22 | `docs/routing-v2-spec.md` |
| [SPEC-0002](SPEC-0002-context-time.md) | Prompt reinsertion and temporal binding | implemented (historical) | `1f0370a` | 2026-09-22 | `docs/context-time-spec.md` |
| [SPEC-0003](SPEC-0003-relational-grounding.md) | Relational grounding | implemented (historical; day-14 acceptance unverified) | `21a9136` | 2026-09-23 | `docs/Keeper Patch 03 — Relational Grounding.md` |
| [SPEC-0004](SPEC-0004-autonomous-loop-hygiene.md) | Autonomous-loop hygiene | implemented | coding assistant: `7dc2270`, `0534cf9`, `23d2c72` (local) | 2026-10-15 | `docs/specs/SPEC-nnnn-autonomous-loop-hygiene.md` |
