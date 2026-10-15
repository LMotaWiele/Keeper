<!-- agent-memory-schema: 1 -->
# Notes index — read this first

Agents: load **this file only**, then the **one** row matching the task. Do not bulk-read notes/.

This file is the **authority for what happens next**. Notes hold conditions; this holds the queue.

## Notes

| Topic | File | Status | One-liner |
|---|---|---|---|
| How to write notes | [README.md](README.md) | meta | layers, front-matter, access rules |
| _(add rows as findings appear)_ | | | |

`Status` must match the `**Status:**` line in the linked file. Mismatch fails the pre-commit check.

## Active next

Ordered. Top row is what the next session picks up. One line each — detail lives in the source note.

| # | Item | Source | Added |
|---|---|---|---|
| _(empty)_ | | | |

`Source` is the note whose pre-committed condition fired, or `human` if you queued it directly.
Close a row by deleting it. Do not leave completed rows with a status marker — the queue is not a log.

## Do not

Standing constraints. These bind every session in this repo until removed here.

- _(none yet)_

## Glossary (optional)

Project-specific verdicts, modes, and label names. If a label appears in a note's front-matter or in a routing question, it belongs here.

| Name | Means |
|---|---|
| _(project-specific verdicts / modes)_ | |
