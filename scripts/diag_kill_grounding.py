#!/usr/bin/env python3
"""Phase B branch two: kill the grounding task and record the observable signature.

Uses whatever DATA_DIR the process was started with. Do not point this at live
state — copy first.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["DIAG_KILL_GROUNDING"] = "1"

from config.settings import config  # noqa: E402
from core.loop import companion  # noqa: E402


async def amain() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("diag_kill_grounding")
    config.ensure_dirs()
    print(f"DATA_DIR={config.data_dir} DIAG_KILL_GROUNDING={os.environ.get('DIAG_KILL_GROUNDING')}")

    await companion.startup()
    await asyncio.sleep(1.5)

    by_name = {t.get_name(): t for t in companion._background_tasks}
    print("tasks:", {n: ("done" if t.done() else "running") for n, t in by_name.items()})
    grounding = by_name.get("grounding")
    if grounding is None:
        print("FAIL: no grounding task")
        await companion.shutdown()
        return
    print(f"grounding.done={grounding.done()} cancelled={grounding.cancelled()}")
    if grounding.done() and not grounding.cancelled():
        exc = grounding.exception()
        print(f"grounding.exception={exc!r}")

    last_before = companion.state.last_updated
    print(f"last_updated before user turn: {last_before.isoformat()}")

    ids = list(config.allowed_user_ids)
    user_id = ids[0] if ids else 0
    await companion.process(user_id, "diag kill-grounding ping")
    after_process = companion.state.last_updated
    print(f"last_updated after process(): {after_process.isoformat()}")
    print(f"process() advanced last_updated: {after_process != last_before}")

    frozen = companion.state.last_updated
    wait_s = max(5, int(config.grounding_interval) + 2)
    print(f"waiting {wait_s}s for TimePassingEvent (should NOT arrive)")
    await asyncio.sleep(wait_s)
    after_idle = companion.state.last_updated
    print(f"last_updated after idle: {after_idle.isoformat()}")
    print(f"idle advanced last_updated: {after_idle != frozen}")
    print(
        "signature: user turns still advance last_updated even with grounding dead; "
        "only TimePassingEvent ticks freeze. Overnight frozen last_updated PLUS later "
        "episodic rows cannot be the same process unless process() was not the writer."
    )

    await companion.shutdown()
    log.info("shutdown complete")


if __name__ == "__main__":
    asyncio.run(amain())
