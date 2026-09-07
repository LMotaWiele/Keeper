"""Atomic file replace — write tmp, then os.replace onto the real path."""
from __future__ import annotations

import os
from pathlib import Path


def atomic_write_text(path: Path | str, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
