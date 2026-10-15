"""Storage isolation guard shared by SPEC-0004 repair commands."""
from __future__ import annotations

from pathlib import Path


def require_isolated_paths():
    """Report resolved settings before any loader and refuse storage escapes."""
    from config.settings import config

    paths = {name: Path(getattr(config, attr)).resolve() for name, attr in (
        ("DATA_DIR", "data_dir"), ("MIDTERM_DB_PATH", "midterm_db_path"),
        ("CHROMA_DB_PATH", "chroma_db_path"), ("EMBEDDING_CACHE_DIR", "embedding_cache_dir"),
    )}
    for name, path in paths.items():
        print(f"{name}={path}", flush=True)
    root = paths["DATA_DIR"]
    for name, path in paths.items():
        if name != "DATA_DIR" and (path == root or not path.is_relative_to(root)):
            raise ValueError(f"Refusing repair: {name} must be under DATA_DIR")
    return config
