"""Codebase index — AST map of this repo. Injected only when self-reflection needs it."""
from __future__ import annotations

import ast
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def build_index(project_root: Path) -> dict[str, dict]:
    """
    Scan the project and build a structural index from docstrings
    and class/function signatures. No LLM call needed — pure parsing.
    """
    index = {}
    for py_file in project_root.rglob("*.py"):
        if py_file.name.startswith("_") and py_file.name != "__init__.py":
            continue
        # Skip venv / data dirs
        rel = str(py_file.relative_to(project_root))
        if any(part in rel for part in ("venv", ".venv", "data", "__pycache__")):
            continue

        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue

        module_doc = ast.get_docstring(tree) or ""
        classes = []
        functions = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                methods = [
                    m.name for m in node.body
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and not m.name.startswith("_")
                ]
                classes.append({
                    "name": node.name,
                    "doc": (ast.get_docstring(node) or "")[:200],
                    "methods": methods,
                })
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)

        pillar = _classify_pillar(rel)
        index[rel] = {
            "purpose": module_doc[:300],
            "classes": classes,
            "module_functions": functions,
            "pillar": pillar,
        }

    return index


def _classify_pillar(path: str) -> str:
    if path.startswith("memory/"):
        return "1-memory"
    elif path.startswith("core/internal_state") or path.startswith("core/drive"):
        return "2-internal_state"
    elif path.startswith("environment/"):
        return "3-environment"
    elif path.startswith("goals/"):
        return "4-goals"
    elif path.startswith("core/self_model") or path.startswith("core/opinions"):
        return "5-self_model"
    elif path.startswith("core/codebase"):
        return "5-self_model"
    elif path.startswith("agent/"):
        return "orchestration"
    elif path.startswith("tg/") or path.startswith("telegram/"):
        return "interface"
    return "infrastructure"


def get_full_source(project_root: Path, module_path: str) -> str:
    """Load full source of a specific module — used sparingly."""
    fp = project_root / module_path
    if fp.exists():
        return fp.read_text(encoding="utf-8")
    return f"[File not found: {module_path}]"


def summarize_architecture(index: dict) -> str:
    """
    Compact text summary suitable for injection into self-model prompts.
    ~500-800 tokens, structured by pillar.
    """
    by_pillar: dict[str, list] = {}
    for path, info in index.items():
        p = info["pillar"]
        by_pillar.setdefault(p, []).append((path, info))

    lines = ["## My Architecture (derived from my own source code)\n"]
    for pillar in sorted(by_pillar):
        lines.append(f"### {pillar}")
        for path, info in sorted(by_pillar[pillar]):
            cls_names = [c["name"] for c in info["classes"]]
            cls_str = f" — classes: {', '.join(cls_names)}" if cls_names else ""
            purpose = info["purpose"].split("\n")[0] if info["purpose"] else "no docstring"
            lines.append(f"- `{path}`: {purpose}{cls_str}")
        lines.append("")

    return "\n".join(lines)


class CodebaseIndex:
    """Manages the index lifecycle. Rebuilt on startup, cached in RAM."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.index: dict = {}
        self._summary: str = ""

    def rebuild(self) -> None:
        self.index = build_index(self.project_root)
        self._summary = summarize_architecture(self.index)
        log.info("Codebase index built: %d modules", len(self.index))

    @property
    def summary(self) -> str:
        return self._summary

    def get_module_detail(self, module_path: str) -> str:
        """Get detailed info about a specific module (for targeted reflection)."""
        info = self.index.get(module_path)
        if not info:
            return f"Unknown module: {module_path}"

        parts = [
            f"Module: {module_path}",
            f"Pillar: {info['pillar']}",
            f"Purpose: {info['purpose']}",
        ]
        for cls in info["classes"]:
            parts.append(f"\nClass {cls['name']}: {cls['doc']}")
            parts.append(f"  Public methods: {', '.join(cls['methods'])}")
        return "\n".join(parts)

    def get_source(self, module_path: str) -> str:
        return get_full_source(self.project_root, module_path)

    def list_modules(self) -> str:
        """Path, pillar, one-line purpose — no source."""
        if not self.index:
            return "Codebase index is empty."
        lines = []
        for path, info in sorted(self.index.items()):
            purpose = (info.get("purpose") or "").split("\n")[0].strip() or "no docstring"
            lines.append(f"{path}\t{info.get('pillar')}\t{purpose}")
        return "\n".join(lines)
