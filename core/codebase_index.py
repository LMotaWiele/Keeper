"""Codebase index — AST map of this repo. Injected only when self-reflection needs it."""
from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass
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


def project_root() -> Path:
    """Repository root (parent of core/)."""
    return Path(__file__).resolve().parents[1]


def normalize_module_path(module_path: str) -> str:
    rel = (module_path or "").strip().replace("\\", "/")
    if rel.startswith("./"):
        rel = rel[2:]
    return rel


def module_path_problem(module_path: str) -> str | None:
    """Reject a target that names more than one module.

    A single path such as core/user_life.py is one module. A comma, the
    word 'and', or more than one *.py path is a plan, not a proposal.
    """
    text = (module_path or "").strip()
    if not text:
        return "unresolved_module"
    if "," in text or ";" in text or "|" in text:
        return "multi_module"
    if re.search(r"\band\b", text, flags=re.IGNORECASE):
        return "multi_module"
    py_paths = re.findall(r"[\w./\\-]+\.py\b", text)
    if len(py_paths) > 1:
        return "multi_module"
    return None


def _skip_path(path: Path) -> bool:
    return any(part in path.parts for part in ("venv", ".venv", "data", "__pycache__"))


def find_module_file(root: Path, module_path: str) -> Path | None:
    """Resolve one relative module path to a file under root. None if missing."""
    rel = normalize_module_path(module_path)
    if not rel or module_path_problem(rel) == "multi_module":
        return None
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    if candidate.is_file():
        return candidate
    name = Path(rel).name
    if not name.endswith(".py"):
        return None
    matches = [
        p for p in root.rglob(name)
        if p.is_file() and not _skip_path(p) and p.as_posix().endswith(rel)
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def symbol_source(file_path: Path, symbol: str) -> str | None:
    """Return the source of a function or class named symbol, or None."""
    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        tree = ast.parse(raw)
    except SyntaxError:
        return None
    wanted = (symbol or "").strip()
    if not wanted:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == wanted:
                segment = ast.get_source_segment(raw, node)
                return segment if segment else None
    return None


@dataclass
class SymbolResolution:
    path: str
    source: str


def resolve_symbol(
    module_path: str,
    symbol: str,
    project: Path | None = None,
) -> tuple[SymbolResolution | None, str | None]:
    """Resolve a proposal target.

    Returns (resolution, None) or (None, reason) where reason is
    multi_module, unresolved_module, or unresolved_symbol.
    """
    problem = module_path_problem(module_path)
    if problem:
        return None, problem
    root = project or project_root()
    file_path = find_module_file(root, module_path)
    if file_path is None:
        return None, "unresolved_module"
    source = symbol_source(file_path, symbol)
    if source is None:
        return None, "unresolved_symbol"
    rel = file_path.resolve().relative_to(root.resolve()).as_posix()
    return SymbolResolution(path=rel, source=source), None


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

    def resolve_symbol(self, module_path: str, symbol: str):
        """Resolve one module and one symbol against this checkout."""
        return resolve_symbol(module_path, symbol, project=self.project_root)

    def list_modules(self) -> str:
        """Path, pillar, one-line purpose — no source."""
        if not self.index:
            return "Codebase index is empty."
        lines = []
        for path, info in sorted(self.index.items()):
            purpose = (info.get("purpose") or "").split("\n")[0].strip() or "no docstring"
            lines.append(f"{path}\t{info.get('pillar')}\t{purpose}")
        return "\n".join(lines)
