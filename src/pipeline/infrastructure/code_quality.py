"""Codebase review tooling (Agent Directive V7 Section 12).

Provides automated code quality checks:
- Circular import detection
- Architecture map generation
- Module dependency analysis
"""

from __future__ import annotations

import ast
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def check_circular_imports(
    package_path: str | Path | None = None,
) -> list[str]:
    """Detect circular import chains among pipeline submodules.

    Returns a list of issue descriptions (empty if no problems found).
    """
    if package_path is None:
        package_path = Path(__file__).resolve().parents[1]  # src/pipeline
    package_path = Path(package_path)

    # Build import graph from AST analysis
    graph: dict[str, set[str]] = {}
    for py_file in package_path.rglob("*.py"):
        module_name = _path_to_module(py_file, package_path.parent)
        if module_name:
            graph[module_name] = _extract_imports(py_file, "pipeline")

    # Find cycles using DFS
    cycles = _find_cycles(graph)
    return [f"Circular import: {' -> '.join(cycle)}" for cycle in cycles]


def generate_architecture_map(
    package_path: str | Path | None = None,
) -> dict[str, Any]:
    """<architecture_review_report> — Generate a module dependency map.

    Returns a dict with module names, sizes, and their import dependencies.
    """
    if package_path is None:
        package_path = Path(__file__).resolve().parents[1]
    package_path = Path(package_path)

    modules: list[dict[str, Any]] = []
    for py_file in sorted(package_path.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue
        module_name = _path_to_module(py_file, package_path.parent)
        if not module_name:
            continue
        try:
            with open(py_file) as fh:
                line_count = sum(1 for _ in fh)
        except OSError:
            line_count = 0
        imports = _extract_imports(py_file, "pipeline")
        modules.append(
            {
                "module": module_name,
                "file": str(py_file.relative_to(package_path.parent)),
                "lines": line_count,
                "internal_imports": sorted(imports),
            }
        )

    return {
        "report_type": "architecture_map",
        "total_modules": len(modules),
        "total_lines": sum(m["lines"] for m in modules),
        "modules": modules,
        "generated_at": datetime.now(UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _path_to_module(file_path: Path, src_root: Path) -> str | None:
    """Convert a file path to a dotted module name."""
    try:
        rel = file_path.relative_to(src_root)
    except ValueError:
        return None
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return None
    return ".".join(parts)


def _extract_imports(file_path: Path, prefix: str) -> set[str]:
    """Extract internal (same-package) imports from a Python file via AST."""
    imports: set[str] = set()
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, OSError):
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(prefix):
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(prefix):
            imports.add(node.module)
    return imports


def _find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Find all cycles in a directed graph using DFS."""
    white, gray, black = 0, 1, 2
    color: dict[str, int] = dict.fromkeys(graph, white)
    path: list[str] = []
    cycles: list[list[str]] = []

    def dfs(node: str) -> None:
        color[node] = gray
        path.append(node)
        for neighbor in graph.get(node, set()):
            if neighbor not in color:
                continue
            if color[neighbor] == gray:
                # Found a cycle
                idx = path.index(neighbor)
                cycles.append(path[idx:] + [neighbor])
            elif color[neighbor] == white:
                dfs(neighbor)
        path.pop()
        color[node] = black

    for node in graph:
        if color[node] == white:
            dfs(node)

    return cycles
