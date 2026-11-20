"""Module dependency rules of SPEC 8.1: dependencies point down only.

Each top-level subpackage of `entropy` may import only the subpackages listed for
it in SPEC 8.1, and, transitively, what those may import. `app` is the
composition root: it wires the window, game, renderer and interface together and
may import anything, while nothing may import it.
"""

import ast
from pathlib import Path

SRC = Path(__file__).parents[1] / "src" / "entropy"

# SPEC 8.1, "May depend on" column. Direct dependencies only; the closure is computed.
ALLOWED: dict[str, frozenset[str]] = {
    "core": frozenset(),
    "fields": frozenset({"core"}),
    "linalg": frozenset({"fields"}),
    "solvers": frozenset({"fields", "linalg"}),
    "coupling": frozenset({"solvers"}),
    "world": frozenset({"core"}),
    "sim": frozenset({"solvers", "coupling", "world"}),
    "game": frozenset({"sim", "world"}),
    "editor": frozenset({"world"}),
    "render": frozenset({"sim"}),
    "ui": frozenset({"game", "sim"}),
    "io": frozenset({"sim", "world"}),
}
COMPOSITION_ROOT = "app"


def reachable(package: str, allowed: dict[str, frozenset[str]]) -> frozenset[str]:
    """Every package `package` may import: its direct dependencies and theirs."""
    seen: set[str] = set()
    stack = list(allowed[package])
    while stack:
        dep = stack.pop()
        if dep not in seen:
            seen.add(dep)
            stack.extend(allowed[dep])
    return frozenset(seen)


def violations(
    imports: dict[str, set[str]], allowed: dict[str, frozenset[str]]
) -> list[tuple[str, str]]:
    """(importer, imported) pairs that break the rules.

    `imports` maps each subpackage to the `entropy` subpackages it imports.
    A subpackage missing from `allowed` is a violation, so a new package
    cannot slip in without its row in SPEC 8.1.
    """
    found: list[tuple[str, str]] = []
    for package, targets in sorted(imports.items()):
        for target in sorted(targets - {package}):
            if target == COMPOSITION_ROOT:
                found.append((package, target))
            elif package == COMPOSITION_ROOT:
                continue
            elif package not in allowed or target not in reachable(package, allowed):
                found.append((package, target))
    return found


def collect_imports(src: Path) -> dict[str, set[str]]:
    """Map each top-level subpackage (or module) of `src` to the siblings it imports."""
    imports: dict[str, set[str]] = {}
    for path in src.rglob("*.py"):
        relative = path.relative_to(src)
        if relative.parts == ("__init__.py",):
            continue
        package = relative.parts[0].removesuffix(".py")
        targets = imports.setdefault(package, set())
        containing = relative.parts[:-1]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    base = containing[: len(containing) - (node.level - 1)]
                    module = ".".join(("entropy", *base, *filter(None, [node.module])))
                else:
                    module = node.module or ""
                # `from entropy import core` or `from .. import core` name subpackages.
                names = [module, *(f"{module}.{alias.name}" for alias in node.names)]
            for name in names:
                parts = name.split(".")
                if parts[0] == "entropy" and len(parts) > 1:
                    targets.add(parts[1])
    return imports


def test_upward_import_is_a_violation() -> None:
    assert violations({"core": {"render"}}, ALLOWED) == [("core", "render")]


def test_transitive_downward_import_is_allowed() -> None:
    assert violations({"solvers": {"core"}, "game": {"fields"}}, ALLOWED) == []


def test_render_may_not_import_game() -> None:
    assert violations({"render": {"game"}}, ALLOWED) == [("render", "game")]


def test_unknown_package_is_a_violation() -> None:
    assert violations({"physics": {"core"}}, ALLOWED) == [("physics", "core")]


def test_composition_root_imports_anything_but_nothing_imports_it() -> None:
    assert violations({"app": {"render", "ui", "game"}, "ui": {"app"}}, ALLOWED) == [("ui", "app")]


def test_collect_imports_resolves_absolute_and_relative_imports(tmp_path: Path) -> None:
    (tmp_path / "render").mkdir()
    (tmp_path / "render" / "__init__.py").write_text("")
    (tmp_path / "render" / "a.py").write_text(
        "import entropy.sim.clock\nfrom entropy import core\nfrom ..game import tools\n"
    )

    assert collect_imports(tmp_path) == {"render": {"sim", "core", "game"}}


def test_source_tree_respects_spec_8_1() -> None:
    assert violations(collect_imports(SRC), ALLOWED) == []
