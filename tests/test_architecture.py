"""Module dependency rules of SPEC 8.1: dependencies point down only (ADR 0005).

Each top-level subpackage of `entropy` may import only the subpackages listed for
it in SPEC 8.1, and, transitively, what those may import. `app` (with
`__main__`) is the composition root: it wires the window, game, renderer and
interface together and may import anything, while nothing may import it. The
package root `entropy/__init__.py` runs on every import, so it may import no
subpackage.

Known limits, checked by review instead:
- Imports are read statically; `importlib.import_module` and attribute access
  through an imported `entropy` are not seen.
- "Read-only" in the `render`, `ui` and `audio` rows is about writes, not imports.
- Subpackages are the unit, so imports between two solvers inside `solvers` are
  not checked yet (issue #21, before the first two solvers land in M1).
"""

import ast
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

SRC = Path(__file__).parents[1] / "src" / "entropy"

ROOT = "entropy"
COMPOSITION_ROOTS = frozenset({"app", "__main__"})
NOT_IN_SPEC = "<no row in SPEC 8.1>"

# SPEC 8.1, "May depend on" column. Direct dependencies only; the closure is computed.
ALLOWED: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        ROOT: frozenset(),
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
        "audio": frozenset({"sim"}),
    }
)


def reachable(package: str, allowed: Mapping[str, frozenset[str]]) -> frozenset[str]:
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
    imports: Mapping[str, set[str]], allowed: Mapping[str, frozenset[str]]
) -> list[tuple[str, str]]:
    """(importer, imported) pairs that break the rules.

    `imports` maps each subpackage to the `entropy` subpackages it imports.
    A subpackage missing from `allowed` is reported as (package, NOT_IN_SPEC),
    so a new package cannot slip in without its row in SPEC 8.1.
    """
    found: list[tuple[str, str]] = []
    for package, targets in sorted(imports.items()):
        is_root = package in COMPOSITION_ROOTS
        if not is_root and package not in allowed:
            found.append((package, NOT_IN_SPEC))
            continue
        for target in sorted(targets - {package}):
            if target in COMPOSITION_ROOTS:
                if not is_root:
                    found.append((package, target))
            elif not is_root and target not in reachable(package, allowed):
                found.append((package, target))
    return found


def collect_imports(src: Path) -> dict[str, set[str]]:
    """Map each top-level subpackage or module of `src` to the siblings it imports.

    The package root `src/__init__.py` is reported under ROOT.
    """
    siblings = {path.name.removesuffix(".py") for path in src.iterdir()} - {"__init__"}
    imports: dict[str, set[str]] = {}
    for path in src.rglob("*.py"):
        relative = path.relative_to(src)
        package = ROOT if relative.parts == ("__init__.py",) else relative.parts[0]
        package = package.removesuffix(".py")
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
                if parts[0] == "entropy" and len(parts) > 1 and parts[1] in siblings:
                    targets.add(parts[1])
    return imports


def test_upward_import_is_a_violation() -> None:
    assert violations({"core": {"render"}}, ALLOWED) == [("core", "render")]


def test_transitive_downward_import_is_allowed() -> None:
    assert violations({"solvers": {"core"}, "game": {"fields"}}, ALLOWED) == []


def test_render_may_not_import_game() -> None:
    assert violations({"render": {"game"}}, ALLOWED) == [("render", "game")]


def test_unknown_package_is_a_violation_even_without_imports() -> None:
    assert violations({"physics": set()}, ALLOWED) == [("physics", NOT_IN_SPEC)]


def test_composition_root_imports_anything_but_nothing_imports_it() -> None:
    imports = {"app": {"render", "ui", "game"}, "__main__": {"app"}, "ui": {"app"}}

    assert violations(imports, ALLOWED) == [("ui", "app")]


def test_package_root_may_not_import_subpackages() -> None:
    """`entropy/__init__.py` runs on every import, so it would make everything depend on it."""
    assert violations({ROOT: {"render"}}, ALLOWED) == [(ROOT, "render")]


def _write(root: Path, relative: str, text: str = "") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_collect_imports_resolves_absolute_and_relative_imports(tmp_path: Path) -> None:
    for name in ("sim", "core", "game"):
        _write(tmp_path, f"{name}/__init__.py")
    _write(
        tmp_path,
        "render/a.py",
        "import entropy.sim.clock\nfrom entropy import core\nfrom ..game import tools\n",
    )

    assert collect_imports(tmp_path)["render"] == {"sim", "core", "game"}


def test_collect_imports_checks_the_package_root(tmp_path: Path) -> None:
    _write(tmp_path, "__init__.py", "from . import render\n")
    _write(tmp_path, "render/__init__.py")

    assert collect_imports(tmp_path)[ROOT] == {"render"}


def test_collect_imports_ignores_names_that_are_not_modules(tmp_path: Path) -> None:
    _write(tmp_path, "__init__.py", "__version__ = '0'\n")
    _write(tmp_path, "core/a.py", "from entropy import __version__\nfrom .. import __version__\n")

    assert collect_imports(tmp_path)["core"] == set()


def test_source_tree_respects_spec_8_1() -> None:
    imports = collect_imports(SRC)

    assert ROOT in imports, f"{SRC} has no package root: the check would see nothing"
    assert violations(imports, ALLOWED) == []
