"""Module dependency rules of SPEC 8.1: dependencies point down only (ADR 0005).

Each top-level subpackage of `entropy` may import only the subpackages listed for
it in SPEC 8.1, and, transitively, what those may import. `app` (with
`__main__`) is the composition root: it wires the window, game, renderer and
interface together and may import anything, while nothing may import it. The
package root `entropy/__init__.py` runs on every import, so it may import no
subpackage.

Inside `solvers`, each solver (a module or subpackage of `solvers`) is its own
unit, named `solvers.<name>` (SPEC 8: solvers talk to each other only through
the coupling layer):
- A solver may import `solvers.interface`, the package root `solvers` and what
  `solvers` may import (`fields`, `linalg`, `core`), never another solver.
  Every module of `solvers` other than `interface` counts as a solver, so code
  shared by solvers goes to `fields`, `linalg` or the interface.
- The interface names no solver. The package root imports the interface only, to
  re-export it, so a solver written there cannot use `fields`, `linalg` or `core`.
- `coupling` may import the interface and the package root only (SPEC 8.1:
  solver interfaces only).
Other packages see `solvers` as one package.

Known limits, checked by review instead:
- Imports are read statically; `importlib.import_module` and attribute access
  through an imported `entropy` are not seen.
- "Read-only" in the `render`, `ui` and `audio` rows is about writes, not imports.
- Code written in `solvers/interface.py`, or in `solvers/__init__.py` without
  importing an `entropy` package, is not seen: review keeps both to the interface.
"""

import ast
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

SRC = Path(__file__).parents[1] / "src" / "entropy"

ROOT = "entropy"
COMPOSITION_ROOTS = frozenset({"app", "__main__"})
NOT_IN_SPEC = "<no row in SPEC 8.1>"
SOLVERS = "solvers"
SOLVER_INTERFACE = "solvers.interface"

# SPEC 8.1, "May depend on" column. Direct dependencies only; the closure is computed.
ALLOWED: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        ROOT: frozenset(),
        "core": frozenset(),
        "fields": frozenset({"core"}),
        "linalg": frozenset({"fields"}),
        "solvers": frozenset({"fields", "linalg"}),
        "coupling": frozenset({"solvers"}),
        "world": frozenset({"core", "fields"}),
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
    """(importer, imported) unit pairs that break the rules.

    `imports` maps each unit (see `collect_imports`) to the units it imports.
    A unit whose package is missing from `allowed` is reported as
    (unit, NOT_IN_SPEC), so a new package cannot slip in without its row in
    SPEC 8.1.
    """
    found: list[tuple[str, str]] = []
    for unit, targets in sorted(imports.items()):
        package = _package(unit)
        is_root = package in COMPOSITION_ROOTS
        if not is_root and package not in allowed:
            found.append((unit, NOT_IN_SPEC))
            continue
        for target in sorted(targets - {unit}):
            if _breaks_solver_isolation(unit, target):
                found.append((unit, target))
            elif _package(target) == package:
                continue
            elif target in COMPOSITION_ROOTS:
                if not is_root:
                    found.append((unit, target))
            elif not is_root and _package(target) not in reachable(package, allowed):
                found.append((unit, target))
    return found


def _package(unit: str) -> str:
    """`solvers.liquid` belongs to `solvers`; a top-level unit to itself."""
    return unit.split(".")[0]


def _breaks_solver_isolation(unit: str, target: str) -> bool:
    """A specific solver imported from another solver, the interface, the root or coupling;
    or the package root importing anything but the interface."""
    if unit == SOLVERS:
        return target != SOLVER_INTERFACE
    names_a_solver = target.startswith(f"{SOLVERS}.") and target != SOLVER_INTERFACE
    return names_a_solver and _package(unit) in (SOLVERS, "coupling")


def _module_names(directory: Path) -> set[str]:
    """Modules and subpackages directly in `directory`."""
    return {path.name.removesuffix(".py") for path in directory.iterdir()} - {
        "__init__",
        "__pycache__",
    }


def collect_imports(src: Path) -> dict[str, set[str]]:
    """Map each unit of `src` to the units it imports.

    Units are the top-level subpackages and modules, except inside `solvers`,
    where each solver is `solvers.<name>` and `solvers/__init__.py` is `solvers`.
    The package root `src/__init__.py` is reported under ROOT.
    """
    siblings = _module_names(src)
    solvers = _module_names(src / SOLVERS) if (src / SOLVERS).is_dir() else set()

    def unit_of(package: str, below: list[str]) -> str:
        """The unit of `entropy.<package>.<below...>`."""
        if package == SOLVERS and below and below[0] in solvers:
            return f"{SOLVERS}.{below[0]}"
        return package

    imports: dict[str, set[str]] = {}
    for path in src.rglob("*.py"):
        relative = path.relative_to(src)
        if relative.parts == ("__init__.py",):
            importer = ROOT
        else:
            package, *below = (part.removesuffix(".py") for part in relative.parts)
            importer = unit_of(package, below)
        targets = imports.setdefault(importer, set())
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
                    targets.add(unit_of(parts[1], parts[2:]))
    return imports


def test_upward_import_is_a_violation() -> None:
    assert violations({"core": {"render"}}, ALLOWED) == [("core", "render")]


def test_transitive_downward_import_is_allowed() -> None:
    assert violations({"solvers.liquid": {"core"}, "game": {"fields"}}, ALLOWED) == []


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


def test_a_solver_may_not_import_another_solver() -> None:
    """SPEC 8: solvers talk to each other only through the coupling layer."""
    assert violations({"solvers.liquid": {"solvers.gas"}}, ALLOWED) == [
        ("solvers.liquid", "solvers.gas")
    ]


def test_a_solver_may_import_the_interface_and_the_layers_below() -> None:
    imports = {"solvers.liquid": {"solvers", "solvers.interface", "linalg", "fields", "core"}}

    assert violations(imports, ALLOWED) == []


def test_the_interface_and_the_package_root_name_no_solver() -> None:
    imports = {"solvers.interface": {"solvers.rigid"}, "solvers": {"solvers.liquid"}}

    assert violations(imports, ALLOWED) == [
        ("solvers", "solvers.liquid"),
        ("solvers.interface", "solvers.rigid"),
    ]


def test_the_package_root_imports_only_the_interface() -> None:
    """A solver written in `solvers/__init__.py` with `fields` or `linalg` is caught."""
    imports = {"solvers": {"solvers.interface", "linalg", "fields"}}

    assert violations(imports, ALLOWED) == [("solvers", "fields"), ("solvers", "linalg")]


def test_coupling_may_import_only_the_solver_interface() -> None:
    """SPEC 8.1: `coupling` depends on solver interfaces only."""
    imports = {"coupling": {"solvers", "solvers.interface", "solvers.liquid"}}

    assert violations(imports, ALLOWED) == [("coupling", "solvers.liquid")]


def test_sim_may_import_any_solver() -> None:
    assert violations({"sim": {"solvers.liquid", "solvers.gas"}}, ALLOWED) == []


def test_a_solver_still_obeys_its_package_rules() -> None:
    assert violations({"solvers.liquid": {"render"}}, ALLOWED) == [("solvers.liquid", "render")]


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


def test_collect_imports_sees_each_solver_as_its_own_unit(tmp_path: Path) -> None:
    _write(tmp_path, "solvers/__init__.py", "from .interface import Solver\n")
    _write(tmp_path, "solvers/interface.py")
    _write(tmp_path, "solvers/gas.py", "from . import liquid\n")
    _write(tmp_path, "solvers/liquid/__init__.py", "from . import step\n")
    _write(
        tmp_path,
        "solvers/liquid/step.py",
        "from ..interface import Solver\n"
        "from entropy.solvers import gas\n"
        "import entropy.solvers.rigid\n",
    )
    _write(tmp_path, "solvers/rigid/__init__.py")
    _write(tmp_path, "coupling/a.py", "from entropy.solvers import Solver\n")

    imports = collect_imports(tmp_path)

    assert imports["solvers"] == {"solvers.interface"}
    assert imports["solvers.gas"] == {"solvers", "solvers.liquid"}
    assert imports["solvers.liquid"] == {
        "solvers",
        "solvers.interface",
        "solvers.liquid",  # `from . import step`: imports of its own modules, ignored later
        "solvers.gas",
        "solvers.rigid",
    }
    assert imports["coupling"] == {"solvers"}


def test_source_tree_respects_spec_8_1() -> None:
    imports = collect_imports(SRC)

    assert ROOT in imports, f"{SRC} has no package root: the check would see nothing"
    assert violations(imports, ALLOWED) == []
