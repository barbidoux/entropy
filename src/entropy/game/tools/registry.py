"""The tool registry (SPEC 8.2, #53): the core never names a specific tool.

A tool is two things: a class registered here under its name (`@register_tool`), and
`content/tools/<name>.toml`, whose `order` is the key that chooses it (1 to 9).
`discover_tools` reads every content file, in key order, and makes each tool from its
registered class. Every module of `entropy.game.tools` is imported first, so a tool
module's own decorator is all it takes to register it. The table is filled only as
modules are imported, never while the game runs (CLAUDE.md rule 6).
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from entropy.core.paths import content_dir
from entropy.game.tools.base import ToolSpec, load_tool_spec

_CLASSES: dict[str, Callable[[ToolSpec], Any]] = {}


def register_tool(name: str) -> Callable[[type], type]:
    """Class decorator: the tool made for `content/tools/<name>.toml`."""

    def register(cls: type) -> type:
        if _CLASSES.get(name, cls) is not cls:
            raise ValueError(f"a tool is already registered as {name!r}")
        _CLASSES[name] = cls
        return cls

    return register


def _import_tool_modules() -> None:
    package = importlib.import_module("entropy.game.tools")
    for module in pkgutil.iter_modules(package.__path__):
        importlib.import_module(f"{package.__name__}.{module.name}")


def discover_tools(folder: Path | None = None) -> list[Any]:
    """One tool per content file in `folder` (`content/tools` by default), in key order."""
    _import_tool_modules()
    folder = folder if folder is not None else content_dir() / "tools"
    specs = [load_tool_spec(path.stem, folder) for path in sorted(folder.glob("*.toml"))]
    orders = [spec.order for spec in specs]
    if len(set(orders)) != len(orders):
        raise ValueError(f"two tools share an order (a key): {sorted(orders)}")
    tools = []
    for spec in sorted(specs, key=lambda s: s.order):
        cls = _CLASSES.get(spec.name)
        if cls is None:
            raise ValueError(f"no tool class is registered for {spec.name!r}")
        tools.append(cls(spec))
    return tools
