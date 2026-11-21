"""Materials: one TOML file each in `content/materials/`, every constant with its source.

SPEC 5.1 and CLAUDE.md rule 4: no material constant lives in code. A file has a
`name`, its constants in SI units (the unit is in the key) and a `[source]`
table citing where each constant comes from. A constant without a source, an
unknown key or an impossible value is an error naming the file and the key.

M1 needs the density only; constants are added here as solvers need them.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from entropy.core.paths import content_dir

CONSTANTS = ("density_kg_per_m3",)
"""Constant keys a material file may hold."""


@dataclass(frozen=True)
class Material:
    """A material's constants in SI units, and the source of each."""

    name: str
    density_kg_per_m3: float
    sources: Mapping[str, str]


def parse_material(text: str, *, origin: str) -> Material:
    """A material from TOML `text`; `origin` names it in errors (usually the file)."""
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"{origin}: {error}") from error
    sources = raw.pop("source", {})
    if not isinstance(sources, dict):
        raise ValueError(f"{origin}: [source] must be a table of key = citation")
    name = raw.pop("name", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{origin}: name must be a non-empty string")
    for key in raw:
        if key not in CONSTANTS:
            raise ValueError(f"{origin}: unknown key {key}")
    for key in CONSTANTS:
        if key not in raw:
            raise ValueError(f"{origin}: {key} is missing")
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{origin}: {key} must be a number, got {value!r}")
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{origin}: {key} must be positive and finite, got {value}")
        source = sources.get(key)
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"{origin}: {key} has no source")
    unused = set(sources) - set(CONSTANTS)
    if unused:
        raise ValueError(f"{origin}: sources for unknown keys {sorted(unused)}")
    return Material(
        name=name,
        density_kg_per_m3=float(raw["density_kg_per_m3"]),
        sources=MappingProxyType(dict(sources)),
    )


def load_material(name: str) -> Material:
    """The material in `content/materials/<name>.toml`."""
    path = content_dir() / "materials" / f"{name}.toml"
    if not path.is_file():
        raise FileNotFoundError(f"no material file {path}")
    return parse_material(path.read_text(encoding="utf-8"), origin=str(path))
