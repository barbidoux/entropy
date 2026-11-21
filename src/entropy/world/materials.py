"""Materials: one TOML file each in `content/materials/`, every constant with its source.

SPEC 5.1 and CLAUDE.md rule 4: no material constant lives in code. A file has a
`name`, its constants in SI units (the unit is in the key) and a `[source]`
table citing where each constant comes from. A constant without a source, an
unknown key or an impossible value is an error naming the file and the key.

The density is required. Optional constants serve the renderer (SPEC 7.2): the
refractive index (Fresnel reflection, refraction), a linear-RGB albedo, and
Beer-Lambert absorption per metre for the three sRGB primaries. Constants are
added here as solvers and passes need them.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from entropy.core.paths import content_dir

CONSTANTS = ("density_kg_per_m3",)
"""Constant keys every material file holds."""
OPTIONAL = ("refractive_index", "albedo_linear_rgb", "absorption_per_m_rgb")
"""Constant keys a material file may hold."""

Rgb = tuple[float, float, float]


@dataclass(frozen=True)
class Material:
    """A material's constants in SI units, and the source of each."""

    name: str
    density_kg_per_m3: float
    sources: Mapping[str, str]
    refractive_index: float | None = None
    """Real part, dimensionless, for visible light."""
    albedo_linear_rgb: Rgb | None = None
    """Diffuse reflectance in linear sRGB, each in [0, 1]."""
    absorption_per_m_rgb: Rgb | None = None
    """Beer-Lambert absorption coefficient (1/m) for red, green and blue light."""


def _number(value: object, origin: str, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{origin}: {key} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{origin}: {key} must be finite, got {value}")
    return float(value)


def _rgb(value: object, origin: str, key: str, low: float, high: float) -> Rgb:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{origin}: {key} must be three numbers (red, green, blue)")
    r, g, b = (_number(v, origin, key) for v in value)
    if not all(low <= c <= high for c in (r, g, b)):
        raise ValueError(f"{origin}: {key} must lie in [{low}, {high}], got {value}")
    return (r, g, b)


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
        if key not in CONSTANTS + OPTIONAL:
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
    for key in OPTIONAL:
        source = sources.get(key)
        if key in raw and (not isinstance(source, str) or not source.strip()):
            raise ValueError(f"{origin}: {key} has no source")
    unused = set(sources) - set(raw)
    if unused:
        raise ValueError(f"{origin}: sources for keys the file does not set {sorted(unused)}")
    index = None
    if "refractive_index" in raw:
        index = _number(raw["refractive_index"], origin, "refractive_index")
        if index < 1.0:
            raise ValueError(f"{origin}: refractive_index must be at least 1, got {index}")
    albedo = absorption = None
    if "albedo_linear_rgb" in raw:
        albedo = _rgb(raw["albedo_linear_rgb"], origin, "albedo_linear_rgb", 0.0, 1.0)
    if "absorption_per_m_rgb" in raw:
        absorption = _rgb(
            raw["absorption_per_m_rgb"], origin, "absorption_per_m_rgb", 0.0, math.inf
        )
    return Material(
        name=name,
        density_kg_per_m3=float(raw["density_kg_per_m3"]),
        sources=MappingProxyType(dict(sources)),
        refractive_index=index,
        albedo_linear_rgb=albedo,
        absorption_per_m_rgb=absorption,
    )


def load_material(name: str) -> Material:
    """The material in `content/materials/<name>.toml`."""
    path = content_dir() / "materials" / f"{name}.toml"
    if not path.is_file():
        raise FileNotFoundError(f"no material file {path}")
    return parse_material(path.read_text(encoding="utf-8"), origin=str(path))
