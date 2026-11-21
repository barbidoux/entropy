"""Map files, format v1 (SPEC 4.3): a folder of plain text under version control.

- `map.toml`: `schema_version`, `name`, `description`, and an `[environment]` table
  (gravity in m/s², air temperature in K).
- `scene.json`: `entities`, each with a `prefab` id, a `position_m`, a `material`
  (a file in `content/materials/`) and the prefab's `parameters`.

Old maps load forever. The loader migrates any older `schema_version` one step
at a time through `MIGRATIONS`, and each migration has a test. Version 1 is the
first format, so there is no migration yet; the registry is tested with a fake v0.
A map newer than this code is refused rather than guessed at.

What a prefab id means is `entropy.world.prefabs`' business. This module only
reads and writes the files.
"""

from __future__ import annotations

import json
import math
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

SCHEMA_VERSION = 1

Migration = Callable[[dict[str, Any]], dict[str, Any]]
"""Takes {"map": <map.toml>, "scene": <scene.json>} at version n, returns it at n + 1."""

MIGRATIONS: Mapping[int, Migration] = MappingProxyType({})


@dataclass(frozen=True)
class Environment:
    gravity_m_per_s2: tuple[float, float, float] = (0.0, 0.0, -9.81)
    air_temperature_k: float = 293.15


@dataclass(frozen=True)
class Entity:
    """One placed thing: a prefab at a position, made of a material, with parameters."""

    prefab: str
    position_m: tuple[float, float, float]
    material: str
    parameters: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class MapData:
    name: str
    schema_version: int
    description: str
    environment: Environment
    entities: tuple[Entity, ...]


def _frozen(value: Any) -> Any:
    """Lists become tuples and dicts read-only mappings, so maps compare and stay immutable."""
    if isinstance(value, list | tuple):
        return tuple(_frozen(v) for v in value)
    if isinstance(value, dict):
        return MappingProxyType({k: _frozen(v) for k, v in value.items()})
    return value


def _thawed(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_thawed(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _thawed(v) for k, v in value.items()}
    return value


def _vec3(value: Any, where: str) -> tuple[float, float, float]:
    if (
        not isinstance(value, list | tuple)
        or len(value) != 3
        or not all(isinstance(c, int | float) and not isinstance(c, bool) for c in value)
        or not all(math.isfinite(c) for c in value)
    ):
        raise ValueError(f"{where} must be three finite numbers, got {value!r}")
    return (float(value[0]), float(value[1]), float(value[2]))


def _read(folder: Path) -> dict[str, Any]:
    if not folder.is_dir():
        raise FileNotFoundError(f"no map folder {folder}")
    meta_path, scene_path = folder / "map.toml", folder / "scene.json"
    try:
        meta = tomllib.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"{meta_path}: {error}") from error
    try:
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{scene_path}: {error}") from error
    return {"map": meta, "scene": scene}


def _migrate(
    raw: dict[str, Any], migrations: Mapping[int, Migration], folder: Path
) -> dict[str, Any]:
    version = raw["map"].get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise ValueError(f"{folder}: schema_version must be a whole number, got {version!r}")
    if version > SCHEMA_VERSION:
        raise ValueError(f"{folder}: schema {version} is newer than this game ({SCHEMA_VERSION})")
    while version < SCHEMA_VERSION:
        step = migrations.get(version)
        if step is None:
            raise ValueError(f"{folder}: no migration from schema {version}")
        raw = step(raw)
        version += 1
        raw["map"]["schema_version"] = version
    return raw


def _parse(raw: dict[str, Any], folder: Path) -> MapData:
    meta, scene = raw["map"], raw["scene"]
    env = meta.get("environment", {})
    environment = Environment(
        gravity_m_per_s2=_vec3(
            env.get("gravity_m_per_s2", list(Environment.gravity_m_per_s2)),
            f"{folder}: environment.gravity_m_per_s2",
        ),
        air_temperature_k=float(env.get("air_temperature_k", Environment.air_temperature_k)),
    )
    if not (math.isfinite(environment.air_temperature_k) and environment.air_temperature_k > 0):
        raise ValueError(f"{folder}: environment.air_temperature_k must be a positive temperature")
    entities = []
    for index, item in enumerate(scene.get("entities", [])):
        where = f"{folder}: entity {index}"
        if not isinstance(item, dict) or not isinstance(item.get("prefab"), str):
            raise ValueError(f"{where} needs a prefab id")
        if not isinstance(item.get("material"), str):
            raise ValueError(f"{where} needs a material")
        entities.append(
            Entity(
                prefab=item["prefab"],
                position_m=_vec3(item.get("position_m"), f"{where}: position_m"),
                material=item["material"],
                parameters=_frozen(item.get("parameters", {})),
            )
        )
    name = meta.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{folder}: name must be a non-empty string")
    return MapData(
        name=name,
        schema_version=meta["schema_version"],
        description=str(meta.get("description", "")),
        environment=environment,
        entities=tuple(entities),
    )


def load_map(folder: Path, *, migrations: Mapping[int, Migration] = MIGRATIONS) -> MapData:
    """The map in `folder`, migrated to the current schema."""
    folder = Path(folder)
    return _parse(_migrate(_read(folder), migrations, folder), folder)


def _toml_value(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)  # a JSON string is a valid TOML basic string
    if isinstance(value, tuple | list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return repr(float(value)) if isinstance(value, float) else str(value)


def save_map(data: MapData, folder: Path) -> None:
    """Write `data` as `map.toml` and `scene.json` in `folder` (created if needed)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    env = data.environment
    meta = "\n".join(
        [
            f"schema_version = {data.schema_version}",
            f"name = {_toml_value(data.name)}",
            f"description = {_toml_value(data.description)}",
            "",
            "[environment]",
            f"gravity_m_per_s2 = {_toml_value(env.gravity_m_per_s2)}",
            f"air_temperature_k = {_toml_value(env.air_temperature_k)}",
            "",
        ]
    )
    (folder / "map.toml").write_text(meta, encoding="utf-8")
    scene = {
        "entities": [
            {
                "prefab": e.prefab,
                "position_m": list(e.position_m),
                "material": e.material,
                "parameters": _thawed(e.parameters),
            }
            for e in data.entities
        ]
    }
    (folder / "scene.json").write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")
