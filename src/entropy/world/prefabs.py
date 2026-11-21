"""Prefabs: what a map entity's `prefab` id builds (SPEC 4.3, 8.2).

A prefab is a builder function that turns an entity's position and parameters
into solid shapes. The registry is code for now; SPEC 8.2's content file per
prefab (its parameters, limits and icon, for the Build mode library) comes with
Build mode in M6.

M1 has two prefabs:
- `glass_tank`: `position_m` is the centre of the cavity's base. Parameters:
  `inner_size_m` (x, y, z), `wall_m`.
- `floor_slab`: `position_m` is the centre of its top face. Parameters:
  `size_m` (x, y), `thickness_m`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from entropy.world.maps import Entity
from entropy.world.sdf import Box, OpenBox, Shape

Builder = Callable[[Entity, str], tuple[Shape, ...]]
"""Builds an entity's shapes; the str names the entity in errors."""


def _numbers(entity: Entity, key: str, count: int, where: str) -> tuple[float, ...]:
    value = entity.parameters.get(key)
    values: Any = (value,) if count == 1 else value
    numbers: list[float] = []
    if isinstance(values, tuple) and len(values) == count:
        numbers = [
            float(v)
            for v in values
            if isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v) and v > 0
        ]
    if len(numbers) != count:
        raise ValueError(
            f"{where}: parameter {key} must be {count} positive number(s), got {value!r}"
        )
    return tuple(numbers)


def glass_tank(entity: Entity, where: str) -> tuple[Shape, ...]:
    sx, sy, sz = _numbers(entity, "inner_size_m", 3, where)
    (wall,) = _numbers(entity, "wall_m", 1, where)
    x, y, z = entity.position_m
    return (
        OpenBox(
            inner_min_m=(x - sx / 2, y - sy / 2, z),
            inner_max_m=(x + sx / 2, y + sy / 2, z + sz),
            wall_m=wall,
        ),
    )


def floor_slab(entity: Entity, where: str) -> tuple[Shape, ...]:
    sx, sy = _numbers(entity, "size_m", 2, where)
    (thickness,) = _numbers(entity, "thickness_m", 1, where)
    x, y, z = entity.position_m
    return (Box(centre_m=(x, y, z - thickness / 2), half_size_m=(sx / 2, sy / 2, thickness / 2)),)


PREFABS: Mapping[str, Builder] = MappingProxyType(
    {"glass_tank": glass_tank, "floor_slab": floor_slab}
)


def build(entity: Entity, where: str, prefabs: Mapping[str, Builder] = PREFABS) -> tuple[Any, ...]:
    """The shapes of `entity`; `where` names it in errors."""
    builder = prefabs.get(entity.prefab)
    if builder is None:
        raise ValueError(f"{where}: unknown prefab {entity.prefab}")
    return builder(entity, where)
