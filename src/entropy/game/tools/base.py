"""What every tool shares (SPEC 4.2, 8.2): its content file, its aim and its context.

A tool is a class with `on_press(ctx)`, `on_hold(ctx, dt_s)` and `on_release(ctx)`,
plus `content/tools/<name>.toml`: a label, an icon, its parameters in SI units with
their limits and defaults, and which one the mouse wheel sets (`power`). `dt_s` is the
simulated time of the frame (slow motion gives less), so a tool's effect per
simulated second does not depend on the time scale.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from entropy.core.paths import content_dir

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class ToolParameter:
    """One setting of a tool, in SI units (`unit` says which)."""

    name: str
    label: str
    unit: str
    minimum: float
    maximum: float
    default: float
    step: float

    def __post_init__(self) -> None:
        if not (self.minimum <= self.default <= self.maximum and self.step > 0):
            raise ValueError(f"{self.name}: need minimum <= default <= maximum and step > 0")

    def clamp(self, value: float) -> float:
        return min(self.maximum, max(self.minimum, value))


@dataclass(frozen=True)
class ToolSpec:
    """A tool's content file."""

    name: str
    label: str
    icon: str
    power: str
    parameters: Mapping[str, ToolParameter] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.power not in self.parameters:
            raise ValueError(f"{self.name}: power {self.power} is not one of its parameters")


def load_tool_spec(name: str) -> ToolSpec:
    """The tool described by `content/tools/<name>.toml`."""
    path = content_dir() / "tools" / f"{name}.toml"
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    parameters = {
        key: ToolParameter(
            name=key,
            label=str(value["label"]),
            unit=str(value["unit"]),
            minimum=float(value["minimum"]),
            maximum=float(value["maximum"]),
            default=float(value["default"]),
            step=float(value["step"]),
        )
        for key, value in raw.get("parameters", {}).items()
    }
    return ToolSpec(
        name=str(raw["name"]),
        label=str(raw["label"]),
        icon=str(raw["icon"]),
        power=str(raw["power"]),
        parameters=MappingProxyType(parameters),
    )


@dataclass(frozen=True)
class Aim:
    """Where the player points: the camera's position (m) and a unit direction."""

    origin_m: Vec3
    direction: Vec3

    def __post_init__(self) -> None:
        length = math.sqrt(sum(c * c for c in self.direction))
        if not math.isclose(length, 1.0, rel_tol=1e-4):
            raise ValueError(f"direction must be a unit vector, got {self.direction}")


class LiquidTarget(Protocol):
    def add_particles(self, positions_m: Any, *, velocities: Any = None) -> None: ...


@dataclass
class ToolContext:
    """What a tool may act on this frame."""

    aim: Aim
    liquid: LiquidTarget | None
    jet: Any
    """The `Jet` that carries water outside the liquid's grid."""
    particle_volume_m3: float


class Tool(Protocol):
    spec: ToolSpec
    values: dict[str, float]

    def change_power(self, steps: int) -> None: ...

    def on_press(self, ctx: ToolContext) -> None: ...

    def on_hold(self, ctx: ToolContext, dt_s: float) -> None: ...

    def on_release(self, ctx: ToolContext) -> None: ...


class ToolBase:
    """Parameters at their defaults, and the mouse wheel on `power`."""

    def __init__(self, spec: ToolSpec) -> None:
        self.spec = spec
        self.values = {name: p.default for name, p in spec.parameters.items()}

    def change_power(self, steps: int) -> None:
        parameter = self.spec.parameters[self.spec.power]
        value = self.values[parameter.name] + steps * parameter.step
        self.values[parameter.name] = parameter.clamp(value)

    def on_press(self, ctx: ToolContext) -> None:
        pass

    def on_hold(self, ctx: ToolContext, dt_s: float) -> None:
        pass

    def on_release(self, ctx: ToolContext) -> None:
        pass
