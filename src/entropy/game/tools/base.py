"""What every tool shares (SPEC 4.2, 8.2): its content file, its aim and its context.

A tool is a class with `on_press(ctx)`, `on_hold(ctx, dt_s)` and `on_release(ctx)`,
registered under its name (`entropy.game.tools.registry`), plus
`content/tools/<name>.toml`: a label, an icon, its key (`order`), its parameters in SI
units with their limits and defaults, and which one the mouse wheel sets (`power`). `dt_s` is the
simulated time of the frame (slow motion gives less), so a tool's effect per
simulated second does not depend on the time scale.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np

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
    """What one notch of the mouse wheel adds, or multiplies by when `by_factor`."""
    by_factor: bool = False

    def __post_init__(self) -> None:
        if not (self.minimum <= self.default <= self.maximum and self.step > 0):
            raise ValueError(f"{self.name}: need minimum <= default <= maximum and step > 0")
        if self.by_factor and not (self.step > 1.0 and self.minimum > 0.0):
            raise ValueError(f"{self.name}: a factor step needs step > 1 and minimum > 0")

    def stepped(self, value: float, steps: int) -> float:
        """`value` moved by `steps` notches of the wheel, within the limits."""
        moved = value * self.step**steps if self.by_factor else value + steps * self.step
        return self.clamp(moved)

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
    order: int = 0
    """The key that chooses it, 1 to 9 (0 for a spec made in code)."""

    def __post_init__(self) -> None:
        if self.power not in self.parameters:
            raise ValueError(f"{self.name}: power {self.power} is not one of its parameters")


def load_tool_spec(name: str, folder: Path | None = None) -> ToolSpec:
    """The tool described by `<name>.toml` in `folder` (`content/tools` by default)."""
    path = (folder if folder is not None else content_dir() / "tools") / f"{name}.toml"
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
            by_factor=bool(value.get("by_factor", False)),
        )
        for key, value in raw.get("parameters", {}).items()
    }
    return ToolSpec(
        name=str(raw["name"]),
        label=str(raw["label"]),
        icon=str(raw["icon"]),
        power=str(raw["power"]),
        parameters=MappingProxyType(parameters),
        order=int(raw.get("order", 0)),
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

    def room_particles(self) -> int:
        """How many more particles the liquid holds."""
        ...

    def particle_count(self) -> int:
        """How many particles the liquid holds."""
        ...


@runtime_checkable
class ErasableLiquid(Protocol):
    """A liquid the eraser can take water from (#54)."""

    def fields(self) -> Mapping[str, Any]:
        """Its fields; the eraser reads `velocity` (for the grid) and `phi_m`."""
        ...

    def remove_particles_within(self, centre_m: Vec3, radius_m: float) -> int:
        """Remove the particles within `radius_m` of `centre_m`; how many went."""
        ...


class Inflow:
    """The liquid as tools see it: water past its room is lost, and counted.

    The grid holds at most its volume of water (a closed box in M1, ADR 0017). Water a
    tool adds beyond that would have spilled over the rim, which M1 cannot show; it is
    lost instead (named simplification), and `lost_m3` says how much. A tool that does
    more than add water (the eraser) acts on `liquid` itself."""

    def __init__(self, liquid: LiquidTarget, particle_volume_m3: float) -> None:
        self.liquid = liquid
        self.particle_volume_m3 = particle_volume_m3
        self.lost_m3 = 0.0

    def add_particles(self, positions_m: Any, *, velocities: Any = None) -> None:
        positions = np.asarray(positions_m).reshape(-1, 3)
        room = max(int(self.liquid.room_particles()), 0)
        if len(positions) > room:
            self.lost_m3 += (len(positions) - room) * self.particle_volume_m3
            positions = positions[:room]
            if velocities is not None:
                velocities = np.asarray(velocities).reshape(-1, 3)[:room]
        if len(positions):
            self.liquid.add_particles(positions, velocities=velocities)

    def room_particles(self) -> int:
        return self.liquid.room_particles()

    def particle_count(self) -> int:
        return self.liquid.particle_count()


@dataclass
class ToolContext:
    """What a tool may act on this frame."""

    aim: Aim
    liquid: LiquidTarget | None
    jet: Any
    """The `Jet` that carries water outside the liquid's grid."""
    particle_volume_m3: float
    domain: Any = None
    """The `Domain` where the liquid can hold water, or None without a liquid."""


class Tool(Protocol):
    spec: ToolSpec
    values: dict[str, float]

    def change_power(self, steps: int) -> None: ...

    def on_alternate(self) -> None: ...

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
        self.values[parameter.name] = parameter.stepped(self.values[parameter.name], steps)

    def on_alternate(self) -> None:
        """The tool's other action (right click, SPEC 4.5); none by default."""

    def on_press(self, ctx: ToolContext) -> None:
        pass

    def on_hold(self, ctx: ToolContext, dt_s: float) -> None:
        pass

    def on_release(self, ctx: ToolContext) -> None:
        pass
