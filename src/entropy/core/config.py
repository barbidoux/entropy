"""Settings: immutable dataclasses loaded from TOML (CLAUDE.md rule 6).

Each TOML table maps to one dataclass field of `Settings`; missing keys keep
their defaults, unknown keys and wrong types are errors that name the key.
"""

from __future__ import annotations

import dataclasses
import tomllib
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from entropy.core.paths import content_dir


@dataclass(frozen=True)
class WindowSettings:
    """The game window and its OpenGL context."""

    width_px: int = 1600
    height_px: int = 900
    title: str = "ENTROPY"
    vsync: bool = True
    gl_version: tuple[int, int] = (4, 6)


@dataclass(frozen=True)
class CameraSettings:
    """The free camera of God mode (SPEC 4.1, 4.5)."""

    fov_y_deg: float = 70.0
    near_m: float = 0.05
    far_m: float = 1000.0
    speed_m_per_s: float = 5.0
    mouse_rad_per_px: float = 0.0025
    start_position_m: tuple[float, float, float] = (-8.0, -10.0, 4.0)
    start_yaw_deg: float = 50.0
    start_pitch_deg: float = -15.0


@dataclass(frozen=True)
class Settings:
    window: WindowSettings = field(default_factory=WindowSettings)
    camera: CameraSettings = field(default_factory=CameraSettings)


def default_settings_path() -> Path:
    """`content/settings.toml`, the settings shipped with the game."""
    return content_dir() / "settings.toml"


def load_settings(path: Path) -> Settings:
    """Read `path` and return the settings it describes over the defaults."""
    with path.open("rb") as file:
        data = tomllib.load(file)
    return _build(Settings, data, prefix="")


def _build[T](cls: type[T], data: dict[str, Any], prefix: str) -> T:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(f"unknown setting {prefix}{unknown[0]}")
    values: dict[str, Any] = {}
    for name, value in data.items():
        key = f"{prefix}{name}"
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            if not isinstance(value, dict):
                raise TypeError(f"setting {key} must be a table")
            values[name] = _build(hint, value, prefix=f"{key}.")  # type: ignore[arg-type]
        else:
            values[name] = _convert(value, hint, key)
    return cls(**values)


def _convert(value: Any, hint: Any, key: str) -> Any:
    if typing.get_origin(hint) is tuple:
        items = typing.get_args(hint)
        if not isinstance(value, list) or len(value) != len(items):
            raise TypeError(f"setting {key} must be a list of {len(items)} values")
        return tuple(_convert(item, t, key) for item, t in zip(value, items, strict=True))
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if hint is int and isinstance(value, bool):
        raise TypeError(f"setting {key} must be {hint.__name__}")
    if not isinstance(value, hint):
        raise TypeError(f"setting {key} must be {hint.__name__}, not {type(value).__name__}")
    return value
