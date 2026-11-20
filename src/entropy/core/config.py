"""Settings: immutable dataclasses loaded from TOML (CLAUDE.md rule 6).

Each TOML table maps to one dataclass field of `Settings`; missing keys keep
their defaults, unknown keys, wrong types and impossible values are errors
that name the key. The defaults below are the reference; `content/settings.toml`
repeats them so Matt can see and change them.

The loader handles scalars, nested tables and fixed-length tuples, which is
all the settings use; other field types would need loader support first.
"""

from __future__ import annotations

import dataclasses
import math
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

    def __post_init__(self) -> None:
        _require(self.width_px > 0, "width_px must be positive")
        _require(self.height_px > 0, "height_px must be positive")
        _require(self.gl_version >= (3, 2), "gl_version must be 3.2 or later (core profile)")


@dataclass(frozen=True)
class CameraSettings:
    """The free camera of God mode (SPEC 4.1, 4.5)."""

    fov_y_deg: float = 70.0
    near_m: float = 0.05
    far_m: float = 1000.0
    speed_m_per_s: float = 5.0
    mouse_rad_per_px: float = 0.0025
    start_position_m: tuple[float, float, float] = (8.0, -9.0, 4.0)
    start_yaw_deg: float = 130.0
    start_pitch_deg: float = -15.0

    def __post_init__(self) -> None:
        _require(0.0 < self.fov_y_deg < 180.0, "fov_y_deg must be between 0 and 180")
        _require(self.near_m > 0.0, "near_m must be positive")
        _require(self.near_m < self.far_m < math.inf, "far_m must be finite and beyond near_m")
        _require(0.0 < self.speed_m_per_s < math.inf, "speed_m_per_s must be positive")
        _require(
            0.0 <= self.mouse_rad_per_px < math.inf, "mouse_rad_per_px must be zero or positive"
        )
        _require(
            all(math.isfinite(c) for c in self.start_position_m), "start_position_m must be finite"
        )
        _require(math.isfinite(self.start_yaw_deg), "start_yaw_deg must be finite")
        _require(abs(self.start_pitch_deg) <= 89.0, "start_pitch_deg must be within ±89")


@dataclass(frozen=True)
class ControlsSettings:
    """Keys for each action (SPEC 4.5), issue #32.

    A single letter or digit is the label printed on the player's keyboard, so
    "Z" means the key marked Z whatever the layout; other keys are named as in
    GLFW without the KEY_ prefix (SPACE, LEFT_CONTROL, F3, ESCAPE, PAGE_UP...).
    Several keys for one action are separated by commas. The defaults are for a
    QWERTY keyboard; `content/settings.toml` sets Matt's AZERTY keys.
    """

    forward: str = "W"
    back: str = "S"
    left: str = "A"
    right: str = "D"
    up: str = "SPACE"
    down: str = "LEFT_CONTROL, RIGHT_CONTROL"
    toggle_hud: str = "F3"
    menu: str = "ESCAPE"


@dataclass(frozen=True)
class Settings:
    """Everything `content/settings.toml` can set, one table per section."""

    window: WindowSettings = field(default_factory=WindowSettings)
    camera: CameraSettings = field(default_factory=CameraSettings)
    controls: ControlsSettings = field(default_factory=ControlsSettings)


def _require(condition: bool, message: str) -> None:
    # Comparisons with NaN are false, so NaN fails every range check here.
    if not condition:
        raise ValueError(message)


def default_settings_path() -> Path:
    """`content/settings.toml`, the settings shipped with the game."""
    return content_dir() / "settings.toml"


def load_settings(path: Path) -> Settings:
    """Read `path` and return the settings it describes over the defaults."""
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
        return _build(Settings, data, prefix="")
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"{path}: {error}") from error


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
    try:
        return cls(**values)
    except ValueError as error:
        raise ValueError(f"{prefix}{error}") from error


def _convert(value: Any, hint: Any, key: str) -> Any:
    if typing.get_origin(hint) is tuple:
        items = typing.get_args(hint)
        if not isinstance(value, list) or len(value) != len(items):
            raise TypeError(f"setting {key} must be a list of {len(items)} values")
        return tuple(_convert(item, t, key) for item, t in zip(value, items, strict=True))
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if hint is int and isinstance(value, bool):
        raise TypeError(f"setting {key} must be int, not bool")
    if not isinstance(value, hint):
        raise TypeError(f"setting {key} must be {hint.__name__}, not {type(value).__name__}")
    return value
