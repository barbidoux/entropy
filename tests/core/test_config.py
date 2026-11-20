"""Settings: immutable dataclasses loaded from TOML (CLAUDE.md rule 6)."""

import dataclasses
from pathlib import Path

import pytest

from entropy.core.config import (
    CameraSettings,
    Settings,
    WindowSettings,
    default_settings_path,
    load_settings,
)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "settings.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_empty_file_gives_the_defaults(tmp_path: Path) -> None:
    assert load_settings(_write(tmp_path, "")) == Settings()


def test_values_override_defaults_section_by_section(tmp_path: Path) -> None:
    settings = load_settings(
        _write(
            tmp_path, "[window]\nwidth_px = 1280\nvsync = false\n[camera]\nspeed_m_per_s = 9.5\n"
        )
    )

    assert settings.window == dataclasses.replace(WindowSettings(), width_px=1280, vsync=False)
    assert settings.camera.speed_m_per_s == 9.5
    assert settings.camera.fov_y_deg == CameraSettings().fov_y_deg


def test_lists_become_tuples(tmp_path: Path) -> None:
    settings = load_settings(_write(tmp_path, "[camera]\nstart_position_m = [1.0, 2.0, 3.0]\n"))

    assert settings.camera.start_position_m == (1.0, 2.0, 3.0)


def test_integers_are_accepted_for_floats(tmp_path: Path) -> None:
    settings = load_settings(_write(tmp_path, "[camera]\nspeed_m_per_s = 4\n"))

    assert settings.camera.speed_m_per_s == 4.0
    assert isinstance(settings.camera.speed_m_per_s, float)


def test_unknown_keys_are_errors_naming_the_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"window\.widht_px"):
        load_settings(_write(tmp_path, "[window]\nwidht_px = 1280\n"))


def test_wrong_types_are_errors_naming_the_key(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match=r"window\.vsync"):
        load_settings(_write(tmp_path, '[window]\nvsync = "yes"\n'))


def test_tuple_length_is_checked(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match=r"camera\.start_position_m"):
        load_settings(_write(tmp_path, "[camera]\nstart_position_m = [1.0, 2.0]\n"))


def test_settings_are_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        Settings().window.width_px = 1  # type: ignore[misc]


def test_shipped_settings_file_loads() -> None:
    assert isinstance(load_settings(default_settings_path()), Settings)
