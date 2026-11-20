"""God mode (`entropy.game.god_mode`): camera, HUD toggle and mouse capture from input."""

import math

import glfw
import pytest

from entropy.core.config import CameraSettings
from entropy.game.god_mode import GodMode

SETTINGS = CameraSettings(
    speed_m_per_s=5.0,
    mouse_rad_per_px=0.01,
    start_position_m=(1.0, 2.0, 3.0),
    start_yaw_deg=90.0,
    start_pitch_deg=-10.0,
)


def test_starts_where_the_settings_say() -> None:
    mode = GodMode.from_settings(SETTINGS)

    assert mode.pose.position_m == (1.0, 2.0, 3.0)
    assert mode.pose.yaw_rad == pytest.approx(math.pi / 2)
    assert mode.pose.pitch_rad == pytest.approx(math.radians(-10.0))
    assert mode.hud_visible
    assert not mode.input.mouse_captured


def test_holding_w_flies_forward_at_the_set_speed() -> None:
    mode = GodMode.from_settings(SETTINGS)
    level = GodMode.from_settings(
        CameraSettings(start_position_m=(0.0, 0.0, 2.0), start_yaw_deg=0.0, start_pitch_deg=0.0)
    )

    level.input.on_key(glfw.KEY_W, glfw.PRESS)
    level.update(0.5)
    mode.update(0.5)

    assert level.pose.position_m == pytest.approx((2.5, 0.0, 2.0))
    assert mode.pose.position_m == (1.0, 2.0, 3.0)


def test_f3_toggles_the_hud() -> None:
    mode = GodMode.from_settings(SETTINGS)

    mode.input.on_key(glfw.KEY_F3, glfw.PRESS)
    mode.update(1 / 60)
    hidden = mode.hud_visible
    mode.input.on_key(glfw.KEY_F3, glfw.RELEASE)
    mode.input.on_key(glfw.KEY_F3, glfw.PRESS)
    mode.update(1 / 60)

    assert (hidden, mode.hud_visible) == (False, True)


def test_a_click_captures_the_mouse_and_escape_releases_it() -> None:
    mode = GodMode.from_settings(SETTINGS)

    mode.on_mouse_button(pressed=True)
    captured = mode.input.mouse_captured
    mode.input.on_key(glfw.KEY_ESCAPE, glfw.PRESS)
    mode.update(1 / 60)

    assert captured
    assert not mode.input.mouse_captured


def test_the_mouse_turns_the_camera_only_while_captured() -> None:
    mode = GodMode.from_settings(SETTINGS)
    yaw = mode.pose.yaw_rad

    mode.input.on_mouse_move(0.0, 0.0)
    mode.input.on_mouse_move(10.0, 0.0)
    mode.update(1 / 60)
    released_yaw = mode.pose.yaw_rad
    mode.on_mouse_button(pressed=True)
    mode.input.on_mouse_move(0.0, 0.0)
    mode.input.on_mouse_move(10.0, 0.0)
    mode.update(1 / 60)

    assert released_yaw == yaw
    assert mode.pose.yaw_rad == pytest.approx(yaw - 0.1)
