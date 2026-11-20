"""God mode free flight (`entropy.game.free_camera`), SPEC 4.1 and 4.5, Z up."""

import math

import numpy as np
import pytest

from entropy.game.free_camera import MAX_PITCH_RAD, CameraPose, FlightInput, fly

SPEED_M_PER_S = 5.0
MOUSE_RAD_PER_PX = 0.01
LEVEL = CameraPose(position_m=(0.0, 0.0, 2.0), yaw_rad=0.0, pitch_rad=0.0)


def _fly(pose: CameraPose, flight: FlightInput, dt_s: float = 1.0) -> CameraPose:
    return fly(
        pose,
        flight,
        dt_s=dt_s,
        speed_m_per_s=SPEED_M_PER_S,
        mouse_rad_per_px=MOUSE_RAD_PER_PX,
    )


@pytest.mark.parametrize(
    ("yaw_deg", "pitch_deg", "expected"),
    [(0, 0, (1, 0, 0)), (90, 0, (0, 1, 0)), (0, 90, (0, 0, 1)), (180, -90, (0, 0, -1))],
)
def test_forward_from_yaw_and_pitch(
    yaw_deg: float, pitch_deg: float, expected: tuple[float, float, float]
) -> None:
    pose = CameraPose((0.0, 0.0, 0.0), math.radians(yaw_deg), math.radians(pitch_deg))

    assert pose.forward() == pytest.approx(expected, abs=1e-9)


def test_forward_input_moves_along_the_view_at_the_set_speed() -> None:
    moved = _fly(LEVEL, FlightInput(forward=1.0), dt_s=0.5)

    assert moved.position_m == pytest.approx((2.5, 0.0, 2.0))


def test_strafing_right_moves_towards_minus_y_when_looking_along_x() -> None:
    """Z up, looking along +X: right-handed frame puts the right-hand side at -Y."""
    moved = _fly(LEVEL, FlightInput(right=1.0))

    assert moved.position_m == pytest.approx((0.0, -5.0, 2.0))


def test_up_input_moves_along_world_z_whatever_the_pitch() -> None:
    looking_down = CameraPose((0.0, 0.0, 2.0), 0.3, -1.0)

    moved = _fly(looking_down, FlightInput(up=1.0))

    assert moved.position_m == pytest.approx((0.0, 0.0, 7.0))


def test_diagonal_input_is_not_faster() -> None:
    moved = _fly(LEVEL, FlightInput(forward=1.0, right=1.0))

    displacement = np.subtract(moved.position_m, LEVEL.position_m)
    assert np.linalg.norm(displacement) == pytest.approx(SPEED_M_PER_S)


def test_motion_does_not_depend_on_the_frame_rate() -> None:
    flight = FlightInput(forward=1.0, up=0.5)

    one_step = _fly(LEVEL, flight, dt_s=1 / 30)
    two_steps = _fly(_fly(LEVEL, flight, dt_s=1 / 60), flight, dt_s=1 / 60)

    assert two_steps.position_m == pytest.approx(one_step.position_m)


def test_mouse_right_turns_right_and_mouse_up_looks_up() -> None:
    """Screen coordinates grow rightwards and downwards, as GLFW reports them."""
    turned = _fly(LEVEL, FlightInput(look_dx_px=10.0, look_dy_px=-10.0), dt_s=0.0)

    assert turned.yaw_rad == pytest.approx(-0.1)
    assert turned.pitch_rad == pytest.approx(0.1)


def test_pitch_is_clamped_so_the_view_never_flips() -> None:
    up = _fly(LEVEL, FlightInput(look_dy_px=-1e6), dt_s=0.0)
    down = _fly(LEVEL, FlightInput(look_dy_px=1e6), dt_s=0.0)

    assert up.pitch_rad == pytest.approx(MAX_PITCH_RAD)
    assert down.pitch_rad == pytest.approx(-MAX_PITCH_RAD)
    assert math.pi / 2 > MAX_PITCH_RAD


def test_poses_are_immutable() -> None:
    with pytest.raises(AttributeError):
        LEVEL.yaw_rad = 1.0  # type: ignore[misc]
