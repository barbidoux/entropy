"""God mode free flight (SPEC 4.1, 4.5) in the Z-up world frame (ADR 0003).

Yaw is measured from +X towards +Y, pitch upwards from the horizontal plane.
W and S fly along the view direction, A and D strafe horizontally, Space and
Ctrl move along world Z. Speeds are in m/s, so motion does not depend on the
frame rate.
"""

import math
from dataclasses import dataclass

import numpy as np

MAX_PITCH_RAD = math.radians(89.0)
"""Pitch limit, short of straight up or down, so the view never flips."""


@dataclass(frozen=True)
class CameraPose:
    """Where the camera is (m) and where it looks (rad)."""

    position_m: tuple[float, float, float]
    yaw_rad: float
    pitch_rad: float

    def forward(self) -> np.ndarray:
        """Unit view direction."""
        cos_pitch = math.cos(self.pitch_rad)
        return np.array(
            [
                cos_pitch * math.cos(self.yaw_rad),
                cos_pitch * math.sin(self.yaw_rad),
                math.sin(self.pitch_rad),
            ]
        )

    def right(self) -> np.ndarray:
        """Unit horizontal direction to the right of the view."""
        return np.array([math.sin(self.yaw_rad), -math.cos(self.yaw_rad), 0.0])


@dataclass(frozen=True)
class FlightInput:
    """One frame of flight controls.

    `forward`, `right` and `up` are in [-1, 1]; mouse motion is in screen
    pixels, x to the right and y downwards, as GLFW reports it.
    """

    forward: float = 0.0
    right: float = 0.0
    up: float = 0.0
    look_dx_px: float = 0.0
    look_dy_px: float = 0.0


def fly(
    pose: CameraPose,
    flight: FlightInput,
    *,
    dt_s: float,
    speed_m_per_s: float,
    mouse_rad_per_px: float,
) -> CameraPose:
    """The pose after `dt_s` seconds of `flight`: turn first, then move."""
    yaw_rad = pose.yaw_rad - flight.look_dx_px * mouse_rad_per_px
    pitch_rad = pose.pitch_rad - flight.look_dy_px * mouse_rad_per_px
    pitch_rad = max(-MAX_PITCH_RAD, min(MAX_PITCH_RAD, pitch_rad))
    turned = CameraPose(pose.position_m, yaw_rad, pitch_rad)

    direction = (
        flight.forward * turned.forward()
        + flight.right * turned.right()
        + flight.up * np.array([0.0, 0.0, 1.0])
    )
    length = float(np.linalg.norm(direction))
    if length > 1.0:
        direction /= length  # diagonals are not faster
    position = np.asarray(pose.position_m) + direction * speed_m_per_s * dt_s
    return CameraPose(
        (float(position[0]), float(position[1]), float(position[2])), yaw_rad, pitch_rad
    )
