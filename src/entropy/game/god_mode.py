"""God mode (SPEC 4.1): free camera, HUD toggle and mouse capture, driven by input.

The mouse is captured by a click in the window and released by Esc (the menu
key of SPEC 4.5, which has no menu yet), so the game never grabs the cursor
by surprise.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from entropy.core.config import CameraSettings
from entropy.game.free_camera import CameraPose, fly
from entropy.game.input import Action, FrameInput, InputState, flight_input


@dataclass
class GodMode:
    settings: CameraSettings
    pose: CameraPose
    input: InputState = field(default_factory=InputState)
    hud_visible: bool = True
    last_frame: FrameInput | None = None
    """The input of the last update, for the tools."""

    @classmethod
    def from_settings(
        cls, settings: CameraSettings, bindings: Mapping[int, Action] | None = None
    ) -> GodMode:
        pose = CameraPose(
            settings.start_position_m,
            math.radians(settings.start_yaw_deg),
            math.radians(settings.start_pitch_deg),
        )
        state = InputState(bindings) if bindings is not None else InputState()
        return cls(settings=settings, pose=pose, input=state)

    def on_mouse_button(self, *, pressed: bool) -> None:
        """A click in the window captures the mouse for looking around."""
        if pressed:
            self.input.set_mouse_captured(True)

    def update(self, frame_dt_s: float) -> None:
        """Apply this frame's input; `frame_dt_s` is real time (SPEC 4.6)."""
        frame = self.input.take_frame()
        self.last_frame = frame
        if Action.TOGGLE_HUD in frame.pressed:
            self.hud_visible = not self.hud_visible
        if Action.MENU in frame.pressed:
            self.input.set_mouse_captured(False)
        self.pose = fly(
            self.pose,
            flight_input(frame),
            dt_s=frame_dt_s,
            speed_m_per_s=self.settings.speed_m_per_s,
            mouse_rad_per_px=self.settings.mouse_rad_per_px,
        )
