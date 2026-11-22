"""Time control (SPEC 4.5, 4.6, ADR 0008, #55): pause, slow motion and single step.

The player cycles the time scale through 1x, 0.5x, 0.25x and 0.1x, pauses and resumes,
and while paused steps one frame at the chosen scale. The frame loop asks
`frame()` once per frame; game logic and the camera keep running on real
time whatever the scale, so the camera stays free while the world is frozen.
"""

from __future__ import annotations

SCALES = (1.0, 0.5, 0.25, 0.1)
"""Time scales the player cycles through (SPEC 4.5)."""


class TimeControl:
    def __init__(self) -> None:
        self._index = 0
        self.paused = False
        self.step_requested = False

    @property
    def scale(self) -> float:
        """The chosen time scale, paused or not."""
        return SCALES[self._index]

    def slower(self) -> None:
        self._index = (self._index + 1) % len(SCALES)

    def faster(self) -> None:
        self._index = (self._index - 1) % len(SCALES)

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.step_requested = False

    def step_once(self) -> None:
        """While paused, the next frame advances one frame at the chosen scale."""
        self.step_requested = self.paused

    def frame(self) -> tuple[float, bool]:
        """(scale, single step) for this frame: the scale is 0 while paused, except for a
        requested single step, which the loop turns into one 60 fps frame's worth."""
        if not self.paused:
            return self.scale, False
        if self.step_requested:
            self.step_requested = False
            return self.scale, True
        return 0.0, False

    def hud_line(self) -> str:
        scale = f"x{self.scale:g}"
        if self.paused:
            return f"temps : EN PAUSE ({scale})"
        if self.scale < 1.0:
            return f"temps : {scale} (ralenti)"
        return f"temps : {scale}"
