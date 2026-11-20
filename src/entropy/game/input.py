"""Input mapping: GLFW keys and mouse to game actions (SPEC 4.5).

GLFW calls `on_key` and `on_mouse_move` as events arrive; the game loop takes
one `FrameInput` per frame. Held actions persist across frames; presses are
reported once, in the frame they happened, even if the key was released in
the same frame. An action stays held while any key bound to it is down.

Keys are GLFW key tokens, which name physical key positions: on an AZERTY
keyboard W A S D are the keys labelled Z Q S D. Mouse motion is counted only
while the mouse is captured; with raw motion on (ADR 0004) the units are mouse
counts, so the look sensitivity depends on the mouse's resolution.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

import glfw

from entropy.game.free_camera import FlightInput


class Action(StrEnum):
    FORWARD = "forward"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    TOGGLE_HUD = "toggle_hud"
    MENU = "menu"


DEFAULT_BINDINGS: Mapping[int, Action] = MappingProxyType(
    {
        glfw.KEY_W: Action.FORWARD,
        glfw.KEY_S: Action.BACK,
        glfw.KEY_A: Action.LEFT,
        glfw.KEY_D: Action.RIGHT,
        glfw.KEY_SPACE: Action.UP,
        glfw.KEY_LEFT_CONTROL: Action.DOWN,
        glfw.KEY_RIGHT_CONTROL: Action.DOWN,
        glfw.KEY_F3: Action.TOGGLE_HUD,
        glfw.KEY_ESCAPE: Action.MENU,
    }
)


@dataclass(frozen=True)
class FrameInput:
    """Everything the player did during one frame."""

    held: frozenset[Action]
    pressed: frozenset[Action]
    look_dx_px: float
    look_dy_px: float


class InputState:
    """Collects input events between frames."""

    def __init__(self, bindings: Mapping[int, Action] = DEFAULT_BINDINGS) -> None:
        self.bindings = bindings
        self._mouse_captured = False
        self._held_keys: set[int] = set()
        self._pressed: set[Action] = set()
        self._cursor_px: tuple[float, float] | None = None
        self._dx_px = 0.0
        self._dy_px = 0.0

    def on_key(self, key: int, event: int) -> None:
        """A GLFW key event (`glfw.PRESS`, `glfw.RELEASE` or `glfw.REPEAT`)."""
        action = self.bindings.get(key)
        if action is None:
            return
        if event == glfw.PRESS:
            self._held_keys.add(key)
            self._pressed.add(action)
        elif event == glfw.RELEASE:
            self._held_keys.discard(key)

    @property
    def mouse_captured(self) -> bool:
        return self._mouse_captured

    def set_mouse_captured(self, captured: bool) -> None:
        """Follow the window's mouse capture; call both together.

        On capture the next cursor position becomes the reference, so the view
        does not jump to wherever GLFW's virtual cursor restarts.
        """
        if captured and not self._mouse_captured:
            self._cursor_px = None
        self._mouse_captured = captured

    def on_mouse_move(self, x_px: float, y_px: float) -> None:
        """A GLFW cursor position; motion counts only while the mouse is captured."""
        if self._cursor_px is not None and self._mouse_captured:
            self._dx_px += x_px - self._cursor_px[0]
            self._dy_px += y_px - self._cursor_px[1]
        self._cursor_px = (x_px, y_px)

    def take_frame(self) -> FrameInput:
        """This frame's input; presses and mouse motion start again from zero."""
        frame = FrameInput(
            held=frozenset(self.bindings[key] for key in self._held_keys),
            pressed=frozenset(self._pressed),
            look_dx_px=self._dx_px,
            look_dy_px=self._dy_px,
        )
        self._pressed.clear()
        self._dx_px = self._dy_px = 0.0
        return frame


def flight_input(frame: FrameInput) -> FlightInput:
    """Free-flight controls from the held movement keys and the mouse."""

    def axis(positive: Action, negative: Action) -> float:
        return float(positive in frame.held) - float(negative in frame.held)

    return FlightInput(
        forward=axis(Action.FORWARD, Action.BACK),
        right=axis(Action.RIGHT, Action.LEFT),
        up=axis(Action.UP, Action.DOWN),
        look_dx_px=frame.look_dx_px,
        look_dy_px=frame.look_dy_px,
    )
