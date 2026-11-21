"""Input mapping: GLFW keys and mouse to game actions (SPEC 4.5).

GLFW calls `on_key` and `on_mouse_move` as events arrive; the game loop takes
one `FrameInput` per frame. Held actions persist across frames; presses are
reported once, in the frame they happened, even if the key was released in
the same frame. An action stays held while any key bound to it is down.

Internally keys are GLFW key tokens, which name physical key positions (on an
AZERTY keyboard the token KEY_W is the key labelled Z). The player's settings
name keys by their printed labels; `bindings_from_settings` turns those into
tokens through the keyboard layout (issue #32). Mouse motion is counted only
while the mouse is captured; with raw motion on (ADR 0004) the units are mouse
counts, so the look sensitivity depends on the mouse's resolution.
"""

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

import glfw

from entropy.core.config import ControlsSettings
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
    TOOL_1 = "tool_1"
    TOOL_2 = "tool_2"
    TOOL_3 = "tool_3"
    TOOL_4 = "tool_4"
    TOOL_5 = "tool_5"
    TOOL_6 = "tool_6"
    TOOL_7 = "tool_7"
    TOOL_8 = "tool_8"
    TOOL_9 = "tool_9"


TOOL_ACTIONS = tuple(Action(f"tool_{n}") for n in range(1, 10))
"""Keys 1 to 9 choose a tool (SPEC 4.5)."""


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
        **{glfw.KEY_1 + n: Action(f"tool_{n + 1}") for n in range(9)},
    }
)


_PRINTABLE_KEYS = (
    glfw.KEY_APOSTROPHE,
    *range(glfw.KEY_COMMA, glfw.KEY_9 + 1),
    glfw.KEY_SEMICOLON,
    glfw.KEY_EQUAL,
    *range(glfw.KEY_A, glfw.KEY_RIGHT_BRACKET + 1),
    glfw.KEY_GRAVE_ACCENT,
    glfw.KEY_WORLD_1,
    glfw.KEY_WORLD_2,
)
"""Key tokens whose label depends on the keyboard layout."""


def glfw_key_label(key: int) -> str | None:
    """The label printed on key in the current keyboard layout (GLFW must be running).

    Keys this keyboard lacks (often KEY_WORLD_1 and 2) have no label; GLFW's
    warning for them is not an error here.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", glfw.GLFWError)
        label = glfw.get_key_name(key, 0)
    if isinstance(label, bytes):
        label = label.decode("utf-8")
    return label or None


def _key_for(setting: str, name: str, key_label: Callable[[int], str | None]) -> int:
    if name.startswith("@"):  # a position, named after the US-layout key there
        key = getattr(glfw, f"KEY_{name[1:].upper()}", None)
        if len(name) != 2 or not isinstance(key, int) or key not in _PRINTABLE_KEYS:
            raise ValueError(f"controls.{setting}: unknown key position {name!r}")
        return key
    if len(name) == 1:
        for key in _PRINTABLE_KEYS:
            if (key_label(key) or "").upper() == name.upper():
                return key
        # AZERTY prints & é " ... on the digit row: use that row. Not str.isdigit(),
        # which also accepts "²" and other scripts' digits.
        if name in "0123456789":
            return glfw.KEY_0 + int(name)
        raise ValueError(f"controls.{setting}: no key labelled {name!r} on this keyboard")
    key = getattr(glfw, f"KEY_{name.upper()}", None)
    if not isinstance(key, int) or key < 0 or name.upper() in ("UNKNOWN", "LAST"):
        raise ValueError(f"controls.{setting}: unknown key name {name!r}")
    return key


def bindings_from_settings(
    controls: ControlsSettings, *, key_label: Callable[[int], str | None] = glfw_key_label
) -> Mapping[int, Action]:
    """Key tokens for each action named in the settings (see ControlsSettings)."""
    bindings: dict[int, Action] = {}
    for action in Action:
        keys = [name.strip() for name in getattr(controls, action.value).split(",")]
        keys = [name for name in keys if name]
        if not keys:
            raise ValueError(f"controls.{action.value}: no key")
        for name in keys:
            key = _key_for(action.value, name, key_label)
            if key in bindings and bindings[key] is not action:
                raise ValueError(
                    f"controls: key {name!r} is bound to both {bindings[key].value} and "
                    f"{action.value}"
                )
            bindings[key] = action
    return MappingProxyType(bindings)


@dataclass(frozen=True)
class FrameInput:
    """Everything the player did during one frame."""

    held: frozenset[Action]
    pressed: frozenset[Action]
    look_dx_px: float
    look_dy_px: float
    tool_held: bool = False
    """The tool button (left mouse) is down, with the mouse captured."""
    tool_pressed: bool = False
    tool_released: bool = False
    wheel_steps: int = 0
    """Mouse wheel notches this frame, up positive."""


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
        self._tool_held = False
        self._tool_pressed = False
        self._tool_released = False
        self._wheel = 0.0

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

    def on_mouse_button(self, button: int, event: int) -> None:
        """A GLFW mouse button event. The left button uses the tool, but only while the
        mouse is captured: the click that captures it does not."""
        if button != glfw.MOUSE_BUTTON_LEFT:
            return
        if event == glfw.PRESS and self._mouse_captured:
            self._tool_held = True
            self._tool_pressed = True
        elif event == glfw.RELEASE and self._tool_held:
            self._tool_held = False
            self._tool_released = True

    def on_scroll(self, dy: float) -> None:
        """A GLFW scroll event: notches of the wheel, up positive."""
        self._wheel += dy

    def take_frame(self) -> FrameInput:
        """This frame's input; presses, wheel and mouse motion start again from zero."""
        steps = int(self._wheel)
        frame = FrameInput(
            held=frozenset(self.bindings[key] for key in self._held_keys),
            pressed=frozenset(self._pressed),
            look_dx_px=self._dx_px,
            look_dy_px=self._dy_px,
            tool_held=self._tool_held,
            tool_pressed=self._tool_pressed,
            tool_released=self._tool_released,
            wheel_steps=steps,
        )
        self._pressed.clear()
        self._dx_px = self._dy_px = 0.0
        self._tool_pressed = self._tool_released = False
        self._wheel -= steps
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
