"""Keys from the settings (`entropy.game.input.bindings_from_settings`), issue #32.

Letters and digits name the label printed on the player's keyboard, resolved
through the keyboard layout; other keys are named (SPACE, LEFT_CONTROL, F3...).
"""

import glfw
import pytest

from entropy.core.config import ControlsSettings
from entropy.game.input import Action, bindings_from_settings

_AZERTY_SWAPS = {
    glfw.KEY_Q: "a",
    glfw.KEY_A: "q",
    glfw.KEY_W: "z",
    glfw.KEY_Z: "w",
    glfw.KEY_SEMICOLON: "m",
    glfw.KEY_M: ",",
}


def qwerty(key: int) -> str | None:
    """Labels of a US keyboard: the GLFW key names themselves."""
    if glfw.KEY_A <= key <= glfw.KEY_Z or glfw.KEY_0 <= key <= glfw.KEY_9:
        return chr(key).lower()
    return None


def azerty(key: int) -> str | None:
    return _AZERTY_SWAPS.get(key, qwerty(key))


ZQSD = ControlsSettings(forward="Z", left="Q", back="S", right="D")


def test_azerty_labels_land_on_the_usual_positions() -> None:
    bindings = bindings_from_settings(ZQSD, key_label=azerty)

    assert bindings[glfw.KEY_W] is Action.FORWARD
    assert bindings[glfw.KEY_A] is Action.LEFT
    assert bindings[glfw.KEY_S] is Action.BACK
    assert bindings[glfw.KEY_D] is Action.RIGHT


def test_the_default_controls_on_a_qwerty_keyboard_are_wasd() -> None:
    bindings = bindings_from_settings(ControlsSettings(), key_label=qwerty)

    assert {key: bindings[key] for key in (glfw.KEY_W, glfw.KEY_A, glfw.KEY_S, glfw.KEY_D)} == {
        glfw.KEY_W: Action.FORWARD,
        glfw.KEY_A: Action.LEFT,
        glfw.KEY_S: Action.BACK,
        glfw.KEY_D: Action.RIGHT,
    }
    assert bindings[glfw.KEY_SPACE] is Action.UP
    assert bindings[glfw.KEY_LEFT_CONTROL] is Action.DOWN
    assert bindings[glfw.KEY_RIGHT_CONTROL] is Action.DOWN
    assert bindings[glfw.KEY_F3] is Action.TOGGLE_HUD
    assert bindings[glfw.KEY_ESCAPE] is Action.MENU


def test_labels_ignore_case_and_names_allow_several_keys() -> None:
    controls = ControlsSettings(forward="z", up="space, page_up")

    bindings = bindings_from_settings(controls, key_label=azerty)

    assert bindings[glfw.KEY_W] is Action.FORWARD
    assert bindings[glfw.KEY_PAGE_UP] is Action.UP


@pytest.mark.parametrize(
    ("controls", "message"),
    [
        (ControlsSettings(forward="É"), r"controls\.forward: no key labelled 'É'"),
        (ControlsSettings(up="SPACEBAR"), r"controls\.up: unknown key name 'SPACEBAR'"),
        (ControlsSettings(forward="Z", up="Z"), r"key 'Z' is bound to both forward and up"),
        (ControlsSettings(menu=""), r"controls\.menu: no key"),
    ],
)
def test_impossible_controls_are_errors_naming_the_setting(
    controls: ControlsSettings, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        bindings_from_settings(controls, key_label=azerty)
