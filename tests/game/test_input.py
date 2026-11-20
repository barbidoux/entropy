"""Input mapping (`entropy.game.input`): keys to actions, SPEC 4.5."""

import glfw
import pytest

from entropy.game.input import DEFAULT_BINDINGS, Action, InputState, flight_input


@pytest.mark.parametrize(
    ("key", "action"),
    [
        (glfw.KEY_W, Action.FORWARD),
        (glfw.KEY_S, Action.BACK),
        (glfw.KEY_A, Action.LEFT),
        (glfw.KEY_D, Action.RIGHT),
        (glfw.KEY_SPACE, Action.UP),
        (glfw.KEY_LEFT_CONTROL, Action.DOWN),
        (glfw.KEY_RIGHT_CONTROL, Action.DOWN),
        (glfw.KEY_F3, Action.TOGGLE_HUD),
        (glfw.KEY_ESCAPE, Action.MENU),
    ],
)
def test_default_bindings_follow_spec_4_5(key: int, action: Action) -> None:
    assert DEFAULT_BINDINGS[key] is action


def test_held_keys_stay_held_until_released() -> None:
    state = InputState()

    state.on_key(glfw.KEY_W, glfw.PRESS)
    first = state.take_frame()
    second = state.take_frame()
    state.on_key(glfw.KEY_W, glfw.RELEASE)
    third = state.take_frame()

    assert Action.FORWARD in first.held
    assert Action.FORWARD in second.held
    assert Action.FORWARD not in third.held


def test_a_press_is_reported_once_even_with_key_repeat() -> None:
    state = InputState()

    state.on_key(glfw.KEY_F3, glfw.PRESS)
    state.on_key(glfw.KEY_F3, glfw.REPEAT)
    first = state.take_frame()
    second = state.take_frame()

    assert first.pressed == {Action.TOGGLE_HUD}
    assert second.pressed == set()


def test_a_tap_inside_one_frame_is_not_lost() -> None:
    state = InputState()

    state.on_key(glfw.KEY_F3, glfw.PRESS)
    state.on_key(glfw.KEY_F3, glfw.RELEASE)

    assert state.take_frame().pressed == {Action.TOGGLE_HUD}


def test_mouse_motion_accumulates_until_the_frame_is_taken() -> None:
    state = InputState()
    state.set_mouse_captured(True)

    state.on_mouse_move(100.0, 100.0)
    state.on_mouse_move(103.0, 98.0)
    state.on_mouse_move(110.0, 95.0)
    first = state.take_frame()
    state.on_mouse_move(111.0, 95.0)
    second = state.take_frame()

    assert (first.look_dx_px, first.look_dy_px) == (10.0, -5.0)
    assert (second.look_dx_px, second.look_dy_px) == (1.0, 0.0)


def test_the_mouse_starts_released() -> None:
    """The window starts with a normal cursor; moving it must not turn the camera."""
    state = InputState()

    state.on_mouse_move(0.0, 0.0)
    state.on_mouse_move(50.0, 50.0)

    assert not state.mouse_captured
    assert state.take_frame().look_dx_px == 0.0


def test_recapturing_the_mouse_does_not_jump_the_view() -> None:
    """GLFW's virtual cursor restarts where the real one was, maybe outside the window."""
    state = InputState()
    state.set_mouse_captured(True)
    state.on_mouse_move(960.0, 540.0)

    state.set_mouse_captured(False)
    state.set_mouse_captured(True)
    state.on_mouse_move(2500.0, 900.0)
    state.on_mouse_move(2502.0, 900.0)

    frame = state.take_frame()
    assert (frame.look_dx_px, frame.look_dy_px) == (2.0, 0.0)


def test_releasing_one_of_two_keys_for_an_action_keeps_it_held() -> None:
    state = InputState()

    state.on_key(glfw.KEY_LEFT_CONTROL, glfw.PRESS)
    state.on_key(glfw.KEY_RIGHT_CONTROL, glfw.PRESS)
    state.on_key(glfw.KEY_RIGHT_CONTROL, glfw.RELEASE)

    assert Action.DOWN in state.take_frame().held


def test_unbound_keys_are_ignored() -> None:
    state = InputState()

    state.on_key(glfw.KEY_Z, glfw.PRESS)

    assert state.take_frame().held == set()


def test_flight_input_from_held_actions() -> None:
    state = InputState()
    for key in (glfw.KEY_W, glfw.KEY_A, glfw.KEY_SPACE):
        state.on_key(key, glfw.PRESS)

    flight = flight_input(state.take_frame())

    assert (flight.forward, flight.right, flight.up) == (1.0, -1.0, 1.0)


def test_opposite_keys_cancel() -> None:
    state = InputState()
    for key in (glfw.KEY_W, glfw.KEY_S, glfw.KEY_SPACE, glfw.KEY_LEFT_CONTROL):
        state.on_key(key, glfw.PRESS)

    flight = flight_input(state.take_frame())

    assert (flight.forward, flight.up) == (0.0, 0.0)
