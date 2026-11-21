"""The toolbox (`entropy.game.toolbox`): choosing a tool, its power, using it (SPEC 4.5)."""

import glfw

from entropy.game.input import Action, FrameInput, InputState
from entropy.game.toolbox import Toolbox
from entropy.game.tools.base import Aim
from entropy.game.tools.jet import Domain

TANK = Domain(low_m=(-2.0, -2.0, 0.0), high_m=(2.0, 2.0, 2.5), rim_z_m=2.0, ground_z_m=-0.32)
AIM = Aim(origin_m=(0.0, -1.0, 2.8), direction=(0.0, 0.6, -0.8))
PARTICLE_M3 = 0.0625**3 / 8


class _Liquid:
    def __init__(self) -> None:
        self.added = 0

    def add_particles(self, positions_m, *, velocities=None) -> None:
        self.added += len(positions_m)


def _frame(**kw) -> FrameInput:
    base = {
        "held": frozenset(),
        "pressed": frozenset(),
        "look_dx_px": 0.0,
        "look_dy_px": 0.0,
        "tool_held": False,
        "tool_pressed": False,
        "tool_released": False,
        "wheel_steps": 0,
    }
    return FrameInput(**{**base, **kw})


def _toolbox(liquid: _Liquid) -> Toolbox:
    return Toolbox.standard(
        gravity_m_per_s2=(0.0, 0.0, -9.81),
        domain=TANK,
        liquid=liquid,
        particle_volume_m3=PARTICLE_M3,
    )


def test_the_hose_is_tool_1_and_the_default() -> None:
    toolbox = _toolbox(_Liquid())

    assert toolbox.current.spec.name == "hose"
    toolbox.handle(_frame(pressed=frozenset({Action.TOOL_1})))
    assert toolbox.current.spec.name == "hose"


def test_a_key_without_a_tool_changes_nothing() -> None:
    toolbox = _toolbox(_Liquid())

    toolbox.handle(_frame(pressed=frozenset({Action.TOOL_9})))

    assert toolbox.current.spec.name == "hose"


def test_the_wheel_sets_the_power() -> None:
    toolbox = _toolbox(_Liquid())

    toolbox.handle(_frame(wheel_steps=3))

    assert toolbox.current.values["flow_l_per_s"] == 23.0


def test_holding_the_button_pours_water_over_simulated_time() -> None:
    liquid = _Liquid()
    toolbox = _toolbox(liquid)

    toolbox.handle(_frame(tool_pressed=True, tool_held=True))
    for _ in range(60):
        toolbox.apply(AIM, 1.0 / 60.0)
        toolbox.handle(_frame(tool_held=True))
    toolbox.handle(_frame(tool_released=True))
    for _ in range(120):
        toolbox.apply(AIM, 1.0 / 60.0)

    assert abs(liquid.added * PARTICLE_M3 - 0.020) <= 0.0004


def test_a_paused_world_receives_no_water() -> None:
    liquid = _Liquid()
    toolbox = _toolbox(liquid)

    toolbox.handle(_frame(tool_pressed=True, tool_held=True))
    for _ in range(60):
        toolbox.apply(AIM, 0.0)

    assert liquid.added == 0 and len(toolbox.jet) == 0


def test_the_palette_shows_the_tool_and_its_settings_in_si_units() -> None:
    lines = _toolbox(_Liquid()).palette()

    text = "\n".join(lines)
    assert "Tuyau" in text and "20 L/s" in text and "10 m/s" in text


def test_the_mouse_uses_the_tool_only_once_captured() -> None:
    state = InputState()
    state.on_mouse_button(glfw.MOUSE_BUTTON_LEFT, glfw.PRESS)
    assert not state.take_frame().tool_pressed  # this click captures the mouse

    state.set_mouse_captured(True)
    state.on_mouse_button(glfw.MOUSE_BUTTON_LEFT, glfw.PRESS)
    frame = state.take_frame()
    assert frame.tool_pressed and frame.tool_held
    assert state.take_frame().tool_held
    state.on_mouse_button(glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE)
    frame = state.take_frame()
    assert frame.tool_released and not frame.tool_held


def test_wheel_steps_add_up_within_a_frame() -> None:
    state = InputState()
    state.on_scroll(1.0)
    state.on_scroll(2.0)
    state.on_scroll(-1.0)

    assert state.take_frame().wheel_steps == 2
    assert state.take_frame().wheel_steps == 0


def test_the_palette_lists_the_m1_tools_and_says_when_the_bucket_misses() -> None:
    toolbox = _toolbox(_Liquid())
    toolbox.handle(_frame(pressed=frozenset({Action.TOOL_2})))
    toolbox.handle(_frame(tool_pressed=True, tool_held=True))
    toolbox.apply(Aim(origin_m=(0.0, -6.0, 1.0), direction=(0.0, -1.0, 0.0)), 1.0 / 60.0)

    text = " | ".join(toolbox.palette())

    assert "1 Tuyau" in text and "2 Seau" in text and "3 Crue" in text
    assert "visez la cuve" in text
