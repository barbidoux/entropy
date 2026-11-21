"""The toolbox (`entropy.game.toolbox`): choosing a tool, its power, using it (SPEC 4.5)."""

import glfw
import pytest

from entropy.game.input import Action, FrameInput, InputState
from entropy.game.toolbox import Toolbox
from entropy.game.tools.base import Aim, Tool
from entropy.game.tools.jet import Domain

TANK = Domain(low_m=(-2.0, -2.0, 0.0), high_m=(2.0, 2.0, 2.5), rim_z_m=2.0, ground_z_m=-0.32)
AIM = Aim(origin_m=(0.0, -1.0, 2.8), direction=(0.0, 0.6, -0.8))
PARTICLE_M3 = 0.0625**3 / 8


class _Liquid:
    def __init__(self) -> None:
        self.added = 0

    def room_particles(self) -> int:
        return 10**9

    def particle_count(self) -> int:
        return self.added

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


def test_right_click_reaches_the_tool_in_hand() -> None:
    state = InputState()
    state.set_mouse_captured(True)
    state.on_mouse_button(glfw.MOUSE_BUTTON_RIGHT, glfw.PRESS)
    frame = state.take_frame()
    toolbox = _toolbox(_Liquid())
    toolbox.handle(_frame(pressed=frozenset({Action.TOOL_2})))

    toolbox.handle(frame)

    assert frame.alternate_pressed
    assert getattr(toolbox.current, "shape", None) == "barrage"


class _SmallTank(_Liquid):
    """A liquid with room for `room` particles."""

    def __init__(self, room: int) -> None:
        super().__init__()
        self.room = room

    def room_particles(self) -> int:
        return self.room - self.added


def test_water_past_the_tank_s_room_is_lost_and_the_hud_says_so() -> None:
    """The grid holds at most its volume of water: a tool adds what fits, the rest is
    lost, counted, and shown; nothing crashes (B1 of #74's review)."""
    liquid = _SmallTank(room=100)
    toolbox = _toolbox(liquid)
    toolbox.handle(_frame(pressed=frozenset({Action.TOOL_3})))  # the flood source
    toolbox.handle(_frame(tool_held=True, tool_pressed=True))

    toolbox.apply(AIM, 1.0)

    released = int(0.1 / PARTICLE_M3)  # 0.1 m³/s for 1 s
    assert liquid.added == 100
    assert toolbox.overflow_m3 == pytest.approx((released - 100) * PARTICLE_M3)
    assert any("pleine" in line for line in toolbox.palette())


class _Recorder:
    """A tool that records what it is told."""

    def __init__(self, name: str, log: list[tuple[str, str]]) -> None:
        from entropy.game.tools.base import ToolParameter, ToolSpec

        power = ToolParameter("p", "p", "1", 0.0, 1.0, 0.5, 0.1)
        self.spec = ToolSpec(name=name, label=name, icon="r", power="p", parameters={"p": power})
        self.values = {"p": 0.5}
        self.log = log

    def change_power(self, steps: int) -> None: ...

    def on_alternate(self) -> None: ...

    def on_press(self, ctx: object) -> None:
        self.log.append((self.spec.name, "press"))

    def on_hold(self, ctx: object, dt_s: float) -> None:
        self.log.append((self.spec.name, "hold"))

    def on_release(self, ctx: object) -> None:
        self.log.append((self.spec.name, "release"))


def _recording_toolbox() -> tuple[Toolbox, list[tuple[str, str]]]:
    from entropy.game.tools.jet import Jet

    log: list[tuple[str, str]] = []
    tools: list[Tool] = [_Recorder("A", log), _Recorder("B", log)]
    box = Toolbox(tools, jet=Jet((0.0, 0.0, -9.81)), domain=TANK, liquid=_Liquid(),
                  particle_volume_m3=PARTICLE_M3)  # fmt: skip
    return box, log


def test_switching_tools_with_the_button_held_releases_one_and_presses_the_other() -> None:
    """The old tool lets go, the new one starts as if clicked (#74's review)."""
    box, log = _recording_toolbox()
    box.handle(_frame(tool_pressed=True, tool_held=True))
    box.apply(AIM, 1.0 / 60.0)
    box.handle(_frame(pressed=frozenset({Action.TOOL_2}), tool_held=True))
    box.apply(AIM, 1.0 / 60.0)

    assert log == [("A", "press"), ("A", "hold"), ("A", "release"), ("B", "press"), ("B", "hold")]


def test_a_click_while_the_world_stands_still_is_forgotten() -> None:
    """Paused or frozen, no tool acts; a click then must not fire when the world runs
    again, maybe with another tool in hand (#74's review)."""
    box, log = _recording_toolbox()
    box.handle(_frame(tool_pressed=True))
    box.handle(_frame(tool_released=True))

    box.forget_clicks()
    box.apply(AIM, 1.0 / 60.0)

    assert log == []
