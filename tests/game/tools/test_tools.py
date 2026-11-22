"""Tools (SPEC 4.2, 8.2): content files, the jet outside the liquid's grid, the hose."""

import numpy as np
import pytest

from entropy.game.tools.base import Aim, ToolContext, load_tool_spec
from entropy.game.tools.hose import Hose
from entropy.game.tools.jet import Domain, Jet

GRAVITY = (0.0, 0.0, -9.81)
TANK = Domain(low_m=(-2.0, -2.0, 0.0), high_m=(2.0, 2.0, 2.5), rim_z_m=2.0, ground_z_m=-0.32)
PARTICLE_M3 = 0.0625**3 / 8


class _Liquid:
    """Records what a tool adds, like `LiquidSolver.add_particles`."""

    def __init__(self) -> None:
        self.positions = np.zeros((0, 3))
        self.velocities = np.zeros((0, 3))

    def room_particles(self) -> int:
        return 10**9

    def particle_count(self) -> int:
        return len(self.positions)

    def add_particles(self, positions_m, *, velocities=None) -> None:
        added = np.asarray(positions_m, dtype=float).reshape(-1, 3)
        moving = np.zeros_like(added) if velocities is None else np.asarray(velocities)
        self.positions = np.concatenate([self.positions, added])
        self.velocities = np.concatenate([self.velocities, moving.reshape(-1, 3)])


def test_a_tool_s_parameters_come_from_its_content_file() -> None:
    spec = load_tool_spec("hose")

    assert spec.power == "flow_l_per_s"
    flow = spec.parameters["flow_l_per_s"]
    assert (flow.minimum, flow.maximum, flow.default, flow.unit) == (1.0, 50.0, 20.0, "L/s")
    nozzle = spec.parameters["nozzle_speed_m_per_s"]
    assert (nozzle.minimum, nozzle.maximum) == (5.0, 30.0)


def test_power_stays_within_its_limits() -> None:
    hose = Hose(load_tool_spec("hose"))

    hose.change_power(+100)
    assert hose.values["flow_l_per_s"] == 50.0
    hose.change_power(-3)
    assert hose.values["flow_l_per_s"] == 47.0
    hose.change_power(-1000)
    assert hose.values["flow_l_per_s"] == 1.0


def test_a_ballistic_parcel_enters_through_the_open_top() -> None:
    jet = Jet(GRAVITY)
    jet.launch(np.array([[0.0, 0.0, 3.0]]), np.array([[0.0, 0.0, 0.0]]))

    entered = [jet.advance(1.0 / 60.0, TANK) for _ in range(30)]

    positions = np.concatenate([p for p, _ in entered])
    velocities = np.concatenate([v for _, v in entered])
    assert len(positions) == 1 and len(jet) == 0
    assert TANK.contains(positions[0])
    fall_s = np.sqrt(2 * 0.5 / 9.81)  # from 3.0 m to the grid's top at 2.5 m
    assert velocities[0, 2] == pytest.approx(-9.81 * fall_s, abs=9.81 / 60.0)


def test_water_hitting_the_glass_below_the_rim_is_lost() -> None:
    """ADR 0017: water cannot exist outside the tank; the glass stops it."""
    jet = Jet(GRAVITY)
    jet.launch(np.array([[4.0, 0.0, 1.0]]), np.array([[-10.0, 0.0, 0.0]]))

    for _ in range(30):
        positions, _ = jet.advance(1.0 / 60.0, TANK)
        assert len(positions) == 0

    assert len(jet) == 0
    assert jet.lost_parcels == 1


def test_water_over_the_rim_enters_through_the_side() -> None:
    jet = Jet(GRAVITY)
    jet.launch(np.array([[4.0, 0.0, 2.3]]), np.array([[-10.0, 0.0, 0.0]]))

    entered = sum(len(jet.advance(1.0 / 60.0, TANK)[0]) for _ in range(30))

    assert entered == 1 and jet.lost_parcels == 0


def test_water_falling_on_the_floor_outside_is_lost() -> None:
    jet = Jet(GRAVITY)
    jet.launch(np.array([[6.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))

    for _ in range(60):
        jet.advance(1.0 / 60.0, TANK)

    assert len(jet) == 0 and jet.lost_parcels == 1


def _hose_into_tank(time_scale: float, seconds: float) -> tuple[float, Jet, _Liquid]:
    """Aim the hose from above the rim into the tank for `seconds` of real time."""
    hose = Hose(load_tool_spec("hose"))
    jet = Jet(GRAVITY)
    liquid = _Liquid()
    aim = Aim(origin_m=(0.0, -1.0, 2.8), direction=(0.0, 0.6, -0.8))
    context = ToolContext(aim=aim, liquid=liquid, jet=jet, particle_volume_m3=PARTICLE_M3)
    frame_s = 1.0 / 60.0
    dt_s = frame_s * time_scale
    hose.on_press(context)
    for _ in range(round(seconds / frame_s)):
        hose.on_hold(context, dt_s)
        jet.advance(dt_s, TANK, liquid)
    hose.on_release(context)
    for _ in range(round(2.0 / dt_s)):  # let the water in flight land
        jet.advance(dt_s, TANK, liquid)
    return len(liquid.positions) * PARTICLE_M3, jet, liquid


@pytest.mark.parametrize("time_scale", [1.0, 0.1])
def test_the_hose_delivers_its_flow_at_any_time_scale(time_scale: float) -> None:
    """US-01's volume, #35: 20 L/s for 1 s of simulated time is 20 L, within 2 %, at 1x
    and at 0.1x (1.1 particles per frame), with no water lost to rounding."""
    seconds = 1.0 / time_scale
    delivered_m3, jet, _ = _hose_into_tank(time_scale, seconds)

    assert jet.lost_parcels == 0
    assert delivered_m3 == pytest.approx(0.020, rel=0.02)


def test_the_jet_leaves_the_nozzle_at_the_nozzle_speed() -> None:
    _, _, liquid = _hose_into_tank(1.0, 0.2)

    # It falls 0.4 m or so before entering: its speed grows a little, never shrinks.
    speeds = np.linalg.norm(liquid.velocities, axis=1)
    assert speeds.min() >= 10.0 - 1e-6
    assert speeds.max() < 10.0 + 9.81 * 0.2


def test_a_tool_context_needs_a_unit_aim() -> None:
    with pytest.raises(ValueError, match="direction"):
        Aim(origin_m=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 2.0))


def _one_frame_of_hose(aim: Aim, dt_s: float, domain: Domain = TANK) -> tuple[Jet, _Liquid]:
    """One frame as the toolbox runs it: the hose releases, then the jet moves."""
    hose = Hose(load_tool_spec("hose"))
    jet = Jet(GRAVITY)
    liquid = _Liquid()
    context = ToolContext(aim=aim, liquid=liquid, jet=jet, particle_volume_m3=PARTICLE_M3)
    hose.on_press(context)
    hose.on_hold(context, dt_s)
    jet.advance(dt_s, domain, liquid)
    return jet, liquid


def test_after_its_first_frame_the_water_is_one_frame_down_the_jet() -> None:
    """Water leaves the nozzle (0.3 m ahead) at even times through the frame: at the
    frame's end it lies between the nozzle and one frame's flight beyond it, not two
    (#74's review, S1)."""
    aim = Aim(origin_m=(20.0, 0.0, 5.0), direction=(1.0, 0.0, 0.0))  # far from the tank
    dt_s = 1.0 / 60.0

    jet, _ = _one_frame_of_hose(aim, dt_s)

    ahead = jet.positions_m[:, 0] - 20.0
    assert len(ahead) > 0
    assert ahead.min() >= 0.3 - 1e-9
    assert ahead.max() <= 0.3 + 10.0 * dt_s + 1e-9
    # Strung along that stretch, as they left at even times through the frame.
    assert ahead.min() < 0.3 + 0.1 * 10.0 * dt_s
    assert ahead.max() > 0.3 + 0.9 * 10.0 * dt_s


@pytest.mark.parametrize(("distance_m", "fps"), [(0.2, 60.0), (1.0, 30.0)])
def test_the_hose_cannot_spray_through_the_glass(distance_m: float, fps: float) -> None:
    """The camera just outside the glass, aiming at it below the rim: the path from the
    camera on is tested, so no water gets in, however close the nozzle is (S1)."""
    aim = Aim(origin_m=(2.0 + distance_m, 0.0, 1.0), direction=(-1.0, 0.0, 0.0))
    hose = Hose(load_tool_spec("hose"))
    hose.values["nozzle_speed_m_per_s"] = 30.0
    jet = Jet(GRAVITY)
    liquid = _Liquid()
    context = ToolContext(aim=aim, liquid=liquid, jet=jet, particle_volume_m3=PARTICLE_M3)
    hose.on_press(context)
    for _ in range(10):
        hose.on_hold(context, 1.0 / fps)
        jet.advance(1.0 / fps, TANK, liquid)

    assert len(liquid.positions) == 0
    assert jet.lost_parcels > 0


def test_a_parcel_crossing_the_whole_tank_in_one_frame_still_hits_the_glass() -> None:
    """Fast enough to pass through both walls between two frames: the glass on its way
    still stops it (mutant M5)."""
    jet = Jet(GRAVITY)
    jet.launch(np.array([[3.0, 0.0, 1.0]]), np.array([[-600.0, 0.0, 0.0]]))

    entered, _ = jet.advance(1.0 / 60.0, TANK)

    assert len(entered) == 0 and len(jet) == 0 and jet.lost_parcels == 1


def test_a_parcel_from_above_the_grid_that_comes_in_through_a_side_below_the_rim_is_lost() -> None:
    """Starting above the grid's top is not enough to enter: this one dives past the top's
    edge and crosses the side below the rim (mutant M6)."""
    jet = Jet(GRAVITY)
    jet.launch(np.array([[2.05, 0.0, 2.6]]), np.array([[-6.0, 0.0, -120.0]]))

    entered, _ = jet.advance(1.0 / 60.0, TANK)

    assert len(entered) == 0 and jet.lost_parcels == 1


def test_a_parcel_that_misses_the_side_s_face_does_not_count_as_crossing_it() -> None:
    """A path that crosses a side's plane beside the tank, then comes in over the open top,
    enters: the side's plane alone is not its face."""
    jet = Jet(GRAVITY)
    jet.launch(np.array([[2.5, 3.0, 2.7]]), np.array([[-60.0, -120.0, -12.0]]))

    entered, _ = jet.advance(1.0 / 60.0, TANK)

    assert len(entered) == 1


def test_water_is_lost_on_the_frame_it_reaches_the_ground() -> None:
    """From 1 m up, it reaches the ground 0.32 m under the floor after
    sqrt(2 * 1.32 / 9.81) = 0.519 s: lost on frame 32, in flight on frame 31 (M7)."""
    jet = Jet(GRAVITY)
    jet.launch(np.array([[6.0, 0.0, 1.0]]), np.array([[0.0, 0.0, 0.0]]))

    for _ in range(31):
        jet.advance(1.0 / 60.0, TANK)
    assert len(jet) == 1
    jet.advance(1.0 / 60.0, TANK)
    assert len(jet) == 0 and jet.lost_parcels == 1
