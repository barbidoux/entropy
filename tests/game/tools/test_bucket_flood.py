"""The bucket and the flood source (SPEC 4.2, #54)."""

import numpy as np
import pytest

from entropy.game.tools.base import Aim, ToolContext, load_tool_spec
from entropy.game.tools.bucket import Bucket
from entropy.game.tools.flood import Flood
from entropy.game.tools.jet import Domain, Jet

TANK = Domain(low_m=(-2.0, -2.0, 0.0), high_m=(2.0, 2.0, 2.5), rim_z_m=2.0, ground_z_m=-0.32)
DX_M = 0.0625
PARTICLE_M3 = DX_M**3 / 8


class _Liquid:
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


def _context(liquid: _Liquid, aim: Aim) -> ToolContext:
    return ToolContext(
        aim=aim,
        liquid=liquid,
        jet=Jet((0.0, 0.0, -9.81)),
        particle_volume_m3=PARTICLE_M3,
        domain=TANK,
    )


DOWN_INTO_TANK = Aim(origin_m=(0.5, -3.0, 3.0), direction=(0.0, 0.8, -0.6))


@pytest.mark.parametrize("volume_m3", [0.01, 0.25, 2.0])
def test_the_bucket_drops_its_volume_into_the_tank(volume_m3: float) -> None:
    liquid = _Liquid()
    bucket = Bucket(load_tool_spec("bucket"))
    bucket.values["volume_m3"] = volume_m3

    bucket.on_press(_context(liquid, DOWN_INTO_TANK))

    assert abs(len(liquid.positions) * PARTICLE_M3 - volume_m3) <= PARTICLE_M3 / 2
    assert TANK.contains(liquid.positions).all()
    assert np.all(liquid.velocities == 0.0)


def test_the_bucket_drops_where_the_player_aims() -> None:
    liquid = _Liquid()
    bucket = Bucket(load_tool_spec("bucket"))

    bucket.on_press(_context(liquid, DOWN_INTO_TANK))

    # The aim meets the rim's height at y = -3 + 0.8 * (1 / 0.6) = -1.67, x = 0.5.
    centre = liquid.positions[:, :2].mean(axis=0)
    np.testing.assert_allclose(centre, (0.5, -1.667), atol=0.1)


def test_the_bucket_drops_nothing_when_the_aim_misses_the_tank() -> None:
    liquid = _Liquid()
    bucket = Bucket(load_tool_spec("bucket"))
    away = Aim(origin_m=(0.0, -6.0, 1.0), direction=(0.0, -1.0, 0.0))

    bucket.on_press(_context(liquid, away))

    assert len(liquid.positions) == 0
    assert "cuve" in bucket.status


def test_holding_the_bucket_drops_one_bucketful() -> None:
    liquid = _Liquid()
    bucket = Bucket(load_tool_spec("bucket"))
    context = _context(liquid, DOWN_INTO_TANK)

    bucket.on_press(context)
    for _ in range(30):
        bucket.on_hold(context, 1.0 / 60.0)

    assert len(liquid.positions) * PARTICLE_M3 == pytest.approx(0.25, rel=0.02)


@pytest.mark.parametrize("time_scale", [1.0, 0.1])
def test_the_flood_source_adds_its_rate_at_the_rising_level(time_scale: float) -> None:
    """10 s of simulated time at 0.1 m³/s is 1 m³ within 2 %, at 1x and 0.1x. Each frame's
    water appears just under the level the water stands at over the floor, rising at
    the rate over the floor's area (#74's review: added at the floor, it stood at half
    its depth in shallow water)."""
    liquid = _Liquid()
    flood = Flood(load_tool_spec("flood"))
    context = _context(liquid, DOWN_INTO_TANK)
    dt_s = time_scale / 60.0

    flood.on_press(context)
    for _ in range(round(10.0 / dt_s)):
        flood.on_hold(context, dt_s)

    assert len(liquid.positions) * PARTICLE_M3 == pytest.approx(1.0, rel=0.02)
    assert TANK.contains(liquid.positions).all()
    level_m = 1.0 / (4.0 * 4.0)  # 1 m³ over the 16 m² floor
    assert liquid.positions[:, 2].max() == pytest.approx(level_m, abs=DX_M / 4)
    added_last = liquid.positions[-50:, 2]
    assert added_last.min() > level_m - DX_M / 2  # the last water came in at the top
    rise_m_per_s = 0.1 / (4.0 * 4.0)
    np.testing.assert_allclose(liquid.velocities[:, 2], rise_m_per_s)


def test_right_click_turns_the_bucket_into_a_dam_against_the_far_wall() -> None:
    """US-02: the bucket's other shape is Martin & Moyce's column, a wide and 2a high,
    across the tank's whole width, against the wall the player faces."""
    liquid = _Liquid()
    bucket = Bucket(load_tool_spec("bucket"))
    bucket.values["volume_m3"] = 2.0

    bucket.on_alternate()
    bucket.on_press(_context(liquid, DOWN_INTO_TANK))  # facing +y

    p = liquid.positions
    assert len(p) * PARTICLE_M3 == pytest.approx(2.0, rel=0.02)
    a = np.sqrt(2.0 / (2 * 4.0))  # 0.5 m for 2 m³ across 4 m
    assert p[:, 1].min() == pytest.approx(2.0 - a, abs=DX_M / 2)  # against y = 2
    assert p[:, 1].max() == pytest.approx(2.0, abs=DX_M / 2)
    assert p[:, 2].max() == pytest.approx(2 * a, abs=DX_M / 2)
    assert np.ptp(p[:, 0]) == pytest.approx(4.0, abs=DX_M)
    assert "barrage" in bucket.status


def test_right_click_again_brings_the_block_back() -> None:
    bucket = Bucket(load_tool_spec("bucket"))

    bucket.on_alternate()
    bucket.on_alternate()

    assert bucket.shape == "bloc"


def test_the_flood_rate_moves_by_a_factor_per_wheel_notch() -> None:
    """From 0.1 m³/s to the 5 m³/s maximum in under 20 notches, and back to 0.01 m³/s:
    the wheel scales the rate rather than adding to it (#74's review)."""
    flood = Flood(load_tool_spec("flood"))

    flood.change_power(20)
    assert flood.values["rate_m3_per_s"] == 5.0
    flood.change_power(-40)
    assert flood.values["rate_m3_per_s"] == 0.01
    flood.change_power(1)
    assert flood.values["rate_m3_per_s"] == pytest.approx(0.0125)
