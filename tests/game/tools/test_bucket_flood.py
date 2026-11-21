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

    assert len(liquid.positions) * PARTICLE_M3 == pytest.approx(volume_m3, rel=0.02)
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
def test_the_flood_source_adds_its_rate_from_the_floor(time_scale: float) -> None:
    """10 s of simulated time at 0.1 m³/s is 1 m³ within 2 %, at 1x and 0.1x."""
    liquid = _Liquid()
    flood = Flood(load_tool_spec("flood"))
    context = _context(liquid, DOWN_INTO_TANK)
    dt_s = time_scale / 60.0

    flood.on_press(context)
    for _ in range(round(10.0 / dt_s)):
        flood.on_hold(context, dt_s)

    assert len(liquid.positions) * PARTICLE_M3 == pytest.approx(1.0, rel=0.02)
    assert TANK.contains(liquid.positions).all()
    assert liquid.positions[:, 2].max() < DX_M  # from the floor
    rise_m_per_s = 0.1 / (4.0 * 4.0)
    np.testing.assert_allclose(liquid.velocities[:, 2], rise_m_per_s)
