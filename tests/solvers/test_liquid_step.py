"""The liquid's full step (SPEC 5.2, 5.9; ADR 0013, 0016, 0017)."""

import math
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from entropy.fields.mac import GridSpec
from entropy.linalg.mgpcg import Mgpcg, SolveResult
from entropy.sim.simulation import PacedSolver
from entropy.solvers.interface import Solver
from entropy.solvers.liquid.solver import (
    MAX_SUBSTEPS,
    LiquidSetup,
    LiquidSolver,
    NonFiniteStateError,
    fill_box,
    load_dump,
)
from entropy.world.maps import Environment
from entropy.world.materials import load_material

WATER = load_material("water")
GRAVITY = Environment().gravity_m_per_s2
DX_M = 1.0 / 16.0


def _setup(
    shape: tuple[int, int, int],
    dump_dir: Path | None = None,
    solid_phi_m: np.ndarray | None = None,
    particles_per_cell: int = 8,
) -> LiquidSetup:
    return LiquidSetup(
        spec=GridSpec(shape=shape, dx_m=DX_M),
        density_kg_per_m3=WATER.density_kg_per_m3,
        gravity_m_per_s2=GRAVITY,
        dump_dir=dump_dir,
        solid_phi_m=solid_phi_m,
        particles_per_cell=particles_per_cell,
    )


def test_fill_box_seeds_jittered_particles_per_cell() -> None:
    points = fill_box((0.0, 0.0, 0.0), (0.5, 0.25, 0.125), dx_m=DX_M, particles_per_cell=8, seed=1)

    assert len(points) == 8 * 8 * 4 * 2
    assert np.all(points.min(axis=0) >= 0.0)
    assert np.all(points.max(axis=0) <= (0.5, 0.25, 0.125))
    cells = np.floor(points / DX_M).astype(int)
    _, counts = np.unique(cells, axis=0, return_counts=True)
    assert np.all(counts == 8)


def test_the_liquid_is_a_solver_the_simulation_can_pace(device: str) -> None:
    liquid = LiquidSolver.create(_setup((8, 8, 8)), device)

    assert isinstance(liquid, Solver)
    assert isinstance(liquid, PacedSolver)


def test_a_falling_blob_accelerates_at_g(device: str) -> None:
    """Free fall, far from walls: after 0.1 s the water moves at g t downwards (2 %)."""
    liquid = LiquidSolver.create(_setup((16, 16, 32)), device)
    liquid.add_particles(fill_box((0.375, 0.375, 1.25), (0.625, 0.625, 1.5), DX_M, 8, seed=2))

    for _ in range(6):
        liquid.step(1.0 / 60.0)

    velocity = liquid.particle_velocities()
    assert velocity[:, 2].mean() == pytest.approx(GRAVITY[2] * 0.1, rel=0.02)
    assert np.abs(velocity[:, :2]).max() < 0.05  # no spurious sideways motion


def _surface_heights(liquid: LiquidSolver, columns: slice = slice(1, -1)) -> np.ndarray:
    """Height (m) where the level set crosses zero, in each inner column of the grid."""
    phi = liquid.fields()["phi_m"].numpy()[columns, columns]
    z = (np.arange(phi.shape[2]) + 0.5) * DX_M
    below = (phi[..., :-1] < 0.0) & (phi[..., 1:] >= 0.0)
    k = np.argmax(below, axis=2)
    lo = np.take_along_axis(phi, k[..., None], axis=2)[..., 0]
    hi = np.take_along_axis(phi, (k + 1)[..., None], axis=2)[..., 0]
    return z[k] + DX_M * lo / (lo - hi)


def test_still_water_stays_still(device: str) -> None:
    """A tank of still water with a flat surface: after 1 s nothing has moved by 1 mm
    (ADR 0017's falsifier), and the surface is where the water ends, within 1 mm."""
    liquid = LiquidSolver.create(_setup((12, 12, 16)), device)
    liquid.add_particles(fill_box((0.0, 0.0, 0.0), (0.75, 0.75, 0.5), DX_M, 8, seed=3, jitter=0.0))
    start = liquid.particle_positions().copy()

    for _ in range(60):
        liquid.step(1.0 / 60.0)

    moved = np.linalg.norm(liquid.particle_positions() - start, axis=1)
    assert np.percentile(moved, 99) < 1e-3
    assert np.abs(liquid.particle_velocities()).max() < 5e-3
    assert np.abs(_surface_heights(liquid) - 0.5).max() < 1e-3  # inner columns
    # Against the walls the level set has particles on one side only: 2 mm there.
    assert np.abs(_surface_heights(liquid, slice(None)) - 0.5).max() < 2e-3


def test_a_rough_surface_settles(device: str) -> None:
    """Jittered particles make a rough surface; gravity flattens it, as with real water."""
    liquid = LiquidSolver.create(_setup((12, 12, 16)), device)
    liquid.add_particles(fill_box((0.0, 0.0, 0.0), (0.75, 0.75, 0.5), DX_M, 8, seed=3))
    liquid.step(1.0 / 60.0)
    rough = _surface_heights(liquid).std()

    for _ in range(59):
        liquid.step(1.0 / 60.0)

    assert _surface_heights(liquid).std() < rough / 3
    assert _surface_heights(liquid).mean() == pytest.approx(0.5, abs=2e-3)


def test_particles_never_leave_the_grid(device: str) -> None:
    """The grid's walls are the tank's walls (ADR 0017); particles stay inside them."""
    setup = _setup((8, 8, 8))
    liquid = LiquidSolver.create(setup, device)
    points = fill_box((0.0, 0.0, 0.1), (0.5, 0.5, 0.3), DX_M, 8, seed=4)
    liquid.add_particles(points, velocities=np.tile([3.0, -2.0, 1.0], (len(points), 1)))

    for _ in range(20):
        liquid.step(1.0 / 60.0)

    positions = liquid.particle_positions()
    assert positions.min() >= 0.0
    assert np.all(positions.max(axis=0) <= np.asarray(setup.spec.shape) * DX_M)


def test_substeps_follow_the_cfl_condition_and_the_cap(device: str) -> None:
    """ADR 0013: n = ceil(dt (u + sqrt(g dx)) / dx), at most 4."""
    liquid = LiquidSolver.create(_setup((8, 8, 8)), device)
    points = fill_box((0.125, 0.125, 0.125), (0.375, 0.375, 0.25), DX_M, 8, seed=5)
    liquid.add_particles(points, velocities=np.tile([2.0, 0.0, 0.0], (len(points), 1)))
    liquid.step(1e-4)  # learn the speed
    speed = np.linalg.norm(liquid.particle_velocities(), axis=1).max()

    cfl = speed + math.sqrt(abs(GRAVITY[2]) * DX_M)
    assert liquid.max_advance_s() == pytest.approx(MAX_SUBSTEPS * DX_M / cfl, rel=0.05)
    liquid.step(1.0 / 30.0)
    assert liquid.diagnostics()["substeps"] == min(MAX_SUBSTEPS, math.ceil(cfl / 30.0 / DX_M))


def test_a_nan_freezes_the_liquid_and_dumps_its_state(device: str, tmp_path: Path) -> None:
    """SPEC 5.9: a NaN stops the simulation and becomes a reproducible dump."""
    liquid = LiquidSolver.create(_setup((8, 8, 8), dump_dir=tmp_path), device)
    points = fill_box((0.125, 0.125, 0.125), (0.375, 0.375, 0.25), DX_M, 8, seed=6)
    velocities = np.zeros_like(points)
    velocities[0] = np.nan
    liquid.add_particles(points, velocities=velocities)

    with pytest.raises(NonFiniteStateError) as raised:
        for _ in range(3):  # detected at the latest one step later (no sync point)
            liquid.step(1.0 / 60.0)

    dump = load_dump(raised.value.dump_path)
    assert raised.value.dump_path.parent == tmp_path
    assert dump["positions_m"].shape == (len(points), 3)
    assert not np.isfinite(dump["velocities_m_per_s"]).all()
    with pytest.raises(NonFiniteStateError) as again:  # frozen: same dump, no new one
        liquid.max_advance_s()
    assert again.value.dump_path == raised.value.dump_path
    assert len(list(tmp_path.iterdir())) == 1


def test_a_dump_replays_the_failing_frame(device: str, tmp_path: Path) -> None:
    """SPEC 5.9: the dump holds the state the failing frame started from, and its time
    step: loading it and stepping once gives the non-finite state again."""
    liquid = LiquidSolver.create(_setup((8, 8, 8), dump_dir=tmp_path), device)
    points = fill_box((0.125, 0.125, 0.125), (0.375, 0.375, 0.25), DX_M, 8, seed=6)
    liquid.add_particles(points)
    liquid.step(1.0 / 60.0)
    poisoned = liquid.checkpoint()
    poisoned["velocities_m_per_s"][3] = np.inf
    liquid.restore(poisoned)

    with pytest.raises(NonFiniteStateError) as raised:
        for _ in range(3):
            liquid.step(1.0 / 60.0)

    dump = load_dump(raised.value.dump_path)
    assert float(dump["dt_s"]) == pytest.approx(1.0 / 60.0)
    replay = LiquidSolver.create(_setup((8, 8, 8), dump_dir=tmp_path), device)
    replay.restore({k[len("before_") :]: v for k, v in dump.items() if k.startswith("before_")})
    replay.step(float(dump["dt_s"]))
    assert not np.isfinite(replay.particle_velocities()).all()


def test_a_non_finite_position_is_not_hidden_by_the_grid_clamp(device: str) -> None:
    """Clamping NaN into the grid would make it look finite to the guard."""
    liquid = LiquidSolver.create(_setup((8, 8, 8)), device)
    points = fill_box((0.125, 0.125, 0.125), (0.375, 0.375, 0.25), DX_M, 8, seed=6)
    points[0] = np.nan
    liquid.add_particles(points)

    with pytest.raises(NonFiniteStateError) as raised:
        for _ in range(3):
            liquid.step(1.0 / 60.0)

    assert not np.isfinite(load_dump(raised.value.dump_path)["positions_m"][0]).all()


def test_restore_after_a_freeze_runs_again(device: str, tmp_path: Path) -> None:
    liquid = LiquidSolver.create(_setup((8, 8, 8), dump_dir=tmp_path), device)
    points = fill_box((0.125, 0.125, 0.125), (0.375, 0.375, 0.25), DX_M, 8, seed=6)
    liquid.add_particles(points)
    good = liquid.checkpoint()
    bad = liquid.checkpoint()
    bad["velocities_m_per_s"][0] = np.nan
    liquid.restore(bad)
    with pytest.raises(NonFiniteStateError):
        for _ in range(3):
            liquid.step(1.0 / 60.0)

    liquid.restore(good)
    for _ in range(3):
        liquid.step(1.0 / 60.0)
    assert np.isfinite(liquid.particle_velocities()).all()


def test_restore_returns_to_the_checkpoint(device: str) -> None:
    """A checkpoint, stepped again after restore, gives the same water."""
    liquid = LiquidSolver.create(_setup((12, 12, 16)), device)
    liquid.add_particles(fill_box((0.0, 0.0, 0.0), (0.25, 0.75, 0.75), DX_M, 8, seed=11))
    for _ in range(5):
        liquid.step(1.0 / 60.0)
    saved = liquid.checkpoint()
    for _ in range(5):
        liquid.step(1.0 / 60.0)
    first = liquid.particle_positions().copy()

    liquid.restore(saved)
    for _ in range(5):
        liquid.step(1.0 / 60.0)

    # CPU is bit for bit; CUDA's atomics add in any order (named in the solver).
    np.testing.assert_allclose(liquid.particle_positions(), first, atol=2e-3)


def test_fast_water_hits_the_substep_cap_and_counts_an_overrun(device: str) -> None:
    """ADR 0013: 10 m/s in 6.25 cm cells over 1/30 s needs 6 substeps; 4 run, and the
    frame is counted."""
    liquid = LiquidSolver.create(_setup((16, 8, 8)), device)
    # Away from the walls: water cannot pull away from a wall without air getting in.
    points = fill_box((0.125, 0.125, 0.125), (0.375, 0.375, 0.375), DX_M, 8, seed=12)
    liquid.add_particles(points, velocities=np.tile([10.0, 0.0, 0.0], (len(points), 1)))

    liquid.step(1.0 / 30.0)
    liquid.max_advance_s()  # reads the frame's status

    assert liquid.diagnostics()["substeps"] == MAX_SUBSTEPS
    assert liquid.diagnostics()["overruns"] == 1


def test_a_frame_within_the_cap_is_not_an_overrun(device: str) -> None:
    liquid = LiquidSolver.create(_setup((8, 8, 8)), device)
    liquid.add_particles(fill_box((0.0, 0.0, 0.0), (0.5, 0.5, 0.25), DX_M, 8, seed=7))

    for _ in range(3):
        liquid.step(1.0 / 60.0)
    liquid.max_advance_s()

    assert liquid.diagnostics()["overruns"] == 0


def test_the_speed_comes_from_the_end_of_the_frame(device: str) -> None:
    """The status is sent after the substeps: a blob that fell for 0.1 s is known to move
    at g t, so the next advance is limited by that speed."""
    liquid = LiquidSolver.create(_setup((16, 16, 32)), device)
    liquid.add_particles(fill_box((0.375, 0.375, 1.25), (0.625, 0.625, 1.5), DX_M, 8, seed=2))
    for _ in range(6):
        liquid.step(1.0 / 60.0)

    speed = abs(GRAVITY[2]) * 0.1
    cfl = speed + math.sqrt(abs(GRAVITY[2]) * DX_M)
    assert liquid.max_advance_s() == pytest.approx(MAX_SUBSTEPS * DX_M / cfl, rel=0.03)


def test_velocity_is_extrapolated_two_faces_into_the_air(device: str) -> None:
    """Grid-to-particles reaches 1.5 cells: two faces above a falling blob carry its
    velocity; further up, air faces only hold gravity's push."""
    liquid = LiquidSolver.create(_setup((16, 16, 32)), device)
    liquid.add_particles(fill_box((0.25, 0.25, 1.0), (0.75, 0.75, 1.25), DX_M, 8, seed=13))
    for _ in range(3):
        liquid.step(1.0 / 60.0)

    w = liquid.fields()["velocity"].w.numpy()[8, 8, :]
    phi = liquid.fields()["phi_m"].numpy()[8, 8, :]
    top = int(np.nonzero(phi < 0.0)[0].max())  # the blob's top cell; face top + 1 is its lid
    blob = w[top]
    assert w[top + 2] == pytest.approx(blob, rel=0.05)  # first face above the lid
    assert w[top + 3] == pytest.approx(blob, rel=0.05)  # second
    assert abs(w[top + 4] - blob) > 0.2 * abs(blob)  # beyond: not extrapolated


def test_a_static_solid_holds_water_and_pushes_particles_out(device: str) -> None:
    """A raised floor (a static solid, ADR 0017): water poured on it rests on top, and a
    particle placed inside is pushed out."""
    shape = (12, 12, 16)
    centres = (np.indices(shape).transpose(1, 2, 3, 0) + 0.5) * DX_M
    # A slab 0.25 m high over the whole floor: signed distance, negative inside.
    lo, hi = np.array([-1.0, -1.0, -1.0]), np.array([2.0, 2.0, 0.25])
    q = np.abs(centres - (lo + hi) / 2) - (hi - lo) / 2
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    solid = (outside + np.minimum(q.max(axis=-1), 0.0)).astype(np.float32)
    liquid = LiquidSolver.create(_setup(shape, solid_phi_m=solid), device)
    water = fill_box((0.0, 0.0, 0.3), (0.75, 0.75, 0.5), DX_M, 8, seed=14)
    inside = np.array([[0.375, 0.375, 0.2]], dtype=np.float32)
    liquid.add_particles(np.concatenate([water, inside]))

    for _ in range(30):
        liquid.step(1.0 / 60.0)

    p = liquid.particle_positions()
    assert p[:, 2].min() > 0.25 - 0.1 * DX_M  # nothing sank into the slab
    assert p[-1, 2] > 0.25 - 0.1 * DX_M  # the one inside got out
    assert np.abs(liquid.particle_velocities()).max() < 0.1  # and the water came to rest


def test_wall_faces_stay_closed_after_extrapolation(device: str) -> None:
    """ADR 0015: the grid's walls carry no flow, also where velocity was extrapolated: a
    blob flying at a wall, a cell away from it, must not lend its speed to the wall."""
    liquid = LiquidSolver.create(_setup((12, 8, 12)), device)
    blob = fill_box((DX_M, 0.125, 0.25), (0.25 + DX_M, 0.375, 0.5), DX_M, 8, seed=16)
    liquid.add_particles(blob, velocities=np.tile([-0.5, 0.0, 0.0], (len(blob), 1)))
    liquid.step(1.0 / 240.0)

    velocity = liquid.fields()["velocity"]
    u, v, w = (c.numpy() for c in (velocity.u, velocity.v, velocity.w))
    for face in (u[0], u[-1], v[:, 0], v[:, -1], w[:, :, 0], w[:, :, -1]):
        assert np.all(face == 0.0)


def test_crowded_water_spreads_back_to_its_volume(device: str) -> None:
    """ADR 0018: particles packed 15 % too densely (as clustering does over minutes)
    spread back out, so the water regains the volume its particles carry."""
    liquid = LiquidSolver.create(_setup((12, 12, 16)), device)
    packed = fill_box((0.0, 0.0, 0.0), (0.75, 0.75, 0.6), DX_M, 8, seed=17)
    packed[:, 2] *= 0.85  # the same particles in 85 % of the height
    liquid.add_particles(packed)
    liquid.step(1e-5)
    crowded = liquid.density_ratio()[3:-3, 3:-3, 2:6].mean()

    for _ in range(90):
        liquid.step(1.0 / 60.0)

    assert crowded > 1.1
    assert liquid.density_ratio()[3:-3, 3:-3, 2:6].mean() == pytest.approx(1.0, abs=0.03)


def test_the_density_of_evenly_spread_particles_is_one(device: str) -> None:
    """The volume correction measures crowding against the rest density of 8 particles
    per cell (ADR 0018): jittered, evenly spread particles read 1 inside the water."""
    liquid = LiquidSolver.create(_setup((12, 12, 12)), device)
    liquid.add_particles(fill_box((0.0, 0.0, 0.0), (0.75, 0.75, 0.75), DX_M, 8, seed=15))
    liquid.step(1e-5)

    ratio = liquid.density_ratio()[3:-3, 3:-3, 3:-3]  # away from walls and surface
    assert ratio.mean() == pytest.approx(1.0, abs=0.03)


def test_diagnostics_count_particles_and_pressure_iterations(device: str) -> None:
    liquid = LiquidSolver.create(_setup((8, 8, 8)), device)
    liquid.add_particles(fill_box((0.0, 0.0, 0.0), (0.5, 0.5, 0.25), DX_M, 8, seed=7))

    liquid.step(1.0 / 60.0)

    diagnostics = liquid.diagnostics()
    assert diagnostics["particles"] == 8 * 8 * 8 * 4
    assert diagnostics["pressure_iterations"] > 0
    assert diagnostics["substeps"] >= 1


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"particles_per_cell": 6}, "cube"),
        ({"solid_phi_m": np.ones((3, 3, 3))}, "solid_phi_m"),
        ({"density_kg_per_m3": 0.0}, "density"),
        ({"gravity_m_per_s2": (0.0, 0.0, float("nan"))}, "gravity"),
    ],
)
def test_impossible_setups_are_refused(change: dict, message: str) -> None:
    base = {
        "spec": GridSpec(shape=(4, 4, 4), dx_m=DX_M),
        "density_kg_per_m3": WATER.density_kg_per_m3,
        "gravity_m_per_s2": GRAVITY,
    }
    with pytest.raises(ValueError, match=message):
        LiquidSetup(**{**base, **change})


def test_adding_beyond_capacity_is_refused() -> None:
    setup = LiquidSetup(
        spec=GridSpec(shape=(4, 4, 4), dx_m=DX_M),
        density_kg_per_m3=WATER.density_kg_per_m3,
        gravity_m_per_s2=GRAVITY,
        max_particles=10,
    )
    liquid = LiquidSolver.create(setup, "cpu")
    with pytest.raises(ValueError, match="capacity"):
        liquid.add_particles(np.zeros((11, 3)))


@pytest.mark.gpu
def test_cpu_and_cuda_fall_alike() -> None:
    means = []
    for device in ("cpu", "cuda:0"):
        liquid = LiquidSolver.create(_setup((12, 12, 20)), device)
        liquid.add_particles(fill_box((0.25, 0.25, 0.6), (0.5, 0.5, 0.85), DX_M, 8, seed=8))
        for _ in range(6):
            liquid.step(1.0 / 60.0)
        means.append(liquid.particle_positions().mean(axis=0))
        wp.synchronize()

    np.testing.assert_allclose(means[0], means[1], atol=1e-3)


def test_uneven_water_evens_out_with_its_volume_unchanged(device: str) -> None:
    """ADR 0018's local correction: one half packed 15 % too densely, the other as much
    too sparsely, so the total volume is right and the volume control sees nothing.
    The densities even out within a second."""
    liquid = LiquidSolver.create(_setup((16, 8, 12)), device)
    points = fill_box((0.0, 0.0, 0.0), (1.0, 0.5, 0.5), DX_M, 8, seed=18)
    x = points[:, 0]
    split = 0.5 / 1.15  # the left half's particles fit in this
    points[:, 0] = np.where(x < 0.5, x / 1.15, split + (x - 0.5) * (1.0 - split) / 0.5)
    liquid.add_particles(points)
    liquid.step(1e-5)
    inner = (slice(2, -2), slice(2, -2), slice(2, 5))
    uneven = liquid.density_ratio()[inner].std()

    for _ in range(60):
        liquid.step(1.0 / 60.0)

    assert uneven > 0.08
    assert liquid.density_ratio()[inner].std() < uneven / 3


def test_particles_move_by_the_midpoint_rule(device: str) -> None:
    """Second order in time (SPEC 5.2): in a solid-body rotation of 2 rad/s, particles keep
    their radius over 100 steps of 1/60 s (forward Euler would spiral out by 5 %). The
    rotation is linear in space, so the grid interpolates it exactly."""
    liquid = LiquidSolver.create(_setup((16, 16, 4)), device)
    angle = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
    centre = np.array([0.5, 0.5, 0.125])
    ring = centre + 0.2 * np.stack([np.cos(angle), np.sin(angle), 0 * angle], axis=1)
    liquid.add_particles(ring.astype(np.float32))
    spec = liquid.fields()["velocity"].spec
    omega = 2.0
    u = np.zeros(spec.face_shape(0), np.float32)
    v = np.zeros(spec.face_shape(1), np.float32)
    ys = (np.arange(spec.shape[1]) + 0.5) * DX_M
    xs = (np.arange(spec.shape[0]) + 0.5) * DX_M
    u[:, :, :] = (-omega * (ys - centre[1]))[None, :, None]
    v[:, :, :] = (omega * (xs - centre[0]))[:, None, None]
    velocity = liquid.fields()["velocity"]
    velocity.u.assign(u)
    velocity.v.assign(v)
    velocity.w.zero_()

    for _ in range(100):
        liquid.move_particles(1.0 / 60.0)

    radius = np.linalg.norm(liquid.particle_positions()[:, :2] - centre[:2], axis=1)
    np.testing.assert_allclose(radius, 0.2, rtol=1e-3)


def test_the_grid_clamp_keeps_a_fast_particle_inside(device: str) -> None:
    """ADR 0017: the grid's walls are the tank's; a particle the velocity would carry
    through one stays 1e-4 of a cell inside."""
    liquid = LiquidSolver.create(_setup((8, 8, 8)), device)
    liquid.add_particles(np.array([[0.05, 0.25, 0.25]], dtype=np.float32))
    velocity = liquid.fields()["velocity"]
    velocity.u.fill_(-30.0)  # through the x = 0 wall in one step
    velocity.v.zero_()
    velocity.w.zero_()

    liquid.move_particles(1.0 / 60.0)

    x = liquid.particle_positions()[0, 0]
    assert 0.0 < x <= 2e-4 * DX_M


@pytest.mark.parametrize(
    ("per_cell", "depth_cells"), [(8, 8.0), (8, 7.5), (27, 7 + 1 / 3), (27, 7 + 2 / 3)]
)
def test_the_volume_control_measures_still_water_as_its_particles(
    device: str, per_cell: int, depth_cells: float
) -> None:
    """The control's own measure of the water (ADR 0018) agrees with the particles'
    volume within 0.1 %, with the surface on a face, mid-cell, a third and two thirds of
    the way through a cell (27 particles per cell, to put it there)."""
    liquid = LiquidSolver.create(_setup((12, 12, 16), particles_per_cell=per_cell), device)
    depth_m = depth_cells * DX_M
    points = fill_box((0.0, 0.0, 0.0), (0.75, 0.75, depth_m), DX_M, per_cell, seed=19, jitter=0.0)
    liquid.add_particles(points)
    liquid.step(1e-5)
    liquid.max_advance_s()  # reads the frame's status

    diagnostics = liquid.diagnostics()
    particles_m3 = len(points) * DX_M**3 / per_cell
    assert diagnostics["particles_in_water"] == len(points)
    assert diagnostics["water_volume_m3"] == pytest.approx(particles_m3, rel=1e-3)


@pytest.mark.slow
def test_jittered_still_water_stays_still_for_ten_seconds(device: str) -> None:
    """The volume correction must not stir still water (ADR 0018): its dead band ignores
    the measure's small bias, and the local correction is gentle. After 10 s nothing
    moves faster than 5 mm/s."""
    liquid = LiquidSolver.create(_setup((16, 4, 16)), device)
    liquid.add_particles(fill_box((0.0, 0.0, 0.0), (1.0, 0.25, 0.4), DX_M, 8, seed=9))

    for _ in range(600):
        liquid.step(1.0 / 60.0)

    assert np.abs(liquid.particle_velocities()).max() < 5e-3


def test_particles_within_a_sphere_can_be_removed(device: str) -> None:
    """The eraser (#54): the particles within the radius go, the others keep their order,
    velocities and affine matrices."""
    liquid = LiquidSolver.create(_setup((8, 8, 8)), device)
    points = fill_box((0.0, 0.0, 0.0), (0.5, 0.5, 0.25), DX_M, 8, seed=20)
    velocities = np.tile(np.arange(len(points), dtype=np.float32)[:, None], (1, 3))
    liquid.add_particles(points, velocities=velocities)
    centre, radius = np.array([0.25, 0.25, 0.1]), 0.12
    outside = np.linalg.norm(points - centre, axis=1) > radius

    removed = liquid.remove_particles_within(tuple(centre), radius)

    assert removed == int((~outside).sum()) > 0
    assert liquid.diagnostics()["particles"] == int(outside.sum())
    np.testing.assert_allclose(liquid.particle_positions(), points[outside])
    np.testing.assert_allclose(liquid.particle_velocities(), velocities[outside])


def test_removing_nothing_changes_nothing(device: str) -> None:
    liquid = LiquidSolver.create(_setup((8, 8, 8)), device)
    points = fill_box((0.0, 0.0, 0.0), (0.5, 0.5, 0.25), DX_M, 8, seed=21)
    liquid.add_particles(points)

    assert liquid.remove_particles_within((5.0, 5.0, 5.0), 0.1) == 0
    np.testing.assert_allclose(liquid.particle_positions(), points)
@pytest.mark.parametrize("jitter", [0.0, 0.8])
def test_a_grid_full_of_water_stays_still(device: str, jitter: float) -> None:
    """#69: water up to the grid's lid, no air anywhere: still, finite, converged."""
    liquid = LiquidSolver.create(_setup((12, 10, 16)), device)
    liquid.add_particles(
        fill_box((0.0, 0.0, 0.0), (0.75, 0.625, 1.0), DX_M, 8, seed=25, jitter=jitter)
    )

    for _ in range(60):
        liquid.step(1.0 / 60.0)

    assert np.isfinite(liquid.particle_velocities()).all()
    assert np.abs(liquid.particle_velocities()).max() < 5e-3
    assert liquid.diagnostics()["pressure_iterations"] < 200


@pytest.mark.gpu
@pytest.mark.parametrize("jitter", [0.0, 0.8])
def test_the_test_tank_s_grid_full_of_water_never_caps_a_solve(
    jitter: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#69 at the Test Tank's grid size (72 x 72 x 40), water to the lid: every substep's
    pressure solve converges well under its cap, for 600 frames (measured: at most 8
    iterations). Before ADR 0021's fix in MGPCG, 1 to 4 solves hit the 200-iteration cap
    here (60 ms each, a visible hitch)."""
    liquid = LiquidSolver.create(_setup((72, 72, 40)), "cuda:0")
    liquid.add_particles(
        fill_box((0.0, 0.0, 0.0), (4.5, 4.5, 2.5), DX_M, 8, seed=25, jitter=jitter)
    )
    solve = Mgpcg.solve
    iterations: list[int] = []

    def counted(self: Mgpcg, *args, **kwargs) -> SolveResult:
        """Every substep's solve, not only the last one's (`diagnostics`)."""
        result = solve(self, *args, **kwargs)
        iterations.append(result.iterations if result.converged else 10_000)
        return result

    monkeypatch.setattr(Mgpcg, "solve", counted)
    for _ in range(600):
        liquid.step(1.0 / 60.0)

    assert len(iterations) >= 600
    assert max(iterations) <= 16
    assert np.abs(liquid.particle_velocities()).max() < 5e-3
