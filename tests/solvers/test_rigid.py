"""Rigid bodies (`entropy.solvers.rigid`, SPEC 5.4, #86): bodies, XPBD substeps, gravity,
the NaN guard and checkpoints. V-RB-01 is in `test_rigid_validation.py`."""

import math

import numpy as np
import pytest

from entropy.sim.simulation import PacedSolver, Simulation
from entropy.solvers.interface import NonFiniteStateError, Solver
from entropy.solvers.rigid import (
    MAX_SUBSTEP_S,
    Body,
    RigidSetup,
    RigidSolver,
    Shape,
    load_dump,
    mass_properties,
)
from entropy.world.maps import Environment
from entropy.world.sdf import HalfSpace, Solids

GRAVITY = Environment().gravity_m_per_s2
TEST_DENSITY_KG_PER_M3 = 600.0  # any density will do; no material is meant


def _solver(device: str, tmp_path=None, gravity=GRAVITY) -> RigidSolver:
    return RigidSolver.create(
        RigidSetup(gravity_m_per_s2=gravity, max_bodies=64, dump_dir=tmp_path), device
    )


def _box(**kw) -> Body:
    base = {
        "shape": Shape.BOX,
        "size_m": (0.3, 0.2, 0.1),
        "density_kg_per_m3": TEST_DENSITY_KG_PER_M3,
        "position_m": (0.0, 0.0, 5.0),
    }
    base.update(kw)
    return Body(**base)


def test_the_rigid_solver_is_a_solver_and_paced() -> None:
    solver = _solver("cpu")
    assert isinstance(solver, Solver)
    assert isinstance(solver, PacedSolver)


def test_rigid_substeps_never_exceed_1_480_s(device: str) -> None:
    """SPEC 5.4: ⌈advance / (1/480 s)⌉ equal substeps."""
    solver = _solver(device)
    solver.add_bodies([_box()])
    for dt_s, expected in ((1.0 / 30.0, 16), (1.0 / 60.0, 8), (0.001, 1), (1.0 / 480.0, 1)):
        solver.step(dt_s)
        assert solver.diagnostics()["substeps"] == expected
        assert dt_s / expected <= MAX_SUBSTEP_S * (1.0 + 1e-9)


def test_the_rigid_solver_never_limits_the_world_s_advance() -> None:
    """Its substeps follow the advance (ADR 0013): no cap to hold the world to."""
    solver = _solver("cpu")
    solver.add_bodies([_box(velocity_m_per_s=(50.0, 0.0, 0.0))])
    solver.step(1.0 / 60.0)
    assert solver.max_advance_s() == math.inf


@pytest.mark.parametrize(
    ("shape", "size_m"),
    [
        (Shape.BOX, (0.3, 0.2, 0.1)),
        (Shape.SPHERE, (0.25, 0.0, 0.0)),
        (Shape.CAPSULE, (0.1, 0.3, 0.0)),
    ],
)
def test_mass_and_inertia_match_a_numerical_integral(shape: Shape, size_m) -> None:
    """Mass and principal moments (body frame) against a Monte Carlo integral of the
    shape: box half sizes (x, y, z); sphere radius; capsule radius and the half length of
    its cylinder along z."""
    mass_kg, inertia = mass_properties(shape, size_m, 1000.0)
    rng = np.random.default_rng(3)
    if shape == Shape.BOX:
        half = np.asarray(size_m)
    elif shape == Shape.SPHERE:
        half = np.full(3, size_m[0])
    else:
        half = np.array([size_m[0], size_m[0], size_m[0] + size_m[1]])
    points = rng.uniform(-half, half, (2_000_000, 3))
    if shape == Shape.SPHERE:
        inside = (points**2).sum(axis=1) <= size_m[0] ** 2
    elif shape == Shape.CAPSULE:
        z = np.clip(points[:, 2], -size_m[1], size_m[1])
        inside = points[:, 0] ** 2 + points[:, 1] ** 2 + (points[:, 2] - z) ** 2 <= size_m[0] ** 2
    else:
        inside = np.ones(len(points), dtype=bool)
    volume = np.prod(2.0 * half) * inside.mean()
    p = points[inside]
    expected_inertia = (
        1000.0
        * volume
        * np.array(
            [
                (p[:, 1] ** 2 + p[:, 2] ** 2).mean(),
                (p[:, 0] ** 2 + p[:, 2] ** 2).mean(),
                (p[:, 0] ** 2 + p[:, 1] ** 2).mean(),
            ]
        )
    )
    np.testing.assert_allclose(mass_kg, 1000.0 * volume, rtol=3e-3)
    np.testing.assert_allclose(inertia, expected_inertia, rtol=5e-3)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("size_m", (0.3, -0.2, 0.1), "size"),
        ("density_kg_per_m3", 0.0, "density"),
        ("orientation", (0.0, 0.0, 0.0, 0.0), "orientation"),
        ("position_m", (0.0, math.nan, 0.0), "position_m"),
        ("velocity_m_per_s", (math.inf, 0.0, 0.0), "velocity_m_per_s"),
        ("angular_velocity_rad_per_s", (0.0, 0.0, math.nan), "angular_velocity"),
    ],
)
def test_a_body_that_cannot_be_is_refused(field: str, value, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _box(**{field: value})


def test_the_capacity_takes_exactly_its_bodies(device: str) -> None:
    solver = _solver(device)
    with pytest.raises(ValueError, match="capacity"):
        solver.add_bodies([_box()] * 65)
    assert solver.body_count() == 0
    solver.add_bodies([_box()] * 64)
    with pytest.raises(ValueError, match="capacity"):
        solver.add_bodies([_box()])
    assert solver.body_count() == 64


def test_a_body_s_velocity_is_set_only_for_a_body_that_exists(device: str) -> None:
    solver = _solver(device)
    solver.add_bodies([_box(), _box()])
    solver.set_body_velocity(1, (1.0, 2.0, 3.0))
    np.testing.assert_allclose(solver.body_velocities()[1], [1.0, 2.0, 3.0])
    for index in (-1, 2):
        with pytest.raises(IndexError):
            solver.set_body_velocity(index, (0.0, 0.0, 0.0))


def test_an_orientation_is_stored_normalised(device: str) -> None:
    solver = _solver(device)
    solver.add_bodies([_box(orientation=(0.0, 0.0, 2.0, 2.0))])
    np.testing.assert_allclose(
        solver.body_orientations()[0], [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)], atol=1e-6
    )


def test_a_body_spinning_about_a_principal_axis_turns_steadily(device: str) -> None:
    """No gravity: 2 rad/s about z for 1 s turns the box by 2 rad about z, and its spin
    stays."""
    solver = _solver(device, gravity=(0.0, 0.0, 0.0))
    solver.add_bodies([_box(angular_velocity_rad_per_s=(0.0, 0.0, 2.0))])
    for _ in range(60):
        solver.step(1.0 / 60.0)
    x, y, z, w = solver.body_orientations()[0]
    angle = 2.0 * math.atan2(z, w)
    assert abs(x) < 1e-6 and abs(y) < 1e-6
    assert angle == pytest.approx(2.0, abs=2e-3)
    np.testing.assert_allclose(solver.body_angular_velocities()[0], [0.0, 0.0, 2.0], atol=1e-4)


def test_free_rotation_keeps_the_angular_momentum(device: str) -> None:
    """No gravity, no torque: a box spinning about no principal axis tumbles, and its
    angular momentum in world space, R I Rᵀ ω, stays within 1 % over 2 s (the gyroscopic
    term by the implicit midpoint rule, ADR 0030)."""
    solver = _solver(device, gravity=(0.0, 0.0, 0.0))
    solver.add_bodies([_box(angular_velocity_rad_per_s=(1.0, 3.0, 0.5))])
    _, inertia = mass_properties(Shape.BOX, (0.3, 0.2, 0.1), TEST_DENSITY_KG_PER_M3)

    def momentum() -> np.ndarray:
        x, y, z, w = solver.body_orientations()[0]
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        omega = solver.body_angular_velocities()[0]
        return rotation @ np.diag(inertia) @ rotation.T @ omega

    before = momentum()
    for _ in range(120):
        solver.step(1.0 / 60.0)
    after = momentum()
    assert np.linalg.norm(after - before) <= 0.01 * np.linalg.norm(before)
    assert np.abs(solver.body_angular_velocities()[0] - (1.0, 3.0, 0.5)).max() > 0.1  # it tumbles


def _state(solver: RigidSolver) -> dict[str, np.ndarray]:
    return {key: value for key, value in solver.checkpoint().items() if key != "count"}


def _momentum_and_energy(solver: RigidSolver, inertia: np.ndarray) -> tuple[np.ndarray, float]:
    """World angular momentum R I Rᵀ ω and rotational energy ½ ωᵀ R I Rᵀ ω of body 0."""
    x, y, z, w = solver.body_orientations()[0].astype(np.float64)
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    omega = solver.body_angular_velocities()[0].astype(np.float64)
    world_inertia = rotation @ np.diag(inertia) @ rotation.T
    return world_inertia @ omega, 0.5 * omega @ world_inertia @ omega


@pytest.mark.parametrize(
    "omega", [(5.0, 18.0, 3.0), (1.0, 1.0, 30.0)], ids=["tumbling-20", "major-axis-30-wobble"]
)
def test_a_fast_spinning_body_keeps_its_momentum_and_energy(device: str, omega) -> None:
    """No gravity, no torque, 10 s: |L| and the rotational energy within 1e-3. An explicit
    gyroscopic step gains energy every substep (+57 % of |L| in 10 s at 20 rad/s, #98's
    review); the implicit midpoint keeps both."""
    solver = _solver(device, gravity=(0.0, 0.0, 0.0))
    solver.add_bodies([_box(angular_velocity_rad_per_s=omega)])
    _, inertia = mass_properties(Shape.BOX, (0.3, 0.2, 0.1), TEST_DENSITY_KG_PER_M3)
    momentum, energy = _momentum_and_energy(solver, inertia)

    for _ in range(600):
        solver.step(1.0 / 60.0)

    momentum_after, energy_after = _momentum_and_energy(solver, inertia)
    assert abs(np.linalg.norm(momentum_after) / np.linalg.norm(momentum) - 1.0) <= 1e-3
    assert abs(energy_after / energy - 1.0) <= 1e-3


@pytest.mark.parametrize("speed", [600.0, 1500.0, 3000.0])
def test_a_thin_plank_spinning_very_fast_keeps_its_momentum_and_energy(
    device: str, speed: float
) -> None:
    """A plank 8 x 2 x 0.5 cm (its moments of inertia 15 to 1) spinning about a generic
    axis at 600 to 3,000 rad/s, as a hard hit on the floor can spin it, no gravity, 1 s:
    |L| and the rotational energy within 1e-3. Solved over the whole substep, the
    gyroscopic step lost 41 % of the speed at 600 rad/s and went non-finite from about
    1,500 rad/s (#101's fourth review)."""
    solver = _solver(device, gravity=(0.0, 0.0, 0.0))
    axis = np.array([0.6, 0.3, 0.74])
    axis /= np.linalg.norm(axis)
    size = (0.04, 0.01, 0.0025)
    solver.add_bodies([Body(Shape.BOX, size, TEST_DENSITY_KG_PER_M3,
                            angular_velocity_rad_per_s=tuple(speed * axis))])  # fmt: skip
    _, inertia = mass_properties(Shape.BOX, size, TEST_DENSITY_KG_PER_M3)
    momentum, energy = _momentum_and_energy(solver, inertia)

    for _ in range(60):
        solver.step(1.0 / 60.0)

    solver.max_advance_s()  # raises if the last step left a non-finite state
    momentum_after, energy_after = _momentum_and_energy(solver, inertia)
    assert abs(np.linalg.norm(momentum_after) / np.linalg.norm(momentum) - 1.0) <= 1e-3
    assert abs(energy_after / energy - 1.0) <= 1e-3


def test_a_spin_faster_than_half_a_turn_per_substep_keeps_its_sense(device: str) -> None:
    """1,600 rad/s turns more than π in a substep of 1/480 s: the turn's quaternion has a
    negative w, which must not flip the spin when the velocity is recovered (#98)."""
    solver = _solver(device, gravity=(0.0, 0.0, 0.0))
    solver.add_bodies(
        [Body(Shape.SPHERE, (0.1, 0.0, 0.0), 1000.0, angular_velocity_rad_per_s=(0, 0, 1600.0))]
    )

    solver.step(1.0 / 480.0)

    np.testing.assert_allclose(solver.body_angular_velocities()[0], [0.0, 0.0, 1600.0], rtol=1e-4)


def test_a_checkpoint_restores_every_array_exactly(device: str) -> None:
    """Every array, properties included, even after a body was added since."""
    solver = _solver(device)
    solver.add_bodies(
        [_box(), _box(position_m=(1.0, 0.0, 3.0), angular_velocity_rad_per_s=(0.0, 1.0, 0.0))]
    )
    solver.step(1.0 / 60.0)
    saved = solver.checkpoint()
    expected = _state(solver)
    for _ in range(10):
        solver.step(1.0 / 60.0)
    solver.add_bodies([Body(Shape.CAPSULE, (0.1, 0.4, 0.0), 2000.0, position_m=(3.0, 0.0, 1.0))])

    solver.restore(saved)

    assert solver.body_count() == 2
    restored = _state(solver)
    assert restored.keys() == expected.keys()
    for name, values in expected.items():
        np.testing.assert_array_equal(restored[name], values, err_msg=name)


def test_a_checkpoint_larger_than_the_capacity_is_refused(device: str) -> None:
    big = RigidSolver.create(RigidSetup(gravity_m_per_s2=GRAVITY, max_bodies=8), device)
    big.add_bodies([_box()] * 8)
    small = RigidSolver.create(RigidSetup(gravity_m_per_s2=GRAVITY, max_bodies=4), device)
    with pytest.raises(ValueError, match="capacity"):
        small.restore(big.checkpoint())


@pytest.mark.parametrize(
    ("field", "row"),
    [
        ("velocities_m_per_s", (math.nan, 0.0, 0.0)),
        ("angular_velocities_rad_per_s", (0.0, math.inf, 0.0)),
    ],
)
def test_a_non_finite_body_freezes_the_solver_and_dumps_its_frame(
    device: str, tmp_path, field: str, row
) -> None:
    """SPEC 5.9: the guard reads the step's status at the next call and raises with a
    dump of the frame's starting state, its result, its time step and gravity; it keeps
    raising, and refuses new bodies and velocities, until a restore."""
    solver = _solver(device, tmp_path)
    solver.add_bodies([_box(), _box(position_m=(1.0, 0.0, 5.0))])
    saved = solver.checkpoint()
    broken = {key: np.array(value, copy=True) for key, value in saved.items()}
    broken[field][1] = row
    solver.restore(broken)
    start = solver.body_positions()
    solver.step(1.0 / 60.0)

    with pytest.raises(NonFiniteStateError) as raised:
        solver.step(1.0 / 60.0)
    with pytest.raises(NonFiniteStateError):
        solver.max_advance_s()
    with pytest.raises(NonFiniteStateError):
        solver.add_bodies([_box()])
    with pytest.raises(NonFiniteStateError):
        solver.set_body_velocity(0, (0.0, 0.0, 0.0))

    dump = load_dump(raised.value.dump_path)
    assert float(dump["dt_s"]) == pytest.approx(1.0 / 60.0)
    np.testing.assert_allclose(dump["gravity_m_per_s2"], GRAVITY)
    np.testing.assert_array_equal(dump["before_positions_m"], start)
    assert not np.isfinite(dump[field][1]).all() or not np.isfinite(dump["positions_m"][1]).all()
    assert np.isfinite(dump["positions_m"][0]).all()
    solver.restore(saved)
    solver.step(1.0 / 60.0)  # runs again


def test_a_dump_replays_its_frame(tmp_path) -> None:
    """A dump holds the frame's starting state (the friction anchors included), the static
    solids and the room for pairs: `RigidSolver.from_dump` rebuilds the solver at that
    state, and stepping the dump's time step gives its result bit for bit on the CPU,
    contacts and friction included (#101's review). A dump is a regression test."""
    floor = Solids((HalfSpace((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),)).to_device("cpu")
    setup = RigidSetup(gravity_m_per_s2=GRAVITY, max_bodies=64, dump_dir=tmp_path,
                       solids=floor, max_pairs=7, static_friction=0.5,
                       dynamic_friction=0.5)  # fmt: skip
    solver = RigidSolver.create(setup, "cpu")
    mu = {"static_friction": 0.5, "dynamic_friction": 0.5}
    solver.add_bodies(
        [
            _box(position_m=(0.0, 0.0, 0.1), **mu),
            _box(position_m=(0.02, 0.0, 0.3), angular_velocity_rad_per_s=(0.1, 0.2, 0.3), **mu),
            _box(position_m=(3.0, 0.0, 5.0)),
        ]
    )
    for _ in range(20):
        solver.step(1.0 / 60.0)
    solver.set_body_velocity(2, (math.nan, 0.0, 0.0))
    solver.step(1.0 / 30.0)
    with pytest.raises(NonFiniteStateError) as raised:
        solver.step(1.0 / 60.0)
    dump = load_dump(raised.value.dump_path)

    replay = RigidSolver.from_dump(dump, "cpu")
    replay.step(float(dump["dt_s"]))

    assert replay.setup.max_pairs == 7
    np.testing.assert_array_equal(replay.body_positions()[:2], dump["positions_m"][:2])
    np.testing.assert_array_equal(replay.body_orientations()[:2], dump["orientations"][:2])


def test_a_nan_orientation_is_caught(device: str, tmp_path) -> None:
    """A NaN orientation is caught by the NaN guard at the next call: it reaches the
    angular velocity, recovered from the substep's rotation vectors, even where Warp's
    normalize would turn the quaternion itself back into the identity."""
    floor = Solids((HalfSpace((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),)).to_device(device)
    solver = RigidSolver.create(
        RigidSetup(gravity_m_per_s2=GRAVITY, max_bodies=4, dump_dir=tmp_path, solids=floor),
        device,
    )
    solver.add_bodies([_box(position_m=(0.0, 0.0, 0.1))])
    state = solver.checkpoint()
    state["orientations"][0] = math.nan
    solver.restore(state)
    solver.step(1.0 / 60.0)

    with pytest.raises(NonFiniteStateError):
        solver.step(1.0 / 60.0)


def test_a_restore_drops_a_status_not_yet_read(device: str) -> None:
    """A step that went non-finite, then a restore before the next call: the old status
    must not freeze the restored state."""
    solver = _solver(device)
    solver.add_bodies([_box()])
    saved = solver.checkpoint()
    solver.set_body_velocity(0, (math.nan, 0.0, 0.0))
    solver.step(1.0 / 60.0)

    solver.restore(saved)

    solver.step(1.0 / 60.0)
    assert np.isfinite(solver.body_positions()).all()


def test_the_simulation_paces_the_rigid_solver_with_the_liquid_s_advance() -> None:
    """One shared advance per frame (ADR 0013): the rigid solver steps by it."""
    solver = _solver("cpu")
    solver.add_bodies([_box()])
    simulation = Simulation([solver])

    advance = simulation.step(1.0 / 30.0)

    assert advance == pytest.approx(1.0 / 30.0)
    assert solver.diagnostics()["substeps"] == 16
