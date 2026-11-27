"""Friction and restitution of the rigid contacts (`entropy.solvers.rigid`, SPEC 5.4, #88,
ADR 0032). V-RB-02 and V-RB-03 are in `test_rigid_validation.py`."""

import math

import numpy as np
import pytest

from entropy.solvers.rigid import Body, RigidSetup, RigidSolver, Shape
from entropy.world.maps import Environment
from entropy.world.sdf import HalfSpace, Solids

GRAVITY = Environment().gravity_m_per_s2
G = -GRAVITY[2]
FLOOR = Solids((HalfSpace((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),))


def _solver(
    device: str, solids: Solids | None = FLOOR, gravity=GRAVITY, max_bodies: int = 16, **surface
) -> RigidSolver:
    setup = RigidSetup(
        gravity_m_per_s2=gravity,
        max_bodies=max_bodies,
        solids=solids.to_device(device) if solids is not None else None,
        **surface,
    )
    return RigidSolver.create(setup, device)


def test_a_sliding_box_stops_where_kinetic_friction_says(device: str) -> None:
    """A box sliding at 2 m/s on the floor, μd = 0.3 for both: it stops after v²/(2μg) =
    0.680 m, within 5 %, and stays stopped."""
    solver = _solver(device, static_friction=0.3, dynamic_friction=0.3)
    solver.add_bodies([Body(Shape.BOX, (0.2, 0.2, 0.1), 600.0, position_m=(0.0, 0.0, 0.1),
                            velocity_m_per_s=(2.0, 0.0, 0.0), static_friction=0.3,
                            dynamic_friction=0.3)])  # fmt: skip

    for _ in range(90):
        solver.step(1.0 / 60.0)

    assert solver.body_positions()[0, 0] == pytest.approx(2.0**2 / (2 * 0.3 * G), rel=0.05)
    assert np.abs(solver.body_velocities()[0]).max() < 1e-3


def test_a_box_kicked_gently_stops_where_kinetic_friction_says(device: str) -> None:
    """A 20 cm box resting on the floor, kicked to 0.2 m/s, μ = 0.5: it slides v²/(2μg) =
    4.08 mm, within 5 %, and stays there. The anchor keeps the sliding left within its
    0.2 mm window and undoes it later: with a 2 mm window, the box sprang back to 2.1 mm
    (#102's second review)."""
    solver = _solver(device, static_friction=0.5, dynamic_friction=0.5)
    solver.add_bodies([_box((0.0, 0.0, 0.1))])
    for _ in range(30):
        solver.step(1.0 / 60.0)
    start = float(solver.body_positions()[0, 0])
    solver.set_body_velocity(0, (0.2, 0.0, 0.0))

    for _ in range(60):
        solver.step(1.0 / 60.0)

    travelled = float(solver.body_positions()[0, 0]) - start
    assert travelled == pytest.approx(0.2**2 / (2.0 * 0.5 * G), rel=0.05)


def test_without_friction_a_box_keeps_sliding(device: str) -> None:
    """The defaults: no friction and no bounce unless a body or the surface has some."""
    solver = _solver(device)
    solver.add_bodies([Body(Shape.BOX, (0.2, 0.2, 0.1), 600.0, position_m=(0.0, 0.0, 0.1),
                            velocity_m_per_s=(2.0, 0.0, 0.0))])  # fmt: skip

    for _ in range(60):
        solver.step(1.0 / 60.0)

    assert solver.body_velocities()[0, 0] == pytest.approx(2.0, rel=1e-3)


def test_the_pair_takes_the_geometric_mean_of_friction_and_the_larger_restitution(
    device: str,
) -> None:
    """ADR 0032: μ = √(μa μb) for both frictions, e = max(ea, eb). A box at μd = 0.6 on a
    floor at 0.15 slides as at μd = 0.3; a sphere at e = 0.2 dropped on a floor at 0.6
    bounces as at e = 0.6."""
    solver = _solver(device, static_friction=0.15, dynamic_friction=0.15, restitution=0.6)
    solver.add_bodies(
        [
            Body(Shape.BOX, (0.2, 0.2, 0.1), 600.0, position_m=(0.0, 0.0, 0.1),
                 velocity_m_per_s=(2.0, 0.0, 0.0), static_friction=0.6, dynamic_friction=0.6),
            Body(Shape.SPHERE, (0.1, 0.0, 0.0), 600.0, position_m=(5.0, 0.0, 1.1),
                 restitution=0.2),
        ]
    )  # fmt: skip
    peak = 0.0
    landed = False
    for _ in range(150):
        solver.step(1.0 / 60.0)
        z, vz = solver.body_positions()[1, 2], solver.body_velocities()[1, 2]
        landed = landed or vz > 0.0
        if landed:
            peak = max(peak, z - 0.1)

    assert solver.body_positions()[0, 0] == pytest.approx(2.0**2 / (2 * 0.3 * G), rel=0.05)
    assert peak == pytest.approx(0.6**2 * 1.0, rel=0.05)


def test_a_box_dropped_flat_bounces_as_a_sphere_does(device: str) -> None:
    """Restitution acts once per contact patch, at its centre of pressure: a box dropped
    flat from 1 m with e = 0.5 rises to e² of its fall, within 5 %, without tipping."""
    solver = _solver(device, restitution=0.5)
    solver.add_bodies([Body(Shape.BOX, (0.2, 0.2, 0.1), 600.0, position_m=(0.0, 0.0, 1.1),
                            restitution=0.5)])  # fmt: skip
    peak, rising = 0.0, False
    for _ in range(90):
        solver.step(1.0 / 60.0)
        z, vz = solver.body_positions()[0, 2], solver.body_velocities()[0, 2]
        rising = rising or vz > 0.0
        if rising:
            peak = max(peak, z - 0.1)
            if vz < 0.0:
                break

    assert peak == pytest.approx(0.25, rel=0.05)
    assert np.abs(solver.body_angular_velocities()[0]).max() < 1e-3


def test_a_ball_bounces_on_a_stack_as_on_the_floor(device: str) -> None:
    """A ball dropped 1 m onto a heavy slab resting on another, e = 0.8: it rises e² of
    its fall within 10 %, as on the floor. The bounce is shared among the patches that
    bounce, not with the slab's resting one: shared with it, the ball rose 2 cm (#102's
    second review)."""
    solver = _solver(device, restitution=0.8)
    slab = {"half": (0.5, 0.5, 0.05), "density": 2400.0, "restitution": 0.8}
    solver.add_bodies([_box((0.0, 0.0, 0.05), **slab), _box((0.0, 0.0, 0.15), **slab),
                       Body(Shape.SPHERE, (0.1, 0.0, 0.0), 600.0, position_m=(0.0, 0.0, 1.3),
                            restitution=0.8)])  # fmt: skip
    peak, rising = 0.0, False
    for _ in range(2 * 480):
        solver.step(1.0 / 480.0)
        z, vz = solver.body_positions()[2, 2], solver.body_velocities()[2, 2]
        rising = rising or vz > 0.0
        if rising:
            peak = max(peak, z - 0.3)
            if vz < 0.0:
                break

    assert peak == pytest.approx(0.8**2, rel=0.1)


@pytest.mark.parametrize("falling", [2, 0], ids=["falling-last", "falling-first"])
def test_a_box_landing_on_two_boxes_at_once_gains_no_energy(device: str, falling: int) -> None:
    """A 20 cm box dropped 1 m flat across two heavy boxes, e = 0.8: both patches bounce
    in the same substep and share the bounce, so the box rises no higher than e² of its
    fall (0.27 m: a named limit, ADR 0032), whichever index it has. Each giving the whole
    bounce, it rose 2.02 m (#102's second review)."""
    solver = _solver(device, restitution=0.8)
    under = [_box((x, 0.0, 0.1), density=5000.0, restitution=0.8) for x in (-0.1001, 0.1001)]
    top = _box((0.0, 0.0, 1.3), restitution=0.8)
    solver.add_bodies([*under, top] if falling == 2 else [top, *under])
    peak, rising = 0.0, False
    for _ in range(2 * 480):
        solver.step(1.0 / 480.0)
        z, vz = solver.body_positions()[falling, 2], solver.body_velocities()[falling, 2]
        rising = rising or vz > 0.0
        if rising:
            peak = max(peak, z - 0.3)
            if vz < 0.0:
                break

    assert 0.0 < peak <= 0.8**2


def test_friction_between_bodies_keeps_momentum(device: str) -> None:
    """No gravity: a sphere spinning at 20 rad/s grazes a box, both with friction. The
    friction pushes the box sideways (0.24 m/s; exactly 0 without friction) and slows the
    spin, and its impulses are given to both alike: momentum within 1e-4."""
    solver = _solver(device, solids=None, gravity=(0.0, 0.0, 0.0))
    solver.add_bodies(
        [
            Body(Shape.SPHERE, (0.2, 0.0, 0.0), 800.0, position_m=(-1.0, 0.1, 0.0),
                 velocity_m_per_s=(3.0, 0.0, 0.0), angular_velocity_rad_per_s=(0.0, 0.0, 20.0),
                 static_friction=0.6, dynamic_friction=0.5, restitution=0.3),
            Body(Shape.BOX, (0.3, 0.3, 0.3), 500.0, position_m=(0.5, 0.0, 0.0),
                 static_friction=0.6, dynamic_friction=0.5, restitution=0.3),
        ]
    )  # fmt: skip
    masses = solver.checkpoint()["masses_kg"][:, None].astype(np.float64)
    before = (masses * solver.body_velocities()).sum(axis=0)

    for _ in range(60):
        solver.step(1.0 / 60.0)

    after = (masses * solver.body_velocities()).sum(axis=0)
    np.testing.assert_allclose(after, before, atol=1e-4 * np.abs(before).sum())
    assert solver.body_velocities()[1, 1] > 0.1  # friction pushed the box sideways
    assert solver.body_angular_velocities()[0, 2] < 15.0  # and slowed the sphere's spin


def _kinetic_energy(solver: RigidSolver) -> float:
    """The bodies' kinetic energy (J), linear and angular."""
    state = solver.checkpoint()
    m = state["masses_kg"].astype(np.float64)
    inertia = state["inertia_kg_m2"].astype(np.float64)
    v = solver.body_velocities().astype(np.float64)
    w = solver.body_angular_velocities().astype(np.float64)
    total = 0.5 * float((m * (v * v).sum(axis=1)).sum())
    for i, q in enumerate(solver.body_orientations()):
        spin = _rotation(q).T @ w[i]
        total += 0.5 * float(spin @ (inertia[i] * spin))
    return total


@pytest.mark.parametrize("tilt_deg", [0.0, 45.0])
def test_friction_gives_no_energy_to_bodies_pushed_out_of_an_overlap(
    device: str, tilt_deg: float
) -> None:
    """Two 20 cm boxes spawned 1 cm into each other at rest, the second turned, μ = 0.5,
    no gravity: the depths push them apart with no speed to speak of, as without friction
    (ADR 0031). Friction must not take the turn the depth corrections give for sliding:
    cancelling it, it flung them apart with 24 J and 18 rad/s at 45° (#102's second
    review)."""
    solver = _solver(device, solids=None, gravity=(0.0, 0.0, 0.0))
    half = math.radians(tilt_deg) / 2.0
    axis = np.array([1.0, 0.7, 0.3]) / np.linalg.norm([1.0, 0.7, 0.3])
    turn = (*(math.sin(half) * axis), math.cos(half))
    solver.add_bodies([_box((0.0, 0.0, 0.0)), _box((0.19, 0.03, 0.05), orientation=turn)])

    solver.step(1.0 / 60.0)

    assert _kinetic_energy(solver) < 0.05


def _rotation(q) -> np.ndarray:
    """The rotation matrix of a unit quaternion (x, y, z, w)."""
    x, y, z, w = (float(c) for c in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


@pytest.mark.parametrize("value", [-0.1, 1.5, math.nan])
def test_a_restitution_outside_0_to_1_is_refused(value: float) -> None:
    with pytest.raises(ValueError, match="restitution"):
        Body(Shape.SPHERE, (0.1, 0.0, 0.0), 600.0, restitution=value)
    with pytest.raises(ValueError, match="restitution"):
        RigidSetup(gravity_m_per_s2=GRAVITY, restitution=value)


@pytest.mark.parametrize("value", [-0.1, math.inf, math.nan])
def test_a_negative_or_infinite_friction_is_refused(value: float) -> None:
    with pytest.raises(ValueError, match="friction"):
        Body(Shape.SPHERE, (0.1, 0.0, 0.0), 600.0, dynamic_friction=value)
    with pytest.raises(ValueError, match="friction"):
        RigidSetup(gravity_m_per_s2=GRAVITY, static_friction=value)


MU = {"static_friction": 0.5, "dynamic_friction": 0.5}


def _box(position_m, half=(0.1, 0.1, 0.1), density=600.0, **kw) -> Body:
    return Body(Shape.BOX, half, density, position_m=position_m, **{**MU, **kw})


def _drift(solver: RigidSolver, seconds: float) -> float:
    """How far the bodies moved at most over `seconds` (m), after 1 s to settle."""
    return _drift_and_turn(solver, seconds)[0]


def _drift_and_turn(solver: RigidSolver, seconds: float) -> tuple[float, float]:
    """How far the bodies moved (m) and turned (degrees) at most over `seconds`, after
    1 s to settle."""
    for _ in range(60):
        solver.step(1.0 / 60.0)
    settled = solver.body_positions().copy()
    turned = solver.body_orientations().copy()
    for _ in range(round(seconds * 60)):
        solver.step(1.0 / 60.0)
    moved = float(np.linalg.norm(solver.body_positions() - settled, axis=1).max())
    dots = np.abs((solver.body_orientations().astype(np.float64) * turned).sum(axis=1))
    return moved, float(np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0))).max())


@pytest.mark.parametrize(
    ("offset_m", "upper_first"),
    [(0.01, False), (0.04, False), (0.08, False), (0.04, True)],
    ids=["1cm", "4cm", "8cm", "4cm-upper-first"],
)
def test_with_friction_a_box_off_centre_on_another_stays(
    device: str, offset_m: float, upper_first: bool
) -> None:
    """A box resting on an equal one, offset by 1, 4 or 8 cm, μ = 0.5: it stays for 10 s
    (moves under 1 mm), where without friction it slides off (ADR 0031); also with the
    upper box listed first, which takes the other branch of the pair's rule."""
    solver = _solver(device, static_friction=0.5, dynamic_friction=0.5)
    boxes = [_box((0.0, 0.0, 0.1)), _box((offset_m, 0.0, 0.3))]
    solver.add_bodies(boxes[::-1] if upper_first else boxes)

    assert _drift(solver, 10.0) < 1e-3


def test_with_friction_a_pyramid_stands(device: str) -> None:
    """A 4-3-2-1 pyramid of boxes, each row resting on the joints of the one below, μ =
    0.5: it stands for 10 s, every box moving under 1 mm and turning under 0.1° (without
    friction it collapses, ADR 0031). With friction's torque about the normal given on
    resting patches, the boxes turned up to 1.9° in 30 s (#102's third review)."""
    solver = _solver(device, static_friction=0.5, dynamic_friction=0.5)
    boxes = []
    for row, count in enumerate((4, 3, 2, 1)):
        for k in range(count):
            boxes.append(_box((0.2 * (k - (count - 1) / 2.0), 0.0, 0.1 + 0.2 * row)))
    solver.add_bodies(boxes)

    moved, turned = _drift_and_turn(solver, 10.0)
    assert moved < 1e-3
    assert turned < 0.1


@pytest.mark.parametrize("scene", ["board", "bridge"])
def test_with_friction_a_heavy_cube_rests_on_a_wooden_plank(device: str, scene: str) -> None:
    """A heavy 20 cm cube resting near the end of a long wooden plank, μ = 0.5: concrete
    0.85 m out on a 2 m board lying on the floor, or steel 0.6 m out on a plank laid
    across two blocks. For 10 s no body moves 1 mm or turns 0.1°. With friction's torque
    about the normal given on resting patches, the cube on the board slid 153 mm and
    turned 90° (#102's third review)."""
    solver = _solver(device, max_bodies=8, static_friction=0.5, dynamic_friction=0.5)
    if scene == "board":
        bodies = [_box((0.0, 0.0, 0.01), half=(1.0, 0.1, 0.01)),
                  _box((0.85, 0.0, 0.121), density=2450.0)]  # fmt: skip
    else:
        bodies = [_box((-0.9, 0.0, 0.1)), _box((0.9, 0.0, 0.1)),
                  _box((0.0, 0.0, 0.225), half=(1.0, 0.1, 0.025)),
                  _box((0.6, 0.0, 0.351), density=7850.0)]  # fmt: skip
    solver.add_bodies(bodies)

    moved, turned = _drift_and_turn(solver, 10.0)
    assert moved < 1e-3
    assert turned < 0.1


def test_with_friction_a_crate_under_a_heavy_slab_stays(device: str) -> None:
    """A 20 cm crate under a concrete-heavy slab 64 times its mass, μ = 0.5: it is not
    squeezed out in 10 s (without friction it is, within 2 s, ADR 0031)."""
    solver = _solver(device, static_friction=0.5, dynamic_friction=0.5)
    solver.add_bodies(
        [_box((0.0, 0.0, 0.1)), _box((0.0, 0.0, 0.3), half=(0.4, 0.4, 0.1), density=2400.0)]
    )

    assert _drift(solver, 10.0) < 1e-3


@pytest.mark.parametrize("margin_deg", [3.0, 1.0])
def test_with_friction_a_box_on_a_box_holds_on_a_slope(device: str, margin_deg: float) -> None:
    """Two boxes stacked on an incline 3° or 1° below the friction angle (μ = 0.5): neither
    moves by more than 1 mm in 5 s. The upper one rests on a wide patch: friction moves
    it without turning either body. The lower one's load from the pair varies from
    substep to substep, and a substep past its bound is clamped to it: let go there, the
    stack slid 64 mm at 1° (#102's review)."""
    theta = math.atan(0.5) - math.radians(margin_deg)
    normal = np.array([0.0, -math.sin(theta), math.cos(theta)])
    slope = Solids((HalfSpace((0.0, 0.0, 0.0), tuple(normal)),))
    solver = _solver(device, solids=slope, static_friction=0.5, dynamic_friction=0.5)
    tilt = (math.sin(theta / 2.0), 0.0, 0.0, math.cos(theta / 2.0))
    solver.add_bodies([_box(tuple(0.1 * normal), orientation=tilt),
                       _box(tuple(0.3 * normal), orientation=tilt)])  # fmt: skip

    assert _drift(solver, 5.0) < 1e-3


def _tilting_plank(device: str, plank: int, yaw_deg: float, until_deg: float, along_m: float = 0.0):
    """A 20 cm box (turned by `yaw_deg` about z, `along_m` along the plank from its
    middle) resting on a heavy plank that turns about its centre at 0.3 rad/s from 0.5 s
    on, to `until_deg`, driven as a tool would hold it, μ = 0.6 (31°), no floor; `plank`
    is its index. Yields (the plank's angle in degrees, the box's centre in the plank's
    frame) after each 1/480 s step."""
    solver = _solver(device, solids=None, static_friction=0.6, dynamic_friction=0.6)
    half = math.radians(yaw_deg) / 2.0
    bodies = [
        _box((0.0, 0.0, 0.0), half=(1.0, 0.5, 0.05), density=1.0e5,
             static_friction=0.6, dynamic_friction=0.6),
        _box((along_m, 0.0, 0.15), orientation=(0.0, 0.0, math.sin(half), math.cos(half)),
             static_friction=0.6, dynamic_friction=0.6),
    ]  # fmt: skip
    solver.add_bodies(bodies if plank == 0 else bodies[::-1])
    box = 1 - plank
    h, rate = 1.0 / 480.0, 0.3
    for k in range(round((0.5 + math.radians(until_deg) / rate) / h)):
        solver.step(h)
        angle = rate * max(0.0, (k + 1) * h - 0.5)
        state = solver.checkpoint()
        state["positions_m"][plank] = 0.0
        state["orientations"][plank] = (0.0, math.sin(angle / 2.0), 0.0, math.cos(angle / 2.0))
        # Held still: the next substep's gravity is taken back in advance, so the plank
        # does not fall and press the box twice as hard as its weight.
        state["velocities_m_per_s"][plank] = -np.asarray(GRAVITY) * h
        state["angular_velocities_rad_per_s"][plank] = (0.0, rate if angle > 0.0 else 0.0, 0.0)
        solver.restore(state)
        yield (
            math.degrees(angle),
            _rotation(state["orientations"][plank]).T @ (solver.body_positions()[box]),
        )


@pytest.mark.parametrize(
    ("plank", "yaw_deg", "along_m"),
    [(0, 0.0, 0.0), (1, 30.0, 0.3)],
    ids=["plank-first", "box-first-turned-off-centre"],
)
def test_with_friction_a_box_rides_a_tilting_plank(
    device: str, plank: int, yaw_deg: float, along_m: float
) -> None:
    """A box on a plank tilting to 20°, under its friction angle of 31°: it rides the
    plank, moving under 1 mm in the plank's frame, whichever of the two has the lower
    index, also turned and 30 cm from the axis. Static friction holds the upper body's
    centre where the lower body carries it: held above the anchor in the world instead,
    the box slid 35 mm uphill, its height times the tilt (#102's review); carried in the
    upper body's frame, the turned box 3.5 mm (its second review)."""
    start, now = None, np.zeros(3)
    for angle, now in _tilting_plank(device, plank, yaw_deg, 20.0, along_m):
        if start is None and angle > 0.0:
            start = now

    assert start is not None
    assert np.linalg.norm(now - start) < 1e-3


def test_a_box_slides_off_a_plank_tilting_past_its_friction_angle(device: str) -> None:
    """The same plank tilting on to 40°: the box starts sliding (moves 1 mm on the plank)
    between 29 and 35°, its friction angle being 31°. With the plank's fall each substep
    pressing it twice as hard, the box rode a plank held at 40° (#102's second
    review)."""
    start, broke = None, None
    for angle, now in _tilting_plank(device, 0, 0.0, 40.0):
        if start is None and angle > 0.0:
            start = now
        if start is not None and broke is None and np.linalg.norm(now - start) > 1e-3:
            broke = angle

    assert broke is not None
    assert 29.0 < broke < 35.0


def test_a_block_breaking_free_slides_with_kinetic_friction(device: str) -> None:
    """A block released on an incline 1° steeper than its static friction angle, μs = 0.6
    and μd = 0.4: once it breaks free, friction is μd's, so after 1 s it slides at g (sin
    θ - μd cos θ) t = 1.86 m/s, within 3 %. Clamped at μs while slow, it reached 1.70 m/s
    (#102's reviews)."""
    theta = math.atan(0.6) + math.radians(1.0)
    normal = (0.0, -math.sin(theta), math.cos(theta))
    slope = Solids((HalfSpace((0.0, 0.0, 0.0), normal),))
    solver = _solver(device, solids=slope, static_friction=0.6, dynamic_friction=0.4)
    tilt = (math.sin(theta / 2.0), 0.0, 0.0, math.cos(theta / 2.0))
    solver.add_bodies([Body(Shape.BOX, (0.2, 0.2, 0.05), 600.0, orientation=tilt,
                            position_m=tuple(0.05 * np.asarray(normal)),
                            static_friction=0.6, dynamic_friction=0.4)])  # fmt: skip

    for _ in range(60):
        solver.step(1.0 / 60.0)

    speed = float(np.linalg.norm(solver.body_velocities()[0]))
    assert speed == pytest.approx(G * (math.sin(theta) - 0.4 * math.cos(theta)), rel=0.03)


def test_a_box_spun_flat_on_the_floor_stops_turning(device: str) -> None:
    """A 40 cm box spun flat at 10 rad/s on the floor, μ = 0.5: the friction against
    turning about the normal stops it in 0.36 s under an even pressure; its samples weigh
    the corners more, so it stops about 1.4 times sooner (ADR 0032). Without that friction
    it would spin on: the centre of pressure does not slide."""
    solver = _solver(device, static_friction=0.5, dynamic_friction=0.5)
    solver.add_bodies([_box((0.0, 0.0, 0.1), half=(0.2, 0.2, 0.1),
                            angular_velocity_rad_per_s=(0.0, 0.0, 10.0))])  # fmt: skip
    stopped = None
    for k in range(60):
        solver.step(1.0 / 120.0)
        if stopped is None and abs(solver.body_angular_velocities()[0, 2]) < 0.01:
            stopped = (k + 1) / 120.0

    assert stopped is not None
    assert 0.2 <= stopped <= 0.45


def test_a_box_slides_on_a_held_box_as_on_a_slope(device: str) -> None:
    """A 20 cm box on a broad one held on an incline 1° above the pair's friction angle (μ
    = 0.5 between them, 1.0 under the broad one): the upper box slides on the lower at g
    (sin θ - μ cos θ) = 0.1914 m/s², within 5 %, and the lower one stays. A pair at its
    static bound gets no dynamic friction: counted twice, it held the box (#102's
    review)."""
    theta = math.atan(0.5) + math.radians(1.0)
    normal = np.array([0.0, -math.sin(theta), math.cos(theta)])
    down = np.array([0.0, -math.cos(theta), -math.sin(theta)])
    slope = Solids((HalfSpace((0.0, 0.0, 0.0), tuple(normal)),))
    solver = _solver(device, solids=slope, static_friction=2.0, dynamic_friction=2.0)
    tilt = (math.sin(theta / 2.0), 0.0, 0.0, math.cos(theta / 2.0))
    solver.add_bodies(
        [
            _box(tuple(0.05 * normal), half=(0.3, 0.3, 0.05), orientation=tilt),
            _box(tuple(0.15 * normal), half=(0.1, 0.1, 0.05), orientation=tilt),
        ]
    )
    for _ in range(6):
        solver.step(1.0 / 60.0)
    lower = solver.body_positions()[0].copy()
    start = solver.body_velocities()

    for _ in range(60):
        solver.step(1.0 / 60.0)

    end = solver.body_velocities()
    gained = float(np.dot((end[1] - end[0]) - (start[1] - start[0]), down))
    assert gained == pytest.approx(G * (math.sin(theta) - 0.5 * math.cos(theta)), rel=0.05)
    assert np.linalg.norm(solver.body_positions()[0] - lower) < 1e-3


def _resting_pair() -> list[Body]:
    return [_box((0.0, 0.0, 0.1)), _box((0.04, 0.0, 0.3))]


def test_bodies_added_after_a_restore_start_without_anchors() -> None:
    """Restored to an empty checkpoint, a solver whose bodies had friction anchors gets
    new bodies 0.1 mm from the old ones: they start with none, and move as in a fresh
    solver, bit for bit on the CPU (ADR 0032). An anchor left on the floor from the old
    bodies would pull the new ones back to it."""
    used = _solver("cpu", static_friction=0.5, dynamic_friction=0.5)
    empty = used.checkpoint()
    used.add_bodies(_resting_pair())
    for _ in range(30):
        used.step(1.0 / 60.0)
    used.restore(empty)
    fresh = _solver("cpu", static_friction=0.5, dynamic_friction=0.5)

    for solver in (used, fresh):
        solver.add_bodies([_box((1e-4, 0.0, 0.1)), _box((0.04 + 1e-4, 0.0, 0.3))])
        for _ in range(30):
            solver.step(1.0 / 60.0)

    np.testing.assert_array_equal(used.body_positions(), fresh.body_positions())


def test_a_checkpoint_without_anchors_restores_none() -> None:
    """A checkpoint that carries no anchors (an older save) restores none: the stack moves
    on as a fresh solver given the same state does, bit for bit on the CPU."""
    used = _solver("cpu", static_friction=0.5, dynamic_friction=0.5)
    used.add_bodies(_resting_pair())
    for _ in range(30):
        used.step(1.0 / 60.0)
    state = {k: v for k, v in used.checkpoint().items() if "anchor" not in k}
    for _ in range(30):
        used.step(1.0 / 60.0)  # anchors that a restore must clear
    fresh = _solver("cpu", static_friction=0.5, dynamic_friction=0.5)

    for solver in (used, fresh):
        solver.restore(state)
        for _ in range(30):
            solver.step(1.0 / 60.0)

    np.testing.assert_array_equal(used.body_positions(), fresh.body_positions())


def test_a_checkpoint_keeps_the_friction_anchors() -> None:
    """The friction anchors are state (ADR 0032): restored from a checkpoint, a stack
    resting with friction on the floor moves on exactly as it did, bit for bit on the
    CPU."""
    solver = _solver("cpu", static_friction=0.5, dynamic_friction=0.5)
    solver.add_bodies([_box((0.0, 0.0, 0.1)), _box((0.04, 0.0, 0.3))])
    for _ in range(30):
        solver.step(1.0 / 60.0)
    saved = solver.checkpoint()
    for _ in range(30):
        solver.step(1.0 / 60.0)
    expected = solver.body_positions(), solver.body_orientations()

    solver.restore(saved)
    for _ in range(30):
        solver.step(1.0 / 60.0)

    np.testing.assert_array_equal(solver.body_positions(), expected[0])
    np.testing.assert_array_equal(solver.body_orientations(), expected[1])


def test_a_ball_rolls_down_a_slope_at_five_sevenths_g_sin_theta(device: str) -> None:
    """A sphere on a 20° slope with μ = 0.6 rolls without slipping: after 1 s its speed is
    (5/7) g sin θ t within 2 % and its spin v / r. Static friction holds the point that
    touches the slope now, not a point of the ball's surface, which rolls away from it."""
    theta = math.radians(20.0)
    normal = (0.0, -math.sin(theta), math.cos(theta))
    slope = Solids((HalfSpace((0.0, 0.0, 0.0), normal),))
    solver = _solver(device, solids=slope, static_friction=0.6, dynamic_friction=0.6)
    radius = 0.1
    solver.add_bodies([Body(Shape.SPHERE, (radius, 0.0, 0.0), 600.0,
                            position_m=tuple(radius * np.asarray(normal)),
                            static_friction=0.6, dynamic_friction=0.6)])  # fmt: skip

    for _ in range(60):
        solver.step(1.0 / 60.0)

    speed = float(np.linalg.norm(solver.body_velocities()[0]))
    assert speed == pytest.approx(5.0 / 7.0 * G * math.sin(theta) * 1.0, rel=0.02)
    spin = float(np.linalg.norm(solver.body_angular_velocities()[0]))
    assert spin == pytest.approx(speed / radius, rel=0.02)


@pytest.mark.parametrize("stacked", [False, True], ids=["one-box", "box-on-a-box"])
def test_with_friction_boxes_hold_on_a_slope_far_from_the_origin(
    device: str, stacked: bool
) -> None:
    """A box, or a box on a box, on an incline 3° below the friction angle (μ = 0.5), 100 m
    down the slope, where a float32 position resolves only about 8 µm, half a substep's
    slip: neither moves by more than 1 mm in 5 s. Friction's corrections enter the motion
    exactly and its anchors are fixed points, so the rounding of each slip it measures is
    not summed up (#101's sixth and seventh reviews found the sums biased). Without the
    anchors on the slope, the stack crept 1.9 mm (#102's review)."""
    theta = math.atan(0.5) - math.radians(3.0)
    normal = np.array([0.0, -math.sin(theta), math.cos(theta)])
    down = np.array([0.0, -math.cos(theta), -math.sin(theta)])
    slope = Solids((HalfSpace((0.0, 0.0, 0.0), tuple(normal)),))
    solver = _solver(device, solids=slope, static_friction=0.5, dynamic_friction=0.5)
    tilt = (math.sin(theta / 2.0), 0.0, 0.0, math.cos(theta / 2.0))
    base = 100.0 * down
    boxes = [_box(tuple(base + 0.1 * normal), orientation=tilt)]
    if stacked:
        boxes.append(_box(tuple(base + 0.3 * normal), orientation=tilt))
    solver.add_bodies(boxes)

    assert _drift(solver, 5.0) < 1e-3


def test_a_sliding_box_far_from_the_origin_stops_where_kinetic_friction_says(device: str) -> None:
    """A box sliding at 2 m/s on the floor 100 m from the origin, μ = 0.3: it stops after
    v²/2μg = 0.680 m, within 1 %, as it does at the origin."""
    solver = _solver(device, static_friction=0.3, dynamic_friction=0.3)
    solver.add_bodies([_box((100.0, 0.0, 0.1), half=(0.2, 0.2, 0.1),
                            velocity_m_per_s=(2.0, 0.0, 0.0),
                            static_friction=0.3, dynamic_friction=0.3)])  # fmt: skip

    for _ in range(120):
        solver.step(1.0 / 60.0)

    travelled = float(solver.body_positions()[0, 0]) - 100.0
    assert travelled == pytest.approx(4.0 / (2.0 * 0.3 * G), rel=0.01)


def test_with_friction_centred_stacks_stand_at_any_yaw(device: str) -> None:
    """Four centred stacks of five 20 cm boxes at 58 to 82° of yaw over the Test Tank's
    floor, μ = 0.5: for 30 s no box moves 1 mm off the one under it. Without friction a
    stack falls within about a minute, these four at 12 to 22 s on CUDA: float32 tilts
    each box's faces by about 1e-7 rad, a slope a frictionless box slides down (ADR 0031,
    #101's eighth review). Static friction holds it."""
    solver = _solver(device, max_bodies=20, static_friction=0.5, dynamic_friction=0.5)
    spots = [((-1.693, 0.125), 82.1), ((0.143, -0.187), 65.7), ((-0.130, 1.645), 48.7),
             ((1.350, 0.068), 58.2)]  # fmt: skip
    boxes = []
    for (x, y), yaw_deg in spots:
        half = math.radians(yaw_deg) / 2.0
        turn = (0.0, 0.0, math.sin(half), math.cos(half))
        boxes += [_box((x, y, 0.1 + 0.2 * k), orientation=turn) for k in range(5)]
    solver.add_bodies(boxes)

    for _ in range(30 * 60):
        solver.step(1.0 / 60.0)

    x = solver.body_positions().reshape(4, 5, 3)
    offsets = np.linalg.norm(x[:, 1:, :2] - x[:, :-1, :2], axis=2)
    assert offsets.max() < 1e-3


def test_with_friction_a_stack_set_off_by_10_um_stands(device: str) -> None:
    """Five 20 cm boxes stacked at the origin, the top one set off by 10 µm, μ = 0.5: for
    30 s no box moves 1 mm off the one under it. Without friction that offset brings the
    stack down in 7 to 8 s, at the origin as away from it (#101's ninth review)."""
    solver = _solver(device, static_friction=0.5, dynamic_friction=0.5)
    solver.add_bodies([_box((1e-5 if k == 4 else 0.0, 0.0, 0.1 + 0.2 * k)) for k in range(5)])

    for _ in range(30 * 60):
        solver.step(1.0 / 60.0)

    x = solver.body_positions()
    assert np.linalg.norm(x[1:, :2] - x[:-1, :2], axis=1).max() < 1e-3
