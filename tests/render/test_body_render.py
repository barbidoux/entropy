"""Rigid bodies drawn in the solids pass (`entropy.render.bodies`, SPEC 7.2, #91)."""

import math
from collections.abc import Iterator

import numpy as np
import pytest
import warp as wp

from entropy.core.config import WindowSettings
from entropy.render.bodies import BodyRenderer
from entropy.render.camera import CameraView
from entropy.render.context import GameWindow
from entropy.render.frame import HdrTarget
from entropy.render.interop import InteropMode
from entropy.render.scene import SceneBox, SceneRenderer
from entropy.solvers.rigid import Body, RigidSetup, RigidSolver, Shape

pytestmark = pytest.mark.render

SIZE = (160, 120)
RED = (0.6, 0.1, 0.1)


@pytest.fixture(scope="module")
def window(window_settings: WindowSettings) -> Iterator[GameWindow]:
    with GameWindow(window_settings, visible=False) as w:
        yield w


def _view(eye_m, target_m) -> CameraView:
    forward = np.asarray(target_m, float) - np.asarray(eye_m, float)
    x, y, z = (float(c) for c in forward / np.linalg.norm(forward))
    return CameraView(eye_m, (x, y, z), fov_y_rad=np.radians(30.0), near_m=0.05, far_m=100.0)


def _read(target: HdrTarget) -> np.ndarray:
    width, height = target.size_px
    colour = np.frombuffer(target.colour.read(), dtype=np.float16).astype(np.float32)
    return colour.reshape(height, width, 4)[::-1, :, :3]


def _draw(
    window: GameWindow,
    view: CameraView,
    bodies: list[Body] = (),  # type: ignore[assignment]
    boxes: tuple[SceneBox, ...] = (),
    albedo=RED,
) -> np.ndarray:
    """The scene (ground and `boxes`), then `bodies` over it, read back as linear RGB."""
    ctx = window.ctx
    target = HdrTarget(ctx, SIZE)
    solver = RigidSolver.create(RigidSetup(gravity_m_per_s2=(0.0, 0.0, -9.81), max_bodies=8), "cpu")
    solver.add_bodies(list(bodies))
    scene = SceneRenderer(ctx, boxes, ground_z_m=-50.0)
    renderer = BodyRenderer(ctx, capacity=8, device="cpu")
    colours = wp.array(np.tile(albedo, (8, 1)).astype(np.float32), dtype=wp.vec3, device="cpu")
    try:
        scene.render(target.framebuffer, view, SIZE)
        renderer.update(solver.fields(), solver.body_count(), colours)
        renderer.render(target.framebuffer, view, SIZE)
        return _read(target)
    finally:
        renderer.release()
        scene.release()
        target.release()


def _quat(axis, angle_rad: float) -> tuple[float, float, float, float]:
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    s = math.sin(angle_rad / 2.0)
    return (float(a[0] * s), float(a[1] * s), float(a[2] * s), math.cos(angle_rad / 2.0))


OBLIQUE = _view((3.0, -4.0, 3.0), (0.0, 0.0, 0.0))


def test_a_box_body_draws_as_the_scene_draws_the_same_box(window: GameWindow) -> None:
    """Same geometry, same lighting: the body pass draws what the scene pass draws."""
    half = (0.4, 0.3, 0.2)
    body = Body(Shape.BOX, half, 500.0, position_m=(0.1, 0.2, 0.3))

    drawn = _draw(window, OBLIQUE, bodies=[body])
    expected = _draw(window, OBLIQUE, boxes=(SceneBox((0.1, 0.2, 0.3), half, RED),))

    np.testing.assert_allclose(drawn, expected, atol=2e-3)


def test_a_quarter_turn_draws_the_box_with_its_sides_swapped(window: GameWindow) -> None:
    """Turned 90° about z, a box of half sizes (a, b, c) is the box (b, a, c): the body's
    orientation is applied."""
    body = Body(Shape.BOX, (0.5, 0.2, 0.3), 500.0, orientation=_quat((0, 0, 1), math.pi / 2))

    drawn = _draw(window, OBLIQUE, bodies=[body])
    expected = _draw(window, OBLIQUE, boxes=(SceneBox((0.0, 0.0, 0.0), (0.2, 0.5, 0.3), RED),))

    differs = np.abs(drawn - expected).max(axis=-1) > 2e-3
    assert differs.mean() < 0.002, f"{differs.sum()} pixels differ"


def _covered(window: GameWindow, view: CameraView, bodies: list[Body]) -> np.ndarray:
    """Pixels the bodies changed."""
    return np.abs(_draw(window, view, bodies) - _draw(window, view)).max(axis=-1) > 1e-3


def _pixel(view: CameraView, point_m) -> tuple[int, int]:
    """(row from the top, column) of a world point."""
    projection = view.projection(SIZE[0] / SIZE[1]).astype(np.float64) @ view.view()
    clip = projection @ np.array([*point_m, 1.0])
    ndc = clip[:3] / clip[3]
    column = int((ndc[0] * 0.5 + 0.5) * SIZE[0])
    row = int((1.0 - (ndc[1] * 0.5 + 0.5)) * SIZE[1])
    return row, column


SIDE = _view((6.0, 0.0, 0.0), (0.0, 0.0, 0.0))  # looking along -x at the y-z plane


def test_a_sphere_draws_a_disc_of_its_radius(window: GameWindow) -> None:
    covered = _covered(window, SIDE, [Body(Shape.SPHERE, (0.5, 0.0, 0.0), 500.0)])

    for point, inside in (
        ((0, 0, 0), True),
        ((0, 0.45, 0), True),
        ((0, 0, -0.45), True),
        ((0, 0.55, 0), False),
        ((0, 0, 0.55), False),
        ((0, 0.4, 0.4), False),
    ):
        assert covered[_pixel(SIDE, point)] == inside, point


def test_a_capsule_is_as_long_as_its_cylinder_and_caps(window: GameWindow) -> None:
    """Radius 0.2 and a cylinder 1 m long (h = 0.5), laid along y: it spans 1.4 m along y and
    0.4 m across."""
    capsule = Body(Shape.CAPSULE, (0.2, 0.5, 0.0), 500.0, orientation=_quat((1, 0, 0), math.pi / 2))
    covered = _covered(window, SIDE, [capsule])

    # Points 2 pixels or more (about 0.05 m here) from the surface.
    for point, inside in (
        ((0, 0.65, 0), True),
        ((0, -0.65, 0), True),
        ((0, 0, 0.15), True),
        ((0, 0.78, 0), False),
        ((0, 0, 0.25), False),
        ((0, 0.68, 0.15), False),
    ):
        assert covered[_pixel(SIDE, point)] == inside, point


def test_each_body_draws_only_its_own_shape(window: GameWindow) -> None:
    """Every mesh is drawn for every body and a body keeps only its shape's: a sphere
    whose unused sizes would make a large box or a long capsule stays a disc, and a box
    keeps its corners."""
    sphere = Body(Shape.SPHERE, (0.2, 0.5, 0.5), 500.0, position_m=(0.0, -1.0, 0.0))
    box = Body(Shape.BOX, (0.3, 0.3, 0.3), 500.0, position_m=(0.0, 1.0, 0.0))
    covered = _covered(window, SIDE, [sphere, box])

    assert covered[_pixel(SIDE, (0.0, -1.0, 0.0))]
    assert not covered[_pixel(SIDE, (0.0, -1.0, 0.4))]  # a capsule's end
    assert not covered[_pixel(SIDE, (0.0, -0.6, 0.0))]  # a box's side
    assert covered[_pixel(SIDE, (0.0, 1.25, 0.25))]  # a box's corner, outside its sphere


def test_no_bodies_draw_nothing(window: GameWindow) -> None:
    np.testing.assert_array_equal(_draw(window, OBLIQUE, []), _draw(window, OBLIQUE))


def test_the_bodies_are_drawn_where_the_solver_moved_them(window: GameWindow) -> None:
    """The renderer reads the solver's arrays as they are, not a copy made at the start."""
    ctx = window.ctx
    target = HdrTarget(ctx, SIZE)
    solver = RigidSolver.create(RigidSetup(gravity_m_per_s2=(0.0, 0.0, -9.81), max_bodies=2), "cpu")
    solver.add_bodies([Body(Shape.SPHERE, (0.3, 0.0, 0.0), 500.0, position_m=(0.0, 0.0, 1.0))])
    scene = SceneRenderer(ctx, (), ground_z_m=-50.0)
    renderer = BodyRenderer(ctx, capacity=2, device="cpu")
    colours = wp.array(np.array([RED, RED], np.float32), dtype=wp.vec3, device="cpu")
    try:
        for _ in range(30):
            solver.step(1.0 / 60.0)  # falls 1.23 m in 0.5 s
        scene.render(target.framebuffer, SIDE, SIZE)
        renderer.update(solver.fields(), solver.body_count(), colours)
        renderer.render(target.framebuffer, SIDE, SIZE)
        image = _read(target)
        empty = _draw(window, SIDE)
    finally:
        renderer.release()
        scene.release()
        target.release()
    covered = np.abs(image - empty).max(axis=-1) > 1e-3
    z = float(solver.body_positions()[0, 2])
    assert covered[_pixel(SIDE, (0.0, 0.0, z))]
    assert not covered[_pixel(SIDE, (0.0, 0.0, 1.0))]


@pytest.mark.gpu
def test_on_cuda_the_bodies_go_to_opengl_without_a_copy(window: GameWindow) -> None:
    """The instances are written by a CUDA kernel straight into the OpenGL buffer, and
    draw what the CPU path draws."""
    ctx = window.ctx
    target = HdrTarget(ctx, SIZE)
    solver = RigidSolver.create(
        RigidSetup(gravity_m_per_s2=(0.0, 0.0, -9.81), max_bodies=2), "cuda:0"
    )
    solver.add_bodies([Body(Shape.BOX, (0.4, 0.3, 0.2), 500.0, position_m=(0.1, 0.2, 0.3))])
    scene = SceneRenderer(ctx, (), ground_z_m=-50.0)
    renderer = BodyRenderer(ctx, capacity=2, device="cuda:0")
    colours = wp.array(np.array([RED, RED], np.float32), dtype=wp.vec3, device="cuda:0")
    try:
        assert renderer.instances.mode is InteropMode.ZERO_COPY, renderer.instances.fallback_reason
        scene.render(target.framebuffer, OBLIQUE, SIZE)
        renderer.update(solver.fields(), solver.body_count(), colours)
        wp.synchronize_device("cuda:0")
        renderer.render(target.framebuffer, OBLIQUE, SIZE)
        image = _read(target)
    finally:
        renderer.release()
        scene.release()
        target.release()
    expected = _draw(
        window, OBLIQUE, [Body(Shape.BOX, (0.4, 0.3, 0.2), 500.0, position_m=(0.1, 0.2, 0.3))]
    )
    np.testing.assert_allclose(image, expected, atol=2e-3)
