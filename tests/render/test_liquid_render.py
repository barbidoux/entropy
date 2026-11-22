"""The liquid pass (`entropy.render.liquid`, SPEC 7.2, #52): screen-space water."""

from collections.abc import Iterator

import numpy as np
import pytest
import warp as wp

from entropy.core.config import WindowSettings
from entropy.render.camera import CameraView
from entropy.render.context import GameWindow
from entropy.render.frame import HdrTarget
from entropy.render.interop import SharedBuffer
from entropy.render.liquid import LiquidRenderer, fresnel_f0
from entropy.render.scene import HORIZON, SceneBox, SceneRenderer
from entropy.solvers.liquid.solver import fill_box
from entropy.world.materials import load_material

pytestmark = pytest.mark.render

SIZE = (120, 90)
DX_M = 0.0625
WATER = load_material("water")
SPACING_M = DX_M / 2  # 8 particles per cell


@pytest.fixture(scope="module")
def window(window_settings: WindowSettings) -> Iterator[GameWindow]:
    with GameWindow(window_settings, visible=False) as w:
        yield w


def _view(eye_m, target_m) -> CameraView:
    forward = np.asarray(target_m, float) - np.asarray(eye_m, float)
    x, y, z = (float(c) for c in forward / np.linalg.norm(forward))
    return CameraView(eye_m, (x, y, z), fov_y_rad=np.radians(30.0), near_m=0.05, far_m=100.0)


def _read(target: HdrTarget) -> tuple[np.ndarray, np.ndarray]:
    """Linear radiance (rows from the top) and window depth of an HDR target."""
    width, height = target.size_px
    colour = np.frombuffer(target.colour.read(), dtype=np.float16).astype(np.float32)
    depth = np.frombuffer(target.depth.read(), dtype=np.float32)
    return (
        colour.reshape(height, width, 4)[::-1, :, :3],
        depth.reshape(height, width)[::-1],
    )


def _centre(image: np.ndarray, dy: int = 0, dx: int = 0, half: int = 2) -> np.ndarray:
    y, x = SIZE[1] // 2 + dy, SIZE[0] // 2 + dx
    return image[y - half : y + half + 1, x - half : x + half + 1].mean(axis=(0, 1))


def _draw(
    window: GameWindow,
    points: np.ndarray,
    view: CameraView,
    *,
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),
    boxes: tuple[SceneBox, ...] = (),
    size: tuple[int, int] = SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scene (a uniform background, or boxes over sky) then water: both read back."""
    ctx = window.ctx
    scene = HdrTarget(ctx, size)
    out = HdrTarget(ctx, size)
    count = max(len(points), 1)
    with SharedBuffer(ctx, count, wp.vec3, device="cpu") as shared:
        if len(points):
            with shared.writable() as array:
                wp.copy(array, wp.array(points.astype(np.float32), dtype=wp.vec3, device="cpu"))
        renderer = LiquidRenderer(
            ctx,
            shared,
            radius_m=0.75 * SPACING_M,
            particle_volume_m3=SPACING_M**3,
            water=WATER,
        )
        painter = SceneRenderer(ctx, boxes, ground_z_m=-50.0) if boxes else None
        try:
            if painter is not None:
                painter.render(scene.framebuffer, view, size)
            else:
                scene.framebuffer.clear(*background, 1.0, depth=1.0)
            renderer.render(out, scene, view, count=len(points))
            result = (*_read(scene), *_read(out))
        finally:
            if painter is not None:
                painter.release()
            renderer.release()
            scene.release()
            out.release()
    return result


def test_without_water_the_scene_passes_through(window: GameWindow) -> None:
    view = _view((6.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    scene, scene_depth, out, out_depth = _draw(
        window, np.zeros((0, 3)), view, background=(0.3, 0.5, 0.7)
    )

    np.testing.assert_array_equal(out, scene)
    np.testing.assert_array_equal(out_depth, scene_depth)


def test_deep_water_absorbs_red_first(window: GameWindow) -> None:
    """Beer-Lambert through a 2 m slab, square on, over white: each primary keeps
    exp(-a L) of the light behind (a from water.toml), and the surface reflects F0 of
    the horizon behind the camera. Jittered particles, and a patch of pixels: the
    thickness is an average over where rays cross the spheres."""
    slab = fill_box((0.0, -0.5, -0.5), (2.0, 0.5, 0.5), DX_M, 8, seed=1)
    view = _view((6.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    _, _, out, _ = _draw(window, slab, view)

    absorption = np.asarray(WATER.absorption_per_m_rgb)
    f0 = fresnel_f0(WATER.refractive_index or 0.0)
    expected = f0 * np.asarray(HORIZON) + (1.0 - f0) * np.exp(-absorption * 2.0)
    measured = _centre(out, half=8)
    np.testing.assert_allclose(measured, expected, rtol=0.03)
    assert measured[0] < measured[1] < measured[2]


def test_an_opaque_box_in_front_hides_the_water(window: GameWindow) -> None:
    slab = fill_box((0.0, -0.5, -0.5), (2.0, 0.5, 0.5), DX_M, 8, seed=1, jitter=0.0)
    view = _view((6.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    box = SceneBox((3.5, 0.0, 0.0), (0.1, 0.12, 0.12), (0.6, 0.1, 0.1))

    scene, _, out, _ = _draw(window, slab, view, boxes=(box,))

    np.testing.assert_allclose(_centre(out), _centre(scene), atol=1e-3)  # the box
    beside = _centre(out, dx=15)  # beside the box: water, over the sky
    assert np.abs(beside - _centre(scene, dx=15)).max() > 0.05


def test_the_water_writes_its_depth(window: GameWindow) -> None:
    """Passes after this one (glass) are hidden where the water is nearer."""
    slab = fill_box((0.0, -0.5, -0.5), (2.0, 0.5, 0.5), DX_M, 8, seed=1, jitter=0.0)
    view = _view((6.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    _, scene_depth, _, out_depth = _draw(window, slab, view)

    assert _centre(out_depth) < _centre(scene_depth) == 1.0
    assert out_depth[2, 2] == scene_depth[2, 2] == 1.0  # a corner: no water
    # The slab's front: particle centres reach 2 m - half a spacing, spheres a radius
    # more. Its window depth, from the eye 6 m away along x:
    front_m = 2.0 - 0.5 * SPACING_M + 0.75 * SPACING_M
    clip = view.projection(SIZE[0] / SIZE[1]) @ np.array([0.0, 0.0, -(6.0 - front_m), 1.0])
    expected = clip[2] / clip[3] * 0.5 + 0.5
    eye_depth = _eye_depth(view, float(_centre(out_depth, half=0)))
    assert eye_depth == pytest.approx(6.0 - front_m, abs=0.01)
    assert _centre(out_depth, half=0) == pytest.approx(expected, abs=1e-4)


def test_water_s_optics_come_from_its_material() -> None:
    assert WATER.refractive_index == pytest.approx(1.333, abs=1e-3)
    absorption = WATER.absorption_per_m_rgb
    assert absorption is not None and absorption[0] > absorption[1] > absorption[2] > 0.0
    assert fresnel_f0(1.333) == pytest.approx(0.0204, abs=1e-4)


def _eye_depth(view: CameraView, window_depth: float) -> float:
    """Distance along the view axis (m) of a window depth in [0, 1]."""
    p = view.projection(SIZE[0] / SIZE[1]).astype(np.float64)
    ndc = 2.0 * window_depth - 1.0
    return float(p[2, 3] / (ndc + p[2, 2]))


def test_the_water_reflects_the_sky_by_fresnel_s_law(window: GameWindow) -> None:
    """Over black, square on: only the reflection is seen, F0 = 2 % of the horizon behind
    the camera (n = 1.333). At a grazing angle the same surface reflects far more."""
    slab = fill_box((0.0, -0.5, -0.5), (2.0, 0.5, 0.5), DX_M, 8, seed=1, jitter=0.0)
    square = _view((6.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    _, _, out, _ = _draw(window, slab, square, background=(0.0, 0.0, 0.0))

    f0 = fresnel_f0(WATER.refractive_index or 0.0)
    np.testing.assert_allclose(_centre(out, half=4), f0 * np.asarray(HORIZON), rtol=0.10)

    pool = fill_box((-1.0, -1.0, -0.5), (1.0, 1.0, 0.0), DX_M, 8, seed=2, jitter=0.0)
    grazing = _view((-6.0, 0.0, 0.4), (0.0, 0.0, 0.0))  # about 86° from the normal
    _, _, out, _ = _draw(window, pool, grazing, background=(0.0, 0.0, 0.0))
    assert _centre(out, half=4).sum() > 10.0 * f0 * float(np.sum(HORIZON))


def test_looking_down_the_water_reflects_the_zenith(window: GameWindow) -> None:
    """Straight down onto a pool, over black: the reflection is of the sky overhead, F0 of
    the zenith's deep blue, not the pale horizon (a reflected ray turned the wrong way, or
    a view rotation transposed, would see the horizon)."""
    pool = fill_box((-1.0, -1.0, -0.5), (1.0, 1.0, 0.0), DX_M, 8, seed=2, jitter=0.0)
    down = _view((0.001, 0.0, 3.0), (0.0, 0.0, 0.0))

    _, _, out, _ = _draw(window, pool, down, background=(0.0, 0.0, 0.0))

    f0 = fresnel_f0(WATER.refractive_index or 0.0)
    zenith = np.array([0.10, 0.25, 0.70])  # SKY_GLSL's colour straight up
    np.testing.assert_allclose(_centre(out, half=4), f0 * zenith, rtol=0.15)


def test_at_45_degrees_the_water_reflects_the_sky_45_degrees_up(window: GameWindow) -> None:
    """An oblique view, so the view-to-world rotation is not symmetric: the reflection is
    the sky at 45° (SKY_GLSL: horizon to zenith by the square root of the sine), weighted
    by Schlick's Fresnel at 45°."""
    pool = fill_box((-1.5, -1.5, -0.5), (1.5, 1.5, 0.0), DX_M, 8, seed=2, jitter=0.0)
    view = _view((-2.0, 1.0, 2.0), (0.0, 1.0, 0.0))

    _, _, out, _ = _draw(window, pool, view, background=(0.0, 0.0, 0.0))

    f0 = fresnel_f0(WATER.refractive_index or 0.0)
    fresnel = f0 + (1 - f0) * (1 - np.cos(np.radians(45.0))) ** 5
    zenith, horizon = np.array([0.10, 0.25, 0.70]), np.asarray(HORIZON)
    sky = horizon + (zenith - horizon) * np.sqrt(np.sin(np.radians(45.0)))
    np.testing.assert_allclose(_centre(out, half=4), fresnel * sky, rtol=0.15)


def test_a_solid_inside_the_water_shortens_what_light_crosses(window: GameWindow) -> None:
    """A solid halfway into a 2 m slab: the light seen comes from its face 0.9 m deep, so
    only 0.9 m of water absorbs it. Spheres behind the face add no thickness."""
    slab = fill_box((0.0, -0.5, -0.5), (2.0, 0.5, 0.5), DX_M, 8, seed=1)
    view = _view((6.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    block = SceneBox((1.0, 0.0, 0.0), (0.1, 0.4, 0.4), (0.8, 0.8, 0.8))

    scene, _, out, _ = _draw(window, slab, view, boxes=(block,))

    absorption = np.asarray(WATER.absorption_per_m_rgb)
    f0 = fresnel_f0(WATER.refractive_index or 0.0)
    face = _centre(scene, half=8)
    expected = f0 * np.asarray(HORIZON) + (1.0 - f0) * face * np.exp(-absorption * 0.9)
    np.testing.assert_allclose(_centre(out, half=8), expected, rtol=0.03)


@pytest.mark.parametrize(("size", "half"), [((120, 90), (60, 45)), ((121, 91), (61, 46))])
def test_the_spheres_are_drawn_at_half_the_output_s_resolution(
    window: GameWindow, size: tuple[int, int], half: tuple[int, int]
) -> None:
    """#57: drawing the spheres costs per fragment, and up close a million of them cost
    4.6 ms at 1600 x 900 (SPEC 7.2's budget: 2.5 ms). Depth, thickness and smoothing run
    at half the output's resolution; the composite runs at full resolution."""
    slab = fill_box((0.0, -0.5, -0.5), (2.0, 0.5, 0.5), DX_M, 8, seed=1, jitter=0.0)
    ctx = window.ctx
    scene, out = HdrTarget(ctx, size), HdrTarget(ctx, size)
    with SharedBuffer(ctx, len(slab), wp.vec3, device="cpu") as shared:
        with shared.writable() as array:
            wp.copy(array, wp.array(slab.astype(np.float32), dtype=wp.vec3, device="cpu"))
        renderer = LiquidRenderer(
            ctx, shared, radius_m=0.75 * SPACING_M, particle_volume_m3=SPACING_M**3, water=WATER
        )
        try:
            scene.framebuffer.clear(1.0, 1.0, 1.0, 1.0, depth=1.0)
            renderer.render(out, scene, _view((6.0, 0.0, 0.0), (0.0, 0.0, 0.0)), count=len(slab))
            assert renderer.sprite_size_px == half
        finally:
            renderer.release()
            scene.release()
            out.release()


def _eye_depths(view: CameraView, window_depth: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Distance along the view axis (m) of window depths in [0, 1]."""
    projection = view.projection(size[0] / size[1]).astype(np.float64)
    return projection[2, 3] / ((2.0 * window_depth - 1.0) + projection[2, 2])


@pytest.mark.parametrize("offset_m", [0.003, 0.007, 0.010])
def test_a_box_edge_off_the_pixel_blocks_is_never_drawn_over(
    window: GameWindow, offset_m: float
) -> None:
    """A box in front of the water, its edges anywhere on the screen: every box pixel shows
    the box and its depth, and every water pixel beside it is water, absorbed. Spheres
    are drawn in 2 x 2 blocks of output pixels (#78's review, S1)."""
    size = (321, 241)
    slab = fill_box((0.0, -0.5, -0.5), (2.0, 0.5, 0.5), DX_M, 8, seed=1, jitter=0.0)
    view = _view((6.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    box = SceneBox((3.5, offset_m, offset_m * 0.6), (0.1, 0.12, 0.12), (0.6, 0.1, 0.1))

    scene, scene_depth, out, out_depth = _draw(window, slab, view, boxes=(box,), size=size)

    on_box = _eye_depths(view, scene_depth, size) < 3.0
    assert on_box.sum() > 100
    np.testing.assert_allclose(out[on_box], scene[on_box], atol=1e-3)
    np.testing.assert_allclose(out_depth[on_box], scene_depth[on_box], atol=1e-6)
    grown = on_box.copy()
    grown[1:, :] |= on_box[:-1, :]
    grown[:-1, :] |= on_box[1:, :]
    grown[:, 1:] |= on_box[:, :-1]
    grown[:, :-1] |= on_box[:, 1:]
    ring = grown & ~on_box
    absorbed = np.abs(out[ring] - scene[ring]).max(axis=1)
    assert (absorbed > 0.02).all(), f"{(absorbed <= 0.02).sum()} ring pixels show no water"


def test_the_targets_keep_their_own_viewports(window: GameWindow) -> None:
    """The pass draws in its own half-size targets: the scene's and the output's
    framebuffers keep their full-size viewports (#78's review, S2)."""
    slab = fill_box((0.0, -0.5, -0.5), (2.0, 0.5, 0.5), DX_M, 8, seed=1, jitter=0.0)
    ctx = window.ctx
    scene, out = HdrTarget(ctx, SIZE), HdrTarget(ctx, SIZE)
    with SharedBuffer(ctx, len(slab), wp.vec3, device="cpu") as shared:
        with shared.writable() as array:
            wp.copy(array, wp.array(slab.astype(np.float32), dtype=wp.vec3, device="cpu"))
        renderer = LiquidRenderer(
            ctx, shared, radius_m=0.75 * SPACING_M, particle_volume_m3=SPACING_M**3, water=WATER
        )
        try:
            scene.framebuffer.use()
            scene.framebuffer.clear(1.0, 1.0, 1.0, 1.0, depth=1.0)
            for _ in range(2):
                renderer.render(
                    out, scene, _view((6.0, 0.0, 0.0), (0.0, 0.0, 0.0)), count=len(slab)
                )
            assert scene.framebuffer.viewport == (0, 0, *SIZE)
            assert out.framebuffer.viewport == (0, 0, *SIZE)
        finally:
            renderer.release()
            scene.release()
            out.release()


@pytest.mark.parametrize("size", [(120, 90), (121, 91)])
def test_the_water_s_silhouette_is_where_the_water_is(
    window: GameWindow, size: tuple[int, int]
) -> None:
    """A slab centred on the view axis is drawn centred, at even and odd output sizes
    (#78's review, N1)."""
    slab = fill_box((0.0, -0.5, -0.5), (2.0, 0.5, 0.5), DX_M, 8, seed=1, jitter=0.0)
    view = _view((6.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    scene, _, out, _ = _draw(window, slab, view, size=size)

    water = np.abs(out - scene).max(axis=2) > 0.02
    rows, cols = np.nonzero(water)
    assert abs(cols.mean() - (size[0] - 1) / 2) < 0.3
    assert abs(rows.mean() - (size[1] - 1) / 2) < 0.3


@pytest.mark.parametrize("size", [(120, 90), (480, 360), (960, 720)])
def test_the_thickness_does_not_depend_on_how_big_the_spheres_look(
    window: GameWindow, size: tuple[int, int]
) -> None:
    """The same 2 m slab at three output sizes: its spheres span about 0.4, 1.4 and 2.9
    pixels of the half-size targets, and Beer-Lambert holds at all three (#78's review)."""
    slab = fill_box((0.0, -0.5, -0.5), (2.0, 0.5, 0.5), DX_M, 8, seed=1)
    view = _view((6.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    _, _, out, _ = _draw(window, slab, view, size=size)

    absorption = np.asarray(WATER.absorption_per_m_rgb)
    f0 = fresnel_f0(WATER.refractive_index or 0.0)
    expected = f0 * np.asarray(HORIZON) + (1.0 - f0) * np.exp(-absorption * 2.0)
    y, x = size[1] // 2, size[0] // 2
    half = size[0] // 15
    measured = out[y - half : y + half + 1, x - half : x + half + 1].mean(axis=(0, 1))
    np.testing.assert_allclose(measured, expected, rtol=0.03)


@pytest.mark.parametrize("background", [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)], ids=["black", "white"])
def test_the_water_varies_within_each_block_of_pixels(
    window: GameWindow, background: tuple[float, float, float]
) -> None:
    """A ball of water: over black only its reflection shows, which follows its normal;
    over white mostly what it lets through, which follows its thickness. Both are
    interpolated between the half-size targets' pixels, so neighbouring output pixels in
    one 2 x 2 block differ as much as those across blocks; read from the nearest pixel, a
    block would share one value (#78's review, B1)."""
    size = (240, 180)
    cube = fill_box((-0.6, -0.6, -0.6), (0.6, 0.6, 0.6), DX_M, 8, seed=4, jitter=0.0)
    ball = cube[np.linalg.norm(cube, axis=1) <= 0.6]
    view = _view((6.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    scene, _, out, _ = _draw(window, ball, view, background=background, size=size)

    lit = out.sum(axis=2)
    water = np.abs(out - scene).max(axis=2) > 1e-3
    inner = water.copy()
    for shift in (1, 2, 3):
        inner[shift:, :] &= water[:-shift, :]
        inner[:-shift, :] &= water[shift:, :]
        inner[:, shift:] &= water[:, :-shift]
        inner[:, :-shift] &= water[:, shift:]
    step = np.abs(np.diff(lit, axis=1))
    both = inner[:, 1:] & inner[:, :-1]
    within = step[:, 0::2][both[:, 0::2]].mean()  # columns 2k and 2k + 1: one block
    across = step[:, 1::2][both[:, 1::2]].mean()  # columns 2k + 1 and 2k + 2: two blocks
    assert within > 0.5 * across, f"within {within:.5f}, across {across:.5f}"
