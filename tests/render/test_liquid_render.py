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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scene (a uniform background, or boxes over sky) then water: both read back."""
    ctx = window.ctx
    scene = HdrTarget(ctx, SIZE)
    out = HdrTarget(ctx, SIZE)
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
                painter.render(scene.framebuffer, view, SIZE)
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
    np.testing.assert_allclose(measured, expected, rtol=0.06)
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


def test_water_s_optics_come_from_its_material() -> None:
    assert WATER.refractive_index == pytest.approx(1.333, abs=1e-3)
    absorption = WATER.absorption_per_m_rgb
    assert absorption is not None and absorption[0] > absorption[1] > absorption[2] > 0.0
    assert fresnel_f0(1.333) == pytest.approx(0.0204, abs=1e-4)
