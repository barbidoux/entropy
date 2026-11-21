"""A whole frame of the Test Tank (`entropy.render.world_frame`, #52): solids, water,
glass and tone mapping, against a reference image.

The reference is `golden/still_pool.npy`. To make it again after a deliberate
change, run this file with `ENTROPY_UPDATE_GOLDEN=1`, look at the image, and commit it.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from entropy.core.config import WindowSettings
from entropy.core.paths import content_dir
from entropy.render.camera import CameraView
from entropy.render.context import GameWindow
from entropy.render.interop import SharedBuffer
from entropy.render.world_frame import WorldFrame
from entropy.solvers.liquid.solver import fill_box
from entropy.world.maps import load_map
from entropy.world.materials import load_material
from entropy.world.world import build_world

pytestmark = pytest.mark.render

GOLDEN = Path(__file__).parent / "golden" / "still_pool.npy"
SIZE = (320, 240)
DX_M = 0.0625


@pytest.fixture(scope="module")
def window(window_settings: WindowSettings) -> Iterator[GameWindow]:
    with GameWindow(window_settings, visible=False) as w:
        yield w


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean structural similarity of two 8-bit images (Wang et al. 2004): luma, a 7 x 7
    Gaussian window of sigma 1.5, K1 = 0.01, K2 = 0.03."""
    weights = np.array([0.299, 0.587, 0.114])
    x, y = (np.asarray(im, float) @ weights for im in (a, b))
    taps = np.exp(-(np.arange(-3, 4) ** 2) / (2 * 1.5**2))
    taps /= taps.sum()

    def blur(image: np.ndarray) -> np.ndarray:
        rows = np.apply_along_axis(lambda r: np.convolve(r, taps, mode="valid"), 1, image)
        return np.apply_along_axis(lambda c: np.convolve(c, taps, mode="valid"), 0, rows)

    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = blur(x), blur(y)
    vx, vy, cov = blur(x * x) - mx**2, blur(y * y) - my**2, blur(x * y) - mx * my
    index = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))
    return float(index.mean())


def test_ssim_is_one_for_equal_images_and_lower_for_noise() -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (40, 50, 3))
    assert ssim(image, image) == pytest.approx(1.0)
    noisy = np.clip(image + rng.normal(0, 40, image.shape), 0, 255)
    assert ssim(image, noisy) < 0.9


def _still_pool(window: GameWindow) -> np.ndarray:
    """Half a metre of water in the Test Tank, seen from a corner above the rim: particles
    on a lattice and sprites of several pixels, so the image does not hang on the jitter of
    single particles."""
    world = build_world(load_map(content_dir() / "maps" / "test_tank"))
    points = fill_box((-2.0, -2.0, 0.0), (2.0, 2.0, 0.5), DX_M, 8, seed=1, jitter=0.0)
    eye = (3.2, -3.6, 1.9)
    target = (0.0, 0.0, 0.2)
    forward = np.subtract(target, eye) / np.linalg.norm(np.subtract(target, eye))
    x, y, z = (float(c) for c in forward)
    view = CameraView(eye, (x, y, z), fov_y_rad=np.radians(60.0), near_m=0.05, far_m=500.0)
    screen = window.ctx.simple_framebuffer(SIZE)
    with SharedBuffer(window.ctx, len(points), wp.vec3, device="cpu") as shared:
        with shared.writable() as array:
            wp.copy(array, wp.array(points, dtype=wp.vec3, device="cpu"))
        frame = WorldFrame(
            window.ctx,
            world,
            shared,
            particle_volume_m3=DX_M**3 / 8,
            water=load_material("water"),
        )
        try:
            frame.render(screen, view, SIZE, count=len(points))
            image = np.frombuffer(screen.read(components=3), dtype=np.uint8)
        finally:
            frame.release()
            screen.release()
    return image.reshape(SIZE[1], SIZE[0], 3)[::-1].copy()


def test_a_still_pool_looks_like_its_reference(window: GameWindow) -> None:
    """SPEC 9.3, render tests: SSIM >= 0.98 against the reference image."""
    image = _still_pool(window)
    if os.environ.get("ENTROPY_UPDATE_GOLDEN") == "1":
        GOLDEN.parent.mkdir(exist_ok=True)
        np.save(GOLDEN, image)
    reference = np.load(GOLDEN)

    assert image.shape == reference.shape
    similarity = ssim(image, reference)
    assert similarity >= 0.98, f"SSIM {similarity:.4f} against the reference"
    water = image[SIZE[1] // 2, SIZE[0] // 2].astype(int)
    assert water[2] > water[0]  # the pool tints the floor blue-green


def test_the_frame_draws_the_glass(window: GameWindow, monkeypatch: pytest.MonkeyPatch) -> None:
    """The glass pass is part of the frame: without it, the panes' reflection is gone."""
    world = build_world(load_map(content_dir() / "maps" / "test_tank"))
    eye = (5.0, -1.0, 0.8)
    forward = np.subtract((0.0, 2.0, 0.6), eye)
    forward = forward / np.linalg.norm(forward)
    x, y, z = (float(c) for c in forward)
    view = CameraView(eye, (x, y, z), fov_y_rad=np.radians(60.0), near_m=0.05, far_m=500.0)
    screen = window.ctx.simple_framebuffer(SIZE)
    images = []
    with SharedBuffer(window.ctx, 1, wp.vec3, device="cpu") as shared:
        frame = WorldFrame(
            window.ctx, world, shared, particle_volume_m3=DX_M**3 / 8, water=load_material("water")
        )
        try:
            for _ in range(2):
                frame.render(screen, view, SIZE, count=0)
                images.append(np.frombuffer(screen.read(components=3), dtype=np.uint8))
                monkeypatch.setattr(frame.glass, "render", lambda *args, **kwargs: None)
        finally:
            frame.release()
            screen.release()

    with_glass, without = (image.astype(np.int32) for image in images)
    changed = np.abs(with_glass - without).reshape(-1, 3).max(axis=1) > 0
    assert changed.mean() > 0.10  # the panes in view: about 15 % of the pixels
