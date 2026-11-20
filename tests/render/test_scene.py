"""The empty lit scene (`entropy.render.scene`): sky, ground, sun, reference cubes."""

from collections.abc import Iterator

import numpy as np
import pytest

from entropy.core.config import WindowSettings
from entropy.render.camera import CameraView
from entropy.render.context import GameWindow
from entropy.render.scene import SUN_DIRECTION, SceneRenderer, srgb_from_linear

pytestmark = pytest.mark.render

SIZE = (160, 90)


@pytest.fixture(scope="module")
def window(window_settings: WindowSettings) -> Iterator[GameWindow]:
    with GameWindow(window_settings, visible=False) as w:
        yield w


def _view(
    position_m: tuple[float, float, float], forward: tuple[float, float, float]
) -> CameraView:
    return CameraView(position_m, forward, fov_y_rad=np.radians(70.0), near_m=0.05, far_m=500.0)


def _render(window: GameWindow, view: CameraView) -> np.ndarray:
    """RGB image as floats from 0 to 255, row 0 at the top."""
    target = window.ctx.simple_framebuffer(SIZE)
    renderer = SceneRenderer(window.ctx)
    renderer.render(target, view)
    image = np.frombuffer(target.read(components=3), dtype=np.uint8).reshape(SIZE[1], SIZE[0], 3)
    renderer.release()
    target.release()
    return image[::-1].astype(float)


def test_level_view_shows_blue_sky_above_and_grey_ground_below(window: GameWindow) -> None:
    image = _render(window, _view((0.0, -20.0, 1.7), (0.0, 1.0, 0.0)))

    sky = image[: SIZE[1] // 4].mean(axis=(0, 1))
    ground = image[-SIZE[1] // 4 :].mean(axis=(0, 1))
    assert sky[2] > sky[0] + 20, f"sky not blue: {sky}"
    assert abs(ground[0] - ground[2]) < 40, f"ground not grey: {ground}"
    assert ground.mean() > 20, f"ground unlit: {ground}"


def test_sunlit_faces_are_brighter_than_faces_turned_away(window: GameWindow) -> None:
    """The sun shines from +X, +Y, +Z; the reference cube at the origin shows +X or -X."""
    assert SUN_DIRECTION[0] > 0 and SUN_DIRECTION[2] > 0
    from_east = _render(window, _view((6.0, 0.0, 0.5), (-1.0, 0.0, 0.0)))
    from_west = _render(window, _view((-6.0, 0.0, 0.5), (1.0, 0.0, 0.0)))

    centre = (slice(SIZE[1] // 2 - 4, SIZE[1] // 2 + 4), slice(SIZE[0] // 2 - 4, SIZE[0] // 2 + 4))
    assert from_east[centre].mean() > from_west[centre].mean() + 20


def test_looking_straight_down_shows_the_metre_grid(window: GameWindow) -> None:
    """Grid lines every metre: the ground below the camera is not uniform."""
    image = _render(window, _view((0.3, 0.3, 3.0), (1e-3, 0.0, -1.0)))

    assert image.std() > 5.0


def test_srgb_transfer_function() -> None:
    assert srgb_from_linear(0.0) == pytest.approx(0.0)
    assert srgb_from_linear(1.0) == pytest.approx(1.0)
    assert srgb_from_linear(0.0031308) == pytest.approx(0.04045, abs=1e-4)
    assert srgb_from_linear(0.5) == pytest.approx(0.7354, abs=1e-3)
