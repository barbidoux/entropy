"""The empty lit scene (`entropy.render.scene`): sky, ground, sun, reference cubes."""

from collections.abc import Iterator

import numpy as np
import pytest

from entropy.core.config import WindowSettings
from entropy.render.camera import CameraView
from entropy.render.context import GameWindow
from entropy.render.scene import (
    EXPOSURE,
    REFERENCE_CUBES,
    SKY_IRRADIANCE,
    SUN_DIRECTION,
    SUN_IRRADIANCE,
    SceneRenderer,
    srgb_from_linear,
)

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


def _render(window: GameWindow, view: CameraView, size: tuple[int, int] = SIZE) -> np.ndarray:
    """RGB image as floats from 0 to 255, row 0 at the top."""
    target = window.ctx.simple_framebuffer(size)
    renderer = SceneRenderer(window.ctx)
    try:
        renderer.render(target, view, size)
        image = np.frombuffer(target.read(components=3), dtype=np.uint8)
    finally:
        renderer.release()
        target.release()
    return image.reshape(size[1], size[0], 3)[::-1].astype(float)


def _pixel_of(
    view: CameraView, point_m: tuple[float, float, float], size: tuple[int, int]
) -> tuple[int, int]:
    """(row from the top, column) of a world point."""
    clip = view.projection(size[0] / size[1]) @ view.view() @ np.append(point_m, 1.0)
    x, y = clip[:2] / clip[3]
    return int((1.0 - (y + 1.0) / 2.0) * size[1]), int((x + 1.0) / 2.0 * size[0])


def _aces(x: np.ndarray) -> np.ndarray:
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


def _expected_face(normal: tuple[float, float, float]) -> np.ndarray:
    """The shader's lighting model, in Python: the reference for the face colours."""
    albedo = np.array(REFERENCE_CUBES[0][2])
    n = np.array(normal)
    direct = max(float(n @ np.array(SUN_DIRECTION)), 0.0)
    radiance = albedo * (
        np.array(SUN_IRRADIANCE) * direct + np.array(SKY_IRRADIANCE) * (0.5 + 0.5 * n[2])
    )
    display = _aces(radiance * EXPOSURE)
    return 255.0 * np.array([srgb_from_linear(float(c)) for c in display])


def test_level_view_shows_blue_sky_above_and_grey_ground_below(window: GameWindow) -> None:
    image = _render(window, _view((0.0, -20.0, 1.7), (0.0, 1.0, 0.0)))

    sky = image[: SIZE[1] // 4].mean(axis=(0, 1))
    ground = image[-SIZE[1] // 4 :].mean(axis=(0, 1))
    assert sky[2] > sky[0] + 20, f"sky not blue: {sky}"
    assert abs(ground[0] - ground[2]) < 40, f"ground not grey: {ground}"
    assert ground.mean() > 20, f"ground unlit: {ground}"


@pytest.mark.parametrize(
    ("eye_m", "forward", "normal"),
    [
        ((6.0, 0.0, 0.5), (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((-6.0, 0.0, 0.5), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        ((0.0, 6.0, 0.5), (0.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
        ((0.0, -6.0, 0.5), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        ((0.0, 0.0, 6.0), (1e-3, 0.0, -1.0), (0.0, 0.0, 1.0)),
    ],
)
def test_each_face_of_the_origin_cube_has_the_modelled_colour(
    window: GameWindow,
    eye_m: tuple[float, float, float],
    forward: tuple[float, float, float],
    normal: tuple[float, float, float],
) -> None:
    """Sun, sky term, exposure, ACES and sRGB, checked through the real shader."""
    view = _view(eye_m, forward)
    face_centre = tuple(0.5 * np.array(normal) + np.array([0.0, 0.0, 0.5]))
    row, column = _pixel_of(view, face_centre, SIZE)

    pixel = _render(window, view)[row, column]

    np.testing.assert_allclose(pixel, _expected_face(normal), atol=3.0)


def test_sunlit_faces_are_brighter_than_faces_turned_away(window: GameWindow) -> None:
    assert SUN_DIRECTION[0] > 0 and SUN_DIRECTION[2] > 0
    assert _expected_face((1.0, 0.0, 0.0)).mean() > _expected_face((-1.0, 0.0, 0.0)).mean() + 20


def test_grid_lines_lie_on_whole_metres(window: GameWindow) -> None:
    """Away from the cubes: a pixel on the x = 20 m line is darker than a cell centre."""
    size = (320, 180)
    view = _view((20.3, 20.3, 3.0), (1e-3, 0.0, -1.0))
    image = _render(window, view, size).mean(axis=2)

    on_line = image[_pixel_of(view, (20.0, 20.5, 0.0), size)]
    in_cell = image[_pixel_of(view, (20.5, 20.5, 0.0), size)]

    assert on_line < in_cell - 10


def test_the_viewport_follows_the_given_size_not_the_target(window: GameWindow) -> None:
    """moderngl's screen keeps its creation size; the window's real size is passed in."""
    target = window.ctx.simple_framebuffer((200, 100))
    renderer = SceneRenderer(window.ctx)
    try:
        renderer.render(target, _view((0.0, -20.0, 1.7), (0.0, 1.0, 0.0)), (100, 50))
        assert window.ctx.viewport == (0, 0, 100, 50)
    finally:
        renderer.release()
        target.release()


def test_a_minimised_window_is_skipped(window: GameWindow) -> None:
    target = window.ctx.simple_framebuffer((8, 8))
    renderer = SceneRenderer(window.ctx)
    try:
        renderer.render(target, _view((0.0, -20.0, 1.7), (0.0, 1.0, 0.0)), (0, 0))
    finally:
        renderer.release()
        target.release()


def test_srgb_transfer_function() -> None:
    assert srgb_from_linear(0.0) == pytest.approx(0.0)
    assert srgb_from_linear(1.0) == pytest.approx(1.0)
    assert srgb_from_linear(0.0031308) == pytest.approx(0.04045, abs=1e-4)
    assert srgb_from_linear(0.5) == pytest.approx(0.7354, abs=1e-3)
