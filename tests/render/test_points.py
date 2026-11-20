"""The interop test pattern and its point renderer (`entropy.render.points`)."""

from collections.abc import Iterator

import numpy as np
import pytest
import warp as wp

from entropy.core.config import WindowSettings
from entropy.render.camera import CameraView
from entropy.render.context import GameWindow
from entropy.render.interop import SharedBuffer
from entropy.render.points import (
    PATTERN_CENTRE_M,
    PATTERN_RADIUS_M,
    PointsRenderer,
    write_test_pattern,
)

COUNT = 4096


def test_pattern_stays_in_its_volume_and_moves_with_time() -> None:
    points = wp.zeros(COUNT, dtype=wp.vec3, device="cpu")

    write_test_pattern(points, time_s=0.0)
    first = points.numpy().copy()
    write_test_pattern(points, time_s=1.0)
    later = points.numpy()

    offset = first - np.array(PATTERN_CENTRE_M)
    assert np.isfinite(first).all()
    assert np.linalg.norm(offset[:, :2], axis=1).max() <= PATTERN_RADIUS_M + 1e-5
    assert np.abs(offset[:, 2]).max() <= PATTERN_RADIUS_M
    assert not np.allclose(first, later)


@pytest.fixture(scope="module")
def window(window_settings: WindowSettings) -> Iterator[GameWindow]:
    with GameWindow(window_settings, visible=False) as w:
        yield w


@pytest.mark.render
def test_points_are_drawn_where_the_pattern_is(window: GameWindow) -> None:
    size = (64, 64)
    target = window.ctx.simple_framebuffer(size)
    eye = (PATTERN_CENTRE_M[0] - 15.0, PATTERN_CENTRE_M[1], PATTERN_CENTRE_M[2])
    view = CameraView(eye, (1.0, 0.0, 0.0), np.radians(60.0), 0.05, 100.0)

    with (
        SharedBuffer(window.ctx, COUNT, wp.vec3, device="cpu") as shared,
        PointsRenderer(window.ctx, shared) as points,
    ):
        with shared.writable() as array:
            write_test_pattern(array, time_s=0.0)
        target.use()
        target.clear(0.0, 0.0, 0.0, 1.0)
        points.render(target, view)
        image = np.frombuffer(target.read(components=3), dtype=np.uint8)

    assert image.reshape(size[1], size[0], 3).sum(axis=2).astype(bool).mean() > 0.01
    target.release()
