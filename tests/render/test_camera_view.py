"""`CameraView`: what the renderer needs to know about the camera."""

import math

import numpy as np
import pytest

from entropy.render.camera import CameraView


def test_matrices_from_a_view() -> None:
    view = CameraView((0.0, 0.0, 2.0), (1.0, 0.0, 0.0), math.radians(60.0), 0.1, 100.0)

    clip = view.projection(16 / 9) @ view.view() @ np.array([10.0, 0.0, 2.0, 1.0])

    assert clip[:2] / clip[3] == pytest.approx([0.0, 0.0], abs=1e-6)


def test_views_are_immutable() -> None:
    view = CameraView((0.0, 0.0, 2.0), (1.0, 0.0, 0.0), 1.0, 0.1, 100.0)

    with pytest.raises(AttributeError):
        view.near_m = 1.0  # type: ignore[misc]
