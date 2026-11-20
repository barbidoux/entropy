"""View and projection matrices (`entropy.render.camera`), world Z up (ADR 0003)."""

import numpy as np
import pytest

from entropy.render.camera import WORLD_UP, gl_bytes, perspective, view_matrix

EYE = np.array([1.0, 2.0, 3.0])
FORWARD = np.array([1.0, 0.0, 0.0])
UP = np.array([0.0, 0.0, 1.0])


def _ndc(point: np.ndarray, near_m: float = 0.1, far_m: float = 100.0) -> np.ndarray:
    clip = (
        perspective(np.radians(60.0), 16 / 9, near_m, far_m)
        @ view_matrix(EYE, FORWARD)
        @ (np.append(point, 1.0))
    )
    return clip[:3] / clip[3]


def test_a_point_straight_ahead_lands_at_the_screen_centre() -> None:
    ndc = _ndc(EYE + 10.0 * FORWARD)

    assert ndc[:2] == pytest.approx([0.0, 0.0], abs=1e-6)


def test_world_up_is_up_on_screen_and_right_is_right() -> None:
    right = np.cross(FORWARD, UP)

    # 1 m off-axis at 10 m: y = 0.1 / tan(30°) ≈ 0.17, x = that / (16/9) ≈ 0.097.
    assert _ndc(EYE + 10.0 * FORWARD + UP)[1] == pytest.approx(0.1 / np.tan(np.radians(30.0)))
    assert _ndc(EYE + 10.0 * FORWARD + right)[0] == pytest.approx(
        0.1 / np.tan(np.radians(30.0)) / (16 / 9)
    )


def test_near_and_far_planes_map_to_the_ends_of_the_depth_range() -> None:
    assert _ndc(EYE + 0.1 * FORWARD)[2] == pytest.approx(-1.0, abs=1e-5)
    assert _ndc(EYE + 100.0 * FORWARD)[2] == pytest.approx(1.0, abs=1e-5)


def test_field_of_view_edge_lands_on_the_screen_edge() -> None:
    edge = EYE + 10.0 * FORWARD + 10.0 * np.tan(np.radians(30.0)) * UP

    assert _ndc(edge)[1] == pytest.approx(1.0, abs=1e-5)


def test_matrices_are_float32_4x4() -> None:
    for matrix in (view_matrix(EYE, FORWARD), perspective(1.0, 1.0, 0.1, 10.0)):
        assert matrix.shape == (4, 4)
        assert matrix.dtype == np.float32


def test_gl_bytes_are_column_major() -> None:
    matrix = np.arange(16, dtype=np.float32).reshape(4, 4)

    columns = np.frombuffer(gl_bytes(matrix), dtype=np.float32)

    assert list(columns[:4]) == [0.0, 4.0, 8.0, 12.0]


@pytest.mark.parametrize(
    ("fov_y_rad", "aspect", "near_m", "far_m"),
    [
        (0.0, 1.0, 0.1, 10.0),
        (np.pi, 1.0, 0.1, 10.0),
        (1.0, 0.0, 0.1, 10.0),
        (1.0, 1.0, 0.0, 10.0),
        (1.0, 1.0, 1.0, 1.0),
    ],
)
def test_impossible_projections_are_rejected(
    fov_y_rad: float, aspect: float, near_m: float, far_m: float
) -> None:
    """A minimised window has a 0x0 framebuffer: the caller must skip the frame."""
    with pytest.raises(ValueError):
        perspective(fov_y_rad, aspect, near_m, far_m)


def test_world_up_cannot_be_changed() -> None:
    with pytest.raises((TypeError, ValueError, AttributeError)):
        WORLD_UP[2] = -1.0  # type: ignore[index]


def test_looking_straight_up_is_rejected() -> None:
    with pytest.raises(ValueError, match="parallel"):
        view_matrix(EYE, UP)
