"""Static solids as signed distance fields (`entropy.world.sdf`), SPEC 5.2."""

import math

import numpy as np
import pytest
import warp as wp

from entropy.world.sdf import Box, Solids, open_box, sample_on_grid, signed_distance

BOX = Box(centre_m=(0.0, 0.0, 0.0), half_size_m=(1.0, 2.0, 3.0))


def _query(solids: Solids, points_m: list[tuple[float, float, float]], device: str):
    points = wp.array(np.asarray(points_m, dtype=np.float32), dtype=wp.vec3, device=device)
    distance = wp.zeros(len(points_m), dtype=float, device=device)
    normal = wp.zeros(len(points_m), dtype=wp.vec3, device=device)
    signed_distance(solids.to_device(device), points, distance, normal)
    return distance.numpy(), normal.numpy()


def test_box_distance_is_exact_outside_and_inside(device: str) -> None:
    points = [(2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (2.0, 3.0, 0.0), (0.0, 0.0, 0.0), (0.5, 0.0, 0.0)]

    distance, _ = _query(Solids((BOX,)), points, device)

    np.testing.assert_allclose(distance, [1.0, 1.0, math.sqrt(2.0), -1.0, -0.5], atol=1e-6)


def test_normals_point_out_of_the_nearest_face(device: str) -> None:
    points = [(2.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, -2.5, 0.0), (0.0, 0.0, -2.9)]

    _, normal = _query(Solids((BOX,)), points, device)

    np.testing.assert_allclose(normal, [(1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 0, -1)], atol=1e-6)


def test_normals_are_the_distance_gradient(device: str) -> None:
    """Away from edges and corners, the normal is the unit gradient of the distance."""
    rng = np.random.default_rng(1)
    points = rng.uniform(-4.0, 4.0, size=(64, 3))
    step = 1e-3
    distance, normal = _query(Solids((BOX,)), points.tolist(), device)
    shifted = [_query(Solids((BOX,)), (points + step * e).tolist(), device)[0] for e in np.eye(3)]
    gradient = (np.stack(shifted, axis=1) - distance[:, None]) / step

    smooth = np.abs(np.linalg.norm(gradient, axis=1) - 1.0) < 1e-2  # not across a crease
    assert smooth.sum() > 40
    np.testing.assert_allclose(normal[smooth], gradient[smooth], atol=2e-2)
    np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1.0, atol=1e-5)


def test_a_union_is_as_near_as_its_nearest_solid(device: str) -> None:
    other = Box(centre_m=(5.0, 0.0, 0.0), half_size_m=(1.0, 1.0, 1.0))

    distance, normal = _query(Solids((BOX, other)), [(3.5, 0.0, 0.0)], device)

    np.testing.assert_allclose(distance, [0.5], atol=1e-6)  # 0.5 from `other`, 2.5 from BOX
    np.testing.assert_allclose(normal, [(-1.0, 0.0, 0.0)], atol=1e-6)


def test_an_open_box_holds_water_and_is_open_at_the_top(device: str) -> None:
    tank = open_box(inner_min_m=(0.0, 0.0, 0.0), inner_max_m=(4.0, 4.0, 2.0), wall_m=0.02)
    points = [
        (2.0, 2.0, 1.0),
        (-0.01, 2.0, 1.0),
        (2.0, 2.0, -0.01),
        (2.0, 2.0, 3.0),
        (4.5, 2.0, 1.0),
    ]

    distance, _ = _query(Solids(tank), points, device)

    assert distance[0] == pytest.approx(1.0, abs=1e-6)  # cavity: the floor is nearest
    assert distance[1] < 0.0 and distance[2] < 0.0  # inside a wall, inside the floor
    assert distance[3] > 1.0  # above the cavity: nothing closes the top
    assert distance[4] == pytest.approx(0.48, abs=1e-6)  # outside, 0.48 m from the wall


def test_sampling_on_a_grid_equals_point_queries_at_cell_centres(device: str) -> None:
    solids = Solids((BOX,))
    origin_m, dx_m, shape = (-1.5, -2.5, -3.5), 0.5, (6, 10, 14)
    grid = wp.zeros(shape, dtype=float, device=device)

    sample_on_grid(solids.to_device(device), origin_m, dx_m, grid)

    index = np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing="ij"), axis=-1)
    centres = np.asarray(origin_m) + (index.reshape(-1, 3) + 0.5) * dx_m
    expected, _ = _query(solids, centres.tolist(), device)
    np.testing.assert_allclose(grid.numpy().reshape(-1), expected, atol=1e-6)


@pytest.mark.parametrize("half_size_m", [(0.0, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, math.inf)])
def test_boxes_must_have_a_positive_finite_size(half_size_m: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError, match="half_size_m"):
        Box(centre_m=(0.0, 0.0, 0.0), half_size_m=half_size_m)


def test_an_open_box_needs_ordered_bounds_and_a_wall() -> None:
    with pytest.raises(ValueError, match="inner_min_m"):
        open_box(inner_min_m=(0.0, 0.0, 2.0), inner_max_m=(4.0, 4.0, 1.0), wall_m=0.02)
    with pytest.raises(ValueError, match="wall_m"):
        open_box(inner_min_m=(0.0, 0.0, 0.0), inner_max_m=(4.0, 4.0, 2.0), wall_m=0.0)


def test_no_solids_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Solids(())
