"""Static solids as signed distance fields (`entropy.world.sdf`), SPEC 5.2."""

import math

import numpy as np
import pytest
import warp as wp

from entropy.world.sdf import Box, HalfSpace, OpenBox, Solids, sample_on_grid, signed_distance

BOX = Box(centre_m=(0.0, 0.0, 0.0), half_size_m=(1.0, 2.0, 3.0))
TANK = OpenBox(inner_min_m=(0.0, 0.0, 0.0), inner_max_m=(4.0, 4.0, 2.0), wall_m=0.02)


def _query(solids: Solids, points_m, device: str):
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


def test_ties_resolve_x_first_then_y_then_z_and_positive(device: str) -> None:
    """At the centre every face is as near: the documented tie order picks +x."""
    _, normal = _query(Solids((Box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),)), [(0.0, 0.0, 0.0)], device)

    np.testing.assert_allclose(normal, [(1.0, 0.0, 0.0)], atol=1e-6)


def test_normals_are_the_distance_gradient(device: str) -> None:
    """Away from creases, the normal is the unit gradient of the distance."""
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


# 5 mm inside each of the five parts of the tank, next to the cavity, and 5 mm outside.
INSIDE_WALLS = [
    ((-0.005, 2.0, 1.0), (1, 0, 0)),
    ((4.005, 2.0, 1.0), (-1, 0, 0)),
    ((2.0, -0.005, 1.0), (0, 1, 0)),
    ((2.0, 4.005, 1.0), (0, -1, 0)),
    ((2.0, 2.0, -0.005), (0, 0, 1)),
]
OUTSIDE_WALLS = [
    ((-0.025, 2.0, 1.0), (-1, 0, 0)),
    ((4.025, 2.0, 1.0), (1, 0, 0)),
    ((2.0, -0.025, 1.0), (0, -1, 0)),
    ((2.0, 4.025, 1.0), (0, 1, 0)),
    ((2.0, 2.0, -0.025), (0, 0, -1)),
]


@pytest.mark.parametrize(("points", "sign"), [(INSIDE_WALLS, -1.0), (OUTSIDE_WALLS, 1.0)])
def test_every_wall_and_the_floor_of_an_open_box_are_there(device: str, points, sign) -> None:
    distance, normal = _query(Solids((TANK,)), [p for p, _ in points], device)

    np.testing.assert_allclose(distance, np.full(5, sign * 0.005), atol=1e-6)
    np.testing.assert_allclose(normal, [n for _, n in points], atol=1e-6)


def test_junctions_inside_the_glass_point_to_the_nearest_real_surface(device: str) -> None:
    """Where a wall meets the floor or another wall there is no internal face to mistake."""
    points = [(-0.005, 2.0, 0.001), (0.001, -0.005, 1.0), (-0.005, -0.005, 1.0)]

    distance, normal = _query(Solids((TANK,)), points, device)

    # In the corner of the glass the nearest surface is the cavity's vertical edge.
    np.testing.assert_allclose(distance, [-0.005, -0.005, -0.005 * math.sqrt(2.0)], atol=1e-6)
    half = math.sqrt(0.5)
    np.testing.assert_allclose(normal, [(1, 0, 0), (0, 1, 0), (half, half, 0)], atol=1e-6)


def test_an_open_box_holds_water_and_is_open_at_the_top(device: str) -> None:
    points = [(2.0, 2.0, 1.0), (2.0, 2.0, 3.0), (4.5, 2.0, 1.0), (2.0, 2.0, 1.99)]

    distance, normal = _query(Solids((TANK,)), points, device)

    assert distance[0] == pytest.approx(1.0, abs=1e-6)  # cavity: the floor is nearest
    np.testing.assert_allclose(normal[0], (0.0, 0.0, 1.0), atol=1e-6)
    assert distance[1] > 1.0  # above the cavity: nothing closes the top
    assert distance[2] == pytest.approx(0.48, abs=1e-6)  # outside, 0.48 m from the wall
    assert distance[3] == pytest.approx(1.99, abs=1e-5)  # just under the rim: still the floor


def test_a_half_space_is_a_floor(device: str) -> None:
    floor = HalfSpace(point_m=(0.0, 0.0, -0.02), normal=(0.0, 0.0, 2.0))

    distance, normal = _query(Solids((floor,)), [(1.0, 2.0, 3.0), (5.0, -7.0, -1.0)], device)

    np.testing.assert_allclose(distance, [3.02, -0.98], atol=1e-6)
    np.testing.assert_allclose(normal, [(0, 0, 1), (0, 0, 1)], atol=1e-6)


def test_solids_mix_boxes_open_boxes_and_half_spaces(device: str) -> None:
    floor = HalfSpace(point_m=(0.0, 0.0, -0.02), normal=(0.0, 0.0, 1.0))
    solids = Solids((TANK, floor, Box((10.0, 10.0, 1.0), (0.5, 0.5, 0.5))))

    distance, _ = _query(solids, [(2.0, 2.0, 1.0), (10.0, 10.0, 2.0), (8.0, 8.0, 0.5)], device)

    np.testing.assert_allclose(distance, [1.0, 0.5, 0.52], atol=1e-6)


def test_sampling_on_a_grid_equals_point_queries_at_cell_centres(device: str) -> None:
    solids = Solids((BOX, TANK))
    origin_m, dx_m, shape = (-1.5, -2.5, -3.5), 0.5, (12, 14, 14)
    grid = wp.zeros(shape, dtype=float, device=device)

    sample_on_grid(solids.to_device(device), origin_m, dx_m, grid)

    index = np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing="ij"), axis=-1)
    centres = np.asarray(origin_m) + (index.reshape(-1, 3) + 0.5) * dx_m
    expected, _ = _query(solids, centres.tolist(), device)
    np.testing.assert_allclose(grid.numpy().reshape(-1), expected, atol=1e-5)


def test_a_wall_thinner_than_a_cell_can_miss_every_cell_centre(device: str) -> None:
    """Named limit: at 6.25 cm cells the 2 cm glass may have no negative sample, so a
    solver must use point queries, face fractions or thicker walls (decided in #49).
    Here every wall falls between two rows of cell centres (±3.125 cm around it)."""
    grid = wp.zeros((66, 66, 34), dtype=float, device=device)

    sample_on_grid(Solids((TANK,)).to_device(device), (-0.0625, -0.0625, -0.0625), 0.0625, grid)

    assert (grid.numpy() < 0.0).sum() == 0
    assert _query(Solids((TANK,)), [(-0.01, 2.0, 1.0)], device)[0][0] < 0.0


@pytest.mark.parametrize("half_size_m", [(0.0, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, math.inf)])
def test_boxes_must_have_a_positive_finite_size(half_size_m: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError, match="half_size_m"):
        Box(centre_m=(0.0, 0.0, 0.0), half_size_m=half_size_m)


def test_an_open_box_needs_ordered_bounds_and_a_wall() -> None:
    with pytest.raises(ValueError, match="inner_min_m"):
        OpenBox(inner_min_m=(0.0, 0.0, 2.0), inner_max_m=(4.0, 4.0, 1.0), wall_m=0.02)
    with pytest.raises(ValueError, match="wall_m"):
        OpenBox(inner_min_m=(0.0, 0.0, 0.0), inner_max_m=(4.0, 4.0, 2.0), wall_m=0.0)


def test_a_half_space_needs_a_normal() -> None:
    with pytest.raises(ValueError, match="normal"):
        HalfSpace(point_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 0.0))


def test_no_solids_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Solids(())
