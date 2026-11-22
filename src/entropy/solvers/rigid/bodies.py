"""What a rigid body is (SPEC 5.4): its shape, size, density and initial motion.

Shapes, in the body's own frame, centred on its centre of mass:
- `Shape.BOX`: `size_m` holds its half sizes along x, y and z;
- `Shape.SPHERE`: `size_m[0]` is its radius;
- `Shape.CAPSULE`: `size_m[0]` is its radius, `size_m[1]` the half length of its
  cylinder along z (a sphere when 0).
Unused entries of `size_m` are ignored. Units: metres, kilograms, seconds, radians.

The principal axes of every shape are the body's x, y and z axes, so its inertia is a
diagonal (`mass_properties`). Orientations are unit quaternions (x, y, z, w), the
rotation from the body's frame to the world's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

Vec3 = tuple[float, float, float]


class Shape(IntEnum):
    BOX = 0
    SPHERE = 1
    CAPSULE = 2


@dataclass(frozen=True)
class Body:
    """A body to add to the rigid solver."""

    shape: Shape
    size_m: Vec3
    density_kg_per_m3: float
    position_m: Vec3 = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Unit quaternion (x, y, z, w); normalised when added, refused when zero."""
    velocity_m_per_s: Vec3 = (0.0, 0.0, 0.0)
    angular_velocity_rad_per_s: Vec3 = (0.0, 0.0, 0.0)
    """In the world's frame."""

    def __post_init__(self) -> None:
        shape = Shape(self.shape)
        size = tuple(float(s) for s in self.size_m)
        needed = {Shape.BOX: 3, Shape.SPHERE: 1, Shape.CAPSULE: 1}[shape]
        if len(size) != 3 or not all(math.isfinite(s) for s in size):
            raise ValueError(f"size_m must be three finite numbers, got {self.size_m}")
        if any(s <= 0.0 for s in size[:needed]) or (shape == Shape.CAPSULE and size[1] < 0.0):
            raise ValueError(f"a {shape.name.lower()}'s size must be positive, got {self.size_m}")
        if not (math.isfinite(self.density_kg_per_m3) and self.density_kg_per_m3 > 0.0):
            raise ValueError(f"density must be positive, got {self.density_kg_per_m3}")
        q = np.asarray(self.orientation, dtype=float)
        if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9:
            raise ValueError(f"orientation must be a non-zero quaternion, got {self.orientation}")
        for name in ("position_m", "velocity_m_per_s", "angular_velocity_rad_per_s"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,):
                raise ValueError(f"{name} must have three components, got {getattr(self, name)}")


def mass_properties(
    shape: Shape, size_m: Vec3, density_kg_per_m3: float
) -> tuple[float, np.ndarray]:
    """(mass in kg, principal moments of inertia in kg·m² about the body's x, y, z axes)."""
    shape = Shape(shape)
    rho = density_kg_per_m3
    if shape == Shape.BOX:
        hx, hy, hz = size_m
        mass = rho * 8.0 * hx * hy * hz
        return mass, mass / 3.0 * np.array([hy**2 + hz**2, hx**2 + hz**2, hx**2 + hy**2])
    if shape == Shape.SPHERE:
        r = size_m[0]
        mass = rho * 4.0 / 3.0 * math.pi * r**3
        return mass, np.full(3, 0.4 * mass * r**2)
    r, h = size_m[0], size_m[1]
    length = 2.0 * h
    cylinder = rho * math.pi * r**2 * length
    hemisphere = rho * 2.0 / 3.0 * math.pi * r**3  # each of the two
    axial = cylinder * r**2 / 2.0 + 2.0 * hemisphere * 0.4 * r**2
    # Each hemisphere about its own centroid (83/320 m r²), moved 3r/8 + h off the centre.
    across = cylinder * (length**2 / 12.0 + r**2 / 4.0) + 2.0 * hemisphere * (
        0.4 * r**2 + length**2 / 4.0 + 3.0 * length * r / 8.0
    )
    return cylinder + 2.0 * hemisphere, np.array([across, across, axial])
