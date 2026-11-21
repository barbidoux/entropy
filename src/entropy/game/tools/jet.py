"""Water outside the liquid's grid: ballistic parcels until they reach the tank (#53).

The liquid exists only inside its grid, the tank's cavity and some air above it
(ADR 0017). Water a tool throws from outside (the hose, from the camera) flies as
parcels of one particle each, under gravity only, until it:
- enters the grid through its open top, or through a side above the glass's rim:
  it then joins the liquid where it entered, with its velocity;
- reaches a side below the rim: the glass stops it, and it is lost;
- falls below the ground: it is lost.

Simplifications, named (CLAUDE.md rule 7):
- no air drag and no breakup of the jet outside the grid;
- water stopped by the glass or the ground is lost, since water cannot exist outside
  the tank in M1 (ADR 0017); `lost_parcels` counts it;
- a parcel joins the liquid at the end of the frame it entered in, at its end-of-frame
  position: at most one frame of motion is approximated, inside the air above the
  water.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Domain:
    """Where the liquid can hold water: its grid's box, the rim's height (water enters
    the sides only above it) and the ground's height (m)."""

    low_m: Vec3
    high_m: Vec3
    rim_z_m: float
    ground_z_m: float

    def contains(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=float)
        low, high = np.asarray(self.low_m), np.asarray(self.high_m)
        return np.all((p > low) & (p < high), axis=-1)


class Jet:
    """Parcels of water in flight outside the liquid's grid."""

    def __init__(self, gravity_m_per_s2: Vec3) -> None:
        self.gravity = np.asarray(gravity_m_per_s2, dtype=float)
        self.positions_m = np.zeros((0, 3))
        self.velocities_m_per_s = np.zeros((0, 3))
        self.lost_parcels = 0

    def __len__(self) -> int:
        return len(self.positions_m)

    def clear(self) -> None:
        """Drop every parcel in flight (a checkpoint restored: they belong to later)."""
        self.positions_m = np.zeros((0, 3))
        self.velocities_m_per_s = np.zeros((0, 3))

    def launch(self, positions_m: np.ndarray, velocities_m_per_s: np.ndarray) -> None:
        self.positions_m = np.concatenate([self.positions_m, np.reshape(positions_m, (-1, 3))])
        self.velocities_m_per_s = np.concatenate(
            [self.velocities_m_per_s, np.reshape(velocities_m_per_s, (-1, 3))]
        )

    def advance(
        self, dt_s: float, domain: Domain, liquid: Any = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Move every parcel by `dt_s`; those that entered the liquid's grid join
        `liquid` (`add_particles`) and are returned as (positions, velocities); the
        others stay in flight or are dropped."""
        if not len(self):
            return np.zeros((0, 3)), np.zeros((0, 3))
        start = self.positions_m
        end = start + self.velocities_m_per_s * dt_s + 0.5 * self.gravity * dt_s**2
        velocity = self.velocities_m_per_s + self.gravity * dt_s
        inside = domain.contains(end)
        low, high = np.asarray(domain.low_m), np.asarray(domain.high_m)
        from_above = start[:, 2] >= high[2]
        # Height where the path crosses the side it came through (linear along the path).
        crossing_z = np.full(len(start), np.inf)
        for axis in (0, 1):
            for bound in (low[axis], high[axis]):
                d0, d1 = start[:, axis] - bound, end[:, axis] - bound
                crosses = (d0 * d1 < 0) & inside
                share = np.where(crosses, d0 / np.where(crosses, d0 - d1, 1.0), 0.0)
                z = start[:, 2] + share * (end[:, 2] - start[:, 2])
                crossing_z = np.where(crosses, np.minimum(crossing_z, z), crossing_z)
        through_side_above_rim = crossing_z >= domain.rim_z_m
        enters = inside & (from_above | through_side_above_rim)
        stopped = inside & ~enters
        # A side below the rim is glass, whether the path ended inside or beyond it.
        hits_glass = np.zeros(len(start), dtype=bool)
        for axis in (0, 1):
            other = 1 - axis
            for bound in (low[axis], high[axis]):
                d0, d1 = start[:, axis] - bound, end[:, axis] - bound
                crosses = d0 * d1 < 0
                share = np.where(crosses, d0 / np.where(crosses, d0 - d1, 1.0), 0.0)
                at = start + share[:, None] * (end - start)
                on_face = (at[:, other] > low[other]) & (at[:, other] < high[other])
                below_rim = (at[:, 2] < domain.rim_z_m) & (at[:, 2] > low[2])
                hits_glass |= crosses & on_face & below_rim
        on_ground = end[:, 2] < domain.ground_z_m
        lost = (stopped | hits_glass | on_ground) & ~enters
        self.lost_parcels += int(lost.sum())
        keep = ~enters & ~lost
        entered = (end[enters], velocity[enters])
        self.positions_m, self.velocities_m_per_s = end[keep], velocity[keep]
        if liquid is not None and len(entered[0]):
            liquid.add_particles(entered[0], velocities=entered[1])
        return entered
