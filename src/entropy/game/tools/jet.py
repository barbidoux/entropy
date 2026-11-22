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
  water;
- whether a frame's path crosses the glass or enters is judged on the straight chord
  from where it started to where it ends (for a new parcel, from where the tool says
  its path began, such as the camera): over one frame the arc is close to it.
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
        self._flight_s = np.zeros(0)  # the next advance's duration per parcel; NaN: dt
        self._from_m = np.zeros((0, 3))  # where the next advance's path starts

    def __len__(self) -> int:
        return len(self.positions_m)

    def clear(self) -> None:
        """Drop every parcel in flight (a checkpoint restored: they belong to later)."""
        self.positions_m = np.zeros((0, 3))
        self.velocities_m_per_s = np.zeros((0, 3))
        self._flight_s = np.zeros(0)
        self._from_m = np.zeros((0, 3))

    def launch(
        self,
        positions_m: np.ndarray,
        velocities_m_per_s: np.ndarray,
        *,
        flight_s: np.ndarray | None = None,
        from_m: np.ndarray | None = None,
    ) -> None:
        """New parcels at `positions_m` with `velocities_m_per_s`. On the next advance
        each flies `flight_s` (what is left of the frame after it was launched; the whole
        advance if None), and its path is judged from `from_m` (default: where it is), so
        a tool can have the stretch it came along tested too."""
        positions = np.reshape(positions_m, (-1, 3)).astype(float)
        n = len(positions)
        flight = np.full(n, np.nan) if flight_s is None else np.reshape(flight_s, n)
        start = positions if from_m is None else np.reshape(from_m, (-1, 3)).astype(float)
        self.positions_m = np.concatenate([self.positions_m, positions])
        self.velocities_m_per_s = np.concatenate(
            [self.velocities_m_per_s, np.reshape(velocities_m_per_s, (-1, 3))]
        )
        self._flight_s = np.concatenate([self._flight_s, flight])
        self._from_m = np.concatenate([self._from_m, start])

    def advance(
        self, dt_s: float, domain: Domain, liquid: Any = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Move every parcel by `dt_s`; those that entered the liquid's grid join
        `liquid` (`add_particles`) and are returned as (positions, velocities); the
        others stay in flight or are dropped."""
        if not len(self):
            return np.zeros((0, 3)), np.zeros((0, 3))
        flight = np.where(np.isnan(self._flight_s), dt_s, self._flight_s)[:, None]
        end = self.positions_m + self.velocities_m_per_s * flight + 0.5 * self.gravity * flight**2
        velocity = self.velocities_m_per_s + self.gravity * flight
        start = self._from_m
        inside = domain.contains(end)
        low, high = np.asarray(domain.low_m), np.asarray(domain.high_m)
        # Where the chord enters the grid's box: the last of its three slab entries (a
        # ray-box test). The face it enters by says whether it may: the open top, or a
        # side above the rim; a side below it is glass.
        chord = end - start
        with np.errstate(divide="ignore", invalid="ignore"):
            to_low, to_high = (low - start) / chord, (high - start) / chord
        slab_entry = np.where(chord != 0.0, np.minimum(to_low, to_high), -np.inf)
        entry_axis = np.argmax(slab_entry, axis=1)
        entry = start + np.clip(slab_entry.max(axis=1), 0.0, 1.0)[:, None] * chord
        started_inside = domain.contains(start)
        through_top = (entry_axis == 2) & (chord[:, 2] < 0.0)
        through_side_above_rim = (entry_axis < 2) & (entry[:, 2] >= domain.rim_z_m)
        enters = inside & (started_inside | through_top | through_side_above_rim)
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
        self._flight_s = np.full(int(keep.sum()), np.nan)
        self._from_m = self.positions_m.copy()
        if liquid is not None and len(entered[0]):
            liquid.add_particles(entered[0], velocities=entered[1])
        return entered
