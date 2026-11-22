"""Liquid particles sorted by half-cell (#65, ADR 0025).

Neighbouring particles then sit near each other in memory, so the transfers read and
write the grid in order instead of at random. The key is the half-cell: each cell is
split in two along each axis (`SUBDIVISION`), so the 8 particles of a cell do not sit
in consecutive threads, which would all add to the same 27 grid nodes at once. Measured
at 1 M particles on the Test Tank's grid (unofficial, ADR 0025): particles to grid 3.55
ms in random order, 2.38 ms by cell, 0.98 ms by half-cell.

Half-cells are ordered x-major (x, then y, then z), the grid's own memory order; a
particle outside the grid sorts with the nearest half-cell, each axis clamped on its
own; one whose position is not finite takes key 0. The sort is stable (a radix sort of
the keys), so particles that share a half-cell keep their order, and the order on CPU
is reproducible.
"""

from __future__ import annotations

import warp as wp

from entropy.fields.mac import GridSpec
from entropy.solvers.liquid import _sorting_kernels as _k
from entropy.solvers.liquid.particles import LiquidParticles

SUBDIVISION = 2
"""Each cell's side is split in this many: the sort's key is the half-cell."""


class CellSort:
    """Sorts up to `capacity` particles by half-cell on one device; its key arrays are
    made once (a radix sort needs twice the length as scratch)."""

    def __init__(self, spec: GridSpec, *, capacity: int, device: str) -> None:
        if capacity < 0:
            raise ValueError(f"capacity must be zero or more, got {capacity}")
        self.spec = spec
        self.capacity = capacity
        self.device = device
        self._keys = wp.zeros(2 * max(capacity, 1), dtype=wp.int32, device=device)  # pyright: ignore[reportArgumentType]
        self._order = wp.zeros(2 * max(capacity, 1), dtype=wp.int32, device=device)  # pyright: ignore[reportArgumentType]

    def sort(self, particles: LiquidParticles, out: LiquidParticles) -> None:
        """Write `particles` into `out` in half-cell order. `out` holds at least as many and
        is other arrays: the sort gathers, it does not move particles in place."""
        n = particles.count
        if any(
            a.ptr == b.ptr and n > 0
            for a, b in zip(
                (particles.position_m, particles.velocity_m_per_s, particles.affine_per_s),
                (out.position_m, out.velocity_m_per_s, out.affine_per_s),
                strict=True,
            )
        ):
            raise ValueError("out must be other arrays than particles: the sort gathers")
        if n > self.capacity:
            raise ValueError(f"{n} particles, more than the sort's capacity of {self.capacity}")
        if out.count < n:
            raise ValueError(f"out holds {out.count} particles, fewer than {n}")
        if n == 0:
            return
        spec = self.spec
        wp.launch(
            _k.cell_keys,
            dim=n,
            inputs=[
                particles.position_m,
                wp.vec3(*spec.origin_m),
                spec.dx_m / SUBDIVISION,
                wp.vec3i(*(SUBDIVISION * n for n in spec.shape)),
                self._keys,
                self._order,
            ],
            device=self.device,
        )
        wp.utils.radix_sort_pairs(self._keys, self._order, n)
        wp.launch(
            _k.gather,
            dim=n,
            inputs=[
                self._order,
                particles.position_m,
                particles.velocity_m_per_s,
                particles.affine_per_s,
                out.position_m,
                out.velocity_m_per_s,
                out.affine_per_s,
            ],
            device=self.device,
        )
