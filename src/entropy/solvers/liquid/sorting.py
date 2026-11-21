"""Liquid particles sorted by cell (#65).

Particles that share a cell then sit together in memory, so the transfers read and
write the grid in order instead of at random. Measured on the laptop at 1 M particles
in the Test Tank's grid (unofficial, GPU shared): particles to grid 3.4 to 2.2 ms per
substep, grid to particles 0.87 to 0.15 ms, for a sort of 0.8 ms per frame.

Cells are ordered x-major (x, then y, then z), the grid's own memory order; a
particle outside the grid sorts with the nearest cell. The sort is stable (a radix
sort of the cell keys), so particles that share a cell keep their order, and the
order on CPU is reproducible.
"""

from __future__ import annotations

import warp as wp

from entropy.fields.mac import GridSpec
from entropy.solvers.liquid import _sorting_kernels as _k
from entropy.solvers.liquid.particles import LiquidParticles


class CellSort:
    """Sorts up to `capacity` particles by cell on one device; its key arrays are made
    once (a radix sort needs twice the length as scratch)."""

    def __init__(self, spec: GridSpec, *, capacity: int, device: str) -> None:
        if capacity < 0:
            raise ValueError(f"capacity must be zero or more, got {capacity}")
        self.spec = spec
        self.capacity = capacity
        self.device = device
        self._keys = wp.zeros(2 * max(capacity, 1), dtype=wp.int32, device=device)  # pyright: ignore[reportArgumentType]
        self._order = wp.zeros(2 * max(capacity, 1), dtype=wp.int32, device=device)  # pyright: ignore[reportArgumentType]

    def sort(self, particles: LiquidParticles, out: LiquidParticles) -> None:
        """Write `particles` into `out` in cell order. `out` holds at least as many and is
        other arrays: the sort gathers, it does not move particles in place."""
        n = particles.count
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
                spec.dx_m,
                wp.vec3i(*spec.shape),
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
