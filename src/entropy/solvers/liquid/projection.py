"""Pressure projection of the liquid: free surface and static solids (SPEC 5.2, ADR 0017).

Each substep, after transfer and forces, the face velocities u* are made
divergence-free in the water:

1. Faces against static solids and the grid's walls are set to 0 (they do not move).
2. One θ per fluid-air face comes from the liquid's level set `phi_m` (negative in
   the water, metres): the fraction of the way to the air cell's centre where it
   crosses zero. This is the second-order free surface (Enright et al. 2003).
3. Solve A p = -(rho/dt) (div(u*) - d) with MGPCG (ADR 0016), starting from the last
   pressure. d is a target divergence (1/s), zero unless the caller asks for one: the
   liquid's volume correction asks crowded cells to expand (ADR 0018).
4. u = u* - dt/rho grad(p). A missing air pressure is the ghost value that puts p = 0 at θ.

Water sealed from air (#69, ADR 0021): when no cell is air, as when water fills the
whole grid, the pressure is fixed only up to a constant. MGPCG then drops the
right-hand side's part along the constant and sets the pressure's mean to 0; the
velocity update uses only differences, so it is unaffected.

Each substep the caller writes `labels` (fluid, air, solid; `entropy.linalg.mgpcg`
codes) and `phi_m` into this object's own arrays. The arrays stay the same objects,
so the solver's CUDA graphs stay valid (ADR 0016). Faces between two air cells are
left alone: the liquid extrapolates velocity there.

Units: density kg/m³, Δt s, velocity m/s, pressure Pa (relative to the air, which is
at 0).
"""

from __future__ import annotations

import math

import warp as wp

from entropy.fields.mac import GridSpec, MacVelocity, divergence
from entropy.linalg.mgpcg import Mgpcg, SolveResult, system_from_labels
from entropy.solvers.liquid import _projection_kernels as _k

_OFFSETS = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


class Projection:
    """The liquid's pressure projection on one grid, with its arrays and solver."""

    def __init__(self, spec: GridSpec, density_kg_per_m3: float, device: str) -> None:
        if not (math.isfinite(density_kg_per_m3) and density_kg_per_m3 > 0):
            raise ValueError(f"density must be positive, got {density_kg_per_m3}")
        self.spec = spec
        self.density_kg_per_m3 = density_kg_per_m3
        self.device = device
        shape = spec.shape
        self.labels = wp.zeros(shape, dtype=wp.int32, device=device)  # pyright: ignore[reportArgumentType] (Warp's stubs type dtype as float)
        """Cell labels, written by the caller every substep (all air at first)."""
        self.phi_m = wp.zeros(shape, dtype=float, device=device)
        """The liquid's signed distance at cell centres (m), negative in the water."""
        self.pressure_pa = wp.zeros(shape, dtype=float, device=device)
        """The last pressure (Pa): the result, and the next solve's starting point."""
        self.target_divergence_per_s = wp.zeros(shape, dtype=float, device=device)
        """Divergence (1/s) the fluid cells should have after projection; 0 by default."""
        self._theta = tuple(
            wp.ones(spec.face_shape(axis), dtype=float, device=device) for axis in range(3)
        )
        self._divergence = wp.zeros(shape, dtype=float, device=device)
        self._rhs = wp.zeros(shape, dtype=float, device=device)
        system = system_from_labels(self.labels, dx_m=spec.dx_m, face_theta=self._theta)  # pyright: ignore[reportArgumentType]
        self.solver = Mgpcg(system)

    def zero_solid_faces(self, velocity: MacVelocity) -> None:
        """Set the faces against solids and the grid's walls to 0, from the current labels."""
        for offset, component in zip(_OFFSETS, (velocity.u, velocity.v, velocity.w), strict=True):
            wp.launch(
                _k.zero_solid_faces,
                dim=component.shape,
                inputs=[self.labels, *offset, component],
                device=self.device,
            )

    def project(
        self, velocity: MacVelocity, dt_s: float, *, tolerance: float = 1e-5
    ) -> SolveResult:
        """Make `velocity` divergence-free in the fluid cells, in place; return the solve."""
        if not (math.isfinite(dt_s) and dt_s > 0):
            raise ValueError(f"dt_s must be a positive time step, got {dt_s}")
        if velocity.spec != self.spec:
            raise ValueError(f"velocity is on {velocity.spec}, the projection on {self.spec}")
        for offset, theta in zip(_OFFSETS, self._theta, strict=True):
            wp.launch(
                _k.face_theta,
                dim=theta.shape,
                inputs=[self.labels, self.phi_m, *offset, theta],
                device=self.device,
            )
        self.solver.refresh()
        self.zero_solid_faces(velocity)
        divergence(velocity, self._divergence)
        wp.launch(
            _k.right_hand_side,
            dim=self.spec.shape,
            inputs=[
                self.labels,
                self._divergence,
                self.target_divergence_per_s,
                -self.density_kg_per_m3 / dt_s,
                self._rhs,
            ],
            device=self.device,
        )
        result = self.solver.solve(self._rhs, self.pressure_pa, tolerance=tolerance)
        coefficient = dt_s / (self.density_kg_per_m3 * self.spec.dx_m)
        for offset, theta, component in zip(
            _OFFSETS, self._theta, (velocity.u, velocity.v, velocity.w), strict=True
        ):
            wp.launch(
                _k.subtract_gradient,
                dim=component.shape,
                inputs=[self.labels, self.pressure_pa, theta, *offset, coefficient, component],
                device=self.device,
            )
        return result
