"""Multigrid-preconditioned conjugate gradient for the pressure Poisson equation (SPEC 5.2).

The system lives on the cells of a grid (ADR 0015). Each cell is labelled:
- fluid: an unknown;
- air: Dirichlet, p = 0;
- solid: Neumann, no flux.

The grid boundary is a wall. For a fluid cell c the operator is

    (A p)_c = (diag_c · p_c - Σ_fluid neighbours n of c: w_cn · p_n) / Δx²,

with w the open fraction of each face (1 or 0 here; fractional weights for
solids arrive with the variational coupling) and diag_c = Σ_fluid w_cn +
Σ_air w_cn / θ_cn. θ is the fraction of the way from the fluid cell's centre to
the air cell's centre at which p = 0. θ = 1 puts the zero at the air cell's centre
(first order). θ = ½ puts it on the face between them, which is second order
for a cell-centred grid (the ghost-fluid method of Enright et al. 2003). A is
symmetric positive definite when every connected body of fluid touches air.
It is ≈ -∇², so solving A p = f solves -∇²p = f.

CG is preconditioned by one multigrid V-cycle (McAdams, Sifakis, Teran 2010):
- **Coarse grids:** each halves the shape. A coarse cell is air if any child is
  air, else fluid if any is, else solid; coarse operators use unit weights and
  θ = 1.
- **Smoothing:** damped Jacobi, the same number of sweeps before and after the
  coarse correction.
- **Transfers:** trilinear prolongation P between cell centres, and restriction
  R = Pᵀ/8 by scattering, so R is P's exact transpose.
- **Coarsest grid:** a fixed number of Jacobi sweeps.

Together these make the preconditioner symmetric, as CG requires.

CG's scalars stay on the device in float64, so no step of an iteration waits
for the CPU. The residual is read back once every `check_every` iterations. On
CUDA those iterations replay as one CUDA graph (SPEC 3: solver loops launch
from graphs). Iterations past convergence do nothing (alpha = beta = 0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import warp as wp

from entropy.linalg import _mgpcg_kernels as _k

CELL_AIR = 0
CELL_FLUID = 1
CELL_SOLID = 2

_DOT_THREADS = 65536


@dataclass(frozen=True)
class PoissonSystem:
    """Cell labels, face weights and diagonal of A on one grid (see the module docstring)."""

    labels: wp.array
    diag: wp.array
    wx: wp.array
    wy: wp.array
    wz: wp.array
    dx_m: float

    def __post_init__(self) -> None:
        nx, ny, nz = self.labels.shape
        expected = {
            "diag": (nx, ny, nz),
            "wx": (nx + 1, ny, nz),
            "wy": (nx, ny + 1, nz),
            "wz": (nx, ny, nz + 1),
        }
        for name, shape in expected.items():
            array = getattr(self, name)
            if array.shape != shape:
                raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
            if array.device != self.labels.device:
                raise ValueError(f"{name} is on {array.device}, labels on {self.labels.device}")
        if not (math.isfinite(self.dx_m) and self.dx_m > 0):
            raise ValueError(f"dx_m must be a positive length, got {self.dx_m}")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.labels.shape

    @property
    def device(self) -> wp.Device | None:
        return self.labels.device

    def operator_args(self) -> list:
        return [self.labels, self.diag, self.wx, self.wy, self.wz, 1.0 / self.dx_m**2]


def system_from_labels(labels: wp.array, *, dx_m: float, air_theta: float = 1.0) -> PoissonSystem:
    """The system with unit face weights between non-solid cells and one θ for every air face.

    `labels` is an int32 cell array; `air_theta` in (0, 1] places p = 0 between a
    fluid cell and an air neighbour (½: on the face, second order; 1: at the air
    cell's centre).
    """
    if not (math.isfinite(air_theta) and 0.0 < air_theta <= 1.0):
        raise ValueError(f"air_theta must be in (0, 1], got {air_theta}")
    if labels.dtype != wp.int32 or labels.ndim != 3:
        raise ValueError("labels must be a 3D int32 array")
    nx, ny, nz = labels.shape
    device = labels.device
    faces = []
    for offset, shape in (
        ((1, 0, 0), (nx + 1, ny, nz)),
        ((0, 1, 0), (nx, ny + 1, nz)),
        ((0, 0, 1), (nx, ny, nz + 1)),
    ):
        w = wp.empty(shape, dtype=float, device=device)
        wp.launch(_k.face_weights, dim=shape, inputs=[labels, *offset, w], device=device)
        faces.append(w)
    diag = wp.empty(labels.shape, dtype=float, device=device)
    wp.launch(
        _k.diagonal, dim=labels.shape, inputs=[labels, *faces, 1.0 / air_theta, diag], device=device
    )
    return PoissonSystem(labels, diag, *faces, dx_m=dx_m)


def _check_field(system: PoissonSystem, name: str, array: wp.array) -> None:
    if array.shape != system.shape:
        raise ValueError(f"{name} has shape {array.shape}, the system {system.shape}")
    if array.device != system.device:
        raise ValueError(f"{name} is on {array.device}, the system on {system.device}")


def apply(system: PoissonSystem, x: wp.array, out: wp.array) -> None:
    """out = A x on fluid cells, 0 elsewhere."""
    _check_field(system, "x", x)
    _check_field(system, "out", out)
    wp.launch(
        _k.apply, dim=system.shape, inputs=[*system.operator_args(), x, out], device=system.device
    )


@dataclass(frozen=True)
class SolveResult:
    iterations: int
    relative_residual: float
    """‖b - A x‖ / ‖b‖ over fluid cells, from CG's recurrence."""
    converged: bool


@dataclass
class _Level:
    system: PoissonSystem
    x: wp.array
    b: wp.array
    scratch: wp.array


def _coarsen(system: PoissonSystem) -> PoissonSystem:
    shape = tuple((n + 1) // 2 for n in system.shape)
    labels = wp.empty(shape, dtype=wp.int32, device=system.device)  # pyright: ignore[reportArgumentType] (Warp's stubs type dtype as float)
    wp.launch(_k.coarsen_labels, dim=shape, inputs=[system.labels, labels], device=system.device)
    return system_from_labels(labels, dx_m=2.0 * system.dx_m, air_theta=1.0)


@dataclass
class Mgpcg:
    """CG preconditioned by a multigrid V-cycle (or by Jacobi, for comparison)."""

    system: PoissonSystem
    multigrid: bool = True
    smoothing_sweeps: int = 2
    coarsest_sweeps: int = 40
    omega: float = 6.0 / 7.0
    """Jacobi damping: 6/7 smooths the 7-point Laplacian's high frequencies best."""
    min_coarse_cells: int = 8
    """Coarsening stops once a coarse grid would have an axis shorter than this."""
    _levels: list[_Level] = field(init=False, repr=False)
    _vectors: dict[str, wp.array] = field(init=False, repr=False)
    _graph: wp.Graph | None = field(init=False, default=None, repr=False)
    _graph_iterations: int = field(init=False, default=0, repr=False)
    _graph_x: wp.array | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.smoothing_sweeps < 1 or self.coarsest_sweeps < 1 or not 0 < self.omega <= 1:
            raise ValueError("need at least one sweep and 0 < omega ≤ 1")
        device = self.system.device
        levels = [self.system]
        while (
            self.multigrid and min((n + 1) // 2 for n in levels[-1].shape) >= self.min_coarse_cells
        ):
            levels.append(_coarsen(levels[-1]))
        self._levels = [
            _Level(
                s,
                wp.zeros(s.shape, dtype=float, device=device),
                wp.zeros(s.shape, dtype=float, device=device),
                wp.zeros(s.shape, dtype=float, device=device),
            )
            for s in levels
        ]
        self._vectors = {
            name: wp.zeros(self.system.shape, dtype=float, device=device)
            for name in ("r", "z", "p", "q")
        }
        self._vectors["scalars"] = wp.zeros(4, dtype=wp.float64, device=device)

    @property
    def levels(self) -> int:
        return len(self._levels)

    # The preconditioner ----------------------------------------------------------------

    def _jacobi(self, level: _Level, sweeps: int) -> None:
        """`sweeps` damped Jacobi sweeps on level.x towards level.b (ping-pong, even count kept)."""
        s = level.system
        for _ in range(sweeps):
            wp.launch(
                _k.jacobi,
                dim=s.shape,
                inputs=[*s.operator_args(), self.omega, level.x, level.b, level.scratch],
                device=s.device,
            )
            wp.copy(level.x, level.scratch)

    def _vcycle(self, index: int) -> None:
        level = self._levels[index]
        level.x.zero_()
        if index == len(self._levels) - 1:
            self._jacobi(level, self.coarsest_sweeps)
            return
        s = level.system
        self._jacobi(level, self.smoothing_sweeps)
        wp.launch(
            _k.residual,
            dim=s.shape,
            inputs=[*s.operator_args(), level.x, level.b, level.scratch],
            device=s.device,
        )
        coarse = self._levels[index + 1]
        coarse.b.zero_()
        wp.launch(
            _k.restrict, dim=s.shape, inputs=[s.labels, level.scratch, coarse.b], device=s.device
        )
        self._vcycle(index + 1)
        wp.launch(
            _k.prolong_add, dim=s.shape, inputs=[s.labels, coarse.x, level.x], device=s.device
        )
        self._jacobi(level, self.smoothing_sweeps)

    def precondition(self, r: wp.array, z: wp.array) -> None:
        """z = M r: one V-cycle from zero (or z = D⁻¹ r without multigrid)."""
        s = self.system
        if not self.multigrid:
            wp.launch(
                _k.scale_by_inverse_diagonal,
                dim=s.shape,
                inputs=[s.labels, s.diag, 1.0 / s.dx_m**2, r, z],
                device=s.device,
            )
            return
        top = self._levels[0]
        wp.copy(top.b, r)
        self._vcycle(0)
        wp.copy(z, top.x)

    # CG ----------------------------------------------------------------------------------

    def _dot(self, a: wp.array, b: wp.array, slot: int) -> None:
        scalars = self._vectors["scalars"]
        flat_a, flat_b = a.flatten(), b.flatten()
        threads = min(_DOT_THREADS, flat_a.shape[0])
        wp.launch(_k.clear, dim=1, inputs=[scalars, slot], device=self.system.device)
        wp.launch(
            _k.dot,
            dim=threads,
            inputs=[flat_a, flat_b, threads, scalars, slot],
            device=self.system.device,
        )

    def _iterate(self, x: wp.array) -> None:
        """One CG iteration, entirely on the device."""
        v, s = self._vectors, self.system
        n = int(np.prod(s.shape))
        wp.launch(
            _k.apply, dim=s.shape, inputs=[*s.operator_args(), v["p"], v["q"]], device=s.device
        )
        self._dot(v["p"], v["q"], int(_k.PQ))
        wp.launch(
            _k.step_x_and_r,
            dim=n,
            inputs=[
                v["p"].flatten(),
                v["q"].flatten(),
                v["scalars"],
                x.flatten(),
                v["r"].flatten(),
            ],
            device=s.device,
        )
        self.precondition(v["r"], v["z"])
        self._dot(v["r"], v["z"], int(_k.RZ_NEW))
        wp.launch(
            _k.step_p,
            dim=n,
            inputs=[v["z"].flatten(), v["scalars"], v["p"].flatten()],
            device=s.device,
        )
        wp.launch(_k.roll, dim=1, inputs=[v["scalars"]], device=s.device)

    def _batch(self, x: wp.array, iterations: int) -> None:
        """`iterations` CG iterations, replayed as one CUDA graph on a GPU."""
        device = self.system.device
        if device is None or not device.is_cuda:
            for _ in range(iterations):
                self._iterate(x)
            return
        if self._graph is None or self._graph_iterations != iterations or self._graph_x is not x:
            with wp.ScopedCapture(device=device) as capture:
                for _ in range(iterations):
                    self._iterate(x)
            self._graph, self._graph_iterations, self._graph_x = capture.graph, iterations, x
        assert self._graph is not None
        wp.capture_launch(self._graph)

    def solve(
        self,
        b: wp.array,
        x: wp.array,
        *,
        tolerance: float = 1e-5,
        max_iterations: int = 200,
        check_every: int = 4,
    ) -> SolveResult:
        """Solve A x = b on fluid cells, starting from `x`; x is 0 on other cells after.

        Stops when ‖b - A x‖ / ‖b‖ ≤ `tolerance`, checked every `check_every`
        iterations (one read-back each), or after `max_iterations`.
        """
        _check_field(self.system, "b", b)
        _check_field(self.system, "x", x)
        if check_every < 1 or max_iterations < 0:
            raise ValueError("check_every must be ≥ 1 and max_iterations ≥ 0")
        v, s = self._vectors, self.system
        scalars = v["scalars"]
        wp.launch(_k.mask, dim=s.shape, inputs=[s.labels, x], device=s.device)
        wp.launch(
            _k.residual, dim=s.shape, inputs=[*s.operator_args(), x, b, v["r"]], device=s.device
        )
        wp.copy(v["q"], b)
        wp.launch(_k.mask, dim=s.shape, inputs=[s.labels, v["q"]], device=s.device)
        self._dot(v["q"], v["q"], int(_k.RR))
        b_norm = math.sqrt(float(scalars.numpy()[int(_k.RR)]))
        if b_norm == 0.0:
            x.zero_()
            return SolveResult(iterations=0, relative_residual=0.0, converged=True)
        self.precondition(v["r"], v["z"])
        wp.copy(v["p"], v["z"])
        self._dot(v["r"], v["z"], int(_k.RZ))
        done = 0
        relative = math.inf
        while True:
            self._dot(v["r"], v["r"], int(_k.RR))
            relative = math.sqrt(max(float(scalars.numpy()[int(_k.RR)]), 0.0)) / b_norm
            if relative <= tolerance or done >= max_iterations:
                break
            batch = min(check_every, max_iterations - done)
            self._batch(x, batch)
            done += batch
        return SolveResult(
            iterations=done, relative_residual=relative, converged=relative <= tolerance
        )
