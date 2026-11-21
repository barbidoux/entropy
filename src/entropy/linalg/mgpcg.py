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
from graphs). Iterations past convergence only lower the residual further.

Named limit: every connected body of fluid must touch air, or A is singular. A
sealed pocket does not converge (`converged` is false after `max_iterations`), and
an isolated fluid cell gets an arbitrary pressure through the diagonal's 1e-12
floor. The liquid labels such pockets (#49).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import warp as wp

from entropy.linalg import _mgpcg_kernels as _k

CELL_AIR = 0
CELL_FLUID = 1
CELL_SOLID = 2

_DOT_PARTIALS = 32768
"""Threads of a dot product's first stage: enough to stream the arrays at full bandwidth."""
_DOT_GROUP = 128


@dataclass(frozen=True)
class PoissonSystem:
    """Cell labels, face weights and diagonal of A on one grid (see the module docstring)."""

    labels: wp.array
    diag: wp.array
    wx: wp.array
    wy: wp.array
    wz: wp.array
    dx_m: float
    unit_weights: bool = False
    """True when every weight is 0 or 1 as `system_from_labels` sets them: kernels then
    take 1 between fluid cells without reading the weight arrays."""
    air_theta: float = 1.0
    """θ of every air face, used again when the operator is refreshed from new labels."""
    face_theta: tuple[wp.array, wp.array, wp.array] | None = None
    """One θ per face (x, y and z faces), replacing `air_theta` when given: the liquid
    writes them from its level set (second-order free surface). Clamped to [1e-3, 1]."""

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
        if self.face_theta is not None:
            for name, theta, w in zip(
                "xyz", self.face_theta, (self.wx, self.wy, self.wz), strict=True
            ):
                if theta.shape != w.shape or theta.device != w.device:
                    raise ValueError(
                        f"face_theta {name} has shape {theta.shape}, its faces {w.shape}"
                    )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.labels.shape

    @property
    def device(self) -> wp.Device | None:
        return self.labels.device

    def operator_args(self) -> list:
        return [
            self.labels,
            self.diag,
            self.wx,
            self.wy,
            self.wz,
            int(self.unit_weights),
            1.0 / self.dx_m**2,
        ]


def system_from_labels(
    labels: wp.array,
    *,
    dx_m: float,
    air_theta: float = 1.0,
    face_theta: tuple[wp.array, wp.array, wp.array] | None = None,
) -> PoissonSystem:
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
    faces = [
        wp.empty(shape, dtype=float, device=device)
        for shape in ((nx + 1, ny, nz), (nx, ny + 1, nz), (nx, ny, nz + 1))
    ]
    diag = wp.empty(labels.shape, dtype=float, device=device)
    system = PoissonSystem(
        labels,
        diag,
        *faces,
        dx_m=dx_m,
        unit_weights=True,
        air_theta=air_theta,
        face_theta=face_theta,
    )
    _assemble(system)
    return system


def _assemble(system: PoissonSystem) -> None:
    """Recompute, in place, the weights (when unit) and the diagonal from the labels."""
    device = system.device
    if system.unit_weights:
        faces = (system.wx, system.wy, system.wz)
        for offset, w in zip(((1, 0, 0), (0, 1, 0), (0, 0, 1)), faces, strict=True):
            wp.launch(
                _k.face_weights, dim=w.shape, inputs=[system.labels, *offset, w], device=device
            )
    if system.face_theta is not None:
        wp.launch(
            _k.diagonal_face_theta,
            dim=system.shape,
            inputs=[
                system.labels,
                system.wx,
                system.wy,
                system.wz,
                *system.face_theta,
                system.diag,
            ],
            device=device,
        )
        return
    wp.launch(
        _k.diagonal,
        dim=system.shape,
        inputs=[
            system.labels,
            system.wx,
            system.wy,
            system.wz,
            1.0 / system.air_theta,
            system.diag,
        ],
        device=device,
    )


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
    """‖r‖ / ‖b‖ over fluid cells, with r CG's recursively updated residual: an estimate
    of ‖b - A x‖ / ‖b‖, close to it down to float32's floor (about 1e-6)."""
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


def _recoarsen(fine: PoissonSystem, coarse: PoissonSystem) -> None:
    """`_coarsen` into the existing coarse arrays."""
    wp.launch(
        _k.coarsen_labels, dim=coarse.shape, inputs=[fine.labels, coarse.labels], device=fine.device
    )
    _assemble(coarse)


@dataclass(frozen=True)
class Mgpcg:
    """CG preconditioned by a multigrid V-cycle (or by Jacobi, for comparison).

    Settings are fixed at construction: on CUDA the V-cycle is captured once, so a
    later change would silently not apply there.
    """

    system: PoissonSystem
    multigrid: bool = True
    smoothing_sweeps: int = 2
    coarsest_sweeps: int = 16
    omega: float = 6.0 / 7.0
    """Jacobi damping: 6/7 smooths the 7-point Laplacian's high frequencies best."""
    min_coarse_cells: int = 4
    """Coarsening stops once a coarse grid would have an axis shorter than this."""
    _levels: list[_Level] = field(init=False, repr=False)
    _vectors: dict[str, wp.array] = field(init=False, repr=False)
    _graphs: dict[str, tuple[wp.Graph, tuple]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.smoothing_sweeps < 1 or self.coarsest_sweeps < 1 or not 0 < self.omega <= 1:
            raise ValueError("need at least one sweep and 0 < omega ≤ 1")
        device = self.system.device
        vectors = {
            name: wp.zeros(self.system.shape, dtype=float, device=device)
            for name in ("r", "z", "p", "q")
        }
        vectors["scalars"] = wp.zeros(5, dtype=wp.float64, device=device)
        partials = min(_DOT_PARTIALS, int(np.prod(self.system.shape)))
        vectors["partials"] = wp.zeros(partials, dtype=wp.float64, device=device)
        systems = [self.system]
        while (
            self.multigrid and min((n + 1) // 2 for n in systems[-1].shape) >= self.min_coarse_cells
        ):
            systems.append(_coarsen(systems[-1]))
        levels = []
        for index, s in enumerate(systems):
            scratch = wp.zeros(s.shape, dtype=float, device=device)
            if index == 0:  # the top level works in the caller's arrays (`precondition`)
                levels.append(_Level(s, vectors["z"], vectors["r"], scratch))
                continue
            x = wp.zeros(s.shape, dtype=float, device=device)
            b = wp.zeros(s.shape, dtype=float, device=device)
            levels.append(_Level(s, x, b, scratch))
        # Frozen: internal state is set once here; only the arrays' contents change later.
        object.__setattr__(self, "_levels", levels)
        object.__setattr__(self, "_vectors", vectors)
        object.__setattr__(self, "_graphs", {})

    def refresh(self) -> None:
        """Recompute the operator from the labels' current contents, in place.

        The liquid relabels cells every substep: it writes the new labels into
        `system.labels` and calls this. Every array stays the same object, so the
        CUDA graphs captured on them stay valid (ADR 0016). With stored weights
        (`unit_weights` False) the caller updates them first; this then redoes the
        diagonal and the coarse levels.
        """
        _assemble(self.system)
        for fine, coarse in zip(self._levels, self._levels[1:], strict=False):
            _recoarsen(fine.system, coarse.system)

    @property
    def levels(self) -> int:
        return len(self._levels)

    # The preconditioner ----------------------------------------------------------------

    def _jacobi(self, level: _Level, sweeps: int) -> None:
        """`sweeps` damped Jacobi sweeps on level.x towards level.b.

        Sweeps alternate between level.x and level.scratch; an odd count ends
        with one copy back into level.x."""
        s = level.system
        src, dst = level.x, level.scratch
        for _ in range(sweeps):
            wp.launch(
                _k.jacobi,
                dim=s.shape,
                inputs=[*s.operator_args(), self.omega, src, level.b, dst],
                device=s.device,
            )
            src, dst = dst, src
        if src is not level.x:
            wp.copy(level.x, src)

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
        top.b, top.x = r, z  # the fine level works in the caller's arrays: no copies
        self._vcycle(0)

    # CG ----------------------------------------------------------------------------------

    def _dot(self, a: wp.array, b: wp.array, slot: int) -> None:
        """scalars[slot] = a·b in two stages: partial sums, then a few atomics."""
        scalars, partials = self._vectors["scalars"], self._vectors["partials"]
        device = self.system.device
        threads = partials.shape[0]
        wp.launch(_k.clear, dim=1, inputs=[scalars, slot], device=device)
        wp.launch(
            _k.dot_partial,
            dim=threads,
            inputs=[a.flatten(), b.flatten(), threads, partials],
            device=device,
        )
        groups = (threads + _DOT_GROUP - 1) // _DOT_GROUP
        wp.launch(
            _k.sum_partials, dim=groups, inputs=[partials, _DOT_GROUP, scalars, slot], device=device
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

    def _run(self, name: str, key: tuple, enqueue: Callable[[], None]) -> None:
        """Run `enqueue` directly on the CPU; on CUDA capture it once per `key` as a graph
        and replay it (SPEC 3: solver loops launch from graphs). `key` holds the arrays
        the work reads or writes, so a graph is never replayed on other arrays."""
        device = self.system.device
        if device is None or not device.is_cuda:
            enqueue()
            return
        cached = self._graphs.get(name)
        if (
            cached is None
            or len(cached[1]) != len(key)
            or any(a is not b for a, b in zip(cached[1], key, strict=True))
        ):
            with wp.ScopedCapture(device=device) as capture:
                enqueue()
            assert capture.graph is not None
            cached = (capture.graph, key)
            self._graphs[name] = cached
        wp.capture_launch(cached[0])

    def _start(self, b: wp.array, x: wp.array) -> None:
        """r = b - A x, z = M r, p = z, and r·z, r·r and b·b, all on the device."""
        v, s = self._vectors, self.system
        wp.launch(_k.mask, dim=s.shape, inputs=[s.labels, x], device=s.device)
        wp.launch(
            _k.residual, dim=s.shape, inputs=[*s.operator_args(), x, b, v["r"]], device=s.device
        )
        wp.copy(v["q"], b)
        wp.launch(_k.mask, dim=s.shape, inputs=[s.labels, v["q"]], device=s.device)
        self._dot(v["q"], v["q"], int(_k.BB))
        self.precondition(v["r"], v["z"])
        wp.copy(v["p"], v["z"])
        self._dot(v["r"], v["z"], int(_k.RZ))
        self._dot(v["r"], v["r"], int(_k.RR))

    def _batch(self, x: wp.array, iterations: int) -> None:
        """`iterations` CG iterations, then r·r for the convergence check."""
        for _ in range(iterations):
            self._iterate(x)
        self._dot(self._vectors["r"], self._vectors["r"], int(_k.RR))

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

        Stops when the relative residual (see `SolveResult`) is at most
        `tolerance`, checked every `check_every` iterations (one read-back each), or
        after `max_iterations`. `tolerance` must be at least 1e-6, float32's floor.
        On CUDA, pass the same `b` and `x` arrays at every call: graphs are captured
        per array, and new arrays mean a new capture (several ms).
        """
        _check_field(self.system, "b", b)
        _check_field(self.system, "x", x)
        if not (math.isfinite(tolerance) and tolerance >= 1e-6):
            raise ValueError(f"tolerance must be at least 1e-6 (float32), got {tolerance}")
        if check_every < 1 or max_iterations < 0:
            raise ValueError("check_every must be ≥ 1 and max_iterations ≥ 0")
        scalars = self._vectors["scalars"]
        self._run("start", (b, x), lambda: self._start(b, x))
        values = scalars.numpy()  # the first read-back: b·b and r·r together
        b_norm = math.sqrt(float(values[int(_k.BB)]))
        if b_norm == 0.0:
            x.zero_()
            return SolveResult(iterations=0, relative_residual=0.0, converged=True)
        relative = math.sqrt(max(float(values[int(_k.RR)]), 0.0)) / b_norm
        done = 0
        while relative > tolerance and done < max_iterations:
            batch = min(check_every, max_iterations - done)
            self._run(f"batch{batch}", (x,), lambda n=batch: self._batch(x, n))
            done += batch
            relative = math.sqrt(max(float(scalars.numpy()[int(_k.RR)]), 0.0)) / b_norm
        return SolveResult(
            iterations=done, relative_residual=relative, converged=relative <= tolerance
        )
