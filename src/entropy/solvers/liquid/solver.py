"""The liquid: APIC particles on a MAC grid, stepped in the order of SPEC 5.2.

Each substep:
1. Particles to grid (APIC).
2. Gravity.
3. The level set from the particles, and cell labels from it and the static solids:
   phi = |x - x̄| - r, with x̄ a weighted mean of the particles within R
   (Zhu & Bridson 2005). It is smooth where nearest-particle distances are lumpy,
   so a flat layer of jittered particles gives a flat surface.
4. Pressure projection with the free surface (ADR 0017).
5. Velocity extrapolated two faces into the air.
6. Grid to particles.
7. Second-order (midpoint) motion, kept inside the grid and pushed out of static solids.

Substeps (ADR 0013): the advance is split into n = ceil(advance (u + sqrt(g dx)) /
dx) substeps, at most `MAX_SUBSTEPS`. u is the largest particle speed at the end of
the previous frame. That speed, and a check that nothing is NaN or infinite, travel
to the host in a small status array: it is copied asynchronously after each frame
and read at the next, so the step adds no synchronisation point.

Simplification, named: ADR 0013 lets the GPU count substeps from the current
speed (conditional graph nodes, #36). Until then the CPU fixes n from the
previous frame's speed. A frame whose speed outgrew that estimate moved water
more than one cell per substep; `diagnostics()["overruns"]` counts such frames.

NaN guard (SPEC 5.9): a non-finite state found in the status raises
`NonFiniteStateError` at the next step, after dumping the state to an `.npz` file that
`load_dump` reads back for a regression test.

`solvers` does not import `world` (SPEC 8.1): `LiquidSetup` carries what the liquid
needs from it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import warp as wp

from entropy.core.paths import out_dir
from entropy.fields.mac import GridSpec, MacVelocity, sample_cells, sample_velocity
from entropy.solvers.liquid import _solver_kernels as _k
from entropy.solvers.liquid.particles import LiquidParticles
from entropy.solvers.liquid.projection import Projection
from entropy.solvers.liquid.transfer import grid_to_particles, particles_to_grid

MAX_SUBSTEPS = 4
"""Substep cap per frame (SPEC 5.9, ADR 0013)."""
LEVEL_SET_KERNEL_SPACINGS = 2.0
"""Zhu & Bridson's kernel radius R, in particle spacings."""
LEVEL_SET_RADIUS_SPACINGS = 0.75
"""The particle radius r of the level set, in particle spacings: with R = 2, a flat
layer of particles puts the surface half a spacing above its top particles, where the
water ends (measured on the still tank, within 1 mm)."""


@dataclass(frozen=True)
class LiquidSetup:
    """What the liquid needs from the world (SPEC 8.1: solvers do not import `world`)."""

    spec: GridSpec
    density_kg_per_m3: float
    gravity_m_per_s2: tuple[float, float, float]
    particles_per_cell: int = 8
    solid_phi_m: np.ndarray | None = None
    """Signed distance of the static solids at cell centres (m), negative inside; None
    when the grid's walls are the only solids (the Test Tank, ADR 0017)."""
    max_particles: int = 2_000_000
    dump_dir: Path | None = None
    """Where the NaN guard writes its dumps; `out/nan_dumps` by default."""


def _head(a: wp.array, n: int) -> wp.array:
    """The first n entries of a 1D array: a contiguous view, not a copy (Warp's stubs
    type every slice as possibly indexed)."""
    return cast(wp.array, a[:n])


class NonFiniteStateError(RuntimeError):
    """The liquid's state holds NaN or infinity; it was dumped to `dump_path`."""

    def __init__(self, dump_path: Path) -> None:
        super().__init__(f"the liquid's state is not finite; dumped to {dump_path}")
        self.dump_path = dump_path


def fill_box(
    low_m: tuple[float, float, float],
    high_m: tuple[float, float, float],
    dx_m: float,
    particles_per_cell: int,
    seed: int,
    *,
    jitter: float = 0.8,
) -> np.ndarray:
    """Particle positions filling a box: `particles_per_cell` (a cube, such as 8) per
    cell of `dx_m`, one in each sub-cube, so the water starts evenly spread. Each is
    moved randomly by up to `jitter` / 2 of a sub-cube from its centre; 0 gives a
    regular lattice and a flat surface."""
    per_axis = round(particles_per_cell ** (1.0 / 3.0))
    if per_axis**3 != particles_per_cell:
        raise ValueError(f"particles_per_cell must be a cube, got {particles_per_cell}")
    spacing = dx_m / per_axis
    low, high = np.asarray(low_m, float), np.asarray(high_m, float)
    counts = np.round((high - low) / spacing).astype(int)
    index = np.stack(np.meshgrid(*(np.arange(n) for n in counts), indexing="ij"), axis=-1)
    if not 0.0 <= jitter <= 1.0:
        raise ValueError(f"jitter must be between 0 and 1, got {jitter}")
    offset = 0.5 + np.random.default_rng(seed).uniform(-jitter / 2, jitter / 2, size=index.shape)
    return (low + (index + offset) * spacing).reshape(-1, 3).astype(np.float32)


def load_dump(path: Path) -> dict[str, np.ndarray]:
    """The arrays of a NaN-guard dump, for a regression test."""
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


class LiquidSolver:
    """The liquid of SPEC 5.2; implements `Solver` and the simulation's `PacedSolver`."""

    def __init__(self) -> None:
        self._setup: LiquidSetup | None = None

    @classmethod
    def create(cls, setup: LiquidSetup, device: str) -> LiquidSolver:
        solver = cls()
        solver.allocate(setup, device)
        return solver

    # Solver interface ---------------------------------------------------------------

    def allocate(self, world: Any, device: str) -> None:
        """`world` is a `LiquidSetup`: the simulation builds it from the map."""
        setup: LiquidSetup = world
        if not (setup.particles_per_cell > 0 and setup.max_particles > 0):
            raise ValueError("particles_per_cell and max_particles must be positive")
        self._setup, self._device = setup, device
        spec = setup.spec
        self._count = 0
        capacity = setup.max_particles
        self._particles = LiquidParticles(
            wp.zeros(capacity, dtype=wp.vec3, device=device),
            wp.zeros(capacity, dtype=wp.vec3, device=device),
            wp.zeros(capacity, dtype=wp.mat33, device=device),
        )
        self._velocity = MacVelocity.zeros(spec, device)
        self._mass = MacVelocity.zeros(spec, device)
        self._projection = Projection(spec, setup.density_kg_per_m3, device)
        self._solid_phi = wp.array(
            setup.solid_phi_m if setup.solid_phi_m is not None else np.ones(spec.shape),
            dtype=float,
            device=device,
        )
        self._valid = [
            (
                wp.zeros(spec.face_shape(a), dtype=wp.int32, device=device),  # pyright: ignore[reportArgumentType]
                wp.zeros(spec.face_shape(a), dtype=wp.int32, device=device),  # pyright: ignore[reportArgumentType]
                wp.zeros(spec.face_shape(a), dtype=float, device=device),
            )
            for a in range(3)
        ]
        self._weight = wp.zeros(spec.shape, dtype=float, device=device)
        self._weighted_x = wp.zeros(spec.shape, dtype=wp.vec3, device=device)
        self._scratch = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self._scratch_v = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self._status = wp.zeros(2, dtype=float, device=device)
        self._status_host = wp.zeros(
            2, dtype=float, device="cpu", pinned=wp.get_device(device).is_cuda
        )
        self._status_event: wp.Event | None = None
        self._speed_m_per_s = 0.0
        self._last_substep_s = 0.0
        self._diagnostics = {"substeps": 0, "pressure_iterations": 0, "overruns": 0}

    def step(self, dt_s: float, ctx: Any = None) -> None:
        """Advance the liquid by `dt_s` seconds, in at most `MAX_SUBSTEPS` substeps."""
        self._read_status()
        if not (math.isfinite(dt_s) and dt_s > 0):
            raise ValueError(f"dt_s must be a positive time step, got {dt_s}")
        spec = self._spec()
        n = math.ceil(dt_s * self._cfl_speed() / spec.dx_m)
        n = min(MAX_SUBSTEPS, max(1, n))
        for _ in range(n):
            self._substep(dt_s / n)
        self._diagnostics["substeps"] = n
        self._last_substep_s = dt_s / n
        self._send_status()

    def max_advance_s(self) -> float:
        """Largest advance within the substep cap, from the previous frame's speed."""
        self._read_status()
        speed = self._cfl_speed()
        return math.inf if speed <= 0.0 else MAX_SUBSTEPS * self._spec().dx_m / speed

    def fields(self) -> dict[str, Any]:
        return {
            "positions_m": self._active().position_m,
            "velocity": self._velocity,
            "pressure_pa": self._projection.pressure_pa,
            "phi_m": self._projection.phi_m,
        }

    def checkpoint(self) -> dict[str, Any]:
        active = self._active()
        return {
            "count": self._count,
            "positions_m": active.position_m.numpy().copy(),
            "velocities_m_per_s": active.velocity_m_per_s.numpy().copy(),
            "affine_per_s": active.affine_per_s.numpy().copy(),
            "pressure_pa": self._projection.pressure_pa.numpy().copy(),
        }

    def restore(self, state: dict[str, Any]) -> None:
        self._count = 0
        self.add_particles(state["positions_m"], velocities=state["velocities_m_per_s"])
        self._particles.affine_per_s[: self._count].assign(state["affine_per_s"])
        self._projection.pressure_pa.assign(state["pressure_pa"])

    def diagnostics(self) -> dict[str, float]:
        return {"particles": self._count, **self._diagnostics}

    # Particles ------------------------------------------------------------------------

    def add_particles(
        self, positions_m: np.ndarray, *, velocities: np.ndarray | None = None
    ) -> None:
        """Add particles at `positions_m` (n x 3, m) with `velocities` (m/s, at rest if None)."""
        positions = np.asarray(positions_m, dtype=np.float32).reshape(-1, 3)
        n = len(positions)
        setup = self._require_setup()
        if self._count + n > setup.max_particles:
            raise ValueError(
                f"{self._count + n} particles exceed the capacity of {setup.max_particles}"
            )
        added = LiquidParticles.from_numpy(positions, velocities=velocities, device=self._device)
        end = self._count + n
        wp.copy(self._particles.position_m, added.position_m, dest_offset=self._count, count=n)
        wp.copy(
            self._particles.velocity_m_per_s,
            added.velocity_m_per_s,
            dest_offset=self._count,
            count=n,
        )
        wp.copy(self._particles.affine_per_s, added.affine_per_s, dest_offset=self._count, count=n)
        self._count = end

    def particle_positions(self) -> np.ndarray:
        return self._active().position_m.numpy()

    def particle_velocities(self) -> np.ndarray:
        return self._active().velocity_m_per_s.numpy()

    # Internals --------------------------------------------------------------------------

    def _require_setup(self) -> LiquidSetup:
        if self._setup is None:
            raise RuntimeError("allocate the liquid first")
        return self._setup

    def _spec(self) -> GridSpec:
        return self._require_setup().spec

    def _active(self) -> LiquidParticles:
        n = self._count
        p = self._particles
        return LiquidParticles(
            _head(p.position_m, n), _head(p.velocity_m_per_s, n), _head(p.affine_per_s, n)
        )

    def _cfl_speed(self) -> float:
        """u + sqrt(|g| dx): the speed the CFL condition holds to one cell per substep."""
        g = math.sqrt(sum(c * c for c in self._require_setup().gravity_m_per_s2))
        return self._speed_m_per_s + math.sqrt(g * self._spec().dx_m)

    def _particle_mass_kg(self) -> float:
        setup = self._require_setup()
        return setup.density_kg_per_m3 * setup.spec.dx_m**3 / setup.particles_per_cell

    def _send_status(self) -> None:
        active = self._active()
        self._status.zero_()
        if self._count:
            wp.launch(
                _k.status,
                dim=self._count,
                inputs=[active.position_m, active.velocity_m_per_s, self._status],
                device=self._device,
            )
        wp.copy(self._status_host, self._status)
        device = wp.get_device(self._device)
        if device.is_cuda:
            event = wp.Event(device)
            device.stream.record_event(event)
            self._status_event = event

    def _read_status(self) -> None:
        """The previous frame's status: waits only if that frame is somehow still running."""
        if self._status_event is not None:
            wp.synchronize_event(self._status_event)
            self._status_event = None
        speed, nonfinite = (float(v) for v in self._status_host.numpy())
        if nonfinite > 0.0:
            raise NonFiniteStateError(self._dump())
        if self._last_substep_s and speed * self._last_substep_s > self._spec().dx_m:
            self._diagnostics["overruns"] += 1
            self._last_substep_s = 0.0
        self._speed_m_per_s = speed

    def _dump(self) -> Path:
        setup = self._require_setup()
        folder = setup.dump_dir or out_dir() / "nan_dumps"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"liquid-{time.strftime('%Y%m%d-%H%M%S')}-{time.perf_counter_ns()}.npz"
        state = self.checkpoint()
        np.savez_compressed(
            path,
            positions_m=state["positions_m"],
            velocities_m_per_s=state["velocities_m_per_s"],
            affine_per_s=state["affine_per_s"],
            pressure_pa=state["pressure_pa"],
            origin_m=np.asarray(setup.spec.origin_m),
            dx_m=np.asarray(setup.spec.dx_m),
            shape=np.asarray(setup.spec.shape),
        )
        return path

    def _substep(self, dt_s: float) -> None:
        setup = self._require_setup()
        spec, device = setup.spec, self._device
        active = self._active()
        n = self._count
        origin = wp.vec3(*spec.origin_m)
        particles_to_grid(active, self._particle_mass_kg(), self._velocity, self._mass)
        for g, component in zip(
            setup.gravity_m_per_s2,
            (self._velocity.u, self._velocity.v, self._velocity.w),
            strict=True,
        ):
            if g:
                wp.launch(
                    _k.add_constant,
                    dim=component.shape,
                    inputs=[component, g * dt_s],
                    device=device,
                )
        projection = self._projection
        spacing = spec.dx_m / round(setup.particles_per_cell ** (1.0 / 3.0))
        self._weight.zero_()
        self._weighted_x.zero_()
        if n:
            wp.launch(
                _k.splat_weights,
                dim=n,
                inputs=[
                    active.position_m,
                    origin,
                    spec.dx_m,
                    LEVEL_SET_KERNEL_SPACINGS * spacing,
                    self._weight,
                    self._weighted_x,
                ],
                device=device,
            )
        wp.launch(
            _k.level_set,
            dim=spec.shape,
            inputs=[
                self._weight,
                self._weighted_x,
                origin,
                spec.dx_m,
                LEVEL_SET_RADIUS_SPACINGS * spacing,
                3.0 * spec.dx_m,
                projection.phi_m,
            ],
            device=device,
        )
        has_solids = int(setup.solid_phi_m is not None)
        wp.launch(
            _k.label_cells,
            dim=spec.shape,
            inputs=[projection.phi_m, self._solid_phi, has_solids, projection.labels],
            device=device,
        )
        result = projection.project(self._velocity, dt_s)
        self._diagnostics["pressure_iterations"] = result.iterations
        self._extrapolate()
        if not n:
            return
        grid_to_particles(self._velocity, active)
        self._advect(active, dt_s)

    def _extrapolate(self) -> None:
        """Two layers of velocity into the air, so grid-to-particles reads sound values
        for particles near the surface."""
        device = self._device
        offsets = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        components = (self._velocity.u, self._velocity.v, self._velocity.w)
        for offset, component, (valid, valid_next, scratch) in zip(
            offsets, components, self._valid, strict=True
        ):
            wp.launch(
                _k.valid_faces,
                dim=component.shape,
                inputs=[self._projection.labels, *offset, valid],
                device=device,
            )
            for _ in range(2):
                wp.launch(
                    _k.extrapolate,
                    dim=component.shape,
                    inputs=[component, valid, scratch, valid_next],
                    device=device,
                )
                wp.copy(component, scratch)
                wp.copy(valid, valid_next)

    def _advect(self, active: LiquidParticles, dt_s: float) -> None:
        """Midpoint rule through the grid velocity, then inside the grid and out of solids."""
        spec, device, n = self._spec(), self._device, self._count
        mid, velocity_at = _head(self._scratch, n), _head(self._scratch_v, n)
        sample_velocity(self._velocity, active.position_m, velocity_at)
        wp.launch(
            _k.midpoint,
            dim=n,
            inputs=[active.position_m, velocity_at, 0.5 * dt_s, mid],
            device=device,
        )
        sample_velocity(self._velocity, mid, velocity_at)
        margin = 1e-4 * spec.dx_m
        low = np.asarray(spec.origin_m) + margin
        high = np.asarray(spec.origin_m) + np.asarray(spec.shape) * spec.dx_m - margin
        wp.launch(
            _k.move,
            dim=n,
            inputs=[active.position_m, velocity_at, dt_s, wp.vec3(*low), wp.vec3(*high)],
            device=device,
        )
        if self._require_setup().solid_phi_m is not None:
            self._push_out(active)

    def _push_out(self, active: LiquidParticles) -> None:
        spec, device, n = self._spec(), self._device, self._count
        step = 0.1 * spec.dx_m
        distance = wp.zeros(n, dtype=float, device=device)
        sample_cells(spec, self._solid_phi, active.position_m, distance)
        shifted = wp.zeros(3 * n, dtype=wp.vec3, device=device)
        wp.launch(_k.offsets, dim=n, inputs=[active.position_m, step, shifted], device=device)
        plus = wp.zeros(3 * n, dtype=float, device=device)
        sample_cells(spec, self._solid_phi, shifted, plus)
        wp.launch(
            _k.push_out, dim=n, inputs=[active.position_m, distance, plus, step], device=device
        )
