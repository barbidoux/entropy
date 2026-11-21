"""The liquid: APIC particles on a MAC grid, stepped in the order of SPEC 5.2.

Each substep:
1. Particles to grid (APIC).
2. Gravity.
3. Density and level set (ADR 0018): each cell's mass from step 1 over the mass of
   water at rest there gives its density (1 inside still water). The surface is
   where it is one half: the particles' mass is smoothed by a symmetric kernel, so
   its half height lies where the water ends, however the particles are arranged.
   Cell labels come from it and the static solids.
4. Volume correction (ADR 0018), a target divergence for the projection:
   - volume control (Kim et al. 2007), proportional and integral: when the level set
     holds less water than the particles inside it carry, every fluid cell expands a
     little, and the other way round. Particles in flight (spray, a hose's jet) are
     not counted until they join the water;
   - local density: a cell inside the water denser than at rest expands a little, a
     sparser one contracts, which undoes the slow clustering of particles.
5. Pressure projection with the free surface (ADR 0017).
6. Velocity extrapolated two faces into the air; wall and solid faces set to 0 again.
7. Grid to particles.
8. Second-order (midpoint) motion, kept inside the grid and pushed out of static solids.

Substeps (ADR 0013): the advance is split into n = ceil(advance (u + sqrt(g dx)) /
dx) substeps, at most `MAX_SUBSTEPS`. u is the largest particle speed at the end of
the previous frame. That speed, and a check that nothing is NaN or infinite, travel
to the host in a small status array: it is copied asynchronously after each frame
and read at the next. (The pressure solve still reads its residual back every few
iterations, ADR 0016, so the step as a whole is not yet free of sync points.)

Simplifications, named:
- ADR 0013 lets the GPU count substeps from the current speed (conditional graph
  nodes, #36). Until then the CPU fixes n from the previous frame's speed. When the
  frame's end speed shows it needed more substeps than it ran, the frame is counted
  in `diagnostics()["overruns"]`. A speed peak inside a frame that is gone by its
  end is not seen.
- On CUDA, particles add to the grid with atomics in no fixed order, so two runs
  differ in the last bits and drift apart over time; on the CPU runs are identical.

NaN guard (SPEC 5.9): a non-finite state found in the status raises
`NonFiniteStateError` at the next call, and every call after it until `restore`. The
dump (an `.npz` file `load_dump` reads) holds the state the failing frame started
from (`before_*`), its time step (`dt_s`) and the state it ended in: stepping the
first once by `dt_s` replays the failure, for a regression test.

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
SURFACE_FILL = 0.5
"""Density at the free surface: half of water at rest (ADR 0018)."""
SURFACE_SLOPE_PER_CELL = 2.0 / 3.0
"""How fast the density rises across a flat surface, per cell, at its middle: the
quadratic B-spline's 3/4 on the faces level with the cell, 1/2 on those above and
below, averaged over the six faces. phi = (fill - density) * dx / slope is then close
to the distance to the surface (m) near it."""
VOLUME_CORRECTION_RATE = 0.1
"""Share of a cell's density error the local correction removes per substep (ADR 0018)."""
VOLUME_CONTROL_PER_S = 1.2
"""Volume control, proportional gain (1/s): divergence per unit of relative volume error."""
VOLUME_CONTROL_PER_S2 = 0.36
"""Volume control, integral gain (1/s²): critically damped with the proportional gain
(0.36 = 1.2² / 4), so a steady loss leaves no lasting error."""
VOLUME_CONTROL_DEAD_BAND = 0.0025
"""Relative volume error the control leaves alone: above the measure's bias on still
water (0.075 %) and its frame-to-frame noise, so still water stays still."""
VOLUME_CONTROL_INTEGRAL_LIMIT_S = 0.02
"""Bound on the integral of the error (s): 2 % for a second at most, against wind-up."""


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

    def __post_init__(self) -> None:
        per_axis = (
            round(self.particles_per_cell ** (1.0 / 3.0)) if self.particles_per_cell > 0 else 0
        )
        if per_axis < 1 or per_axis**3 != self.particles_per_cell:
            raise ValueError(f"particles_per_cell must be a cube, got {self.particles_per_cell}")
        if self.max_particles <= 0:
            raise ValueError(f"max_particles must be positive, got {self.max_particles}")
        if not (math.isfinite(self.density_kg_per_m3) and self.density_kg_per_m3 > 0):
            raise ValueError(f"density must be positive, got {self.density_kg_per_m3}")
        if len(self.gravity_m_per_s2) != 3 or not all(
            math.isfinite(g) for g in self.gravity_m_per_s2
        ):
            raise ValueError(f"gravity must be a finite 3D vector, got {self.gravity_m_per_s2}")
        if self.solid_phi_m is not None and np.shape(self.solid_phi_m) != self.spec.shape:
            raise ValueError(
                f"solid_phi_m has shape {np.shape(self.solid_phi_m)}, the grid {self.spec.shape}"
            )


def _head(a: wp.array, n: int) -> wp.array:
    """The first n entries of a 1D array: a contiguous view, not a copy (Warp's stubs
    type every slice as possibly indexed)."""
    return cast(wp.array, a[:n])


def _arrays(particles: LiquidParticles) -> tuple[wp.array, wp.array, wp.array]:
    return particles.position_m, particles.velocity_m_per_s, particles.affine_per_s


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
        self._setup, self._device = setup, device
        spec = setup.spec
        self._count = 0
        capacity = setup.max_particles
        self._particles = LiquidParticles(
            wp.zeros(capacity, dtype=wp.vec3, device=device),
            wp.zeros(capacity, dtype=wp.vec3, device=device),
            wp.zeros(capacity, dtype=wp.mat33, device=device),
        )
        # The state each frame starts from, for the NaN guard's dump.
        self._before = LiquidParticles(
            wp.zeros(capacity, dtype=wp.vec3, device=device),
            wp.zeros(capacity, dtype=wp.vec3, device=device),
            wp.zeros(capacity, dtype=wp.mat33, device=device),
        )
        self._before_pressure = wp.zeros(spec.shape, dtype=float, device=device)
        self._before_count = 0
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
        self._scratch = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self._scratch_v = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self._solid_distance = wp.zeros(capacity, dtype=float, device=device)
        self._shifted = wp.zeros(3 * capacity, dtype=wp.vec3, device=device)
        self._shifted_distance = wp.zeros(3 * capacity, dtype=float, device=device)
        self._status = wp.zeros(2, dtype=float, device=device)
        cuda = wp.get_device(device).is_cuda
        self._status_host = wp.zeros(2, dtype=float, device="cpu", pinned=cuda)
        self._status_event = wp.Event(device) if cuda else None
        self._status_pending = False
        self._frozen: NonFiniteStateError | None = None
        self._speed_m_per_s = 0.0
        self._last_frame: tuple[float, int] | None = None  # (dt_s, substeps) of the last step
        self._diagnostics = {"substeps": 0, "pressure_iterations": 0, "overruns": 0}
        self._density = wp.zeros(spec.shape, dtype=float, device=device)
        self._rest_mass = self._mass_at_rest()
        self._volume_cells = wp.zeros(1, dtype=wp.float64, device=device)
        self._inside_particles = wp.zeros(1, dtype=wp.float64, device=device)
        self._volume_error = wp.zeros(2, dtype=wp.float64, device=device)
        """[relative volume error, its integral over time (s)]: the volume control's state."""
        self._phi_at = wp.zeros(capacity, dtype=float, device=device)

    def step(self, dt_s: float, ctx: Any = None) -> None:
        """Advance the liquid by `dt_s` seconds, in at most `MAX_SUBSTEPS` substeps."""
        self._read_status()
        if not (math.isfinite(dt_s) and dt_s > 0):
            raise ValueError(f"dt_s must be a positive time step, got {dt_s}")
        self._keep_before()
        n = self._substeps_needed(dt_s, self._speed_m_per_s)
        n = min(MAX_SUBSTEPS, n)
        for _ in range(n):
            self._substep(dt_s / n)
        self._diagnostics["substeps"] = n
        self._last_frame = (dt_s, n)
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
            "volume_error": self._volume_error.numpy().copy(),
        }

    def restore(self, state: dict[str, Any]) -> None:
        """Return to `state`; a NaN freeze is lifted, and the speed comes from `state`."""
        if self._status_pending and self._status_event is not None:
            wp.synchronize_event(self._status_event)
        self._status_pending = False
        self._status_host.zero_()
        self._frozen = None
        self._last_frame = None
        self._count = 0
        self.add_particles(state["positions_m"], velocities=state["velocities_m_per_s"])
        _head(self._particles.affine_per_s, self._count).assign(state["affine_per_s"])
        self._projection.pressure_pa.assign(state["pressure_pa"])
        self._volume_error.assign(state.get("volume_error", np.zeros(2)))
        velocities = np.asarray(state["velocities_m_per_s"], dtype=float).reshape(-1, 3)
        speeds = np.linalg.norm(velocities, axis=1) if len(velocities) else np.zeros(1)
        finite = speeds[np.isfinite(speeds)]
        self._speed_m_per_s = float(finite.max()) if len(finite) else 0.0

    def density_ratio(self) -> np.ndarray:
        """Each cell's mass over that of water at rest there, from the last substep."""
        return self._density.numpy()

    def diagnostics(self) -> dict[str, float]:
        return {"particles": self._count, **self._diagnostics}

    # Particles ------------------------------------------------------------------------

    def add_particles(
        self, positions_m: np.ndarray, *, velocities: np.ndarray | None = None
    ) -> None:
        """Add particles at `positions_m` (n x 3, m) with `velocities` (m/s, at rest if None).

        Their speed joins the speed estimate at once, so the next frame takes enough
        substeps for them (a hose knows how fast its water leaves)."""
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
        if velocities is not None and n:
            speeds = np.linalg.norm(np.asarray(velocities, dtype=float).reshape(-1, 3), axis=1)
            finite = speeds[np.isfinite(speeds)]
            if len(finite):
                self._speed_m_per_s = max(self._speed_m_per_s, float(finite.max()))

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

    def _cfl_speed(self, speed_m_per_s: float | None = None) -> float:
        """u + sqrt(|g| dx): the speed the CFL condition holds to one cell per substep."""
        g = math.sqrt(sum(c * c for c in self._require_setup().gravity_m_per_s2))
        u = self._speed_m_per_s if speed_m_per_s is None else speed_m_per_s
        return u + math.sqrt(g * self._spec().dx_m)

    def _substeps_needed(self, dt_s: float, speed_m_per_s: float) -> int:
        return max(1, math.ceil(dt_s * self._cfl_speed(speed_m_per_s) / self._spec().dx_m))

    def _keep_before(self) -> None:
        """Copy the state this frame starts from (the dump's `before_*`)."""
        n = self._count
        if n:
            for source, target in zip(_arrays(self._particles), _arrays(self._before), strict=True):
                wp.copy(target, source, count=n)
        wp.copy(self._before_pressure, self._projection.pressure_pa)
        self._before_count = n

    def _mass_at_rest(self) -> wp.array:
        """Each cell's mass (kg) when still water fills every cell outside the solids.

        Measured once, with the same transfer as the step: against walls and solids
        the faces gather less, and dividing by this field makes the density 1 there too.
        """
        setup, device = self._require_setup(), self._device
        spec = setup.spec
        (ox, oy, oz), (nx, ny, nz) = spec.origin_m, spec.shape
        high = (ox + nx * spec.dx_m, oy + ny * spec.dx_m, oz + nz * spec.dx_m)
        lattice = fill_box(spec.origin_m, high, spec.dx_m, setup.particles_per_cell, 0, jitter=0.0)
        if setup.solid_phi_m is not None:
            points = wp.array(lattice, dtype=wp.vec3, device=device)
            distance = wp.zeros(len(lattice), dtype=float, device=device)
            sample_cells(spec, self._solid_phi, points, distance)
            lattice = lattice[distance.numpy() >= 0.0]
        particles = LiquidParticles.from_numpy(lattice, device=device)
        velocity, mass = MacVelocity.zeros(spec, device), MacVelocity.zeros(spec, device)
        particles_to_grid(particles, self._particle_mass_kg(), velocity, mass)
        rest = wp.zeros(spec.shape, dtype=float, device=device)
        wp.launch(
            _k.cell_mass, dim=spec.shape, inputs=[mass.u, mass.v, mass.w, rest], device=device
        )
        return rest

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
        if self._status_event is not None:
            wp.get_device(self._device).stream.record_event(self._status_event)
        self._status_pending = True

    def _read_status(self) -> None:
        """The previous frame's status: waits only if that frame is somehow still running.
        Raises the same `NonFiniteStateError` again while the liquid is frozen."""
        if self._frozen is not None:
            raise self._frozen
        if not self._status_pending:
            return
        if self._status_event is not None:
            wp.synchronize_event(self._status_event)
        self._status_pending = False
        speed, nonfinite = (float(v) for v in self._status_host.numpy())
        if nonfinite > 0.0:
            self._frozen = NonFiniteStateError(self._dump())
            raise self._frozen
        if self._last_frame is not None:
            dt_s, ran = self._last_frame
            if self._substeps_needed(dt_s, speed) > ran:
                self._diagnostics["overruns"] += 1
        self._speed_m_per_s = speed

    def _dump(self) -> Path:
        setup = self._require_setup()
        folder = setup.dump_dir or out_dir() / "nan_dumps"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"liquid-{time.strftime('%Y%m%d-%H%M%S')}-{time.perf_counter_ns()}.npz"
        state = self.checkpoint()
        n = self._before_count
        before: dict[str, Any] = {
            f"before_{name}": _head(array, n).numpy().copy()
            for name, array in zip(
                ("positions_m", "velocities_m_per_s", "affine_per_s"),
                _arrays(self._before),
                strict=True,
            )
        }
        np.savez_compressed(
            path,
            positions_m=state["positions_m"],
            velocities_m_per_s=state["velocities_m_per_s"],
            affine_per_s=state["affine_per_s"],
            pressure_pa=state["pressure_pa"],
            before_pressure_pa=self._before_pressure.numpy().copy(),
            dt_s=np.asarray(self._last_frame[0] if self._last_frame else np.nan),
            origin_m=np.asarray(setup.spec.origin_m),
            dx_m=np.asarray(setup.spec.dx_m),
            shape=np.asarray(setup.spec.shape),
            **before,
        )
        return path

    def _substep(self, dt_s: float) -> None:
        setup = self._require_setup()
        spec, device = setup.spec, self._device
        active = self._active()
        n = self._count
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
        mass = self._mass
        wp.launch(
            _k.density_and_level_set,
            dim=spec.shape,
            inputs=[
                mass.u,
                mass.v,
                mass.w,
                self._rest_mass,
                SURFACE_FILL,
                spec.dx_m / SURFACE_SLOPE_PER_CELL,
                self._density,
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
        self._volume_cells.zero_()
        wp.launch(
            _k.fluid_volume,
            dim=spec.shape,
            inputs=[projection.phi_m, projection.labels, spec.dx_m, self._volume_cells],
            device=device,
        )
        self._inside_particles.zero_()
        if n:
            phi_at = _head(self._phi_at, n)
            sample_cells(spec, projection.phi_m, active.position_m, phi_at)
            wp.launch(
                _k.count_inside, dim=n, inputs=[phi_at, self._inside_particles], device=device
            )
        wp.launch(
            _k.volume_error,
            dim=1,
            inputs=[
                self._volume_cells,
                self._inside_particles,
                float(setup.particles_per_cell),
                VOLUME_CONTROL_DEAD_BAND,
                VOLUME_CONTROL_INTEGRAL_LIMIT_S,
                dt_s,
                self._volume_error,
            ],
            device=device,
        )
        wp.launch(
            _k.volume_correction,
            dim=spec.shape,
            inputs=[
                self._density,
                projection.labels,
                VOLUME_CORRECTION_RATE / dt_s,
                self._volume_error,
                VOLUME_CONTROL_PER_S,
                VOLUME_CONTROL_PER_S2,
                projection.target_divergence_per_s,
            ],
            device=device,
        )
        result = projection.project(self._velocity, dt_s)
        self._diagnostics["pressure_iterations"] = result.iterations
        self._extrapolate()
        projection.zero_solid_faces(self._velocity)
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
        distance = _head(self._solid_distance, n)
        sample_cells(spec, self._solid_phi, active.position_m, distance)
        shifted = _head(self._shifted, 3 * n)
        wp.launch(_k.offsets, dim=n, inputs=[active.position_m, step, shifted], device=device)
        plus = _head(self._shifted_distance, 3 * n)
        sample_cells(spec, self._solid_phi, shifted, plus)
        wp.launch(
            _k.push_out, dim=n, inputs=[active.position_m, distance, plus, step], device=device
        )
