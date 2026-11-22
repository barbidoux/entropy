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
  nodes, measured in ADR 0026, not yet adopted). Until then the CPU fixes n from the
  previous frame's speed. When the
  frame's end speed shows it needed more substeps than it ran, the frame is counted
  in `diagnostics()["overruns"]` and logged (the 1st, 10th, 100th...). A speed peak
  inside a frame that is gone by its end is not seen.
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

import logging
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import warp as wp

from entropy.core.paths import out_dir
from entropy.fields.mac import GridSpec, MacVelocity, sample_cells, sample_velocity
from entropy.solvers.interface import NonFiniteStateError
from entropy.solvers.liquid import _solver_kernels as _k
from entropy.solvers.liquid.particles import LiquidParticles
from entropy.solvers.liquid.projection import Projection
from entropy.solvers.liquid.sorting import CellSort
from entropy.solvers.liquid.transfer import grid_to_particles, particles_to_grid

_log = logging.getLogger(__name__)


def _worth_logging(count: int) -> bool:
    """The 1st, 10th, 100th... occurrence: a sustained one does not flood the log."""
    return count > 0 and 10 ** round(math.log10(count)) == count


__all__ = ["LiquidSetup", "LiquidSolver", "NonFiniteStateError", "fill_box", "load_dump"]

MAX_SUBSTEPS = 4
"""Substep cap per frame (SPEC 5.9, ADR 0013)."""
SURFACE_FILL = 0.5
"""Density at the free surface: half of water at rest (ADR 0018)."""
SURFACE_SLOPE_PER_CELL = 2.0 / 3.0
"""How fast the density rises across a flat surface, per cell, at its middle: the
quadratic B-spline's 3/4 on the faces level with the cell, 1/2 on those above and
below, averaged over the six faces. phi = (fill - density) * dx / slope is then close
to the distance to the surface (m) near it."""
VOLUME_CORRECTION_PER_S = 6.0
"""Local correction's rate (1/s): a cell's density error asks for this divergence per
unit of error (ADR 0018). Per second, not per substep, so slow motion corrects as fast
per simulated second (ADR 0013); capped so that one substep never asks for more than
its whole error."""
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

    @property
    def particle_volume_m3(self) -> float:
        """The volume of water one particle stands for: a cell over its particles."""
        return self.spec.dx_m**3 / self.particles_per_cell

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


class PendingCheckpoint:
    """A copy of the liquid's state on its way to host memory (`checkpoint_async`).

    On CUDA the copies go to pinned memory on the device's stream, and an event marks
    their end; `result()` waits for it, so only restoring can wait, not taking. The host
    buffers (`buffers`) can be larger than the state (`arrays` are views of their heads),
    so that a checkpoint handed back to `checkpoint_async` lends them to the next one."""

    def __init__(
        self,
        arrays: dict[str, wp.array],
        event: wp.Event | None,
        extra: dict,
        buffers: dict[str, wp.array] | None = None,
    ) -> None:
        self._arrays = arrays
        self._event = event
        self._extra = extra
        self.buffers = buffers if buffers is not None else dict(arrays)

    def wait(self) -> None:
        """Until the copies have landed."""
        if self._event is not None:
            wp.synchronize_event(self._event)
            self._event = None

    def result(self) -> dict[str, Any]:
        """The state, as `checkpoint()` gives it."""
        self.wait()
        state: dict[str, Any] = {name: a.numpy().copy() for name, a in self._arrays.items()}
        state.update(self._extra)
        return state


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
        # The state each frame starts from, for the NaN guard's dump; each frame the two
        # buffers swap, as the particles are sorted from one into the other (#65).
        self._before = LiquidParticles(
            wp.zeros(capacity, dtype=wp.vec3, device=device),
            wp.zeros(capacity, dtype=wp.vec3, device=device),
            wp.zeros(capacity, dtype=wp.mat33, device=device),
        )
        self._before_pressure = wp.zeros(spec.shape, dtype=float, device=device)
        self._before_count = 0
        self._sort = CellSort(spec, capacity=capacity, device=device)
        self._velocity = MacVelocity.zeros(spec, device)
        self._mass = MacVelocity.zeros(spec, device)
        self._projection = Projection(spec, setup.density_kg_per_m3, device)
        solid_phi = setup.solid_phi_m if setup.solid_phi_m is not None else np.ones(spec.shape)
        self._solid_phi = wp.array(solid_phi, dtype=float, device=device)
        open_cells = int(np.count_nonzero(np.asarray(solid_phi) > 0.0))
        self._water_capacity = min(capacity, open_cells * setup.particles_per_cell)
        """The most particles the grid holds: its volume of water at rest, outside the
        solids (a closed box in M1, ADR 0017), and at most the particle capacity."""
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
        self._affine_scratch: wp.array | None = None  # made on the first removal
        self._solid_distance = wp.zeros(capacity, dtype=float, device=device)
        self._shifted = wp.zeros(3 * capacity, dtype=wp.vec3, device=device)
        self._shifted_distance = wp.zeros(3 * capacity, dtype=float, device=device)
        self._status = wp.zeros(5, dtype=float, device=device)
        cuda = wp.get_device(device).is_cuda
        self._status_host = wp.zeros(5, dtype=float, device="cpu", pinned=cuda)
        self._water = (0.0, 0.0, 0.0)  # (volume in cells, particles in it, fluid cells)
        self._timer: Any = None
        self._status_event = wp.Event(device) if cuda else None
        self._status_pending = False
        self._frozen: NonFiniteStateError | None = None
        self._speed_m_per_s = 0.0
        self._last_frame: tuple[float, int] | None = None  # (dt_s, substeps) of the last step
        self._diagnostics = {
            "substeps": 0,
            "pressure_iterations": 0,
            "overruns": 0,
            "unconverged_solves": 0,
        }
        self._density = wp.zeros(spec.shape, dtype=float, device=device)
        self._rest_mass = self._mass_at_rest()
        self._volume_slots = wp.zeros(int(_k.SLOTS), dtype=wp.float64, device=device)
        self._inside_slots = wp.zeros(int(_k.SLOTS), dtype=wp.float64, device=device)
        self._cell_slots = wp.zeros(int(_k.SLOTS), dtype=wp.float64, device=device)
        self._totals = wp.zeros(3, dtype=wp.float64, device=device)
        self._volume_error = wp.zeros(2, dtype=wp.float64, device=device)
        self._before_volume_error = wp.zeros(2, dtype=wp.float64, device=device)
        """[relative volume error, its integral over time (s)]: the volume control's state."""
        self._phi_at = wp.zeros(capacity, dtype=float, device=device)

    def step(self, dt_s: float, ctx: Any = None) -> None:
        """Advance the liquid by `dt_s` seconds, in at most `MAX_SUBSTEPS` substeps.

        When `ctx` has a `timer` (a stage timer, `entropy.core.profiler`), the phases are
        timed as "liquid transfers", "liquid pressure" and "liquid motion" (SPEC 4.7)."""
        self._timer = getattr(ctx, "timer", None)
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

    def checkpoint_async(self, reuse: PendingCheckpoint | None = None) -> PendingCheckpoint:
        """A copy of the full state to host memory, enqueued without waiting (SPEC 4.6):
        the game takes one every 10 s without a hitch. `restore` accepts it.

        `reuse`, a checkpoint no longer needed (the one falling off the ring), lends its
        pinned buffers: allocating pinned memory costs milliseconds of CPU per save
        (measured 5.6 ms at 262 k particles, 11 ms at 655 k). Buffers grow by a quarter
        more than needed when they are too small. `reuse` must not be used again."""
        n, device = self._count, self._device
        cuda = wp.get_device(device).is_cuda
        if reuse is not None:
            reuse.wait()
        lent = reuse.buffers if reuse is not None else {}

        def buffer(name: str, length: int, dtype: Any, grow: bool) -> wp.array:
            host = lent.get(name)
            if host is None or host.dtype != dtype or host.shape[0] < length:
                size = max(length + length // 4 if grow else length, 1)  # a view needs one
                host = wp.empty(size, dtype=dtype, device="cpu", pinned=cuda)
            return host

        buffers: dict[str, wp.array] = {}
        arrays: dict[str, wp.array] = {}
        for name, source, dtype in (
            ("positions_m", self._particles.position_m, wp.vec3),
            ("velocities_m_per_s", self._particles.velocity_m_per_s, wp.vec3),
            ("affine_per_s", self._particles.affine_per_s, wp.mat33),
        ):
            buffers[name] = buffer(name, n, dtype, grow=True)
            arrays[name] = _head(buffers[name], n)
            if n:
                wp.copy(arrays[name], source, count=n)
        for name, source in (
            ("pressure_pa", self._projection.pressure_pa),
            ("volume_error", self._volume_error),
        ):
            host = lent.get(name)
            if host is None or host.shape != source.shape or host.dtype != source.dtype:
                host = wp.empty(source.shape, dtype=source.dtype, device="cpu", pinned=cuda)
            wp.copy(host, source)
            buffers[name] = arrays[name] = host
        event = None
        if cuda:
            event = wp.Event(device)
            wp.get_device(device).stream.record_event(event)
        return PendingCheckpoint(arrays, event, {"count": n}, buffers)

    def restore(self, state: dict[str, Any] | PendingCheckpoint) -> None:
        """Return to `state` (a `checkpoint()`, or a `checkpoint_async()` waited for here);
        a NaN freeze is lifted, and the speed comes from `state`."""
        if isinstance(state, PendingCheckpoint):
            state = state.result()
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
        self._water = (0.0, 0.0, 0.0)
        velocities = np.asarray(state["velocities_m_per_s"], dtype=float).reshape(-1, 3)
        speeds = np.linalg.norm(velocities, axis=1) if len(velocities) else np.zeros(1)
        finite = speeds[np.isfinite(speeds)]
        self._speed_m_per_s = float(finite.max()) if len(finite) else 0.0

    def density_ratio(self) -> np.ndarray:
        """Each cell's mass over that of water at rest there, from the last substep."""
        return self._density.numpy()

    def diagnostics(self) -> dict[str, float]:
        """Counts for the HUD. The water's volume (from the level set, m³) and the
        particles inside it come from the last frame's status, so reading them never
        waits for the GPU."""
        volume_cells, inside, fluid_cells = self._water
        return {
            "particles": self._count,
            "water_volume_m3": volume_cells * self._spec().dx_m ** 3,
            "particles_in_water": inside,
            "water_cells": fluid_cells,
            **self._diagnostics,
        }

    def move_particles(self, dt_s: float) -> None:
        """Step 8 alone: move the particles through the grid's current velocity by
        `dt_s`, kept inside the grid and out of the solids (to check the integrator)."""
        if self._count:
            self._advect(self._active(), dt_s)

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

    def remove_particles_within(self, centre_m: tuple[float, float, float], radius_m: float) -> int:
        """Remove the particles within `radius_m` of `centre_m` (the eraser, #54); the
        others keep their order. Returns how many went. It reads the new count back, so
        it waits for the GPU once: a tool's action, not part of the step."""
        n = self._count
        if n == 0 or radius_m <= 0.0:
            return 0
        device = self._device
        active = self._active()
        keep = wp.zeros(n, dtype=wp.int32, device=device)  # pyright: ignore[reportArgumentType]
        index = wp.zeros(n, dtype=wp.int32, device=device)  # pyright: ignore[reportArgumentType]
        wp.launch(
            _k.outside_sphere,
            dim=n,
            inputs=[active.position_m, wp.vec3(*centre_m), float(radius_m) ** 2, keep],
            device=device,
        )
        wp.utils.array_scan(keep, index, inclusive=False)
        # Scratch arrays free between steps; not `_before`, which the NaN guard may still
        # need for the frame that just ran.
        if self._affine_scratch is None:
            capacity = self._require_setup().max_particles
            self._affine_scratch = wp.zeros(capacity, dtype=wp.mat33, device=device)
        target = LiquidParticles(self._scratch, self._scratch_v, self._affine_scratch)
        wp.launch(
            _k.compact,
            dim=n,
            inputs=[keep, index, *_arrays(active), *_arrays(target)],
            device=device,
        )
        # Only the last entry of each is read back (two values, not 2 n).
        last_keep, last_index = keep[n - 1 :].numpy()[0], index[n - 1 :].numpy()[0]
        kept = int(last_index + last_keep)
        for source, destination in zip(_arrays(target), _arrays(self._particles), strict=True):
            if kept:
                wp.copy(destination, source, count=kept)
        self._count = kept
        return n - kept

    def particle_count(self) -> int:
        """How many particles the liquid holds now (no GPU read)."""
        return self._count

    def room_particles(self) -> int:
        """How many more particles the grid holds (`_water_capacity`): tools add no more
        than this, and lose the rest (`entropy.game.tools.base.Inflow`)."""
        return max(self._water_capacity - self._count, 0)

    def particle_positions(self) -> np.ndarray:
        """The particles' positions (m), copied from the GPU. On the CPU this is a view,
        valid until the next step only: the sort swaps the particle buffers (ADR 0025)."""
        return self._active().position_m.numpy()

    def particle_velocities(self) -> np.ndarray:
        """The particles' velocities (m/s); on the CPU a view, as `particle_positions`."""
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
        """Keep the state this frame starts from (the dump's `before_*`), and sort the
        particles by cell for the frame (#65). The sort gathers them, in cell order, into
        the other buffer, which then becomes the live one: the buffer left behind holds
        the starting state as it was, for the dump, at the cost of the copy it replaces."""
        n = self._count
        if n:
            self._sort.sort(self._active(), self._before)
            self._particles, self._before = self._before, self._particles
        wp.copy(self._before_pressure, self._projection.pressure_pa)
        wp.copy(self._before_volume_error, self._volume_error)
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
        wp.launch(_k.copy_totals, dim=1, inputs=[self._totals, self._status], device=self._device)
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
        speed, nonfinite, volume_cells, inside, cells = (
            float(v) for v in self._status_host.numpy()
        )
        self._water = (volume_cells, inside, cells)
        if nonfinite > 0.0:
            self._frozen = NonFiniteStateError(self._dump())
            raise self._frozen
        if self._last_frame is not None:
            dt_s, ran = self._last_frame
            needed = self._substeps_needed(dt_s, speed)
            if needed > ran:
                self._diagnostics["overruns"] += 1
                if _worth_logging(self._diagnostics["overruns"]):
                    _log.warning(
                        "liquid overrun %d: a frame needed %d substeps and ran %d (SPEC 5.9)",
                        self._diagnostics["overruns"],
                        needed,
                        ran,
                    )
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
            before_volume_error=self._before_volume_error.numpy().copy(),
            dt_s=np.asarray(self._last_frame[0] if self._last_frame else np.nan),
            origin_m=np.asarray(setup.spec.origin_m),
            dx_m=np.asarray(setup.spec.dx_m),
            shape=np.asarray(setup.spec.shape),
            **before,
        )
        return path

    def _stage(self, name: str):
        """A timed stage of the step, when a timer was given; else nothing."""
        return self._timer.stage(name) if self._timer is not None else nullcontext()

    def _substep(self, dt_s: float) -> None:
        setup = self._require_setup()
        spec, device = setup.spec, self._device
        active = self._active()
        n = self._count
        with self._stage("liquid transfers (CUDA)"):
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
        self._volume_slots.zero_()
        self._cell_slots.zero_()
        wp.launch(
            _k.fluid_volume,
            dim=spec.shape,
            inputs=[
                projection.phi_m,
                projection.labels,
                spec.dx_m,
                self._volume_slots,
                self._cell_slots,
            ],
            device=device,
        )
        self._inside_slots.zero_()
        if n:
            phi_at = _head(self._phi_at, n)
            sample_cells(spec, projection.phi_m, active.position_m, phi_at)
            wp.launch(_k.count_inside, dim=n, inputs=[phi_at, self._inside_slots], device=device)
        wp.launch(
            _k.volume_error,
            dim=1,
            inputs=[
                self._volume_slots,
                self._inside_slots,
                self._cell_slots,
                float(setup.particles_per_cell),
                VOLUME_CONTROL_DEAD_BAND,
                VOLUME_CONTROL_INTEGRAL_LIMIT_S,
                dt_s,
                self._volume_error,
                self._totals,
            ],
            device=device,
        )
        wp.launch(
            _k.volume_correction,
            dim=spec.shape,
            inputs=[
                self._density,
                projection.labels,
                min(VOLUME_CORRECTION_PER_S, 1.0 / dt_s),
                self._volume_error,
                VOLUME_CONTROL_PER_S,
                VOLUME_CONTROL_PER_S2,
                projection.target_divergence_per_s,
            ],
            device=device,
        )
        with self._stage("liquid pressure (CUDA)"):
            result = projection.project(self._velocity, dt_s)
        self._diagnostics["pressure_iterations"] = result.iterations
        if not result.converged:  # divergence left behind: counted, not hidden (M1 audit)
            self._diagnostics["unconverged_solves"] += 1
            if _worth_logging(self._diagnostics["unconverged_solves"]):
                _log.warning(
                    "liquid pressure solve %d did not converge: relative residual %.2g "
                    "after %d iterations",
                    self._diagnostics["unconverged_solves"],
                    result.relative_residual,
                    result.iterations,
                )
        self._extrapolate()
        projection.zero_solid_faces(self._velocity)
        if not n:
            return
        with self._stage("liquid transfers (CUDA)"):
            grid_to_particles(self._velocity, active)
        with self._stage("liquid motion (CUDA)"):
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
