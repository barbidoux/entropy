"""The rigid-body solver (SPEC 5.4): extended position-based dynamics with substeps.

Each advance is split into ⌈advance / (1/480 s)⌉ equal substeps (`MAX_SUBSTEP_S`):
16 for a 1/30 s frame, 8 for 1/60 s. A substep predicts every body's motion under
gravity and the gyroscopic torque, then recovers its velocities from that motion
(Müller et al. 2020, "Detailed rigid body simulation with extended position based
dynamics"); constraints such as contacts (#87) will project the positions in between.
The substeps follow the advance, so the solver never limits it (ADR 0013).

State on the device, one entry per body (ADR 0030): position (m), orientation (unit
quaternion x, y, z, w, body to world), velocity (m/s), angular velocity (rad/s, world
frame), principal moments of inertia (kg·m², body frame), mass (kg), shape and size.

Simplifications, named (CLAUDE.md rule 7):
- symplectic Euler per substep: a body in free fall drops ½gt² + ½ght, 0.21 % more
  than ½gt² after 1 s at h = 1/480 s (V-RB-01 allows 0.5 %);
- the gyroscopic torque by the implicit midpoint rule, solved by four fixed-point
  iterations: it keeps a tumbling body's |L| and energy (within 1e-3 over 10 s at 20 and
  30 rad/s in the tests) while |ω|·h is well under 1, a few hundred rad/s;
- positions accumulate in float32: far from the origin a body's position drifts from
  x₀ + vt (1.5 cm after 60 s at 1.26 m/s, 20 to 80 m out), its velocity stays exact;
- checkpoints are synchronous: on CUDA, taking one waits for the queued GPU work (#99);
- no contacts, friction or sleeping yet (#87, #88, #89).

NaN guard (SPEC 5.9): after each step a kernel counts non-finite bodies into a small
status array, copied to the host without waiting; the next call reads it. A non-finite
state is dumped (the frame's starting state, its result and its time step) and raises
`NonFiniteStateError` at that call and every call after it, until `restore`.

Units: SI throughout; gravity in m/s².
"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from entropy.core.paths import out_dir
from entropy.solvers.interface import NonFiniteStateError
from entropy.solvers.rigid import _kernels as _k
from entropy.solvers.rigid.bodies import Body, Shape, mass_properties

MAX_SUBSTEP_S = 1.0 / 480.0
"""The longest rigid substep (SPEC 5.4, ADR 0013)."""

_STATE = (
    ("positions_m", wp.vec3),
    ("orientations", wp.quat),
    ("velocities_m_per_s", wp.vec3),
    ("angular_velocities_rad_per_s", wp.vec3),
)
_PROPERTIES = (
    ("masses_kg", float),
    ("inertia_kg_m2", wp.vec3),
    ("shapes", wp.int32),
    ("sizes_m", wp.vec3),
)


@dataclass(frozen=True)
class RigidSetup:
    """What the rigid solver needs from the world (SPEC 8.1: solvers do not import it)."""

    gravity_m_per_s2: tuple[float, float, float]
    max_bodies: int = 5000
    """SPEC 3's capacity."""
    dump_dir: Path | None = None
    """Where the NaN guard writes its dumps; `out/nan_dumps` by default."""

    def __post_init__(self) -> None:
        if self.max_bodies <= 0:
            raise ValueError(f"max_bodies must be positive, got {self.max_bodies}")
        if len(self.gravity_m_per_s2) != 3 or not all(
            math.isfinite(g) for g in self.gravity_m_per_s2
        ):
            raise ValueError(f"gravity must be three finite numbers, got {self.gravity_m_per_s2}")


def load_dump(path: Path) -> dict[str, np.ndarray]:
    """The arrays of a NaN-guard dump, for a regression test."""
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


class RigidSolver:
    """The rigid bodies of SPEC 5.4; implements `Solver` and the simulation's
    `PacedSolver`."""

    def __init__(self) -> None:
        self._setup: RigidSetup | None = None

    @classmethod
    def create(cls, setup: RigidSetup, device: str) -> RigidSolver:
        solver = cls()
        solver.allocate(setup, device)
        return solver

    # Solver interface ---------------------------------------------------------------

    def allocate(self, world: Any, device: str) -> None:
        """`world` is a `RigidSetup`: the simulation builds it from the map."""
        setup: RigidSetup = world
        self._setup, self._device = setup, device
        n = setup.max_bodies
        self._count = 0
        self._state = {name: wp.zeros(n, dtype=dtype, device=device) for name, dtype in _STATE}
        self._before = {name: wp.zeros(n, dtype=dtype, device=device) for name, dtype in _STATE}
        self._before_count = 0
        self._properties = {
            name: wp.zeros(n, dtype=dtype, device=device)  # pyright: ignore[reportArgumentType]
            for name, dtype in _PROPERTIES
        }
        cuda = wp.get_device(device).is_cuda
        self._status = wp.zeros(1, dtype=float, device=device)
        self._status_host = wp.zeros(1, dtype=float, device="cpu", pinned=cuda)
        self._status_event = wp.Event(device) if cuda else None
        self._status_pending = False
        self._frozen: NonFiniteStateError | None = None
        self._last_dt_s = 0.0
        self._timer: Any = None
        self._diagnostics = {"bodies": 0, "substeps": 0}

    def step(self, dt_s: float, ctx: Any = None) -> None:
        """Advance every body by `dt_s` seconds in substeps of at most `MAX_SUBSTEP_S`.
        With a stage timer in `ctx`, the step is timed as "rigid bodies (CUDA)"."""
        self._timer = getattr(ctx, "timer", None)
        self._read_status()
        if not (math.isfinite(dt_s) and dt_s > 0):
            raise ValueError(f"dt_s must be a positive time step, got {dt_s}")
        setup = self._require_setup()
        n = max(1, math.ceil(dt_s / MAX_SUBSTEP_S - 1e-9))
        h = dt_s / n
        count = self._count
        self._before_count = count
        with self._stage("rigid bodies (CUDA)"):
            for name, _ in _STATE:
                if count:
                    wp.copy(self._before[name], self._state[name], count=count)
            if count:
                for _ in range(n):
                    wp.launch(
                        _k.substep,
                        dim=count,
                        inputs=[
                            self._state["positions_m"],
                            self._state["orientations"],
                            self._state["velocities_m_per_s"],
                            self._state["angular_velocities_rad_per_s"],
                            self._properties["inertia_kg_m2"],
                            wp.vec3(*setup.gravity_m_per_s2),
                            h,
                        ],
                        device=self._device,
                    )
        self._diagnostics["substeps"] = n
        self._last_dt_s = dt_s
        self._send_status()

    def max_advance_s(self) -> float:
        """Unlimited: the substeps follow the advance (ADR 0013). Raises if the last step
        left a non-finite state."""
        self._read_status()
        return math.inf

    def fields(self) -> dict[str, Any]:
        """The bodies' device arrays (the first `body_count()` entries are live)."""
        return {**self._state, **self._properties}

    def checkpoint(self) -> dict[str, Any]:
        n = self._count
        state: dict[str, Any] = {"count": n}
        for name, array in (*self._state.items(), *self._properties.items()):
            state[name] = array.numpy()[:n].copy()
        return state

    def restore(self, state: dict[str, Any]) -> None:
        """Return to a checkpoint; a frozen solver runs again, and a status not yet read
        is dropped."""
        n = int(state["count"])
        capacity = self._require_setup().max_bodies
        if n > capacity:
            raise ValueError(f"the checkpoint holds {n} bodies, over the capacity of {capacity}")
        if self._status_event is not None and self._status_pending:
            wp.synchronize_event(self._status_event)  # its copy must not land on later state
        for name, array in (*self._state.items(), *self._properties.items()):
            if n:
                values = wp.array(state[name], dtype=array.dtype, device=self._device)
                wp.copy(array, values, count=n)
        self._count = n
        self._diagnostics["bodies"] = n
        self._frozen = None
        self._status_pending = False

    def diagnostics(self) -> dict[str, float]:
        return dict(self._diagnostics)

    # Bodies -------------------------------------------------------------------------

    def add_bodies(self, bodies: list[Body]) -> None:
        """Add bodies after the existing ones; refused whole if over the capacity. Raises
        while the solver is frozen by the NaN guard."""
        setup = self._require_setup()
        self._read_status()
        if self._count + len(bodies) > setup.max_bodies:
            raise ValueError(
                f"{len(bodies)} more bodies exceed the capacity of {setup.max_bodies}"
                f" ({self._count} already)"
            )
        if not bodies:
            return
        rows: dict[str, list[Any]] = {name: [] for name, _ in (*_STATE, *_PROPERTIES)}
        for body in bodies:
            mass, inertia = mass_properties(body.shape, body.size_m, body.density_kg_per_m3)
            q = np.asarray(body.orientation, dtype=float)
            rows["positions_m"].append(body.position_m)
            rows["orientations"].append(q / np.linalg.norm(q))
            rows["velocities_m_per_s"].append(body.velocity_m_per_s)
            rows["angular_velocities_rad_per_s"].append(body.angular_velocity_rad_per_s)
            rows["masses_kg"].append(mass)
            rows["inertia_kg_m2"].append(inertia)
            rows["shapes"].append(int(Shape(body.shape)))
            rows["sizes_m"].append(body.size_m)
        start, count = self._count, len(bodies)
        for name, array in (*self._state.items(), *self._properties.items()):
            values = np.asarray(rows[name], dtype=np.int32 if name == "shapes" else np.float32)
            source = wp.array(values, dtype=array.dtype, device=self._device)
            wp.copy(array, source, dest_offset=start, count=count)
        self._count += count
        self._diagnostics["bodies"] = self._count

    def set_body_velocity(self, index: int, velocity_m_per_s: tuple[float, float, float]) -> None:
        """Set one body's velocity (m/s), as a tool would. Raises while frozen."""
        self._read_status()
        self._check_index(index)
        values = wp.array(
            np.asarray([velocity_m_per_s], np.float32), dtype=wp.vec3, device=self._device
        )
        wp.copy(self._state["velocities_m_per_s"], values, dest_offset=index, count=1)

    def body_count(self) -> int:
        return self._count

    def body_positions(self) -> np.ndarray:
        return self._head("positions_m")

    def body_orientations(self) -> np.ndarray:
        return self._head("orientations")

    def body_velocities(self) -> np.ndarray:
        return self._head("velocities_m_per_s")

    def body_angular_velocities(self) -> np.ndarray:
        return self._head("angular_velocities_rad_per_s")

    # Internals ----------------------------------------------------------------------

    def _require_setup(self) -> RigidSetup:
        if self._setup is None:
            raise RuntimeError("allocate the rigid solver first")
        return self._setup

    def _head(self, name: str) -> np.ndarray:
        return self._state[name].numpy()[: self._count].copy()

    def _check_index(self, index: int) -> None:
        if not 0 <= index < self._count:
            raise IndexError(f"no body {index}: there are {self._count}")

    def _stage(self, name: str):
        return self._timer.stage(name) if self._timer is not None else nullcontext()

    def _send_status(self) -> None:
        self._status.zero_()
        if self._count:
            wp.launch(
                _k.count_nonfinite,
                dim=self._count,
                inputs=[*(self._state[name] for name, _ in _STATE), self._status],
                device=self._device,
            )
        wp.copy(self._status_host, self._status)
        if self._status_event is not None:
            wp.get_device(self._device).stream.record_event(self._status_event)
        self._status_pending = True

    def _read_status(self) -> None:
        if self._frozen is not None:
            raise self._frozen
        if not self._status_pending:
            return
        if self._status_event is not None:
            wp.synchronize_event(self._status_event)
        self._status_pending = False
        if float(self._status_host.numpy()[0]) > 0.0:
            self._frozen = NonFiniteStateError(self._dump())
            raise self._frozen

    def _dump(self) -> Path:
        setup = self._require_setup()
        folder = setup.dump_dir or out_dir() / "nan_dumps"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"rigid-{time.strftime('%Y%m%d-%H%M%S')}-{time.perf_counter_ns()}.npz"
        state = self.checkpoint()
        n = self._before_count
        before = {f"before_{name}": self._before[name].numpy()[:n].copy() for name, _ in _STATE}
        arrays = {key: np.asarray(value) for key, value in state.items()}
        arrays["gravity_m_per_s2"] = np.asarray(setup.gravity_m_per_s2)
        np.savez_compressed(path, dt_s=np.asarray(self._last_dt_s), **arrays, **before)  # pyright: ignore[reportArgumentType]
        return path
