"""The rigid-body solver (SPEC 5.4): extended position-based dynamics with substeps.

Each advance is split into ⌈advance / (1/480 s)⌉ equal substeps (`MAX_SUBSTEP_S`):
16 for a 1/30 s frame, 8 for 1/60 s. A substep predicts every body's motion under
gravity and the gyroscopic torque, solves the contacts' depths once (each pair of
bodies by its own thread, the pairs in parallel, then each body against the static
solids by its own thread), recovers the velocities from the motion, then runs the
velocity pass at the contact patches (Müller et al. 2020, "Detailed rigid body
simulation with extended position based dynamics"; ADR 0031). The substeps follow the
advance, so the solver never limits it (ADR 0013).

State on the device, one entry per body (ADR 0030): position (m), orientation (unit
quaternion x, y, z, w, body to world), velocity (m/s), angular velocity (rad/s, world
frame), principal moments of inertia (kg·m², body frame), mass (kg), shape and size.

Simplifications, named (CLAUDE.md rule 7):
- symplectic Euler per substep: a body in free fall drops ½gt² + ½ght, 0.21 % more
  than ½gt² after 1 s at h = 1/480 s (V-RB-01 allows 0.5 %);
- the gyroscopic torque by the implicit midpoint rule, solved by four fixed-point
  iterations over pieces of the substep short enough for them to converge: it keeps a
  tumbling body's |L| and energy within 1e-3 in the tests (20 and 30 rad/s over 10 s, a
  thin plank at 600 to 3,000 rad/s over 1 s); its energy is capped at its start, so
  a spin never gains energy;
- positions accumulate in float32: far from the origin a body's position drifts from
  x₀ + vt (1.5 cm after 60 s at 1.26 m/s, 20 to 80 m out), its velocity stays exact in
  free flight (contacts see positions to float32's resolution, ADR 0031);
- checkpoints are synchronous: on CUDA, taking one waits for the queued GPU work (#99);
- contacts (ADR 0031, which gives the numbers): edge-on-edge contacts between bodies
  are sampled; a capsule's envelope sags by up to 0.13 r between samples; a fast body
  can cross a thin one where it has no sample, from about 480 s⁻¹ times its thickness;
  small boxes can leave the Test Tank at 30 m/s; without friction a stack is unstable
  (10 µm off brings five boxes down in 7 s, and off the origin float32 positions seed
  it), which friction holds; pairs are Jacobi between them, so stacks sink under load
  (7.7 mm for 10 boxes of 20 cm), a far heavier body sinks into the one under it, and
  several bodies at once still pop and bounce (27 boxes spawned 5 cm into each other
  fly apart at up to 12 m/s); the static solids never give a body energy (its speed is
  scaled down to the kinetic energy it had before their corrections); angular momentum
  is kept within 1.35e-3 of its terms in V-RB-04's scenes, the linear within 1.6e-7; a
  pair overflow loses support; the tank's rims are not sampled as edges;
- friction and restitution (ADR 0032, which gives the numbers): once per contact patch,
  at its centre of pressure; on a wide patch static friction holds the resting body at
  the patch's anchor, at most μs times the patch's normal impulse, and between bodies,
  static or dynamic, it turns neither body, so a box sliding onto another in free fall
  loses angular momentum, and one sliding across another off its centre does not turn
  it; bounces on several bodies at once share one bounce, and the velocity pass does
  not know a body's support; friction can give a little energy between bodies, in fast
  hits or a box tipping off a plank's edge; no anchor on a patch
  that is not wide, so a plank leaning on a wall creeps; no rolling resistance, and no
  torsion at a single contact point; a gentle push springs back by up to 0.2 mm; far
  from the origin, contacts see float32's resolution, and a block just past its
  friction angle may stick from about 20 m; one surface material for all the static
  solids; torsion about 1.4 times too strong on box faces, and dynamic only; bounces
  1.3 to 1.7 % low;
- no sleeping yet (#89).

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
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from entropy.core.paths import out_dir
from entropy.fields.solids import DeviceSolids
from entropy.solvers.interface import NonFiniteStateError
from entropy.solvers.rigid import _kernels as _k
from entropy.solvers.rigid.bodies import Body, Shape, check_surface, mass_properties

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
    ("bounds_m", float),
    ("static_friction", float),
    ("dynamic_friction", float),
    ("restitution", float),
)
PAIRS_PER_BODY = 16
"""Room for pairs of bodies in contact, per body by default: a box in a pile touches up
to about 10 others."""
_BOX_SIGNS = np.array(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    + [[0, sy, sz] for sy in (-1, 1) for sz in (-1, 1)]
    + [[sx, 0, sz] for sx in (-1, 1) for sz in (-1, 1)]
    + [[sx, sy, 0] for sx in (-1, 1) for sy in (-1, 1)]
    + [[s, 0, 0] for s in (-1, 1)]
    + [[0, s, 0] for s in (-1, 1)]
    + [[0, 0, s] for s in (-1, 1)],
    dtype=np.float32,
)
"""A box's samples in units of its half sizes: 8 corners, 12 edge midpoints, 6 face
centres (`_kernels.BOX_SAMPLES`)."""


@dataclass(frozen=True)
class RigidSetup:
    """What the rigid solver needs from the world (SPEC 8.1: solvers do not import it)."""

    gravity_m_per_s2: tuple[float, float, float]
    max_bodies: int = 5000
    """SPEC 3's capacity."""
    dump_dir: Path | None = None
    """Where the NaN guard writes its dumps; `out/nan_dumps` by default."""
    solids: DeviceSolids | None = None
    """The map's static solids, on the solver's device; none when None."""
    max_pairs: int | None = None
    """Room for pairs of bodies in contact each substep; `PAIRS_PER_BODY` per body when
    None. Pairs past it are dropped and counted (`diagnostics()["pair_overflows"]`); the
    contacts with the static solids need no room."""
    static_friction: float = 0.0
    dynamic_friction: float = 0.0
    restitution: float = 0.0
    """The static solids' surface coefficients, one set for all of them (a named
    simplification, ADR 0032); 0, none, when not given."""

    def __post_init__(self) -> None:
        if self.max_bodies <= 0:
            raise ValueError(f"max_bodies must be positive, got {self.max_bodies}")
        if self.max_pairs is not None and self.max_pairs <= 0:
            raise ValueError(f"max_pairs must be positive, got {self.max_pairs}")
        check_surface(self.static_friction, self.dynamic_friction, self.restitution)
        if len(self.gravity_m_per_s2) != 3 or not all(
            math.isfinite(g) for g in self.gravity_m_per_s2
        ):
            raise ValueError(f"gravity must be three finite numbers, got {self.gravity_m_per_s2}")


def _bound(body: Body) -> float:
    """The radius of a sphere about the body's centre that holds it (m)."""
    size = body.size_m
    if body.shape == Shape.BOX:
        return math.sqrt(size[0] ** 2 + size[1] ** 2 + size[2] ** 2)
    if body.shape == Shape.SPHERE:
        return size[0]
    return size[0] + size[1]


def _no_solids(device: str) -> DeviceSolids:
    """No static solids: every array empty."""
    vectors = wp.zeros(0, dtype=wp.vec3, device=device)
    return DeviceSolids(
        box_centres_m=vectors,
        box_halves_m=vectors,
        tank_lows_m=vectors,
        tank_highs_m=vectors,
        tank_walls_m=wp.zeros(0, dtype=float, device=device),
        plane_points_m=vectors,
        plane_normals=vectors,
    )


_ANCHOR_FIELDS = (
    ("other", wp.int32),
    ("stuck", wp.int32),
    ("normal", wp.vec3),
    ("offset_a_m", wp.vec3),
    ("offset_b_m", wp.vec3),
)


def _anchors(size: int, device: str) -> Any:  # a `_kernels.Anchors` (a Warp struct)
    """`size` friction anchors, none set (ADR 0032)."""
    anchors = _k.Anchors()
    for name, dtype in _ANCHOR_FIELDS:
        setattr(anchors, name, wp.zeros(size, dtype=dtype, device=device))  # pyright: ignore[reportArgumentType]
    return anchors


def load_dump(path: Path) -> dict[str, np.ndarray]:
    """The arrays of a NaN-guard dump, for a regression test (`RigidSolver.from_dump`):
    the frame's starting state (`before_*`), its result, its time step, the setup
    (`setup_*`) and the static solids (`solids_*`)."""
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

    @classmethod
    def from_dump(cls, dump: dict[str, np.ndarray], device: str) -> RigidSolver:
        """A solver at a NaN-guard dump's starting state, with its setup (the static
        solids and the room for pairs included): stepping the dump's `dt_s` replays the
        frame, bit for bit on the CPU (rule 8)."""
        values: dict[str, Any] = {}
        for field in fields(RigidSetup):
            key = f"setup_{field.name}"
            if key in dump:
                value = dump[key]
                if value.ndim == 1:
                    values[field.name] = tuple(float(v) for v in value)
                elif np.issubdtype(value.dtype, np.integer):
                    values[field.name] = int(value)
                else:
                    values[field.name] = float(value)
        solids = DeviceSolids(
            **{
                field.name: wp.array(
                    dump[f"solids_{field.name}"],
                    dtype=float if field.name == "tank_walls_m" else wp.vec3,
                    device=device,
                )
                for field in fields(DeviceSolids)
            }
        )
        solver = cls.create(RigidSetup(**values, solids=solids), device)
        start = {
            key.removeprefix("before_"): v for key, v in dump.items() if key.startswith("before_")
        }
        # A dump from before friction (#88) has no coefficients: none made it. Any other
        # property missing is a broken dump, and raises.
        count = int(dump["count"])
        coefficients = ("static_friction", "dynamic_friction", "restitution")
        properties = {
            name: np.zeros(count, np.float32)
            if name in coefficients and name not in dump
            else dump[name]
            for name, _ in _PROPERTIES
        }
        solver.restore({"count": count, **start, **properties})
        return solver

    @property
    def setup(self) -> RigidSetup:
        return self._require_setup()

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
        self._displacement = wp.zeros(n, dtype=wp.vec3, device=device)
        self._turn = wp.zeros(n, dtype=wp.quat, device=device)
        self._shift_m = wp.zeros(n, dtype=wp.vec3, device=device)
        self._spin_rad = wp.zeros(n, dtype=wp.vec3, device=device)
        capacity = setup.max_pairs or PAIRS_PER_BODY * n
        patches = _k.Patches()
        patches.count = wp.zeros(1, dtype=wp.int32, device=device)  # pyright: ignore[reportArgumentType]
        patches.a = wp.zeros(capacity, dtype=wp.int32, device=device)  # pyright: ignore[reportArgumentType]
        patches.b = wp.zeros(capacity, dtype=wp.int32, device=device)  # pyright: ignore[reportArgumentType]
        patches.impulse = wp.zeros(capacity, dtype=float, device=device)
        patches.moment = wp.zeros(capacity, dtype=wp.vec3, device=device)
        patches.normal = wp.zeros(capacity, dtype=wp.vec3, device=device)
        patches.spread = wp.zeros(capacity, dtype=wp.mat33, device=device)
        patches.upper = wp.zeros(capacity, dtype=wp.int32, device=device)  # pyright: ignore[reportArgumentType]
        patches.per_body = wp.zeros(n, dtype=wp.int32, device=device)  # pyright: ignore[reportArgumentType]
        patches.anchor = wp.zeros(capacity, dtype=wp.int32, device=device)  # pyright: ignore[reportArgumentType]
        patches.clamped = wp.zeros(capacity, dtype=wp.int32, device=device)  # pyright: ignore[reportArgumentType]
        self._patches = patches
        # Friction anchors (ADR 0032): the pairs' table is written again each substep from
        # the last one's, so there are two; the static solids' are kept in place.
        self._pair_anchors = _anchors(PAIRS_PER_BODY * n, device)
        self._spare_anchors = _anchors(PAIRS_PER_BODY * n, device)
        self._static_anchors = _anchors(_k.STATIC_PATCHES * n, device)
        self._anchors_before = (
            _anchors(PAIRS_PER_BODY * n, device),
            _anchors(_k.STATIC_PATCHES * n, device),
        )
        static = _k.STATIC_PATCHES * n
        self._static_patches = [
            wp.zeros(static, dtype=float, device=device),
            wp.zeros(static, dtype=wp.vec3, device=device),
            wp.zeros(static, dtype=wp.vec3, device=device),
            wp.zeros(static, dtype=wp.mat33, device=device),
            wp.zeros(static, dtype=wp.int32, device=device),  # pyright: ignore[reportArgumentType]
        ]
        self._static_touch = wp.zeros(static, dtype=wp.vec4, device=device)
        self._static_contacts = wp.zeros(1, dtype=wp.int32, device=device)  # pyright: ignore[reportArgumentType]
        self._rotated_rad = wp.zeros(n, dtype=wp.vec3, device=device)
        self._velocity_before_m_per_s = wp.zeros(n, dtype=wp.vec3, device=device)
        self._angular_before_rad_per_s = wp.zeros(n, dtype=wp.vec3, device=device)
        self._kick_m_per_s = wp.zeros(n, dtype=wp.vec3, device=device)
        self._twist_rad_per_s = wp.zeros(n, dtype=wp.vec3, device=device)
        self._bouncing = wp.zeros(n, dtype=wp.int32, device=device)  # pyright: ignore[reportArgumentType]
        self._kinetic_j = wp.zeros(n, dtype=float, device=device)
        self._contact_count = wp.zeros(1, dtype=wp.int32, device=device)  # pyright: ignore[reportArgumentType]
        self._overflows = wp.zeros(1, dtype=wp.int32, device=device)  # pyright: ignore[reportArgumentType]
        self._signs = wp.array(_BOX_SIGNS, dtype=wp.vec3, device=device)
        self._grid = wp.HashGrid(128, 128, 128, device=device)
        self._max_bound_m = 0.0
        self._solids = setup.solids or _no_solids(device)
        cuda = wp.get_device(device).is_cuda
        self._status = wp.zeros(3, dtype=float, device=device)
        self._status_host = wp.zeros(3, dtype=float, device="cpu", pinned=cuda)
        self._status_event = wp.Event(device) if cuda else None
        self._status_pending = False
        # The dump of the frame that went non-finite; each call raises a new error for it,
        # since a kept one would hold its traceback, and with it this solver, in a cycle.
        self._frozen: Path | None = None
        self._last_dt_s = 0.0
        self._timer: Any = None
        self._diagnostics = {"bodies": 0, "substeps": 0, "contacts": 0, "pair_overflows": 0}

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
            for (_, table, rows), before in zip(
                self._anchor_tables(), self._anchors_before, strict=True
            ):
                for name, _ in _ANCHOR_FIELDS:
                    if count:
                        wp.copy(getattr(before, name), getattr(table, name), count=count * rows)
            self._overflows.zero_()  # the HUD shows the last step's
            if count:
                for _ in range(n):
                    self._substep(h, wp.vec3(*setup.gravity_m_per_s2))
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
        """Every body's state and properties, and the friction anchors (ADR 0032)."""
        n = self._count
        state: dict[str, Any] = {"count": n}
        for name, array in (*self._state.items(), *self._properties.items()):
            state[name] = array.numpy()[:n].copy()
        tables = [table for _, table, _ in self._anchor_tables()]
        return state | self._anchor_state(tables, n)

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
        for prefix, table, rows in self._anchor_tables():
            for name, _ in _ANCHOR_FIELDS:
                array = getattr(table, name)
                key = f"{prefix}_{name}"
                if key not in state:  # a checkpoint without anchors: none is set
                    array.zero_()
                elif n:
                    values = np.asarray(state[key])  # (n, entries per body, ...)
                    flat = values.reshape(n * rows, *values.shape[2:])
                    wp.copy(array, wp.array(flat, dtype=array.dtype, device=self._device),
                            count=n * rows)  # fmt: skip
        self._count = n
        self._max_bound_m = float(np.max(state["bounds_m"])) if n else 0.0
        self._diagnostics.update(bodies=n, contacts=0, pair_overflows=0)
        self._contact_count.zero_()
        self._static_contacts.zero_()
        self._overflows.zero_()
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
            rows["bounds_m"].append(_bound(body))
            rows["static_friction"].append(body.static_friction)
            rows["dynamic_friction"].append(body.dynamic_friction)
            rows["restitution"].append(body.restitution)
        start, count = self._count, len(bodies)
        for name, array in (*self._state.items(), *self._properties.items()):
            values = np.asarray(rows[name], dtype=np.int32 if name == "shapes" else np.float32)
            source = wp.array(values, dtype=array.dtype, device=self._device)
            wp.copy(array, source, dest_offset=start, count=count)
        for _, table, per_body in self._anchor_tables():  # none set for the new bodies
            for name, _ in _ANCHOR_FIELDS:
                getattr(table, name)[start * per_body : (start + count) * per_body].zero_()
        self._count += count
        self._max_bound_m = max(self._max_bound_m, *rows["bounds_m"])
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

    def _anchor_tables(self) -> tuple[tuple[str, Any, int], ...]:
        """The friction anchors' tables, with their checkpoint names and entries per body:
        the pairs' of the last substep, and the static solids'."""
        return (
            ("pair_anchor", self._pair_anchors, PAIRS_PER_BODY),
            ("static_anchor", self._static_anchors, _k.STATIC_PATCHES),
        )

    def _anchor_state(self, tables: list[Any], n: int) -> dict[str, np.ndarray]:
        """The first `n` bodies' rows of the anchor `tables` (as `_anchor_tables` orders
        them), one array per field and table, shaped (n, entries per body, ...)."""
        state: dict[str, np.ndarray] = {}
        for (prefix, _, rows), table in zip(self._anchor_tables(), tables, strict=True):
            for name, _ in _ANCHOR_FIELDS:
                values = getattr(table, name).numpy()[: n * rows]
                state[f"{prefix}_{name}"] = values.reshape(n, rows, *values.shape[1:]).copy()
        return state

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

    def _substep(self, h: float, gravity: wp.vec3) -> None:
        """Predict; solve each pair of bodies by its own thread, the pairs in parallel, then
        each body's contacts with the static solids by its own thread (ADR 0031); recover
        the velocities; then the velocity pass, the static patches first."""
        count, device = self._count, self._device
        setup = self._require_setup()
        g = math.sqrt(sum(c * c for c in setup.gravity_m_per_s2))
        slow = 2.0 * g * h
        state = [self._state[name] for name, _ in _STATE]
        positions, orientations = state[0], state[1]
        velocities = state[2:]
        increments = [self._displacement, self._turn, self._rotated_rad]
        properties = self._properties
        masses, inertia = properties["masses_kg"], properties["inertia_kg_m2"]
        before = [self._velocity_before_m_per_s, self._angular_before_rad_per_s]
        wp.copy(before[0], velocities[0], count=count)
        wp.copy(before[1], velocities[1], count=count)
        wp.launch(_k.predict, dim=count, inputs=[*state, inertia, gravity, h, *increments],
                  device=device)  # fmt: skip
        patches = self._patches
        self._contact_count.zero_()
        self._static_contacts.zero_()
        patches.count.zero_()
        patches.per_body.zero_()
        self._grid.build(positions[:count], max(2.0 * self._max_bound_m, 1e-3))
        shapes, sizes = properties["shapes"], properties["sizes_m"]
        self._pair_anchors, self._spare_anchors = self._spare_anchors, self._pair_anchors
        wp.launch(
            _k.detect,
            dim=count,
            inputs=[self._grid.id, positions, orientations, shapes, sizes, properties["bounds_m"],
                    self._max_bound_m, self._signs, self._overflows, patches,
                    self._spare_anchors, self._pair_anchors, PAIRS_PER_BODY],
            device=device,
        )  # fmt: skip
        corrections = [self._shift_m, self._spin_rad]
        wp.launch(
            _k.solve_pairs,
            dim=patches.a.shape[0],
            inputs=[patches, positions, orientations, shapes, sizes, masses, inertia, self._signs,
                    self._displacement, self._turn, properties["static_friction"],
                    properties["dynamic_friction"], gravity, self._contact_count, *corrections,
                    self._pair_anchors],
            device=device,
        )  # fmt: skip
        wp.launch(_k.apply_corrections, dim=count,
                  inputs=[*corrections, positions, orientations, *increments, masses, inertia, h,
                          self._kinetic_j],
                  device=device)  # fmt: skip
        static = [positions, orientations, *increments, properties["shapes"], properties["sizes_m"],
                  properties["bounds_m"], masses, inertia, self._signs, *self._solids.arrays(),
                  self._static_contacts, *self._static_patches, self._static_touch,
                  properties["static_friction"], setup.static_friction,
                  properties["dynamic_friction"], setup.dynamic_friction,
                  self._static_anchors]  # fmt: skip
        wp.launch(_k.solve_static, dim=count, inputs=static, device=device)
        wp.launch(_k.recover, dim=count,
                  inputs=[self._displacement, self._rotated_rad, h, *velocities],
                  device=device)  # fmt: skip
        wp.launch(_k.solve_static_velocities, dim=count,
                  inputs=[positions, orientations, *velocities, *before, masses, inertia,
                          *self._static_patches, properties["dynamic_friction"],
                          properties["restitution"], setup.dynamic_friction, setup.restitution,
                          slow, h, self._kinetic_j],
                  device=device)  # fmt: skip
        kicks = [self._kick_m_per_s, self._twist_rad_per_s]
        self._bouncing.zero_()
        wp.launch(_k.count_bouncing, dim=patches.a.shape[0],
                  inputs=[patches, positions, *before, properties["restitution"], slow,
                          self._bouncing],
                  device=device)  # fmt: skip
        wp.launch(_k.solve_velocities, dim=patches.a.shape[0],
                  inputs=[patches, positions, orientations, *velocities, *before, masses, inertia,
                          properties["dynamic_friction"], properties["restitution"], slow, h,
                          *kicks, self._bouncing],
                  device=device)  # fmt: skip
        wp.launch(_k.apply_velocity_corrections, dim=count, inputs=[*kicks, *velocities],
                  device=device)  # fmt: skip

    def _send_status(self) -> None:
        self._status.zero_()
        wp.launch(
            _k.contact_status,
            dim=1,
            inputs=[self._contact_count, self._static_contacts, self._overflows, self._status],
            device=self._device,
        )
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
            raise NonFiniteStateError(self._frozen)
        if not self._status_pending:
            return
        if self._status_event is not None:
            wp.synchronize_event(self._status_event)
        self._status_pending = False
        status = self._status_host.numpy()
        self._diagnostics["contacts"] = int(status[1])
        self._diagnostics["pair_overflows"] = int(status[2])
        if float(status[0]) > 0.0:
            self._frozen = self._dump()
            raise NonFiniteStateError(self._frozen)

    def _dump(self) -> Path:
        setup = self._require_setup()
        folder = setup.dump_dir or out_dir() / "nan_dumps"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"rigid-{time.strftime('%Y%m%d-%H%M%S')}-{time.perf_counter_ns()}.npz"
        state = self.checkpoint()
        n = self._before_count
        before = {f"before_{name}": self._before[name].numpy()[:n].copy() for name, _ in _STATE}
        anchors = self._anchor_state(list(self._anchors_before), n)
        before |= {f"before_{key}": value for key, value in anchors.items()}
        arrays = {key: np.asarray(value) for key, value in state.items()}
        arrays["gravity_m_per_s2"] = np.asarray(setup.gravity_m_per_s2)
        for field in fields(setup):
            value = getattr(setup, field.name)
            if field.name not in ("dump_dir", "solids") and value is not None:
                arrays[f"setup_{field.name}"] = np.asarray(value)
        for field in fields(self._solids):
            arrays[f"solids_{field.name}"] = getattr(self._solids, field.name).numpy()
        np.savez_compressed(path, dt_s=np.asarray(self._last_dt_s), **arrays, **before)  # pyright: ignore[reportArgumentType]
        return path
