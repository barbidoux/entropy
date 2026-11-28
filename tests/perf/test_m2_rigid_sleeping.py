"""M2 measurement: what sleeping stacks cost against awake ones (#89, ADR 0033).

1,000 boxes of 20 cm in 100 stacks of 10 on the floor (mu = 0.5) settle until all sleep;
the frame's GPU time (CUDA events around `step`, 1/30 s frames, 16 substeps) is then
measured over 60 frames with sleeping on, and with it off after 5 s to settle. ADR 0033
asks the sleeping stacks' frame to cost at most half the awake ones'. A random heap of
1,000 boxes keeps a few of them moving and its main island awake (a named limit, ADR
0033), so it is not the scene. SPEC 3's budget for rigid bodies is #90's.
"""

import numpy as np
import warp as wp

from bench.harness import BenchSession
from entropy.solvers.rigid import Body, RigidSetup, RigidSolver, Shape
from entropy.world.sdf import HalfSpace, Solids

DEVICE = "cuda:0"
BODIES = 5000
FRAME_S = 1.0 / 30.0
MU = 0.5


def _stacks(sleeping: bool) -> RigidSolver:
    floor = Solids((HalfSpace((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),)).to_device(DEVICE)
    solver = RigidSolver.create(
        RigidSetup(gravity_m_per_s2=(0.0, 0.0, -9.81), max_bodies=BODIES, solids=floor,
                   sleeping=sleeping, static_friction=MU, dynamic_friction=MU),
        DEVICE,
    )  # fmt: skip
    bodies = []
    for k in range(BODIES):
        column, level = divmod(k, 10)
        x, y = 0.4 * (column % 25 - 12.0), 0.4 * (column // 25 - 9.5)
        bodies.append(Body(Shape.BOX, (0.1, 0.1, 0.1), 600.0, position_m=(x, y, 0.1 + 0.2 * level),
                           static_friction=MU, dynamic_friction=MU))  # fmt: skip
    solver.add_bodies(bodies)
    return solver


def _frame_ms(solver: RigidSolver, frames: int) -> list[float]:
    times = []
    for _ in range(frames):
        start, end = wp.Event(DEVICE, enable_timing=True), wp.Event(DEVICE, enable_timing=True)
        wp.record_event(start)
        solver.step(FRAME_S)
        wp.record_event(end)
        wp.synchronize_event(end)
        times.append(wp.get_event_elapsed_time(start, end))
    return times


def test_sleeping_stacks_cost_at_most_half_awake_ones(bench_session: BenchSession) -> None:
    asleep = _stacks(sleeping=True)
    for _ in range(round(20.0 / FRAME_S)):
        asleep.step(FRAME_S)
        if asleep.diagnostics()["sleeping"] == BODIES:
            break
    asleep.step(FRAME_S)
    assert asleep.diagnostics()["sleeping"] == BODIES
    awake = _stacks(sleeping=False)
    for _ in range(round(5.0 / FRAME_S)):
        awake.step(FRAME_S)

    bench_session.start("rigid_sleeping_1000")
    asleep_ms = float(np.median(_frame_ms(asleep, 60)))
    awake_ms = float(np.median(_frame_ms(awake, 60)))
    path = bench_session.record(
        "rigid_sleeping_1000",
        {"bodies": BODIES, "frame_s": FRAME_S, "asleep_median_ms": asleep_ms,
         "awake_median_ms": awake_ms, "ratio": asleep_ms / awake_ms},
    )  # fmt: skip
    print(f"\n1,000 boxes: asleep {asleep_ms:.2f} ms, awake {awake_ms:.2f} ms per frame -> {path}")
    assert asleep_ms <= 0.5 * awake_ms
