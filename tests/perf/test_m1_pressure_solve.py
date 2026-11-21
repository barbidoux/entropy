"""M1 measurement: one MGPCG pressure solve (issue #46, SPEC 3 liquid row).

SPEC 3 estimates the pressure solve on the full house grid (160 x 128 x 96 at
6.25 cm) at 5.0-8.7 ms per substep with constant coefficients, from the M0 sweep
cost and a model of 12 iterations of about 12 sweep equivalents. This measures
the real solver on the worst case of that row: the whole grid fluid under one
layer of air, solved from zero to a relative residual of 1e-5, for a random
right-hand side. The 72 x 72 x 40 grid is the Test Tank of M1 (4 x 4 x 2 m at
6.25 cm, with a margin).

The solve's own read-backs (one every `check_every` iterations) are inside the
measured span, as they would be in a frame. The first solve, which compiles and
captures the CUDA graph, is excluded.
"""

import numpy as np
import pytest
import warp as wp

from bench.harness import BenchSession, summarize, time_device
from entropy.linalg.mgpcg import CELL_AIR, CELL_FLUID, Mgpcg, SolveResult, system_from_labels

DEVICE = "cuda:0"
DX_M = 0.0625
ESTIMATE_MS = (5.0, 8.7)  # SPEC 3, house grid, constant coefficients, per substep


@pytest.mark.parametrize(
    "shape", [(160, 128, 96), (128, 128, 128), (72, 72, 40)], ids=["house", "128", "test-tank"]
)
def test_pressure_solve_cost(bench_session: BenchSession, shape: tuple[int, int, int]) -> None:
    labels = np.full(shape, CELL_FLUID, dtype=np.int32)
    labels[:, :, -1] = CELL_AIR
    system = system_from_labels(
        wp.array(labels, dtype=wp.int32, device=DEVICE), dx_m=DX_M, air_theta=0.5
    )
    solver = Mgpcg(system)
    rhs = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    b = wp.array(rhs, dtype=float, device=DEVICE)
    x = wp.zeros(shape, dtype=float, device=DEVICE)
    results: list[SolveResult] = []

    def solve() -> None:
        x.zero_()
        results.append(solver.solve(b, x, tolerance=1e-5))

    solve()  # compile, capture the graph
    name = "pressure_solve_" + "x".join(map(str, shape))
    bench_session.start(name)
    timed = summarize(time_device(solve, repeats=20, device=DEVICE))

    iterations = results[-1].iterations
    metrics = {
        "shape": list(shape),
        "cells": int(np.prod(shape)),
        "levels": solver.levels,
        "iterations": iterations,
        "relative_residual": results[-1].relative_residual,
        "solve": timed.as_ms(),
        "per_iteration_ms": timed.median_s / max(iterations, 1) * 1e3,
        "spec3_estimate_ms": list(ESTIMATE_MS),
    }
    path = bench_session.record(name, metrics)
    print(
        f"\n{name}: {timed.median_s * 1e3:.2f} ms per solve, {iterations} iterations, "
        f"{metrics['per_iteration_ms']:.3f} ms each, {solver.levels} levels "
        f"(SPEC 3 estimate {ESTIMATE_MS[0]}-{ESTIMATE_MS[1]} ms) -> {path}"
    )
    assert all(r.converged for r in results)
