"""M0 measurement: cost of a Jacobi sweep of the 7-point Poisson stencil (issue #6).

SPEC 11 asks for pressure-solve costs at 128³ and 192³ (the gas box of
SPEC 3 and 5.3). A pressure solve is a number of stencil sweeps; this measures
one sweep of `entropy.linalg.jacobi`, three ways:

- single: one launch between two CUDA events;
- batched: BATCH launches back to back from Python, divided by BATCH;
- graph: the same BATCH launches captured once as a CUDA graph and replayed.

The 160 x 128 x 96 grid is the house at the 6.25 cm liquid cell (SPEC 3).
Ideal traffic is 12 bytes per cell (read p and rhs, write the result; the six
neighbours come from cache), which gives the effective bandwidth. At 128³ the three
arrays (24 MiB) fit in the 64 MiB L2 cache, so that figure can exceed the DRAM
peak. A streaming pass with the same traffic and no stencil (read p and rhs,
write once) is measured alongside as a reference for this kernel.
"""

from collections.abc import Callable

import pytest
import warp as wp

from bench.harness import BenchSession, summarize, time_interleaved
from entropy.linalg.jacobi import jacobi_step

DEVICE = "cuda:0"
H_M = 0.05  # gas cell size in m; does not change the cost
OMEGA = 6.0 / 7.0
BATCH = 100
GAS_BUDGET_S = 6e-3  # SPEC 3, gas, smoke and fire
BYTES_PER_CELL = 12


@wp.kernel
def _stream(p: wp.array3d(dtype=float), rhs: wp.array3d(dtype=float), out: wp.array3d(dtype=float)):
    i, j, k = wp.tid()  # pyright: ignore[reportGeneralTypeIssues, reportAssignmentType]
    out[i, j, k] = p[i, j, k] + rhs[i, j, k]


def _graph_of(count: int, enqueue: Callable[[], object]) -> wp.Graph:
    enqueue()  # load the module before capturing
    with wp.ScopedCapture(device=DEVICE) as capture:
        for _ in range(count):
            enqueue()
    assert capture.graph is not None
    return capture.graph


@pytest.mark.parametrize(
    "shape", [(128, 128, 128), (192, 192, 192), (160, 128, 96)], ids=["128", "192", "house"]
)
def test_jacobi_sweep_cost(bench_session: BenchSession, shape: tuple[int, int, int]) -> None:
    p = wp.zeros(shape, dtype=float, device=DEVICE)
    out = wp.zeros(shape, dtype=float, device=DEVICE)
    rhs = wp.full(shape, 1.0, dtype=float, device=DEVICE)

    def sweeps(count: int) -> None:
        for k in range(count):
            src, dst = (p, out) if k % 2 == 0 else (out, p)
            jacobi_step(src, rhs, dst, h_m=H_M, omega=OMEGA)

    graph = _graph_of(BATCH // 2, lambda: sweeps(2))
    streaming = _graph_of(BATCH, lambda: wp.launch(_stream, shape, [p, rhs, out], device=DEVICE))

    name = "stencil_" + "x".join(map(str, shape))
    bench_session.start(name)
    # Interleaved repeat by repeat, so the ratios between them hold while clocks drift (#25).
    timed = time_interleaved(
        {
            "single": lambda: sweeps(1),
            "batched": lambda: sweeps(BATCH),
            "graph": lambda: wp.capture_launch(graph),
            "streaming": lambda: wp.capture_launch(streaming),
        },
        rounds=20,
        device=DEVICE,
    )
    single, batched, graphed, streamed = (
        summarize(timed[key]) for key in ("single", "batched", "graph", "streaming")
    )

    cells = shape[0] * shape[1] * shape[2]
    per_sweep_s = graphed.median_s / BATCH
    metrics = {
        "shape": list(shape),
        "cells": cells,
        "single": single.as_ms(),
        "batched_per_sweep_ms": batched.median_s / BATCH * 1e3,
        "graph_per_sweep_ms": per_sweep_s * 1e3,
        "graph_p10_p90_per_sweep_ms": [graphed.p10_s / BATCH * 1e3, graphed.p90_s / BATCH * 1e3],
        "effective_gb_per_s": BYTES_PER_CELL * cells / per_sweep_s / 1e9,
        "ns_per_cell": per_sweep_s / cells * 1e9,
        "sweeps_in_gas_budget": GAS_BUDGET_S / per_sweep_s,
        "streaming_per_pass_ms": streamed.median_s / BATCH * 1e3,
        "streaming_gb_per_s": BYTES_PER_CELL * cells / (streamed.median_s / BATCH) / 1e9,
        "fraction_of_streaming_speed": (streamed.median_s / BATCH) / per_sweep_s,
    }
    path = bench_session.record(name, metrics)
    print(
        f"\n{name}: {per_sweep_s * 1e3:.3f} ms/sweep (graph), "
        f"{metrics['batched_per_sweep_ms']:.3f} batched, {single.median_s * 1e3:.3f} single, "
        f"{metrics['effective_gb_per_s']:.0f} GB/s, "
        f"{metrics['sweeps_in_gas_budget']:.0f} sweeps in 6 ms, "
        f"{metrics['fraction_of_streaming_speed']:.0%} of streaming -> {path}"
    )
    assert per_sweep_s > 0.0
