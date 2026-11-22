"""M1 measurement: a solver's iteration loop as a CUDA graph conditional node (#36).

ADR 0016's pressure solve replays fixed batches of 4 iterations as graphs and reads the
residual back after each batch, about 4 synchronisations per solve. A conditional node
(`wp.capture_while`) keeps the loop on the GPU: one graph launch, no read-back until the
end. This compares, for the same 40 iterations of a two-kernel body (a Jacobi sweep on a
96³ grid and a kernel that counts down the loop's condition, as a convergence test
would):

- `batched`: a graph of 4 iterations, launched 10 times, the condition read after each;
- `conditional`: one graph whose loop is a conditional node;
- `unrolled`: one graph of the 40 iterations, no condition at all (the floor).

Host time is the CPU's wall time from the first launch until the result is known on the
host; enqueue time is the CPU time spent launching, which the batched loop cannot
separate from its waits (each batch's read-back blocks it). GPU time is CUDA events
around the work. Both are timed round by round, interleaved
(ADR 0022).
"""

import time

import numpy as np
import warp as wp

from bench.harness import BenchSession, summarize

DEVICE = "cuda:0"
SHAPE = (96, 96, 96)
ITERATIONS = 40
BATCH = 4
ROUNDS = 50


@wp.kernel
def _sweep(
    src: wp.array3d(dtype=float), rhs: wp.array3d(dtype=float), dst: wp.array3d(dtype=float)
):
    i, j, k = wp.tid()  # pyright: ignore[reportGeneralTypeIssues, reportAssignmentType]
    nx, ny, nz = src.shape[0], src.shape[1], src.shape[2]
    if i == 0 or j == 0 or k == 0 or i == nx - 1 or j == ny - 1 or k == nz - 1:
        dst[i, j, k] = 0.0
        return
    total = src[i - 1, j, k] + src[i + 1, j, k] + src[i, j - 1, k]
    total += src[i, j + 1, k] + src[i, j, k - 1] + src[i, j, k + 1]
    dst[i, j, k] = (total + rhs[i, j, k]) / 6.0


@wp.kernel
def _count_down(left: wp.array(dtype=wp.int32)):
    left[0] = left[0] - 1


@wp.kernel
def _reset(left: wp.array(dtype=wp.int32), value: int):
    left[0] = value


def test_a_solver_loop_as_a_conditional_graph_node(bench_session: BenchSession) -> None:
    assert wp.is_conditional_graph_supported(), "CUDA 12.4+ is needed for conditional nodes"
    a = wp.zeros(SHAPE, dtype=float, device=DEVICE)
    b = wp.zeros(SHAPE, dtype=float, device=DEVICE)
    rhs = wp.array(np.random.default_rng(0).random(SHAPE, dtype=np.float32), device=DEVICE)
    left = wp.zeros(1, dtype=wp.int32, device=DEVICE)
    left_host = wp.zeros(1, dtype=wp.int32, device="cpu", pinned=True)

    def iteration() -> None:
        wp.launch(_sweep, dim=SHAPE, inputs=[a, rhs, b], device=DEVICE)
        wp.launch(_sweep, dim=SHAPE, inputs=[b, rhs, a], device=DEVICE)
        wp.launch(_count_down, dim=1, inputs=[left], device=DEVICE)

    def capture(body) -> wp.Graph:
        with wp.ScopedCapture(device=DEVICE) as capture:
            body()
        assert capture.graph is not None
        return capture.graph

    batch = capture(lambda: [iteration() for _ in range(BATCH)])
    unrolled = capture(lambda: [iteration() for _ in range(ITERATIONS)])
    conditional = capture(lambda: wp.capture_while(left, iteration))

    def run_batched() -> None:
        wp.launch(_reset, dim=1, inputs=[left, ITERATIONS], device=DEVICE)
        while True:
            wp.capture_launch(batch)
            wp.copy(left_host, left)
            wp.synchronize_device(DEVICE)  # the read-back the batches need
            if int(left_host.numpy()[0]) <= 0:
                return

    enqueue_s: dict[str, list[float]] = {"conditional": [], "unrolled": []}

    def run_graph(graph: wp.Graph, key: str):
        def run() -> None:
            t0 = time.perf_counter()
            wp.launch(_reset, dim=1, inputs=[left, ITERATIONS], device=DEVICE)
            wp.capture_launch(graph)
            wp.copy(left_host, left)
            enqueue_s[key].append(time.perf_counter() - t0)
            wp.synchronize_device(DEVICE)

        return run

    runs = {
        "batched": run_batched,
        "conditional": run_graph(conditional, "conditional"),
        "unrolled": run_graph(unrolled, "unrolled"),
    }
    for run in runs.values():
        for _ in range(3):
            run()
    for samples in enqueue_s.values():
        samples.clear()
    host_s: dict[str, list[float]] = {name: [] for name in runs}
    gpu_s: dict[str, list[float]] = {name: [] for name in runs}
    stream = wp.get_stream(DEVICE)
    name = "conditional_graph_" + "x".join(map(str, SHAPE))
    bench_session.start(name)
    for _ in range(ROUNDS):
        for key, run in runs.items():
            start, end = wp.Event(DEVICE, enable_timing=True), wp.Event(DEVICE, enable_timing=True)
            stream.record_event(start)
            t0 = time.perf_counter()
            run()
            host_s[key].append(time.perf_counter() - t0)
            stream.record_event(end)
            wp.synchronize_event(end)
            gpu_s[key].append(wp.get_event_elapsed_time(start, end) / 1e3)
            assert int(left_host.numpy()[0]) == 0, f"{key} did not run its {ITERATIONS}"

    host = {k: summarize(v) for k, v in host_s.items()}
    gpu = {k: summarize(v) for k, v in gpu_s.items()}
    kernels = 3 * ITERATIONS
    metrics = {
        "shape": list(SHAPE),
        "iterations": ITERATIONS,
        "batch": BATCH,
        "kernels_per_solve": kernels,
        "host": {k: s.as_ms() for k, s in host.items()},
        "gpu": {k: s.as_ms() for k, s in gpu.items()},
        "enqueue": {k: summarize(v).as_ms() for k, v in enqueue_s.items()},
        "conditional_overhead_per_iteration_us": (
            (gpu["conditional"].median_s - gpu["unrolled"].median_s) / ITERATIONS * 1e6
        ),
        "read_backs_saved": ITERATIONS // BATCH - 1,
    }
    path = bench_session.record(name, metrics)
    print(
        f"\n{name}, {ITERATIONS} iterations: host batched {host['batched'].median_s * 1e3:.3f} ms, "
        f"conditional {host['conditional'].median_s * 1e3:.3f} ms, unrolled "
        f"{host['unrolled'].median_s * 1e3:.3f} ms; GPU batched {gpu['batched'].median_s * 1e3:.3f}"
        f" ms, conditional {gpu['conditional'].median_s * 1e3:.3f} ms, unrolled "
        f"{gpu['unrolled'].median_s * 1e3:.3f} ms; conditional overhead "
        f"{metrics['conditional_overhead_per_iteration_us']:.2f} µs per iteration; enqueue "
        f"conditional {summarize(enqueue_s['conditional']).median_s * 1e6:.0f} µs -> {path}"
    )
    assert host["conditional"].median_s > 0.0
