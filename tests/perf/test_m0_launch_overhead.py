"""M0 measurement: Warp kernel launch overhead, with and without CUDA graphs (issue #7).

Risk R3 (Python launch overhead) and ADR 0002's falsification criterion:
orchestration must stay under 2 ms per frame once each substep is a CUDA graph.

A tiny kernel (256 threads) is launched three ways:

- `wp.launch` from Python, the plain way;
- a recorded launch command (`record_cmd=True`), which skips argument packing;
- captured as CUDA graphs of GRAPH_SIZES kernels, replayed with `wp.capture_launch`.

Host cost is wall time on the Python thread to enqueue, starting from an idle
GPU; GPU cost is stream time between CUDA events divided by the number of
kernels, which for tiny kernels is the per-kernel overhead on the GPU side.
A probe showed the host cost of one graph launch is not linear in its size,
hence several sizes.
"""

import time
from collections.abc import Callable

import warp as wp

from bench.harness import BenchSession, Summary, summarize, time_device

DEVICE = "cuda:0"
N_LAUNCHES = 1000
GRAPH_SIZES = (10, 100, 1000)
REPEATS = 30
ORCHESTRATION_BUDGET_S = 2e-3  # SPEC 3, game logic, UI and Python orchestration
US = 1e6


@wp.kernel
def _tiny(a: wp.array(dtype=float)):
    i = wp.tid()
    a[i] = a[i] + 1.0


def _host(enqueue: Callable[[], object], count: int) -> Summary:
    """Wall time per call to enqueue `count` calls, from an idle GPU each repeat."""
    samples = []
    for _ in range(REPEATS):
        wp.synchronize_device(DEVICE)
        begin = time.perf_counter()
        for _ in range(count):
            enqueue()
        samples.append((time.perf_counter() - begin) / count)
    wp.synchronize_device(DEVICE)
    return summarize(samples)


def _repeat(enqueue: Callable[[], object], count: int) -> Callable[[], None]:
    def run() -> None:
        for _ in range(count):
            enqueue()

    return run


def test_kernel_launch_overhead(bench_session: BenchSession) -> None:
    a = wp.zeros(256, dtype=float, device=DEVICE)

    def launch() -> None:
        wp.launch(_tiny, dim=256, inputs=[a], device=DEVICE)

    command = wp.launch(_tiny, dim=256, inputs=[a], device=DEVICE, record_cmd=True)
    assert command is not None
    launch()
    graphs: dict[int, wp.Graph] = {}
    for size in GRAPH_SIZES:
        with wp.ScopedCapture(device=DEVICE) as capture:
            _repeat(launch, size)()
        assert capture.graph is not None
        graphs[size] = capture.graph

    bench_session.start("launch_overhead")
    host_launch = _host(launch, N_LAUNCHES)
    host_command = _host(command.launch, N_LAUNCHES)
    gpu_launch = summarize(time_device(_repeat(launch, N_LAUNCHES), repeats=REPEATS, device=DEVICE))
    gpu_command = summarize(
        time_device(_repeat(command.launch, N_LAUNCHES), repeats=REPEATS, device=DEVICE)
    )
    metrics: dict[str, object] = {
        "plain_launch": {
            "host_us_per_kernel": host_launch.median_s * US,
            "host_p10_p90_us": [host_launch.p10_s * US, host_launch.p90_s * US],
            "gpu_us_per_kernel": gpu_launch.median_s / N_LAUNCHES * US,
        },
        "recorded_command": {
            "host_us_per_kernel": host_command.median_s * US,
            "host_p10_p90_us": [host_command.p10_s * US, host_command.p90_s * US],
            "gpu_us_per_kernel": gpu_command.median_s / N_LAUNCHES * US,
        },
        "kernels_in_budget_plain": ORCHESTRATION_BUDGET_S / host_launch.median_s,
    }
    for size, graph in graphs.items():
        host = _host(lambda graph=graph: wp.capture_launch(graph), 1)
        gpu = summarize(
            time_device(
                lambda graph=graph: wp.capture_launch(graph), repeats=REPEATS, device=DEVICE
            )
        )
        metrics[f"graph_{size}"] = {
            "host_us_per_graph": host.median_s * US,
            "host_p10_p90_us": [host.p10_s * US, host.p90_s * US],
            "host_us_per_kernel": host.median_s / size * US,
            "gpu_us_per_kernel": gpu.median_s / size * US,
            "kernels_in_budget": ORCHESTRATION_BUDGET_S / (host.median_s / size),
        }
    path = bench_session.record("launch_overhead", metrics)
    print(f"\nlaunch overhead -> {path}")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # R3 mitigation: graphs must cut the host cost per kernel several-fold.
    best_graph = min(
        metrics[f"graph_{size}"]["host_us_per_kernel"]  # type: ignore[index]
        for size in GRAPH_SIZES
    )
    assert best_graph * 5 < host_launch.median_s * US
