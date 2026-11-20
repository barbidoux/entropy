"""M0 measurement: effective GPU memory bandwidth (issue #5, SPEC 3, risk R1).

Effective bandwidth = bytes moved / GPU time, with 1 GB = 10⁹ bytes. A copy
moves 2 x BUFFER_BYTES (one read, one write per byte); a read-only pass moves
BUFFER_BYTES. Buffers are 8 times the 64 MiB L2 cache of the AD103 chip, so
the cache cannot inflate the result and it measures device memory (DRAM).
"""

import warp as wp

from bench.harness import BenchSession, dram_peak_gb_per_s, summarize, time_device

DEVICE = "cuda:0"
BUFFER_BYTES = 512 * 2**20
PEAK_GB_PER_S = 576.0
"""RTX 4090 Laptop GPU: 256-bit GDDR6 at 18 Gb/s per pin (NVIDIA specification)."""
BUS_WIDTH_BITS = 256
REPEATS = 50


@wp.kernel
def _copy_float(src: wp.array(dtype=float), dst: wp.array(dtype=float)):
    i = wp.tid()
    dst[i] = src[i]


@wp.kernel
def _copy_vec4(src: wp.array(dtype=wp.vec4), dst: wp.array(dtype=wp.vec4)):
    i = wp.tid()
    dst[i] = src[i]


@wp.kernel
def _read_vec4(src: wp.array(dtype=wp.vec4), sink: wp.array(dtype=float)):
    i = wp.tid()
    v = src[i]
    # Never true for the input of ones, but the compiler cannot know: every load stays.
    if v[0] + v[1] + v[2] + v[3] == -1.0:
        sink[0] = v[0]


def _gb_per_s(bytes_moved: int, seconds: float) -> float:
    return bytes_moved / seconds / 1e9


def test_memory_bandwidth(bench_session: BenchSession) -> None:
    floats = BUFFER_BYTES // 4
    src = wp.zeros(floats, dtype=float, device=DEVICE)
    dst = wp.zeros(floats, dtype=float, device=DEVICE)
    # Non-zero data, so no compression of zero pages can flatter the read pass.
    src4 = wp.full(floats // 4, wp.vec4(1.0, 1.0, 1.0, 1.0), dtype=wp.vec4, device=DEVICE)
    dst4 = wp.zeros(floats // 4, dtype=wp.vec4, device=DEVICE)
    sink = wp.zeros(1, dtype=float, device=DEVICE)

    cases = {
        "copy_float": (2 * BUFFER_BYTES, lambda: wp.launch(_copy_float, floats, [src, dst])),
        "copy_vec4": (2 * BUFFER_BYTES, lambda: wp.launch(_copy_vec4, floats // 4, [src4, dst4])),
        "memcpy": (2 * BUFFER_BYTES, lambda: wp.copy(dst, src)),
        "read_vec4": (BUFFER_BYTES, lambda: wp.launch(_read_vec4, floats // 4, [src4, sink])),
    }

    telemetry_before = bench_session.telemetry()
    bench_session.start("bandwidth")
    metrics: dict[str, object] = {
        "buffer_mib": BUFFER_BYTES / 2**20,
        "peak_gb_per_s": PEAK_GB_PER_S,
    }
    with wp.ScopedDevice(DEVICE):
        for name, (bytes_moved, run) in cases.items():
            summary = summarize(time_device(run, repeats=REPEATS, device=DEVICE))
            metrics[name] = {
                **summary.as_ms(),
                "gb_per_s": _gb_per_s(bytes_moved, summary.median_s),
                "fraction_of_peak": _gb_per_s(bytes_moved, summary.median_s) / PEAK_GB_PER_S,
            }
    peak_from_clock = dram_peak_gb_per_s(
        [telemetry_before, bench_session.telemetry()], bus_width_bits=BUS_WIDTH_BITS
    )
    if peak_from_clock is not None:
        metrics["peak_from_clock_gb_per_s"] = peak_from_clock
    # Under the power cap the memory clock can sit below the nominal 9001 MHz, so the
    # bound comes from the clock actually seen when nvidia-smi can read it.
    bound_gb_per_s = peak_from_clock or PEAK_GB_PER_S
    path = bench_session.record("bandwidth", metrics)
    print(f"\nbandwidth -> {path}")
    for name in cases:
        result = metrics[name]
        assert isinstance(result, dict)
        print(f"  {name:10s} {result['gb_per_s']:7.1f} GB/s ({result['fraction_of_peak']:.0%})")
        # Above the DRAM peak means the timing or the byte count is wrong.
        assert 0.0 < result["gb_per_s"] <= bound_gb_per_s * 1.05
