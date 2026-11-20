# ADR 0007: How performance is measured

- Status: accepted
- Date: 2026-11-20

## Context

SPEC section 3 sets every budget for the laptop "plugged in, in performance power mode",
"after a 5-minute warm-up, so thermal throttling is included in the numbers". M0 has to
turn that sentence into a harness whose results can be trusted by later sessions that
did not watch the run.

## Decision

- Benchmarks are pytest tests under `tests/perf/`, all marked `perf` by their conftest.
  They run with `uv run pytest -m perf` on the laptop only, never in CI.
- **Official conditions**, checked by `bench/harness.py` and written into every result:
  - on mains power (`GetSystemPowerStatus`);
  - energy saver off;
  - performance power mode: the Windows 11 slider on *Best performance* under the
    Balanced plan, or a *High performance* / *Ultimate performance* plan, which Windows
    offers instead of the slider on some machines. Any other plan is not performance,
    and the slider value is ignored outside the Balanced plan;
  - at least 300 s of GPU warm-up with a memory-streaming load, after every kernel module
    of the session is compiled.
- The power state is read again right before and right after each benchmark; a change
  can only make a result unofficial.
- Official results go to `docs/benchmarks/results/` and are committed; others go to
  `out/benchmarks/`, ignored by git. Each file holds metrics, conditions, `nvidia-smi`
  telemetry before and after (clocks, temperature, power, enforced power limit, clock
  event reasons) and versions, including `git describe --dirty`.
- GPU durations come from CUDA events recorded on the device stream around the measured
  work; costs on the Python side use wall time. Results are medians with the 10th and
  90th percentiles over repeated runs.
- The Dell thermal profile cannot be read from Windows; the enforced GPU power limit in
  the telemetry shows its effect, and benchmark reports state the profile Matt set.

## Alternatives considered

- **pytest-benchmark:** good statistics, but wall time only, no warm-up or power checks,
  and its own storage format. Our needs are GPU-side timing and conditions, which it does
  not cover.
- **Refusing to run outside official conditions:** slows down writing benchmarks;
  marking results unofficial and keeping them out of `docs/` gives the same guarantee.

## Consequences

- A benchmark result in `docs/benchmarks/results/` can be trusted without knowing who ran
  it or how.
- A session takes at least 5 minutes; `ENTROPY_WARMUP_S` shortens it for unofficial trials.

## How we would know it is wrong

Two official runs of the same benchmark on the same commit differ by more than their
10–90 % spread, or the telemetry of an official run shows the GPU below its sustained
clocks without a thermal or power reason.
