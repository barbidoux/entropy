# Benchmarks

Performance is measured, never assumed (SPEC section 3). Benchmarks are pytest tests
marked `perf` under `tests/perf/`; they run only on the laptop.

## Running

On the laptop, plugged in, Windows power mode set to **Best performance** (Settings >
System > Power & battery > Power mode), nothing heavy running:

```powershell
uv run pytest -m perf -s
```

The first benchmark of the session warms the GPU up for 5 minutes. For a quick trial
while developing a benchmark, shorten it:

```powershell
$env:ENTROPY_WARMUP_S = "20"; uv run pytest -m perf -s
```

## Official and unofficial results

Each benchmark writes one JSON file with its metrics, the session conditions, GPU
telemetry (clocks, temperature, power, throttle reasons) before and after, and the
software versions (`bench/harness.py`).

- **Official**: on mains power, in a performance power mode, after at least 300 s of
  warm-up. Written to `docs/benchmarks/results/` and committed.
- **Unofficial**: anything else. Written to `out/benchmarks/`, ignored by git, and the
  reasons are listed in the file.

GPU times come from CUDA events recorded around the measured work; CPU-side costs
(kernel launches from Python) use wall-clock time. Each benchmark repeats its
measurement and reports the median with the 10th and 90th percentiles.

## Reports

Dated reports in this folder summarise a session and say what it changes in the
budgets of SPEC section 3.
