# Changelog

All notable changes to ENTROPY are recorded here, newest first.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Benchmark harness (#4): `bench/harness.py` and the `-m perf` session fixture. Checks
  mains power, energy saver and the Windows power mode around every benchmark, warms
  the GPU up for 5 minutes, times work with CUDA events, and writes JSON results with
  GPU telemetry. Only runs meeting the SPEC 3 conditions are official and go to
  `docs/benchmarks/results/`. ADR 0007 records the method.
  mains power and the Windows power mode, warms the GPU up for 5 minutes, times work
  with CUDA events, and writes JSON results with GPU telemetry. Only runs meeting the
  SPEC 3 conditions are official and go to `docs/benchmarks/results/`.
- Test harness (#3): `device` fixture over CPU and, with `-m gpu`, CUDA; a test
  enforcing the module dependency rules of SPEC 8.1, with `app` as the composition
  root; coverage measured in CI with a 90 % floor, Warp kernels excluded; ruff enforces
  `import warp as wp`.
- ADR 0005: transitive layering and the `app` composition root. SPEC 8.1 gains the
  `app` and `audio` rows.
- ADR 0006: `docs/SPEC.md` is the reference copy of the specification, as Matt decided
  when M0 started; `CLAUDE.md` and the SPEC header say so.
- Specification v0.1 (`docs/SPEC.md`), mirrored from the living document.
- Walking skeleton: `entropy` package, Warp toolchain tests (CPU float atomics,
  bit-reproducible CPU sums, CUDA availability on the laptop).
- Tooling: uv, pytest with `gpu`, `perf`, `soak`, `slow` and `render` markers, ruff,
  pyright, pre-commit, GitHub Actions CPU CI.
- ADR 0001 (record decisions), 0002 (Python and Warp), 0003 (SI units, Z-up frame).
- Maintainer guide (`CLAUDE.md`).
