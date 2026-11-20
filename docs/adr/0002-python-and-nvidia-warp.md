# ADR 0002: Python 3.12 with NVIDIA Warp kernels

- Status: accepted
- Date: 2026-11-20

## Context

The engine needs GPU compute for coupled fluid, gas, rigid and granular solvers on an
NVIDIA RTX 4090 Laptop GPU. Development follows strict TDD, and the maintainer's
workspace has no GPU, so the test loop must run on CPU.

## Decision

Python 3.12 orchestrates; physics runs as NVIDIA Warp kernels, written in Python and
JIT-compiled for CPU or CUDA.

Evidence gathered on 2026-11-20 in the maintainer's workspace (Warp 1.19.0, no GPU):

- The same kernels compile and run on the CPU backend.
- Float atomic additions, needed for particle-to-grid transfers, work on CPU.
- CPU atomic sums are bit-for-bit reproducible across runs (now a test in
  `tests/test_toolchain.py`).

## Alternatives considered

- **C++20, CUDA and Vulkan:** maximum control and speed, but GPU code cannot run in the
  GPU-less workspace, so every red-green cycle would wait for the laptop. Much more code.
- **Rust and wgpu (WGSL):** safe, portable to every GPU vendor, but WGSL has no float
  atomic addition, which the particle-to-grid transfers depend on; the workarounds
  (fixed-point atomics or sorting) cost speed and complexity.

## Consequences

- NVIDIA GPUs only. Accepted: the target machine is fixed (SPEC section 3).
- Python 3.12 is used because every planned dependency ships Windows wheels for it.
- Pyright's `reportInvalidTypeForm` is disabled, because Warp's kernel signatures
  (`wp.array(dtype=float)`) are calls in type positions.

## How we would know it is wrong

M0 benchmarks show Python and kernel-launch overhead above 2 ms per frame after each
substep is captured as a CUDA graph (SPEC section 3, frame budget).
