# ADR 0025: The liquid sorts its particles by half-cell each frame; particles-to-grid stays a scatter

- Status: accepted
- Date: 2026-11-22

## Context

At 1 M particles the APIC transfers took much of the liquid's 9 ms per frame (SPEC 3).
Issue #65 measured them in random order: about 3.5 ms per substep for particles to grid
and 0.9 ms for grid to particles, at 2 to 4 substeps a frame.

Particles-to-grid scatters 162 float atomics per particle: 3 components × 27 nodes ×
mass and momentum. Its cost depends on the order of the particles in memory:
- **Random order.** The atomics and the reads land all over the grid.
- **By cell.** The 8 particles of a cell sit in consecutive threads of a warp. They add
  to nearly the same 27 nodes at the same moment, and same-address atomics serialise.
- **By half-cell** (each cell split in two along each axis). Neighbouring threads touch
  neighbouring nodes, but rarely the same ones at once. In x-major order, consecutive
  threads also run along z, the grid's contiguous axis.
- **In the bucket's layers** (bottom first, then x, y fastest). Neighbouring threads
  share no nodes, but run along y, across the grid's memory.

The independent review of #75 measured this, interleaved over 30 rounds with the L2
cache flushed before each pass:

| Order of 1,048,576 particles | Particles to grid | Grid to particles |
| --- | --- | --- |
| Random | 3.55 ms | 0.90 ms |
| By cell | 2.38 ms | 0.171 ms |
| By half-cell | 0.979 ms | 0.156 ms |
| As the bucket and the dam pour it (layers, y or x fastest) | 3.38 ms | 0.169 ms |
| As `fill_box` lays it out (a lattice, already in half-cell order) | 0.977 ms | 0.157 ms |

Every tool adds water in an order the sort improves: the bucket and the dam in layers,
the flood source and the hose at random. Mixing during a flow drifts any order towards
random.

## Decision

- **Sort every frame by half-cell.** At the start of each frame the liquid sorts its
  particles by half-cell, x-major (`entropy.solvers.liquid.sorting.CellSort`,
  `SUBDIVISION = 2`).
  - The sort is a stable radix sort of the keys, so particles in one half-cell keep
    their order, and still water keeps its order frame after frame.
  - The particles are gathered into the second particle buffer, which then becomes the
    live one.
  - The buffer left behind holds the frame's starting state as it was, for the NaN
    guard's dump (ADR 0018). The sort replaces the copy the dump needed anyway.
  - Keys fit in int32: 1.66 M half-cells on the Test Tank's grid, 15.7 M on the house
    grid.
- **Keep the scatter.** Particles-to-grid stays a scatter of atomics. At about 1 ms per
  substep in half-cell order it is below #65's threshold of about 2 ms, so a
  shared-memory version is not needed.

The cost of the sort, per frame:
- about 0.8 ms on shuffled particles;
- 0.42 ms on the previous frame's sorted order, which is the liquid's case;
- minus the copy of the three arrays it replaces, 0.25 ms;
- so about 0.2 ms net in steady state.

The `-m perf` benchmark (`tests/perf/test_m1_transfers.py`) times all of these, together
with the transfers in random, bucket, lattice and half-cell order. It requires the
half-cell order to cost no more than the lattice, and the sort to pay for itself at one
substep against the bucket's order and random order.

Consequences:
- **CPU views.** On the CPU, `particle_positions()` is a view. Since the swap, it is
  valid until the next step only.
- **Frame graphs.** The particle buffers change places every frame. A future CUDA graph
  of the whole liquid frame must be keyed by which buffer is live.

## Alternatives considered

- **Sort by cell.** This was the first version of this ADR. It costs 2.38 ms of
  particles-to-grid, 2.4 times the half-cell order. It also made water seeded by
  `fill_box` (already in half-cell order) about 2 ms per frame slower than not sorting.
- **A counting sort** (histogram, scan, scatter). It took 1.07 ms against 0.80 ms for
  the radix sort, and its order within a bin is not reproducible on the GPU.
- **Cells grouped in 4 × 4 × 4 bricks.** Particles-to-grid is slower in that order
  (2.78 ms): it groups even more particles on the same nodes.
- **Particles-to-grid in shared memory, one tile per brick** (Gao et al. 2018). It took
  about 2.06 ms. That beats the cell order, but not the half-cell order, and needs three
  times the code.
- **Particles-to-grid as a gather** (one thread per face node over its cells' particles).
  It is correct but takes 13.8 ms.
- **Sorting every few frames.** A dam-break front moves 0.5–1.1 cells per frame at
  60 fps, and the CFL cap allows 4. The order would decay within a few frames, and the
  sort is cheap on sorted input.

## How we would know it is wrong

- The M1 performance gate (#57) finds particles-to-grid above about 2 ms per substep in
  play, or the sort a large share of the liquid's frame.
- The NaN guard's dump no longer replays its frame, which would mean the buffer swap
  went wrong.
