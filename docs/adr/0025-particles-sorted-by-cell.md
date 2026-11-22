# ADR 0025: The liquid sorts its particles by cell each frame; particles-to-grid stays a scatter

- Status: accepted
- Date: 2026-11-22

## Context

At 1 M particles the APIC transfers took most of the liquid's 9 ms per frame (SPEC 3).
Issue #65 measured them in random order: about 3.4 ms per substep for particles to grid
and 0.9 ms for grid to particles, at 2 to 4 substeps a frame.

Two causes:
- Particles that share a cell sit far apart in memory, so every read of the grid and
  every atomic add to it lands at random.
- Particles-to-grid scatters 162 float atomics per particle: 3 components × 27 nodes
  × mass and momentum.

## Decision

- **Sort every frame.** At the start of each frame the liquid sorts its particles by
  cell (`entropy.solvers.liquid.sorting.CellSort`):
  - a stable radix sort of x-major cell keys, which is the grid's own memory order;
  - a gather into the second particle buffer, which then becomes the live one;
  - the buffer left behind still holds the frame's starting state, so the NaN guard's
    dump keeps it.

  This replaces the copy the dump needed anyway (ADR 0018), so the sort adds no memory
  traffic beyond its keys.
- **Keep the scatter.** Particles to grid stays a scatter of atomics. Sorted, it runs at
  about 2.3 ms per substep.

Measured on the laptop at 1 M particles on the Test Tank's grid (unofficial: 20 s warm-up,
GPU shared with other runs; the official figures come with the M1 performance gate, #57):

| Pass | Random order | Sorted by cell |
| --- | --- | --- |
| Particles to grid | 3.42 ms | 2.34 ms |
| Grid to particles | 0.87 ms | 0.14 ms |
| Both, per substep | 4.29 ms | 2.49 ms |

The sort costs 0.76 ms per frame, so it pays for itself from the first substep. At 2
substeps it saves about 2.8 ms per frame.

## Alternatives considered

- **A counting sort** (cell histogram, scan, scatter): 1.07 ms against 0.80 ms for the
  radix sort, and its order within a cell is not reproducible on the GPU.
- **Cells ordered by 4 × 4 × 4 bricks**: the scatter is slower in that order (2.78 ms).
- **Particles to grid in shared memory, one tile per brick** (Gao et al. 2018, with Warp's
  `tile_scatter_add`): 2.06 to 2.13 ms, against 2.2 to 2.9 ms for the scatter in other
  runs on the shared GPU. That is within noise of the sorted scatter, for a brick order, a
  fallback for particles that leave their brick, and code three times larger. It is not
  adopted.
- **Particles to grid as a gather** (one thread per face node over its cells' particles,
  no atomics): correct, but 13.8 ms.
- **Sorting every few frames**: it would save part of the 0.76 ms, and particles move
  under a third of a cell per frame. Every frame is simpler, and its cost is known.

## How we would know it is wrong

- The M1 performance gate (#57) finds the transfers above about 2 ms per substep, or
  the liquid over its 9 ms with the sort as a large share. The shared-memory tile is
  the next step then.
- The NaN guard's dump no longer replays its frame (the buffer swap went wrong).
