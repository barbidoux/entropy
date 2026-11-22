# ADR 0024: Checkpoints keep in step with the NaN guard

- Status: accepted
- Date: 2026-11-21

## Context

SPEC 4.6 asks for checkpoints in host memory: one every 10 s of simulated time in a
ring of 6, and one on F5; F9 restores. SPEC 5.9's NaN guard freezes the world when a
solver finds its state non-finite. The liquid reads each step's status at the start of
the next frame, so it adds no sync point (ADR 0018). Two consequences showed up in
review of #74:

- A checkpoint saved after the last step has not been checked yet. If that step went
  non-finite, the checkpoint holds NaN.
- F5 while frozen would save the non-finite state. F9 would then restore it, freeze
  again, and leave the player stuck, because F9 only went to the latest checkpoint.

Saving also cost time: allocating pinned host memory on every save took 5.6 ms of CPU
at 262 k particles and 11 ms at 655 k, a visible hitch every 10 s.

## Decision

`entropy.sim.checkpoints.Rewind` keeps the ring and the guard together:

- **A checkpoint at the start.** There is always somewhere to go back to, even if the
  world freezes in its first 10 s with no F5.
- **No saving while frozen.** F5 does nothing until F9 has restored finite water.
- **A freeze drops the unchecked checkpoints.** When a step finds the world non-finite,
  the checkpoints saved since the last good step (those at the current clock) are
  dropped; the ones before were checked by a later step.
- **Restoring checks first.** F9 reads the last step's verdict inside the guard
  (`Simulation.check`), dropping what it condemns, then restores and unfreezes. So an
  unchecked checkpoint is never restored.
- **F9 again goes further back**, one checkpoint per press, down to the oldest. Once the
  world runs on from a restored checkpoint, the later ones are forgotten.
- **Pinned buffers are reused.** The checkpoint that falls off the ring lends its host
  buffers to the new one (`checkpoint_async(reuse=...)`). They grow by a quarter when too
  small.

The logic lives in `sim`; the app only wires keys to it (SPEC 8.1).

Named simplification (CLAUDE.md rule 7): water in flight outside the grid (the hose's
jet) is not saved. A restore drops it, so water that was in the air when the checkpoint
was taken is lost.

## Alternatives considered

- **Reading the status at once after each step:** every checkpoint would be checked,
  but every frame would wait for the GPU.
- **Marking entries checked or unchecked:** this is the same rule, but kept as state on
  each entry. Reading the verdict before restoring needs nothing stored.
- **F9 to the latest only:** five of the six slots could never be reached.
- **Pinned buffers for the full capacity:** 2 M particles × 60 B × 6 slots is 720 MB of
  locked host memory.

## How we would know it is wrong

- F9 restores a state that freezes again at once.
- A save still shows as a frame-time spike in the HUD at 1 M particles.
- Players lose water they expected to keep: the jet in flight at save time.
