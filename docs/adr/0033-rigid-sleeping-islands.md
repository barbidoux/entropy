# ADR 0033: Rigid bodies sleep in islands, rebuilt each frame on the GPU

- Status: proposed
- Date: 2026-11-28

## Context

SPEC 5.4: "bodies at rest sleep in islands and wake on contact, to save time". Issue #89
asks for V-RB-05 (a stack of 10 boxes for 60 s, drift ≤ 1 mm), for islands that sleep and
wake on contact or when a tool touches them, for a count of sleeping bodies on the HUD,
and for tests that a sleeping stack costs less and that a body dropped on it wakes it.

V-RB-05 already passes on the CPU and CUDA: #88's friction holds the stack. What is left
is sleeping. Without it, everything at rest is solved every substep: a settled pile of
5,000 boxes costs as much as a falling one (22 ms per 1/30 s frame, SPEC 3 allows 3 ms,
#90), and piles never stop: 4 to 17 of 100 boxes still move a few centimetres between 15
and 25 s (ADR 0032).

The solver (ADR 0030 to 0032) runs 16 substeps per 1/30 s frame. Each substep finds the
pairs of bodies in contact again (a hash grid, `detect`), solves pairs in parallel
(Jacobi), each body against the static solids by its own thread, then the velocity pass.
Pairs keep friction anchors from substep to substep in per-body rows; static patches keep
theirs in place.

How the engines do it (studied 2026-11-28, from their sources):
- **Box2D v3** and **PhysX** keep persistent islands with serial or CPU bookkeeping:
  merged when a contact begins, split later. Box2D sleeps a body slower than 0.05 m/s
  (plus its turn times its extent) for 0.5 s; PhysX filters the velocity over 0.4 s.
- **Jolt** rebuilds islands every step with a lock-free union-find and tests a
  displacement window: three points of the body may wander 1.5 cm (0.03 m/s × 0.5 s)
  before the timer resets, so jitter that cancels out does not keep a body awake.
- **Bullet** rebuilds islands every step from broadphase pairs; 0.8 m/s and 1 rad/s for
  2 s.
- **NVIDIA Newton's Steel solver** (Warp) rebuilds islands each step with a
  compare-and-swap union-find, requires an island to touch static geometry, and skips
  sleepers by flags. **MuJoCo Warp** finds the gain mostly in contacts not solved.
- All of them sleep an island only as a whole, and one touch wakes it whole.

Matt decided (2026-11-28) that a pile still sliding very slowly, a few centimetres in
10 s, may fall asleep: it is frozen where it is, as the engines above do.

## Decision

- **Islands are rebuilt once per frame, on the GPU**, after the frame's last substep:
  a union-find with compare-and-swap over that substep's pairs of bodies in contact,
  each link joining the larger root to the smaller, then pointer jumping. The static
  solids join no island. The root is the island's smallest body index, whatever the
  order of the threads, so islands are deterministic.
- **A body is quiet** when its speed has stayed under `SLEEP_SPEED_M_PER_S` = 0.05 m/s
  for `SLEEP_TIME_S` = 0.5 s (Box2D's rule and values). Its speed is the larger of
  |v| + |ω| r and (|Δx| + |Δθ| r) / Δt, r its bounding radius and Δ over the frame, so
  both the recovered velocities and the frame's motion count. A faster frame starts the
  timer again. A body is tested once per frame. A displacement window (Jolt's way) was
  considered first: a body speeding up from rest slower than 0.2 m/s² stays within its
  2.5 cm for the first 0.5 s and would sleep before it got going, and V-RB-03's block
  1° past its friction angle (0.19 m/s²) with it. With the speed rule the block passes
  0.05 m/s after 0.26 s and slides.
- **An island sleeps only as a whole**, when every body in it is quiet, at least one
  touches a static solid (so a body in flight, at the top of a throw, never sleeps), and
  none is held back: a body a tool set this frame, or a pair dropped for want of room
  (`pair_overflows`). A body can then never sleep under one that moves.
- **A sleeping body is frozen.** Its velocities are zero; it skips the prediction (no
  gravity), the static solve and the velocity pass. Pairs of two sleeping bodies are not
  tested: their anchors are carried over as they are. Sleeping bodies stay in the hash
  grid, so awake bodies find them.
- **Waking:**
  - **Contact:** when `detect` finds an awake body touching a sleeping one, it marks the
    sleeper's island, and a kernel run before the pairs are solved wakes every body of
    it. From then on it is solved as usual.
  - **Tools:** `set_body_velocity` (what the spawner and the grabber will use) wakes the
    body's whole island and holds it back from sleeping that frame, and
    `wake_in_box(low, high)` wakes every island with a body whose bounds meet the box
    (for tools that act on a region, and for Build mode's edits, M6). New bodies start
    awake; one spawned touching a sleeping island wakes it by contact.
  - **Water** pushes bodies only once the coupling exists (#92); the coupling wakes the
    bodies it pushes (#92's work, noted in that issue).
  - A woken body's timer starts again.
- **Sleeping can be turned off:** `RigidSetup.sleeping`, on by default. The tests of the
  contact and friction physics (stacks, slopes, piles, V-RB-05's stacking) run with it
  off, so a stack that falls asleep cannot pass them trivially; the tests of sleeping
  run with it on, and V-RB-05 runs both ways.
- **Sleep is state.** Each body's sleep flag, island root, window start and window age
  go into checkpoints and NaN dumps; a checkpoint or dump without them restores every
  body awake with a fresh window.
- **The count:** `diagnostics()["sleeping"]`, counted in the frame's sleep kernel and
  read with the rest of the status, without waiting. The HUD shows "N corps, dont M
  endormis" once the game runs rigid bodies (the spawner, #93); #89 provides the line.

## Tests

Written first, each seen failing:
- V-RB-05 (already passing, committed first).
- A stack of boxes resting on the floor falls asleep, all its bodies in the same frame,
  within 1.5 s of being settled; while asleep its positions do not change at all, bit for
  bit, over 10 s.
- A sleeping stack costs less: the contacts the solver reports drop to 0 for it, and a
  `-m perf` benchmark on the laptop shows a sleeping 1,000-box pile's frame at most half
  an awake one's, recorded (SPEC 3's budget is #90's).
- A box dropped 0.5 m on a sleeping stack wakes it, every body, in the substep of the
  contact, and the stack takes the blow as an awake stack does: each box ends within
  1 mm of where it ends on an awake stack, and no box moves faster than 1.1 times the
  awake case's fastest. If the one-substep delay below fails it, the woken bodies' own
  pairs are found in a second pass of `detect` in the same substep.
- A box thrown up never sleeps in flight; a box sliding on the floor faster than
  0.05 m/s does not sleep; a ball rolling at 0.5 m/s does not sleep; V-RB-03 passes with
  sleeping on.
- A box resting on a box that a tool keeps pushing stays awake; the whole stack sleeps
  together once both rest.
- `set_body_velocity` on one box of a sleeping stack wakes the whole stack.
- A pile of 100 boxes on the floor falls asleep within a few seconds of settling
  (Matt's decision: its slow creep is frozen).
- A checkpoint keeps the sleep state bit for bit; one without it restores everything
  awake; a dump replays a frame with sleeping bodies.
- Every test runs on CPU and CUDA except the bit-for-bit ones (CPU).

## Named simplifications

- **Slow creep is frozen:** an island whose bodies all stay under 0.05 m/s for 0.5 s
  sleeps, so a pile still settling by a few centimetres a minute stops where it is
  (Matt's decision). A body speeding up from rest slower than 0.1 m/s² freezes before it
  gets going: a block less than about 0.5° past its friction angle (V-RB-03 asks ±1°),
  or a box tipping very slowly over an edge.
- **One touch wakes the whole island**, however large the pile; a thrown box landing on
  a corner of a 1,000-box pile wakes it all, as in every engine studied.
- **A woken island's own contacts come back one substep later** (unless the test above
  needs the second pass): for that substep its bodies are not held against each other,
  so a struck body can be pushed into the one under it by up to about half the striking
  body's step, and is pushed back in the next substep.
- **Water does not wake a sleeping body** until the coupling (#92) does it.
- **Removing a support does not wake.** No body or solid can be removed yet; when Build
  mode (M6) edits solids or a tool deletes bodies, the islands they touched must be
  woken.
- **Sleeping bodies still cost their place in the hash grid** and their thread's early
  exit in each kernel; the active bodies are not compacted (a later optimisation, #90).
- Islands are tested once per frame, so a body is asleep or awake for a whole frame,
  apart from waking.

## Alternatives considered

- **Persistent islands** (Box2D, PhysX): cheaper to maintain on a CPU, but the merges and
  splits are serial and our contacts are rebuilt on the GPU every substep; the
  bookkeeping would cross between them every substep.
- **Each body sleeping on its own, without islands**: a box could sleep under one still
  moving, and a stack would come apart; every engine avoids it.
- **A displacement window** (Jolt): not woken by jitter that cancels out, but a body
  speeding up slowly from rest sleeps before it gets going (above). Our resting bodies'
  recovered velocities are near zero (#101, #102), so jitter does not need the window.
- **Compacting the awake bodies into a list** (Jolt, MuJoCo Warp): saves the early exits,
  at the cost of a scan each frame and indirection in every kernel; left to #90, which
  will measure what remains.

## How we would know it is wrong

- Things freeze in mid-air, mid-slide or while visibly tipping.
- A pile wakes and pops when touched, or a woken stack jumps.
- A pile never falls asleep (watch the HUD's count), or sleeps and keeps costing.
- A stack hangs in the air after the box under it was thrown away.
