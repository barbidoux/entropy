# ADR 0030: Rigid bodies: their state on the device, and how a substep keeps its precision

- Status: accepted
- Date: 2026-11-22

## Context

M2 starts the rigid-body solver of SPEC 5.4: extended position-based dynamics (XPBD,
Müller et al. 2020) with substeps of at most 1/480 s, one constraint iteration each.
Contacts, friction and sleeping follow (#87, #88, #89); the liquid's pressure solve will
couple with the bodies (#92). This first step (#86) fixes what a body is on the device,
how a substep moves it, the NaN guard and checkpoints.

Two float32 effects showed up in its first tests:
- **Recovering velocities from positions.** XPBD recovers a body's velocity as
  (x − x₀)/h after the constraints moved it. At 20 m from the origin a float32
  position's spacing is 1.9e-6 m, so at h = 1/480 s the recovered speed is only good to
  about 1e-3 m/s. The same holds for orientations: a quaternion's components near 1 keep
  about 1e-7 of a rotation of 1e-3 rad per substep, and a body spinning at 2 rad/s read
  2.0005 rad/s after a second.
- **Round trips through rotations.** Turning the angular velocity into the body's frame
  and back scales it by |q|⁴, which float32 leaves about 1e-7 off 1 after each
  normalisation: a drift of about 1e-4 a second at 480 substeps.

## Decision

- **A structure of arrays** in `entropy.solvers.rigid`, one entry per body, sized at
  allocation (`RigidSetup.max_bodies`, 5,000 by default after SPEC 3):
  - state: position (m), orientation (unit quaternion x, y, z, w, body to world, as
    Warp's `quat`), velocity (m/s), angular velocity (rad/s, world frame);
  - properties: mass (kg), principal moments of inertia (kg·m², body frame), shape and
    size.

  Shapes are a box, a sphere and a capsule along z (`Shape`), centred on their centre of
  mass with their principal axes on the body's axes, so the inertia is a diagonal.
  Coupling and rendering read these arrays (`fields()`); only the solver writes them.
- **The count lives on the CPU.** Bodies come from the CPU (maps, the spawner), so the
  solver knows how many are live without reading back.
- **A substep keeps its motion as increments**, a displacement and a small rotation, and
  recovers the velocities from them, not from the difference of two nearby positions or
  orientations. Constraints will add their corrections to those increments (#87).
- **Orientations turn by the exact exponential** of ω·h, and angular velocities come
  back by the exact logarithm, so a spin is not shortened by the integration.
- **The gyroscopic torque is integrated explicitly** in the body's frame, once per
  substep, and only its change of the angular velocity goes through the rotations.
- **The substeps follow the advance**: ⌈advance / (1/480 s)⌉ equal substeps, so
  `max_advance_s` is unlimited (ADR 0013). One kernel launch per substep: 16 launches a
  frame at 30 fps, about 0.25 ms of CPU. CUDA graphs wait for the contacts and the
  benchmark (#90).
- **NaN guard and checkpoints as the liquid's** (SPEC 5.9, ADR 0024). The status is read
  at the next call, and a dump holds the frame's starting state, its result and its time
  step. Checkpoints are synchronous copies: 5,000 bodies are about 0.5 MB.

Named simplifications (SPEC 5.4 and the module):
- symplectic Euler per substep: a body in free fall drops ½gt² + ½ght, 0.21 % more than
  ½gt² after 1 s (V-RB-01 allows 0.5 %);
- the explicit gyroscopic step: a tumbling body's angular momentum drifts, under 1 % in
  2 s in the tests.

## Alternatives considered

- **Velocities from position differences**, as in the XPBD paper and most float32
  implementations. That gives the precision loss above, which would show as jitter in
  resting contact.
- **Warp's own XPBD integrator** (`warp.sim`). Its model and state objects would sit
  beside ours. The coupling with the liquid's pressure solve (SPEC 6), the static SDF,
  and our checkpoints and NaN guard need our own arrays. Kept as a reference.
- **An array of structs** (one Warp struct per body). The coupling and the renderer read
  single fields of every body, which a structure of arrays serves directly.
- **Positions relative to a moving origin** (ADR 0026's answer for far maps). Not needed
  for M2's rooms; the increments already keep the velocities exact near the origin.

## How we would know it is wrong

- Resting bodies jitter or creep in the contact tests (#87, #89).
- The 5,000-body benchmark (#90) finds the launches, not the kernels, set the cost.
- A tumbling body visibly gains or loses spin over a minute of play.
