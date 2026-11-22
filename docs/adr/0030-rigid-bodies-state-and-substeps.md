# ADR 0030: Rigid bodies: their state on the device, and how a substep keeps its precision

- Status: accepted
- Date: 2026-11-22

## Context

M2 starts the rigid-body solver of SPEC 5.4: extended position-based dynamics (XPBD,
Müller et al. 2020) with substeps of at most 1/480 s, one constraint iteration each.
Contacts, friction and sleeping follow (#87, #88, #89); the liquid's pressure solve will
couple with the bodies (#92). This first step (#86) fixes what a body is on the device,
how a substep moves it, the NaN guard and checkpoints.

Three effects showed up in its first tests and its review:
- **Recovering velocities from positions.** XPBD recovers a body's velocity as
  (x − x₀)/h after the constraints moved it. At 20 m from the origin a float32
  position's spacing is 1.9e-6 m, and the errors accumulate: a body falling there read a
  speed 1.24 % low after 1 s (#98's review). The same holds for orientations: a
  quaternion's components near 1 keep about 1e-7 of a rotation of 1e-3 rad per substep,
  and a body spinning at 2 rad/s read 2.0002–2.0005 rad/s after a second.
- **Round trips through rotations.** Turning the angular velocity into the body's frame
  and back scales it by |q|⁴, which float32 leaves about 1e-7 off 1 after each
  normalisation: a drift of about 1e-4 a second at 480 substeps.
- **The explicit gyroscopic step gains energy.** Explicit Euler raises |L_body|² by
  h²·|L × ω|² every substep. A box tumbling at 10 rad/s gained 10 % of |L| in 10 s and
  116 % in 60 s; one spinning at 30 rad/s about its major axis with a 5 % wobble
  reached 1,410 rad/s in 60 s (#98's review, the same in float64).

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
- **The gyroscopic torque by the implicit midpoint rule** in the body's frame,
  I(ω₁ − ω₀)/h = −ω_m × (I ω_m) with ω_m = (ω₀ + ω₁)/2, solved by four fixed-point
  iterations; only its change of the angular velocity goes through the rotations. In the
  review's runs |L| stayed within 3e-4 and the energy within 6e-4 over 60 s at 1–20
  rad/s. It holds to about 100 rad/s (|ω|·h about 0.2); faster spins lose energy, never
  gain it (a 2 m plank at 200 rad/s lost 26 % in 10 s), which is safe for a game.
- **The logarithm keeps a fast spin's sense.** The substep's rotation is built from ω,
  so a turn of more than π (above about 1,500 rad/s) is real and is not flipped.
- **The substeps follow the advance**: ⌈advance / (1/480 s)⌉ equal substeps, so
  `max_advance_s` is unlimited (ADR 0013). One kernel launch per substep: with the
  status kernel and the copies, 0.39 ms of CPU per 1/30 s step at 5,000 bodies (#98's
  review). CUDA graphs wait for the contacts and the benchmark (#90).
- **NaN guard as the liquid's** (SPEC 5.9). The status is read at the next call. An
  orientation stays a unit quaternion or turns NaN (Warp's normalize gives the identity
  for a zero quaternion), and a NaN angular velocity is itself caught. A dump holds the frame's starting state, its result, its time step and gravity,
  and replays bit for bit on the CPU. A frozen solver refuses new bodies and velocities.
- **Checkpoints are synchronous copies** (5,000 bodies are about 0.4 MB). On CUDA taking
  one waits for the queued GPU work, the save hitch ADR 0024 avoided for the liquid;
  asynchronous checkpoints are #99.

Named simplifications (SPEC 5.4 and the module):
- symplectic Euler per substep: a body in free fall drops ½gt² + ½ght, 0.21 % more than
  ½gt² after 1 s (V-RB-01 allows 0.5 %);
- the gyroscopic iteration holds to about 100 rad/s; faster spins lose energy;
- positions accumulate in float32: 20 to 80 m from the origin, a body moving at 1.26 m/s
  ends 1.5 cm from x₀ + vt after 60 s; its velocity stays exact.

## Alternatives considered

- **Velocities from position differences**, as in the XPBD paper and most float32
  implementations. That gives the precision loss above, which would show as jitter in
  resting contact.
- **Warp's own XPBD integrator** (`warp.sim`). Its model and state objects would sit
  beside ours. The coupling with the liquid's pressure solve (SPEC 6), the static SDF,
  and our checkpoints and NaN guard need our own arrays. Kept as a reference.
- **An array of structs** (one Warp struct per body). The coupling and the renderer read
  single fields of every body, which a structure of arrays serves directly.
- **Explicit gyroscopic Euler** (the first version): unstable at a game's spins, above.
  Catto's single Newton step of implicit Euler is stable but lost 16 % of the energy in
  10 s at 10 rad/s.
- **Positions relative to a moving origin** (ADR 0026's answer for far maps). Not needed
  for M2's rooms; the increments already keep the velocities exact.

## How we would know it is wrong

- Resting bodies jitter or creep in the contact tests (#87, #89).
- The 5,000-body benchmark (#90) finds the launches, not the kernels, set the cost.
- A tumbling body visibly gains or loses spin over a minute of play, or a body spins
  faster than about 100 rad/s in play.
