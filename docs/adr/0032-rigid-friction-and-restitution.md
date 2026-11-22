# ADR 0032: Rigid friction and restitution, once per contact patch

- Status: accepted
- Date: 2026-11-22

## Context

SPEC 5.4 asks for Coulomb friction, static and dynamic, plus restitution, all from the
material files. SPEC 9.4 validates them:
- V-RB-02: a bounce rises e² of its fall, within 3 %;
- V-RB-03: a block on an incline holds below the friction angle and slides above it,
  with the threshold within ±1°.

ADR 0031 solves contacts with the static solids by each body's thread (Gauss–Seidel),
contacts between bodies in parallel (Jacobi), and runs a velocity pass at contact
patches: a body's contacts with the static solids along one direction, or the contacts
of a pair of bodies.

Müller et al. 2020 (§3.5–3.6) give friction for XPBD:
- static friction is a position correction that undoes a contact's sliding over the
  substep, kept when it is within μs times the normal correction;
- dynamic friction and restitution act in a velocity pass after the velocities are
  recovered.

Their solver is serial Gauss–Seidel over contacts. A first try applied their rules per
contact in parallel. It failed V-RB-03 at the angle where the block should hold: the
block crept at 5 mm/s. Jacobi closes only part of the sliding each substep, and the
rest becomes speed.

## Decision

- **Coefficients.** A body has `static_friction`, `dynamic_friction` and `restitution`.
  The static solids have one set for all of them, in `RigidSetup` (a named
  simplification: the map's surfaces cannot differ yet). All default to 0.
  - Frictions must be finite and non-negative; a restitution lies in [0, 1].
  - Bodies' coefficients are per-body properties, so checkpoints and dumps carry them.
  - The spawner (#93) takes them from the material files, with their sources (CLAUDE.md
    rule 4).
- **A pair's coefficients:** each friction is the geometric mean √(μa μb), and the
  restitution is the larger, max(ea, eb). The static solids count as the other body.
  Both are common game-engine rules. The mean keeps a frictionless surface frictionless;
  the max lets a bouncy ball bounce on anything.
- **Friction acts once per patch, at its centre of pressure** (the mean of its
  contacts' points, weighted by their normal corrections), along the patch's mean
  normal. One exact correction there replaces several partial ones.
- **Static friction**, after the depths:
  - It is the correction that undoes the sliding of the material point at the centre of
    pressure over the substep.
  - It is kept when its size is within μs times the patch's normal correction.
  - For the static solids it runs in the body's thread, after its depth sweep. Between
    bodies it runs in parallel, scaled by 1 / (the larger patch count of the two bodies)
    and given to both alike.
- **The velocity pass**, after the velocities are recovered:
  - **Restitution:** the separating speed at the centre of pressure becomes e times the
    approaching speed before the substep, or the separating speed it had if the bodies
    were already separating. There is no bounce below 2|g|h, so resting bodies do not
    jitter.
  - **Dynamic friction:** an impulse against the sliding there, of at most μd times the
    patch's normal impulse in the substep. That impulse is its depth corrections over h
    plus the restitution's impulse, so a contact met right at the surface still rubs.
  - **Friction against turning about the normal** (torsion): at most μd times the
    normal impulse times the patch's torsion radius, the root-mean-square distance of
    its points from the centre of pressure. Without it, one point has no lever against
    a spin, and a box spun flat would never stop.
- **A wide patch moves the body without turning it** (static solids only).
  - A patch is wide along the sliding when its points reach √3 σ along it (σ is their
    spread, √3 σ the half length of an even spread). That reach must be at least μ times
    the height of the body's centre above the patch. The normal forces can then shift
    and hold friction's tipping torque, which one point cannot show.
  - On a wide patch, both frictions move the body without turning it. Otherwise (a
    sphere, a capsule lying across the slope, a tall narrow box) they act at the centre
    of pressure and turn it, so a ball rolls.
  - The width counts every sample within 5 mm of the surface (`TOUCH_M`), sunk in or
    not. Lifted by its bottom face's centre (ADR 0031), a box resting flat has a single
    contact, yet its whole face holds it.
  - Without this rule, the friction at the centre of pressure tipped the box a little
    every substep. The next depth sweep levelled it about its centre, and the block
    crept down the slope by rolling.

## Measured

On CUDA, 2026-11-22.

| Case | Result | Expected |
| --- | --- | --- |
| A box sliding at 2 m/s, μ = 0.3 | stops after 0.682 m, untilted | v²/2μg = 0.680 m |
| A box sliding at 2 m/s, μ = 0.6 | stops after 0.342 m, untilted | 0.340 m |
| V-RB-02: a sphere or a box dropped flat from 1 m, e = 0.5 | rises 0.246 m (−1.7 %) | e² = 0.25 m |
| V-RB-02: the same, e = 0.8 | rises 0.632 m (−1.3 %) | e² = 0.64 m |
| V-RB-03: a block, μ = 0.5, 1° below atan μ | does not move in 1 s | holds |
| V-RB-03: the same, 1° above | accelerates at 0.1914 m/s² | 0.1914 m/s² |
| V-RB-03: the same, 3° above | accelerates at 0.5740 m/s² | 0.5740 m/s² |
| A sphere on a 20° slope, μ = 0.6 | rolls at 2.397 m/s after 1 s, spin = v/r | (5/7) g sin θ t = 2.397 m/s |
| A box spun flat at 10 rad/s, μ = 0.5 | stops in 0.25 s | 0.36 s under an even pressure |

## Named simplifications

These are named in SPEC 5.4 and in the module.

- **One surface material for all the static solids** (`RigidSetup`), until maps give
  their surfaces materials.
- **Torsion is too strong on box faces.** A box's samples weigh its corners and edges
  more than an even pressure would. The torsion radius comes out about 1.4 times too
  large, so a spinning box stops about 1.4 times too soon. Torsion has dynamic friction
  only, with no static hold against turning.
- **Between bodies, friction always turns them** (no wide rule), so both momenta are
  kept. A box resting on another box on a slope can creep as the first try did; V-RB-05
  (#89) will show whether stacks need more.
- **The bounce is 1.3 to 1.7 % low**, within V-RB-02's 3 %. The target takes the
  approach speed at the start of the landing substep, which misses that substep's
  gravity: up to e g h, about 1 % of the height here.
- **The tipping test is a threshold.** A patch just under the width tips the body by its
  full lever, and one just over it does not tip it at all.

## Alternatives considered

- **Müller's per-contact friction, in parallel.** It is the paper's rule. Under Jacobi
  it leaves part of the sliding each substep, and the block crept at 5 mm/s where it
  should hold.
- **A per-contact velocity pass in the body's thread.** It was tried for the normal
  velocities: one sweep left normal speeds into the surface and made a resting capsule
  jitter.
- **Friction at the centre of pressure, always turning.** The block crept by rolling, as
  described under the wide rule.
- **Several Jacobi iterations per substep.** They would converge, at several times the
  cost, with SPEC 3's budget already exceeded (#90).
- **Persistent contacts with friction anchors.** They hold static friction exactly
  across substeps, but contacts are found again each substep (ADR 0031), with no
  identity to keep.

## How we would know it is wrong

- Boxes creep on ramps in the Room, or stacks slide apart (V-RB-05, #89).
- Spinning boxes stop visibly too fast, or keep spinning.
- A ball on a box does not roll off, or a box rolls like a ball.
- Bounces look dead or too lively against the material's e.
