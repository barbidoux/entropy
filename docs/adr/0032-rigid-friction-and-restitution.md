# ADR 0032: Rigid friction and restitution, once per contact patch

- Status: accepted
- Date: 2026-11-27

## Context

SPEC 5.4 asks for Coulomb friction, static and dynamic, plus restitution, all from the
material files. SPEC 9.4 validates them:
- V-RB-02: a bounce rises e² of its fall, within 3 %;
- V-RB-03: a block on an incline holds below the friction angle and slides above it,
  with the threshold within ±1°.

ADR 0031 solves contacts with the static solids by each body's thread (Gauss–Seidel),
contacts between bodies in parallel (Jacobi), and runs a velocity pass at contact
patches: a body's contacts with the static solids along one direction, or the contacts
of a pair of bodies. ADR 0031 also names what friction must hold, all of which slide or
fall without it:
- a box resting 1 to 8 cm off-centre on an equal one;
- a 4-3-2-1 pyramid of boxes;
- a crate under a slab 64 times heavier.

Müller et al. 2020 (§3.5–3.6) give friction for XPBD:
- static friction is a position correction that undoes a contact's sliding over the
  substep, kept when it is within μs times the normal correction;
- dynamic friction and restitution act in a velocity pass after the velocities are
  recovered.

Their solver is serial Gauss–Seidel over contacts. Ours shares each pair's correction
with its bodies' other pairs, and the part a substep leaves over adds up: friction must
hold across substeps, not only within one.

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
- **A wide patch holds the body's centre; any other holds the point touching now.**
  - A patch is wide along the sliding when its points reach √3 σ along it (σ is their
    spread, √3 σ the half length of an even spread). That reach must be at least μ times
    the height of the body's centre above the patch. The normal forces can then shift
    and hold friction's tipping torque, which one point cannot show. The width counts
    every sample within 5 mm of the surface (`TOUCH_M`), sunk in or not: lifted by its
    bottom face's centre (ADR 0031), a box resting flat has a single contact, yet its
    whole face holds it.
  - On a wide patch, friction moves the resting body without turning it. Without this
    rule, friction at the centre of pressure tipped a box on a slope a little every
    substep; the next depth sweep levelled it about its centre, and the block crept down
    by rolling.
  - Between bodies, a wide patch needs one body on top of the other (the normal within
    60° of the vertical). Static friction then turns neither body. The torque the
    shifted normal forces would pass to the lower body, (x_lower − x_upper) × P, is left
    to the lower body's own support, as the static solids take a resting body's. Given
    as a turn each substep and levelled by the next sweep, it moved the anchor on the
    lower body's face by about 60 µm a substep against 17 µm of sliding: a box resting
    on another on a slope 3° under its friction angle crept 0.04 to 4.5 mm in 5 s on
    CUDA, depending on its start. Dynamic friction, between bodies that slide, still
    gives the lower body that torque, so both momenta are kept.
  - On a patch that is not wide (a sphere, a capsule lying across the slope, a tall
    narrow box), friction acts at the centre of pressure and turns the body, so a ball
    rolls.
- **Static friction keeps an anchor on a wide patch** (`Anchors`), as PhysX's patch
  friction and Solver2D's "sticky" solver do:
  - The anchor is where static friction holds the resting body, kept from substep to
    substep while the patch lasts. The correction undoes the body's drift since the
    patch began, not only this substep's: Jacobi's leftover is undone in later substeps,
    as the normal overlap's is. In the steady state both are amplified alike by the
    Jacobi share, so a pair holds up to μs.
  - The anchor follows the resting body's centre (a world offset from it, set with the
    anchor), not its rocking, which the normal forces hold. On a pair it is fixed to the
    lower body, so a box on a tilting plank follows the plank.
  - A patch that is not wide holds the point touching now, as Müller et al. do, and
    leaves its anchor as it is. A point of a ball's surface rolls away from the slope,
    and an anchor there dragged the ball back: it rolled at 2.15 m/s after 1 s where
    (5/7) g sin θ t is 2.40 m/s.
  - Identity:
    - a body's patches with the static solids find their anchors by normal, within
      about 11° (`ANCHOR_COS`);
    - a pair's anchor sits in its lower-indexed body's row of `PAIRS_PER_BODY` entries,
      found by the other body's index;
    - `detect` carries the anchors over from the last substep; a pair no longer in
      contact loses its anchor.
  - A new patch's anchor is set where the patch is, and so is the anchor of a patch that
    slides.
- **The bound is μs times the patch's normal impulse, averaged.**
  - While the anchor holds, the normal impulse is averaged over about 8 substeps
    (`ANCHOR_MEAN`, an exponential mean). One substep's normal impulse dips and peaks as
    the Jacobi shares and the sinking samples change. Against it, the anchors of a pair
    of boxes on a slope 3° under the friction angle (a 13 % margin) broke, and the pair
    slid.
  - Past the bound, the correction is clamped to it, as impulse solvers clamp friction
    to its cone (PhysX, Box2D, Bullet). The anchor keeps the drift left while it is
    within 0.2 mm (`ANCHOR_SLIP_M`) and undoes it when the bound allows. Past that, the
    patch slides and its anchor is set again.
  - A clamped patch gets no dynamic friction in the velocity pass that substep: its
    friction is already Coulomb's largest, min(needed, μ N), and is counted once.
- **The velocity pass**, after the velocities are recovered:
  - **Restitution:** the separating speed at the centre of pressure becomes e times the
    approaching speed before the substep, or the separating speed it had if the bodies
    were already separating. There is no bounce below 2|g|h, so resting bodies do not
    jitter.
  - **Dynamic friction:** an impulse against the sliding there, of at most μd times the
    patch's normal impulse in the substep. That impulse is its depth corrections over h
    plus the restitution's impulse, so a contact met right at the surface still rubs.
    On a wide patch it acts as static friction does, without turning the resting body.
  - **Friction against turning about the normal** (torsion): at most μd times the
    normal impulse times the patch's torsion radius, the root-mean-square distance of
    its points from the centre of pressure. Without it, one point has no lever against
    a spin, and a box spun flat would never stop.
- **The pairs' Jacobi average is over-relaxed** (`PAIR_RELAXATION`): a pair's
  corrections are scaled by min(1, ω / n), with ω = 1.8 and n the larger pair count of
  its two bodies (Macklin et al. 2014, "Unified particle physics for real-time
  applications", §4.2). This replaces ADR 0031's 1 / n.
  - A lone pair keeps a scale of 1, so it is solved exactly, not overshot.
  - With 1 / n, the pyramid's rows sank and tilted by 0.36°. The side-by-side boxes
    overlapped their neighbours by about 0.5 mm and pushed each other apart, every pair
    asked for more friction than its bound, and the top box crept 29 mm in 10 s.
  - Several passes per substep also hold it (below), at that many times the contacts'
    cost.
- **Anchors are state.** Checkpoints and dumps carry them, so a replay is exact; new
  bodies start with none.

## Measured

On CUDA, 2026-11-27, except where noted.

| Case | Result | Expected |
| --- | --- | --- |
| A box sliding at 2 m/s, μ = 0.3 | stops after 0.681 m, untilted | v²/2μg = 0.680 m |
| A box sliding at 2 m/s, μ = 0.6 | stops after 0.342 m, untilted | 0.340 m |
| V-RB-02: a sphere or a box dropped flat from 1 m, e = 0.5 | rises 0.246 m (−1.7 %) | e² = 0.25 m |
| V-RB-02: the same, e = 0.8 | rises 0.632 m (−1.3 %) | e² = 0.64 m |
| V-RB-03: a block, μ = 0.5, 1° below atan μ | does not move in 1 s | holds |
| V-RB-03: the same, 1° above | accelerates at 0.1915 m/s² | 0.1914 m/s² |
| V-RB-03: the same, 3° above | accelerates at 0.5740 m/s² | 0.5740 m/s² |
| A sphere on a 20° slope, μ = 0.6 | rolls at 2.397 m/s after 1 s, spin = v/r | (5/7) g sin θ t = 2.397 m/s |
| A box spun flat at 10 rad/s, μ = 0.5 | stops in 0.25 s | 0.36 s under an even pressure |
| μ = 0.5, each body's largest move over 10 s after 1 s to settle: | | |
| a box 1, 4 or 8 cm off-centre on an equal one | 0.000 mm | under 1 mm (the tests) |
| a 4-3-2-1 pyramid | 0.012 mm | under 1 mm |
| a crate under a slab 64 times heavier | 0.37 mm | under 1 mm |
| two boxes on a slope 3° under atan μ (5 s; 0.000 mm from 8 starts) | 0.000 mm | under 1 mm |

The over-relaxation changes ADR 0031's numbers between bodies. Measured by one probe on
CUDA, with 1 / n and then with min(1, 1.8 / n):

| Case | 1 / n | min(1, 1.8 / n) |
| --- | --- | --- |
| The top of 3, 5 and 10 stacked 20 cm boxes sinks (ADR 0031: 0.5, 1.7, 7.7 mm) | as ADR 0031 | 0.28, 0.95, 4.26 mm |
| A box 100 times heavier on a 3-stack: sinks, jitters (ADR 0031: 43, 41 mm) | as ADR 0031 | 23.9, 15.9 mm |
| Two boxes squeezing a sphere at 1 m/s come back at (ADR 0031: 0.31 m/s) | as ADR 0031 | 0.26 m/s |
| 12 spheres of 5 cm spawned in a 30 cm clump: peak speed | 2.6 m/s | 3.4 m/s |
| 12 spheres of 10 cm, the same (the fourth review's scene, seed 0) | 19.2 m/s | 24.6 m/s |
| 27 boxes spawned 5 cm into each other: peak speed, after 2 s | 12.2, 11.4 m/s | 16.3, 13.3 m/s |
| A box dropped 0.5 m onto one box, onto two: rebound | 0.35, 0.95 m/s | 0.35, 0.96 m/s |
| A box at 4 m/s into a row of 3: the row and the thrown box leave at | 0.74, 1.28, 1.65, 0.33 m/s | 0.74, 1.27, 1.41, 0.58 m/s |
| V-RB-04, angular momentum over its terms' sizes: seed 4, worst of seeds 0 to 9 | 8.7e-4, 1.35e-3 | 6.3e-4, 1.20e-3 |
| V-RB-04, linear momentum (with the pairs' exact shifts, #101's eighth round) | 1.6e-7 | 1.6e-7 |

The clump scenes differ from ADR 0031's, so the table compares this probe's two columns.

## Named simplifications

These are named in SPEC 5.4 and in the module.

- **One surface material for all the static solids** (`RigidSetup`), until maps give
  their surfaces materials.
- **Static friction holds on average.** Its bound follows the normal impulse over about
  8 substeps (1/60 s), so it lags a load that changes faster. A body at its bound may
  creep up to 0.2 mm before it slides.
- **Torsion is too strong on box faces.** A box's samples weigh its corners and edges
  more than an even pressure would. The torsion radius comes out about 1.4 times too
  large, so a spinning box stops about 1.4 times too soon. Torsion has dynamic friction
  only, with no static hold against turning.
- **A patch that is not wide uses no anchor.** Its static friction undoes one substep's
  sliding, so Jacobi's leftover is lost there, as in the first try below. So is a pair
  past its body's row of `PAIRS_PER_BODY` anchors.
- **Static friction between bodies does not turn the lower one.** Its tipping torque is
  left to the lower body's support, which holds it when that support is wide (a box on
  the floor or on another box). A lower body that cannot hold it, a ball under a box,
  misses it while static friction holds; dynamic friction gives it.
- **Over-relaxation makes bodies spawned overlapping pop harder** (16 m/s instead of
  12 m/s for 27 boxes 5 cm into each other, 24.6 instead of 19.2 m/s for 12 spheres of
  10 cm in a clump). The spawner (#93) places bodies apart. ω is
  tuned on the pyramid, not derived (ω = 1.5 left the top box creeping 11.6 mm in
  10 s). The graph colouring of #90 would replace it with Gauss–Seidel between pairs.
- **The bounce is 1.3 to 1.7 % low**, within V-RB-02's 3 %. The target takes the
  approach speed at the start of the landing substep, which misses that substep's
  gravity: up to e g h, about 1 % of the height here.
- **The tipping test is a threshold.** A patch just under the width tips the body by its
  full lever, and one just over it does not tip it at all.

## Alternatives considered

Measured on the CPU, in the tests' scenes.

- **Müller's per-contact friction, in parallel.** It is the paper's rule. Under Jacobi
  it leaves part of the sliding each substep, and the block crept at 5 mm/s where it
  should hold.
- **Per-patch friction without anchors.** It held single bodies (V-RB-03), not stacks.
  A box 1 cm off-centre on another crept 59 mm in 10 s, and two boxes on a slope 3°
  under the friction angle slid 0.80 m in 5 s.
- **A support rule** (shock propagation: the lower body counts as immovable for the
  upper one's friction). Dense piles exploded: bodies flew apart and fell through the
  floor, in each variant tried (per contact, one level only). It was removed.
- **Anchors that let go at the first substep past the bound** (Müller's all-or-nothing
  rule). The noisy per-substep normal impulse broke them: the slope pair slid 0.11 m
  in 5 s.
- **Static friction turning the lower body**, by the torque of the shifted normal
  forces, as dynamic friction does. It kept both momenta, but the turn and the next sweep's
  levelling moved the anchor on the lower body each substep, and a box on another on a
  slope crept up to 4.5 mm in 5 s.
- **Anchors on every patch.** A ball's anchor, a point of its surface, rolled away and
  dragged it back: 10 % too slow down a slope.
- **Clamping without skipping dynamic friction.** It counted friction twice: V-RB-03's
  block 1° above the friction angle crept 0.44 mm in 1 s instead of sliding 10 cm.
- **A slip tolerance alone**, without the averaged bound and the centre's slip. No value
  held both the pyramid and the pairs of boxes: 0.1 mm held the pyramid (0.03 mm) and
  let the offset boxes creep 2.8 to 5.1 mm; 0.2 mm did the opposite.
- **Several passes per substep.** They converge: the pyramid's top crept 29, 9.5, 0.69
  and 0.25 mm in 10 s with 1, 2, 4 and 8 passes. Each pass costs a full contact solve,
  with SPEC 3's budget already exceeded (#90).
- **A per-contact velocity pass in the body's thread.** It was tried for the normal
  velocities: one sweep left normal speeds into the surface and made a resting capsule
  jitter.

## How we would know it is wrong

- Boxes creep on ramps in the Room, or stacks slide apart (V-RB-05, #89).
- A box on a tilting plank lags it, or a body sticks for a moment above its friction
  angle.
- A ball on a box does not roll off, or a box rolls like a ball.
- Spinning boxes stop visibly too fast, or keep spinning.
- Bounces look dead or too lively against the material's e.
- Spawned piles fly apart (the spawner, #93).
