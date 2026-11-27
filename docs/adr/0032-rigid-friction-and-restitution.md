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
    60° of the vertical). Friction there, static and dynamic, turns neither body. The
    torque the shifted normal forces would pass to the lower body, (x_lower − x_upper) ×
    P, is left to the lower body's own support, as the static solids take a resting
    body's. Given each substep, the support answered only in the next one:
    - in static friction, the turn and the next sweep's levelling moved the anchor on
      the lower body's face by about 60 µm a substep against 17 µm of sliding, and a box
      resting on another on a slope 3° under its friction angle crept up to 4.5 mm in
      5 s on CUDA;
    - in dynamic friction, the residual sliding of resting stacks gave the lower bodies
      a turn each substep, and they crept apart (boxes 4 and 8 cm off-centre on others,
      the pyramid, the crate, centred stacks); kept to bodies sliding at μd N, the
      torque still made a box slide 6 % too fast on a broad box held on a slope;
    - its part about the normal, which no normal force can hold, was given to both
      bodies in both passes after #102's second review: the impulse was sized without
      the turn it gives, and a held patch's turn was never taken back, so resting
      bodies turned. A concrete cube resting on the end of a wooden board slid 153 mm
      and turned 90° in 10 s, a steel cube on a plank bridge moved 40 mm, and a pyramid
      turned 1.9° in 30 s (#102's third review).
    So the angular momentum is not kept (below).
  - Between bodies, static friction measures the sliding from the motion before the
    pair's own corrections: the turn its depth corrections give the bodies is no
    sliding. Cancelling it, friction flung two boxes spawned 1 cm into each other apart
    with 24 J and 18 rad/s (#102's second review), and fed a capsule resting on a box a
    vertical speed of 8 mm/s that did not move it; both are now gone.
  - On a patch that is not wide (a sphere, a capsule lying across the slope, a tall
    narrow box), friction acts at the centre of pressure and turns the body, so a ball
    rolls.
- **Static friction keeps an anchor on a wide patch** (`Anchors`), as PhysX's patch
  friction and Solver2D's "sticky" solver do:
  - The anchor is where static friction holds the resting body, kept from substep to
    substep while the patch lasts. The correction undoes the body's drift since the
    patch began, not only this substep's.
  - Between bodies, Jacobi's leftover is undone in later substeps, as the normal
    overlap's is. In the steady state both are amplified alike by the Jacobi share, so
    a pair holds up to μs. Without anchors, a box 1 cm off-centre on another crept
    59 mm in 10 s.
  - The anchor follows the resting body's centre, not its rocking, which the normal
    forces hold. Between bodies it sits on the lower body, which carries it as it moves
    and turns: the upper body's centre is kept in the lower body's frame. Held above a
    fixed point in the world instead, a box on a plank tilting to 20° slid 35 mm uphill,
    its height times the tilt (#102's review); it now rides the plank within 0.1 mm
    (0.7 mm turned 30° and 30 cm from the axis).
  - On a static solid the anchor is a fixed point, so the rounding of a resting body's
    float32 position is undone instead of summed. Without it, a box on a box 100 m down
    a slope 3° under the friction angle crept 1.9 mm in 5 s (#102's review), and now
    0.001 mm.
  - A patch that is not wide holds the point touching now, as Müller et al. do, and
    leaves its anchor as it is; a patch that turns wide again with its anchor left far
    behind lets go at once. A point of a ball's surface rolls away from the slope, and
    an anchor there dragged the ball back: it rolled at 2.15 m/s after 1 s where (5/7)
    g sin θ t is 2.40 m/s.
  - Identity:
    - a body's patches with the static solids find their anchors by normal, within
      about 11° (`ANCHOR_COS`);
    - a pair's anchor sits in its lower-indexed body's row of `PAIRS_PER_BODY` entries,
      found by the other body's index;
    - `detect` carries the anchors over from the last substep; a pair no longer in
      contact loses its anchor.
  - A new patch's anchor is set where the patch is, and so is the anchor of a patch that
    slides.
- **The bound is μs times the patch's normal impulse in the substep.**
  - Past it the patch breaks free and slides with Coulomb's kinetic friction: the
    correction is clamped to μd times the normal impulse, as impulse solvers clamp
    friction to its cone (PhysX, Box2D, Bullet), while the sliding left is within
    0.2 mm (`ANCHOR_SLIP_M`); the anchor keeps it, and on a pair undoes it when the bound
    allows. Past that, the friction is left to the velocity pass and the anchor is set
    again.
  - One substep's normal impulse dips and peaks as the Jacobi shares and the sinking
    samples change; the clamp holds a stack through the dips. Let go there instead, two
    boxes on a slope 1° under the friction angle slid 64 mm in 5 s.
  - Clamped at μs instead of μd, a body that broke free kept its static friction until
    it slid at about 0.1 m/s (0.2 mm a substep): with μs = 0.6 and μd = 0.4, a box
    pushed past its bound was 16 % too slow after 1 s (#102's review). Now it follows
    Coulomb within 0.1 %.
  - A clamped patch gets no dynamic friction in the velocity pass that substep: its
    friction is already Coulomb's, and is counted once.
- **The velocity pass**, after the velocities are recovered:
  - **Restitution:** the separating speed at the centre of pressure becomes e times the
    approaching speed before the substep, or the separating speed it had if the bodies
    were already separating. There is no bounce below 2|g|h, so resting bodies do not
    jitter. Between bodies, a bounce is shared only with the bodies' other bouncing
    patches (`count_bouncing`), not with all their pairs: in the next substep the pair
    counts as separating, so a share given then is never completed. Shared with all, a
    ball dropped 1 m with e = 0.8 onto a slab resting on another rose 0.020 m instead of
    0.64 m (#102's second review), and now 0.60 m; with no share at all, a box landing
    across two boxes gained energy and rose 2.02 m.
  - **Dynamic friction:** an impulse against the sliding there, of at most μd times the
    patch's normal impulse in the substep. That impulse is its depth corrections over h
    plus the restitution's impulse, so a contact met right at the surface still rubs.
    On a wide patch it acts as static friction does, without turning either body.
  - **Friction against turning about the normal** (torsion): at most μd times the
    normal impulse times the patch's torsion radius, the root-mean-square distance of
    its points from the centre of pressure. Without it, one point has no lever against
    a spin, and a box spun flat would never stop.
- **The pairs stay Jacobi with ADR 0031's 1 / n.** An over-relaxation, min(1, 1.8 / n)
  (Macklin et al. 2014, §4.2), was part of this change until its review: see the
  alternatives.
- **Anchors are state.** Checkpoints and dumps carry them, so a replay is exact; new
  bodies start with none, and a checkpoint without anchors restores none.

## Measured

On CUDA, 2026-11-27, except where noted.

| Case | Result | Expected |
| --- | --- | --- |
| A box sliding at 2 m/s, μ = 0.3 | stops after 0.681 m, untilted | v²/2μg = 0.680 m |
| A box sliding at 2 m/s, μ = 0.6 | stops after 0.342 m, untilted | 0.340 m |
| V-RB-02: a sphere or a box dropped flat from 1 m, e = 0.5 | rises 0.246 m (−1.7 %) | e² = 0.25 m |
| V-RB-02: the same, e = 0.8 | rises 0.632 m (−1.3 %) | e² = 0.64 m |
| V-RB-03: a block, μ = 0.5, 1° below atan μ | does not move in 1 s | holds |
| V-RB-03: the same, 1° above | accelerates at 0.1912 m/s² | 0.1914 m/s² |
| V-RB-03: the same, 3° above | accelerates at 0.5740 m/s² | 0.5740 m/s² |
| A box pushed by a force rising 40 N/s, μs = 0.6, μd = 0.4 | breaks free at 113.1 N, 3.306 m/s at 3.9 s | 113.0 N, 3.304 m/s |
| A block 1° past its static angle, μs = 0.6, μd = 0.4 | slides at g (sin θ − μd cos θ) | the test, within 3 % |
| A small box on a broad one held on a slope 1° past the pair's angle | 0.1904 m/s², the lower one 0.002 mm | 0.1914 m/s² |
| A sphere on a 20° slope, μ = 0.6 | rolls at 2.397 m/s after 1 s, spin = v/r | (5/7) g sin θ t = 2.397 m/s |
| A box spun flat at 10 rad/s, μ = 0.5 | stops in 0.25 s | 0.36 s under an even pressure |
| A 20 cm box on a plank tilting to 20° at 0.3 rad/s, μ = 0.6 | moves 0.09 mm on the plank; 0.74 mm turned 30° and 30 cm from the axis | rides it |
| The same plank tilting on to 40° | the box slides from 32.9° | 31° |
| A box kicked to 0.2 m/s on the floor, μ = 0.5 | slides 4.10 mm | v²/2μg = 4.08 mm |
| A ball dropped 1 m, e = 0.8, onto the floor; a slab on the floor; a slab on a slab | rises 0.632; 0.615; 0.596 m | e² = 0.64 m |
| Two 20 cm boxes spawned 1 to 6 cm into each other, turned up to 45°, μ = 0.5, no gravity | 0.00 J after one frame | 0.00 J, as frictionless |
| μ = 0.5, each body's largest move over 10 s after 1 s to settle: | | |
| a box 1, 4 or 8 cm off-centre on an equal one | 0.000 mm | under 1 mm (the tests) |
| a 4-3-2-1 pyramid; a 6-row one | 0.013 to 0.084 mm, unturned; 0.25 to 0.98 mm, turned up to 0.17° | under 1 mm, 0.1° (the test: 4-3-2-1) |
| a concrete cube 0.85 m out on a 2 m wooden board; a steel cube 0.6 m out on a plank bridge | 0.003 mm; 0.16 mm; unturned | under 1 mm and 0.1° (the tests) |
| a crate under a slab 64 times heavier | 0.37 mm | under 1 mm |
| four boxes stacked as stairs, each 3 cm past the one under it | 0.7 to 1.0 mm | |
| boxes on a slope (5 s, 4 starts): 2 at 3°, 1° and 0.5° under atan μ; 3 at 1°; 5 at 3° | at most 0.087 mm | under 1 mm |
| a box on a box 3° under atan μ, 5 to 100 m down the slope (5 s) | at most 0.008 mm | under 1 mm |

Between bodies, ADR 0031's numbers stand: stacks sink 0.5, 1.7 and 7.7 mm for 3, 5 and
10 boxes; 27 boxes spawned 5 cm into each other fly apart at 12.2 m/s; V-RB-04 keeps the
angular momentum within 8.7e-4 of its terms (seed 4; 1.35e-3 at worst over seeds 0 to 9)
and the linear within 1.9e-7. Run to run on CUDA, the stacks' small moves vary (the
pyramid 0.013 to 0.084 mm, the 6-row one 0.25 to 0.98 mm).

## Named simplifications

These are named in SPEC 5.4 and in the module.

- **One surface material for all the static solids** (`RigidSetup`), until maps give
  their surfaces materials.
- **Friction between bodies on a wide patch turns neither body** (above). A lower body
  that cannot hold the tipping torque, a ball under a box, misses it. In free fall,
  where nothing supports either, a box sliding onto another at 2 m/s, μ = 0.5, loses
  29 % of their angular momentum about their centre of mass in 0.2 s (1 % with no
  gravity, where no body is on top). Its torque about the normal is lost too: a box
  sliding at 2 m/s across a broad box on a frictionless floor, 25 cm off its centre
  line, leaves it unturned, and 89 % of their angular momentum about the vertical is
  lost in 0.5 s. A torsional anchor, static friction holding the pair's relative turn
  as it holds their sliding, would be the way to give it without setting resting
  bodies turning.
- **Bounces on several bodies at once share one bounce**, and the velocity pass does not
  know a body's support. A plank landing flat across two blocks rises 0.03 m where e² =
  0.64 m, a box landing across two boxes 0.27 m, and a ball on a 20 cm box resting on a
  slab 0.23 m, the box being pushed into the slab. #90's graph colouring (Gauss–Seidel
  between pairs) is the way.
- **Friction can give a little energy in fast hits between bodies.** Its position
  correction undoes the sliding the bodies' motion would have made, which the depth
  corrections may already have turned away: over 60 random hits of two boxes at 2 to
  8 m/s, e = 1 and μ = 0.5, the worst ends with 1.075 times its energy (frictionless
  1.000); most lose some. A dense box pushed off a plank's edge onto a frictionless
  floor gains up to 13 % of its kinetic energy for a moment. Real materials have e < 1.
- **A patch that is not wide has no anchor.** Its static friction undoes one substep's
  sliding, so Jacobi's leftover is lost there, and between two narrow patches one
  Gauss–Seidel pass leaves a little sliding each substep. A plank leaning on a wall
  (both ends on edges), μ = 0.5, holds in statics up to 53° from the vertical; it
  creeps 11, 26 and 47 mm in 20 s at 30, 40 and 45°, and slides off at 50°. So is a
  pair past its body's row of `PAIRS_PER_BODY` anchors.
- **No rolling resistance, and no torsion at one point.** A ball rolled on the floor
  keeps its speed (0.465 m/s, 5 m in 10 s), and a sphere knocked out of a stack rolls
  on; with spheres in a pile, the spheres keep moving after the boxes stop. A sphere
  spun about the vertical on the floor keeps spinning: its single contact gives the
  torsion no lever.
- **A round body resting off-centre on another drifts.** ADR 0031's off-centre load
  tilts the box under it by 1 to 3 µrad, so a capsule lying 5 cm off-centre on a box
  rolls 3.9 mm in 60 s, even with friction.
- **A gentle push springs back a little.** The anchor keeps the sliding left within the
  0.2 mm window and undoes it later: a box kicked to 0.1 m/s slides 0.96 mm and settles
  back to 0.93 mm, where v²/2μg is 1.02 mm.
- **A dynamic friction above the static one** is taken as given: past μs the patch's
  friction is clamped to μd, so it holds up to μd. Materials have μd ≤ μs.
- **Far from the origin, contacts see float32's resolution** (ADR 0030, 0031): depths and
  slips are rounded to about 1 µm at 10 m and 8 µm at 100 m, against 0.8 µm of a
  substep's sliding 1° past the friction angle. A block 1° above it slides 9 % short
  over 1 s at 10 m (V-RB-03 still holds), half as far at 20 m and not at all at 100 m;
  below it, it holds everywhere. The Street (M8) will need positions relative to a
  local origin.
- **Piles still move.** A pile of 100 random boxes on the floor has 4 to 17 of them
  moving over 1 mm between 15 and 25 s (seeds 0 to 4, run to run on CUDA), up to
  0.15 m; sleeping (#89) stops them.
- **Torsion is too strong on box faces.** A box's samples weigh its corners and edges
  more than an even pressure would. The torsion radius comes out about 1.4 times too
  large, so a spinning box stops about 1.4 times too soon. Torsion has dynamic friction
  only, with no static hold against turning.
- **The bounce is 1.3 to 1.7 % low**, within V-RB-02's 3 %. The target takes the
  approach speed at the start of the landing substep, which misses that substep's
  gravity: up to e g h, about 1 % of the height here.
- **The tipping test is a threshold.** A patch just under the width tips the body by its
  full lever, and one just over it does not tip it at all.

## Alternatives considered

Measured on the CPU in the tests' scenes, except where noted.

- **Müller's per-contact friction, in parallel.** It is the paper's rule. Under Jacobi
  it leaves part of the sliding each substep, and the block crept at 5 mm/s where it
  should hold.
- **Per-patch friction without anchors.** It held single bodies (V-RB-03), not stacks.
  A box 1 cm off-centre on another crept 59 mm in 10 s, and two boxes on a slope 3°
  under the friction angle slid 0.80 m in 5 s.
- **No anchors on the static solids** (the pairs keep theirs). Every scene above holds
  near the origin, but a box on a box 100 m down a slope crept 1.6 to 1.9 mm in 5 s,
  and with the sliding measured from the exact increments, 2.5 mm. A pile does not
  settle better without them: over five seeds, 2 to 17 boxes moved in either case.
- **A support rule** (shock propagation: the lower body counts as immovable for the
  upper one's friction). Dense piles exploded: bodies flew apart and fell through the
  floor, in each variant tried (per contact, one level only). It was removed.
- **Friction let go at the first substep past the bound** (Müller's all-or-nothing
  rule). The dips of the per-substep normal impulse broke it: two boxes on a slope 1°
  under the friction angle slid 64 mm in 5 s, on the CPU as on CUDA.
- **Turning the bodies on a wide patch, in static or dynamic friction** (above). It
  keeps the angular momentum in free fall but not a supported stack at rest. Its part
  about the normal alone set resting bodies turning (above). Sizing the impulse with
  the turn it gives fixed the cube on the board but not the plank bridge (40 mm) or the
  pyramids (2.3°); also leaving the turn out on patches an anchor holds fixed those, at
  the cost of a held pair's turn together (#102's third review). A torsional anchor
  would be the principled way; it is not tried.
- **A pair's sliding measured with its own corrections** (Müller et al.'s current
  positions). Bodies pushed out of an overlap were flung apart (above).
- **A bounce shared with all of a body's pairs** (1 / n, as the depths), **or with none**:
  dead bounces on stacks, or energy gained by bodies hit twice at once (above).
- **Anchors on every patch.** A ball's anchor, a point of its surface, rolled away and
  dragged it back: 10 % too slow down a slope.
- **Clamping without skipping dynamic friction.** It counted friction twice: V-RB-03's
  block 1° above the friction angle crept 0.44 mm in 1 s instead of sliding 10 cm, and
  a box on a held box 1° past the pair's angle did not slide at all.
- **Clamping at μs.** See above: kinetic friction began only at about 0.1 m/s.
- **The bound on a normal impulse averaged over about 8 substeps**, and **the pairs'
  over-relaxation, min(1, 1.8 / n)** (Macklin et al. 2014, "Unified particle physics for
  real-time applications", §4.2). Both came in while a float32 difference of positions
  still biased the corrections: then, with 1 / n, the pyramid's top crept 29 mm in 10 s,
  and an unaveraged bound let the pair of boxes on a slope slide. Once #101's reviews
  made every shift an exact sum, neither helped (#102's review), and both harmed (CUDA,
  over 4 starts):
  - three boxes on a slope 1° under the friction angle slid 3.7 m from one start with
    both, and held within 0.002 mm without;
  - a 6-row pyramid moved 6.1 mm in 10 s with both, 0.28 mm without;
  - the over-relaxation made 27 overlapping boxes pop at 13 to 16 m/s instead of 10 to
    12, and 12 spheres of 10 cm in a clump at 24.6 instead of 19.2 m/s;
  - in return it sank a 10-box stack 4.3 mm instead of 7.7 mm.
  #90's graph colouring is the way to stiffer stacks.
- **Several passes per substep.** Before the exact sums, they converged on the pyramid:
  its top crept 29, 9.5, 0.69 and 0.25 mm in 10 s with 1, 2, 4 and 8 passes. Each pass
  costs a full contact solve, with SPEC 3's budget already exceeded (#90).
- **A per-contact velocity pass in the body's thread.** It was tried for the normal
  velocities: one sweep left normal speeds into the surface and made a resting capsule
  jitter.

## How we would know it is wrong

- Boxes creep on ramps in the Room, or stacks slide apart (V-RB-05, #89).
- A box on a tilting plank lags it or slides on it, or a body sticks for a moment above
  its friction angle.
- A pushed box starts sliding late, or too slowly once it does.
- A ball on a box does not roll off, or a box rolls like a ball.
- Spinning boxes stop visibly too fast, or keep spinning.
- Bounces look dead or too lively against the material's e, on the floor or on a stack.
- Spawned piles fly apart (the spawner, #93), or settled bodies keep trembling.
