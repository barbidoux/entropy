# ADR 0031: Rigid contacts: sample spheres against exact distances, solved per body and per pair

- Status: accepted
- Date: 2026-11-22
- Supersedes ADR 0030's recovery of the angular velocity from the logarithm of the turn:
  it is now the sum of the substep's rotation vectors (below).

## Context

SPEC 5.4 asks for a GPU broad phase, then a convex narrow phase decided by an ADR in M2,
with contacts solved as XPBD position constraints, one iteration per substep of at most
1/480 s (ADR 0030). M2's shapes are boxes, spheres and capsules (#86). The static world
is the map's solids: boxes, open boxes (the Test Tank) and half-spaces, whose exact
signed distance lived in `entropy.world.sdf`. Solvers may not import `world` (SPEC 8.1).

V-RB-04 requires collisions without gravity to keep momentum within 1e-4. On the GPU
the contacts are solved in parallel, so a body's corrections from several contacts are
summed, not applied one after the other.

The first version solved every contact the same way, in parallel. Three reviews of #101
found the following, and they led to the decisions below:
- boxes going through the Test Tank's 2 cm glass;
- long capsules falling through thin beams;
- untested rotations;
- boxes toppling off table edges;
- nine boxes on one slab jittering;
- a dump that did not replay;
- ghost walls at the seam of two flush static boxes;
- phantom contacts beside static box edges;
- limits that nothing named, or named wrongly.

## Decision

- **The static solids' distance maths moves to `fields`** (`entropy.fields.solids`), and
  `world` may import `fields` (SPEC 8.1 is updated). `world.sdf` keeps the shapes and
  builds their device arrays; the liquid and the rigid bodies query the same functions.
- **Sample spheres.** Every body carries sample spheres in its frame:
  - a box has 26 points of radius 0: its 8 corners, 12 edge midpoints and 6 face
    centres. Two equal boxes face to face put every corner and edge midpoint on the
    other's edges, at distance 0, so only the face centres sink in;
  - a sphere has its centre, with its radius;
  - a capsule has an odd number of points along its axis, at most a radius apart and at
    most 25, with its radius. One point sits at its middle, and nothing thinner than
    about its radius slips between two points.
- **A sweep's order** (`_order`): a box's face centres first and its corners last; a
  capsule's middle first, then outwards. A body lying flat is then lifted whole by its
  first contact, through its centre of mass, with no turn. Corners first tilted a box a
  little every substep, and stacks slid off it. Ends first set parallel capsules
  spinning.
- **Each body against the static solids, by its own thread** (`solve_static`): one
  Gauss–Seidel sweep, each correction applied before the next is tested.
  - Each sample is tested against every static plane (`_plane`), taken from where it
    started the substep:
    - a half-space is its plane;
    - a box gives the tangent plane of its surface nearest to that start, but only when
      the sample's path this substep meets the box grown by the sample's radius: near an
      edge, the diagonal tangent plane stopped bodies passing beside the box (a box
      falling 1 mm beside a table was kicked sideways at 1.4 m/s);
    - a box's faces covered by another static solid are left out of that surface, an
      inner face. Outside the box they drop out of the nearest feature's normal, so the
      shared top edge of two flush slabs gives the top's normal; inside it, the nearest
      uncovered face counts. Jolt and Bullet remove internal edges and Box2D's chains
      use ghost vertices for the same reason: a box sliding across the seam of two slabs
      bounced back;
    - a tank gives, from inside its cavity's column, its floor's and four walls' inner
      planes, each on its own; from outside, the tangent plane of its nearest surface.
  - A sample sinks in when its place now passes the plane. A thin wall therefore keeps
    its inner normal however fast a body comes, and a sample near an inner edge or
    corner meets every wall there.
  - Then the edges of the static boxes near the body are tested against the body's
    exact distance, over the body's own extent along the edge, at points no farther
    apart than half the body's smallest size (at most 16 per edge). A table's edge
    under a box's face holds the box: from samples alone it stood on the row under its
    centre of mass and toppled off.
  - The body turns to first order within the sweep, and by the exact rotation once at
    the end. Composing small rotations one by one added a turn about the normal, and a
    box at rest started to spin.
  - The contacts gather into up to 3 patches by direction (normals within about 25°).
- **Between bodies, each pair by its own thread** (`solve_pairs`).
  - A hash grid of the bodies' centres (Warp's `HashGrid`), rebuilt every substep,
    finds the neighbours. A pair whose samples touch gets a patch, and each pair is met
    once.
  - The pair's thread runs a Gauss–Seidel sweep of a's samples against b's exact
    distance, then b's against a's, on its own copies of the two bodies. Each
    correction is the XPBD impulse that closes the depth, given to both with opposite
    signs.
  - The pair's net corrections are scaled by 1 / (the larger pair count of the two
    bodies), Jacobi between pairs, and added atomically. Linear momentum is kept (V-RB-04).
    The scale keeps a pile's corrections from adding up: unscaled, a 400-body pile in
    the Test Tank crept 25 cm in 10 s and gained up to 83 J in one second. It slows
    stacks: unscaled, 10 stacked boxes sink 3.8 mm instead of 7.7 mm (#101's fourth
    review). No test in the suite is a pile that large: the soak test at the M2 gate
    guards it.
  - The first version solved contact by contact in parallel. There, nine boxes on one
    slab jittered and sank, and the step cost half as much again.
  - Against a sphere, only the sphere's own sample is tested: it gives the exact contact,
    and a box's face points against a curved surface would push sideways and tip a
    resting sphere off. A box tests only its corners and edge midpoints against a
    capsule, whose samples cover the box's faces.
  - Two boxes share one normal: the separating axis of least overlap, out of their
    3 + 3 face normals and the 9 cross products of their edges. An edge axis wins only
    under 95 % of the best face's overlap. Each box's own normals gave a corner lying on
    the other's edge a sideways normal, and a box resting on an equal one stayed tilted
    at 7°. The depth stays the exact distance.
- **A velocity pass at the contact patches** (Müller et al. 2020, §3.6), after the
  velocities are recovered. A patch is a pair of bodies, or one body against the static
  solids along one direction.
  - It acts once, at the patch's centre of pressure (the depth-weighted mean of its
    points), along its mean normal.
  - The separating speed there becomes the one it had before the substep if it was
    separating, else 0. For one pair alone, or one body against the static solids, the
    depth corrections then do not turn into speed along the normal: no bounce, and no
    pop of two bodies spawned overlapping (11 m/s before). Between several bodies at
    once this pass is Jacobi and leaves part of that speed (named below).
  - One point cannot take back the spin that a deep correction at a body's edge gives
    it. So every body touching the static solids then has its speed scaled down, if need
    be, to the kinetic energy its motion had before their corrections: free motion,
    gravity and the pairs' corrections (taken after `apply_corrections`). The static
    solids never give energy.
    - Without the rule, a box spawned 30 to 50 cm inside a table or a wall was flung at
      57 to 211 m/s and up to 1,058 rad/s, and a plank thrown at the floor at 30 m/s
      rebounded with up to 2.5 times its energy (#101's fourth review).
    - With a budget of the motion without contact, as first tried, a body in a pair had
      to be left out, or a box hitting a resting one stopped at half its speed; a box
      spawned in a table beside another then flew at 135 to 163 m/s (fifth review).
  - The sweep's shift, and each pair's, enter the bodies' displacements as the sums of
    their corrections, exactly, as ADR 0030 keeps every increment. Taken as float32
    differences of positions, they were biased on any tilted contact:
    - a box sliding down a frictionless 30° slope lost 0.7 % of its speed in 5 s, and a
      sphere sliding 100 m down a slope gained 2.2 to 2.8 % in 10 s, which the energy
      rule did not stop (sixth review);
    - two boxes stacked on a frictionless 30° slope, gravity pressing them onto it,
      parted by 5.9 mm in 10 s on CUDA and 49 mm on the CPU at the origin, and 10 m down
      the slope the upper one slid off (seventh review).
  - A body's static patches are solved by its own thread, one after the other. The
    pairs' patches are solved in parallel, scaled by 1 / (the larger pair count).
  - #88 puts friction and restitution at the same patches.
- **The angular velocity is the sum of the substep's rotation vectors over h**: the
  prediction's and every correction's. A correction I⁻¹(r × p) then changes I ω by
  exactly r × p / h. Corrections turn the orientation by the exact exponential of their
  rotation vector.
- **Fast spins: the gyroscopic step is split** (ADR 0030 solved it over the whole
  substep). A hard hit spins a thin body to thousands of rad/s. The implicit midpoint's
  fixed-point iterations converge only while |ω| h times the spread of the moments (the
  largest over the smallest) is small. Over the whole substep, a plank of 8 × 2 × 0.5 cm
  (moments 16 to 1) lost 41 % of 600 rad/s in 1 s and went non-finite from about
  1,500 rad/s, freezing the solver in 11 of 20 throws at the floor at 30 m/s.
  - The substep is now split into pieces of at most 0.2 of that measure, up to 256:
    the plank keeps |L| and its energy within 4e-4 up to 12,000 rad/s, and loses 7 % of
    its energy a second at 30,000 rad/s (no NaN at 100,000 rad/s). The extra cost is not
    measurable with 2,000 tumbling bodies on CUDA (fifth review).
  - The energy is then capped at its start, so a spin never gains energy.
- **Room for 16 pairs per body** by default (`max_pairs`). Contacts with the static
  solids need no room. Pairs past it are dropped and counted per step
  (`pair_overflows` in the diagnostics, for the HUD), through the status read at the
  next call, never a wait. A restore clears the counts.
- **A NaN dump holds the setup and the static solids.** `RigidSolver.from_dump` rebuilds
  the solver at the frame's start. Stepping the dump's time step replays it bit for bit
  on the CPU, contacts included (rule 8).
- **Quaternions keep a NaN.** Orientations are scaled to unit length by hand: Warp's
  `normalize` turns a NaN quaternion into the identity, which hid a NaN from the guard
  once corrections were applied. Since the angular velocity comes from the rotation
  vectors, a NaN orientation also reaches it, so this is belt and braces.

## Named simplifications

These are named in SPEC 5.4 and in the module. The numbers were measured on CUDA on
2026-11-22, and on 2026-11-27 for the fourth review's, with 20 cm boxes unless stated.

- **Edge-on-edge contacts between bodies are sampled**, at edge midpoints and face
  centres, not found exactly: two boxes crossed edge to edge sink into each other until
  a corner, an edge midpoint or a face centre meets the other's surface.
- **A capsule's envelope sags between its samples** by (1 − cos 30°) r, where it crosses
  an edge or another capsule midway between two samples: 6.7 mm at a 5 cm radius. A
  capsule longer than 24 radii has wider gaps.
- **Fast bodies can pass through a thin one.** A box hitting a thin body where it has
  no sample can cross it once one substep's relative travel exceeds about the thin
  body's thickness: from about 480 s⁻¹ × thickness, 2.4 m/s for 5 mm and 9.6 m/s for
  2 cm. A sphere needs about half its size plus half the thickness. In 20 throws each:
  - a 10 cm box aimed 20 cm off a 5 mm plank's centre, face on, went through in 10 at
    4 m/s, 14 at 6 and 10 m/s and all 20 at 30 m/s; at random orientations, 4 at 10 m/s
    and 8 at 30 m/s; aimed at the centre, where the plank has a sample, 3 at 30 m/s;
  - a 2 cm box went through the 5 mm plank in 8 at 10 m/s;
  - a 4 cm plank stopped every 10 cm box up to 30 m/s.
- **The Test Tank keeps 10 and 20 cm boxes in up to 30 m/s.** The reviews found no
  escape in 1,440 throws of every shape. Boxes of 1 to 4 cm stay in up to 20 m/s. At
  30 m/s, 1 to 3 throws in 40 aimed at the floor-to-wall edge get out: the 2 and 4 cm
  boxes bounce up to 2 m and leave over the rim, and a 1 cm one leaks through the edge.
  A 2 cm box also went through a 10 cm static wall in 1 throw of 20 at 30 m/s.
- **Without friction, bodies resting off-centre slide off.** The body under the load
  tilts by a few hundredths of a degree within a substep, and nothing holds the one
  above on that slope:
  - a box offset by 1, 2, 4 or 8 cm on an equal box falls at 5.4, 4.3, 3.1 and 1.6 s;
  - a 4-3-2-1 pyramid collapses;
  - a crate under a slab 64 times heavier is squeezed out after about 2 s.
  Friction (#88) must hold all three; its tests include them.
- **Without friction a stack is unstable, and off the origin most centred stacks fall.**
  The top box of five set off by 10 µm brings the stack down in 7 to 8 s, before and
  after the pairs' exact shift, at the origin as at 1.35 m: the off-centre slide above.
  Off the origin, the contacts' float32 positions (the samples' centres and the lever
  arms) tilt the boxes by 1e-7 to 1e-6 rad, the tilts line up and grow 3 to 4 times a
  second, and seed the same fall; at the origin a centred stack does not move. Of nine
  centred stacks at random yaws over the Test Tank's floor, on CUDA in 60 s, 5 of 9
  five-box stacks fell (at 12 to 54 s), 2 of 9 three-box stacks and 1 of 9 two-box
  stacks (eighth and ninth reviews).
  - Before the pairs' exact shift, a float32 difference of positions rounded those
    pushes away where the stack stood off the origin (at 1.35 m it froze even a 1 mm
    offset for 60 s), so those stacks held for 90 s; it did not at the origin, and it
    biased every tilted contact (the seventh review's slope stack, below).
  - With friction (#88, μ = 0.5) none of the nine five-box stacks moves 1 mm in 60 s,
    nor do ten-box stacks with μ = 0.1.
  - On a frictionless slope, gravity along its normal, two stacked boxes part the same
    way, faster as they go: at the origin on CUDA the upper one slides off between 24
    and 28 s; after 10 s it is off by 1.4 to 1.5 mm 2 m down the slope, 12 to 17 mm 5 m
    down, and 100 m down the stack falls apart.
- **Far from the origin, contacts see positions to float32's resolution** (8 µm at
  100 m), so contact speeds there carry that rounding: a body's speed stays exact in
  free flight only (ADR 0030 said so of every body before contacts). The sums of the
  corrections themselves are exact.
- **Contacts between bodies converge slowly under load.**
  - Stacks of 3, 5 and 10 boxes: the top box sinks 0.5, 1.7 and 7.7 mm (the 10-stack's
    first interface 1.5 mm).
  - A body much heavier than the one under it sinks into it by about the mass ratio
    times g h²: 2.8 mm at 64:1. A box 100 times heavier on a 3-stack sinks 43 mm and
    jitters 41 mm (peak to peak, from 2 to 5 s).
  - Two 20 cm boxes meeting a 20 cm sphere from both sides at 1 m/s come back at
    0.31 m/s: each of the sphere's two patches takes half.
  - V-RB-05 (#89) holds stacks to its bound.
- **Several bodies at once still pop and bounce.** The velocity pass between pairs is
  one Jacobi sweep, so it leaves part of the speed the depth corrections make:
  - bodies spawned overlapping fly apart: 12 spheres of 10 cm radius in a 30 cm clump
    on the floor at up to 19 m/s, still 19 m/s after 2 s; 27 boxes in a 3 x 3 x 3
    block, 5 cm into each other, at 9.8 m/s (two alone: none); two boxes spawned 50 cm
    into a table and into each other, at 4.6, 6.2, 12.7, 23.4 and 35 m/s for 1, 2, 5, 10
    and 15 cm of overlap, since the static solids' budget holds the pair's push apart;
  - a tumbling 20 cm box dropped 0.5 m onto an equal one resting on the floor: in 2 of
    40 throws the two gain more than 5 % of their energy, up to 43 %, and the lower box
    leaves at up to 1.8 m/s (the pair's corrections, with or without the energy rule);
  - a thin plank (8 × 2 × 0.5 cm) thrown at 20 to 30 m/s onto a small body lying on the
    floor gains up to 47 % of its energy in 1 to 5 throws of 40, from the pair's
    corrections, which the static solids' rule cannot tell from a real push;
  - a box dropped 0.5 m onto another rebounds at 0.35 m/s, onto a stack of two at
    0.95 m/s, with no restitution;
  - a box thrown at 4 m/s at a row of 3 leaves them at 0.74, 1.28, 1.65 and 0.33 m/s,
    where an inelastic shock would give 1.0 each;
  - a pile never settles: 400 mixed bodies dropped into the Test Tank keep 4 to 43 J of
    kinetic energy over 60 s, gain up to 32 to 38 J in some one-second windows (the GPU
    runs differ), spin at up
    to 153 rad/s, and the median body moves 21 cm in the last 10 s; one 4 cm box left
    the tank in the first 2 s. Without friction nothing stops them; friction (#88) and
    sleeping (#89) settle piles.
  The spawner (#93) places bodies apart. Contacts kept from one substep to the next,
  with warm-started impulses, which #88's friction anchors begin, and the graph
  colouring of #90 are what Box2D and PhysX use against this.
- **Angular momentum.** In V-RB-04's scene (seed 4) it is kept within 8.7e-4 of the sum
  of the bodies' terms' sizes (|x × m v| + |I ω|), 5.8e-3 of the total's size. Seeds 0
  to 9 give 1.5e-4 to 1.35e-3 (seed 8) of the terms' sizes. Linear momentum is kept
  within 1.8e-7 (1.4e-5 before the pairs' shifts were exact sums).
  - The depth corrections turn each body about where it is during the substep, not where
    it started it. Turning about the start cancels the error, but it adds a torque to
    bodies sliding over each other, so it was not kept.
  - SPEC 9.4's V-RB-04 says momentum. The test checks linear momentum within its 1e-4,
    and angular momentum within 2e-3, which is provisional and covers the ten seeds. The physics auditor decides
    at the M2 gate whether V-RB-04 covers angular momentum.
- **A pair overflow loses support.** A body short of a pair sinks through the other, and
  which pairs are dropped depends on the order of the atomics on CUDA. 16 per body
  leaves room: no overflow with 5,000 bodies piled in the Test Tank.
- **One large body coarsens the broad phase.** The grid's cell is twice the largest
  bound, so one 4 m plank puts every body in a few cells and the pair search tends to
  quadratic. Each sample also scans every static primitive. Both are #90's, before the
  house map.
- **The tank's rims are not sampled as edges.** A body resting across the top of the
  glass is held by its own samples only.
- **No friction or restitution yet.** Contacts are inelastic along the normal and
  frictionless across it (#88).

## Measured cost

Unofficial. The GPU was at P8 and 2 % before the run, with desktop programs only. Mains
power and power mode were not checked.

| Bodies piled in the Test Tank (contacts) | 1/30 s frame, 16 substeps |
| --- | --- |
| 5,000 (28,000) | 20.7 ms |
| 1,000 (5,500) | 11.5 ms |
| 10 | 5.1 ms |

The contact counts depend on how the pile is made: these are from the cost probe's pile
of mixed shapes dropped in rows. The 10-body figure is about 0.3 ms fixed per substep.
SPEC 3's budget for rigid bodies is 3 ms per 1/30 s frame, about 7 times under this. Bringing it down is #90's: CUDA
graphs, fewer launches, a hash grid rebuilt once per frame with a margin.

## Alternatives considered

- **GJK and EPA for every pair.** General for any convex shape, including the convex
  hulls of fracture fragments later, but iterative per pair, divergent on the GPU, and
  one contact point per call: resting boxes need a manifold. Revisit for M4's
  fragments.
- **Analytic pairs with a box–box clipping manifold** (ODE's `dBoxBox`). Exact
  edge–edge contacts and four-point manifolds, but six pair routines to write and test,
  and box–box clipping is long. Sampling with a shared axis covers M2's shapes with one
  routine; revisit if piles show the edge–edge limit.
- **A sampled grid of the static solids** (as the liquid uses). Its 6.25 cm cells cannot
  hold a box on the floor to the millimetre; the exact distances can.
- **Every contact in one parallel Jacobi solve** (the first version). A face contact
  closed about 40 % of its depth per substep, so fast boxes went past the glass's
  mid-plane and out; piles on a slab jittered; it cost more.
- **The static solids swept twice per substep**, before and after the pairs. It moves a
  heavy body's sink from the light body under it into the floor, but does not shrink
  it, for a second sweep's cost.
- **A depenetration cap** (Box2D's contact push speed, PhysX's maximum depenetration
  velocity, 3 m/s) **and a split push** (Bullet's split impulse: the old overlap pushed
  out without speed). They brought a spawned clump's pop to about 3 m/s, but without
  warm starting a box 100 times heavier on a stack could no longer push its steady
  43 mm overlap out fast enough and fell through (and pushed out without speed, a
  stack of 10 sank 0.4 m). They come back once contacts are kept from one substep to
  the next.
- **A velocity pass per contact** (the first review's prototype). It stopped the escapes
  too, but Jacobi per contact leaves part of the speed (a box thrown at the glass came
  back at 0.5 m/s); a patch is one exact impulse for a flat face.
- **Graph colouring, Gauss–Seidel on the GPU across pairs.** Faster convergence for
  piles, but a colouring every substep. Revisit with #89 if stacks need it.
- **Detection once per frame with a margin.** Cheaper, and #90 may adopt it; per
  substep is the simplest correct start.

## How we would know it is wrong

- Boxes crossing edge to edge visibly sink into each other in play.
- Piles in the Room jitter or sink under their own weight (V-RB-05, #89).
- With friction, boxes still slide off each other or off edges (#88).
- Capsule piles look sunk into each other.
- Something thrown in play goes through a wall or a plank.
- Bodies sliding along a map built of boxes catch on its seams or edges.
- The rigid step at 5,000 bodies stays above 3 ms after #90's work.
