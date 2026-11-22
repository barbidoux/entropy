# ADR 0031: Rigid contacts: sample spheres against exact distances, the static solids per body, pairs in parallel

- Status: accepted
- Date: 2026-11-22

## Context

SPEC 5.4 asks for a GPU broad phase, then a convex narrow phase decided by an ADR in M2,
with contacts solved as XPBD position constraints, one iteration per substep of at most
1/480 s (ADR 0030). M2's shapes are boxes, spheres and capsules (#86). The static world
is the map's solids: boxes, open boxes (the Test Tank) and half-spaces, whose exact
signed distance lived in `entropy.world.sdf`. Solvers may not import `world` (SPEC 8.1).

V-RB-04 requires collisions without gravity to keep momentum within 1e-4. On the GPU
the contacts are solved in parallel, so a body's corrections from several contacts are
summed, not applied one after the other.

The first version solved every contact the same way, in parallel. Its review (#101)
found boxes going through the Test Tank's 2 cm glass from 5 m/s, long capsules falling
through thin beams, rotations that no test checked, and limits that nothing named. That
review led to the decisions below.

## Decision

- **The static solids' distance maths moves to `fields`** (`entropy.fields.solids`), and
  `world` may import `fields` (SPEC 8.1 is updated). `world.sdf` keeps the shapes and
  builds their device arrays; the liquid and the rigid bodies query the same functions.
- **Broad phase: a hash grid of the bodies' centres** (Warp's `HashGrid`), rebuilt every
  substep. Each body has a bounding sphere, and a pair is kept when the spheres overlap.
  Each pair is met once.
- **Narrow phase: sample spheres against exact distances.** Every body carries sample
  spheres in its frame:
  - a box has 26 points of radius 0: its 8 corners, 12 edge midpoints and 6 face
    centres. Two equal boxes face to face put every corner and edge midpoint on the
    other's edges, at distance 0, so only the face centres sink in;
  - a sphere has its centre, with its radius;
  - a capsule has ⌈2h/r⌉ + 1 points along its axis, at most a radius apart and at most
    25, with its radius. Nothing thinner than about its radius slips between them.
  A body's samples are tested against the exact signed distance of the other body (box
  by Quilez's formula, sphere, capsule as a segment) in that body's frame. A sample that
  sinks in makes a contact: its depth, the outward normal of the surface it sank into,
  and its point on the sampled body. Each pair tests both bodies' samples, with three
  exceptions:
  - against a sphere, only the sphere's own sample is tested, which gives the exact
    contact. A box's face points against a curved surface would push sideways and tip a
    resting sphere off;
  - a box tests only its corners and edge midpoints against a capsule, because the
    capsule's samples cover the box's faces;
  - two boxes share one normal: the separating axis of least overlap, out of their
    3 + 3 face normals and the 9 cross products of their edges. An edge axis wins only
    under 95 % of the best face's overlap. A sample inside the other box sinks along
    that axis. With each box's own normals, a corner lying on the other's edge got a
    sideways normal, and a box resting on an equal one stayed tilted at 7°.
- **Each body against the static solids, by its own thread.** One Gauss–Seidel sweep of
  its samples, each correction applied before the next sample is tested.
  - A sample takes the surface nearest to where it was at the start of the substep.
    It sinks in when its place now passes that surface's tangent plane, so a thin wall
    keeps its inner normal however fast a body comes.
  - A box's samples go face centres first and corners last. A box lying flat is then
    lifted by its bottom face's centre, with no turn. Corners first tilted it a little
    every substep, and frictionless stacks slid off it within seconds.
  - Within the sweep the body turns to first order, and by the exact rotation once at
    the end. Composing the samples' small rotations added a second-order turn about the
    normal, and a box at rest started to spin.
  - The contacts gather into up to 3 patches by direction (normals within about 25°).
- **Between bodies, solved in parallel, keeping momentum.** Each contact gets the XPBD
  impulse that closes its depth, with compliance 0 and the generalised inverse masses
  of both bodies. It is scaled by 1 / (the larger contact count of its two bodies), and
  both bodies get it, with opposite signs. The corrections add up atomically and are
  applied to the positions and to the substep's increments, so the velocities come from
  them (ADR 0030). V-RB-04's linear momentum holds within 1e-4. The bodies' contacts
  with the static solids are solved after the pairs: what a pile pushes into the floor
  comes out within the same substep.
- **A velocity pass at contact patches** (Müller et al. 2020, §3.6), after the
  velocities are recovered.
  - A patch is the contacts of a pair of bodies, or of one body with the static solids
    along one direction. It acts once, at its centre of pressure (the depth-weighted
    mean of its points), along its mean normal.
  - The separating speed there becomes the one it had before the substep if it was
    separating, else 0. The depth corrections then never turn into speed: no bounce
    (restitution is #88's), and no pop: boxes spawned 5 cm into each other or into the
    floor are pushed out at under 1 mm/s (11 m/s before).
  - A body's static patches are solved by its own thread, one after the other. The
    pairs' patches are solved in parallel, scaled by 1 / (the larger patch count of the
    two bodies).
  - #88 puts friction and restitution at the same patches.
- **The angular velocity is the sum of the rotation vectors over h**, the prediction's
  and every correction's, not the logarithm of the turn quaternion (ADR 0030 recovered
  it from the turn). A correction I⁻¹(r × p) then changes I ω by exactly r × p / h.
  Corrections turn the orientation by the exact exponential of their rotation vector.
- **Detected every substep. Pair contacts go into a fixed capacity** of 32 per body by
  default; static contacts need no room. Contacts past the capacity are dropped and
  counted for each step (`contact_overflows`, on the HUD through the status read at the
  next call, never a wait). A restore clears the counts.
- **Quaternions keep a NaN.** Orientations are scaled to unit length by hand: Warp's
  `normalize` turns a NaN quaternion into the identity, which hid a NaN from the guard
  once corrections were applied.

## Named simplifications

These are named in SPEC 5.4 and in the module. The numbers were measured on CUDA on
2026-11-22, with 20 cm boxes unless stated.

- **Edge-on-edge contacts are sampled**, at edge midpoints and face centres, not found
  exactly: two boxes crossed edge to edge sink into each other until a corner, an edge
  midpoint or a face centre meets the other's surface.
- **A capsule's envelope sags between its samples** by (1 − cos 30°) r where it crosses
  an edge or another capsule midway between two samples: 6.7 mm at a 5 cm radius. A
  capsule longer than 24 radii has wider gaps.
- **Jacobi between bodies converges slowly under load.**
  - Stacks of 3, 5 and 10 boxes: the top box sinks 0.5, 1.7 and 7.7 mm (the first
    interface of the 10-stack by 1.5 mm; the review measured 1.5, 3.5 and 11.6 mm before).
  - A 64:1 slab on a crate overlaps it by 6 mm (13 mm before).
  - A box 100 times heavier on a 3-stack sinks 43 mm and jitters 41 mm.
  - Two boxes meeting a sphere from both sides at 1 m/s come back at 0.31 m/s: each of
    the sphere's two patches takes half.
  - V-RB-05 (#89) holds stacks to its bound.
- **Angular momentum is kept within 8e-4** of the sum of its terms' sizes in V-RB-04's
  scene. The depth corrections turn each body about where it is during the substep,
  not where it started it, an error of order h.
  - Turning about the start brought it to 9.5e-5. It was not kept, because it adds a
    torque of order h to bodies sliding over each other.
  - V-RB-04's 1e-4 is checked on the linear momentum (SPEC 9.4 now says so), the
    angular one within 1e-3. The physics auditor confirms this reading at the M2 gate.
- **A contact overflow loses support.** A body short of contacts sinks into the one
  under it, and which contacts are dropped depends on the order of the atomics on CUDA.
  32 per body leaves room: no overflow with 5,000 bodies piled in the Test Tank.
- **One large body coarsens the broad phase.** The grid's cell is twice the largest
  bound, so one 4 m plank puts every body in a few cells and the pair search tends to
  quadratic. That is #90's.
- **Thin walls hold through the static solve**: no box, tumbling or not, 10 or 20 cm,
  left the tank in 8 throws at each of 5, 8, 12, 20 and 30 m/s. Contacts between bodies
  use the bodies' current places. A box thrown at a free plank body held at every speed
  up to 30 m/s, even at 5 mm thick, thanks to the shared axis. A sphere or a capsule
  went through a plank 1 cm thick from 20 m/s, and 2 cm thick at 30 m/s; one 4 cm thick
  held.
- **No friction or restitution yet.** Contacts are inelastic along the normal and
  frictionless across it (#88).

## Measured cost

Unofficial. The GPU was idle at P8 before the run, with desktop programs only. Mains
power and power mode were not checked.

| Bodies piled in the Test Tank (contacts) | 1/30 s frame, 16 substeps |
| --- | --- |
| 5,000 (34,000) | 32.9 ms |
| 1,000 (5,600) | 16.0 ms |
| 10 | 5.2 ms |

The 10-body figure is about 0.3 ms fixed per substep, for 10 launches. SPEC 3's budget
for rigid bodies is 3 ms per 1/30 s frame, so this is about 11 times over. Bringing it
down is #90's: CUDA graphs, fewer launches, a hash grid rebuilt once per frame with a
margin.

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
- **The static contacts in the same parallel Jacobi as the pairs** (the first version).
  A face contact closed about 40 % of its depth per substep, so fast boxes went past the
  glass's mid-plane and out, and a box on a slope crept, which friction (#88) could not
  stop.
- **A velocity pass per contact** (the review's prototype). It stopped the escapes too,
  but per contact Jacobi leaves part of the speed (a box thrown at the glass came back at
  0.5 m/s); a patch is one exact impulse for a flat face.
- **Graph colouring, Gauss–Seidel on the GPU for the pairs.** Faster convergence for
  piles, but a colouring every substep; Jacobi with symmetric scaling keeps momentum and
  is simpler. Revisit with #89 if stacks need it.
- **Detection once per frame with a margin.** Cheaper, and #90 may adopt it; per
  substep is the simplest correct start.

## How we would know it is wrong

- Boxes crossing edge to edge visibly sink into each other in play.
- Piles in the Room jitter or sink under their own weight (V-RB-05, #89).
- Capsule piles look sunk into each other.
- Something thrown hard leaves the tank, or goes through a plank thicker than 2 cm.
- The rigid step at 5,000 bodies stays above 3 ms after #90's work.
