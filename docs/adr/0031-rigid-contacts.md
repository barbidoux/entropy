# ADR 0031: Rigid contacts: sample spheres against exact distances, solved in parallel

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
  - a capsule has 5 points along its axis, at −h, −h/2, 0, h/2 and h, with its radius.

  A body's samples are tested against the exact signed distance of the other body (box
  by Quilez's formula, sphere, capsule as a segment) in that body's frame, and against
  the static solids. A sample that sinks in makes a contact: its depth, the outward
  normal of the surface it sank into, and its point on the sampled body. Each pair tests
  both bodies' samples against each other.
- **Solved in parallel, keeping momentum.** Each contact gets the XPBD impulse that
  closes its depth, with compliance 0 and the generalised inverse masses of both bodies.
  It is scaled by 1 / (the larger contact count of its two bodies), and both bodies get
  it, with opposite signs. The corrections add up atomically and are applied to the
  positions and to the substep's increments, so the velocities come from them
  (ADR 0030). With the same scale for both bodies, every contact moves no net momentum:
  V-RB-04 holds within 1e-4.
- **Detected every substep, into a fixed capacity:** 32 contacts per body by default.
  Contacts past it are dropped and counted (`contact_overflows`, on the HUD's counts
  through the status read at the next call, never a wait).
- **No friction or restitution yet.** Contacts are inelastic along the normal and
  frictionless across it (#88).
- **Quaternions keep a NaN.** Orientations are scaled to unit length by hand: Warp's
  `normalize` turns a NaN quaternion into the identity, which hid a NaN from the guard
  once corrections were applied.

Named simplifications (SPEC 5.4 and the module):
- **Edge-on-edge contacts are sampled**, at edge midpoints and face centres, not found
  exactly: two boxes crossed edge to edge sink into each other until a corner, an edge
  midpoint or a face centre meets the other's surface.
- **Scaling by the larger contact count** slows the convergence of a body in many
  contacts (a box in a pile): it sinks a little more under load before the next substep
  corrects it.

Measured (unofficial, the GPU about 17 % busy with another process): bodies of the three
shapes piling in the Test Tank, the solver's step per 1/60 s frame of 8 substeps took
5.5 ms with 1,000 bodies (5,600 contacts) and 15.4 ms with 5,000 (39,000 contacts).
SPEC 3's budget for rigid bodies is 3 ms. Bringing it there is #90's: CUDA graphs, a
hash grid rebuilt once per frame with a margin, fewer launches.

## Alternatives considered

- **GJK and EPA for every pair.** General for any convex shape, including the convex
  hulls of fracture fragments later, but iterative per pair, divergent on the GPU, and
  one contact point per call: resting boxes need a manifold. Revisit for M4's
  fragments.
- **Analytic pairs with a box–box clipping manifold** (ODE's `dBoxBox`). Exact
  edge–edge contacts and four-point manifolds, but six pair routines to write and test,
  and box–box clipping is long. Sampling covers M2's shapes with one routine; revisit
  if piles show the edge–edge limit.
- **A sampled grid of the static solids** (as the liquid uses). Its 6.25 cm cells cannot
  hold a box on the floor to the millimetre; the exact distances can.
- **Graph colouring, Gauss–Seidel on the GPU.** Faster convergence for piles, but a
  colouring every substep; Jacobi with symmetric scaling keeps momentum and is simpler.
- **Detection once per frame with a margin.** Cheaper, and #90 may adopt it; per
  substep is the simplest correct start.

## How we would know it is wrong

- Boxes crossing edge to edge visibly sink into each other in play.
- Piles in the Room jitter or sink under their own weight (V-RB-05, #89).
- The rigid step at 5,000 bodies stays above 3 ms after #90's work.
