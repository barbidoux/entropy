# ADR 0027: The liquid pass draws its spheres at half resolution

- Status: accepted
- Date: 2026-11-22

## Context

The liquid pass (ADR 0023) draws every particle as a sphere sprite twice, once for depth
and once for thickness. It then smooths the depth and composites the water at full
resolution. The sprite passes cost per fragment: the more screen the spheres cover, the
more they cost.

Measured at 1 M particles and 1600 × 900 (unofficial, #57's gate benchmark):

| View | Whole pass | Sprite passes |
| --- | --- | --- |
| The game's start camera, from a corner | 1.46 ms | 1.3 ms |
| Above the rim, looking in | 3.58 ms | 3.2 ms |
| Leaning over the water | 4.64 ms | 4.1 ms |

SPEC 7.2's budget for the pass is 2.5 ms. In the M1 gate's hose scenario, the pass took
4.3 to 8.5 ms of the frame, sharing the GPU with the simulation.

A bias was also already named: a sphere smaller than a pixel was drawn as a one-pixel
sprite, so far water read too thick. It read +48 % too thick, and +484 % once spheres
were a third to a half of a pixel across.

## Decision

- **Spheres at half resolution.** The sprite passes and the depth's smoothing run in
  targets half the output's size, rounded up (`SPRITE_DOWNSCALE`).
  - A sphere is dropped only where it is behind the scene at all four output pixels its
    sprite pixel covers. Those four are read in one `textureGather`, and the farthest
    counts.
  - The smoothing filter's largest radius, `MAX_FILTER_PX = 24`, is now in half-size
    pixels: 48 output pixels. Water nearer than about 1.3 m at 1600 × 900 is smoothed
    less than its radius asks; before, that applied to water nearer than 2.5 m.
- **The composite stays at full resolution.**
  - Each output pixel finds its place among the half-size pixels at its centre.
  - Where the four nearest are water within the smoothing's range of each other, it
    interpolates their depth and their normals bilinearly, and so its reflection and
    refraction. At a silhouette it takes the nearest pixel's.
  - The thickness is interpolated too.
  - Water behind the output pixel's own solid is not drawn there: the scene and its
    depth show instead. There is no allowance: none was visibly needed, at grazing
    views or at waterlines, and one of a few sphere radii let water show over a solid's
    edge just in front of it. Seen within 20° of the floor, a thin sheet's side face is
    dropped where the smoothing pulls its depth behind the floor: at most one 8-bit
    level of difference, on the review's light-grey floor.
- **Sub-pixel spheres weighted by d².**
  - Averaged over its position in the pixel, a one-pixel sprite adds the chords of a
    one-pixel disc. For a sphere of true diameter d < 1 pixel, that is 1/d² times its
    own.
  - Weighting its thickness by d² makes it add its true volume on average (πr·d²/3 per
    pixel). The weight is continuous at d = 1.
  - Without it, half resolution would have doubled the bias above.

Measured after this decision (same benchmark and conditions):

| View | Whole pass |
| --- | --- |
| Start camera | 0.58 ms |
| Above the rim | 1.13 ms |
| Leaning over the water | 1.62 ms |

All three views are within the 2.5 ms budget.

The thickness of a 2 m slab is now independent of how big the spheres look. Tests cover
diameters of about 0.4, 1.4 and 2.9 half-size pixels, within 3 %; the review measured
within 1.5 % from 0.17 to 6.

The still pool's reference image is re-recorded. Its SSIM is 0.969 against the image
before half resolution. Far water is a little clearer, and refracted lines are smooth.

Named simplifications (in `entropy.render.liquid` and SPEC 7.2):
- **Coarser silhouettes.** They are two output pixels coarse, and detail smaller than
  two output pixels is lost.
- **More shimmer.** The water shimmers more as the camera turns. The independent review
  turned the camera by one pixel and took the mean change over the centre of a 2 m pool
  at 1600 × 900 (8-bit levels):

  | View | Before | After |
  | --- | --- | --- |
  | Start camera | 0.82 | about 1.9 |
  | Above the rim | 1.15 | 1.37 |
  | Up close | 0.81 | 0.86 |

  Most of the increase comes from drawing the sprites at half resolution, not from the
  upsampling. Matt's play session is the judge.
- **Far water's thickness is right on average,** not pixel by pixel.
- **Thickness blends across a solid's outline inside the water** over one or two output
  pixels. The review measured, for a submerged box, a mean error of 4 % in that band
  and 15 % at the 95th percentile.

## Alternatives considered

- **Quarter resolution for the thickness only.** The depth pass costs as much as the
  thickness pass, so this halves less.
- **The nearest half-size pixel for depth and normals.** This was the first version. It
  stair-stepped refracted lines in 2 × 2 blocks: 2-pixel "fishbones" up close.
  Interpolating costs about 0.01–0.04 ms.
- **Temporal accumulation against the shimmer.** It needs motion vectors for the water,
  which the pass does not have. Revisit if play shows the shimmer.
- **Fewer, larger sprites far away (level of detail).** It would change what the
  smoothing sees, and needs the particles grouped on the GPU first.

## How we would know it is wrong

- Matt finds the water's edges blocky, or its surface shimmering, in play.
- The pass exceeds 2.5 ms in a view the game reaches. The benchmark covers three views.
- Solids inside the water (M2) show water drawn over their edges, or unabsorbed gaps
  beside them. Tests cover off-grid box edges, water just behind a solid's edge, and
  water in front of water.
