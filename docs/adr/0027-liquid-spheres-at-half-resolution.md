# ADR 0027: The liquid pass draws its spheres at half resolution

- Status: accepted
- Date: 2026-11-22

## Context

The liquid pass (ADR 0023) draws every particle as a sphere sprite twice: once for depth
and once for thickness. It then smooths the depth and composites the water at full
resolution. The sprite passes cost per fragment, so their cost grows with the area the
spheres cover on screen. At 1 M particles and 1600 × 900, measured on the laptop
(unofficial, #57's gate benchmark):

| View | Whole pass | Sprites |
| --- | --- | --- |
| The game's start camera, from a corner | 1.46 ms | 1.3 ms |
| Above the rim, looking in | 3.58 ms | 3.2 ms |
| Leaning over the water | 4.64 ms | 4.1 ms |

SPEC 7.2's budget for the pass is 2.5 ms. In the M1 gate's hose scenario the pass took
4.3 to 8.5 ms of the frame, sharing the GPU with the simulation.

Sprites also had a known bias, already named as a simplification: a sphere smaller than
a pixel was still drawn as a one-pixel sprite, so far water read too thick.

## Decision

- **Spheres at half resolution.** The sprite passes (depth, thickness) and the depth's
  smoothing run in targets of half the output's size, rounded up (`SPRITE_DOWNSCALE`).
  - The composite runs at full resolution over the scene. It reads the water's depth at
    the nearest half-resolution pixel and its thickness interpolated. It refracts in the
    scene's own pixels.
  - Each sphere is tested for being behind a solid against one of the four scene pixels
    it covers.
- **Sub-pixel spheres weighted by d².** A one-pixel sprite integrates a disc's chords
  over a one-pixel disc. For a sphere of true diameter d < 1 pixel, that is 1/d² times
  the sphere's own area, so its thickness is weighted by d². Far water then adds its true
  volume on average. This fixes the bias that half resolution would otherwise double:
  without the weight, deep water seen small read 65 % too thick.

Measured after this decision (same benchmark, unofficial):

| View | Whole pass |
| --- | --- |
| Start camera | 0.57 ms |
| Above the rim | 1.12 ms |
| Leaning over the water | 1.56 ms |

All three are within the 2.5 ms budget. The still pool's reference image changed (SSIM
0.967 against the old one): far water is a little clearer, and silhouettes are two output
pixels coarse. The reference is re-recorded.

Named simplifications (in `entropy.render.liquid` and SPEC 7.2):
- silhouettes two output pixels coarse;
- one scene pixel of four tested behind solids;
- far water's thickness right on average, not pixel by pixel.

## Alternatives considered

- **Quarter resolution for thickness only.** Depth costs as much as thickness, so this
  halves less.
- **Bilateral upsampling of the depth.** It gives smoother silhouettes, at the cost of a
  pass. Leave it until silhouettes look wrong in play.
- **Fewer, larger sprites far away (level of detail).** It would change what the
  smoothing sees, and needs the particles grouped on the GPU first.

## How we would know it is wrong

- Matt finds the water's edges blocky or shimmering in play.
- The pass exceeds 2.5 ms in a view the game reaches. The gate benchmark covers three
  views.
