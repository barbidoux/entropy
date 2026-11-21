# ADR 0023: The frame is drawn in linear light, pass by pass, and looks live in material files

- Status: accepted
- Date: 2026-11-21

## Context

M1 draws the Test Tank and its water (issues #51, #52). The water's colour comes from
Beer–Lambert absorption and Fresnel reflection, which are sums and products of light,
so they need linear light. The glass must not show through water in front of it, and
must be seen over water behind it. SPEC 7.2 lists passes and a post pass with exposure
and ACES, but not how they share their buffers. The renderer also needs values that are
look, not physics, such as a concrete's colour, and the solver files forbid constants
in code (CLAUDE.md rule 4).

## Decision

- **Half-float HDR targets.** Every pass draws linear radiance (relative units, until
  SPEC 7.2's calibration) into a half-float colour target with a depth texture.
  Tone mapping comes last: exposure, then the ACES fit, then sRGB.
- **Pass order** (`entropy.render.world_frame`):
  1. solids into target A;
  2. the liquid, reading A (colour and depth) and writing target B, with the water's
     own depth where it is nearer;
  3. glass blended into B, depth-tested against that depth;
  4. tone mapping onto the screen;
  5. the HUD.

  Each pass is timed for the HUD.
- **Looks live in material files.** Optional constants (`albedo_linear_rgb`,
  `refractive_index`, `absorption_per_m_rgb`) sit beside the physical ones, each with
  its source. A value chosen by eye says so in its source.

## Alternatives considered

- **Drawing straight to the 8-bit screen:** absorption and blending in display space
  are wrong by tens of per cent at mid-tones, and there is no way to read the scene
  behind the water for refraction.
- **Glass before the water:** water behind a pane would then hide the pane's
  reflection, and water in front would be blended under the glass.
- **Render colours in a separate look file:** two files per material to keep in step,
  for a handful of values.

## How we would know it is wrong

- Fire's blackbody emission (M3) or bloom needs a target with more range or precision
  than half floats.
- Transparent objects other than glass and water (smoke, M3) need an order-independent
  scheme, not a fixed pass order.
