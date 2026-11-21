# ADR 0020: Tools, and water thrown from outside the liquid's grid

- Status: accepted
- Date: 2026-11-21

## Context

M1's tools add water to the Test Tank (SPEC 4.2, issues #35, #53, #54). The liquid
exists only inside its grid: the tank's cavity and 0.5 m of air above it (ADR 0017).
The camera, and so the hose's nozzle, is usually outside that grid. ADR 0013 asks
which nozzle speed to offer by default: the liquid takes at most 4 substeps per frame,
so water faster than about 4 cells per frame slows the world down (at 6.25 cm cells:
about 14 m/s at 60 fps, 6.7 m/s at 30 fps).

## Decision

- **Tools are plug-ins** (SPEC 8.2): a class with `on_press`, `on_hold(dt)` and
  `on_release`, and `content/tools/<name>.toml` with its parameters in SI units, their
  limits and defaults, and the parameter the mouse wheel sets. `dt` is simulated time,
  so slow motion gives each frame less water and the flow per simulated second holds.
- **Accumulated volume** (#35): a source carries the fraction of a particle it could not
  release to the next frame. At 20 L/s and 0.1x that is 1.1 particles per frame, and
  nothing is lost to rounding.
- **Water outside the grid flies ballistically**, as parcels of one particle each, under
  gravity only. A parcel joins the liquid where it enters the grid, through its open top
  or a side above the glass's rim. A side below the rim is glass, and the water that
  reaches it is lost; so is water that falls to the ground. Parcels in flight are drawn
  with the liquid.
- **Default nozzle speed 10 m/s** (range 5–30 m/s). From above the rim it lands below
  14 m/s, which runs at full speed at 60 fps. Faster jets are allowed: the world then
  slows down rather than becoming unstable (ADR 0013), and the HUD says so.

Named simplifications: no air drag or breakup of the jet outside the grid; water that
misses the tank is lost (it cannot exist outside the tank in M1, ADR 0017); a parcel
joins the liquid at the end of the frame it entered in.

## Alternatives considered

- **A grid that includes the camera:** the domain would grow with every move of the
  player, and water outside the tank needs the glass as a solid the grid can see
  (ADR 0017's named limit).
- **Emitting at the grid's edge, straight away:** the water would appear in the tank
  before it could have flown there, and the jet would be invisible.
- **Rounding particles per frame:** at 0.1x it loses about 8 % of the water (#35).

## How we would know it is wrong

- US-01 misses 200 L ± 2 % at 1x or 0.1x.
- Matt finds the jet's look or reach wrong in play (no drag: long throws go further than
  a real hose's).
