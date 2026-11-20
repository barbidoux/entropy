# ADR 0005: Module layering is transitive, with `app` as the composition root

- Status: accepted
- Date: 2026-11-20

Number 0004 is reserved for the rendering stack, decided later in M0.

## Context

SPEC 8.1 lists, for each subpackage, the packages it "may depend on", and says a test
enforces the rules (`tests/test_architecture.py`). Two questions were left open:

1. Is the list direct-only or transitive? `fields` may use `core`, and `linalg` may use
   `fields`: may `linalg` use `core` directly?
2. Something has to open the window, run the main loop and hand the camera from `game`
   to `render` and the statistics from `sim` to `ui`. In the table, `render` may not
   import `game`, and `game` may not import `render`, so no listed package can do it.

## Decision

- "May depend on" is **transitive**: a package may import what its dependencies may
  import. Dependencies still point down only, which is the property that matters.
- A composition root, `entropy.app` (with `entropy/__main__.py`), wires the window, the
  game, the renderer and the interface. It may import everything, and nothing may import
  it. It holds wiring only: no physics, no rendering passes, no game rules.
- The package root `entropy/__init__.py` imports no subpackage, because it runs on every
  import and would otherwise make every package depend on what it imports.
- SPEC 8.1 gains rows for `app` and for `audio` (listed in 10.2, missing from 8.1), and a
  sentence on transitivity.

## Alternatives considered

- **Direct-only dependencies:** every package would re-list `core`, and every new
  low-level helper would need table edits. No layering benefit.
- **Putting the main loop in `game` or `sim`:** either would have to import `render` or
  `ui`, inverting the arrows of SPEC 8.
- **Passing objects through callbacks registered at import time:** hides the wiring and
  needs global mutable registries (CLAUDE.md rule 6).

## Consequences

- One place shows how the program is assembled; every other package stays testable alone.
- The architecture test treats `app` and `__main__` as the composition root and the
  package root as importing nothing.
- Limits of the static check (dynamic imports, read-only rules, solver-to-solver imports)
  are listed in the test's docstring; solver isolation is issue #21 before M1.

## How we would know it is wrong

`app` grows logic of its own that needs unit tests reaching into it, or two packages need
to call each other and the only way out is routing through `app`.
