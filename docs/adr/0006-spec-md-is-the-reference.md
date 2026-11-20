# ADR 0006: `docs/SPEC.md` is the reference copy of the specification

- Status: accepted
- Date: 2026-11-20

## Context

The specification was written as a living Claude document and exported to
`docs/SPEC.md` with a header saying the living document wins. The maintainer sessions
cannot open the living document, yet they have to change the specification with the
code (SPEC 9.5), for example the SPEC 8.1 rows added with ADR 0005.

On 2026-11-20, when starting M0, Matt (product owner) said: from now on `docs/SPEC.md`
is the reference; if he changes the specification in the living document, he will say so.

## Decision

- `docs/SPEC.md` is the reference copy. It changes through pull requests, like code.
- The living document is no longer re-exported over it. When Matt changes the living
  document, he tells the maintainer, who brings the change into `docs/SPEC.md` by pull
  request.
- The header of `docs/SPEC.md` and `CLAUDE.md` say so.

## Alternatives considered

- **Living document wins, re-exported now and then:** maintainer edits made with the code
  (such as the SPEC 8.1 rows) would be lost at the next export.
- **Maintainer edits both:** impossible, the maintainer has no access to the living
  document.

## Consequences

- Specification changes are reviewed and versioned with the code that needs them.
- The living document drifts from the reference; Matt reads `docs/SPEC.md` (or the
  maintainer's summaries) for the current state.

## How we would know it is wrong

A specification change Matt made in the living document never reaches `docs/SPEC.md`, or
Matt finds he needs to edit the specification himself more easily than through the
maintainer.
