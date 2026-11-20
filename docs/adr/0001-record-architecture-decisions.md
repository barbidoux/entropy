# ADR 0001: Record architecture decisions

- Status: accepted
- Date: 2026-11-20

## Context

ENTROPY is maintained by successive Claude sessions that do not share memory, and by a
product owner who does not read code. Decisions must survive both.

## Decision

Every significant technical decision is recorded as a short numbered file in `docs/adr/`,
with: context, decision, alternatives considered, consequences, and the test or
measurement that would show the decision was wrong.

An accepted ADR is never edited. A new ADR supersedes it and says so in both files'
status lines.

## Consequences

- A new session can reconstruct why the code is the way it is.
- Reversing a decision is explicit and traceable.

## How we would know it is wrong

Sessions repeatedly re-open settled questions because the ADRs are missing or unclear.
