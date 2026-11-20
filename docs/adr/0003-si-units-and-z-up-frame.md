# ADR 0003: SI units, kelvin, and a right-handed Z-up frame

- Status: accepted
- Date: 2026-11-20

## Context

Physical validation (SPEC 9.4) compares results against laws and experiments stated in
SI units. Recorded sequences are rendered offline in Blender, which is Z-up and uses
metres.

## Decision

- All quantities are SI: metre, kilogram, second, kelvin, pascal, joule. The interface
  may display °C; storage and computation use kelvin.
- The world frame is right-handed with Z up. Gravity is 9.81 m/s² along −Z.

## Alternatives considered

- **Y-up, the OpenGL convention:** would need an axis swap at every Blender export and
  invite sign errors in physics code. The camera handles the conversion once instead.

## Consequences

- Exports to Blender need no conversion.
- Docstrings name units; ambiguous variable names carry them (`dt_s`, `temperature_k`).

## How we would know it is wrong

Axis or unit conversion bugs appear at module boundaries despite the convention.
