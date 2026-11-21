"""APIC transfers between particles and the MAC grid (SPEC 5.2; Jiang et al. 2015).

Each velocity component is transferred on its own face grid (ADR 0015) with
quadratic B-spline weights over the 3 x 3 x 3 nodes around the particle.

- Particles to grid: face mass Σ w m_p, and face momentum Σ w m_p (v_pc + c_pc ·
  (x_i - x_p)), with c_pc row c of the particle's affine matrix. The velocity is
  momentum / mass, and 0 on faces that got no mass. This conserves mass, momentum
  and the APIC angular momentum Σ m_p (x_p x v_p + (Δx²/4) Σ_c c_pc x e_c).
- Grid to particles: v_pc = Σ w v_i, and c_pc = (4/Δx²) Σ w v_i (x_i - x_p). For
  quadratic B-splines Σ w (x_i - x_p)(x_i - x_p)ᵀ = Δx²/4 I, so an affine field
  and its gradient are read exactly.

Nodes outside the grid are skipped, so the transfers are exact only for
particles whose stencil lies inside: at least one cell in from every side. The
liquid keeps a margin of cells around its particles for that reason.
"""

from __future__ import annotations

import math

import warp as wp

from entropy.fields.mac import MacVelocity
from entropy.solvers.liquid import _transfer_kernels as _k
from entropy.solvers.liquid.particles import LiquidParticles


def particles_to_grid(
    particles: LiquidParticles, particle_mass_kg: float, velocity: MacVelocity, mass: MacVelocity
) -> None:
    """Overwrite `velocity` (m/s) and `mass` (kg per face) from the particles.

    `mass` is a face field on the same grid; every particle weighs `particle_mass_kg`.
    """
    if velocity.spec != mass.spec:
        raise ValueError("velocity and mass must be on one grid")
    if not (math.isfinite(particle_mass_kg) and particle_mass_kg > 0):
        raise ValueError(f"particle mass must be positive, got {particle_mass_kg}")
    if not particles.device == velocity.device == mass.device:
        raise ValueError("particles, velocity and mass must be on one device")
    spec, device = velocity.spec, velocity.device
    for array in (velocity.u, velocity.v, velocity.w, mass.u, mass.v, mass.w):
        array.zero_()
    if particles.count:
        wp.launch(
            _k.particles_to_grid,
            dim=particles.count,
            inputs=[
                particles.position_m,
                particles.velocity_m_per_s,
                particles.affine_per_s,
                wp.vec3(*spec.origin_m),
                spec.dx_m,
                particle_mass_kg,
                velocity.u,
                velocity.v,
                velocity.w,
                mass.u,
                mass.v,
                mass.w,
            ],
            device=device,
        )
    for component, weight in zip(
        (velocity.u, velocity.v, velocity.w), (mass.u, mass.v, mass.w), strict=True
    ):
        wp.launch(
            _k.momentum_to_velocity, dim=component.shape, inputs=[component, weight], device=device
        )


def grid_to_particles(velocity: MacVelocity, particles: LiquidParticles) -> None:
    """Overwrite the particles' velocities (m/s) and affine matrices (1/s) from the grid."""
    if particles.device != velocity.device:
        raise ValueError("particles and velocity must be on one device")
    if not particles.count:
        return
    spec = velocity.spec
    wp.launch(
        _k.grid_to_particles,
        dim=particles.count,
        inputs=[
            velocity.u,
            velocity.v,
            velocity.w,
            wp.vec3(*spec.origin_m),
            spec.dx_m,
            particles.position_m,
            particles.velocity_m_per_s,
            particles.affine_per_s,
        ],
        device=velocity.device,
    )
