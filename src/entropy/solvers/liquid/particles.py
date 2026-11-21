"""Liquid particles: position, velocity and the APIC affine matrix (SPEC 5.2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass(frozen=True)
class LiquidParticles:
    """Structure of arrays on one device.

    `affine_per_s` is APIC's C (Jiang et al. 2015): row c is the gradient of
    velocity component c around the particle, in 1/s.
    """

    position_m: wp.array
    velocity_m_per_s: wp.array
    affine_per_s: wp.array

    def __post_init__(self) -> None:
        n = self.position_m.shape
        if not (self.velocity_m_per_s.shape == self.affine_per_s.shape == n):
            raise ValueError("positions, velocities and affine matrices must have one length")
        if not (self.position_m.device == self.velocity_m_per_s.device == self.affine_per_s.device):
            raise ValueError("particle arrays must be on one device")

    @property
    def count(self) -> int:
        return self.position_m.shape[0]

    @property
    def device(self) -> wp.Device | None:
        return self.position_m.device

    @classmethod
    def from_numpy(
        cls,
        positions_m: np.ndarray,
        *,
        velocities: np.ndarray | None = None,
        affine: np.ndarray | None = None,
        device: str,
    ) -> LiquidParticles:
        """Particles at `positions_m` (n x 3, m), at rest unless velocities (m/s) and
        affine matrices (n x 3 x 3, 1/s) are given."""
        positions = np.asarray(positions_m, dtype=np.float32).reshape(-1, 3)
        n = len(positions)
        velocities = np.zeros((n, 3)) if velocities is None else np.asarray(velocities)
        affine = np.zeros((n, 3, 3)) if affine is None else np.asarray(affine)
        if velocities.shape != (n, 3):
            raise ValueError(f"velocities have shape {velocities.shape}, expected {(n, 3)}")
        if affine.shape != (n, 3, 3):
            raise ValueError(f"affine matrices have shape {affine.shape}, expected {(n, 3, 3)}")
        return cls(
            wp.array(positions, dtype=wp.vec3, device=device),
            wp.array(velocities.astype(np.float32), dtype=wp.vec3, device=device),
            wp.array(np.ascontiguousarray(affine, dtype=np.float32), dtype=wp.mat33, device=device),
        )
