"""Rigid bodies: extended position-based dynamics with substeps (SPEC 5.4)."""

from entropy.solvers.rigid.bodies import Body, Shape, mass_properties
from entropy.solvers.rigid.solver import MAX_SUBSTEP_S, RigidSetup, RigidSolver, load_dump

__all__ = [
    "MAX_SUBSTEP_S",
    "Body",
    "RigidSetup",
    "RigidSolver",
    "Shape",
    "load_dump",
    "mass_properties",
]
