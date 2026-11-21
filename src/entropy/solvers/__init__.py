"""Solvers, one per phenomenon, each implementing `entropy.solvers.interface.Solver` (SPEC 8).

The package root re-exports the interface only, so importing it names no solver
(architecture test).
"""

from entropy.solvers.interface import Solver

__all__ = ["Solver"]
