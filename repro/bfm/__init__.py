"""BFM-SWM / BFM-VM reproduction package (arXiv 2605.00411)."""

from .oracles import Oracle, CoverageFunction, WeightedCoverage, CutFunction, brute_force_optimal
from .mechanisms import bfm_swm, bfm_vm, MechanismResult

__all__ = [
    "Oracle",
    "CoverageFunction",
    "WeightedCoverage",
    "CutFunction",
    "brute_force_optimal",
    "bfm_swm",
    "bfm_vm",
    "MechanismResult",
]
