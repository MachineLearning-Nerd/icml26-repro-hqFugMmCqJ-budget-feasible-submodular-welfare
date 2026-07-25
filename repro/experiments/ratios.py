"""Approximation-ratio experiments for BFM-SWM (Claims 1, 2) and BFM-VM (Claim 3).

For each instance we run the mechanism, compute the exact optimum by brute
force (instances are kept small enough for that), and record the achieved
ratio. The worst observed ratio across all instances is the empirical
corroboration of the worst-case guarantee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np

from ..bfm.oracles import Oracle, brute_force_optimal
from ..bfm.mechanisms import bfm_swm, bfm_vm
from .instances import coverage_family, cut_family


@dataclass
class RatioRow:
    instance_id: int
    kind: str
    n: int
    budget: float
    rounds: int
    queries: int
    mechanism_value: float
    mechanism_welfare: float
    opt_value: float
    opt_welfare: float
    ratio: float


def _welfare_ratio_sweep(family, alpha, beta, eps, ell, budget, kind, tag):
    rows = []
    for i, (make, costs) in enumerate(family):
        orc = make()
        orc.reset()
        res = bfm_swm(orc, costs, budget, alpha, beta, eps, ell)
        v = orc.value
        vS = v(res.winners)
        cS = float(sum(costs[u] for u in res.winners))
        _, vO = brute_force_optimal(orc, orc.n, "welfare", costs)
        ratio = (vS - cS) / vO if vO > 1e-12 else 1.0
        rows.append(RatioRow(i, f"{kind}/{tag}", orc.n, budget, res.rounds, res.queries, vS, vS - cS, vO, vO, ratio))
    return rows


def _valuation_ratio_sweep(family, alpha, ell, budget):
    rows = []
    for i, (make, costs) in enumerate(family):
        orc = make()
        orc.reset()
        res = bfm_vm(orc, costs, budget, alpha, ell)
        v = orc.value
        vS = v(res.winners)
        _, vO = brute_force_optimal(orc, orc.n, "val", None)
        ratio = vS / vO if vO > 1e-12 else 1.0
        rows.append(RatioRow(i, "vm-coverage", orc.n, budget, res.rounds, res.queries, vS, vS, vO, vO, ratio))
    return rows


def claim1_general_welfare(n=14, k=24, budget=1.0, eps=0.1):
    """Claim 1 (Theorem 4.8): general (non-monotone) welfare ratio >= 0.0328."""
    alpha = 1 + 2 * math.sqrt(6) / 3
    fam = cut_family(n, k)  # non-monotone
    rows = _welfare_ratio_sweep(fam, alpha, 4.0, eps, 2, budget, "cut", "general")
    return rows, alpha, 4.0, 2


def claim2_monotone_welfare(n=16, k=24, budget=1.0, eps=0.1):
    """Claim 2 (Theorem 4.10): monotone welfare ratio >= 0.0877."""
    alpha = 1 + math.sqrt(6) / 2
    fam = coverage_family(n, k)  # monotone
    rows = _welfare_ratio_sweep(fam, alpha, 3.0, eps, 1, budget, "coverage", "monotone")
    return rows, alpha, 3.0, 1


def claim3_valuation(n=16, k=24, budget=1.0):
    """Claim 3 (Theorem 5.4): BFM-VM valuation ratio >= 1/(12+4 sqrt3)."""
    alpha = 1 + math.sqrt(3)
    fam = coverage_family(n, k)
    rows = _valuation_ratio_sweep(fam, alpha, 2, budget)
    return rows, alpha, 2
