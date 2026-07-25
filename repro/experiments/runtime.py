"""Runtime / query-complexity scaling experiments (Claims 2 and 4).

The paper's efficiency metric is query complexity (Section 6.1) -- the number of
value-oracle queries -- which is hardware-independent. We measure it as a
function of ``n`` for BFM-SWM, BFM-VM, and the quadratic ``naive_greedy_vm``
baseline, then fit a log-log slope.  BFM-VM should show slope ~1 (O(n log n))
and the naive baseline slope ~2 (O(n^2)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

from ..bfm.oracles import CoverageFunction
from ..bfm.mechanisms import bfm_swm, bfm_vm
from ..bfm.baselines import naive_greedy_vm
from .instances import coverage_instance


@dataclass
class ScalingRow:
    mechanism: str
    n: int
    queries: int
    rounds: int


def _loglog_slope(ns, qs):
    x = np.log(np.asarray(ns, dtype=float))
    y = np.log(np.asarray(qs, dtype=float))
    A = np.vstack([x, np.ones_like(x)]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(np.exp(sol[1]))


def run_scaling(ns=(32, 64, 128, 256, 512), seed=7, budget=1.0, eps=0.1, repeats=3):
    rows: list[ScalingRow] = []
    bfm_vm_q: list[float] = []
    naive_q: list[float] = []
    bfm_swm_q: list[float] = []
    for n in ns:
        # average over a few seeds to smooth jitter
        q_vm = q_naive = q_swm = 0
        rounds_vm = rounds_swm = 0
        for r in range(repeats):
            make, costs = coverage_instance(n, universe=max(4 * n, 60), degree=max(6, n // 6),
                                            cost_seed=seed + r, struct_seed=100 + seed + r)
            orc = make(); orc.reset()
            res_vm = bfm_vm(orc, costs, budget, alpha=1 + math.sqrt(3), ell=2)
            q_vm += res_vm.queries
            rounds_vm = max(rounds_vm, res_vm.rounds)

            orc = make(); orc.reset()
            res_naive = naive_greedy_vm(orc, costs, budget)
            q_naive += res_naive.queries

            orc = make(); orc.reset()
            res_swm = bfm_swm(orc, costs, budget, alpha=1 + math.sqrt(6) / 2, beta=3.0, eps=eps, ell=1)
            q_swm += res_swm.queries
            rounds_swm = max(rounds_swm, res_swm.rounds)
        q_vm /= repeats; q_naive /= repeats; q_swm /= repeats
        rows.append(ScalingRow("BFM-VM", n, int(round(q_vm)), rounds_vm))
        rows.append(ScalingRow("naive-greedy-VM", n, int(round(q_naive)), 0))
        rows.append(ScalingRow("BFM-SWM(monotone)", n, int(round(q_swm)), rounds_swm))
        bfm_vm_q.append(q_vm); naive_q.append(q_naive); bfm_swm_q.append(q_swm)

    vm_slope, vm_c = _loglog_slope(ns, bfm_vm_q)
    naive_slope, naive_c = _loglog_slope(ns, naive_q)
    swm_slope, swm_c = _loglog_slope(ns, bfm_swm_q)
    fit = {
        "ns": list(ns),
        "BFM-VM_loglog_slope": vm_slope,
        "BFM-VM_loglog_intercept": vm_c,
        "naive-greedy_loglog_slope": naive_slope,
        "BFM-SWM_loglog_slope": swm_slope,
    }
    return rows, fit
