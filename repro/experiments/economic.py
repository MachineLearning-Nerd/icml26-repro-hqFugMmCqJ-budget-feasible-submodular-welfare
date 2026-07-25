"""Economic-property verification for BFM-SWM (Claim 5, Theorem 4.1).

Theorem 4.1 states BFM-SWM satisfies:
  * obvious strategyproofness (which implies truthfulness),
  * individual rationality,
  * non-negative auctioneer surplus,
  * budget feasibility.

The first three "hard" invariants (budget, IR, surplus) are checked on every
mechanism output. Strategyproofness is checked operationally: for each seller we
try a battery of misreported costs and confirm no deviation raises the seller's
utility (payment minus true cost) -- a descending-clock auction is
strategyproof exactly when truthful reporting is a (weakly) dominant strategy.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..bfm.oracles import Oracle, CoverageFunction, CutFunction
from ..bfm.mechanisms import bfm_swm, MechanismResult


@dataclass
class PropertyReport:
    n_instances: int
    budget_violations: int
    ir_violations: int
    surplus_violations: int
    strategyproof_violations: int
    total_deviation_tests: int
    min_surplus: float
    max_budget_ratio: float
    passed: bool


def _hard_invariants(res: MechanismResult, v: Callable, costs: np.ndarray, budget: float, beta: float):
    """Return list of violation strings for budget / IR / surplus / beta-ratio."""
    vial = []
    p = res.payments
    S = res.winners
    pS = sum(p[u] for u in S)
    vS = v(S)
    if pS > budget + 1e-9:
        vial.append(f"budget {pS}>{budget}")
    for u in S:
        if p[u] < costs[u] - 1e-9:
            vial.append(f"IR u={u} p={p[u]}<c={costs[u]}")
    if vS - pS < -1e-9:
        vial.append(f"surplus {vS}-{pS}={vS - pS}<0")
    return vial, pS, vS


def _utility(res: MechanismResult, u: int, true_cost: float) -> float:
    if u in res.winners:
        return res.payments[u] - true_cost
    return 0.0


def check_strategyproofness(
    make_oracle: Callable[[], Oracle],
    costs: np.ndarray,
    budget: float,
    alpha: float,
    beta: float,
    eps: float,
    ell: int,
    deviations_per_seller: int = 5,
    rng: random.Random | None = None,
) -> tuple[int, list[str]]:
    """For each seller, try ``deviations_per_seller`` misreported costs and
    confirm none beats truthful reporting. Returns (#tests, violations)."""
    rng = rng or random.Random(0)
    n = len(costs)
    # truthful run
    orc = make_oracle()
    orc.reset()
    truth = bfm_swm(orc, costs, budget, alpha, beta, eps, ell)
    v_fn = lambda S: orc.value(S)  # noqa: E731  (oracle bound to truth's sets)
    # NOTE: oracle.value depends on the function, not costs, so reuse is fine
    violations: list[str] = []
    tests = 0
    for u in range(n):
        u_truth_util = _utility(truth, u, float(costs[u]))
        for _ in range(deviations_per_seller):
            factor = rng.choice([0.25, 0.5, 0.75, 1.5, 2.0, 4.0, 8.0])
            bid = max(0.0, min(float(costs[u]) * factor, budget * 1.5))
            if abs(bid - float(costs[u])) < 1e-9:
                continue
            dev_costs = costs.copy()
            dev_costs[u] = bid
            orc2 = make_oracle()
            orc2.reset()
            devres = bfm_swm(orc2, dev_costs, budget, alpha, beta, eps, ell)
            u_dev_util = _utility(devres, u, float(costs[u]))
            tests += 1
            if u_dev_util > u_truth_util + 1e-7:
                violations.append(
                    f"seller {u}: truthful util {u_truth_util:.6f} < dev util "
                    f"{u_dev_util:.6f} (bid {bid:.4f} vs true {costs[u]:.4f})"
                )
    return tests, violations


def run_economic_suite(
    instances: list[tuple[Callable[[], Oracle], np.ndarray, float]],
    alpha: float,
    beta: float,
    eps: float,
    ell: int,
    deviations_per_seller: int = 5,
    seed: int = 0,
) -> PropertyReport:
    rng = random.Random(seed)
    n_inst = len(instances)
    bv = ir = sv = spv = 0
    total_tests = 0
    min_surplus = math.inf
    max_budget_ratio = 0.0
    for (make_oracle, costs, budget) in instances:
        orc = make_oracle()
        orc.reset()
        res = bfm_swm(orc, costs, budget, alpha, beta, eps, ell)
        v_fn = lambda S: orc.value(S)  # noqa: E731
        vial, pS, vS = _hard_invariants(res, v_fn, costs, budget, beta)
        if any("budget" in s for s in vial):
            bv += 1
        if any("IR" in s for s in vial):
            ir += 1
        if any("surplus" in s for s in vial):
            sv += 1
        min_surplus = min(min_surplus, vS - pS)
        max_budget_ratio = max(max_budget_ratio, pS / budget if budget > 0 else 0.0)
        tests, violations = check_strategyproofness(
            make_oracle, costs, budget, alpha, beta, eps, ell, deviations_per_seller, rng
        )
        total_tests += tests
        if violations:
            spv += 1
    passed = (bv == 0 and ir == 0 and sv == 0 and spv == 0)
    return PropertyReport(
        n_instances=n_inst,
        budget_violations=bv,
        ir_violations=ir,
        surplus_violations=sv,
        strategyproof_violations=spv,
        total_deviation_tests=total_tests,
        min_surplus=min_surplus,
        max_budget_ratio=max_budget_ratio,
        passed=passed,
    )
