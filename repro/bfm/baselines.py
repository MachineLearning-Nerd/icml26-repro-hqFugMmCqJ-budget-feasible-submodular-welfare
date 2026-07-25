"""Baseline mechanisms for the welfare and valuation experiments.

Welfare baselines (Claim 6, Section 6.1): the three regularized-submodular
greedy algorithms converted to budget-feasible mechanisms by the Deng et al.
(2025) framework, exactly as the paper describes -- "after executing each
mechanism to completion, we select the longest prefix of the returned solution
that satisfies the budget constraint":

* ``deng_distorted``  -- Distorted Greedy (Harshaw et al. 2019)
* ``deng_roi``        -- ROI Greedy (Jin 2021)
* ``deng_costscaled`` -- Cost-Scaled Greedy (Nikolakai et al. 2021)

Each greedy produces an ordering of the elements (by its selection rule); the
budget-feasible prefix is the returned set. Payments are set to the reported
cost (truthful, IR, budget-feasible by construction).

Valuation baseline (Claim 4, Theorem 5.4): ``naive_greedy_vm`` -- a textbook
greedy budget-feasible mechanism for valuation maximization whose query
complexity is Theta(n^2) (it recomputes every marginal at every step), standing
in for the O(n^2 log n) prior deterministic mechanisms (Balkanski et al. 2022).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .oracles import Oracle
from .mechanisms import MechanismResult


def _truncate_to_budget(order: list[int], costs: np.ndarray, budget: float) -> list[int]:
    """Longest prefix of ``order`` whose total cost is within ``budget``."""
    out: list[int] = []
    spent = 0.0
    for u in order:
        if spent + costs[u] <= budget + 1e-12:
            out.append(u)
            spent += float(costs[u])
        else:
            break
    return out


def _greedy_order(oracle: Oracle, costs: np.ndarray, rule: str, n: int) -> list[int]:
    """Generic greedy that builds an ordering by repeatedly picking the best
    remaining element under ``rule`` until no element has positive gain."""
    c = np.asarray(costs, dtype=float)
    S: list[int] = []
    remaining = list(range(n))
    v_S = oracle.value(S)
    while remaining:
        best_u, best_score = -1, -math.inf
        for u in remaining:
            vS_u = oracle.value(S + [u])
            marg = vS_u - v_S
            cu = max(c[u], 1e-12)
            if rule == "distorted":
                score = marg - cu                       # marginal welfare
            elif rule == "roi":
                score = (marg - cu) / cu                # return on investment
            elif rule == "costscaled":
                score = marg / (1.0 + cu)               # cost-scaled marginal
            else:
                raise ValueError(rule)
            if score > best_score:
                best_score, best_u = score, u
        if best_u < 0 or best_score <= 0:
            break
        S.append(best_u)
        v_S = oracle.value(S)
        remaining.remove(best_u)
    return S


def _welfare_baseline(oracle: Oracle, costs: np.ndarray, budget: float, rule: str) -> MechanismResult:
    n = oracle.n
    q0 = oracle.queries
    order = _greedy_order(oracle, costs, rule, n)
    S = _truncate_to_budget(order, np.asarray(costs, dtype=float), budget)
    payments = {u: float(costs[u]) for u in S}
    return MechanismResult(
        winners=S,
        payments=payments,
        rounds=0,
        queries=oracle.queries - q0,
        history={},
        u_star=[],
    )


def deng_distorted(oracle: Oracle, costs: np.ndarray, budget: float) -> MechanismResult:
    return _welfare_baseline(oracle, costs, budget, "distorted")


def deng_roi(oracle: Oracle, costs: np.ndarray, budget: float) -> MechanismResult:
    return _welfare_baseline(oracle, costs, budget, "roi")


def deng_costscaled(oracle: Oracle, costs: np.ndarray, budget: float) -> MechanismResult:
    return _welfare_baseline(oracle, costs, budget, "costscaled")


def naive_greedy_vm(oracle: Oracle, costs: np.ndarray, budget: float) -> MechanismResult:
    """Quadratic-complexity greedy valuation mechanism (prior-work stand-in).

    Greedily adds the element with the largest marginal value per unit cost,
    recomputing every marginal at every step -> Theta(n^2) value queries. This
    is the structural reason prior deterministic mechanisms (Balkanski et al.
    2022, SIP) run in O(n^2 log n); BFM-VM avoids the per-step rescan.
    """
    n = oracle.n
    c = np.asarray(costs, dtype=float)
    q0 = oracle.queries
    S: list[int] = []
    spent = 0.0
    v_S = oracle.value(S)
    feasible = [u for u in range(n) if c[u] <= budget]
    while feasible:
        best_u, best_eff = -1, -math.inf
        for u in feasible:
            marg = oracle.value(S + [u]) - v_S
            eff = marg / max(c[u], 1e-12)
            if eff > best_eff:
                best_eff, best_u = eff, u
        if best_u < 0 or best_eff <= 0:
            break
        if spent + c[best_u] > budget + 1e-12:
            feasible.remove(best_u)
            continue
        S.append(best_u)
        spent += float(c[best_u])
        v_S = oracle.value(S)
        feasible.remove(best_u)
    payments = {u: float(c[u]) for u in S}
    return MechanismResult(
        winners=S,
        payments=payments,
        rounds=0,
        queries=oracle.queries - q0,
        history={},
        u_star=[],
    )
