"""Faithful implementation of the two mechanisms from arXiv 2605.00411.

* :func:`bfm_swm` -- Algorithm 1 (BFM-SWM), the budget-feasible mechanism for
  submodular *welfare* maximization (Section 4).
* :func:`bfm_vm`  -- Algorithm 2 (BFM-VM), the deterministic budget-feasible
  mechanism for submodular *valuation* maximization (Appendix B / Section 5).

Both follow the pseudocode line-for-line. The "seller accepts price p" test is
``c(u) <= p`` (a descending-clock auction with truthful cost reporting). Query
counts come from the :class:`~repro.bfm.oracles.Oracle` wrapper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .oracles import Oracle


@dataclass
class MechanismResult:
    winners: list[int]
    payments: dict[int, float]
    rounds: int
    queries: int
    # per-round candidate sets (for analysis / debugging)
    history: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    u_star: list[int] = field(default_factory=list)


def _accepts(c: np.ndarray, u: int, price: float) -> bool:
    return bool(c[u] <= price)


def bfm_swm(
    oracle: Oracle,
    costs: np.ndarray,
    budget: float,
    alpha: float,
    beta: float,
    eps: float,
    ell: int,
) -> MechanismResult:
    """Algorithm 1: BFM-SWM (welfare maximization).

    Parameters as fixed by the paper's theorems:
      * general submodular   (Thm 4.8): alpha = 1+2*sqrt(6)/3, beta = 4, ell = 2
      * monotone submodular  (Thm 4.10): alpha = 1+sqrt(6)/2, beta = 3, ell = 1
    """
    n = oracle.n
    c = np.asarray(costs, dtype=float)
    # Line 1: R = sellers who accept price B; p(u) <- B
    R = [u for u in range(n) if _accepts(c, u, budget)]
    p = {u: float(budget) for u in R}
    # Line 2: t <- 0; rho_0 <- eps/alpha; u* <- empty; owner <- 0
    t = 0
    rho = eps / alpha
    u_star: list[int] = []
    owner = {u: 0 for u in R}
    # previous-round candidate sets (round 0 = empty)
    S_prev = {i: [] for i in range(1, ell + 1)}
    history: dict[tuple[int, int], list[int]] = {}

    M = 0
    while True:
        # Line 4: t <- t+1; rho_t <- alpha*rho_{t-1}; S_{i,t} <- empty
        t += 1
        rho = alpha * rho
        S_cur = {i: [] for i in range(1, ell + 1)}
        v_S = {i: 0.0 for i in range(1, ell + 1)}
        p_S = {i: 0.0 for i in range(1, ell + 1)}
        # Line 5: iterate over R \ (union S_{i,t-1} ∪ u*)
        prev_union = set().union(*[set(S_prev[i]) for i in range(1, ell + 1)]) if S_prev else set()
        exclude = prev_union | set(u_star)
        candidates = [u for u in R if u not in exclude]
        broke = False
        for u in candidates:
            if u not in R:
                continue
            # Line 6-7: determine candidate sequence j
            if owner[u] != 0:
                j = owner[u]
            else:
                j = 1
                best_marg = -math.inf
                for i in range(1, ell + 1):
                    marg = oracle.marginal(u, S_cur[i])
                    if marg > best_marg:
                        best_marg = marg
                        j = i
            # Line 8: p(u) <- min{p(u), v(u|S_{j,t})/(beta + rho_t/B)}
            marg_u = oracle.marginal(u, S_cur[j])
            new_price = marg_u / (beta + rho / budget)
            if new_price < p[u]:
                p[u] = new_price
            # Line 9: if u accepts price p(u)
            if _accepts(c, u, p[u]):
                # Line 10: if v(S_{j,t}∪{u}) - p(S_{j,t}∪{u}) > rho_t
                surplus_new = (v_S[j] + marg_u) - (p_S[j] + p[u])
                if surplus_new > rho:
                    # Line 11: u* <- {u}; break
                    u_star = [u]
                    broke = True
                    break
                else:
                    # Line 13: S_{j,t} <- S_{j,t} ∪ {u}
                    S_cur[j].append(u)
                    v_S[j] += marg_u
                    p_S[j] += p[u]
                    # Line 14: if owner(u)=0: owner(u) <- j
                    if owner[u] == 0:
                        owner[u] = j
            else:
                # Line 15: R <- R \ {u}
                R = [x for x in R if x != u]
        # record history (S_{i,t} = current round)
        for i in range(1, ell + 1):
            history[(i, t)] = list(S_cur[i])
        # Line 16: until R \ (union (S_{i,t-1} ∪ S_{i,t}) ∪ u*) = empty.
        # S_prev still holds round t-1 here; S_cur holds round t.
        prev_round_union = set().union(*[set(S_prev[i]) for i in range(1, ell + 1)]) if S_prev else set()
        cur_union = set().union(*[set(S_cur[i]) for i in range(1, ell + 1)])
        remaining = set(R) - (prev_round_union | cur_union | set(u_star))
        S_prev = S_cur
        if not remaining:
            M = t
            break
        if t > 10000:  # safety guard against pathological loops
            M = t
            break

    # Line 17: S* <- argmax over {S_{i,t}: i in [ell], t in {M-1,M}} ∪ {u*} of v(A)-p(A)
    best_A: list[int] = []
    best_obj = -math.inf
    cand_rounds = {max(M - 1, 1), M}
    for tt in cand_rounds:
        for i in range(1, ell + 1):
            A = history.get((i, tt), [])
            if not A:
                continue
            vA = oracle.value(A)
            pA = sum(p[u] for u in A)
            obj = vA - pA
            if obj > best_obj:
                best_obj = obj
                best_A = list(A)
    # singleton candidate u*
    if u_star:
        vU = oracle.value(u_star)
        pU = sum(p[u] for u in u_star)
        if vU - pU > best_obj:
            best_obj = vU - pU
            best_A = list(u_star)
    # empty set fallback
    vE = oracle.value(set())
    if 0 - 0 > best_obj:
        best_obj = 0.0
        best_A = []

    return MechanismResult(
        winners=best_A,
        payments={u: p[u] for u in best_A},
        rounds=M,
        queries=oracle.queries,
        history=history,
        u_star=u_star,
    )


def bfm_vm(
    oracle: Oracle,
    costs: np.ndarray,
    budget: float,
    alpha: float,
    ell: int,
) -> MechanismResult:
    """Algorithm 2: BFM-VM (valuation maximization).

    Parameter fixed by Theorem 5.4: alpha = 1+sqrt(3), ell = 2.
    """
    n = oracle.n
    c = np.asarray(costs, dtype=float)
    # Line 1: R = sellers accepting price B; p(u) <- B
    R = [u for u in range(n) if _accepts(c, u, budget)]
    p = {u: float(budget) for u in R}
    if not R:
        return MechanismResult([], {}, 0, oracle.queries, {}, [])
    owner = {u: 0 for u in R}
    # Line 2: t <- 1; rho_1 <- max_{u in R} v(u); S_{i,1} <- empty;
    #         a <- argmax v(u); S_{1,1} <- {a}; owner(a) <- 1
    t = 1
    a = max(R, key=lambda u: oracle.value({u}))
    rho = oracle.value({a})
    S_prev = {i: [] for i in range(1, ell + 1)}
    S_prev[1] = [a]
    owner[a] = 1
    history: dict[tuple[int, int], list[int]] = {(i, 1): list(S_prev[i]) for i in range(1, ell + 1)}

    M = 1
    while True:
        # Line 4: t <- t+1; rho_t <- alpha*rho_{t-1}; S_{i,t} <- empty
        t += 1
        rho = alpha * rho
        S_cur = {i: [] for i in range(1, ell + 1)}
        v_S = {i: 0.0 for i in range(1, ell + 1)}
        # initialize v_S from any seed element already in S_cur (none here)
        # Line 5: iterate over R \ (union S_{i,t-1})
        prev_union = set().union(*[set(S_prev[i]) for i in range(1, ell + 1)])
        candidates = [u for u in R if u not in prev_union]
        for u in candidates:
            if u not in R:
                continue
            # Line 6-7: determine j
            if owner[u] != 0:
                j = owner[u]
            else:
                j = 1
                best_marg = -math.inf
                for i in range(1, ell + 1):
                    marg = oracle.marginal(u, S_cur[i])
                    if marg > best_marg:
                        best_marg = marg
                        j = i
            # Line 8: p(u) <- min{p(u), v(u|S_{j,t})/(rho_t/B)}   [beta=0]
            marg_u = oracle.marginal(u, S_cur[j])
            new_price = marg_u / (rho / budget)
            if new_price < p[u]:
                p[u] = new_price
            # Line 9: if u accepts
            if _accepts(c, u, p[u]):
                # Line 10: if v(S_{j,t}∪{u}) > rho_t  -> break
                if v_S[j] + marg_u > rho:
                    break
                else:
                    # Line 11: add; set owner
                    S_cur[j].append(u)
                    v_S[j] += marg_u
                    if owner[u] == 0:
                        owner[u] = j
            else:
                R = [x for x in R if x != u]
        for i in range(1, ell + 1):
            history[(i, t)] = list(S_cur[i])
        # Line 12: until R \ (union (S_{i,t-1} ∪ S_{i,t})) = empty.
        # S_prev still holds round t-1; S_cur holds round t.
        prev_round_union = set().union(*[set(S_prev[i]) for i in range(1, ell + 1)])
        cur_union = set().union(*[set(S_cur[i]) for i in range(1, ell + 1)])
        remaining = set(R) - (prev_round_union | cur_union)
        S_prev = S_cur
        if not remaining:
            M = t
            break
        if t > 10000:
            M = t
            break

    # Line 14: S* <- argmax over {S_{i,t}: i in [ell], t in {M-1,M}} of v(A)
    best_A: list[int] = []
    best_obj = -math.inf
    cand_rounds = {max(M - 1, 1), M}
    for tt in cand_rounds:
        for i in range(1, ell + 1):
            A = history.get((i, tt), [])
            if not A:
                continue
            vA = oracle.value(A)
            if vA > best_obj:
                best_obj = vA
                best_A = list(A)
    vE = oracle.value(set())
    if vE > best_obj:
        best_A = []

    return MechanismResult(
        winners=best_A,
        payments={u: p[u] for u in best_A},
        rounds=M,
        queries=oracle.queries,
        history=history,
        u_star=[],
    )
