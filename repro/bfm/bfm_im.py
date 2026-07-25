"""Fast numpy-based BFM-SWM for the influence-maximization coverage graph.

Identical algorithm to :func:`repro.bfm.mechanisms.bfm_swm` (Algorithm 1, with
the monotone parameters ell=1), but value/marginal queries use a numpy boolean
``covered`` array and the active-set uses a numpy boolean mask, so a 77K-node
SNAP graph runs in a few seconds rather than a minute.
"""

from __future__ import annotations

import math
import numpy as np

from .im_baselines import IMGraph
from .mechanisms import MechanismResult


def bfm_swm_im(g: IMGraph, budget: float, alpha: float, beta: float, eps: float, ell: int) -> MechanismResult:
    n = g.n
    cost = g.cost
    active = cost <= budget                      # boolean mask for R
    n_active = int(active.sum())
    p = np.full(n, float(budget))
    t = 0
    rho = eps / alpha
    u_star: list[int] = []
    owner = np.zeros(n, dtype=np.int32)
    S_prev = {i: [] for i in range(1, ell + 1)}
    history: dict[tuple[int, int], list[int]] = {}
    queries = 0
    M = 0

    def covered_of(S):
        c = np.zeros(n, dtype=bool)
        for u in S:
            a = g.adj_np[u]
            if a.size:
                c[a] = True
        return c

    def val(S):
        nonlocal queries
        queries += 1
        return int(covered_of(S).sum())

    def marg(u, covered):
        nonlocal queries
        queries += 1
        a = g.adj_np[u]
        return int((~covered[a]).sum()) if a.size else 0

    while True:
        t += 1
        rho = alpha * rho
        S_cur = {i: [] for i in range(1, ell + 1)}
        covered_cur = {i: np.zeros(n, dtype=bool) for i in range(1, ell + 1)}
        v_S = {i: 0 for i in range(1, ell + 1)}
        p_S = {i: 0.0 for i in range(1, ell + 1)}
        in_prev = np.zeros(n, dtype=bool)
        for i in range(1, ell + 1):
            for u in S_prev[i]:
                in_prev[u] = True
        for u in u_star:
            in_prev[u] = True
        cand_mask = active & (~in_prev)
        candidates = np.nonzero(cand_mask)[0].tolist()
        for u in candidates:
            if not active[u]:
                continue
            if owner[u] != 0:
                j = int(owner[u])
            else:
                j = 1
                best_m = -1.0
                for i in range(1, ell + 1):
                    m = marg(u, covered_cur[i])
                    if m > best_m:
                        best_m = m; j = i
            marg_u = marg(u, covered_cur[j])
            new_price = marg_u / (beta + rho / budget)
            if new_price < p[u]:
                p[u] = new_price
            if cost[u] <= p[u]:
                if (v_S[j] + marg_u) - (p_S[j] + p[u]) > rho:
                    u_star = [u]
                    break
                else:
                    S_cur[j].append(u)
                    v_S[j] += marg_u
                    p_S[j] += p[u]
                    a = g.adj_np[u]
                    if a.size:
                        covered_cur[j][a] = True
                    if owner[u] == 0:
                        owner[u] = j
            else:
                active[u] = False
        for i in range(1, ell + 1):
            history[(i, t)] = list(S_cur[i])
        in_prev_round = np.zeros(n, dtype=bool)
        for i in range(1, ell + 1):
            for u in S_prev[i]:
                in_prev_round[u] = True
        cur_union = np.zeros(n, dtype=bool)
        for i in range(1, ell + 1):
            for u in S_cur[i]:
                cur_union[u] = True
        for u in u_star:
            in_prev_round[u] = True
        remaining = active & (~in_prev_round) & (~cur_union)
        S_prev = S_cur
        if not remaining.any():
            M = t
            break
        if t > 10000:
            M = t
            break

    best_A: list[int] = []
    best_obj = -math.inf
    for tt in {max(M - 1, 1), M}:
        for i in range(1, ell + 1):
            A = history.get((i, tt), [])
            if not A:
                continue
            vA = val(A)
            pA = sum(p[u] for u in A)
            obj = vA - pA
            if obj > best_obj:
                best_obj = obj; best_A = list(A)
    if u_star:
        vU = val(u_star); pU = sum(p[u] for u in u_star)
        if vU - pU > best_obj:
            best_obj = vU - pU; best_A = list(u_star)
    if 0 > best_obj:
        best_A = []
    return MechanismResult(best_A, {u: float(p[u]) for u in best_A}, M, queries, history, u_star)
