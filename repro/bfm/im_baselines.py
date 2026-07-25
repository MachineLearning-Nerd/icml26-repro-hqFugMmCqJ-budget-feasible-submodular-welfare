"""Faithful Python port of the paper's influence-maximization baselines
(``Deng-Distorted``, ``Deng-ROI``, ``Deng-CostScaled``) from the authors' C++
reference (github.com/xue74193-dot/BFM-SWM, "Influence Maximization/").

Each baseline:
  1. runs its greedy to build a candidate set S (with snapshots),
  2. computes Myerson threshold payments (>= cost) for each element of S,
  3. truncates S from the end until total payment <= budget,
and reports the welfare ``v(S) - c(S)`` with ``v`` the coverage function.

Cost model and parameters exactly match the reference code:
  cost(u) = 1 + sqrt(degree(u));  budgets 100..1000.

Coverage marginals use numpy boolean arrays so 77K-node graphs run in minutes.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class IMGraph:
    n: int
    adj: list[list[int]]
    adj_np: list[np.ndarray]
    degree: np.ndarray
    cost: np.ndarray

    @staticmethod
    def load(data_dir: Path | str) -> "IMGraph":
        d = Path(data_dir)
        deg: dict[int, int] = {}
        for line in (d / "renum_node.txt").read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                try:
                    nid = int(parts[0]); dg = int(parts[2])
                except ValueError:
                    continue
                deg[nid] = dg
        n = max(deg) + 1 if deg else 0
        adj: list[list[int]] = [[] for _ in range(n)]
        for line in (d / "renum_edge.txt").read_text().splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                a, b = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if a != b and 0 <= a < n and 0 <= b < n:
                adj[a].append(b)
        adj = [sorted(set(xs)) for xs in adj]
        adj_np = [np.array(a, dtype=np.int32) for a in adj]
        degree = np.array([len(adj[i]) for i in range(n)], dtype=float)
        cost = 1.0 + np.sqrt(degree)
        return IMGraph(n, adj, adj_np, degree, cost)

    def marginal(self, u: int, covered: np.ndarray) -> int:
        a = self.adj_np[u]
        return int((~covered[a]).sum()) if a.size else 0

    def add_cover(self, u: int, covered: np.ndarray) -> None:
        if self.adj_np[u].size:
            covered[self.adj_np[u]] = True

    def cover_of(self, S: list[int]) -> np.ndarray:
        covered = np.zeros(self.n, dtype=bool)
        for u in S:
            self.add_cover(u, covered)
        return covered


@dataclass
class IMRunResult:
    winners: list[int]
    welfare: float
    payment: float
    cost: float
    queries: int


def _roi_score(mg, c):
    return (mg - c) / c


def _cs_score(mg, c):
    return mg - 2 * c


def _greedy_pq(g: IMGraph, score_fn, type_kind: int):
    """Lazy priority-queue greedy. Returns (solution_order, snapshots, queries).
    snapshots[i] = list copy of S just before solution_order[i] was added."""
    n = g.n
    covered = np.zeros(n, dtype=bool)
    S: list[int] = []
    in_S = np.zeros(n, dtype=bool)
    snapshots: list[list[int]] = [[]]
    queries = 0
    pq: list[tuple[float, int, int, int]] = []
    counter = 0
    for u in range(n):
        mg = g.marginal(u, covered); queries += 1
        sc = score_fn(mg, g.cost[u])
        heapq.heappush(pq, (-sc, counter, u, 0)); counter += 1
    while pq:
        negsc, _, u, it = heapq.heappop(pq)
        if in_S[u]:
            continue
        if it < len(S):
            mg = g.marginal(u, covered); queries += 1
            sc = score_fn(mg, g.cost[u])
            heapq.heappush(pq, (-sc, counter, u, len(S))); counter += 1
            continue
        sc = -negsc
        if sc > 1e-9:
            S.append(u); in_S[u] = True
            g.add_cover(u, covered)
            snapshots.append(list(S))
        else:
            break
    return S, snapshots, queries


def _greedy_distorted(g: IMGraph, rng: random.Random):
    n = g.n
    perm = [rng.randrange(n) for _ in range(n)]
    base = 1.0 - 1.0 / n
    covered = np.zeros(n, dtype=bool)
    in_S = np.zeros(n, dtype=bool)
    S: list[int] = []
    snapshots: list[list[int]] = [[]]
    queries = 0
    for k in range(1, n + 1):
        u = perm[k - 1]
        if in_S[u]:
            continue
        mg = g.marginal(u, covered); queries += 1
        gamma = base ** (n - k)
        if gamma * mg - g.cost[u] > 1e-9:
            S.append(u); in_S[u] = True
            g.add_cover(u, covered)
            snapshots.append(list(S))
    return S, snapshots, queries, perm


def _feasible_prefix(g: IMGraph, order: list[int], budget: float) -> int:
    """Largest k such that sum of costs of order[:k] <= budget. Since every
    Myerson payment p_i >= cost_i, no element past this prefix can survive the
    budget-truncation step, so we only need payments for order[:k]."""
    acc = 0.0
    for k, u in enumerate(order):
        if acc + g.cost[u] > budget:
            return k
        acc += g.cost[u]
    return len(order)


def _myerson_payments_pq(g: IMGraph, order: list[int], snapshots, score_fn, type_kind: int, budget: float):
    """Myerson threshold payments for the ROI/CostScaled greedy. Payments are
    computed only for the cost-feasible prefix (the rest cannot survive)."""
    n = g.n
    limit = _feasible_prefix(g, order, budget)
    payments = np.zeros(n)
    queries = 0
    for idx in range(limit):
        i = order[idx]
        snap = snapshots[idx]
        covered = g.cover_of(snap)
        in_cur = np.zeros(n, dtype=bool)
        for u in snap:
            in_cur[u] = True
        p_i = float(g.cost[i])
        cur_S = list(snap)
        pq: list[tuple[float, int, int, int]] = []
        counter = 0
        for u in range(n):
            if u == i or in_cur[u]:
                continue
            mg = g.marginal(u, covered); queries += 1
            sc = score_fn(mg, g.cost[u])
            heapq.heappush(pq, (-sc, counter, u, len(cur_S))); counter += 1
        for _ in range(len(snap), n - 1):
            mg_i = g.marginal(i, covered); queries += 1
            best_sc = -1e18; best_u = -1
            while pq:
                negsc, _, u, it = heapq.heappop(pq)
                if in_cur[u]:
                    continue
                if it < len(cur_S):
                    mg_u = g.marginal(u, covered); queries += 1
                    sc = score_fn(mg_u, g.cost[u])
                    heapq.heappush(pq, (-sc, counter, u, len(cur_S))); counter += 1
                    continue
                best_sc = -negsc; best_u = u
                heapq.heappush(pq, (negsc, counter, u, it)); counter += 1
                break
            if best_sc < 1e-9:
                break
            max_other = max(0.0, best_sc)
            if type_kind == 1:
                threshold = mg_i / (1.0 + max_other)
            else:
                threshold = (mg_i - max_other) / 2.0
            if threshold > p_i:
                p_i = threshold
            if best_u != -1 and best_sc > 1e-9:
                cur_S.append(best_u); in_cur[best_u] = True
                g.add_cover(best_u, covered)
        payments[i] = p_i
    return payments, queries


def _distorted_payments(g: IMGraph, order: list[int], snapshots, perm: list[int], gammas, budget: float):
    n = g.n
    limit = _feasible_prefix(g, order, budget)
    payments = np.zeros(n)
    queries = 0
    for idx in range(limit):
        i = order[idx]
        snap = snapshots[idx]
        covered = g.cover_of(snap)
        in_cur = np.zeros(n, dtype=bool)
        for u in snap:
            in_cur[u] = True
        p_i = float(g.cost[i])
        cur_S = list(snap)
        for k in range(len(snap) + 1, n + 1):
            u = perm[k - 1]
            if u == i or in_cur[u]:
                continue
            mg_i = g.marginal(i, covered); queries += 1
            mg_u = g.marginal(u, covered); queries += 1
            threshold = gammas[k] * mg_i
            if threshold > p_i:
                p_i = threshold
            if gammas[k] * mg_u - g.cost[u] > 1e-9:
                cur_S.append(u); in_cur[u] = True
                g.add_cover(u, covered)
        payments[i] = p_i
    return payments, queries


def _finalize(g: IMGraph, order: list[int], payments: np.ndarray, budget: float, queries: int) -> IMRunResult:
    total = float(sum(payments[u] for u in order))
    S = list(order)
    while total > budget + 1e-9 and S:
        last = S.pop()
        total -= payments[last]
        payments[last] = 0.0
    covered = g.cover_of(S)
    vS = int(covered.sum())
    cS = float(sum(g.cost[u] for u in S))
    return IMRunResult(S, vS - cS, total, cS, queries)


def deng_roi_im(g: IMGraph, budget: float) -> IMRunResult:
    order, snaps, q = _greedy_pq(g, _roi_score, 1)
    pay, q2 = _myerson_payments_pq(g, order, snaps, _roi_score, 1, budget)
    return _finalize(g, order, pay, budget, q + q2)


def deng_costscaled_im(g: IMGraph, budget: float) -> IMRunResult:
    order, snaps, q = _greedy_pq(g, _cs_score, 2)
    pay, q2 = _myerson_payments_pq(g, order, snaps, _cs_score, 2, budget)
    return _finalize(g, order, pay, budget, q + q2)


def deng_distorted_im(g: IMGraph, budget: float, seed: int = 12345) -> IMRunResult:
    rng = random.Random(seed)
    n = g.n
    order, snaps, q, perm = _greedy_distorted(g, rng)
    base = 1.0 - 1.0 / n
    gammas = [0.0] + [base ** (n - k) for k in range(1, n + 1)]
    pay, q2 = _distorted_payments(g, order, snaps, perm, gammas, budget)
    return _finalize(g, order, pay, budget, q + q2)
