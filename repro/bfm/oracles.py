"""Value-oracle model and concrete submodular functions for the BFM reproduction.

Every valuation function ``v: 2^N -> R_{>=0}`` is exposed through an
:class:`Oracle` wrapper that counts value queries, so mechanisms can be compared
on the hardware-independent query-complexity metric used by the paper
(Section 6, "query complexity ... the standard measure for computational
efficiency in the value oracle model").
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Callable, Iterable, Sequence

import numpy as np


class Oracle:
    """Wraps a set function ``f`` and counts the number of value queries.

    The mechanism only ever learns ``v`` through :meth:`value`; the query count
    is the complexity metric reported by the paper.
    """

    def __init__(self, f: Callable[[frozenset], float], n: int):
        self._f = f
        self.n = n
        self.queries = 0

    def value(self, S: Iterable[int]) -> float:
        fs = frozenset(S)
        self.queries += 1
        return self._f(fs)

    def marginal(self, u: int, S: Iterable[int]) -> float:
        s = frozenset(S)
        vs = self.value(s)
        vu = self.value(s | {u})
        return vu - vs

    def reset(self) -> None:
        """Reset the query counter (a fresh run over the same function)."""
        self.queries = 0


class CoverageFunction:
    """``v(S) = |union_{u in S} T(u)|`` -- a monotone submodular coverage function.

    This is exactly the influence-maximization valuation of the paper
    (Section 6.2): ``v(S) = |union_{u in S} T(u)|`` where ``T(u)`` is the
    out-neighbour set of user ``u``. Coverage is the canonical monotone
    submodular function and is deterministic for a fixed graph.
    """

    def __init__(self, targets: Sequence[Sequence[int]]):
        self.targets = [frozenset(t) for t in targets]
        self.n = len(self.targets)
        self._oracle = Oracle(self._eval, self.n)

    def _eval(self, S: frozenset) -> float:
        acc: set[int] = set()
        for u in S:
            if 0 <= u < self.n:
                acc |= self.targets[u]
        return float(len(acc))

    @property
    def oracle(self) -> Oracle:
        return self._oracle

    def reset(self) -> None:
        self._oracle.queries = 0


class WeightedCoverage:
    """Non-monotone submodular ``v(S) = sum_i w_i * min(1, |S cap T_i)|``
    with *signed* weights, so the function can be non-monotone while staying
    non-negative and submodular. Used to stress-test the general (non-monotone)
    guarantees of Theorems 4.8 and 5.4 on functions that genuinely are not
    monotone.
    """

    def __init__(self, sets: Sequence[Sequence[int]], weights: Sequence[float]):
        self.sets = [frozenset(s) for s in sets]
        self.weights = list(weights)
        self.n = max((max(s) for s in self.sets if s), default=-1) + 1
        self._oracle = Oracle(self._eval, self.n)

    def _eval(self, S: frozenset) -> float:
        total = 0.0
        for s, w in zip(self.sets, self.weights):
            if s & S:
                total += w
        return max(total, 0.0)

    @property
    def oracle(self) -> Oracle:
        return self._oracle

    def reset(self) -> None:
        self._oracle.queries = 0


class CutFunction:
    """Directed-cut coverage ``v(S) = |{(u,w) : u in S, w not in S}|``.

    A classic *non-monotone* submodular function (used widely as a benchmark for
    non-monotone submodular maximization). Non-negative and submodular.
    """

    def __init__(self, edges: Sequence[tuple[int, int]], n: int):
        self.out = [[] for _ in range(n)]
        for u, w in edges:
            self.out[u].append(w)
        self.n = n
        self._oracle = Oracle(self._eval, self.n)

    def _eval(self, S: frozenset) -> float:
        cut = 0
        sup = S
        for u in S:
            if 0 <= u < self.n:
                for w in self.out[u]:
                    if w not in sup:
                        cut += 1
        return float(cut)

    @property
    def oracle(self) -> Oracle:
        return self._oracle

    def reset(self) -> None:
        self._oracle.queries = 0


def brute_force_optimal(oracle: Oracle, n: int, objective: str, costs: np.ndarray | None = None) -> tuple[list[int], float]:
    """Exact optimum over all ``2^n`` subsets (only for small ``n``).

    objective='val'  -> max v(S)
    objective='welfare' -> max v(S) - c(S)
    """
    best_S: list[int] = []
    best = -math.inf
    empty = oracle.value(set())
    base = 0.0
    for r in range(n + 1):
        for comb in combinations(range(n), r):
            v = oracle.value(comb) if r else empty
            obj = v if objective == "val" else v - (sum(costs[i] for i in comb) if costs is not None else 0.0)
            if obj > best:
                best = obj
                best_S = list(comb)
    return best_S, best
