"""Deterministic instance generators for the approximation-ratio and runtime
experiments. All randomness is seeded so every run is reproducible."""

from __future__ import annotations

import random
from typing import Callable

import numpy as np

from ..bfm.oracles import Oracle, CoverageFunction, CutFunction, WeightedCoverage


def coverage_instance(n: int, universe: int, degree: int, cost_seed: int, struct_seed: int):
    """A monotone coverage function: each of ``n`` sellers covers ``degree``
    random elements of a ``universe``-sized ground set. Returns ``(make, costs)``
    where ``make()`` yields a fresh :class:`~repro.bfm.oracles.Oracle`."""

    def make():
        rng = random.Random(struct_seed)
        targets = [set(rng.sample(range(universe), degree)) for _ in range(n)]
        return CoverageFunction(targets).oracle

    rngc = random.Random(cost_seed)
    costs = np.array([rngc.uniform(0.02, 0.5) for _ in range(n)])
    return make, costs


def cut_instance(n: int, edge_seed: int, cost_seed: int, edge_mult: float = 3.0):
    """A non-monotone directed-cut function on ``n`` nodes with ~edge_mult*n
    random edges."""

    def make():
        rng = random.Random(edge_seed)
        edges = [(rng.randrange(n), rng.randrange(n)) for _ in range(int(edge_mult * n))]
        return CutFunction(edges, n).oracle

    rngc = random.Random(cost_seed)
    costs = np.array([rngc.uniform(0.02, 0.5) for _ in range(n)])
    return make, costs


def weighted_coverage_instance(n: int, m: int, p: float, cost_seed: int, struct_seed: int):
    """A possibly-non-monotone weighted coverage with signed weights
    (clipped to be non-negative), giving a genuine non-monotone submodular
    function."""

    def make():
        rng = random.Random(struct_seed)
        sets = []
        for _ in range(m):
            s = frozenset(u for u in range(n) if rng.random() < p)
            sets.append(s)
        weights = [rng.uniform(-1.0, 1.0) for _ in range(m)]
        return WeightedCoverage(sets, weights).oracle

    rngc = random.Random(cost_seed)
    costs = np.array([rngc.uniform(0.02, 0.5) for _ in range(n)])
    return make, costs


def coverage_family(n: int, k: int, universe: int = 40, degree: int = 6, base_seed: int = 1000):
    """``k`` monotone coverage instances of size ``n`` (different seeds)."""
    return [coverage_instance(n, universe, degree, base_seed + 2 * i, base_seed + 2 * i + 1) for i in range(k)]


def cut_family(n: int, k: int, base_seed: int = 2000, edge_mult: float = 3.0):
    """``k`` non-monotone cut instances of size ``n``."""
    return [cut_instance(n, base_seed + 2 * i, base_seed + 2 * i + 1, edge_mult) for i in range(k)]


def stressed_coverage_family(k: int = 8, base_seed: int = 5000):
    """Structured coverage instances with a few high-value/low-cost 'premium'
    sellers and many low-value/high-cost 'junk' sellers, so the optimal welfare
    set has high value and low cost.  This makes the theorem's RHS
    ``gamma*v(O) - c(O)`` positive and large, so the inequality check is
    non-vacuous (a bad mechanism actually violates it)."""

    def make_one(seed: int):
        def factory():
            rng = random.Random(seed)
            n = 16
            universe = 60
            n_premium = 3
            per = universe // n_premium
            targets: list[set[int]] = []
            # premium sellers: each covers a disjoint block of `per` targets
            for j in range(n_premium):
                targets.append(set(range(j * per, (j + 1) * per)))
            # junk sellers: cover 0-2 random targets each
            for _ in range(n - n_premium):
                targets.append(set(rng.sample(range(universe), rng.randint(0, 2))))
            from ..bfm.oracles import CoverageFunction
            return CoverageFunction(targets).oracle

        rngc = random.Random(seed + 999)
        costs = np.array([0.04, 0.05, 0.04] + [rngc.uniform(0.3, 0.5) for _ in range(13)])
        return factory, costs

    return [make_one(base_seed + i) for i in range(k)]

