"""Quick local smoke test (must finish < 60 s): mechanisms run + invariants hold.

Run:  uv run python -m repro.smoke
"""

from __future__ import annotations

import math
import random

import numpy as np

from repro.bfm import (
    CoverageFunction,
    WeightedCoverage,
    CutFunction,
    bfm_swm,
    bfm_vm,
    brute_force_optimal,
)


def check_swm_invariants(res, v, c, B, beta):
    p = res.payments
    S = res.winners
    pS = sum(p[u] for u in S)
    vS = v(S)
    # budget feasibility
    assert pS <= B + 1e-9, f"budget violated: {pS} > {B}"
    # IR
    for u in S:
        assert p[u] >= c[u] - 1e-9, f"IR violated for {u}: {p[u]} < {c[u]}"
    # non-negative surplus
    assert vS - pS >= -1e-9, f"surplus negative: {vS}-{pS}={vS - pS}"
    # beta ratio (v(S) >= beta*p(S))
    if S:
        assert vS >= beta * pS - 1e-6, f"beta ratio violated: {vS} < {beta}*{pS}"


def check_vm_invariants(res, v, c, B):
    p = res.payments
    S = res.winners
    pS = sum(p[u] for u in S)
    vS = v(S)
    assert pS <= B + 1e-9, f"budget violated: {pS} > {B}"
    for u in S:
        assert p[u] >= c[u] - 1e-9, f"IR violated for {u}"


def main():
    rng = random.Random(42)
    print("== BFM-SWM on coverage (monotone, ell=1) ==")
    targets = [set(rng.sample(range(30), rng.randint(2, 8))) for _ in range(12)]
    cov = CoverageFunction(targets)
    c = np.array([rng.uniform(0.05, 0.5) for _ in range(12)])
    B = 1.0
    alpha = 1 + math.sqrt(6) / 2
    res = bfm_swm(cov.oracle, c, B, alpha=alpha, beta=3.0, eps=0.1, ell=1)
    O, vO = brute_force_optimal(cov.oracle, cov.n, "welfare", c)
    vS = cov.oracle.value(res.winners)
    cS = sum(c[u] for u in res.winners)
    ratio = (vS - cS) / vO if vO > 0 else float("inf")
    check_swm_invariants(res, lambda S: cov.oracle.value(S), c, B, 3.0)
    print(f"  winners={res.winners} rounds={res.rounds} queries={res.queries}")
    print(f"  v(S)-c(S)={vS - cS:.4f}  OPT welfare={vO:.4f}  ratio={ratio:.4f} (>=0.0877?)")
    assert ratio >= 0.0877 - 1e-3, f"monotone ratio {ratio} < 0.0877"

    print("== BFM-SWM on cut (non-monotone, ell=2) ==")
    edges = [(rng.randrange(10), rng.randrange(10)) for _ in range(30)]
    cut = CutFunction(edges, 10)
    c2 = np.array([rng.uniform(0.05, 0.5) for _ in range(10)])
    alpha2 = 1 + 2 * math.sqrt(6) / 3
    res2 = bfm_swm(cut.oracle, c2, B, alpha=alpha2, beta=4.0, eps=0.1, ell=2)
    O2, vO2 = brute_force_optimal(cut.oracle, cut.n, "welfare", c2)
    vS2 = cut.oracle.value(res2.winners)
    cS2 = sum(c2[u] for u in res2.winners)
    ratio2 = (vS2 - cS2) / vO2 if vO2 > 0 else float("inf")
    check_swm_invariants(res2, lambda S: cut.oracle.value(S), c2, B, 4.0)
    print(f"  winners={res2.winners} rounds={res2.rounds} queries={res2.queries}")
    print(f"  v(S)-c(S)={vS2 - cS2:.4f}  OPT={vO2:.4f}  ratio={ratio2:.4f} (>=0.0328?)")

    print("== BFM-VM on coverage (valuation, ell=2) ==")
    cov2 = CoverageFunction([set(rng.sample(range(25), rng.randint(2, 7))) for _ in range(10)])
    c3 = np.array([rng.uniform(0.05, 0.5) for _ in range(10)])
    alpha3 = 1 + math.sqrt(3)
    res3 = bfm_vm(cov2.oracle, c3, B, alpha=alpha3, ell=2)
    O3, vO3 = brute_force_optimal(cov2.oracle, cov2.n, "val", None)
    vS3 = cov2.oracle.value(res3.winners)
    ratio3 = vS3 / vO3 if vO3 > 0 else float("inf")
    check_vm_invariants(res3, lambda S: cov2.oracle.value(S), c3, B)
    print(f"  winners={res3.winners} rounds={res3.rounds} queries={res3.queries}")
    print(f"  v(S)={vS3:.4f}  OPT val={vO3:.4f}  ratio={ratio3:.4f} (>=0.0528?)")
    assert ratio3 >= 1 / (12 + 4 * math.sqrt(3)) - 1e-3, f"VM ratio {ratio3} < 0.0528"

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
