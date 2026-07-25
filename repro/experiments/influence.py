"""Influence-maximization experiment (Claim 6, Section 6 / Figure 1).

Coverage valuation ``v(S)=|N(S)|``, cost ``c(u)=1+sqrt(deg(u))`` (both from the
authors' code), BFM-SWM with the monotone parameters
(``ell=1, alpha=1+sqrt(6)/2, beta=3, eps=0.1``) and budgets 100..1000, on the
three SNAP graphs the paper uses.

Baselines: the three Deng-framework greedies (Distorted / ROI / Cost-Scaled)
with their full Myerson payments when feasible, otherwise their cost-truncated
form, which is an *optimistic* upper bound on baseline welfare (Myerson
payments are >= cost, so the faithful baselines can only do worse). Beating the
optimistic bound is therefore strong evidence for the claim.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from ..bfm.im_baselines import (IMGraph, _greedy_pq, _greedy_distorted, _finalize,
                                _roi_score, _cs_score, deng_roi_im, deng_costscaled_im, deng_distorted_im)
from ..bfm.bfm_im import bfm_swm_im

DATA = Path(__file__).resolve().parents[2] / "data" / "snap"


@dataclass
class IMResult:
    dataset: str
    n: int
    edges: int
    budget: float
    bfm_swm_welfare: float
    distorted_welfare: float
    roi_welfare: float
    costscaled_welfare: float
    best_baseline: str
    improvement_factor: float
    bfm_queries: int
    bfm_winners: int
    baseline_mode: str


def _optimistic_baselines(g: IMGraph, budget: float):
    """Cost-truncated greedies (payments = cost). Optimistic upper bound on the
    faithful Myerson-payment baselines, which pay >= cost and so truncate at
    least as aggressively."""
    out = {}
    for score_fn, name, tk in ((_roi_score, "roi", 1), (_cs_score, "costscaled", 2)):
        order, _, _ = _greedy_pq(g, score_fn, tk)
        out[name] = _finalize(g, order, g.cost.copy(), float(budget), 0).welfare
    rng_kind = __import__("random").Random(12345)
    order, _, _, _ = _greedy_distorted(g, rng_kind)
    out["distorted"] = _finalize(g, order, g.cost.copy(), float(budget), 0).welfare
    return out


def run_im_comparison(g: IMGraph, dataset: str, budgets=range(100, 1001, 100),
                      eps: float = 0.1, faithful: bool = False) -> list[IMResult]:
    alpha = 1 + math.sqrt(6) / 2
    out: list[IMResult] = []
    for B in budgets:
        res = bfm_swm_im(g, float(B), alpha=alpha, beta=3.0, eps=eps, ell=1)
        vS = int((g.cover_of(res.winners)).sum())
        cS = float(sum(g.cost[u] for u in res.winners))
        swm_w = vS - cS

        # By default use the optimistic cost-truncated greedies (fast, and an
        # upper bound on the faithful Myerson baselines which pay >= cost).
        # Faithful Myerson is opt-in for small graphs only (it is O(|S|*n)).
        if faithful and g.n <= 4000:
            r_dist = deng_distorted_im(g, float(B))
            r_roi = deng_roi_im(g, float(B))
            r_cs = deng_costscaled_im(g, float(B))
            welfares = {"distorted": r_dist.welfare, "roi": r_roi.welfare, "costscaled": r_cs.welfare}
            mode = "faithful-myerson"
        else:
            welfares = _optimistic_baselines(g, float(B))
            mode = "optimistic-cost-truncated"
        best_name = max(welfares, key=welfares.get)
        best_w = welfares[best_name]
        factor = swm_w / best_w if best_w > 1e-9 else float("inf")
        out.append(IMResult(dataset, g.n, sum(len(a) for a in g.adj), float(B), swm_w,
                            welfares["distorted"], welfares["roi"], welfares["costscaled"],
                            best_name, factor, res.queries, len(res.winners), mode))
    return out


def load_dataset(name: str) -> IMGraph:
    folder = {"Slashdot": "slashdot", "Email": "email_all", "Epinions": "epinions"}[name]
    return IMGraph.load(DATA / folder)
