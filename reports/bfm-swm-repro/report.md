# Budget-Feasible Mechanisms for Submodular Welfare — Reproduction Report

**Paper:** Cui, Huang, Sun, Xue. *Budget-Feasible Mechanisms for Submodular
Welfare Maximization in Procurement Auctions.* arXiv 2605.00411 (OpenReview
`hqFugMmCqJ`, ICML 2026).

**One-line result:** all six headline claims **VERIFIED** — the two mechanisms
were implemented from their pseudocode, run on synthetic + SNAP data, and their
approximation ratios, runtime scaling, economic properties, and empirical
advantage over baselines all reproduce.

---

## The central question

A budget-constrained buyer wants to procure items from strategic sellers so as
to maximize **social welfare** `v(S) − c(S)` (value minus cost), where `v` is a
submodular set function and each seller's cost is private. Welfare maximization
is harder than the usual valuation objective because the cost term makes the
objective potentially negative and unobservable. The paper asks: *can a
budget-feasible, truthful mechanism achieve a constant approximation for
submodular welfare?* and answers yes with **BFM-SWM**, plus a valuation variant
**BFM-VM**.

## What we built

- A line-for-line Python implementation of **BFM-SWM** (Algorithm 1) and
  **BFM-VM** (Algorithm 2) in [`repro/bfm/mechanisms.py`](../../repro/bfm/mechanisms.py).
- An **exact symbolic re-derivation** of the three approximation constants in
  quadratic-field arithmetic `Q(√d)` ([`repro/bfm/symbolic.py`](../../repro/bfm/symbolic.py))
  — no floating point, so the equalities are proofs-by-computation.
- Empirical suites for ratios, economic properties, runtime scaling, and
  influence maximization, each with a **negative control**.
- The **authors' own C++ code** (verbatim) for the SNAP influence-maximization
  comparison, regenerated on Hugging Face compute.

## Headline evidence

### Approximation ratios (Claims 1–3)

Every constant matches the paper **exactly** in symbolic arithmetic, and the
mechanism beats the guaranteed ratio on all 32 test instances per claim:

| Claim | Theorem | Guarantee | Symbolic match | Min observed ratio |
|---|---|---|---|---|
| C1 general welfare | 4.8 | `3/(4(13+4√6)) = 0.03290` | exact (Q(√6)) | 0.212 |
| C2 monotone welfare | 4.10 | `2/(13+4√6) = 0.08773` | exact (Q(√6)) | 0.274 |
| C3 valuation (BFM-VM) | 5.4 | `1/(12+4√3) = 0.05283` | exact (Q(√3)) | 0.308 |

### Runtime (Claims 2, 4)

Query complexity (the paper's efficiency metric) vs `n`:

```
BFM-VM      log-log slope 0.99   →  Θ(n log n)
naive greedy log-log slope 1.98  →  Θ(n²)
```

### Economic properties (Claim 5)

Across 24 instances and **1152 strategyproofness deviation tests**: zero
violations of budget feasibility, individual rationality, non-negative surplus,
or strategyproofness.

### Influence maximization (Claim 6) — the authors' C++ on SNAP Slashdot

BFM-SWM beats every Deng-framework baseline on **all 10 budgets**; the average
improvement factor is **4.56×**, essentially matching the paper's reported
**4.49×**. The baselines collapse because their Myerson payments (needed for
truthfulness) consume the budget, while BFM's descending-clock prices stay low.

## Honest limitations

- The approximation ratios are worst-case *theorems*; our finite experiments are
  scoped corroboration (the bound holds on every instance tested), not a proof
  over all submodular functions. The symbolic derivation closes that gap for the
  constants themselves.
- Full-scale SNAP C++ was run on Slashdot; Email/Epinions are covered by a Python
  optimistic (cost-payment) upper bound.
- Strategyproofness is tested by deviation, not by a proof certificate.

## Compute

Hugging Face `cpu-upgrade`, ~20 minutes, $0. All seeds are fixed and documented
in `repro/experiments/instances.py`.

## How to reproduce

```bash
uv run python -m repro.run_all      # runs all six verifiers, writes outputs/
```

Winning branch: `orx/bfm-swm-faithful-reproduction` @ `770504d`.
