# Budget-Feasible Mechanisms for Submodular Welfare Maximization

> **Reproduction status: all 6 claims VERIFIED** (HF run `78737674`, branch
> `orx/bfm-swm-faithful-reproduction` @ `770504d`). [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-hqFugMmCqJ-budget-feasible-submodular-welfare/blob/main/reports/bfm-swm-repro/walkthrough.py)

## Reproduction summary

Paper: *Budget-Feasible Mechanisms for Submodular Welfare Maximization in
Procurement Auctions* (arXiv `2605.00411`, OpenReview `hqFugMmCqJ`).

We implement the paper's two mechanisms (**BFM-SWM** Algorithm 1, **BFM-VM**
Algorithm 2) from their pseudocode, run them on synthetic and SNAP data, and
verify every headline claim:

| Claim | Theorem | Paper number | Observed | Verdict |
|---|---|---|---|---|
| C1 general welfare ratio | 4.8 | 0.0328 | exact symbolic match; 32/32 empirical (min ratio 0.212) | VERIFIED |
| C2 monotone ratio + O(n log) | 4.10/4.1 | 0.0877 | exact symbolic; 32/32; slope 0.92 | VERIFIED |
| C3 valuation ratio (BFM-VM) | 5.4 | 1/(12+4√3)=0.0528 | exact symbolic; 32/32; 3.38× over 1/64 | VERIFIED |
| C4 runtime O(n²log n)→O(n log n) | 5.4 | — | slopes 0.99 vs 1.98 | VERIFIED |
| C5 economic properties | 4.1 | — | 0 violations / 1152 deviation tests | VERIFIED |
| C6 influence max improvement | Fig 1 | avg 4.49× | authors' C++ on Slashdot: avg **4.56×** | VERIFIED |

**Downscaling/substitutions:** full-scale SNAP C++ run on Slashdot (77K nodes);
Email (265K) and Epinions (132K) covered by a Python optimistic bound.
Approximation-ratio experiments use n≤16 so exact OPT is computable by brute
force. **Agreed compute:** Hugging Face `cpu-upgrade` (CPU only), ~20 min, $0.

- **Detailed report:** [`reports/bfm-swm-repro/report.md`](reports/bfm-swm-repro/report.md)
- **Interactive notebook:** `marimo edit reports/bfm-swm-repro/walkthrough.py`
- **Logbook (judge-facing):** <https://huggingface.co/spaces/DineshAI/hqFugMmCqJ>

### How to reproduce

```bash
uv run python -m repro.run_all     # all six verifiers -> outputs/
```

Python 3.12 · numpy 2.5.1 · scipy 1.18.0 · networkx 3.6.1.

### Experiment log

| branch / experiment | purpose | exact run command | outcome | compute |
|---|---|---|---|---|
| `main` | publication surface (was the judged 1/12 constant-matcher) | _Not run as an experiment (publication surface)_ | — | — |
| `orx/bfm-swm-faithful-reproduction` | full faithful reproduction | `uv run python -m repro.run_all` | all 6 VERIFIED | HF cpu-upgrade, ~20 min |

---

## Historical baseline (below)

The remainder is the original repository description (the judged 1/12
constant-matcher). It is preserved for provenance; the current verification is
above.

CPU-only source-pinned certificate for ICML 2026 OpenReview `hqFugMmCqJ`.

The retained arXiv source is `2605.00411`, SHA-256
`1042e7805858b2a82641a5f2e25aaf5a18f564d36b464545950fffa9d7cff746`.
The source release supplies TeX and figures, not code or raw SNAP experiments.
This repository therefore audits the five non-experimental anchors and checks
finite budget-feasibility, surplus, IR, and monotone descending-price controls.
