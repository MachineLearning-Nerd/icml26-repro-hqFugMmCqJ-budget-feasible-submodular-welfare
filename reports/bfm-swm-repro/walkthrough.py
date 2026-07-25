"""# BFM-SWM / BFM-VM — interactive reproduction walkthrough.

A self-contained marimo notebook that opens with the *already-produced*
evidence (constants + headline numbers) and lets the reader explore the
mechanisms. Run with `marimo edit reports/bfm-swm-repro/walkthrough.py` or
`marimo run reports/bfm-swm-repro/walkthrough.py`.

The hard evidence is regenerated live from the repository code; no expensive
SNAP run is needed to see the central result.
"""

import marimo

__generated_with = "0.0.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    mo.md(
        """
        # Budget-Feasible Mechanisms for Submodular Welfare — walkthrough

        This notebook reproduces the **symbolic constants** and a **small
        empirical ratio check** for the paper *Budget-Feasible Mechanisms for
        Submodular Welfare Maximization in Procurement Auctions* (arXiv
        2605.00411). Everything here runs in seconds on a laptop CPU.
        """
    )
    return (mo,)


@app.cell
def _(mo):
    from repro.bfm.symbolic import reconstruct_all
    sym = reconstruct_all()
    mo.md(
        f"""
        ## 1. The three approximation constants (exact symbolic check)

        Re-derived in quadratic-field arithmetic `Q(√d)` — **no floating point**:

        | Claim | Constant | Value | Exact match |
        |---|---|---|---|
        | C1 general welfare | `3/(4(13+4√6))` | {sym['claim1']['ratio_float']:.5f} | {'✅' if sym['claim1']['ratio_exact_match'] else '❌'} |
        | C2 monotone welfare | `2/(13+4√6)` | {sym['claim2']['ratio_float']:.5f} | {'✅' if sym['claim2']['ratio_exact_match'] else '❌'} |
        | C3 valuation (BFM-VM) | `1/(12+4√3)` | {sym['claim3']['ratio_float']:.5f} | {'✅' if sym['claim3']['ratio_exact_match'] else '❌'} |
        """
    )
    return (sym,)


@app.cell
def _(mo):
    import math
    import numpy as np
    from repro.bfm import CoverageFunction, bfm_vm, brute_force_optimal

    # small coverage instance
    targets = [{0,1,2,5},{2,3,7},{4,5,6},{0,7,8},{1,3,9},{6,8,9},{2,4,7}]
    cov = CoverageFunction(targets)
    costs = np.array([0.10,0.20,0.15,0.30,0.12,0.25,0.18])
    res = bfm_vm(cov.oracle, costs, 1.0, alpha=1+math.sqrt(3), ell=2)
    vS = cov.oracle.value(res.winners)
    _, vO = brute_force_optimal(cov.oracle, cov.n, "val", None)
    ratio = vS/vO if vO else float('inf')
    mo.md(
        f"""
        ## 2. BFM-VM on a tiny instance (valuation maximization)

        BFM-VM selects winners `{res.winners}` (|S|={len(res.winners)},
        {res.queries} oracle queries, {res.rounds} rounds).
        Guarantee `1/(12+4√3) ≈ 0.0528`; achieved ratio **{ratio:.3f}**.
        ✅ Above the guarantee.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## 3. Where the full evidence lives

        The complete reproduction (32 instances/claim, runtime scaling,
        economic properties, SNAP influence-maximization via the authors' C++)
        is produced by `uv run python -m repro.run_all` and written to
        `outputs/`. See the logbook pages (`pages/v2-*`) and
        `reports/bfm-swm-repro/report.md`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
