"""Fixed run command for every experiment node:

    uv run python -m repro.run_all

Runs every claim verifier (C1-C6), writes machine-readable evidence under
``outputs/`` and a human-readable ``outputs/EVAL.md``. Exits nonzero if any
claim that is expected to pass fails its check.

The C6 (influence-maximization) verifier attempts the SNAP datasets used by the
paper and falls back to a synthetic graph (clearly labelled) when the download
is unavailable, so the suite always completes.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sha = _git_sha()
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": str(_cpu_count()),
        "host": socket.gethostname(),
        "git_sha": sha,
        "run_utc": datetime.now(timezone.utc).isoformat(),
    }
    print(f"BFM-SWM reproduction — git {sha} — {env['python']} — {env['host']}")

    from .verify import (verify_claim1, verify_claim2, verify_claim3,
                          verify_claim4, verify_claim5, verify_claim6)

    _section("C1: general submodular welfare ratio (Theorem 4.8, 0.0328)")
    v1 = verify_claim1()
    _emit(v1)

    _section("C2: monotone welfare ratio (Theorem 4.10, 0.0877)")
    v2 = verify_claim2()
    _emit(v2)

    _section("C3: BFM-VM valuation ratio (Theorem 5.4, 1/(12+4sqrt3))")
    v3 = verify_claim3()
    _emit(v3)

    _section("C4: runtime complexity (O(n log n) vs O(n^2))")
    v4 = verify_claim4()
    _emit(v4)

    _section("C5: economic properties (Theorem 4.1)")
    v5 = verify_claim5()
    _emit(v5)

    _section("C6: influence maximization (Figure 1)")
    use_snap = not _flag_synthetic_only()
    v6 = verify_claim6(use_snap=use_snap)
    _emit(v6)

    elapsed = time.time() - t0
    verdicts = {v.claim: v.status for v in (v1, v2, v3, v4, v5, v6)}
    summary = {
        "env": env,
        "elapsed_sec": round(elapsed, 1),
        "verdicts": verdicts,
        "all_expected_pass": all(v.passed for v in (v1, v2, v3, v4, v5, v6)),
    }
    (OUT / "claim_verdicts.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    _write_eval_md(env, elapsed, v1, v2, v3, v4, v5, v6)
    print(f"\nVERDICTS: {verdicts}")
    print(f"elapsed: {elapsed:.1f}s  ->  outputs/")
    # exit nonzero only if a VERIFIED-expected claim failed (not BLOCKED)
    failed = [v.claim for v in (v1, v2, v3, v4, v5, v6) if not v.passed and v.status != "BLOCKED"]
    return 1 if failed else 0


def _cpu_count() -> int:
    import os
    return os.cpu_count() or 1


def _flag_synthetic_only() -> bool:
    import os
    # Synthetic-only is selected by a committed config file, never an env var.
    return (ROOT / ".synthetic_only").exists()


def _emit(v) -> None:
    data = {"claim": v.claim, "status": v.status, "passed": v.passed,
            "evidence": v.evidence, "negative_control": v.negative_control}
    (OUT / f"claim_{v.claim.lower()}.json").write_text(json.dumps(data, indent=2, default=str) + "\n")
    mark = "OK" if v.passed else ("BLOCK" if v.status == "BLOCKED" else "FAIL")
    nc = v.negative_control
    print(f"  {v.claim}: {v.status} [{mark}]  negative-control: {nc}")


def _write_eval_md(env, elapsed, *verdicts) -> None:
    lines = [
        "# EVAL — BFM-SWM reproduction (hqFugMmCqJ)",
        "",
        f"- git SHA: `{env['git_sha']}`",
        f"- python: {env['python']} on {env['platform']} ({env['host']}, {env['cpu_count']} cores)",
        f"- run UTC: {env['run_utc']}",
        f"- elapsed: {elapsed:.1f}s",
        "",
        "| Claim | Status | Passed | Negative control |",
        "|---|---|---|---|",
    ]
    for v in verdicts:
        lines.append(f"| {v.claim} | {v.status} | {v.passed} | {v.negative_control} |")
    lines += ["", f"Source paper: arXiv 2605.00411 (OpenReview hqFugMmCqJ).", ""]
    (OUT / "EVAL.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
