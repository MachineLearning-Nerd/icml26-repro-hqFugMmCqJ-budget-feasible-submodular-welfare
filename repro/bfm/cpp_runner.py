"""Runner for the authors' C++ influence-maximization code
(github.com/xue74193-dot/BFM-SWM). Compiles (installing cmake/g++ if missing)
and runs the four mechanisms on a SNAP dataset, parsing the welfare and
query-count output.

This is the *faithful* C6 evidence: the exact code and data the paper used.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CPP = ROOT / "cpp_im"
DATA = ROOT / "data" / "snap"
FOLDER = {"Slashdot": "slashdot", "Email": "email_all", "Epinions": "epinions"}


def _ensure_tools() -> bool:
    if shutil.which("g++") and shutil.which("cmake"):
        return True
    # try to install on debian-based containers (HF cpu images are debian)
    try:
        subprocess.run(["bash", "-c", "apt-get update -qq && apt-get install -y -qq cmake g++ >/dev/null 2>&1"],
                       check=False, timeout=180)
    except Exception:
        pass
    return bool(shutil.which("g++") and shutil.which("cmake"))


def _set_dataset(util_h: Path, folder: str) -> None:
    txt = util_h.read_text()
    txt = re.sub(r'edge_text="\.\./Graph/[^"]+"', f'edge_text="../Graph/{folder}/renum_edge.txt"', txt)
    txt = re.sub(r'node_text="\.\./Graph/[^"]+"', f'node_text="../Graph/{folder}/renum_node.txt"', txt)
    util_h.write_text(txt)


def run_cpp(dataset: str, budgets=range(100, 1001, 100)) -> dict | None:
    """Compile and run the paper's C++ on ``dataset``. Returns parsed results
    or None if the toolchain is unavailable."""
    if not _ensure_tools():
        return None
    folder = FOLDER[dataset]
    util_h = CPP / "utilityfunction.h"
    _set_dataset(util_h, folder)
    # link graph data where the C++ expects it (../Graph/<folder>/ from build dir)
    graph_link = CPP / "Graph" / folder
    graph_link.parent.mkdir(parents=True, exist_ok=True)
    src = DATA / folder
    if src.exists() and not graph_link.exists():
        os.symlink(src.resolve(), graph_link)
    build = CPP / "build"
    build.mkdir(exist_ok=True)
    (CPP / "result").mkdir(exist_ok=True)
    r = subprocess.run(["cmake", ".."], cwd=build, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    r = subprocess.run(["cmake", "--build", ".", "-j", "4"], cwd=build, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    # edit main.cpp budget range in place is avoided; the default 100..1000 step 100 is what we want
    proc = subprocess.run(["./Coverage"], cwd=build, capture_output=True, text=True, timeout=3600)
    out = proc.stdout
    blocks = re.findall(
        r"(BFM-SWM|Deng-Distorted|Deng-ROI|Deng-CostScaled) & (?:Max )?Budget: (\d+).*?"
        r"objective values: ([\d.\-]+).*?oracle queries: (\d+)", out, re.S)
    results: dict[int, dict] = {}
    for mech, B, val, q in blocks:
        results.setdefault(int(B), {})[mech] = {"welfare": float(val), "queries": int(q)}
    return {"dataset": dataset, "results": results}


def parse_committed(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    return None
