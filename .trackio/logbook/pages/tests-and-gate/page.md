# Tests and gate


---
<!-- trackio-cell
{"type": "code", "id": "cell_b7a2848d5cd9", "created_at": "2026-07-22T12:58:46+00:00", "title": "Run publication gate", "command": [".venv/bin/python", "repro/src/run_publication_gate.py"], "exit_code": 0, "duration_s": 0.21}
-->
````bash
$ .venv/bin/python repro/src/run_publication_gate.py
````

exit 0 · 0.2s


````python title=run_publication_gate.py
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
subprocess.run([sys.executable,'repro/src/verify_budget_feasible.py','--output','outputs/verification.json'],cwd=ROOT,check=True)
subprocess.run([sys.executable,'-m','unittest','discover','-s','repro/tests','-v'],cwd=ROOT,check=True)
v=json.loads((ROOT/'outputs/verification.json').read_text()); assert v['verified_claims']==5
g={'paper':'hqFugMmCqJ','gate':'passed','tests_passed':True,'publication_gate_passed':True,'verified_claims':5,'scope':v['scope']}
(ROOT/'outputs/publication_gate.json').write_text(json.dumps(g,indent=2)+'\n');print(json.dumps(g,indent=2))

````


````output
{
  "paper": "hqFugMmCqJ",
  "source_sha256": "1042e7805858b2a82641a5f2e25aaf5a18f564d36b464545950fffa9d7cff746",
  "scope": "Source-pinned theorem contract plus finite descending-clock budget/IR/monotonicity controls; not a rerun of unreleased large SNAP experiments.",
  "claims": {
    "C1": {
      "status": "verified",
      "general_constant": 0.032897681803431035
    },
    "C2": {
      "status": "verified",
      "monotone_constant": 0.0877271514758161,
      "runtime_form": "O(n log(OPT/epsilon))"
    },
    "C3": {
      "status": "verified",
      "valg_constant": 0.052831216351296784
    },
    "C4": {
      "status": "verified",
      "complexity_improvement": "O(n^2 log n) to O(n log n)"
    },
    "C5": {
      "status": "verified",
      "economic_cells": 15
    }
  },
  "verified_claims": 5,
  "falsified_claims": 0
}
test_five_claims (test_certificate.TestCertificate.test_five_claims) ... {
  "paper": "hqFugMmCqJ",
  "source_sha256": "1042e7805858b2a82641a5f2e25aaf5a18f564d36b464545950fffa9d7cff746",
  "scope": "Source-pinned theorem contract plus finite descending-clock budget/IR/monotonicity controls; not a rerun of unreleased large SNAP experiments.",
  "claims": {
    "C1": {
      "status": "verified",
      "general_constant": 0.032897681803431035
    },
    "C2": {
      "status": "verified",
      "monotone_constant": 0.0877271514758161,
      "runtime_form": "O(n log(OPT/epsilon))"
    },
    "C3": {
      "status": "verified",
      "valg_constant": 0.052831216351296784
    },
    "C4": {
      "status": "verified",
      "complexity_improvement": "O(n^2 log n) to O(n log n)"
    },
    "C5": {
      "status": "verified",
      "economic_cells": 15
    }
  },
  "verified_claims": 5,
  "falsified_claims": 0
}
ok

----------------------------------------------------------------------
Ran 1 test in 0.051s

OK
{
  "paper": "hqFugMmCqJ",
  "gate": "passed",
  "tests_passed": true,
  "publication_gate_passed": true,
  "verified_claims": 5,
  "scope": "Source-pinned theorem contract plus finite descending-clock budget/IR/monotonicity controls; not a rerun of unreleased large SNAP experiments."
}

````
