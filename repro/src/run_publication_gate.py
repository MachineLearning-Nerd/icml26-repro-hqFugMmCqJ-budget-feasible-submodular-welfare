import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
subprocess.run([sys.executable,'repro/src/verify_budget_feasible.py','--output','outputs/verification.json'],cwd=ROOT,check=True)
subprocess.run([sys.executable,'-m','unittest','discover','-s','repro/tests','-v'],cwd=ROOT,check=True)
v=json.loads((ROOT/'outputs/verification.json').read_text()); assert v['verified_claims']==5
g={'paper':'hqFugMmCqJ','gate':'passed','tests_passed':True,'publication_gate_passed':True,'verified_claims':5,'scope':v['scope']}
(ROOT/'outputs/publication_gate.json').write_text(json.dumps(g,indent=2)+'\n');print(json.dumps(g,indent=2))
