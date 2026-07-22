import json, subprocess, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
class TestCertificate(unittest.TestCase):
 def test_five_claims(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'v.json'; subprocess.run([sys.executable,'repro/src/verify_budget_feasible.py','--output',str(out)],cwd=ROOT,check=True)
   self.assertEqual(json.loads(out.read_text())['verified_claims'],5)
