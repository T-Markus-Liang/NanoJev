import json
import tempfile
import unittest
from pathlib import Path

import bind_financial_r1_pit_receipt_v1 as binder


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research/financial_experiment_protocol_v2.json"
CORE = ROOT / "research/financial_r1_pit_validator_core_v2.json"
REPORT = ROOT / "results/financial_pit_r1_rebase_preflight_20260920_v2.json"


class FinancialR1PitBindingTests(unittest.TestCase):
    def test_current_receipt_binds(self):
        result = binder.bind(PROTOCOL, CORE, REPORT)
        # The available receipt audits the nine-feature pilot, not the 11-feature
        # R1 cohort.  Structural PIT success must not be misreported as R1 readiness.
        self.assertIn("validator input feature schema does not match R1 closed allowlist", result["errors"])
        self.assertIn("validator input label identity does not match R1 event definition", result["errors"])
        self.assertEqual(result["status"], "binding_failed")
        self.assertFalse(result["training_authorized"])
        self.assertEqual(len(result["folds"]), 3)

    def test_core_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "core.json"
            value = json.loads(CORE.read_text(encoding="utf-8"))
            value["asof_ns"] += 1
            path.write_text(json.dumps(value), encoding="utf-8")
            result = binder.bind(PROTOCOL, path, REPORT)
            self.assertIn("projected PIT core differs from full protocol.pit_validator_core", result["errors"])

    def test_report_core_binding_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            value = json.loads(REPORT.read_text(encoding="utf-8"))
            value["protocol"]["sha256"] = "0" * 64
            path.write_text(json.dumps(value), encoding="utf-8")
            result = binder.bind(PROTOCOL, CORE, path)
            self.assertIn("validator report is not bound to the supplied core", result["errors"])


if __name__ == "__main__":
    unittest.main()
