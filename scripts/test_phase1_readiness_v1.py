import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "phase1_readiness_gate_v1.json"


class Phase1ReadinessTest(unittest.TestCase):
    def test_readiness_gate_is_ready_but_not_authorized(self):
        receipt = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["status"], "ready_for_owner_phase1_authorization")
        self.assertFalse(receipt["phase1_authorized"])
        self.assertFalse(receipt["provider_calls_allowed"])
        self.assertFalse(receipt["active_filtering_allowed"])
        self.assertEqual(receipt["failures"], [])

    def test_milestone_metrics(self):
        receipt = json.loads(RESULT.read_text(encoding="utf-8"))
        metrics = receipt["metrics"]
        self.assertEqual(metrics["phase0_cases_passed"], 17)
        self.assertEqual(metrics["phase1_dry_run_cases_passed"], 8)
        self.assertEqual(metrics["provider_calls"], 0)
        self.assertEqual(metrics["unsafe_actions"], 0)
        self.assertTrue(metrics["kill_switch_verified"])
        self.assertTrue(metrics["restore_round_trip_verified"])
        self.assertTrue(metrics["unsupported_path_bypass_verified"])
        self.assertTrue(metrics["scorer_fail_open_verified"])
        self.assertFalse(metrics["active_filtering_authorized"])
        self.assertFalse(metrics["phase1_authorized"])


if __name__ == "__main__":
    unittest.main()
