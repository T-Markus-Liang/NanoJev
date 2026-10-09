import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "phase1_shadow_measurement_v1.json"
GRANT = ROOT / "research" / "phase1_shadow_scope_synthetic_v1.json"


class Phase1ShadowMeasurementTest(unittest.TestCase):
    def test_scope_grant_shape(self):
        grant = json.loads(GRANT.read_text(encoding="utf-8"))
        scope = grant["scope"]
        self.assertFalse(scope["provider_calls_allowed"])
        self.assertFalse(scope["active_filtering_allowed"])
        self.assertFalse(scope["reduced_bytes_may_be_sent"])
        self.assertFalse(scope["real_user_data_allowed"])
        self.assertEqual(scope["mode"], "shadow")
        self.assertLessEqual(scope["max_requests"], 25)

    def test_measurement_receipt(self):
        receipt = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["status"], "phase1_shadow_measurement_pass")
        self.assertEqual(receipt["provider_calls"], 0)
        self.assertEqual(receipt["requests_measured"], 12)
        self.assertLessEqual(receipt["requests_measured"],
                             receipt["grant"]["max_requests"])
        self.assertEqual(receipt["aggregates"]["unsafe_proposals"], 0)
        self.assertEqual(receipt["aggregates"]["changed_bytes"], 0)
        self.assertEqual(receipt["aggregates"]["restore_headers"], 0)
        for case in receipt["cases"]:
            self.assertTrue(case["forwarded_unchanged"])
            self.assertFalse(case["restore_header_present"])
            self.assertEqual(case["unsafe_proposals"], [])
        # Every declared retain-expectation was honoured by the live scorer.
        expected_retain = [c for c in receipt["cases"]
                           if c["expected_all_retain"]]
        self.assertTrue(expected_retain)
        self.assertTrue(all(not c["proposed_pointers"]
                            for c in expected_retain))


if __name__ == "__main__":
    unittest.main()
