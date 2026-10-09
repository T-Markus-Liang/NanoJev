import unittest

from scorer_stress_receipt_v1 import build_receipt


class ScorerStressReceiptTest(unittest.TestCase):
    def test_receipt_scenarios_cover_fallback_and_propagation(self):
        receipt = build_receipt()
        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["failures"], [])
        scenarios = {item["scenario"]: item for item in receipt["scenarios"]}
        self.assertEqual(
            scenarios["partial_fast_error_falls_back"]["observed"]["fast_paths"],
            ["segment_0"])
        self.assertEqual(
            scenarios["partial_fast_error_falls_back"]["observed"]["strong_paths"],
            ["segment_1", "segment_2"])
        self.assertEqual(
            scenarios["fast_timeout_falls_back"]["observed"]["fast_errors"], 3)
        self.assertTrue(scenarios["strong_error_propagates_for_gate_fail_open"]["propagated"])
        self.assertEqual(
            scenarios["strong_malformed_propagates_for_gate_fail_open"]["propagated_type"],
            "ScorerError")


if __name__ == "__main__":
    unittest.main()
