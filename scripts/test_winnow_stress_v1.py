import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "winnow_stress_v1.json"


class WinnowStressTest(unittest.TestCase):
    def test_stress_receipt(self):
        receipt = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["status"], "winnow_stress_pass")
        self.assertEqual(receipt["provider_calls"], 0)
        self.assertEqual(receipt["total_calls"], 20)
        self.assertTrue(receipt["deterministic"])
        self.assertTrue(receipt["consistent_under_concurrency"])
        self.assertTrue(receipt["isolated"])
        means = receipt["state_means"]
        self.assertGreater(means["0"], 0.5)
        self.assertGreater(means["3"], 0.5)
        self.assertLess(means["1"], 0.5)
        self.assertLess(means["2"], 0.5)


if __name__ == "__main__":
    unittest.main()
