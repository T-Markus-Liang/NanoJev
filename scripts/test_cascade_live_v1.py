import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "cascade_live_v1.json"


class CascadeLiveTest(unittest.TestCase):
    def test_live_cascade_receipt(self):
        receipt = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["status"], "cascade_live_pass")
        self.assertEqual(receipt["provider_calls"], 0)
        cascade = receipt["cascade"]
        self.assertEqual(sorted(cascade["fast_paths"]),
                         ["segment_0", "segment_1"])
        self.assertEqual(cascade["strong_paths"], [])
        self.assertGreater(cascade["scores"]["segment_0"], 0.9)
        self.assertLess(cascade["scores"]["segment_1"], 0.1)
        fallback = receipt["fallback"]
        self.assertEqual(sorted(fallback["strong_paths"]),
                         ["segment_0", "segment_1"])
        self.assertEqual(fallback["fast_paths"], [])
        self.assertEqual(fallback["fast_errors"], 2)
        self.assertGreater(fallback["scores"]["segment_0"], 0.9)
        self.assertLess(fallback["scores"]["segment_1"], 0.1)


if __name__ == "__main__":
    unittest.main()
