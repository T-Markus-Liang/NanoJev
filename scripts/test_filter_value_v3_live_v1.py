import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "filter_value_v3_live_cascade_v1.json"


class FilterValueV3LiveTest(unittest.TestCase):
    def test_live_cascade_vs_winnow_receipt(self):
        receipt = json.loads(RESULT.read_text(encoding="utf-8"))
        arms = {arm["arm"]: arm["totals"] for arm in receipt["results"]}
        self.assertEqual(set(arms), {"cascade", "systemone", "control"})
        for arm, totals in arms.items():
            self.assertEqual(totals["unsafe_removals"], 0, arm)
            self.assertEqual(totals["paired_answer_regressions"], 0, arm)
        # Cascade reaches the same removal decision as Winnow-only.
        self.assertEqual(arms["cascade"]["removed_bytes"],
                         arms["systemone"]["removed_bytes"])
        self.assertGreater(arms["cascade"]["removed_bytes"], 0)
        self.assertEqual(arms["control"]["removed_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
