import json
from pathlib import Path
import tempfile
import unittest

from replay_track_a_review_v1 import check_result_invariants, file_receipt

ROOT = Path(__file__).resolve().parents[1]


class TrackAReviewReplayTest(unittest.TestCase):
    def test_existing_v2_result_invariants_pass(self):
        check = check_result_invariants(ROOT / "results/filter_value_v2_selftest_v1.json",
                                        expected_unsafe={"stub": 1})
        self.assertEqual(check["failures"], [])

    def test_result_invariants_detect_unsafe_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"results": [{"arm": "bad", "totals": {
                "unsafe_removals": 1,
                "paired_answer_regressions": 0,
                "round_trip_ok": 0,
                "round_trip_attempted": 0,
            }}]}))
            check = check_result_invariants(path)
            self.assertEqual(check["failures"], ["bad: unsafe_removals=1, expected=0"])

    def test_file_receipt_hashes_existing_policy(self):
        receipt = file_receipt(ROOT / "research/context_filter_threshold_policy_v1.json")
        self.assertTrue(receipt["path"].endswith("context_filter_threshold_policy_v1.json"))
        self.assertEqual(len(receipt["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
