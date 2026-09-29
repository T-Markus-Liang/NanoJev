import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "phase1_shadow_dry_run_v1.json"


class Phase1ShadowDryRunTest(unittest.TestCase):
    def test_phase1_dry_run_receipt_passes(self):
        receipt = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["status"], "phase1_shadow_dry_run_pass")
        self.assertEqual(receipt["provider_calls"], 0)
        self.assertFalse(receipt["active_filtering_applied"])
        self.assertEqual(receipt["failures"], [])
        self.assertEqual({case["case_id"] for case in receipt["cases"]}, {
            "shadow_openai_eligible_plan",
            "shadow_anthropic_eligible_plan",
            "shadow_responses_eligible_plan",
            "shadow_no_sidecar",
            "shadow_internal_headers",
            "shadow_unsupported_embeddings",
            "shadow_kill_switch",
            "shadow_scorer_error",
        })
        self.assertTrue(all(case["forwarded_unchanged"] is True
                            for case in receipt["cases"]))
        self.assertTrue(all(not case["restore_header_present"]
                            for case in receipt["cases"]))


if __name__ == "__main__":
    unittest.main()
