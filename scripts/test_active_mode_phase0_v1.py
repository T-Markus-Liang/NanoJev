import unittest

from run_active_mode_phase0_v1 import run_phase0


class ActiveModePhase0Test(unittest.TestCase):
    def test_phase0_loopback_receipt_passes(self):
        receipt = run_phase0()
        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["status"], "phase0_pass")
        self.assertEqual(receipt["provider_calls"], 0)
        self.assertEqual({case["case_id"] for case in receipt["cases"]}, {
            "active_eligible_drop",
            "provider_accounting_baseline",
            "anthropic_eligible_drop",
            "responses_eligible_drop",
            "unsupported_embeddings",
            "active_no_reduction",
            "shadow_control",
            "kill_switch",
            "scorer_error",
            "malformed_sidecar",
            "dependency_closure",
            "uncertain_score",
            "invalid_score_response",
            "scorer_timeout",
            "restore_manifest_too_large",
            "oversized_request_body",
            "internal_header_stripping",
        })
        eligible = next(case for case in receipt["cases"]
                        if case["case_id"] == "active_eligible_drop")
        self.assertEqual(eligible["applied_pointers"], ["/messages/2/content"])
        self.assertTrue(eligible["restore_header_present"])
        self.assertFalse(eligible["restore_header_contains_raw_text"])


if __name__ == "__main__":
    unittest.main()
