import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "gateway_live_active_v1.json"


class GatewayLiveActiveTest(unittest.TestCase):
    def test_live_active_receipt(self):
        receipt = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["status"], "live_active_pass")
        self.assertEqual(receipt["provider_calls"], 0)
        self.assertEqual(receipt["scorer"]["kind"], "cascade")
        self.assertEqual(receipt["scorer"]["fast_model"], "kev-latest")
        self.assertEqual(receipt["scorer"]["strong_model"], "Winnow-12B")
        for wire in ("openai_chat", "anthropic_messages", "openai_responses"):
            self.assertEqual(receipt["outcomes"][wire], "dropped")
            case = next(c for c in receipt["cases"]
                        if c["case_id"] == f"live_active_{wire}")
            self.assertFalse(case["forwarded_unchanged"])
            self.assertTrue(case["restore_header_present"])
            self.assertTrue(case["restore_round_trip"]
                            ["original_request_sha256_matches"])
            self.assertFalse(case["restore_header_contains_raw_text"])
            self.assertEqual(case["forward_reason"], "active_reduced")
        control = next(c for c in receipt["cases"]
                       if c["case_id"] == "live_active_kill_switch")
        self.assertTrue(control["forwarded_unchanged"])


if __name__ == "__main__":
    unittest.main()
