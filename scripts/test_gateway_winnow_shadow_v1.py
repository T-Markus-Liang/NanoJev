import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "gateway_winnow_shadow_v1.json"


class GatewayWinnowShadowTest(unittest.TestCase):
    def test_live_winnow_shadow_receipt(self):
        receipt = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["status"], "winnow_shadow_pass")
        self.assertEqual(receipt["provider_calls"], 0)
        self.assertEqual(receipt["scorer"]["model"], "Winnow-12B")
        self.assertEqual(receipt["scorer_calls"], 4)
        self.assertEqual({case["case_id"] for case in receipt["cases"]}, {
            "winnow_shadow_openai_chat",
            "winnow_shadow_anthropic_messages",
            "winnow_shadow_openai_responses",
            "winnow_shadow_no_sidecar",
            "winnow_shadow_unsupported",
            "winnow_shadow_kill_switch",
            "winnow_shadow_internal_headers",
        })
        self.assertTrue(all(case["forwarded_unchanged"] is True
                            for case in receipt["cases"]))
        self.assertTrue(all(not case["restore_header_present"]
                            for case in receipt["cases"]))
        # The live scorer proposed the declared eligible pointer on every
        # supported wire format.
        self.assertEqual(receipt["proposed_pointers"]["winnow_shadow_openai_chat"],
                         ["/messages/2/content"])
        self.assertEqual(
            receipt["proposed_pointers"]["winnow_shadow_anthropic_messages"],
            ["/messages/1/content/0/text"])
        self.assertEqual(
            receipt["proposed_pointers"]["winnow_shadow_openai_responses"],
            ["/input/1/content"])

    def test_live_cascade_shadow_receipt(self):
        receipt = json.loads(
            (ROOT / "results" / "gateway_cascade_shadow_v1.json").read_text(
                encoding="utf-8"))
        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["provider_calls"], 0)
        self.assertEqual(receipt["scorer"]["kind"], "cascade")
        self.assertEqual(receipt["scorer"]["model"], "kev-latest")
        self.assertEqual(receipt["scorer"]["strong_model"], "Winnow-12B")
        self.assertEqual(receipt["scorer_calls"], 4)
        self.assertTrue(all(case["forwarded_unchanged"] is True
                            for case in receipt["cases"]))
        self.assertEqual(receipt["proposed_pointers"]["winnow_shadow_openai_chat"],
                         ["/messages/2/content"])


if __name__ == "__main__":
    unittest.main()
