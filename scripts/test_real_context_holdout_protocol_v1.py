import json
from pathlib import Path
import unittest

from validate_real_context_holdout_protocol_v1 import validate_protocol

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research" / "real_context_holdout_protocol_v1.json"


class RealContextHoldoutProtocolTest(unittest.TestCase):
    def test_protocol_is_valid_but_not_collecting(self):
        receipt = validate_protocol(PROTOCOL)
        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["status"], "protocol_valid_not_collecting")
        self.assertFalse(receipt["collection_started"])
        self.assertFalse(receipt["provider_calls_allowed"])
        self.assertFalse(receipt["active_filtering_allowed"])
        self.assertEqual(receipt["failures"], [])

    def test_protocol_contract_keeps_raw_local_and_labels_manual(self):
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertFalse(protocol["collection"]["started"])
        self.assertTrue(protocol["collection"]["requires_owner_authorization"])
        self.assertTrue(protocol["privacy"]["raw_files_local_only"])
        self.assertTrue(protocol["labeling"]["labels_are_manual"])
        self.assertTrue(protocol["labeling"]["scorer_outputs_must_not_be_labels"])
        self.assertTrue(protocol["isolation"]["exclude_v2_v3_cases"])
        self.assertTrue(protocol["isolation"]["training_calibration_forbidden"])

    def test_evaluation_order_is_shadow_before_provider(self):
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(protocol["evaluation_order"][0],
                         "manifest_schema_and_hash_validation")
        self.assertIn("deterministic_shadow_evaluation", protocol["evaluation_order"])
        self.assertIn("pinned_local_generation_pair_evaluation", protocol["evaluation_order"])
        self.assertFalse(protocol["provider_calls_allowed"])
        self.assertFalse(protocol["active_filtering_allowed"])


if __name__ == "__main__":
    unittest.main()
