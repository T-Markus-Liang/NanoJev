import json
from pathlib import Path
import unittest

from validate_active_mode_canary_protocol_v1 import validate_protocol

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research" / "active_mode_canary_protocol_v1.json"


class ActiveModeCanaryProtocolTest(unittest.TestCase):
    def test_protocol_is_valid_but_not_authorized(self):
        receipt = validate_protocol(PROTOCOL)
        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["status"], "protocol_valid_not_authorized")
        self.assertFalse(receipt["active_filtering_allowed"])
        self.assertFalse(receipt["provider_calls_allowed"])
        self.assertEqual(receipt["failures"], [])

    def test_protocol_requires_owner_for_provider_phases(self):
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        phases = {phase["id"]: phase for phase in protocol["phases"]}
        self.assertFalse(phases["phase0_loopback_fake_upstream"]["requires_owner_authorization"])
        self.assertTrue(phases["phase1_shadow_provider_measure"]["requires_owner_authorization"])
        self.assertTrue(phases["phase2_paired_canary"]["requires_owner_authorization"])
        self.assertTrue(all(not phase["provider_calls_allowed"] for phase in phases.values()))

    def test_protocol_keeps_threshold_diagnostic_and_accounting_provider_paired(self):
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(protocol["threshold_policy"]["diagnostic_threshold"], 0.90)
        self.assertIsNone(protocol["threshold_policy"]["production_threshold"])
        accounting = protocol["provider_accounting"]
        self.assertTrue(accounting["actual_savings_requires_paired_provider_usage"])
        self.assertTrue(accounting["no_local_estimate_is_billing"])


if __name__ == "__main__":
    unittest.main()
