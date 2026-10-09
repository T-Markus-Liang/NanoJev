import json
from pathlib import Path
import unittest

from validate_phase1_shadow_authorization_v1 import validate_authorization

ROOT = Path(__file__).resolve().parents[1]
AUTH = ROOT / "research" / "phase1_shadow_authorization_v1.json"


class Phase1ShadowAuthorizationTest(unittest.TestCase):
    def test_template_is_valid_but_not_authorized(self):
        receipt = validate_authorization(AUTH)
        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["status"], "template_ready_not_authorized")
        self.assertFalse(receipt["phase1_authorized"])
        self.assertFalse(receipt["provider_calls_allowed"])
        self.assertFalse(receipt["active_filtering_allowed"])

    def test_measurement_contract_is_shadow_only(self):
        auth = json.loads(AUTH.read_text(encoding="utf-8"))
        contract = auth["measurement_contract"]
        self.assertEqual(contract["mode"], "shadow")
        self.assertTrue(contract["measurement_only"])
        self.assertFalse(contract["reduced_bytes_may_be_sent"])
        self.assertFalse(contract["provider_calls_allowed_in_template"])
        self.assertFalse(contract["active_filtering_allowed"])

    def test_milestone_metrics_and_checklist(self):
        auth = json.loads(AUTH.read_text(encoding="utf-8"))
        metrics = auth["milestone_metrics"]
        self.assertEqual(metrics["phase0_cases_required"], 17)
        self.assertEqual(metrics["phase1_dry_run_cases_required"], 8)
        self.assertEqual(metrics["phase1_dry_run_status_required"],
                         "phase1_shadow_dry_run_pass")
        self.assertEqual(metrics["provider_calls_required"], 0)
        self.assertEqual(metrics["unsafe_count_required"], 0)
        checklist = {item["id"]: item for item in auth["operator_checklist"]}
        for item_id in ("scope_named", "measurement_only_confirmed",
                        "accounting_confirmed", "privacy_confirmed",
                        "stop_conditions_confirmed"):
            self.assertTrue(checklist[item_id]["required"])
            self.assertFalse(checklist[item_id]["completed"])


if __name__ == "__main__":
    unittest.main()
