import json
from pathlib import Path
import unittest

from validate_real_context_collection_authorization_v1 import validate_authorization

ROOT = Path(__file__).resolve().parents[1]
AUTH = ROOT / "research" / "real_context_collection_authorization_v1.json"


class RealContextCollectionAuthorizationTest(unittest.TestCase):
    def test_authorization_template_is_valid_but_not_authorized(self):
        receipt = validate_authorization(AUTH)
        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["status"], "template_ready_not_authorized")
        self.assertFalse(receipt["collection_authorized"])
        self.assertFalse(receipt["provider_calls_allowed"])
        self.assertFalse(receipt["active_filtering_allowed"])
        self.assertEqual(receipt["failures"], [])

    def test_template_requires_owner_scope_and_checklist(self):
        auth = json.loads(AUTH.read_text(encoding="utf-8"))
        self.assertFalse(auth["collection"]["authorized"])
        self.assertIsNone(auth["collection"]["scope"])
        checklist = {item["id"]: item for item in auth["operator_checklist"]}
        for item_id in ("scope_named", "sources_reviewed", "privacy_reviewed",
                        "labels_prepared", "manifest_built", "preflight_passed",
                        "rollback_ready"):
            self.assertTrue(checklist[item_id]["required"])
            self.assertFalse(checklist[item_id]["completed"])

    def test_forbidden_actions_and_stop_conditions_are_declared(self):
        auth = json.loads(AUTH.read_text(encoding="utf-8"))
        self.assertIn("provider calls", auth["forbidden_during_collection"])
        self.assertIn("active filtering", auth["forbidden_during_collection"])
        self.assertIn("using holdout data for training or calibration",
                      auth["forbidden_during_collection"])
        self.assertIn("credential-like pattern is detected", auth["stop_conditions"])
        self.assertIn("owner revokes scope or requests stop", auth["stop_conditions"])


if __name__ == "__main__":
    unittest.main()
