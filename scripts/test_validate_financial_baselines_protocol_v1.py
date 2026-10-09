import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import validate_financial_baselines_protocol_v1 as validator

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research/financial_baselines_protocol_v1.json"
RECEIPT = ROOT / "results/financial_pit_r1_bound_receipt_20260920_v5.json"


class T11ProtocolTests(unittest.TestCase):
    def _write_modified_protocol(self, tmp, mutate):
        value = json.loads(PROTOCOL.read_text())
        mutate(value)
        p = Path(tmp) / "protocol.json"
        p.write_text(json.dumps(value))
        return p
    def test_current_protocol_passes(self):
        result = validator.validate(PROTOCOL, RECEIPT)
        self.assertEqual(result["status"], "protocol_valid_not_fit_authorized")
        self.assertEqual([x["primary_fit_counts"] for x in result["folds"]], [
            {"train": 746, "dev": 24, "calibration": 24},
            {"train": 928, "dev": 24, "calibration": 24},
            {"train": 1108, "dev": 24, "calibration": 24},
        ])
        self.assertFalse(result["authorization"]["training_authorized"])

    def test_protocol_hash_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write_modified_protocol(tmp, lambda x: x["instrument_holdout"]["groups"]["primary"].__setitem__(0, "tampered"))
            with self.assertRaisesRegex(ValueError, "protocol hash"):
                validator.validate(p, RECEIPT)

    def test_upstream_receipt_tamper_rejected(self):
        value = json.loads(PROTOCOL.read_text())
        value["upstream_r1"]["bound_receipt_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "protocol.json"
            p.write_text(json.dumps(value))
            with mock.patch.object(validator, "EXPECTED_PROTOCOL_SHA256", hashlib.sha256(p.read_bytes()).hexdigest()):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    validator.validate(p, RECEIPT)

    def test_authorization_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write_modified_protocol(tmp, lambda x: x["authorization"].__setitem__("training_authorized", True))
            with mock.patch.object(validator, "EXPECTED_PROTOCOL_SHA256", hashlib.sha256(p.read_bytes()).hexdigest()):
                with self.assertRaisesRegex(ValueError, "authorization boundary"):
                    validator.validate(p, RECEIPT)

    def test_budget_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write_modified_protocol(tmp, lambda x: x["candidates"]["linear"].__setitem__("steps", 201))
            with mock.patch.object(validator, "EXPECTED_PROTOCOL_SHA256", hashlib.sha256(p.read_bytes()).hexdigest()):
                with self.assertRaisesRegex(ValueError, "budget"):
                    validator.validate(p, RECEIPT)


if __name__ == "__main__":
    unittest.main()
