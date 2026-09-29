import json
import tempfile
import unittest
from pathlib import Path

import validate_nanojev_v2_t9d_fp32_mps_protocol_v2 as validator


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research/nanojev_v2_t9d_fp32_mps_protocol_v2.json"
REPIN_DIR = ROOT / "research"


def apply_repins(protocol):
    """Overlay accepted repin chain links on a frozen protocol's runtime pins.

    Frozen protocol documents are never edited. Legitimate file evolution is
    recorded in research/nanojev_*_repin_v*.json chain links; each link must name
    the old pinned hash it supersedes so unrelated drift is still caught.
    """
    protocol = json.loads(json.dumps(protocol))
    runtime = protocol.get("runtime", {})
    pin_keys = {"scripts/train_pipeline_decisions.py": "trainer_sha256",
                "scripts/predict_toy_decisions.py": "predictor_sha256"}
    for repin in sorted(REPIN_DIR.glob("nanojev_*_repin_v*.json")):
        link = validator.load(repin)
        target = link.get("repinned_file")
        if target not in pin_keys or link.get("status") != "active_repin_chain_link":
            continue
        key = pin_keys[target]
        if runtime.get(key) != link.get("old_sha256"):
            raise AssertionError(
                f"repin {repin.name} does not chain: protocol pin is not its old_sha256")
        if validator.sha256(ROOT / target) != link.get("new_sha256"):
            raise AssertionError(
                f"repin {repin.name} does not match current {target} hash")
        runtime[key] = link["new_sha256"]
    return protocol


class T9dProtocolValidatorTests(unittest.TestCase):
    def test_current_protocol_passes(self):
        errors = validator.validate(apply_repins(validator.load(PROTOCOL)), ROOT)
        self.assertEqual(errors, [])

    def test_unrepinned_drift_still_detected(self):
        protocol = apply_repins(validator.load(PROTOCOL))
        protocol["runtime"]["predictor_sha256"] = "0" * 64
        self.assertIn("hash mismatch: scripts/predict_toy_decisions.py",
                      validator.validate(protocol, ROOT))

    def test_training_authorization_is_fail_closed(self):
        value = validator.load(PROTOCOL)
        value["authorization"]["training_authorized"] = True
        self.assertIn("authorization.training_authorized must be false", validator.validate(value, ROOT))

    def test_recipe_is_not_silently_defaulted(self):
        value = validator.load(PROTOCOL)
        value["recipe"]["batch_questions"] = 12
        self.assertIn("recipe.batch_questions mismatch", validator.validate(value, ROOT))

    def test_hash_binding_detects_drift(self):
        value = validator.load(PROTOCOL)
        value["runtime"]["trainer_sha256"] = "0" * 64
        self.assertIn("hash mismatch: scripts/train_pipeline_decisions.py", validator.validate(value, ROOT))


if __name__ == "__main__":
    unittest.main()
