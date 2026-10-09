import json
from pathlib import Path
import tempfile
import unittest

from local_main_model_evaluator_v1 import sha256_text
from real_context_holdout_manifest_v1 import build_manifest
from run_real_context_holdout_preflight_v1 import run_preflight

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research" / "real_context_holdout_protocol_v1.json"


def write_case(root):
    requests = root / "requests"
    contracts = root / "contracts"
    requests.mkdir()
    contracts.mkdir()
    body = {"model": "local-main", "messages": [
        {"role": "system", "content": "instructions"},
        {"role": "assistant", "content": "queue depth is 42"},
        {"role": "assistant", "content": "unrelated picnic note"},
        {"role": "user", "content": "What is the queue depth?"},
    ]}
    (requests / "case_a.json").write_text(json.dumps(body), encoding="utf-8")
    (contracts / "case_a.contract.json").write_text(json.dumps({
        "expected_answer": "42",
        "required_strings": ["queue depth is 42"],
    }), encoding="utf-8")
    case = {
        "case_id": "case_a",
        "family": "preflight",
        "language": "en",
        "wire_format": "openai_chat",
        "source_type": "owner_selected_local_requests",
        "source_hash": "0" * 64,
        "request_file": "requests/case_a.json",
        "captured_or_redacted_at": "2026-09-23T00:00:00Z",
        "expected_gate_status": "scored",
        "protected_pointers": ["/messages/0/content", "/messages/1/content"],
        "eligible_candidate_pointers": ["/messages/2/content"],
        "downstream": {
            "required_evidence_pointers": ["/messages/1/content"],
            "required_evidence_sha256": [sha256_text("queue depth is 42")],
            "expected_answer_sha256": sha256_text("42"),
        },
        "labels": {"author": "owner", "confidence": "high"},
    }
    (root / "case_a.json").write_text(json.dumps(case), encoding="utf-8")


def manifest_path(root):
    manifest = build_manifest(root, PROTOCOL, "data/real_context_holdout_v1")
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class RealContextHoldoutPreflightTest(unittest.TestCase):
    def test_protocol_only_preflight(self):
        receipt = run_preflight(PROTOCOL)
        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["status"], "protocol_ready_no_manifest")
        self.assertEqual(receipt["provider_calls"], 0)
        self.assertFalse(receipt["active_filtering_applied"])

    def test_full_chain_passes_on_valid_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_case(root)
            receipt = run_preflight(PROTOCOL, manifest_path(root), root,
                                    root / "contracts")
            self.assertTrue(receipt["ok"], receipt["failures"])
            self.assertEqual(receipt["status"], "e2e_preflight_pass")
            self.assertTrue(receipt["stages"]["protocol"]["ok"])
            self.assertTrue(receipt["stages"]["shadow"]["ok"])
            self.assertTrue(receipt["stages"]["local_generation"]["ok"])

    def test_require_ready_fails_on_partial_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_case(root)
            receipt = run_preflight(PROTOCOL, manifest_path(root), root,
                                    root / "contracts", require_ready=True)
            self.assertFalse(receipt["ok"])
            self.assertIn("manifest_not_ready_for_evaluation", receipt["failures"])

    def test_bad_case_fails_chain(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_case(root)
            case_path = root / "case_a.json"
            case = json.loads(case_path.read_text())
            case["protected_pointers"] = ["/messages/0/content"]
            case["eligible_candidate_pointers"] = ["/messages/1/content"]
            case_path.write_text(json.dumps(case), encoding="utf-8")
            receipt = run_preflight(PROTOCOL, manifest_path(root), root,
                                    root / "contracts")
            self.assertFalse(receipt["ok"])
            self.assertIn("shadow", receipt["failures"])
            self.assertIn("local_generation", receipt["failures"])


if __name__ == "__main__":
    unittest.main()
