import json
from pathlib import Path
import tempfile
import unittest

from local_main_model_evaluator_v1 import sha256_text
from real_context_holdout_manifest_v1 import build_manifest
from run_real_context_holdout_localgen_v1 import evaluate_manifest_pairs

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research" / "real_context_holdout_protocol_v1.json"


def write_contract(root, name, expected="42", required=None):
    contracts = root / "contracts"
    contracts.mkdir(exist_ok=True)
    contract = {"expected_answer": expected,
                "required_strings": required or ["queue depth is 42"]}
    (contracts / f"{name}.contract.json").write_text(json.dumps(contract),
                                                    encoding="utf-8")
    return contract


def write_case(root, name, eligible=None, protected=None, downstream=None,
               request=None, contract=None, extra=None):
    requests = root / "requests"
    requests.mkdir(exist_ok=True)
    body = request or {"model": "local-main", "messages": [
        {"role": "system", "content": "instructions"},
        {"role": "assistant", "content": "queue depth is 42"},
        {"role": "assistant", "content": "unrelated picnic note"},
        {"role": "user", "content": "What is the queue depth?"},
    ]}
    (requests / f"{name}.json").write_text(json.dumps(body), encoding="utf-8")
    contract = contract or {"expected_answer": "42",
                            "required_strings": ["queue depth is 42"]}
    downstream = downstream or {
        "required_evidence_pointers": ["/messages/1/content"],
        "required_evidence_sha256": [sha256_text("queue depth is 42")],
        "expected_answer_sha256": sha256_text("42"),
    }
    case = {
        "case_id": name,
        "family": "localgen_pair",
        "language": "en",
        "wire_format": "openai_chat",
        "source_type": "owner_selected_local_requests",
        "source_hash": "0" * 64,
        "request_file": f"requests/{name}.json",
        "captured_or_redacted_at": "2026-09-23T00:00:00Z",
        "expected_gate_status": "scored",
        "protected_pointers": protected or ["/messages/0/content",
                                            "/messages/1/content"],
        "eligible_candidate_pointers": eligible or ["/messages/2/content"],
        "downstream": downstream,
        "labels": {"author": "owner", "confidence": "high"},
    }
    case.update(extra or {})
    (root / f"{name}.json").write_text(json.dumps(case), encoding="utf-8")


def build(root):
    manifest = build_manifest(root, PROTOCOL, "data/real_context_holdout_v1")
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class RealContextHoldoutLocalGenTest(unittest.TestCase):
    def test_deterministic_pair_passes_when_evidence_survives(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_contract(root, "case_a")
            write_case(root, "case_a")
            manifest = build(root)
            receipt = evaluate_manifest_pairs(manifest, root, root / "contracts", PROTOCOL)
            self.assertTrue(receipt["ok"], receipt["failures"])
            self.assertEqual(receipt["paired_count"], 1)
            case = receipt["cases"][0]
            self.assertEqual(case["original_status"], "answered")
            self.assertEqual(case["reduced_status"], "answered")
            self.assertFalse(case["answer_regression"])

    def test_evidence_drop_is_paired_regression(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_contract(root, "case_b")
            write_case(root, "case_b", protected=["/messages/0/content"],
                       eligible=["/messages/1/content"])
            manifest = build(root)
            receipt = evaluate_manifest_pairs(manifest, root, root / "contracts", PROTOCOL)
            self.assertFalse(receipt["ok"])
            case = receipt["cases"][0]
            self.assertTrue(case["answer_regression"])
            self.assertTrue(case["unsafe"])

    def test_contract_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_contract(root, "case_c", expected="not-42")
            write_case(root, "case_c")
            manifest = build(root)
            receipt = evaluate_manifest_pairs(manifest, root, root / "contracts", PROTOCOL)
            self.assertFalse(receipt["ok"])
            self.assertTrue(any("expected_answer_sha256 mismatch" in failure
                                for failure in receipt["failures"]))

    def test_non_scored_case_is_skipped(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_case(root, "case_d", downstream={},
                       extra={"expected_gate_status": "bypass"})
            manifest = build(root)
            receipt = evaluate_manifest_pairs(manifest, root, root / "contracts", PROTOCOL)
            self.assertTrue(receipt["ok"], receipt["failures"])
            self.assertEqual(receipt["paired_count"], 0)
            self.assertEqual(receipt["skipped_count"], 1)


if __name__ == "__main__":
    unittest.main()
