import json
from pathlib import Path
import tempfile
import unittest

from real_context_holdout_manifest_v1 import build_manifest
from run_real_context_holdout_shadow_v1 import evaluate_manifest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research" / "real_context_holdout_protocol_v1.json"


def write_request(root, name, messages):
    requests = root / "requests"
    requests.mkdir(parents=True, exist_ok=True)
    path = requests / f"{name}.json"
    path.write_text(json.dumps({"model": "local-main", "messages": messages},
                               ensure_ascii=False), encoding="utf-8")


def write_case(root, name, expected_status="scored", eligible=None, protected=None,
               downstream=None, request_messages=None, extra=None):
    messages = request_messages or [
        {"role": "system", "content": "instructions"},
        {"role": "assistant", "content": "unrelated archived note"},
        {"role": "user", "content": "answer"},
    ]
    write_request(root, name, messages)
    case = {
        "case_id": name,
        "family": "synthetic_intake",
        "language": "en",
        "wire_format": "openai_chat",
        "source_type": "owner_selected_local_requests",
        "source_hash": "0" * 64,
        "request_file": f"requests/{name}.json",
        "captured_or_redacted_at": "2026-09-23T00:00:00Z",
        "expected_gate_status": expected_status,
        "protected_pointers": protected or ["/messages/0/content"],
        "eligible_candidate_pointers": eligible or [],
        "downstream": downstream or {},
        "labels": {"author": "owner", "confidence": "high"},
    }
    case.update(extra or {})
    (root / f"{name}.json").write_text(json.dumps(case), encoding="utf-8")


def build(root):
    manifest = build_manifest(root, PROTOCOL, "data/real_context_holdout_v1")
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class RealContextHoldoutShadowTest(unittest.TestCase):
    def test_scored_case_suggests_only_eligible_pointer_and_restores(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_case(root, "case_a", eligible=["/messages/1/content"],
                       downstream={
                           "required_evidence_pointers": ["/messages/0/content"],
                           "expected_answer_sha256": "1" * 64,
                       },
                       extra={"expected_suggestions": ["/messages/1/content"]})
            manifest = build(root)
            receipt = evaluate_manifest(manifest, root, PROTOCOL)
            self.assertTrue(receipt["ok"], receipt["failures"])
            self.assertEqual(receipt["provider_calls"], 0)
            case = receipt["cases"][0]
            self.assertEqual(case["suggested_pointers"], ["/messages/1/content"])
            self.assertEqual(case["protected_drop_pointers"], [])
            self.assertTrue(case["restore"]["ok"])

    def test_bypass_and_protected_only_statuses_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_case(root, "case_b", expected_status="bypass",
                       downstream={})
            write_case(root, "case_c", expected_status="protected_only",
                       downstream={}, request_messages=[
                           {"role": "system", "content": "instructions"},
                           {"role": "user", "content": "protected-only request"},
                       ])
            manifest = build(root)
            receipt = evaluate_manifest(manifest, root, PROTOCOL)
            self.assertTrue(receipt["ok"], receipt["failures"])
            by_id = {case["case_id"]: case for case in receipt["cases"]}
            self.assertEqual(by_id["case_b"]["gate_reason"], "caller_bypass")
            self.assertEqual(by_id["case_c"]["gate_reason"], "no_eligible_segments")

    def test_required_evidence_drop_is_flagged_unsafe(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_case(root, "case_d", eligible=["/messages/1/content"],
                       downstream={
                           "required_evidence_pointers": ["/messages/1/content"],
                           "expected_answer_sha256": "1" * 64,
                       })
            manifest = build(root)
            receipt = evaluate_manifest(manifest, root, PROTOCOL)
            self.assertFalse(receipt["ok"])
            case = receipt["cases"][0]
            self.assertTrue(case["unsafe"])
            self.assertEqual(case["protected_drop_pointers"], ["/messages/1/content"])

    def test_manifest_validation_failure_still_returns_failed_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"schema_version": "bad", "cases": []}),
                                     encoding="utf-8")
            receipt = evaluate_manifest(manifest_path, root, PROTOCOL)
            self.assertFalse(receipt["ok"])
            self.assertFalse(receipt["manifest_validation_ok"])


if __name__ == "__main__":
    unittest.main()
