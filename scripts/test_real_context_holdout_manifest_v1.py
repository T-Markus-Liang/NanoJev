import json
from pathlib import Path
import tempfile
import unittest

from real_context_holdout_manifest_v1 import (
    ManifestError, build_manifest, validate_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research" / "real_context_holdout_protocol_v1.json"


def write_case(root, name="case_a", overrides=None, request=None):
    requests = root / "requests"
    requests.mkdir(parents=True, exist_ok=True)
    request_path = requests / f"{name}.json"
    body = request or {"model": "local-main", "messages": [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "answer from evidence"},
    ]}
    request_path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    case = {
        "case_id": name,
        "family": "tool_result",
        "language": "en",
        "wire_format": "openai_chat",
        "source_type": "owner_selected_local_requests",
        "source_hash": "0" * 64,
        "request_file": f"requests/{name}.json",
        "captured_or_redacted_at": "2026-09-23T00:00:00Z",
        "expected_gate_status": "scored",
        "protected_pointers": ["/messages/0/content"],
        "eligible_candidate_pointers": ["/messages/1/content"],
        "downstream": {
            "required_evidence_pointers": ["/messages/0/content"],
            "expected_answer_sha256": "1" * 64,
        },
        "labels": {"author": "owner", "confidence": "high"},
    }
    case.update(overrides or {})
    (root / f"{name}.json").write_text(json.dumps(case, ensure_ascii=False),
                                       encoding="utf-8")


class RealContextHoldoutManifestTest(unittest.TestCase):
    def test_build_manifest_is_content_free_and_validate_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_case(root)
            manifest = build_manifest(root, PROTOCOL, "data/real_context_holdout_v1")
            self.assertTrue(manifest["content_free"])
            self.assertEqual(manifest["case_count"], 1)
            self.assertFalse(manifest["coverage"]["ready_for_evaluation"])
            case = manifest["cases"][0]
            self.assertEqual(case["request_file"], "requests/case_a.json")
            self.assertNotIn("messages", case)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            receipt = validate_manifest(manifest_path, PROTOCOL)
            self.assertTrue(receipt["ok"], receipt["failures"])
            self.assertFalse(receipt["provider_calls_allowed"])
            self.assertFalse(receipt["active_filtering_allowed"])

    def test_duplicate_request_hash_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_case(root, "case_a")
            write_case(root, "case_b")
            with self.assertRaisesRegex(ManifestError, "duplicate request_sha256"):
                build_manifest(root, PROTOCOL)

    def test_raw_metadata_key_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_case(root, overrides={"messages": [{"role": "user", "content": "raw"}]})
            with self.assertRaisesRegex(ManifestError, "forbidden"):
                build_manifest(root, PROTOCOL)

    def test_credential_like_request_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_case(root, request={"api_key": "sk-abcdefghijklmnop"})
            with self.assertRaisesRegex(ManifestError, "credential-like"):
                build_manifest(root, PROTOCOL)

    def test_protected_eligible_overlap_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_case(root, overrides={
                "protected_pointers": ["/messages/1/content"],
                "eligible_candidate_pointers": ["/messages/1/content"],
            })
            with self.assertRaisesRegex(ManifestError, "both protected and eligible"):
                build_manifest(root, PROTOCOL)


if __name__ == "__main__":
    unittest.main()
