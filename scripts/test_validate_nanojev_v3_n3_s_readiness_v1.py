#!/usr/bin/env python3
"""Bounded regression tests for the read-only N3-S readiness validator.

The tests are standard-library only. They run the real N2 domain-pack preflight
producer against a synthetic in-memory pack, stage the frozen N3 receipts, and
then exercise the readiness validator with contract, boundary, path and read
faults. No model, tokenizer, real corpus or network is touched and no training,
merge or measurement is performed. A passing readiness report is asserted to
remain a prerequisite check only: it never authorizes measurement, training,
deployment or promotion and never claims model evidence or human approval.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import validate_nanojev_v3_domain_pack_v1 as domain_pack
import validate_nanojev_v3_n3_s_readiness_v1 as validator
import test_validate_nanojev_v3_domain_pack_v1 as domain_pack_fixtures


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = ROOT / "research" / "nanojev_v3_n3_s_readiness_v1.json"
N2_EVIDENCE_ID = "n2_domain_pack_preflight"
N3_EVIDENCE_IDS = ("n3_protocol_preflight", "n3_synthetic_controls")

SAFE_AUTHORIZATION_KEYS = (
    "authorizes_execution",
    "measurement_authorized",
    "real_n3_measurement_authorized",
    "training_authorized",
    "deployment_authorized",
    "production_pruning_authorized",
    "promotion_authorized",
)


def protocol_copy():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def evidence_entry(protocol, evidence_id):
    for entry in protocol["required_evidence"]:
        if entry["evidence_id"] == evidence_id:
            return entry
    raise KeyError(evidence_id)


def approve_reviews(protocol):
    protocol["review_requirements"] = [
        dict(item, status="approved") for item in protocol["review_requirements"]
    ]
    return protocol


def repin(root, entry):
    entry["expected_sha256"] = hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest()


def stage_genuine_receipt(root, protocol, evidence_id):
    """Copy one of the frozen, genuinely produced N3 receipts into the fixture root."""
    entry = evidence_entry(protocol, evidence_id)
    source = ROOT / entry["path"]
    target = root / entry["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return entry, target


def ready_fixture(method):
    """A ready protocol whose N2 receipt is produced by the real N2 validator.

    Temporary directories are registered with ``addCleanup`` so every test cleans
    up after itself even when an assertion fails.
    """
    tmp = tempfile.TemporaryDirectory()
    method.addCleanup(tmp.cleanup)
    root = Path(tmp.name)
    protocol = approve_reviews(protocol_copy())

    # Genuine staged N3 receipts already pinned by the frozen contract.
    for evidence_id in N3_EVIDENCE_IDS:
        stage_genuine_receipt(root, protocol, evidence_id)

    # Real producer integration: run the N2 preflight on a synthetic pack and
    # persist its actual report. Never fabricate N2 result fields.
    pack_root = domain_pack_fixtures.write_pack(root / "n2_pack")
    n2_report = domain_pack.validate_pack(pack_root)
    n2_entry = evidence_entry(protocol, N2_EVIDENCE_ID)
    write_json(root / n2_entry["path"], n2_report)
    repin(root, n2_entry)
    return root, protocol


def fresh_root(method):
    tmp = tempfile.TemporaryDirectory()
    method.addCleanup(tmp.cleanup)
    return Path(tmp.name)


class N3SReadinessTests(unittest.TestCase):
    # ------------------------------------------------------------------ ready
    def test_current_contract_blocks_missing_n2_and_pending_reviews(self):
        report = validator.validate_file(PROTOCOL_PATH)
        self.assertFalse(report["contract_valid"])
        self.assertFalse(report["ready"])
        self.assertEqual(report["status"], "n3_s_readiness_blocked")
        self.assertIn("missing_evidence", report["block_reasons"])
        self.assertIn("review_pending", report["block_reasons"])
        self.assertIs(report["measurement_authorized"], False)
        self.assertIs(report["real_n3_measurement_authorized"], False)
        self.assertEqual(report["network_model_calls"], 0)
        self.assertIs(report["model_loaded"], False)

    def test_ready_fixture_uses_real_n2_producer_output(self):
        root, protocol = ready_fixture(self)
        entry = evidence_entry(protocol, N2_EVIDENCE_ID)
        evidence = json.loads((root / entry["path"]).read_text(encoding="utf-8"))
        self.assertEqual(evidence["schema_version"], domain_pack.REPORT_SCHEMA)
        self.assertEqual(evidence["status"], "preflight_passed_not_training_authorized")
        expected_files = {"manifest.json"} | {
            f"splits/{split}.jsonl" for split in domain_pack.REQUIRED_SPLITS
        }
        self.assertEqual(set(evidence["source_hashes"]), expected_files)
        self.assertEqual(evidence["source_hashes"], evidence["source_hashes_after"])
        self.assertFalse(evidence["input_files_changed"])

        report = validator.validate_protocol(protocol, root=root)
        self.assertTrue(report["contract_valid"])
        self.assertTrue(report["ready"])
        self.assertEqual(report["status"], "n3_s_readiness_passed_not_authorized")
        self.assertEqual(report["block_reasons"], [])
        self.assertTrue(all(item["present"] for item in report["evidence"]))
        self.assertTrue(all(item["approved"] for item in report["reviews"]))

    def test_receipt_never_claims_model_evidence_or_human_approval(self):
        root, protocol = ready_fixture(self)
        report = validator.validate_protocol(protocol, root=root)
        for key in SAFE_AUTHORIZATION_KEYS:
            self.assertIs(report[key], False, key)
        self.assertIs(report["model_loaded"], False)
        self.assertEqual(report["network_model_calls"], 0)
        self.assertIs(report["measurement_performed"], False)
        self.assertIs(report["training_performed"], False)
        self.assertIs(report["deployment_performed"], False)
        self.assertIs(report["review_declarations_only"], True)

    # --------------------------------------------------------------- evidence
    def test_missing_evidence_blocks(self):
        root, protocol = ready_fixture(self)
        (root / protocol["required_evidence"][0]["path"]).unlink()
        report = validator.validate_protocol(protocol, root=root)
        self.assertFalse(report["ready"])
        self.assertIn("missing_evidence", report["block_reasons"])

    def test_pinned_hash_mismatch_blocks(self):
        root, protocol = ready_fixture(self)
        protocol["required_evidence"][1]["expected_sha256"] = "0" * 64
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("evidence_hash_mismatch", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_status_mismatch_blocks(self):
        root, protocol = ready_fixture(self)
        entry = protocol["required_evidence"][1]
        path = root / entry["path"]
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["status"] = "wrong_status"
        write_json(path, evidence)
        repin(root, entry)
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("evidence_status", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_required_evidence_field_mismatch_blocks(self):
        root, protocol = ready_fixture(self)
        entry = protocol["required_evidence"][2]
        path = root / entry["path"]
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["network_model_calls"] = 1
        write_json(path, evidence)
        repin(root, entry)
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("evidence_field", report["block_reasons"])
        self.assertIn("evidence_authorization_flag", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_malformed_evidence_is_rejected(self):
        root, protocol = ready_fixture(self)
        entry = protocol["required_evidence"][0]
        path = root / entry["path"]
        path.write_text("{", encoding="utf-8")
        repin(root, entry)
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("malformed_evidence", report["block_reasons"])
        self.assertFalse(report["ready"])

    # ------------------------------------------- producer integration blocked
    def test_blocked_n2_producer_output_blocks(self):
        root = fresh_root(self)
        protocol = approve_reviews(protocol_copy())
        for evidence_id in N3_EVIDENCE_IDS:
            stage_genuine_receipt(root, protocol, evidence_id)
        manifest = domain_pack_fixtures.default_manifest()
        del manifest["heldout"]
        pack_root = domain_pack_fixtures.write_pack(root / "n2_pack", manifest=manifest)
        n2_report = domain_pack.validate_pack(pack_root)
        self.assertEqual(n2_report["status"], "blocked")
        n2_entry = evidence_entry(protocol, N2_EVIDENCE_ID)
        write_json(root / n2_entry["path"], n2_report)
        repin(root, n2_entry)
        report = validator.validate_protocol(protocol, root=root)
        self.assertFalse(report["ready"])
        self.assertIn("evidence_status", report["block_reasons"])
        self.assertIn("evidence_field", report["block_reasons"])

    # ------------------------------------------------ strict type substitution
    def test_bool_for_int_substitution_blocks(self):
        root, protocol = ready_fixture(self)
        entry = evidence_entry(protocol, N2_EVIDENCE_ID)
        path = root / entry["path"]
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["merged_rows_written"] = False
        write_json(path, evidence)
        repin(root, entry)
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("evidence_field", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_int_for_bool_substitution_blocks(self):
        root, protocol = ready_fixture(self)
        entry = evidence_entry(protocol, "n3_protocol_preflight")
        path = root / entry["path"]
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["valid"] = 1
        write_json(path, evidence)
        repin(root, entry)
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("evidence_field", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_float_for_int_substitution_blocks(self):
        root, protocol = ready_fixture(self)
        entry = evidence_entry(protocol, "n3_synthetic_controls")
        path = root / entry["path"]
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["network_model_calls"] = 0.0
        write_json(path, evidence)
        repin(root, entry)
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("evidence_field", report["block_reasons"])
        self.assertIn("evidence_authorization_flag", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_safety_fields_reject_equal_valued_wrong_types(self):
        cases = (
            (N2_EVIDENCE_ID, "merged_rows_written", 0.0),
            (N2_EVIDENCE_ID, "input_files_changed", 0),
            ("n3_protocol_preflight", "model_loaded", 0),
            ("n3_protocol_preflight", "network_model_calls", False),
            ("n3_synthetic_controls", "synthetic_only", 1),
        )
        for evidence_id, key, value in cases:
            with self.subTest(evidence_id=evidence_id, key=key):
                root, protocol = ready_fixture(self)
                entry = evidence_entry(protocol, evidence_id)
                path = root / entry["path"]
                evidence = json.loads(path.read_text(encoding="utf-8"))
                evidence[key] = value
                write_json(path, evidence)
                repin(root, entry)
                report = validator.validate_protocol(protocol, root=root)
                self.assertFalse(report["ready"])
                self.assertIn("evidence_field", report["block_reasons"])

    def test_contract_expected_values_cannot_coerce_types(self):
        for key, value in (("merged_rows_written", False), ("input_files_changed", 0)):
            with self.subTest(key=key):
                root, protocol = ready_fixture(self)
                evidence_entry(protocol, N2_EVIDENCE_ID)["required_values"][key] = value
                report = validator.validate_protocol(protocol, root=root)
                self.assertFalse(report["ready"])
                self.assertIn("evidence_contract_mismatch", report["block_reasons"])

    def test_producer_version_fields_are_required(self):
        cases = (
            (N2_EVIDENCE_ID, "schema_version"),
            ("n3_protocol_preflight", "report_version"),
            ("n3_protocol_preflight", "schema_version"),
            ("n3_synthetic_controls", "report_version"),
            ("n3_synthetic_controls", "protocol_schema_version"),
        )
        for evidence_id, key in cases:
            for value in (None, "unsupported-v0"):
                with self.subTest(evidence_id=evidence_id, key=key, value=value):
                    root, protocol = ready_fixture(self)
                    entry = evidence_entry(protocol, evidence_id)
                    path = root / entry["path"]
                    evidence = json.loads(path.read_text(encoding="utf-8"))
                    if value is None:
                        evidence.pop(key)
                    else:
                        evidence[key] = value
                    write_json(path, evidence)
                    repin(root, entry)
                    report = validator.validate_protocol(protocol, root=root)
                    self.assertFalse(report["ready"])
                    self.assertIn("evidence_field", report["block_reasons"])

    def test_n2_cli_summary_cannot_replace_complete_report(self):
        root, protocol = ready_fixture(self)
        full_path = root / "complete-n2-report.json"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = domain_pack.main([
                "--pack", str(root / "n2_pack"), "--output", str(full_path),
            ])
        self.assertEqual(code, 0)
        entry = evidence_entry(protocol, N2_EVIDENCE_ID)
        path = root / entry["path"]
        path.write_text(stdout.getvalue(), encoding="utf-8")
        repin(root, entry)
        report = validator.validate_protocol(protocol, root=root)
        self.assertFalse(report["ready"])
        self.assertIn("evidence_field", report["block_reasons"])
        path.write_bytes(full_path.read_bytes())
        repin(root, entry)
        self.assertTrue(validator.validate_protocol(protocol, root=root)["ready"])

    def test_nonfinite_and_duplicate_json_evidence_is_rejected(self):
        for raw in ('{"status": NaN}', '{"status": Infinity}',
                    '{"status": -Infinity}', '{"valid": false, "valid": true}'):
            with self.subTest(raw=raw):
                root, protocol = ready_fixture(self)
                entry = evidence_entry(protocol, N2_EVIDENCE_ID)
                (root / entry["path"]).write_text(raw, encoding="utf-8")
                repin(root, entry)
                report = validator.validate_protocol(protocol, root=root)
                self.assertFalse(report["ready"])
                self.assertIn("malformed_evidence", report["block_reasons"])
                json.dumps(report, allow_nan=False)

    def test_nonfinite_and_duplicate_json_protocol_is_rejected(self):
        for raw in ('{"status": NaN}', '{"status": "pending", "status": "approved"}'):
            with self.subTest(raw=raw):
                root = fresh_root(self)
                path = root / "protocol.json"
                path.write_text(raw, encoding="utf-8")
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = validator.main(["--protocol", str(path), "--root", str(root)])
                self.assertEqual(code, 2)
                report = json.loads(stdout.getvalue())
                self.assertFalse(report["ready"])
                self.assertIn("protocol_unreadable", report["block_reasons"])

    def test_evidence_root_must_be_object(self):
        for value in (None, [], True, "report"):
            with self.subTest(value=value):
                root, protocol = ready_fixture(self)
                entry = evidence_entry(protocol, N2_EVIDENCE_ID)
                write_json(root / entry["path"], value)
                repin(root, entry)
                report = validator.validate_protocol(protocol, root=root)
                self.assertFalse(report["ready"])
                self.assertIn("malformed_evidence", report["block_reasons"])

    # -------------------------------------------------------- review gating
    def test_pending_review_blocks_even_when_receipts_are_clean(self):
        root, protocol = ready_fixture(self)
        protocol["review_requirements"][0]["status"] = "pending"
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("review_pending", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_review_rejection_blocks(self):
        root, protocol = ready_fixture(self)
        protocol["review_requirements"][0]["status"] = "rejected"
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("review_pending", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_duplicate_and_unknown_evidence_declarations_block(self):
        root, protocol = ready_fixture(self)
        protocol["required_evidence"].append(copy.deepcopy(protocol["required_evidence"][0]))
        protocol["required_evidence"][-1]["evidence_id"] = "unexpected"
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("unknown_evidence_declaration", report["block_reasons"])
        self.assertIn("evidence_count", report["block_reasons"])

    def test_duplicate_and_unknown_review_declarations_block(self):
        root, protocol = ready_fixture(self)
        protocol["review_requirements"].append(copy.deepcopy(protocol["review_requirements"][0]))
        protocol["review_requirements"][-1]["review_id"] = "unexpected_review"
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("unknown_review_declaration", report["block_reasons"])
        self.assertIn("review_count", report["block_reasons"])

    # -------------------------------------------------------------- contract
    def test_contract_kind_downgrade_blocks(self):
        root, protocol = ready_fixture(self)
        entry = evidence_entry(protocol, N2_EVIDENCE_ID)
        entry["kind"] = "generic_preflight"
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("evidence_contract_mismatch", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_contract_status_downgrade_blocks(self):
        root, protocol = ready_fixture(self)
        entry = evidence_entry(protocol, N2_EVIDENCE_ID)
        entry["expected_status"] = "preflight_passed"
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("evidence_contract_mismatch", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_contract_required_values_downgrade_blocks(self):
        root, protocol = ready_fixture(self)
        entry = evidence_entry(protocol, N2_EVIDENCE_ID)
        del entry["required_values"]["violations"]
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("evidence_contract_mismatch", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_malformed_required_values_declaration_blocks(self):
        for malformed in ([], {}, "required_values"):
            with self.subTest(malformed=malformed):
                root, protocol = ready_fixture(self)
                evidence_entry(protocol, N2_EVIDENCE_ID)["required_values"] = malformed
                report = validator.validate_protocol(protocol, root=root)
                self.assertIn("malformed_evidence_declaration", report["block_reasons"])
                self.assertFalse(report["ready"])

    def test_malformed_required_evidence_list_blocks(self):
        root, protocol = ready_fixture(self)
        protocol["required_evidence"] = {"not": "a list"}
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("missing_or_malformed_field", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_malformed_evidence_declaration_entry_blocks(self):
        root, protocol = ready_fixture(self)
        protocol["required_evidence"][0] = "not-an-object"
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("malformed_evidence_declaration", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_malformed_review_list_blocks(self):
        root, protocol = ready_fixture(self)
        protocol["review_requirements"] = "not-a-list"
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("missing_or_malformed_field", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_wrong_schema_version_blocks(self):
        root, protocol = ready_fixture(self)
        protocol["schema_version"] = "nanojev-v3-n3-s-readiness-v0"
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("schema_version", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_missing_schema_version_blocks(self):
        root, protocol = ready_fixture(self)
        del protocol["schema_version"]
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("schema_version", report["block_reasons"])
        self.assertFalse(report["ready"])

    # ------------------------------------------------------------------ paths
    def test_path_traversal_is_rejected(self):
        root, protocol = ready_fixture(self)
        protocol["required_evidence"][0]["path"] = "../outside.json"
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("unsafe_evidence_path", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_absolute_path_is_rejected(self):
        root, protocol = ready_fixture(self)
        protocol["required_evidence"][0]["path"] = str(root / "absolute.json")
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("unsafe_evidence_path", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_symlink_escape_is_rejected(self):
        root, protocol = ready_fixture(self)
        outside = fresh_root(self)
        target = outside / "outside.json"
        target.write_text('{"status": "preflight_passed_not_training_authorized"}', encoding="utf-8")
        link = root / "escape.json"
        link.symlink_to(target)
        entry = evidence_entry(protocol, N2_EVIDENCE_ID)
        entry["path"] = "escape.json"
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("unsafe_evidence_path", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_symlink_loop_is_rejected(self):
        root, protocol = ready_fixture(self)
        first = root / "loop-a.json"
        second = root / "loop-b.json"
        first.symlink_to(second)
        second.symlink_to(first)
        entry = evidence_entry(protocol, N2_EVIDENCE_ID)
        entry["path"] = "loop-a.json"
        report = validator.validate_protocol(protocol, root=root)
        self.assertFalse(report["ready"])
        self.assertTrue(report["block_reasons"])
        self.assertTrue(
            {"unsafe_evidence_path", "missing_evidence"} & set(report["block_reasons"])
        )

    # --------------------------------------------------------------- read I/O
    def test_permission_error_reading_evidence_blocks(self):
        root, protocol = ready_fixture(self)
        with mock.patch.object(Path, "read_bytes", side_effect=PermissionError("denied")):
            report = validator.validate_protocol(protocol, root=root)
        self.assertIn("evidence_unreadable", report["block_reasons"])
        self.assertNotIn("missing_evidence", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_file_not_found_while_reading_blocks(self):
        root, protocol = ready_fixture(self)
        with mock.patch.object(Path, "read_bytes", side_effect=FileNotFoundError("gone")):
            report = validator.validate_protocol(protocol, root=root)
        self.assertIn("evidence_unreadable", report["block_reasons"])
        self.assertNotIn("missing_evidence", report["block_reasons"])
        self.assertFalse(report["ready"])

    # ------------------------------------------------------------- hashes
    def test_missing_pinned_hash_is_rejected(self):
        root, protocol = ready_fixture(self)
        protocol["required_evidence"][0]["expected_sha256"] = ""
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("missing_pinned_hash", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_short_pinned_hash_is_rejected(self):
        root, protocol = ready_fixture(self)
        protocol["required_evidence"][0]["expected_sha256"] = "a" * 63
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("missing_pinned_hash", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_nonhex_pinned_hash_is_rejected(self):
        root, protocol = ready_fixture(self)
        protocol["required_evidence"][0]["expected_sha256"] = "g" * 64
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("malformed_pinned_hash", report["block_reasons"])
        self.assertFalse(report["ready"])

    # ---------------------------------------------------------- N2 immutability
    def test_n2_source_hash_after_mutation_blocks(self):
        root, protocol = ready_fixture(self)
        entry = evidence_entry(protocol, N2_EVIDENCE_ID)
        path = root / entry["path"]
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["source_hashes_after"] = dict(evidence["source_hashes"])
        evidence["source_hashes_after"]["manifest.json"] = "0" * 64
        write_json(path, evidence)
        repin(root, entry)
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("n2_source_hashes", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_n2_source_hash_key_removed_blocks(self):
        root, protocol = ready_fixture(self)
        entry = evidence_entry(protocol, N2_EVIDENCE_ID)
        path = root / entry["path"]
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["source_hashes"].pop("splits/train.jsonl")
        evidence["source_hashes_after"] = dict(evidence["source_hashes"])
        write_json(path, evidence)
        repin(root, entry)
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("n2_source_hashes", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_n2_source_paths_must_bind_hash_keys(self):
        root, protocol = ready_fixture(self)
        entry = evidence_entry(protocol, N2_EVIDENCE_ID)
        path = root / entry["path"]
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["source_paths"]["splits"]["train"] = "splits/not-the-hashed-file.jsonl"
        write_json(path, evidence)
        repin(root, entry)
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("n2_source_hashes", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_n2_source_paths_must_cover_exact_five_splits(self):
        root, protocol = ready_fixture(self)
        entry = evidence_entry(protocol, N2_EVIDENCE_ID)
        path = root / entry["path"]
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["source_paths"]["splits"].pop("ood")
        write_json(path, evidence)
        repin(root, entry)
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("n2_source_hashes", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_n2_source_paths_reject_escape(self):
        root, protocol = ready_fixture(self)
        entry = evidence_entry(protocol, N2_EVIDENCE_ID)
        path = root / entry["path"]
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["source_paths"]["splits"]["ood"] = "../outside.json"
        write_json(path, evidence)
        repin(root, entry)
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("n2_source_hashes", report["block_reasons"])
        self.assertFalse(report["ready"])

    # ------------------------------------------------------------------ bounds
    def test_boundary_true_is_rejected(self):
        root, protocol = ready_fixture(self)
        protocol["boundaries"]["real_n3_measurement_authorized"] = True
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("boundary_violation", report["block_reasons"])
        self.assertIn("authorization_flag", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_boundary_int_substitution_is_rejected(self):
        root, protocol = ready_fixture(self)
        protocol["boundaries"]["real_n3_measurement_authorized"] = 0
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("boundary_violation", report["block_reasons"])
        self.assertFalse(report["ready"])

    def test_nested_authorization_true_is_rejected(self):
        root, protocol = ready_fixture(self)
        protocol["metadata"] = {"future_authorized": True}
        report = validator.validate_protocol(protocol, root=root)
        self.assertIn("authorization_flag", report["block_reasons"])
        self.assertFalse(report["ready"])

    # --------------------------------------------------------------- mutation
    def test_protocol_is_not_mutated(self):
        root, protocol = ready_fixture(self)
        before = copy.deepcopy(protocol)
        validator.validate_protocol(protocol, root=root)
        self.assertEqual(protocol, before)

    # ------------------------------------------------------------------- CLI
    def test_cli_writes_exclusive_report(self):
        root, protocol = ready_fixture(self)
        research = root / "research"
        research.mkdir(parents=True, exist_ok=True)
        protocol_path = research / "protocol.json"
        write_json(protocol_path, protocol)
        output = root / "report.json"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(validator.main(["--protocol", str(protocol_path), "--root", str(root), "--output", str(output)]), 0)
        saved = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(saved["ready"])
        self.assertEqual(saved, json.loads(stdout.getvalue()))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(validator.main(["--protocol", str(protocol_path), "--root", str(root), "--output", str(output)]), 2)

    def test_cli_blocked_summary_is_written(self):
        root, protocol = ready_fixture(self)
        protocol["review_requirements"][0]["status"] = "pending"
        research = root / "research"
        research.mkdir(parents=True, exist_ok=True)
        protocol_path = research / "protocol.json"
        write_json(protocol_path, protocol)
        output = root / "blocked.json"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = validator.main(["--protocol", str(protocol_path), "--root", str(root), "--output", str(output)])
        self.assertEqual(code, 2)
        written = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(written["status"], "n3_s_readiness_blocked")
        self.assertFalse(written["ready"])
        for key in SAFE_AUTHORIZATION_KEYS:
            self.assertIs(written[key], False, key)
        self.assertIn("review_pending", written["block_reasons"])

    def test_cli_current_contract_returns_two(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(validator.main(["--protocol", str(PROTOCOL_PATH)]), 2)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["status"], "n3_s_readiness_blocked")
        self.assertIs(report["measurement_authorized"], False)


if __name__ == "__main__":
    unittest.main()
