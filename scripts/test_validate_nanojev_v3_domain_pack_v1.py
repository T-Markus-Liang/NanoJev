#!/usr/bin/env python3
"""Synthetic stdlib tests for validate_nanojev_v3_domain_pack_v1.py.

All packs are fabricated in temporary directories from in-memory dicts. No real
corpus is read, no network/model/agent is touched, and no training/merge occurs.
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import validate_nanojev_v3_domain_pack_v1 as validator

SCHEMA = validator.SCHEMA
REQUIRED_SPLITS = validator.REQUIRED_SPLITS


def base_record(rid, split, state=None, source_group_id=None, lineage_id=None,
                question=None, provenance=None, extra=None):
    """One minimal, schema-valid domain-pack record."""
    record = {
        "id": rid,
        "split": split,
        "source_group_id": source_group_id if source_group_id is not None else f"src-{rid}",
        "lineage_id": lineage_id if lineage_id is not None else f"lin-{rid}",
        "provenance": provenance if provenance is not None else {
            "source_alias": f"fact-rule-{rid}", "derived_from_evaluation_corpus": False},
        "state": state if state is not None else f"state {rid}",
        "question": question if question is not None else {
            "type": "choice", "instructions": f"instructions {rid}",
            "criteria": {"a": "A", "b": "B"}},
        "target": {"a": 1.0, "b": 0.0},
    }
    if extra:
        record.update(extra)
    return record


def default_splits():
    return {split: [base_record(f"{split}-1", split)] for split in REQUIRED_SPLITS}


def default_manifest(splits=None, protected_cases=None, heldout=None):
    return {
        "schema_version": SCHEMA,
        "pack_id": "pack-1",
        "pack_version": "1.0.0",
        "heldout": heldout if heldout is not None else {
            "identity": "heldout-engineering-v1", "description": "Frozen OOD cohort"},
        "protected_cases": protected_cases if protected_cases is not None else [
            {"id": "test-1", "split": "test", "reason": "regression guard"}],
        "splits": {split: f"splits/{split}.jsonl" for split in REQUIRED_SPLITS},
    }


def write_pack(root, splits=None, manifest=None, write_files=True):
    """Write a manifest and split JSONL files; return the resolved pack root."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if splits is None:
        splits = default_splits()
    if manifest is None:
        manifest = default_manifest(splits)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if write_files:
        for split, records in splits.items():
            path = root / "splits" / f"{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return root


def all_pack_files(root):
    return sorted(str(path.relative_to(root)) for path in Path(root).rglob("*") if path.is_file())


class CleanPreflightTests(unittest.TestCase):
    def test_clean_pack_passes_but_is_not_training_authorized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_pack(Path(tmp) / "pack")
            report = validator.validate_pack(root)
        self.assertEqual(report["status"], "preflight_passed_not_training_authorized")
        self.assertEqual(report["block_reasons"], [])
        self.assertEqual(report["violations"], [])
        self.assertEqual(report["record_count"], 5)
        self.assertEqual(report["counts_by_split"],
                         {split: 1 for split in REQUIRED_SPLITS})
        self.assertIs(report["training_authorized"], False)
        self.assertIs(report["training_performed"], False)
        self.assertEqual(report["merged_rows_written"], 0)
        self.assertFalse(report["input_files_changed"])
        self.assertEqual(report["heldout"]["identity"], "heldout-engineering-v1")
        self.assertEqual(report["protected_cases"], {"declared": 1, "resolved": 1})

    def test_clean_pack_hashes_every_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_pack(Path(tmp) / "pack")
            report = validator.validate_pack(root)
        expected = {"manifest.json"} | {f"splits/{s}.jsonl" for s in REQUIRED_SPLITS}
        self.assertEqual(set(report["source_hashes"]), expected)
        self.assertEqual(report["source_paths"], {
            "manifest": "manifest.json",
            "splits": {split: f"splits/{split}.jsonl" for split in REQUIRED_SPLITS},
        })
        for value in report["source_hashes"].values():
            self.assertIsInstance(value, str)
            self.assertEqual(len(value), 64)

    def test_clean_report_identity_is_portable_across_pack_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_a = write_pack(Path(tmp) / "a")
            root_b = write_pack(Path(tmp) / "b")
            report_a = validator.validate_pack(root_a)
            report_b = validator.validate_pack(root_b)
        self.assertEqual(report_a, report_b)
        self.assertEqual(report_a["pack"], "pack-1@1.0.0")

    def test_validation_does_not_write_any_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_pack(Path(tmp) / "pack")
            before = all_pack_files(root)
            validator.validate_pack(root)
            self.assertEqual(all_pack_files(root), before)


class SchemaFailClosedTests(unittest.TestCase):
    def test_missing_pack_root_is_blocked(self):
        report = validator.validate_pack("/nonexistent/domain-pack")
        self.assertEqual(report["status"], "blocked")
        self.assertIn("pack_root_missing", report["block_reasons"])

    def test_missing_manifest_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pack"
            root.mkdir()
            report = validator.validate_pack(root)
        self.assertIn("manifest_missing", report["block_reasons"])

    def test_malformed_manifest_json_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pack"
            root.mkdir()
            (root / "manifest.json").write_text("{not json", encoding="utf-8")
            report = validator.validate_pack(root)
        self.assertIn("manifest_malformed", report["block_reasons"])
        self.assertTrue(report["manifest_errors"])

    def test_wrong_schema_version_is_blocked(self):
        manifest = default_manifest()
        manifest["schema_version"] = "nanojev-v3-domain-pack-v0"
        with tempfile.TemporaryDirectory() as tmp:
            report = validator.validate_pack(write_pack(Path(tmp) / "pack", manifest=manifest))
        self.assertIn("schema_version_mismatch", report["block_reasons"])

    def test_missing_heldout_identity_is_blocked(self):
        manifest = default_manifest()
        del manifest["heldout"]
        with tempfile.TemporaryDirectory() as tmp:
            report = validator.validate_pack(write_pack(Path(tmp) / "pack", manifest=manifest))
        self.assertIn("missing_heldout_identity", report["block_reasons"])

    def test_missing_split_declaration_is_blocked(self):
        manifest = default_manifest()
        del manifest["splits"]["ood"]
        with tempfile.TemporaryDirectory() as tmp:
            report = validator.validate_pack(write_pack(Path(tmp) / "pack", manifest=manifest))
        self.assertIn("missing_split", report["block_reasons"])

    def test_missing_split_file_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_pack(Path(tmp) / "pack")
            (root / "splits" / "ood.jsonl").unlink()
            report = validator.validate_pack(root)
        self.assertIn("split_file_missing", report["block_reasons"])

    def test_empty_split_is_blocked(self):
        splits = default_splits()
        splits["dev"] = []
        with tempfile.TemporaryDirectory() as tmp:
            report = validator.validate_pack(write_pack(Path(tmp) / "pack", splits=splits))
        self.assertIn("empty_split", report["block_reasons"])

    def test_malformed_record_line_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_pack(Path(tmp) / "pack")
            path = root / "splits" / "dev.jsonl"
            path.write_text(path.read_text(encoding="utf-8") + "{broken\n", encoding="utf-8")
            report = validator.validate_pack(root)
        self.assertIn("malformed_record", report["block_reasons"])

    def test_record_required_fields_fail_closed(self):
        cases = (
            ("missing_target", {"target": None}),
            ("malformed_question", {"question": {"type": "choice"}}),
            ("malformed_provenance", {"provenance": {"source_alias": "fact-rule"}}),
        )
        for expected, extra in cases:
            with self.subTest(expected=expected):
                splits = default_splits()
                row = base_record("train-1", "train", extra=extra)
                if expected == "missing_target":
                    row.pop("target")
                splits["train"] = [row]
                with tempfile.TemporaryDirectory() as tmp:
                    report = validator.validate_pack(
                        write_pack(Path(tmp) / "pack", splits=splits))
                self.assertIn(expected, report["block_reasons"])


class IdentityFailClosedTests(unittest.TestCase):
    def test_duplicate_record_id_is_blocked(self):
        splits = default_splits()
        splits["train"].append(base_record("train-1", "train", state="other state"))
        with tempfile.TemporaryDirectory() as tmp:
            report = validator.validate_pack(write_pack(Path(tmp) / "pack", splits=splits))
        self.assertIn("duplicate_record_id", report["block_reasons"])

    def test_missing_record_id_is_blocked(self):
        splits = default_splits()
        splits["dev"] = [base_record("dev-1", "dev", extra={"id": ""})]
        with tempfile.TemporaryDirectory() as tmp:
            report = validator.validate_pack(write_pack(Path(tmp) / "pack", splits=splits))
        self.assertIn("missing_record_id", report["block_reasons"])

    def test_record_split_mismatch_is_blocked(self):
        splits = default_splits()
        splits["dev"] = [base_record("dev-1", "test", state="mismatch state")]
        with tempfile.TemporaryDirectory() as tmp:
            report = validator.validate_pack(write_pack(Path(tmp) / "pack", splits=splits))
        self.assertIn("record_split_mismatch", report["block_reasons"])


class CrossSplitIsolationTests(unittest.TestCase):
    def test_source_group_crossing_splits_is_blocked(self):
        splits = default_splits()
        splits["dev"] = [base_record("dev-1", "dev", source_group_id="src-train-1")]
        with tempfile.TemporaryDirectory() as tmp:
            report = validator.validate_pack(write_pack(Path(tmp) / "pack", splits=splits))
        self.assertIn("source_group_id_crosses_splits", report["block_reasons"])

    def test_lineage_crossing_splits_is_blocked(self):
        splits = default_splits()
        splits["dev"] = [base_record("dev-1", "dev", lineage_id="lin-train-1")]
        with tempfile.TemporaryDirectory() as tmp:
            report = validator.validate_pack(write_pack(Path(tmp) / "pack", splits=splits))
        self.assertIn("lineage_id_crosses_splits", report["block_reasons"])

    def test_missing_source_or_lineage_is_blocked(self):
        splits = default_splits()
        splits["train"] = [base_record("train-1", "train", source_group_id="",
                                       lineage_id="")]
        with tempfile.TemporaryDirectory() as tmp:
            report = validator.validate_pack(write_pack(Path(tmp) / "pack", splits=splits))
        self.assertIn("missing_source_group_id", report["block_reasons"])
        self.assertIn("missing_lineage_id", report["block_reasons"])


class CanonicalInputTests(unittest.TestCase):
    def _pack_with(self, train_record, test_record, tmp):
        splits = default_splits()
        splits["train"] = [train_record]
        splits["test"] = [test_record]
        return write_pack(Path(tmp) / "pack", splits=splits)

    def test_identical_visible_inputs_across_splits_are_blocked(self):
        shared = {"type": "choice", "instructions": "pick shared",
                  "criteria": {"a": "A", "b": "B"}}
        with tempfile.TemporaryDirectory() as tmp:
            root = self._pack_with(
                base_record("train-1", "train", state="shared evidence",
                            source_group_id="train-src", lineage_id="train-lin",
                            question=shared),
                base_record("test-1", "test", state="shared evidence",
                            source_group_id="test-src", lineage_id="test-lin",
                            question=shared),
                tmp)
            report = validator.validate_pack(root)
        self.assertIn("canonical_input_crosses_splits", report["block_reasons"])

    def test_choice_criteria_dict_order_is_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._pack_with(
                base_record("train-1", "train", state="same state",
                            question={"type": "choice", "instructions": "pick",
                                      "criteria": {"a": "A", "b": "B", "c": "C"}}),
                base_record("test-1", "test", state="same state",
                            question={"type": "choice", "instructions": "pick",
                                      "criteria": {"c": "C", "a": "A", "b": "B"}}),
                tmp)
            report = validator.validate_pack(root)
        self.assertIn("canonical_input_crosses_splits", report["block_reasons"])

    def test_ordinal_score_list_order_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._pack_with(
                base_record("train-1", "train", state="same state",
                            question={"type": "score", "instructions": "rate",
                                      "criteria": ["low", "mid", "high"]}),
                base_record("test-1", "test", state="same state",
                            question={"type": "score", "instructions": "rate",
                                      "criteria": ["high", "mid", "low"]}),
                tmp)
            report = validator.validate_pack(root)
        self.assertNotIn("canonical_input_crosses_splits", report["block_reasons"])

    def test_changed_state_does_not_collide(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._pack_with(
                base_record("train-1", "train", state="evidence alpha"),
                base_record("test-1", "test", state="evidence beta"),
                tmp)
            report = validator.validate_pack(root)
        self.assertNotIn("canonical_input_crosses_splits", report["block_reasons"])

    def test_visible_input_excludes_ids_and_gold(self):
        row = base_record("r1", "train")
        visible = validator.visible_input(row)
        self.assertEqual(set(visible), {"state", "type", "instructions", "criteria"})
        for excluded in ("id", "split", "source_group_id", "lineage_id", "target"):
            self.assertNotIn(excluded, visible)


class ProvenanceTests(unittest.TestCase):
    def test_aliases_are_recognized(self):
        for alias in ("abstention-survey", "abstention_survey", "Abstention Survey",
                      "workflow challenge", "workflow-v2_evaluation",
                      "context relevance test", "CONTEXT-RELEVANCE-OOD"):
            self.assertTrue(validator.provenance_is_evaluation_derived(alias), alias)
        self.assertFalse(validator.provenance_is_evaluation_derived("family_fact_rule_17"))

    def test_false_flag_with_alias_is_still_blocked(self):
        splits = default_splits()
        splits["train"] = [base_record(
            "train-1", "train",
            provenance={"source_alias": "abstention-survey",
                        "derived_from_evaluation_corpus": False})]
        with tempfile.TemporaryDirectory() as tmp:
            report = validator.validate_pack(write_pack(Path(tmp) / "pack", splits=splits))
        self.assertIn("evaluation_derived_provenance", report["block_reasons"])

    def test_true_flag_is_blocked(self):
        splits = default_splits()
        splits["dev"] = [base_record(
            "dev-1", "dev",
            provenance={"source_alias": "family_fact_rule_17",
                        "derived_from_evaluation_corpus": True})]
        with tempfile.TemporaryDirectory() as tmp:
            report = validator.validate_pack(write_pack(Path(tmp) / "pack", splits=splits))
        self.assertIn("evaluation_derived_provenance", report["block_reasons"])


class ProtectedCoverageTests(unittest.TestCase):
    def test_empty_protected_declarations_are_blocked(self):
        manifest = default_manifest(protected_cases=[])
        with tempfile.TemporaryDirectory() as tmp:
            report = validator.validate_pack(write_pack(Path(tmp) / "pack", manifest=manifest))
        self.assertIn("missing_protected_case_declarations", report["block_reasons"])

    def test_protected_case_reason_is_required(self):
        manifest = default_manifest(protected_cases=[{"id": "test-1", "split": "test"}])
        with tempfile.TemporaryDirectory() as tmp:
            report = validator.validate_pack(write_pack(Path(tmp) / "pack", manifest=manifest))
        self.assertIn("malformed_protected_case_declaration", report["block_reasons"])

    def test_absent_protected_case_is_blocked(self):
        manifest = default_manifest(protected_cases=[
            {"id": "test-999", "split": "test", "reason": "missing"}])
        with tempfile.TemporaryDirectory() as tmp:
            report = validator.validate_pack(write_pack(Path(tmp) / "pack", manifest=manifest))
        self.assertIn("protected_case_absent_from_test_or_ood", report["block_reasons"])

    def test_protected_case_declared_for_train_is_blocked(self):
        manifest = default_manifest(protected_cases=[
            {"id": "train-1", "split": "train", "reason": "wrong split"}])
        with tempfile.TemporaryDirectory() as tmp:
            report = validator.validate_pack(write_pack(Path(tmp) / "pack", manifest=manifest))
        self.assertIn("protected_case_declared_for_wrong_split", report["block_reasons"])

    def test_ood_protected_case_resolves(self):
        splits = default_splits()
        splits["ood"].append(base_record("ood-protected", "ood"))
        manifest = default_manifest(protected_cases=[
            {"id": "ood-protected", "split": "ood", "reason": "ood guard"}])
        with tempfile.TemporaryDirectory() as tmp:
            report = validator.validate_pack(write_pack(Path(tmp) / "pack",
                                                        splits=splits, manifest=manifest))
        self.assertEqual(report["protected_cases"], {"declared": 1, "resolved": 1})
        self.assertEqual(report["status"], "preflight_passed_not_training_authorized")


class HashImmutabilityTests(unittest.TestCase):
    def test_hashes_recorded_before_and_after_are_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_pack(Path(tmp) / "pack")
            paths = [root / "manifest.json"] + [
                root / "splits" / f"{s}.jsonl" for s in REQUIRED_SPLITS]
            before = {str(p.relative_to(root)): validator.file_hash(p) for p in paths}
            report = validator.validate_pack(root)
            after = {str(p.relative_to(root)): validator.file_hash(p) for p in paths}
        self.assertEqual(before, after)
        self.assertEqual(report["source_hashes"], before)
        self.assertEqual(report["source_hashes_after"], before)
        self.assertFalse(report["input_files_changed"])

    def test_recorded_hashes_change_with_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_a = write_pack(Path(tmp) / "a")
            root_b = write_pack(Path(tmp) / "b")
            (root_b / "splits" / "train.jsonl").write_text(
                json.dumps(base_record("train-1", "train", state="changed")) + "\n",
                encoding="utf-8")
            report_a = validator.validate_pack(root_a)
            report_b = validator.validate_pack(root_b)
        self.assertNotEqual(report_a["source_hashes"]["splits/train.jsonl"],
                            report_b["source_hashes"]["splits/train.jsonl"])


class CLITests(unittest.TestCase):
    def run_main(self, argv):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = validator.main(argv)
        return code, stream.getvalue()

    def test_clean_pack_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_pack(Path(tmp) / "pack")
            code, out = self.run_main(["--pack", str(root)])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["status"], "preflight_passed_not_training_authorized")
        self.assertEqual(payload["block_reasons"], [])
        self.assertFalse(payload["training_authorized"])

    def test_blocked_pack_exits_two(self):
        manifest = default_manifest()
        del manifest["heldout"]
        with tempfile.TemporaryDirectory() as tmp:
            root = write_pack(Path(tmp) / "pack", manifest=manifest)
            code, out = self.run_main(["--pack", str(root)])
        self.assertEqual(code, 2)
        payload = json.loads(out)
        self.assertEqual(payload["status"], "blocked")
        self.assertIn("missing_heldout_identity", payload["block_reasons"])

    def test_output_is_exclusive_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_pack(Path(tmp) / "pack")
            output = Path(tmp) / "report.json"
            code, out = self.run_main(["--pack", str(root), "--output", str(output)])
            self.assertEqual(code, 0)
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(written["status"], "preflight_passed_not_training_authorized")
            self.assertIn("output", json.loads(out))

            original = output.read_text(encoding="utf-8")
            code, out = self.run_main(["--pack", str(root), "--output", str(output)])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(out)["status"], "error")
            self.assertEqual(output.read_text(encoding="utf-8"), original,
                             "an existing report must never be overwritten")

    def test_blocked_report_is_written_without_overwrite(self):
        manifest = default_manifest()
        del manifest["heldout"]
        with tempfile.TemporaryDirectory() as tmp:
            root = write_pack(Path(tmp) / "pack", manifest=manifest)
            output = Path(tmp) / "blocked.json"
            code, out = self.run_main(["--pack", str(root), "--output", str(output)])
            self.assertEqual(code, 2)
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(written["status"], "blocked")
            self.assertFalse(written["training_authorized"])
            self.assertEqual(written["merged_rows_written"], 0)

    def test_output_inside_pack_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_pack(Path(tmp) / "pack")
            output = root / "report.json"
            with mock.patch.object(validator, "validate_pack") as mocked:
                with self.assertRaises(SystemExit) as caught:
                    with contextlib.redirect_stderr(io.StringIO()):
                        validator.main(["--pack", str(root), "--output", str(output)])
            self.assertEqual(caught.exception.code, 2)
            mocked.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
