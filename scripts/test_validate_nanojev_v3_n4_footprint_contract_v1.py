#!/usr/bin/env python3
"""Synthetic, network-free tests for the N4 footprint contract validator."""
from __future__ import annotations

import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import validate_nanojev_v3_n4_footprint_contract_v1 as validator


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = ROOT / "research" / "nanojev_v3_n4_footprint_contract_v1.json"


def protocol_copy():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


class N4ContractTests(unittest.TestCase):
    def assert_blocked(self, mutate, code):
        protocol = protocol_copy()
        mutate(protocol)
        report = validator.validate_protocol(protocol)
        self.assertFalse(report["valid"])
        self.assertIn(code, report["block_reasons"])
        self.assertIs(report["training_authorized"], False)
        self.assertIs(report["deployment_authorized"], False)
        self.assertIs(report["model_loaded"], False)
        self.assertEqual(report["network_model_calls"], 0)

    def test_clean_contract_is_valid_but_never_authorizes(self):
        report = validator.validate_protocol(protocol_copy())
        self.assertTrue(report["valid"])
        self.assertEqual(report["status"], "footprint_contract_valid_not_authorized")
        self.assertEqual(report["violations"], [])
        self.assertEqual(report["counts"]["candidates"], 5)
        self.assertEqual(report["counts"]["required_metrics"], len(validator.REQUIRED_METRICS))
        self.assertEqual(report["counts"]["receipt_fields"], len(validator.REQUIRED_RECEIPT_FIELDS))

    def test_file_validation_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.json"
            path.write_text(json.dumps(protocol_copy()), encoding="utf-8")
            before = path.read_bytes()
            report = validator.validate_file(path)
            self.assertTrue(report["valid"])
            self.assertEqual(path.read_bytes(), before)

    def test_cli_valid_output_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol = Path(tmp) / "protocol.json"
            protocol.write_text(json.dumps(protocol_copy()), encoding="utf-8")
            output = Path(tmp) / "report.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(validator.main(["--protocol", str(protocol), "--output", str(output)]), 0)
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "footprint_contract_valid_not_authorized")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(validator.main(["--protocol", str(protocol), "--output", str(output)]), 2)

    def test_missing_candidate_is_blocked(self):
        self.assert_blocked(lambda p: p["candidate_ladder"].pop(), "missing_candidate")

    def test_duplicate_candidate_is_blocked(self):
        self.assert_blocked(lambda p: p["candidate_ladder"].append(copy.deepcopy(p["candidate_ladder"][0])), "duplicate_candidate")

    def test_unknown_candidate_is_blocked(self):
        self.assert_blocked(lambda p: p["candidate_ladder"][0].update(candidate_id="fp8_unknown"), "unknown_candidate")

    def test_baseline_parent_is_blocked(self):
        self.assert_blocked(lambda p: p["candidate_ladder"][0].update(parent_candidate_id="fp16_cast"), "baseline_parent")

    def test_candidate_pairing_is_blocked(self):
        self.assert_blocked(lambda p: p["candidate_ladder"][1].update(paired_with="int8_weight_only"), "unpaired_candidate")

    def test_unknown_parent_is_blocked(self):
        self.assert_blocked(lambda p: p["candidate_ladder"][1].update(parent_candidate_id="fp8_unknown"), "unknown_parent")

    def test_candidate_kind_is_fixed(self):
        self.assert_blocked(lambda p: p["candidate_ladder"][1].update(kind="distillation"), "candidate_kind")

    def test_pairing_invariant_is_blocked(self):
        self.assert_blocked(lambda p: p["pairing"].update(same_tokenizer=False), "pairing_invariant")

    def test_test_split_cannot_fit(self):
        self.assert_blocked(lambda p: p["split_roles"]["test"].update(usable_for=["fit"]), "unfrozen_split")

    def test_unknown_split_is_blocked(self):
        self.assert_blocked(lambda p: p["split_roles"].update(extra={"frozen": True, "usable_for": ["evaluate"]}), "unknown_split")

    def test_metric_contract_is_exact(self):
        self.assert_blocked(lambda p: p["required_metrics"].pop(), "metric_contract")

    def test_receipt_contract_rejects_duplicates(self):
        self.assert_blocked(lambda p: p["receipt_fields"].append("candidate_id"), "receipt_contract")

    def test_quality_gate_protected_error_is_zero(self):
        self.assert_blocked(lambda p: p["quality_gates"].update(protected_error_count_max=1), "quality_gate")

    def test_authorization_nested_flag_is_blocked(self):
        self.assert_blocked(lambda p: p.update(metadata={"future_authorized": True}), "authorization_flag")

    def test_network_calls_are_blocked(self):
        self.assert_blocked(lambda p: p.update(network_model_calls=1), "network_boundary")

    def test_model_load_is_blocked(self):
        self.assert_blocked(lambda p: p.update(model_loaded=True), "execution_boundary")

    def test_quantization_execution_is_blocked(self):
        self.assert_blocked(lambda p: p.update(quantization_performed=True), "execution_boundary")

    def test_bad_evaluation_splits_are_blocked(self):
        self.assert_blocked(lambda p: p["candidate_ladder"][1].update(evaluation_splits=["test"]), "malformed_field")

    def test_bad_calibration_split_is_blocked(self):
        self.assert_blocked(lambda p: p["candidate_ladder"][1].update(calibration_split="test"), "split_contract")

    def test_missing_boundary_flag_is_blocked(self):
        self.assert_blocked(lambda p: p["boundaries"].pop("promotion_authorized"), "missing_field")

    def test_malformed_file_returns_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{", encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(validator.main(["--protocol", str(path)]), 2)
            report = json.loads(output.getvalue())
            self.assertEqual(report["status"], "footprint_contract_blocked")
            self.assertIs(report["deployment_authorized"], False)


if __name__ == "__main__":
    unittest.main()
