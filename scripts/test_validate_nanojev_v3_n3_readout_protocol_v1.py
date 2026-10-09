#!/usr/bin/env python3
"""Synthetic, network-free tests for the N3 protocol validator."""

from __future__ import annotations

import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import validate_nanojev_v3_n3_readout_protocol_v1 as validator


ARMS = list(validator.REQUIRED_ARMS)
CONTROLS = list(validator.REQUIRED_CONTROLS)
RECEIPT_FIELDS = [
    "arm_id", "arm_role", "split", "case_id", "question_type", "option_order",
    "option_permutation_id", "label_permutation_id", "label_assignment",
    "predicted_label", "probabilities", "confidence", "coverage", "correct",
    "constant_true_correct", "constant_false_correct", "control_id", "protected_case",
    "latency_ms", "memory_bytes", "network_model_calls", "receipt_sha256",
    "training_authorized", "deployment_authorized",
]


def valid_protocol():
    arms = [
        {
            "arm_id": arm_id,
            "role": "incumbent" if arm_id == "trained_head" else "candidate",
            "description": f"synthetic {arm_id}",
            "input_serialization": "canonical state and question",
            "readout": "typed probability distribution",
            "paired": True,
        }
        for arm_id in ARMS
    ]
    controls = [
        {
            "control_id": control_id,
            "kind": "synthetic",
            "description": f"synthetic {control_id} control",
            "required": True,
            "applies_to": list(ARMS),
        }
        for control_id in CONTROLS
    ]
    return {
        "schema_version": validator.SCHEMA_VERSION,
        "protocol_id": "synthetic-n3",
        "status": "frozen_research_protocol",
        "purpose": "synthetic paired readout contract",
        "mode": "read_only_measurement",
        "frozen_before_inference": True,
        "read_only": True,
        "network_model_calls": 0,
        "paired_arms": arms,
        "pairing": {
            "paired": True,
            "same_frozen_splits": True,
            "same_controls": True,
            "same_receipt_fields": True,
        },
        "split_roles": {
            name: {
                "description": f"synthetic {name}",
                "frozen": True,
                "usable_for": ["fit" if name == "train" else "select" if name == "dev" else
                               "calibrate" if name == "calibration" else "evaluate"],
            }
            for name in validator.REQUIRED_SPLIT_ROLES
        },
        "controls": controls,
        "option_permutation": {
            "enabled": True,
            "identity_first": True,
            "all_permutations": True,
            "semantic_ids_fixed": True,
            "max_options": 4,
            "applies_to_question_types": ["boolean", "score", "choice"],
        },
        "label_permutation": {
            "enabled": True,
            "preserve_option_set": True,
            "preserve_label_multiset": True,
        },
        "coverage": {
            "boolean": {"required": True, "option_counts": [2]},
            "score": {"required": True, "option_counts": [2, 3, 4]},
        },
        "ood": {
            "required": True,
            "split": "ood",
            "abstain_required": True,
            "no_threshold_tuning_on_ood": True,
        },
        "protected_cases": [
            {"case_id": "protected-1", "split": "test", "reason": "synthetic guard"},
            {"case_id": "protected-2", "split": "ood", "reason": "synthetic OOD guard"},
        ],
        "constant_baselines": ["constant_true", "constant_false", "constant_majority", "abstain_all"],
        "paired_receipt_fields": list(RECEIPT_FIELDS),
        "boundaries": {
            "training_authorized": False,
            "deployment_authorized": False,
            "production_pruning_authorized": False,
            "active_pruning_authorized": False,
            "promotion_authorized": False,
            "authorizes_execution": False,
            "authorizes_training": False,
            "authorizes_deployment": False,
            "model_weights_modified": False,
            "checkpoint_written": False,
            "network_allowed": False,
            "shadow_only": True,
        },
        "decision_rule": "No global winner; all controls must pass before a per-pack Pareto report.",
        "limitations": ["synthetic contract only", "no model is loaded"],
    }


class ValidProtocolTests(unittest.TestCase):
    def test_clean_protocol_is_valid_but_never_authorizes(self):
        report = validator.validate_protocol(valid_protocol())
        self.assertTrue(report["valid"])
        self.assertEqual(report["status"], "protocol_valid_not_authorized")
        self.assertEqual(report["violations"], [])
        self.assertEqual(report["counts"]["arms"], 3)
        self.assertIs(report["training_authorized"], False)
        self.assertIs(report["deployment_authorized"], False)
        self.assertEqual(report["network_model_calls"], 0)

    def test_file_round_trip_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.json"
            path.write_text(json.dumps(valid_protocol()), encoding="utf-8")
            before = path.read_bytes()
            report = validator.validate_file(path)
            self.assertTrue(report["valid"])
            self.assertEqual(path.read_bytes(), before)


class CLITests(unittest.TestCase):
    def _write(self, root, protocol):
        path = Path(root) / "protocol.json"
        path.write_text(json.dumps(protocol), encoding="utf-8")
        return path

    def test_valid_cli_writes_exclusive_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol_path = self._write(tmp, valid_protocol())
            report_path = Path(tmp) / "report.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(
                    validator.main(["--protocol", str(protocol_path), "--output", str(report_path)]),
                    0,
                )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "protocol_valid_not_authorized")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    validator.main(["--protocol", str(protocol_path), "--output", str(report_path)]),
                    2,
                )

    def test_blocked_cli_returns_two_and_keeps_no_authorization(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol = valid_protocol()
            protocol["boundaries"]["deployment_authorized"] = True
            protocol_path = self._write(tmp, protocol)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(validator.main(["--protocol", str(protocol_path)]), 2)
            report = json.loads(stdout.getvalue())
            self.assertEqual(report["status"], "protocol_blocked")
            self.assertIs(report["deployment_authorized"], False)


class FailClosedTests(unittest.TestCase):
    def assert_blocked(self, mutate, code):
        protocol = valid_protocol()
        mutate(protocol)
        report = validator.validate_protocol(protocol)
        self.assertFalse(report["valid"])
        self.assertIn(code, report["block_reasons"])
        self.assertIs(report["training_authorized"], False)
        self.assertIs(report["deployment_authorized"], False)

    def test_missing_arm_is_blocked(self):
        self.assert_blocked(lambda p: p["paired_arms"].pop(), "missing_arm")

    def test_duplicate_arm_is_blocked(self):
        self.assert_blocked(lambda p: p["paired_arms"].append(copy.deepcopy(p["paired_arms"][0])), "duplicate_arm")

    def test_unpaired_arm_is_blocked(self):
        self.assert_blocked(lambda p: p["paired_arms"][0].update(paired=False), "unpaired_arm")

    def test_duplicate_control_is_blocked(self):
        self.assert_blocked(lambda p: p["controls"].append(copy.deepcopy(p["controls"][0])), "duplicate_control")

    def test_control_missing_an_arm_is_blocked(self):
        self.assert_blocked(lambda p: p["controls"][0].update(applies_to=ARMS[:2]), "incomplete_control")

    def test_unfrozen_test_split_is_blocked(self):
        self.assert_blocked(lambda p: p["split_roles"]["test"].update(usable_for=["fit"]), "unfrozen_split")

    def test_disabled_permutation_control_is_blocked(self):
        self.assert_blocked(lambda p: p["option_permutation"].update(enabled=False), "disabled_control")

    def test_missing_label_permutation_invariant_is_blocked(self):
        self.assert_blocked(lambda p: p["label_permutation"].update(preserve_label_multiset=False), "malformed_field")

    def test_bad_option_count_is_blocked(self):
        self.assert_blocked(lambda p: p["coverage"]["score"].update(option_counts=[5]), "malformed_field")

    def test_bad_ood_source_is_blocked(self):
        self.assert_blocked(lambda p: p["ood"].update(split="test"), "missing_ood")

    def test_protected_case_without_reason_is_blocked(self):
        self.assert_blocked(lambda p: p["protected_cases"][0].update(reason=""), "malformed_field")

    def test_duplicate_baseline_is_blocked(self):
        self.assert_blocked(lambda p: p.update(constant_baselines=["constant_true", "constant_true"]), "duplicate_control")

    def test_required_receipt_field_cannot_be_removed(self):
        self.assert_blocked(lambda p: p["paired_receipt_fields"].remove("arm_role"), "missing_receipt_field")

    def test_duplicate_receipt_field_is_blocked(self):
        self.assert_blocked(lambda p: p["paired_receipt_fields"].append("arm_id"), "duplicate_receipt_field")

    def test_missing_boundary_flag_is_blocked(self):
        self.assert_blocked(lambda p: p["boundaries"].pop("promotion_authorized"), "missing_field")

    def test_truthy_nested_authorization_flag_is_blocked(self):
        self.assert_blocked(lambda p: p.update(metadata={"future_authorized": True}), "authorization_flag")

    def test_network_calls_must_be_zero(self):
        self.assert_blocked(lambda p: p.update(network_model_calls=1), "network_enabled")

    def test_missing_protocol_file_is_blocked(self):
        report = validator.validate_file("/nonexistent/n3-protocol.json")
        self.assertFalse(report["valid"])
        self.assertIn("unreadable_protocol", report["block_reasons"])


if __name__ == "__main__":
    unittest.main()
