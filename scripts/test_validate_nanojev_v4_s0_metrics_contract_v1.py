#!/usr/bin/env python3
"""Network-free tests for the V4-S0 metrics contract validator."""
from __future__ import annotations

import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import validate_nanojev_v4_s0_metrics_contract_v1 as validator


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = ROOT / "research" / "nanojev_v4_s0_metrics_contract_v1.json"


def protocol_copy():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


class V4S0MetricsContractTests(unittest.TestCase):
    def assert_blocked(self, mutate, code):
        protocol = protocol_copy()
        mutate(protocol)
        report = validator.validate_protocol(protocol)
        self.assertFalse(report["valid"])
        self.assertIn(code, report["block_reasons"])
        self.assertIs(report["measurement_authorized"], False)
        self.assertIs(report["training_authorized"], False)
        self.assertIs(report["deployment_authorized"], False)
        self.assertEqual(report["network_model_calls"], 0)

    def test_clean_contract_is_valid_but_not_authorized(self):
        report = validator.validate_protocol(protocol_copy())
        self.assertTrue(report["valid"])
        self.assertEqual(report["status"], "metrics_contract_valid_not_authorized")
        self.assertEqual(report["violations"], [])
        self.assertEqual(report["counts"], {"candidates": 5, "scopes": 3, "required_receipt_fields": 24})

    def test_file_validation_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.json"
            path.write_text(json.dumps(protocol_copy()), encoding="utf-8")
            before = path.read_bytes()
            report = validator.validate_file(path)
            self.assertTrue(report["valid"])
            self.assertRegex(report["protocol_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(path.read_bytes(), before)

    def test_cli_output_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol = Path(tmp) / "protocol.json"
            protocol.write_text(json.dumps(protocol_copy()), encoding="utf-8")
            output = Path(tmp) / "report.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(validator.main(["--protocol", str(protocol), "--output", str(output)]), 0)
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "metrics_contract_valid_not_authorized")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(validator.main(["--protocol", str(protocol), "--output", str(output)]), 2)

    def test_missing_candidate_is_blocked(self):
        self.assert_blocked(lambda p: p["candidate_ladder"].pop(), "missing_candidate")

    def test_duplicate_candidate_is_blocked(self):
        self.assert_blocked(lambda p: p["candidate_ladder"].append(copy.deepcopy(p["candidate_ladder"][0])), "duplicate_candidate")

    def test_candidate_parent_is_fixed(self):
        self.assert_blocked(lambda p: p["candidate_ladder"][1].update(parent_candidate_id="int8_weight_only"), "candidate_contract")

    def test_candidate_requirements_are_fixed(self):
        self.assert_blocked(lambda p: p["candidate_ladder"][1].update(requires_quantization=False), "candidate_contract")

    def test_scope_set_is_fixed(self):
        self.assert_blocked(lambda p: p["hardware_workload_contract"]["scopes"].pop(), "scope_contract")

    def test_scope_latency_fields_are_fixed(self):
        self.assert_blocked(lambda p: p["hardware_workload_contract"]["scopes"][0].update(required_latency_fields=["p95_ms"]), "scope_contract")

    def test_local_serving_requires_cold_p99(self):
        self.assert_blocked(lambda p: p["hardware_workload_contract"]["scopes"][2].update(required_latency_fields=["cold_p95_ms", "warm_p50_ms", "warm_p95_ms", "warm_p99_ms"]), "scope_contract")

    def test_targets_are_bound_to_local_serving(self):
        self.assert_blocked(lambda p: p["target_profiles"]["v4_m1_research"].update(warm_single_decision_scope_id="model_compute"), "scope_binding")

    def test_research_cold_target_is_bound_to_local_serving(self):
        self.assert_blocked(lambda p: p["target_profiles"]["v4_m1_research"].update(cold_start_scope_id="paper_decision_e2e"), "scope_binding")

    def test_release_warm_target_is_bound_to_local_serving(self):
        self.assert_blocked(lambda p: p["target_profiles"]["v4_m2_release"].update(warm_single_decision_scope_id="model_compute"), "scope_binding")

    def test_release_cold_target_is_bound_to_local_serving(self):
        self.assert_blocked(lambda p: p["target_profiles"]["v4_m2_release"].update(cold_start_scope_id="model_compute"), "scope_binding")

    def test_targets_must_be_stricter_for_release(self):
        self.assert_blocked(lambda p: p["target_profiles"]["v4_m2_release"].update(package_bytes_max=1500000000), "threshold_contract")

    def test_zero_target_is_blocked(self):
        self.assert_blocked(lambda p: p["target_profiles"]["v4_m1_research"].update(package_bytes_max=0), "threshold_contract")

    def test_quality_metric_order_is_fixed(self):
        self.assert_blocked(lambda p: p["quality_contract"].update(required_metrics=["accuracy"]), "quality_contract")

    def test_quality_cannot_allow_protected_errors(self):
        self.assert_blocked(lambda p: p["quality_contract"].update(protected_error_count_max=1), "threshold_contract")

    def test_quality_cannot_fit_test_or_ood(self):
        self.assert_blocked(lambda p: p["quality_contract"].update(test_ood_fit_allowed=True), "quality_contract")

    def test_cost_cannot_claim_absolute_currency(self):
        self.assert_blocked(lambda p: p["local_cost_contract"].update(absolute_currency_allowed=True), "cost_contract")

    def test_cost_fields_are_fixed(self):
        self.assert_blocked(lambda p: p["local_cost_contract"].update(required_fields=["wall_time_ms"]), "cost_contract")

    def test_reliability_requires_zero_network(self):
        self.assert_blocked(lambda p: p["reliability_contract"].update(network_model_calls_max=1), "reliability_contract")

    def test_active_pruning_default_is_blocked(self):
        self.assert_blocked(lambda p: p["reliability_contract"].update(active_pruning_default=True), "reliability_contract")

    def test_boundary_authorization_is_blocked(self):
        self.assert_blocked(lambda p: p["boundaries"].update(measurement_authorized=True), "authorization_flag")

    def test_nested_authorization_is_blocked(self):
        self.assert_blocked(lambda p: p.update(metadata={"future_authorized": True}), "authorization_flag")

    def test_owner_approval_is_blocked(self):
        self.assert_blocked(lambda p: p.update(owner_approved=True), "execution_boundary")

    def test_receipt_field_order_and_set_are_fixed(self):
        self.assert_blocked(lambda p: p["required_receipt_fields"].pop(), "receipt_contract")

    def test_wrong_status_is_blocked(self):
        self.assert_blocked(lambda p: p.update(status="measurement_authorized"), "status_contract")

    def test_freeze_scope_is_explicit(self):
        self.assert_blocked(lambda p: p.update(freeze_scope="numeric_targets"), "freeze_scope")

    def test_approval_mode_is_pre_approval_only(self):
        self.assert_blocked(lambda p: p.update(approval_mode="approved"), "approval_mode")

    def test_wrong_root_is_blocked(self):
        report = validator.validate_protocol([], source="<test>")
        self.assertFalse(report["valid"])
        self.assertIn("malformed_root", report["block_reasons"])

    def test_malformed_file_returns_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{", encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(validator.main(["--protocol", str(path)]), 2)
            report = json.loads(output.getvalue())
            self.assertEqual(report["status"], "metrics_contract_blocked")
            self.assertIs(report["measurement_authorized"], False)


if __name__ == "__main__":
    unittest.main()
