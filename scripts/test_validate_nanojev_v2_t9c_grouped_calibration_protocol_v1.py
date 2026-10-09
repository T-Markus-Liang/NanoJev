from __future__ import annotations

import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import validate_nanojev_v2_t9c_grouped_calibration_protocol_v1 as validator


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = ROOT / "research/nanojev_v2_t9c_grouped_calibration_protocol_v1.json"


def valid_protocol():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


class ValidProtocolTests(unittest.TestCase):
    def test_clean_protocol_is_valid_but_never_authorizes(self):
        report = validator.validate_protocol(valid_protocol())
        self.assertTrue(report["valid"])
        self.assertEqual(report["status"], "protocol_valid_not_authorized")
        self.assertEqual(report["violations"], [])
        self.assertEqual(report["network_model_calls"], 0)
        for key in (
            "measurement_authorized",
            "training_authorized",
            "deployment_authorized",
            "production_pruning_authorized",
            "promotion_authorized",
            "authorizes_execution",
        ):
            self.assertIs(report[key], False)

    def test_file_validation_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.json"
            path.write_text(json.dumps(valid_protocol()), encoding="utf-8")
            before = path.read_bytes()
            self.assertTrue(validator.validate_file(path)["valid"])
            self.assertEqual(path.read_bytes(), before)

    def test_malformed_root_is_blocked(self):
        report = validator.validate_protocol([])
        self.assertFalse(report["valid"])
        self.assertIn("malformed_root", report["block_reasons"])


class SplitAndGroupingTests(unittest.TestCase):
    def assert_blocked(self, mutate, code):
        protocol = valid_protocol()
        mutate(protocol)
        report = validator.validate_protocol(protocol)
        self.assertFalse(report["valid"])
        self.assertIn(code, report["block_reasons"])
        self.assertIs(report["authorizes_execution"], False)

    def test_test_cannot_fit(self):
        self.assert_blocked(lambda p: p["split_roles"]["test"].update(usable_for=["fit_grouped_temperature"]), "heldout_fit")

    def test_ood_cannot_calibrate(self):
        self.assert_blocked(lambda p: p["split_roles"]["ood"].update(usable_for=["calibrate"]), "heldout_fit")

    def test_calibration_is_only_fit_split(self):
        self.assert_blocked(lambda p: p["split_roles"]["dev"].update(usable_for=["fit_grouped_temperature"]), "wrong_fit_split")

    def test_dev_cannot_select_candidate(self):
        self.assert_blocked(lambda p: p["split_roles"]["dev"].update(usable_for=["select"]), "wrong_split_usage")

    def test_group_keys_are_exact(self):
        self.assert_blocked(lambda p: p["grouping"].update(keys=["question_type"]), "wrong_group_keys")

    def test_option_counts_are_bounded(self):
        self.assert_blocked(lambda p: p["grouping"].update(option_counts=[2, 5]), "wrong_option_counts")

    def test_missing_group_policy_keeps_baseline(self):
        self.assert_blocked(lambda p: p["grouping"].update(missing_group_policy="borrow_global"), "unsafe_missing_group_policy")

    def test_group_identity_is_recomputed(self):
        self.assert_blocked(lambda p: p["grouping"].update(group_identity_source="reported"), "unrecomputed_group")

    def test_minimum_group_size_must_be_positive_integer(self):
        self.assert_blocked(lambda p: p["grouping"].update(minimum_calibration_questions=16), "wrong_minimum")


class ObjectiveAndBoundaryTests(unittest.TestCase):
    def assert_blocked(self, mutate, code):
        protocol = valid_protocol()
        mutate(protocol)
        report = validator.validate_protocol(protocol)
        self.assertFalse(report["valid"])
        self.assertIn(code, report["block_reasons"])
        self.assertIs(report["training_authorized"], False)
        self.assertIs(report["deployment_authorized"], False)

    def test_primary_objective_is_nll(self):
        self.assert_blocked(lambda p: p["objectives"].update(primary="ece"), "wrong_primary_objective")

    def test_search_parameters_are_frozen(self):
        self.assert_blocked(lambda p: p["objectives"]["search"].update(grid_points=100), "wrong_search_grid")

    def test_thresholds_are_not_lowered(self):
        self.assert_blocked(lambda p: p["objectives"].update(thresholds=[0.5, 0.7, 0.8]), "wrong_thresholds")

    def test_uncertainty_is_clustered_and_paired(self):
        self.assert_blocked(lambda p: p["uncertainty"].update(resampling_unit="row"), "wrong_resampling_unit")
        self.assert_blocked(lambda p: p["uncertainty"].update(paired_against="candidate_only"), "unpaired_baseline")

    def test_service_temperature_change_is_forbidden(self):
        self.assert_blocked(lambda p: p["model"].update(service_files_may_change=True), "malformed_field")

    def test_network_calls_must_be_zero(self):
        self.assert_blocked(lambda p: p.update(network_model_calls=1), "network_not_zero")

    def test_authorization_flags_fail_closed(self):
        self.assert_blocked(lambda p: p["boundaries"].update(promotion_authorized=True), "authorization_flag")

    def test_shadow_only_is_required(self):
        self.assert_blocked(lambda p: p["boundaries"].update(shadow_only=False), "shadow_required")

    def test_protected_error_limit_is_zero(self):
        self.assert_blocked(lambda p: p["acceptance"].update(protected_confident_error_limit=1), "protected_error_limit")

    def test_missing_control_is_blocked(self):
        self.assert_blocked(lambda p: p.update(controls=[p["controls"][0]]), "missing_control")

    def test_duplicate_receipt_field_is_blocked(self):
        self.assert_blocked(lambda p: p["receipt_fields"].append("nll"), "duplicate_receipt_field")


class CLITests(unittest.TestCase):
    def test_valid_cli_writes_exclusive_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(validator.main(["--protocol", str(PROTOCOL_PATH), "--output", str(output)]), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "protocol_valid_not_authorized")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(validator.main(["--protocol", str(PROTOCOL_PATH), "--output", str(output)]), 2)

    def test_missing_protocol_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(validator.main(["--protocol", str(Path(tmp) / "missing.json")]), 2)
            report = json.loads(stdout.getvalue())
            self.assertEqual(report["status"], "protocol_blocked")
            self.assertIn("protocol_read_error", report["block_reasons"])


if __name__ == "__main__":
    unittest.main()
