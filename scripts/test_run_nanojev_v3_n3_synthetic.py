#!/usr/bin/env python3
"""Synthetic-only tests for the N3 paired-control runner."""

from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import run_nanojev_v3_n3_synthetic as runner


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = ROOT / "research" / "nanojev_v3_n3_readout_protocol_v1.json"


def protocol_copy():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


class SyntheticRunnerTests(unittest.TestCase):
    def test_clean_run_passes_all_controls_without_authorization(self):
        report = runner.run_protocol(protocol_copy())
        self.assertEqual(report["status"], runner.REPORT_STATUS)
        self.assertTrue(report["synthetic_only"])
        self.assertEqual(report["network_model_calls"], 0)
        self.assertIs(report["model_loaded"], False)
        self.assertIs(report["training_authorized"], False)
        self.assertIs(report["deployment_authorized"], False)
        self.assertIs(report["production_pruning_authorized"], False)
        self.assertGreater(report["receipt_count"], 0)
        self.assertTrue(all(item["passed"] for item in report["control_results"].values()))

    def test_report_and_receipt_hashes_are_deterministic(self):
        first = runner.run_protocol(protocol_copy())
        second = runner.run_protocol(protocol_copy())
        self.assertEqual(first["report_sha256"], second["report_sha256"])
        self.assertEqual(first["fixture_sha256"], second["fixture_sha256"])
        self.assertEqual(first["receipts"], second["receipts"])
        required = set(protocol_copy()["paired_receipt_fields"])
        for row in first["receipts"]:
            self.assertEqual(set(row), required)
            candidate = dict(row)
            expected = candidate.pop("receipt_sha256")
            candidate["receipt_sha256"] = None
            self.assertEqual(expected, runner.sha256_value(candidate))

    def test_receipts_never_contain_raw_input_fields(self):
        report = runner.run_protocol(protocol_copy())
        for row in report["receipts"]:
            self.assertNotIn("state", row)
            self.assertNotIn("instructions", row)
            self.assertNotIn("criteria", row)
            self.assertNotIn("raw_text", row)

    def test_ood_rows_are_abstentions(self):
        report = runner.run_protocol(protocol_copy())
        rows = [row for row in report["receipts"]
                if row["control_id"] == "option_permutation" and row["split"] == "ood"]
        self.assertTrue(rows)
        self.assertTrue(all(row["coverage"] is False and row["predicted_label"] is None for row in rows))

    def test_protected_rows_have_zero_confident_errors(self):
        report = runner.run_protocol(protocol_copy())
        self.assertEqual(report["control_results"]["protected_cases"]["confident_errors"], 0)

    def test_protocol_authorization_is_fail_closed(self):
        protocol = protocol_copy()
        protocol["boundaries"]["deployment_authorized"] = True
        with self.assertRaises(runner.SyntheticRunError):
            runner.run_protocol(protocol)

    def test_option_control_detects_tampering(self):
        protocol = protocol_copy()
        cases = runner.synthetic_cases(protocol)
        receipts = runner.build_receipts(protocol, cases)
        row = next(row for row in receipts if row["control_id"] == "option_permutation"
                   and not row["protected_case"] and row["split"] == "test")
        row["predicted_label"] = "a" if row["predicted_label"] != "a" else "b"
        with self.assertRaises(runner.SyntheticRunError):
            runner.validate_receipts(protocol, cases, receipts)

    def test_coverage_control_detects_missing_score_family(self):
        protocol = protocol_copy()
        cases = tuple(case for case in runner.synthetic_cases(protocol) if case.question_type != "score")
        receipts = runner.build_receipts(protocol, cases)
        with self.assertRaisesRegex(runner.SyntheticRunError, "Boolean and Score"):
            runner.validate_receipts(protocol, cases, receipts)

    def test_baseline_control_detects_missing_baseline(self):
        protocol = protocol_copy()
        cases = runner.synthetic_cases(protocol)
        receipts = [row for row in runner.build_receipts(protocol, cases)
                     if row["label_permutation_id"] != "baseline-abstain_all"]
        with self.assertRaisesRegex(runner.SyntheticRunError, "constant baseline"):
            runner.validate_receipts(protocol, cases, receipts)

    def test_cli_writes_report_exclusively(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "n3.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(runner.main(["--protocol", str(PROTOCOL_PATH), "--output", str(output)]), 0)
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], runner.REPORT_STATUS)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(["--protocol", str(PROTOCOL_PATH), "--output", str(output)]), 2)

    def test_cli_invalid_protocol_returns_two(self):
        protocol = protocol_copy()
        protocol["boundaries"]["training_authorized"] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blocked.json"
            path.write_text(json.dumps(protocol), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(["--protocol", str(path)]), 2)

    def test_required_baselines_are_emitted_for_each_arm(self):
        report = runner.run_protocol(protocol_copy())
        names = {row["label_permutation_id"].removeprefix("baseline-")
                 for row in report["receipts"] if row["control_id"] == "constant_baselines"}
        self.assertEqual(names, {"constant_true", "constant_false", "constant_majority", "abstain_all"})


if __name__ == "__main__":
    unittest.main()
