#!/usr/bin/env python3
"""Synthetic-only tests for the N4 footprint-control runner.

These tests intentionally assert that the runner never fabricates a footprint,
latency, quality, or cost measurement.  The runner is only a contract/receipt
smoke test until an independently reviewed phase authorizes real measurement.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import run_nanojev_v3_n4_synthetic as runner


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = ROOT / "research" / "nanojev_v3_n4_footprint_contract_v1.json"


def protocol_copy():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


class SyntheticFootprintRunnerTests(unittest.TestCase):
    def test_clean_run_passes_controls_without_authorization(self):
        report = runner.run_protocol(protocol_copy())
        self.assertEqual(report["status"], runner.REPORT_STATUS)
        self.assertTrue(report["synthetic_only"])
        self.assertFalse(report["measurement_evidence"])
        self.assertEqual(report["candidate_count"], 5)
        self.assertEqual(report["receipt_count"], 5)
        self.assertEqual(report["network_model_calls"], 0)
        self.assertIs(report["model_loaded"], False)
        self.assertIs(report["artifact_generation_performed"], False)
        self.assertIs(report["quantization_performed"], False)
        self.assertIs(report["training_performed"], False)
        self.assertIs(report["deployment_performed"], False)
        self.assertIs(report["training_authorized"], False)
        self.assertIs(report["deployment_authorized"], False)
        self.assertIs(report["production_pruning_authorized"], False)
        self.assertIs(report["promotion_authorized"], False)
        controls = report["control_results"]
        self.assertTrue(controls["candidate_receipts"]["passed"])
        self.assertTrue(controls["no_fabricated_measurements"])
        self.assertTrue(controls["no_raw_or_path_fields"])

    def test_receipts_follow_frozen_candidate_order_and_field_set(self):
        protocol = protocol_copy()
        report = runner.run_protocol(protocol)
        expected_ids = [item["candidate_id"] for item in protocol["candidate_ladder"]]
        self.assertEqual([row["candidate_id"] for row in report["receipts"]], expected_ids)
        required_fields = set(protocol["receipt_fields"])
        for row in report["receipts"]:
            self.assertEqual(set(row), required_fields)

    def test_hashes_are_deterministic_and_recomputable(self):
        first = runner.run_protocol(protocol_copy())
        second = runner.run_protocol(protocol_copy())
        self.assertEqual(first["report_sha256"], second["report_sha256"])
        self.assertEqual(first["protocol_sha256"], second["protocol_sha256"])
        self.assertEqual(first["receipts"], second["receipts"])
        for row in first["receipts"]:
            payload = dict(row)
            expected = payload["receipt_sha256"]
            payload["receipt_sha256"] = None
            self.assertEqual(expected, runner.sha256_value(payload))
        payload = {key: value for key, value in first.items() if key != "report_sha256"}
        self.assertEqual(first["report_sha256"], runner.sha256_value(payload))

    def test_measurement_fields_are_null_and_no_raw_or_path_content(self):
        report = runner.run_protocol(protocol_copy())
        forbidden = runner.RAW_OR_PATH_KEYS
        for row in report["receipts"]:
            self.assertTrue(all(row[field] is None for field in runner.MEASUREMENT_FIELDS))
            self.assertTrue(forbidden.isdisjoint(row))
        self.assertTrue(report["control_results"]["no_fabricated_measurements"])
        self.assertTrue(report["control_results"]["no_raw_or_path_fields"])

    def test_candidate_metadata_is_preserved_without_measurements(self):
        protocol = protocol_copy()
        report = runner.run_protocol(protocol)
        expected = {
            item["candidate_id"]: item
            for item in protocol["candidate_ladder"]
        }
        for row in report["receipts"]:
            candidate = expected[row["candidate_id"]]
            self.assertEqual(row["representation"], candidate["representation"])
            self.assertEqual(row["parent_candidate_id"], candidate["parent_candidate_id"])
            self.assertEqual(row["paired_with"], candidate["paired_with"])

    def test_contract_authorization_is_fail_closed(self):
        protocol = protocol_copy()
        protocol["boundaries"]["deployment_authorized"] = True
        with self.assertRaises(runner.SyntheticFootprintError):
            runner.run_protocol(protocol)

    def test_contract_execution_boundary_is_fail_closed(self):
        protocol = protocol_copy()
        protocol["quantization_performed"] = True
        with self.assertRaisesRegex(runner.SyntheticFootprintError, "contract_blocked"):
            runner.run_protocol(protocol)

    def test_receipt_mutation_is_rejected_by_hash_check(self):
        protocol = protocol_copy()
        receipts = runner.build_receipts(protocol)
        receipts[0]["candidate_id"] = "tampered"
        with self.assertRaisesRegex(runner.SyntheticFootprintError, "hash mismatch"):
            runner.validate_receipts(protocol, receipts)

    def test_receipt_measurement_fabrication_is_rejected(self):
        protocol = protocol_copy()
        receipts = runner.build_receipts(protocol)
        receipts[0]["warm_p95_ms"] = 1.0
        runner._with_hash(receipts[0])
        with self.assertRaisesRegex(runner.SyntheticFootprintError, "fabricated measurement"):
            runner.validate_receipts(protocol, receipts)

    def test_receipt_model_or_network_activity_is_rejected(self):
        protocol = protocol_copy()
        receipts = runner.build_receipts(protocol)
        receipts[0]["network_model_calls"] = 1
        runner._with_hash(receipts[0])
        with self.assertRaisesRegex(runner.SyntheticFootprintError, "model/network activity"):
            runner.validate_receipts(protocol, receipts)

    def test_cli_writes_report_exclusively_and_matches_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "n4.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(
                    runner.main(["--protocol", str(PROTOCOL_PATH), "--output", str(output)]),
                    0,
                )
            saved = json.loads(output.read_text(encoding="utf-8"))
            printed = json.loads(stdout.getvalue())
            self.assertEqual(saved, printed)
            self.assertEqual(saved["status"], runner.REPORT_STATUS)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    runner.main(["--protocol", str(PROTOCOL_PATH), "--output", str(output)]),
                    2,
                )

    def test_cli_invalid_protocol_returns_two_without_execution(self):
        protocol = protocol_copy()
        protocol["boundaries"]["training_authorized"] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blocked.json"
            path.write_text(json.dumps(protocol), encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(runner.main(["--protocol", str(path)]), 2)
            blocked = json.loads(stdout.getvalue())
            self.assertEqual(blocked["status"], "synthetic_footprint_controls_blocked")
            self.assertFalse(blocked["measurement_evidence"])
            self.assertIs(blocked["training_authorized"], False)
            self.assertEqual(blocked["network_model_calls"], 0)

    def test_cli_unreadable_protocol_returns_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(runner.main(["--protocol", str(missing)]), 2)
            blocked = json.loads(stdout.getvalue())
            self.assertEqual(blocked["status"], "synthetic_footprint_controls_blocked")

    def test_build_receipts_does_not_mutate_protocol(self):
        protocol = protocol_copy()
        before = copy.deepcopy(protocol)
        runner.build_receipts(protocol)
        self.assertEqual(protocol, before)


if __name__ == "__main__":
    unittest.main()
