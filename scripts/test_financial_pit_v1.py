from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from financial_pit_v1 import model_input, partition, validate_dataset, validate_record, walk_forward


def row(time=10, horizon=5, available=None):
    return {"schema_version": "nanojev-financial-pit-v1", "id": f"r{time}", "asset_id": "synthetic-asset",
            "venue": "synthetic-venue", "decision_ns": time, "universe_available_ns": 0,
            "features": {"return": {"value": 0.01, "event_ns": time-2, "available_ns": time-1,
                                    "fit_cutoff_ns": 0, "source_id": "synthetic-only", "version": "v1"}},
            "label": {"event": "forward-positive", "definition_sha256": "a"*64, "end_ns": time+horizon,
                      "available_ns": time+horizon if available is None else available, "outcome": True}}


FOLD = {"train": [0, 100], "dev": [100, 200], "calibration": [200, 300], "test": [300, 400]}


class FinancialPITTest(unittest.TestCase):
    def test_cli_audit_success_and_empty_phase_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, protocol, output = root / "rows.jsonl", root / "protocol.json", root / "audit.json"
            source.write_text("".join(json.dumps(row(t)) + "\n" for t in (10, 110, 210, 310)))
            protocol.write_text(json.dumps({"folds": [FOLD], "embargo_ns": 2, "asof_ns": 500}))
            command = [sys.executable, str(Path(__file__).with_name("financial_pit_v1.py")),
                       "--input", str(source), "--protocol", str(protocol), "--output", str(output)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(output.read_text())["folds"][0]["all_phases_nonempty"])
            source.write_text(json.dumps(row()) + "\n")
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not training-ready", result.stderr)
            self.assertFalse(json.loads(output.read_text())["folds"][0]["all_phases_nonempty"])

    def test_future_feature_availability_event_fit_and_universe_rejected(self):
        for field in ("event_ns", "available_ns", "fit_cutoff_ns"):
            sample = row()
            sample["features"]["return"][field] = 11
            with self.assertRaisesRegex(ValueError, "future"):
                validate_record(sample)
        sample = row(); sample["universe_available_ns"] = 11
        with self.assertRaisesRegex(ValueError, "universe"):
            validate_record(sample)

    def test_nonfinite_boolean_timestamp_and_invalid_label_rejected(self):
        for bad in (float("nan"), float("inf"), "0.1", None):
            sample = row(); sample["features"]["return"]["value"] = bad
            with self.assertRaises(ValueError):
                validate_record(sample)
        for bad in (True, 10.0, -1):
            sample = row(); sample["decision_ns"] = bad
            with self.assertRaises(ValueError):
                validate_record(sample)
        sample = row(); sample["label"]["outcome"] = 1
        with self.assertRaisesRegex(ValueError, "binary"):
            validate_record(sample)

    def test_model_inputs_cannot_contain_future_outcomes(self):
        first = row(); changed = deepcopy(first)
        changed["label"].update(outcome=False, end_ns=10000, available_ns=20000)
        changed["id"] = "different-private-record-id"
        self.assertEqual(model_input(first), model_input(changed))
        self.assertNotIn("label", model_input(first))
        self.assertNotIn("id", model_input(first))

    def test_duplicate_id_snapshot_feature_schema_and_event_drift_rejected(self):
        a, b = row(10), row(20)
        cases = []
        duplicate = deepcopy(a); duplicate["id"] = "different"
        cases.append([a, duplicate])
        cases.append([a, a])
        other = deepcopy(b); other["features"]["other"] = other["features"].pop("return")
        cases.append([a, other])
        other = deepcopy(b); other["label"]["definition_sha256"] = "b"*64
        cases.append([a, other])
        for records in cases:
            with self.assertRaises(ValueError):
                validate_dataset(records)

    def test_purge_embargo_fit_availability_and_test_maturity(self):
        rows = [row(t) for t in (10, 110, 210, 310)]
        rows += [row(94), row(90, horizon=10), row(80, available=101), row(390, available=501)]
        before = deepcopy(rows)
        report = partition(rows, FOLD, embargo_ns=2, asof_ns=500)
        self.assertEqual(report["counts"], dict.fromkeys(FOLD, 1))
        self.assertEqual(report["exclusion_counts"], {"purged_overlap_or_embargo": 2,
            "label_unavailable_at_fit_cutoff": 1, "unmatured_test_label": 1})
        self.assertTrue(report["all_phases_nonempty"])
        self.assertEqual(rows, before)

    def test_exact_boundary_and_empty_phase_are_explicit(self):
        report = partition([row(90, 10), row(100)], FOLD, 0, 500)
        self.assertEqual(report["retained"]["train"], [])
        self.assertEqual(report["retained"]["dev"], ["r100"])
        self.assertFalse(report["all_phases_nonempty"])

    def test_overlapping_windows_and_test_reuse_rejected(self):
        bad = deepcopy(FOLD); bad["dev"][0] = 90
        with self.assertRaisesRegex(ValueError, "chronological"):
            partition([row()], bad, 0, 500)
        with self.assertRaisesRegex(ValueError, "nonoverlapping"):
            walk_forward([row()], [FOLD, FOLD], 0, 500)
        with self.assertRaisesRegex(ValueError, "as-of"):
            partition([row()], FOLD, 0, 399)

    def test_expanding_training_can_use_matured_previous_periods(self):
        second = {"train": [0, 200], "dev": [200, 300], "calibration": [300, 400], "test": [400, 500]}
        rows = [row(t) for t in range(10, 500, 10)]
        reports = walk_forward(rows, [FOLD, second], 2, 550)
        self.assertEqual(len(reports), 2)
        self.assertTrue(all(report["all_phases_nonempty"] for report in reports))
        self.assertTrue(set(reports[0]["retained"]["train"]) < set(reports[1]["retained"]["train"]))


if __name__ == "__main__":
    unittest.main()
