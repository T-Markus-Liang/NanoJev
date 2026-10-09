#!/usr/bin/env python3
"""J-C selective-risk layer tests: curve monotonicity, protected-error counting,
operating-point fit rule, fit/eval isolation hash check, fail-closed loading,
and the routing-spec contract. Pure python — no checkpoint or torch needed."""
import json
import tempfile
import unittest
from pathlib import Path

from selective_risk_v1 import (
    PROTECTED_GATE, SPEC_SCHEMA, TARGET_ACCURACIES, answers_map,
    assert_fit_eval_isolation, build_layer_spec, build_records,
    coverage_risk_curve, curve_point, fit_operating_point,
    fit_operating_points, load_cohort, measure_transfer,
    protected_errors, records_hash, threshold_grid)


def rec(row_id, qid, conf, correct, qtype="choice", state_id=None):
    """Synthetic scored record in the build_records output shape."""
    if correct:
        probs = {"a": conf, "b": 1.0 - conf}
        gold = {"a": 1.0, "b": 0.0}
    else:
        probs = {"a": conf, "b": 1.0 - conf}
        gold = {"a": 0.0, "b": 1.0}
    return {"cohort": "t", "split": "t", "row_id": row_id, "qid": qid,
            "state_id": state_id or row_id, "source_group_id": "g",
            "family_id": "f", "question_type": qtype,
            "group": f"{qtype}:2", "probs": probs, "gold_probs": gold,
            "pred_argmax": max(probs, key=probs.get),
            "gold_argmax": max(gold, key=gold.get),
            "confidence": conf, "correct": correct}


class GridTest(unittest.TestCase):
    def test_grid_endpoints(self):
        grid = threshold_grid(101)
        self.assertEqual(grid[0], 0.0)
        self.assertEqual(grid[-1], 1.0)
        self.assertEqual(len(grid), 101)

    def test_grid_rejects_too_few_points(self):
        with self.assertRaises(ValueError):
            threshold_grid(1)


class CurveTest(unittest.TestCase):
    def test_coverage_nonincreasing(self):
        records = [rec(f"r{i}", "q", 0.05 * i, i % 3 == 0) for i in range(1, 20)]
        points = coverage_risk_curve(records, threshold_grid(51))
        covs = [p["coverage"] for p in points]
        self.assertEqual(covs, sorted(covs, reverse=True))
        ns = [p["n_answered"] for p in points]
        self.assertEqual(ns, sorted(ns, reverse=True))

    def test_unsorted_grid_rejected(self):
        with self.assertRaises(ValueError):
            coverage_risk_curve([rec("r", "q", 0.5, True)], [0.9, 0.1])

    def test_curve_point_counts(self):
        records = [rec("a", "q", 0.95, True), rec("b", "q", 0.92, False),
                   rec("c", "q", 0.4, True), rec("d", "q", 0.3, False)]
        p = curve_point(records, 0.9)
        self.assertEqual(p["n_total"], 4)
        self.assertEqual(p["n_answered"], 2)
        self.assertAlmostEqual(p["coverage"], 0.5)
        self.assertAlmostEqual(p["selective_accuracy"], 0.5)
        self.assertAlmostEqual(p["selective_risk"], 0.5)
        self.assertEqual(p["confident_errors"], 1)
        self.assertIsNotNone(p["ece_fixed_10"])

    def test_empty_answered_subset(self):
        records = [rec("a", "q", 0.1, True)]
        p = curve_point(records, 0.9)
        self.assertEqual(p["n_answered"], 0)
        self.assertIsNone(p["selective_accuracy"])
        self.assertIsNone(p["nll"])
        self.assertEqual(p["coverage"], 0.0)


class ProtectedErrorTest(unittest.TestCase):
    def test_counts_only_wrong_above_gate(self):
        records = [rec("a", "q", 0.95, False),   # protected error
                   rec("b", "q", 0.91, False),   # protected error
                   rec("c", "q", 0.89, False),   # below gate: not counted
                   rec("d", "q", 0.99, True),    # correct: not counted
                   rec("e", "q", 0.5, False)]
        out = protected_errors(records)
        self.assertEqual(out["gate"], PROTECTED_GATE)
        self.assertEqual(out["count"], 2)
        self.assertEqual(out["by_type"]["choice"], 2)
        self.assertEqual(out["by_type"]["boolean"], 0)

    def test_gate_boundary_inclusive(self):
        records = [rec("a", "q", PROTECTED_GATE, False)]
        self.assertEqual(protected_errors(records)["count"], 1)


class OperatingPointTest(unittest.TestCase):
    def _records(self):
        # conf inversely related to correctness: high conf -> correct here.
        return ([rec(f"h{i}", "q", 0.95, True, state_id=f"s{i}") for i in range(40)]
                + [rec(f"m{i}", "q", 0.85, i < 30, state_id=f"sm{i}")
                   for i in range(40)]
                + [rec(f"l{i}", "q", 0.5, i < 10, state_id=f"sl{i}")
                   for i in range(40)])

    def test_fit_picks_max_coverage_feasible(self):
        records = self._records()
        # at t=0: acc = 80/120 = 0.667; at t>=0.85: 70/80 = 0.875; at t>=0.95: 1.0
        p = fit_operating_point(records, 0.80)
        self.assertEqual(p["status"], "estimated")
        self.assertAlmostEqual(p["threshold"], 0.85)
        self.assertAlmostEqual(p["coverage"], 80 / 120)
        self.assertAlmostEqual(p["selective_accuracy"], 70 / 80)

    def test_unreachable_target_is_honest(self):
        records = [rec(f"r{i}", "q", 0.6, False, state_id=f"s{i}")
                   for i in range(40)]
        p = fit_operating_point(records, 0.95)
        self.assertEqual(p["status"], "unreachable_on_this_grid")
        self.assertIsNone(p["threshold"])
        self.assertEqual(p["best_achievable"]["selective_accuracy"], 0.0)

    def test_full_coverage_when_baseline_meets_target(self):
        records = [rec(f"r{i}", "q", 0.6, True, state_id=f"s{i}")
                   for i in range(40)]
        p = fit_operating_point(records, 0.9)
        self.assertEqual(p["status"], "estimated")
        # all conf = 0.6 -> coverage 1.0 for every t <= 0.6; tightest gate wins
        self.assertEqual(p["threshold"], 0.6)
        self.assertEqual(p["coverage"], 1.0)

    def test_minima_gate(self):
        few = [rec(f"r{i}", "q", 0.9, True, state_id=f"s{i}") for i in range(4)]
        fitted = fit_operating_points(few, (0.8,))
        self.assertEqual(fitted["choice"]["status"], "unestimated_below_minimum")
        self.assertEqual(fitted["choice"]["points"], {})


class TransferTest(unittest.TestCase):
    def test_measured_gap_reported(self):
        fit_records = [rec(f"f{i}", "q", 0.95, i < 38, state_id=f"fs{i}")
                       for i in range(40)]
        eval_records = [rec(f"e{i}", "q", 0.95, i < 20, state_id=f"es{i}")
                        for i in range(40)]
        fitted = fit_operating_points(fit_records, (0.9,))
        # fit: acc 38/40 = 0.95 at every t <= 0.95 -> tightest gate t=0.95
        table = measure_transfer(fitted, eval_records, (0.9,))
        cell = table["choice"]["targets"]["0.90"]
        self.assertEqual(cell["fitted_threshold"], 0.95)
        self.assertAlmostEqual(cell["measured_selective_accuracy"], 0.5)
        self.assertAlmostEqual(cell["transfer_gap_accuracy"], 0.5 - 0.9)
        self.assertFalse(cell["target_met_on_eval"])


class IsolationTest(unittest.TestCase):
    def test_shared_question_identity_fails(self):
        fit = [rec("r1", "q", 0.9, True)]
        ev = [rec("r1", "q", 0.8, True, state_id="other")]
        with self.assertRaisesRegex(ValueError, "isolation"):
            assert_fit_eval_isolation(fit, ev, "calibration", "heldout")

    def test_shared_state_fails(self):
        fit = [rec("r1", "q1", 0.9, True, state_id="shared")]
        ev = [rec("r2", "q2", 0.8, True, state_id="shared")]
        with self.assertRaisesRegex(ValueError, "state_id"):
            assert_fit_eval_isolation(fit, ev, "calibration", "heldout")

    def test_disjoint_passes_with_hashes(self):
        fit = [rec("r1", "q", 0.9, True)]
        ev = [rec("r2", "q", 0.8, True)]
        out = assert_fit_eval_isolation(fit, ev, "calibration", "heldout")
        self.assertEqual(out["shared_question_identities"], 0)
        self.assertNotEqual(out["fit_records_sha256"], out["eval_records_sha256"])
        self.assertEqual(out["fit_records_sha256"], records_hash(fit))


class FailClosedLoadTest(unittest.TestCase):
    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_cohort("/nonexistent/nope.jsonl")

    def test_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "empty.jsonl"
            p.write_text("\n")
            with self.assertRaises(ValueError):
                load_cohort(p)

    def test_wrong_split_label_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rows.jsonl"
            row = {"id": "x", "state": "s", "questions": {}, "gold_probs": {},
                   "split": "test"}
            p.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "split"):
                load_cohort(p, expected_split="calibration")

    def test_malformed_row_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rows.jsonl"
            p.write_text(json.dumps({"id": "x"}) + "\n")
            with self.assertRaises(ValueError):
                load_cohort(p)


class BuildRecordsTest(unittest.TestCase):
    def test_missing_prediction_fails(self):
        rows = [{"id": "r", "state": "s", "state_id": "st",
                 "questions": {"q": {"type": "boolean", "instructions": "i"}},
                 "gold_probs": {"q": {"false": 1.0, "true": 0.0}}}]
        with self.assertRaises(KeyError):
            build_records(rows, {}, "c", "fit")

    def test_confidence_and_correctness(self):
        rows = [{"id": "r", "state": "s", "state_id": "st",
                 "questions": {"q": {"type": "boolean", "instructions": "i"}},
                 "gold_probs": {"q": {"false": 0.0, "true": 1.0}}}]
        payload = {"states": [{"id": "r", "answers": {
            "q": {"probabilities": {"false": 0.3, "true": 0.7}}}}]}
        recs = build_records(rows, answers_map(payload), "c", "fit")
        self.assertEqual(len(recs), 1)
        self.assertAlmostEqual(recs[0]["confidence"], 0.7)
        self.assertTrue(recs[0]["correct"])
        self.assertEqual(recs[0]["group"], "boolean:2")


class SpecContractTest(unittest.TestCase):
    def test_spec_shape_and_non_goals(self):
        fitted = fit_operating_points(
            [rec(f"r{i}", "q", 0.95, True, qtype="boolean", state_id=f"s{i}")
             for i in range(40)], (0.9,))
        spec = build_layer_spec("arm", "ckpt", "calibration",
                                __file__, fitted, {"fit_rows": 40})
        self.assertEqual(spec["schema_version"], SPEC_SCHEMA)
        self.assertFalse(spec["authorizes_execution"])
        self.assertFalse(spec["deployment_authorized"])
        self.assertEqual(spec["network_model_calls"], 0)
        self.assertEqual(spec["threshold_source"]["fit_role"],
                         "calibration_fit_only")
        self.assertTrue(spec["abstention_semantics"]["never_silently_answered"])
        self.assertEqual(spec["ood_detector"]["status"],
                         "placeholder_interface_not_implemented")
        self.assertGreaterEqual(len(spec["non_goals"]), 3)
        self.assertEqual(spec["protected_error_policy"]["limit"], 0)
        op = spec["operating_points"]["boolean"]["targets"]["0.90"]
        self.assertIn("abstain_below", op)


if __name__ == "__main__":
    unittest.main()
