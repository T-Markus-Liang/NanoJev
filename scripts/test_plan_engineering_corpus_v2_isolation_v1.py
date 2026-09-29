#!/usr/bin/env python3
"""Tests for the read-only V2 isolation-plan builder."""
from __future__ import annotations

import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import plan_engineering_corpus_v2_isolation_v1 as planner


ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "research" / "engineering_judgment_corpus_v1"


def row(record_id, split, pair_id, member, state, question_text, source="fresh-source"):
    return {
        "id": record_id,
        "state_id": state,
        "family_id": "fixture",
        "split": split,
        "state": state,
        "questions": {
            "choice": {
                "type": "choice",
                "instructions": question_text,
                "criteria": {"a": "A", "b": "B"},
            }
        },
        "metadata": {
            "pair_id": pair_id,
            "member": member,
            "source_group_id": f"source-{pair_id}",
            "provenance_source_id": source,
        },
    }


class IsolationPlanTests(unittest.TestCase):
    def test_lineage_and_same_input_form_one_component(self):
        rows = [
            row("a", "train", "pair-a", "base", "s-a", "same"),
            row("b", "train", "pair-a", "variant", "s-a", "different"),
            row("c", "test", "pair-c", "base", "s-a", "same"),
        ]
        report = planner.component_plan(rows)
        self.assertEqual(report["component_count"], 1)
        shared = [c for c in report["components"] if "pair-a" in c["pair_ids"]][0]
        self.assertIn("pair-c", shared["pair_ids"])
        self.assertTrue(shared["split_conflict"])

    def test_split_conflict_is_blocking_and_no_raw_input_is_emitted(self):
        report = planner.component_plan([
            row("a", "train", "pair-a", "base", "s-a", "same"),
            row("b", "test", "pair-b", "base", "s-a", "same"),
        ])
        self.assertEqual(report["canonical_input_cross_split_groups"], 1)
        self.assertTrue(report["components"][0]["split_conflict"])
        self.assertIn("component_crosses_splits", {
            entry["code"] for entry in report["violations"]
        })
        serialized = json.dumps(report)
        self.assertNotIn('"state":', serialized)
        self.assertNotIn('"questions":', serialized)

    def test_evaluation_provenance_is_quarantined(self):
        report = planner.component_plan([
            row("a", "train", "pair-a", "base", "s-a", "fresh", "abstention-survey-v1"),
        ])
        component = report["components"][0]
        self.assertEqual(component["safe_action"], "quarantine_until_review")
        self.assertEqual(report["evaluation_provenance_record_count"], 1)
        self.assertIn("evaluation_derived_provenance", {
            entry["code"] for entry in report["violations"]
        })

    def test_pair_id_cross_split_is_explicitly_blocked(self):
        first = row("a", "train", "pair-shared", "base", "s-a", "first")
        second = row("b", "test", "pair-shared", "variant", "s-b", "second")
        first["metadata"]["source_group_id"] = "source-train"
        second["metadata"]["source_group_id"] = "source-test"
        report = planner.component_plan([first, second])
        codes = {entry["code"] for entry in report["violations"]}
        self.assertIn("pair_id_crosses_splits", codes)
        self.assertIn("component_crosses_splits", codes)

    def test_source_group_id_cross_split_is_explicitly_blocked(self):
        first = row("a", "train", "pair-a", "base", "s-a", "first")
        second = row("b", "test", "pair-b", "base", "s-b", "second")
        first["metadata"]["source_group_id"] = "source-shared"
        second["metadata"]["source_group_id"] = "source-shared"
        report = planner.component_plan([first, second])
        codes = {entry["code"] for entry in report["violations"]}
        self.assertIn("source_group_id_crosses_splits", codes)

    def test_missing_lineage_metadata_is_blocked(self):
        value = row("a", "train", "pair-a", "base", "s-a", "fresh")
        value["metadata"].pop("pair_id")
        report = planner.component_plan([value])
        self.assertIn("missing_pair_id", {entry["code"] for entry in report["violations"]})

    def test_current_corpus_has_expected_negative_plan(self):
        report = planner.plan_corpus(CORPUS, compare_seed=20260920)
        self.assertEqual(report["status"], "blocked_isolation_plan_only")
        self.assertEqual(report["corpus_identity"], "engineering-judgment-catalog-v1@seed-20260919")
        self.assertNotIn("corpus", report)
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["measurement_authorized"])
        self.assertFalse(report["measurement_performed"])
        self.assertFalse(report["deployment_authorized"])
        self.assertFalse(report["deployment_performed"])
        self.assertEqual(report["merged_rows_written"], 0)
        self.assertEqual(report["record_count"], 246)
        self.assertEqual(report["component_count"], 28)
        self.assertEqual(report["conflicted_component_count"], 6)
        self.assertEqual(report["canonical_input_cross_split_groups"], 27)
        self.assertEqual(report["evaluation_provenance_record_count"], 18)
        self.assertEqual(report["cross_seed"]["common_canonical_inputs"], 129)
        self.assertFalse(report["input_files_changed"])

    def test_cross_seed_comparison_is_required_for_a_plan(self):
        report = planner.plan_corpus(CORPUS)
        self.assertIn("cross_seed_comparison_required", {
            entry["code"] for entry in report["violations"]
        })
        self.assertEqual(report["status"], "blocked_isolation_plan_only")

    def test_cli_report_is_exclusive_and_input_stays_unchanged(self):
        before = {
            path: path.read_bytes()
            for path in planner._source_paths(CORPUS)
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "plan.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(
                    planner.main([
                        "--corpus", str(CORPUS),
                        "--compare-seed", "20260920",
                        "--output", str(output),
                    ]),
                    2,
                )
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "blocked_isolation_plan_only")
            for key in (
                "training_authorized", "training_performed",
                "measurement_authorized", "measurement_performed",
                "deployment_authorized", "deployment_performed",
            ):
                self.assertIs(saved[key], False)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    planner.main([
                        "--corpus", str(CORPUS),
                        "--compare-seed", "20260920",
                        "--output", str(output),
                    ]),
                    2,
                )
        for path, data in before.items():
            self.assertEqual(path.read_bytes(), data)

    def test_plan_is_not_mutable_by_component_copy(self):
        rows = [row("a", "train", "pair-a", "base", "s-a", "same")]
        original = copy.deepcopy(rows)
        planner.component_plan(rows)
        self.assertEqual(rows, original)


if __name__ == "__main__":
    unittest.main()
