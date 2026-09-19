#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluate_raw_dataset_v2 import load_rows, prediction_row, run, target_mapping


def boolean_row(split="test", suffix="1"):
    return {"id": f"r{suffix}", "state_id": f"r{suffix}", "split": split, "family_id": "family",
            "state": "Authorized=true", "questions": {"q": {"type": "boolean", "instructions": "Allowed?"}},
            "gold": {"q": True}, "gold_label_kind": "deterministic_truth",
            "metadata": {"source_group_id": f"g{suffix}"}}


class EvaluateRawDatasetV2Test(unittest.TestCase):
    def test_soft_and_hard_targets_are_preserved(self):
        row = {
            "id": "r1", "state_id": "r1", "split": "test", "family_id": "family",
            "state": "Distribution supplied independently",
            "questions": {"q": {"type": "choice", "instructions": "Choose", "criteria": {"a": "A", "b": "B"}}},
            "gold_probs": {"q": {"a": 0.25, "b": 0.75}},
            "gold_probs_kind": "programmatic_conditional_distribution",
            "gold_label_kind": "unobserved", "metadata": {"source_group_id": "g1"},
        }
        question = row["questions"]["q"]
        target, kind, label_kind = target_mapping(row, "q", question)
        output = prediction_row(row, "q", question, {"probabilities": {"a": 0.3, "b": 0.7}})
        self.assertEqual(target, {"a": 0.25, "b": 0.75})
        self.assertEqual(kind, "programmatic_conditional_distribution")
        self.assertEqual(label_kind, "unobserved")
        self.assertEqual(output["gold_probs"], target)

    def test_boolean_hard_target_uses_false_true_order(self):
        row = {
            "id": "r2", "state_id": "r2", "split": "test", "family_id": "family",
            "state": "True", "gold_label_kind": "deterministic_truth",
            "questions": {"q": {"type": "boolean", "instructions": "True?"}},
            "gold": {"q": True}, "metadata": {"source_group_id": "g2"},
        }
        target, kind, _ = target_mapping(row, "q", row["questions"]["q"])
        self.assertEqual(target, {"false": 0.0, "true": 1.0})
        self.assertEqual(kind, "deterministic_truth")

    def test_hard_observations_and_compatibility_labels_are_not_truth(self):
        for kind in (None, "observed_outcome", "unobserved", "reference_argmax_compatibility"):
            with self.subTest(kind=kind):
                row = boolean_row()
                row["gold_label_kind"] = kind
                with self.assertRaisesRegex(ValueError, "explicit deterministic_truth"):
                    target_mapping(row, "q", row["questions"]["q"])

    def test_invalid_hard_types_are_rejected(self):
        for gold in (1, "true", None):
            row = boolean_row()
            row["gold"]["q"] = gold
            with self.assertRaises(ValueError):
                target_mapping(row, "q", row["questions"]["q"])

    def test_unlabelled_and_invalid_soft_targets_rejected(self):
        row = boolean_row()
        row["gold"] = {}
        with self.assertRaisesRegex(ValueError, "independent gold target"):
            target_mapping(row, "q", row["questions"]["q"])
        row["gold_probs"] = {"q": {"false": 0.3, "true": 0.9}}
        row["gold_probs_kind"] = "programmatic_conditional_distribution"
        with self.assertRaisesRegex(ValueError, "sum"):
            target_mapping(row, "q", row["questions"]["q"])

    def test_soft_target_not_replaced_by_observed_label(self):
        row = boolean_row()
        row.update(gold_label_kind="observed_outcome", gold_probs_kind="programmatic_conditional_distribution",
                   gold_probs={"q": {"false": 0.9, "true": 0.1}})
        target, kind, label_kind = target_mapping(row, "q", row["questions"]["q"])
        self.assertEqual(target, {"false": 0.9, "true": 0.1})
        self.assertEqual(kind, "programmatic_conditional_distribution")
        self.assertEqual(label_kind, "observed_outcome")

    def test_missing_unknown_and_cross_split_source_groups_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test.jsonl").write_text(json.dumps(boolean_row()) + "\n")
            with self.assertRaises(ValueError):
                load_rows(root, {"test", "ood"})
            with self.assertRaises(ValueError):
                load_rows(root, {"typo"})
            with self.assertRaises(ValueError):
                load_rows(root / "test.jsonl", {"ood"})
            other = boolean_row("ood", "2")
            other["metadata"]["source_group_id"] = "g1"
            (root / "ood.jsonl").write_text(json.dumps(other) + "\n")
            with self.assertRaisesRegex(ValueError, "crosses dataset splits"):
                load_rows(root, {"test", "ood"})

    def test_files_are_scoped_to_evaluated_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, suffix in (("train", "0"), ("test", "1"), ("ood", "2")):
                (root / f"{split}.jsonl").write_text(json.dumps(boolean_row(split, suffix)) + "\n")
            rows, files = load_rows(root, {"test", "ood"})
            self.assertEqual({path.name for path in files}, {"test.jsonl", "ood.jsonl"})
            self.assertEqual(len(rows), 2)

    def test_runtime_counts_predictions_provenance_and_no_fake_warm_latency(self):
        row = boolean_row()
        response = {"states": [{"id": "r1", "answers": {"q": {"probabilities": {"false": 0.2, "true": 0.8}}}}],
                    "execution": {"forward_passes": 3, "network_model_calls": 0,
                                  "candidate_paths": 7, "questions": 1, "states": 1}}
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(input=Path(tmp), checkpoint=Path(tmp), splits="test", batch_states=1,
                                   batch_questions=1, device="cpu", precision="fp32", max_length=None,
                                   predictions=Path(tmp) / "predictions.jsonl")
            with patch("evaluate_raw_dataset_v2.load_rows", return_value=([row], [])), \
                    patch("evaluate_raw_dataset_v2.benchmark_manifest", return_value={"manifest_sha256": "old"}) as manifest, \
                    patch("evaluate_raw_dataset_v2.DecisionPredictor") as predictor:
                predictor.return_value.device = "cpu"
                predictor.return_value.precision = "fp32"
                predictor.return_value.limit = 512
                predictor.return_value.predict.return_value = response
                report = run(args)
                sent = predictor.return_value.predict.call_args.args[0]["states"][0]
                self.assertEqual(set(sent), {"id", "state", "questions"})
                self.assertTrue(manifest.call_args.kwargs["script_path"].endswith("evaluate_raw_dataset_v2.py"))
                self.assertEqual(report["execution"]["forward_passes"], 3)
                self.assertEqual(report["execution"]["requests"], 1)
                self.assertEqual(report["execution"]["warm_requests"], 0)
                self.assertIsNone(report["execution"]["warm_request_p99_ms"])
                self.assertEqual(report["coverage"]["test"]["source_groups"], 1)
                self.assertEqual(report["prediction_count"], 1)
                self.assertEqual(len(report["predictions_sha256"]), 64)
                self.assertEqual(len(args.predictions.read_text().splitlines()), 1)
                response["states"][0]["answers"]["unexpected"] = {}
                with self.assertRaisesRegex(ValueError, "question IDs"):
                    run(args)

    def test_empty_requested_split_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test.jsonl").write_text(json.dumps(boolean_row()) + "\n")
            (root / "ood.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "every requested split"):
                load_rows(root, {"test", "ood"})


if __name__ == "__main__":
    unittest.main()
