from copy import deepcopy
import unittest

from build_workflow_challenge_v2 import VARIANTS
from compare_workflow_variants_v2 import compare


def predictions():
    return [{"id": f"base::{name}:q", "split": "test", "family_id": "family", "type": "boolean",
             "candidate_ids": ["false", "true"], "student_probs": [0.1, 0.9],
             "gold_probs": [0, 1], "gold_probs_kind": "deterministic_truth",
             "metadata": {"challenge_variant": name, "base_record_id": "base", "source_group_id": "group"}}
            for name in VARIANTS]


class PairedWorkflowTest(unittest.TestCase):
    def test_no_drift_is_zero_and_reproducible(self):
        result = compare(predictions(), 50)
        self.assertEqual(result, compare(predictions(), 50))
        self.assertEqual(result["base_questions"], 1)
        for values in result["comparisons"].values():
            self.assertEqual(values["source_groups"], 1)
            self.assertEqual(values["top_choice_flips"], 0)
            self.assertEqual(values["deltas_variant_minus_original"]["accuracy"]["ci95"], [0, 0])

    def test_known_error_and_probability_drift(self):
        rows = predictions()
        rows[1]["student_probs"] = [0.95, 0.05]
        result = compare(rows, 50)["comparisons"]["test/family/boolean/archived_distractor"]
        self.assertEqual(result["top_choice_flips"], 1)
        self.assertAlmostEqual(result["maximum_absolute_probability_drift"], 0.85)
        self.assertEqual(result["deterministic_confidence_ge_0_9"]["error_rate"], 1)
        self.assertEqual(result["deltas_variant_minus_original"]["accuracy"]["ci95"], [-1, -1])

    def test_incomplete_duplicate_and_changed_gold_fail(self):
        with self.assertRaisesRegex(ValueError, "coverage"):
            compare(predictions()[:-1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compare(predictions() + predictions()[:1])
        rows = predictions()
        rows[1]["gold_probs"] = [1, 0]
        with self.assertRaisesRegex(ValueError, "semantics"):
            compare(rows)

    def test_source_overlap_between_splits_fails(self):
        rows = predictions()
        other = deepcopy(rows)
        for row in other:
            row["split"] = "ood"
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            compare(rows + other)

    def test_choice_mapping_reorder_is_not_drift(self):
        rows = predictions()
        for row in rows:
            row["type"] = "choice"
            row["candidate_ids"] = ["a", "b"]
        rows[-1].update(candidate_ids=["b", "a"], student_probs=[0.9, 0.1], gold_probs=[1, 0])
        result = compare(rows, 20)["comparisons"]["test/family/choice/choice_order_reversed"]
        self.assertEqual(result["maximum_absolute_probability_drift"], 0)
        self.assertEqual(result["top_choice_flips"], 0)

    def test_analytic_target_has_no_fabricated_accuracy(self):
        rows = predictions()
        for row in rows:
            row.update(gold_probs=[0.4, 0.6], gold_probs_kind="programmatic_conditional_distribution")
        result = compare(rows, 20)
        for values in result["comparisons"].values():
            self.assertNotIn("accuracy", values["deltas_variant_minus_original"])
            self.assertEqual(values["deterministic_confidence_ge_0_9"]["selected"], 0)

    def test_correlated_questions_resample_together(self):
        rows = predictions()
        other = predictions()
        for row in other:
            row["id"] = row["id"].replace("base::", "other::")
            row["metadata"]["base_record_id"] = "other"
        rows[1]["student_probs"] = [0.95, 0.05]
        result = compare(rows + other, 50)["comparisons"]["test/family/boolean/archived_distractor"]
        self.assertEqual(result["paired_questions"], 2)
        self.assertEqual(result["source_groups"], 1)
        self.assertEqual(result["deltas_variant_minus_original"]["accuracy"]["ci95"], [-0.5, -0.5])


if __name__ == "__main__":
    unittest.main()
