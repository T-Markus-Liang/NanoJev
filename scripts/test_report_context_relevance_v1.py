from copy import deepcopy
import random
import unittest

from build_context_relevance_v1 import make_record
from report_context_relevance_v1 import gate_metrics, join_predictions, unique_index


class RelevanceReportTest(unittest.TestCase):
    def fixture(self):
        row = make_record("code", (100, 200, 300, 50), "required_field", "test", "openai_chat", random.Random(17))
        pred = {"id": row["id"] + ":irrelevant", "state_id": row["id"], "family_id": row["family_id"],
                "split": "test", "qid": "irrelevant", "type": "boolean", "candidate_ids": ["false", "true"],
                "gold_index": 0, "student_probs": [0.01, 0.99]}
        return row, pred

    def test_join_preserves_source_identity_and_counts_critical_false_drop(self):
        row, pred = self.fixture()
        samples = join_predictions([row], [pred])
        self.assertEqual(samples[0]["source_group_id"], row["metadata"]["source_group_id"])
        gate = gate_metrics(samples)
        self.assertEqual(gate["false_drops"], 1)
        self.assertEqual(gate["false_drop_groups"], 1)
        self.assertEqual(gate["actual_main_model_tokens_saved"], 0)
        self.assertIsNone(gate["zero_error_group_upper95"])

    def test_missing_duplicate_or_misaligned_prediction_rejected(self):
        row, pred = self.fixture()
        for predictions in ([], [pred, deepcopy(pred)]):
            with self.assertRaises(ValueError):
                join_predictions([row], predictions)
        for key, value in (("gold_index", 1), ("split", "ood"), ("candidate_ids", ["true", "false"]),
                           ("state_id", "wrong"), ("family_id", "wrong")):
            changed = dict(pred, **{key: value})
            with self.assertRaises(ValueError):
                join_predictions([row], [changed])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            unique_index([pred, pred], "id")

    def test_nonfinite_or_malformed_probabilities_rejected(self):
        row, pred = self.fixture()
        for probs in ([float("nan"), 0.1], [float("inf"), 0.0], [-0.1, 1.1], [0.5, 0.6], [1.0]):
            with self.assertRaises(ValueError):
                join_predictions([row], [dict(pred, student_probs=probs)])

    def test_zero_observed_error_not_proof_of_zero_risk(self):
        row, pred = self.fixture()
        pred["student_probs"] = [0.99, 0.01]
        sample = join_predictions([row], [pred])[0]
        gate = gate_metrics([sample, dict(sample, sample_id="second-variant")])
        self.assertEqual(gate["source_groups"], 1)
        self.assertEqual(gate["zero_error_group_upper95"], 0.95)
        self.assertIsNone(gate["false_drop_rate_among_proposals"])
        self.assertEqual(gate["retained_or_abstained_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
