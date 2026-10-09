import unittest

import run_offline_benchmark_bundle_v1 as r


class OfflineBundleRunnerTests(unittest.TestCase):
    def test_semif_payload_maps_options_to_choice(self):
        row = {"id": "x", "state": "state", "question": "which?",
               "options": [{"id": "a", "description": "A"},
                           {"id": "b", "description": "B"}]}
        payload = r.semif_payload(row)
        q = payload["states"][0]["questions"]["decision"]
        self.assertEqual(q["type"], "choice")
        self.assertEqual(q["criteria"], {"a": "A", "b": "B"})

    def test_every_retrieval_uses_yes_label_not_index(self):
        records = [
            {"file": "x/every_rows/inference204.jsonl", "family": "f",
             "question_id": "q", "group_id": "right",
             "probabilities": {"yes": 0.8, "no": 0.2}},
            {"file": "x/every_rows/inference204.jsonl", "family": "f",
             "question_id": "q", "group_id": "wrong",
             "probabilities": {"yes": 0.6, "no": 0.4}},
            {"file": "x/every_rows/gold154.jsonl", "family": "f",
             "question_id": "q", "group_id": "right", "gold_label": "yes"},
            {"file": "x/every_rows/gold154.jsonl", "family": "f",
             "question_id": "q", "group_id": "wrong", "gold_label": "no"},
        ]
        metrics = r.retrieval_metrics(records)
        self.assertEqual(metrics["queries"], 1)
        self.assertEqual(metrics["recall_at_1"], 1.0)
        self.assertEqual(metrics["recall_at_3"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)

    def test_perturbation_compares_option_ids(self):
        records = [
            {"id": "base", "file": "x/authored144.jsonl",
             "predicted_label": "a",
             "probabilities": {"a": 0.8, "b": 0.2}},
            {"id": "pert", "file": "x/perturbations108.jsonl",
             "predicted_label": "b",
             "probabilities": {"b": 0.9, "a": 0.1},
             "provenance": {"base_id": "base"}},
        ]
        metrics = r.perturbation_metrics(records)
        self.assertEqual(metrics["pairs"], 1)
        self.assertEqual(metrics["argmax_flip_rate"], 1.0)
        self.assertAlmostEqual(metrics["mean_prob_l1"], 1.4)


if __name__ == "__main__":
    unittest.main()
