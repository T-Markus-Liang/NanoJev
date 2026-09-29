import unittest

from diagnose_t9d_v5_shift import bucket_cardinality, cardinality_profile, cohort_rows, grouped, metric, receipt_spec


class ShiftDiagnosisTests(unittest.TestCase):
    def test_cardinality_buckets(self):
        self.assertEqual(bucket_cardinality(4), "4")
        self.assertEqual(bucket_cardinality(32), "13-32")
        self.assertEqual(bucket_cardinality(255), "33-255")

    def test_metric_counts_protected_errors(self):
        rows = [
            {"correct": True, "confidence": .95},
            {"correct": False, "confidence": .91},
            {"correct": False, "confidence": .4},
        ]
        value = metric(rows)
        self.assertAlmostEqual(value["accuracy"], 1 / 3)
        self.assertAlmostEqual(value["coverage_at_0.9"], 2 / 3)
        self.assertEqual(value["protected_errors_at_0.9"], 1)

    def test_grouped(self):
        rows = [
            {"type": "choice", "correct": True, "confidence": .6},
            {"type": "boolean", "correct": False, "confidence": .7},
        ]
        value = grouped(rows, "type")
        self.assertEqual(value["choice"]["accuracy"], 1)
        self.assertEqual(value["boolean"]["accuracy"], 0)

    def test_cohort_rows_joins_by_id(self):
        items = [{
            "id": "x", "family_id": "f", "gold": {"q": "b"},
            "questions": {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}},
        }]
        answers = {17: {"x": {"probabilities": {"a": .2, "b": .8}}}}
        rows = cohort_rows(items, answers)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["correct"])
        self.assertEqual(rows[0]["cardinality"], 2)

    def test_shape_profile(self):
        rows = [{
            "questions": {"q": {"type": "score", "criteria": ["a", "b", "c"]}}
        }]
        self.assertEqual(cardinality_profile(rows), {"score:3": 1})

    def test_receipt_spec(self):
        seed, path = receipt_spec("18=results/custom.json")
        self.assertEqual((seed, path.name), (18, "custom.json"))
        seed, path = receipt_spec("results/heldout_seed19.json")
        self.assertEqual((seed, path.name), (19, "heldout_seed19.json"))


if __name__ == "__main__":
    unittest.main()
