#!/usr/bin/env python3
import unittest

from benchmark_nanojev_v2 import percentile, probability_metrics


class BenchmarkHelpersTest(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(percentile([1, 3, 5, 7], 0.5), 4.0)

    def test_probability_metrics_reports_accuracy_and_calibration(self):
        result = probability_metrics([
            {"probabilities": {"false": 0.1, "true": 0.9}, "gold_index": 1},
            {"probabilities": {"false": 0.8, "true": 0.2}, "gold_index": 0},
        ])
        self.assertEqual(result["questions"], 2)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["invalid_outputs"], 0)

    def test_probability_metrics_counts_invalid_outputs(self):
        result = probability_metrics([
            {"probabilities": {"false": 0.5, "true": 0.6}, "gold_index": 1},
        ])
        self.assertEqual(result["invalid_outputs"], 1)
        self.assertIsNone(result["accuracy"])


if __name__ == "__main__":
    unittest.main()
