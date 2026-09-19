#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest

from compare_nanojev_v2 import compare


def report(seed, probability, gold_index=0, sample_id="row:q"):
    return {
        "schema_version": "nanojev-v2-benchmark-v1",
        "run": {"seed_label": seed},
        "samples": [{
            "sample_id": sample_id,
            "row_id": "row",
            "qid": "q",
            "split": "test",
            "source_group_id": "group",
            "family_id": "family",
            "gold_index": gold_index,
            "probabilities": {"false": probability, "true": 1 - probability},
        }],
    }


class CompareNanoJevV2Test(unittest.TestCase):
    def write(self, root, name, value):
        path = Path(root) / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_exact_paired_delta_and_single_seed_null_ci(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = self.write(directory, "baseline.json", report("17", 0.2))
            candidate = self.write(directory, "candidate.json", report("17", 0.8))
            result = compare([baseline], [candidate], bootstrap_samples=50, bootstrap_seed=3)
        delta = result["splits"]["test"]["candidate_minus_baseline"]
        self.assertEqual(delta["by_seed"]["17"]["accuracy"], 1.0)
        self.assertIsNone(delta["metrics"]["accuracy"]["multi_seed_ci95"])
        self.assertEqual(result["splits"]["test"]["paired_source_group_ci95"]["clusters"], 1)

    def test_duplicate_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = self.write(directory, "baseline.json", report("17", 0.2))
            candidate = self.write(directory, "candidate.json", report("17", 0.8))
            with self.assertRaisesRegex(ValueError, "duplicate seed label"):
                compare([baseline, baseline], [candidate], bootstrap_samples=10)

    def test_cohort_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = self.write(directory, "baseline.json", report("17", 0.2))
            candidate = self.write(directory, "candidate.json", report("17", 0.8, gold_index=1))
            with self.assertRaisesRegex(ValueError, "cohort identity differs"):
                compare([baseline], [candidate], bootstrap_samples=10)

    def test_multi_seed_ci_is_present_for_two_matched_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = [self.write(directory, f"b{seed}.json", report(seed, 0.2)) for seed in ("17", "29")]
            candidate = [self.write(directory, f"c{seed}.json", report(seed, 0.8)) for seed in ("17", "29")]
            result = compare(baseline, candidate, bootstrap_samples=20, bootstrap_seed=9)
        interval = result["splits"]["test"]["candidate_minus_baseline"]["metrics"]["accuracy"]["multi_seed_ci95"]
        self.assertIsNotNone(interval)
        self.assertEqual(interval["low"], 1.0)
        self.assertEqual(interval["high"], 1.0)


if __name__ == "__main__":
    unittest.main()
