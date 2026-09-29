"""Offline tests for the T8 real-local parity probe's verification logic."""
import copy
from pathlib import Path
import tempfile
import unittest

import probe_context_batches_v1 as probe
from context_gate_v1 import parse_segments


def receipt(value=0.75, suggestion="retain"):
    return {"segments": [{"pointer": "/p", "p_irrelevant": value, "suggestion": suggestion}]}


class BatchProbeTests(unittest.TestCase):
    def test_all_fixed_fixtures_match_gate_contract_and_preserve_user(self):
        for wire, count in probe.PROTOCOL["cases"]:
            raw, sidecar, pointers = probe.fixture(wire, count)
            segments = parse_segments(raw, wire)
            self.assertEqual(len(segments), count + 1)
            self.assertEqual(list(sidecar["segments"]), pointers)
            self.assertTrue(segments[-1].protected)
            self.assertNotIn(segments[-1].pointer, pointers)

    def test_exact_scores_and_plans_pass(self):
        result = probe.compare(receipt(), receipt(), ["/p"])
        self.assertTrue(result["passed"])
        self.assertEqual(result["max_abs_probability_delta"], 0)
        self.assertEqual(result["plan_sha256_32"], result["plan_sha256_16"])

    def test_missing_scores_fail_even_when_plans_agree(self):
        for a, b in ((None, None), (None, .5), (.5, None)):
            result = probe.compare(receipt(a), receipt(b), ["/p"])
            self.assertFalse(result["passed"])
            self.assertFalse(result["all_candidates_scored"])
            self.assertIsNone(result["max_abs_probability_delta"])

    def test_equal_plans_cannot_hide_numeric_difference(self):
        result = probe.compare(receipt(.75), receipt(.76), ["/p"])
        self.assertFalse(result["passed"])
        self.assertTrue(result["plans_identical"])

    def test_close_scores_cannot_hide_different_plans(self):
        result = probe.compare(receipt(.99, "drop"), receipt(.99 - 1e-8), ["/p"])
        self.assertFalse(result["passed"])
        self.assertFalse(result["plans_identical"])

    def test_receipts_are_exclusive_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            probe.write_new(path, {"v": 1})
            digest = probe.sha256_file(path)
            with self.assertRaises(FileExistsError):
                probe.write_new(path, {"v": 2})
            self.assertEqual(probe.sha256_file(path), digest)


if __name__ == "__main__":
    unittest.main()
