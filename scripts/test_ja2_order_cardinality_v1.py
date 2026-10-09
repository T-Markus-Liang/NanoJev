#!/usr/bin/env python3
"""Unit tests for J-A2 harness helpers (no model load, no MPS required)."""
import json
import unittest
from pathlib import Path

import run_ja2_order_cardinality_v1 as ja2

ROOT = Path(__file__).resolve().parents[1]
PREDICT_INPUT = ROOT / "research/engineering_heldout_v1/predict_input.json"
HELDOUT_ITEMS = ROOT / "research/engineering_heldout_v1/items.jsonl"


def load_payload():
    return json.loads(PREDICT_INPUT.read_text(encoding="utf-8"))


def load_gold():
    gold = {}
    for line in HELDOUT_ITEMS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for qid, q in row["questions"].items():
            if q["type"] == "choice":
                gold[row["id"]] = {qid: row["gold"][qid]}
    return gold


class PermuteKeysTest(unittest.TestCase):
    def test_identity_reverse_rotation(self):
        keys = ["a", "b", "c", "d"]
        self.assertEqual(ja2.permute_keys(keys, "identity", "s"), keys)
        self.assertEqual(ja2.permute_keys(keys, "reversed", "s"), ["d", "c", "b", "a"])
        self.assertEqual(ja2.permute_keys(keys, "rot1", "s"), ["b", "c", "d", "a"])
        self.assertEqual(ja2.permute_keys(keys, "rot2", "s"), ["c", "d", "a", "b"])

    def test_shuffle_is_deterministic_and_a_permutation(self):
        keys = ["a", "b", "c", "d"]
        s1 = ja2.permute_keys(keys, "shuffle_s11", "state-x")
        s2 = ja2.permute_keys(keys, "shuffle_s11", "state-x")
        self.assertEqual(s1, s2)
        self.assertEqual(sorted(s1), sorted(keys))
        self.assertNotEqual(ja2.permute_keys(keys, "shuffle_s11", "state-y"), s1 or s2)

    def test_unknown_permutation_rejected(self):
        with self.assertRaises(ValueError):
            ja2.permute_keys(["a", "b"], "mirror", "s")


class PayloadBuildersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = load_payload()
        cls.gold = load_gold()

    def test_permutation_payload_preserves_pairs(self):
        p = ja2.build_permutation_payload(self.payload, "reversed")
        for s, qid, q in ja2.choice_states(self.payload):
            new_crit = next(x["questions"][qid]["criteria"]
                            for x in p["states"] if x["id"] == s["id"])
            self.assertEqual(new_crit, q["criteria"])  # same key->text mapping
            self.assertEqual(list(new_crit), list(q["criteria"])[::-1])

    def test_permutation_payload_does_not_mutate_input(self):
        before = json.dumps(self.payload, sort_keys=True)
        ja2.build_permutation_payload(self.payload, "shuffle_s23")
        self.assertEqual(json.dumps(self.payload, sort_keys=True), before)

    def test_subset_payload_keeps_gold_and_k(self):
        for k in (2, 3):
            p = ja2.build_subset_payload(self.payload, self.gold, k, seed=101)
            self.assertEqual(len(p["states"]), 23)
            for s in p["states"]:
                qid, q = next(iter(s["questions"].items()))
                self.assertEqual(len(q["criteria"]), k)
                self.assertIn(self.gold[s["id"]][qid], q["criteria"])
        again = ja2.build_subset_payload(self.payload, self.gold, 2, seed=101)
        self.assertEqual(again, ja2.build_subset_payload(self.payload, self.gold, 2, seed=101))

    def test_subset_payload_rejects_bad_k(self):
        with self.assertRaises(ValueError):
            ja2.build_subset_payload(self.payload, self.gold, 5, seed=1)

    def test_padded_payload_shape_and_contract(self):
        p = ja2.build_padded_payload(self.payload, 255)
        self.assertEqual(len(p["states"]), 23)
        for s, qid, q in ja2.choice_states(self.payload):
            new_crit = next(x["questions"][qid]["criteria"]
                            for x in p["states"] if x["id"] == s["id"])
            self.assertEqual(len(new_crit), 255)
            for k, v in q["criteria"].items():
                self.assertEqual(new_crit[k], v)  # real options preserved, identity order first
            self.assertEqual(list(new_crit)[:4], list(q["criteria"]))

    def test_padded_payload_rejects_bad_k(self):
        with self.assertRaises(ValueError):
            ja2.build_padded_payload(self.payload, 4)
        with self.assertRaises(ValueError):
            ja2.build_padded_payload(self.payload, 256)


class MetricsTest(unittest.TestCase):
    def test_tv_and_argmax(self):
        self.assertEqual(ja2.argmax_key({"a": 0.1, "b": 0.9}), "b")
        self.assertAlmostEqual(ja2.tv_distance({"a": 1.0}, {"a": 0.5, "b": 0.5}), 0.5)

    def test_permutation_metrics_counts_flips(self):
        gold = {"s1": {"q": "a"}, "s2": {"q": "x"}}
        answers = {
            "identity": {
                "s1": {"probs": {"a": 0.9, "b": 0.1}, "argmax": "a", "position": 0, "keys": ["a", "b"]},
                "s2": {"probs": {"x": 0.6, "y": 0.4}, "argmax": "x", "position": 0, "keys": ["x", "y"]},
            },
            "reversed": {
                "s1": {"probs": {"b": 0.2, "a": 0.8}, "argmax": "a", "position": 1, "keys": ["b", "a"]},
                "s2": {"probs": {"y": 0.7, "x": 0.3}, "argmax": "y", "position": 0, "keys": ["y", "x"]},
            },
        }
        m = ja2.permutation_metrics(answers, gold)
        self.assertEqual(m["per_permutation"]["identity"]["n_correct"], 2)
        self.assertEqual(m["per_permutation"]["reversed"]["n_correct"], 1)
        self.assertEqual(m["argmax_flips_vs_identity"]["reversed"]["n_flips"], 1)
        self.assertEqual(m["argmax_flips_vs_identity"]["reversed"]["items"], ["s2"])
        self.assertEqual(m["items_stable_all_permutations"], 1)
        self.assertAlmostEqual(m["per_permutation"]["reversed"]["max_abs_dp"], 0.3)


if __name__ == "__main__":
    unittest.main()
