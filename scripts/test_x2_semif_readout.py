#!/usr/bin/env python3
"""Unit tests for X2 Arm B native-logit readout (no model download, CPU only).

Covers: per-type label mapping, candidate-only normalization, the >26 leaf-EOS
fallback contract (255 candidates), determinism, and byte-identity of the prompt
prefix with the production serializer. A stub tokenizer + stub causal LM exercise
the batch paths without any checkpoint.
"""
import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import x2_semif_readout as x2
from predict_toy_decisions import prepare_examples, question_prefix_segments


class StubTokenizer:
    """Deterministic char-level tokenizer with dedicated single ids for labels."""

    LABELS = {label: 500 + i for i, label in enumerate(
        ["False", "True"] + list(x2.CHOICE_LABELS) + [str(i) for i in range(10)])}

    def __init__(self):
        self.eos_token_id = 2
        self.pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        if text in self.LABELS:
            return [self.LABELS[text]]
        return [100 + ord(c) for c in text]

    def decode(self, ids):
        inv = {v: k for k, v in self.LABELS.items()}
        return "".join(inv.get(i, chr(i - 100)) for i in ids)

    def get_vocab(self):
        return dict(self.LABELS)


def choice_q(k):
    return {"type": "choice", "instructions": "Pick the safe action.",
            "criteria": {f"opt_{i:03d}": f"description {i}" for i in range(k)}}


BOOLEAN_Q = {"type": "boolean", "instructions": "Is it safe?",
             "criteria": {"false": "not safe", "true": "safe"}}
SCORE_Q = {"type": "score", "instructions": "Rate risk.",
           "criteria": ["low", "medium", "high"]}
STATE = "INC-1 timeline\n14:00 deploy started"


class LabelSchemeTest(unittest.TestCase):
    def test_boolean_maps_false_then_true(self):
        self.assertEqual(x2.label_scheme(BOOLEAN_Q),
                         [("false", "False"), ("true", "True")])
        self.assertEqual(x2.candidate_ids(BOOLEAN_Q), ["false", "true"])

    def test_choice_letters_in_criteria_order(self):
        q = choice_q(4)
        self.assertEqual([l for _c, l in x2.label_scheme(q)], ["A", "B", "C", "D"])
        self.assertEqual([c for c, _l in x2.label_scheme(q)], list(q["criteria"]))

    def test_score_labels_are_level_indices(self):
        self.assertEqual(x2.label_scheme(SCORE_Q),
                         [("0", "0"), ("1", "1"), ("2", "2")])

    def test_choice_over_26_rejected_by_label_scheme(self):
        with self.assertRaises(ValueError):
            x2.label_scheme(choice_q(27))

    def test_readout_mode_switch(self):
        self.assertEqual(x2.readout_mode(choice_q(26)), "label")
        self.assertEqual(x2.readout_mode(choice_q(27)), "leaf_eos")
        self.assertEqual(x2.readout_mode(choice_q(255)), "leaf_eos")
        self.assertEqual(x2.readout_mode(BOOLEAN_Q), "label")
        self.assertEqual(x2.readout_mode(SCORE_Q), "label")


class NormalizationTest(unittest.TestCase):
    def test_softmax_over_candidates_only(self):
        probs = x2.probs_from_selected_logits([0.0, 1.0, -1.0])
        self.assertAlmostEqual(math.fsum(probs), 1.0, places=12)
        self.assertEqual(probs.index(max(probs)), 1)

    def test_extreme_logits_stay_finite(self):
        probs = x2.probs_from_selected_logits([1000.0, -1000.0, 0.0])
        self.assertAlmostEqual(math.fsum(probs), 1.0, places=12)
        self.assertTrue(all(math.isfinite(p) and 0 <= p <= 1 for p in probs))

    def test_255_way_distribution(self):
        probs = x2.probs_from_selected_logits([math.sin(i) for i in range(255)])
        self.assertEqual(len(probs), 255)
        self.assertAlmostEqual(math.fsum(probs), 1.0, places=12)


class SerializationIdentityTest(unittest.TestCase):
    """Prompt prefix must be byte-identical to the production leaf-path prefix."""

    def test_prefix_segments_shared_with_prepare_examples(self):
        tok = StubTokenizer()
        for q in (BOOLEAN_Q, choice_q(4), SCORE_Q):
            payload = {"states": [{"id": "s1", "state": STATE, "questions": {"q": q}}]}
            ex = prepare_examples(payload, tok, 4096)[0]
            prefix = sum([tok.encode(t, add_special_tokens=False)
                          for t in question_prefix_segments(STATE, q)], [])
            for leaf in ex["leaf_tokens"]:
                self.assertEqual(leaf[:len(prefix)], prefix)
            prompt, cids, _labels = x2.build_prompt_tokens(STATE, q, tok)
            self.assertEqual(prompt[:len(prefix)], prefix)
            self.assertEqual(cids, ex["candidate_ids"])

    def test_anchor_tokens_present_and_shorter_than_leaves(self):
        tok = StubTokenizer()
        payload = {"states": [{"id": "s1", "state": STATE, "questions": {"q": choice_q(3)}}]}
        ex = prepare_examples(payload, tok, 4096)[0]
        self.assertIn("anchor_tokens", ex)
        for leaf in ex["leaf_tokens"]:
            self.assertLessEqual(len(ex["anchor_tokens"]), len(leaf))
        self.assertEqual(ex["anchor_tokens"][-1], tok.eos_token_id)

    def test_prompt_is_deterministic(self):
        tok = StubTokenizer()
        a = x2.build_prompt_tokens(STATE, choice_q(4), tok)
        b = x2.build_prompt_tokens(STATE, choice_q(4), tok)
        self.assertEqual(a, b)

    def test_labels_are_single_clean_tokens(self):
        tok = StubTokenizer()
        for q in (BOOLEAN_Q, choice_q(26), SCORE_Q):
            _ids, _cids, labels = x2.build_prompt_tokens(STATE, q, tok)
            slots = x2.label_token_ids(tok, labels)
            self.assertEqual(len(slots), len(labels))


class StubModelReadoutTest(unittest.TestCase):
    """End-to-end batch readout with a stub causal LM (no checkpoint)."""

    class StubLM:
        """Returns logits = linear hash of last-token id per vocab slot; vocab 600."""

        def __init__(self, label_boost=None):
            self.label_boost = label_boost or {}

        def __call__(self, input_ids, attention_mask, use_cache=False):
            import torch
            b, w = input_ids.shape
            vocab = 600
            logits = torch.zeros(b, w, vocab)
            for i in range(b):
                last = int(input_ids[i, int(attention_mask[i].sum()) - 1])
                logits[i, :, :] = (last % 17) * 0.01
                for token_id, boost in self.label_boost.items():
                    logits[i, :, token_id] += boost
            return type("Out", (), {"logits": logits})()

    def setUp(self):
        import torch  # noqa: F401  (ensures torch present for stub paths)
        self.tok = StubTokenizer()

    def test_label_path_picks_boosted_label(self):
        import torch
        q = choice_q(3)
        # Boost "C" so the argmax lands on the third candidate.
        model = self.StubLM({StubTokenizer.LABELS["C"]: 5.0})
        rows = [{"id": "s1", "state": STATE, "questions": {"q": q}}]
        out = x2.run_readout(model, self.tok, torch.device("cpu"), rows, 4, 4096)[0]
        self.assertEqual(len(out), 1)
        r = out[0]
        self.assertEqual(r["mode"], "label")
        self.assertEqual(r["selected_id"], "opt_002")
        self.assertAlmostEqual(math.fsum(r["probabilities"].values()), 1.0, places=6)
        self.assertEqual(list(r["probabilities"]), list(q["criteria"]))

    def test_boolean_and_score_label_paths(self):
        import torch
        model = self.StubLM({StubTokenizer.LABELS["True"]: 3.0,
                             StubTokenizer.LABELS["2"]: 3.0})
        rows = [{"id": "s1", "state": STATE,
                 "questions": {"b": BOOLEAN_Q, "s": SCORE_Q}}]
        out = x2.run_readout(model, self.tok, torch.device("cpu"), rows, 4, 4096)[0]
        by_qid = {r["qid"]: r for r in out}
        self.assertEqual(by_qid["b"]["selected_id"], "true")
        self.assertEqual(by_qid["s"]["selected_id"], "2")

    def test_255_contract_via_leaf_fallback(self):
        import torch
        q = choice_q(255)
        model = self.StubLM()
        rows = [{"id": "s1", "state": STATE, "questions": {"q": q}}]
        out = x2.run_readout(model, self.tok, torch.device("cpu"), rows, 2, 8192)[0]
        self.assertEqual(len(out), 1)
        r = out[0]
        self.assertEqual(r["mode"], "leaf_eos_fallback")
        self.assertEqual(len(r["probabilities"]), 255)
        self.assertEqual(set(r["probabilities"]), set(q["criteria"]))
        self.assertAlmostEqual(math.fsum(r["probabilities"].values()), 1.0, places=6)
        self.assertIn(r["selected_id"], q["criteria"])

    def test_readout_is_deterministic(self):
        import torch
        q = choice_q(4)
        rows = [{"id": "s1", "state": STATE, "questions": {"q": q}}]
        a = x2.run_readout(self.StubLM(), self.tok, torch.device("cpu"), rows, 4, 4096)[0]
        b = x2.run_readout(self.StubLM(), self.tok, torch.device("cpu"), rows, 4, 4096)[0]
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_gold_key_per_type(self):
        self.assertEqual(x2.gold_key("boolean", True), "true")
        self.assertEqual(x2.gold_key("boolean", False), "false")
        self.assertEqual(x2.gold_key("score", 2), "2")
        self.assertEqual(x2.gold_key("choice", "opt_x"), "opt_x")


class ScoringTest(unittest.TestCase):
    def test_score_rows_cov_and_per_type(self):
        rows = [
            {"id": "a", "qid": "q", "type": "boolean", "label_vocabulary_mass": 0.9,
             "probabilities": {"false": 0.95, "true": 0.05}},
            {"id": "b", "qid": "q", "type": "choice",
             "probabilities": {"x": 0.3, "y": 0.7}},
        ]
        gold = {"a": {"q": False}, "b": {"q": "y"}}
        m = x2.score_rows(rows, gold)
        self.assertEqual(m["questions"], 2)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["answered_at_0.90_rate"], 0.5)
        self.assertEqual(m["by_type"]["boolean"]["answered_at_0.90"], 1)
        self.assertEqual(m["by_type"]["choice"]["answered_at_0.90"], 0)


if __name__ == "__main__":
    unittest.main()
