#!/usr/bin/env python3
"""Unit tests for the X2 Arm C minimal pointer head (DecisionModel set_head='pointer').

CPU only, tiny fake backbone — no checkpoint or download. Covers: output contract
(padded logits + valid mask, softmax over valid sums to 1), boolean null-key
semantics ([0, s] at init), gradient flow into pointer parameters, anchor
conditioning, determinism, missing-anchor failure, and attention-mode backward
compatibility when anchor_tokens are present.
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn

from train_toy_decisions import DecisionModel


class FakeBackbone(nn.Module):
    """Deterministic embedding-only stand-in with the AutoModel output shape."""

    def __init__(self, vocab=64, hidden=16):
        super().__init__()
        self.config = type("Cfg", (), {"hidden_size": hidden})()
        self.embed = nn.Embedding(vocab, hidden)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        h = self.embed(input_ids)
        if attention_mask is not None:
            h = h * attention_mask[..., None].to(h.dtype)
        # Cheap causal mixing so a path's final hidden state depends on all tokens.
        h = h.cumsum(dim=1)
        return type("Out", (), {"last_hidden_state": h})()


def ex(qid, typ, n_ids, n_leaves, anchor=(40, 41, 42, 2)):
    ex = {"id": qid, "qid": qid, "type": typ,
          "candidate_ids": [f"c{i}" for i in range(n_ids)],
          "leaf_tokens": [[10 + i, 11, 12, 2][:4] for i in range(n_leaves)]}
    if anchor is not None:
        ex["anchor_tokens"] = list(anchor)
    return ex


def examples():
    return [ex("b", "boolean", 2, 1),
            ex("c", "choice", 3, 3),
            ex("s", "score", 4, 4)]


class PointerHeadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = DecisionModel(FakeBackbone(), "pointer")

    def test_output_contract_and_normalization(self):
        logits, valid = self.model(examples(), pad_token=0)
        self.assertEqual(logits.shape, (3, 4))  # kmax = 4
        self.assertTrue(torch.isfinite(logits[valid]).all())
        self.assertTrue((logits[~valid] <= -1e8).all())
        for i, ex_ in enumerate(examples()):
            k = len(ex_["candidate_ids"])
            probs = logits[i, :k].softmax(-1)
            self.assertAlmostEqual(float(probs.sum()), 1.0, places=6)

    def test_boolean_null_key_is_zero_logit_at_init(self):
        logits, _ = self.model([ex("b", "boolean", 2, 1)], pad_token=0)
        # pointer_null is zero-init => q·null = 0, matching the scalar head's [0, z].
        self.assertEqual(float(logits[0, 0]), 0.0)
        self.assertNotEqual(float(logits[0, 1]), 0.0)

    def test_gradients_reach_pointer_params(self):
        logits, _ = self.model(examples(), pad_token=0)
        target = torch.zeros_like(logits)
        target[0, 1] = target[1, 2] = target[2, 0] = 1.0
        loss = -(target * logits.log_softmax(-1)).sum(-1).mean()
        loss.backward()
        for name in ("pointer_query.weight", "pointer_key.weight", "pointer_null"):
            param = dict(self.model.named_parameters())[name]
            self.assertIsNotNone(param.grad, name)
            self.assertGreater(float(param.grad.abs().sum()), 0.0, name)

    def test_anchor_conditioning(self):
        a, _ = self.model(examples(), pad_token=0)
        changed = [ex("b", "boolean", 2, 1, anchor=(9, 8, 7, 2)),
                   ex("c", "choice", 3, 3, anchor=(9, 8, 7, 2)),
                   ex("s", "score", 4, 4, anchor=(9, 8, 7, 2))]
        b, _ = self.model(changed, pad_token=0)
        self.assertFalse(torch.allclose(a, b))

    def test_determinism(self):
        a, _ = self.model.eval()(examples(), pad_token=0)
        b, _ = self.model(examples(), pad_token=0)
        self.assertTrue(torch.equal(a, b))

    def test_missing_anchor_fails_closed(self):
        with self.assertRaises(ValueError):
            self.model([ex("c", "choice", 3, 3, anchor=None)], pad_token=0)

    def test_attention_mode_ignores_anchor(self):
        torch.manual_seed(0)
        model = DecisionModel(FakeBackbone(), "attention")
        a, _ = model(examples(), pad_token=0)
        no_anchor = [ex("b", "boolean", 2, 1, anchor=None),
                     ex("c", "choice", 3, 3, anchor=None),
                     ex("s", "score", 4, 4, anchor=None)]
        b, _ = model(no_anchor, pad_token=0)
        self.assertTrue(torch.equal(a, b))

    def test_pointer_params_are_head_group(self):
        """Trainer convention: head = every non-backbone parameter."""
        head = {n for n, _ in self.model.named_parameters()
                if not n.startswith("backbone.")}
        for name in ("pointer_query.weight", "pointer_key.weight",
                     "pointer_null", "norm_anchor.weight"):
            self.assertIn(name, head)


if __name__ == "__main__":
    unittest.main()
