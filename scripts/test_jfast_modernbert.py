import unittest

import torch
from torch import nn

import train_jfast_modernbert as t


class FakeTokenizer:
    mask_token_id = 999
    pad_token_id = 0

    def encode(self, text, add_special_tokens=True):
        ids = [ord(c) % 100 + 1 for c in text]
        return ([1] if add_special_tokens else []) + ids


class JFastEncodingTests(unittest.TestCase):
    def test_row_to_example_targets_labels(self):
        row = {
            "schema_version": "nanojev-engineering-judgment-trainer-row-v1",
            "id": "r1",
            "state": "incident state",
            "questions": {"q": {"type": "choice",
                                "instructions": "pick",
                                "criteria": {"a": "safe", "b": "unsafe"}}},
            "gold_probs": {"q": {"a": 1.0, "b": 0.0}},
        }
        ex = t.row_to_example(row, FakeTokenizer(), 512)
        self.assertEqual(ex["labels"], ["a", "b"])
        self.assertEqual(ex["target"], [1.0, 0.0])
        self.assertEqual(len(ex["marker_positions"]), 2)

    def test_boolean_gets_two_markers(self):
        row = {
            "schema_version": "nanojev-engineering-judgment-trainer-row-v1",
            "id": "r2",
            "state": "incident state",
            "questions": {"q": {"type": "boolean",
                                "instructions": "is it safe?",
                                "criteria": {"false": "no", "true": "yes"}}},
            "gold_probs": {"q": {"false": 0.25, "true": 0.75}},
        }
        ex = t.row_to_example(row, FakeTokenizer(), 512)
        self.assertEqual(ex["labels"], ["false", "true"])
        self.assertEqual(ex["target"], [0.25, 0.75])

    def test_marker_after_option_text(self):
        row = {
            "schema_version": "nanojev-engineering-judgment-trainer-row-v1",
            "id": "r3",
            "state": "incident state",
            "questions": {"q": {"type": "choice",
                                "instructions": "pick",
                                "criteria": {"a": "safe", "b": "unsafe"}}},
            "gold_probs": {"q": {"a": 1.0, "b": 0.0}},
        }
        before = t.row_to_example(row, FakeTokenizer(), 512,
                                  marker_placement="before")
        after = t.row_to_example(row, FakeTokenizer(), 512,
                                 marker_placement="after")
        self.assertEqual(len(before["input_ids"]), len(after["input_ids"]))
        self.assertGreater(after["marker_positions"][0],
                           before["marker_positions"][0])
        self.assertTrue(all(after["input_ids"][p] == 999
                            for p in after["marker_positions"]))


class JFastBatchLossTests(unittest.TestCase):
    def test_batch_padding_and_positions(self):
        examples = [{"input_ids": [1, 2, 3], "marker_positions": [1]},
                    {"input_ids": [4, 5], "marker_positions": [0]}]
        ids, mask, positions = t.batch_examples(examples, 0, torch.device("cpu"))
        self.assertEqual(ids.tolist(), [[1, 2, 3], [4, 5, 0]])
        self.assertEqual(mask.tolist(), [[1, 1, 1], [1, 1, 0]])
        self.assertEqual(positions[0].tolist(), [1])
        self.assertEqual(positions[1].tolist(), [0])

    def test_gold_ce_uses_distribution(self):
        logits = [torch.tensor([0.0, 1.0])]
        examples = [{"target": [0.5, 0.5]}]
        expected = -(0.5 * torch.log_softmax(logits[0], 0)[0] +
                   0.5 * torch.log_softmax(logits[0], 0)[1])
        self.assertAlmostEqual(float(t.gold_ce(logits, examples)), float(expected))

    def test_grouped_batches_respects_question_bound(self):
        examples = [{"input_ids": [1] * n} for n in (10, 10, 10)]
        batches = t.grouped_batches(examples, batch_questions=2, max_tokens=30)
        self.assertEqual([len(x) for x in batches], [2, 1])


class JFastLoraTests(unittest.TestCase):
    def test_injection_targets_modernbert_names(self):
        backbone = nn.Module()
        layer = nn.Module()
        layer.attn = nn.Module()
        layer.attn.Wqkv = nn.Linear(8, 16)
        layer.attn.Wo = nn.Linear(8, 8)
        layer.mlp = nn.Module()
        layer.mlp.Wi = nn.Linear(8, 16)
        layer.mlp.Wo = nn.Linear(16, 8)
        backbone.layers = nn.ModuleList([layer])

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.marker_head = nn.Linear(8, 1)

        model = Model()
        wrapped = t.inject_jfast_lora(model, rank=2, alpha=4, dropout=0.0)
        self.assertEqual(len(wrapped), 4)
        self.assertIn("layers.0.attn.Wqkv", wrapped)
        self.assertIn("layers.0.attn.Wo", wrapped)
        self.assertIn("layers.0.mlp.Wi", wrapped)
        self.assertIn("layers.0.mlp.Wo", wrapped)
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(any("lora_A" in n for n in trainable))
        self.assertTrue(any(n.startswith("marker_head") for n in trainable))
        self.assertFalse(any(n == "backbone.layers.0.attn.Wqkv.weight"
                             for n in trainable))


if __name__ == "__main__":
    unittest.main()
