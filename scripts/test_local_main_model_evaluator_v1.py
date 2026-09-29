import json
import unittest

from local_main_model_evaluator_v1 import (
    DeterministicEvidenceResponder, LocalGenerationResponder,
    evaluate_pair, canonical_request_text, normalize_answer,
)


class LocalMainModelEvaluatorTest(unittest.TestCase):
    def setUp(self):
        self.request = json.dumps({
            "messages": [
                {"role": "system", "content": "Answer from evidence."},
                {"role": "user", "content": "What is the limit?"},
                {"role": "tool", "name": "config", "content": "limit=42"},
            ]
        }, ensure_ascii=False).encode("utf-8")
        self.contract = {
            "expected_answer": "42",
            "required_strings": ["limit=42", "What is the limit?"],
        }

    def test_answered_when_required_evidence_is_present(self):
        result = evaluate_pair(self.request, self.request, self.contract)
        self.assertEqual(result["pair_status"], "passed")
        self.assertTrue(result["original"]["answer_ok"])
        self.assertTrue(result["reduced"]["answer_ok"])
        self.assertFalse(result["answer_regression"])

    def test_local_generation_normalization_handles_format_variants(self):
        self.assertEqual(normalize_answer("In stock"), "in stock")
        self.assertEqual(normalize_answer("in_stock"), "in stock")
        self.assertEqual(normalize_answer('/v1/status, 30 seconds'),
                         'v1 status 30 seconds')
        self.assertEqual(normalize_answer('/v1/status and 30 seconds'),
                         'v1 status 30 seconds')
        self.assertEqual(normalize_answer('<think>x</think>42'), '42')

    def test_missing_evidence_is_a_paired_regression(self):
        reduced = json.dumps({
            "messages": [
                {"role": "system", "content": "Answer from evidence."},
                {"role": "user", "content": "What is the limit?"},
            ]
        }, ensure_ascii=False).encode("utf-8")
        result = evaluate_pair(self.request, reduced, self.contract)
        self.assertEqual(result["pair_status"], "failed")
        self.assertTrue(result["original"]["answer_ok"])
        self.assertFalse(result["reduced"]["answer_ok"])
        self.assertTrue(result["answer_regression"])
        self.assertEqual(len(result["reduced"]["missing_required_string_sha256"]), 1)

    def test_token_counter_records_removed_usage_estimate(self):
        counter = lambda text: len(text)
        responder = DeterministicEvidenceResponder(counter)
        reduced = self.request.replace(b"limit=42", b"")
        result = evaluate_pair(self.request, reduced, self.contract, responder)
        self.assertGreater(result["original"]["prompt_tokens"], 0)
        self.assertLess(result["reduced"]["prompt_tokens"],
                        result["original"]["prompt_tokens"])
        self.assertGreater(result["prompt_tokens_removed"], 0)

    def test_canonical_request_text_accepts_non_json(self):
        self.assertEqual(canonical_request_text(b"not json"), "not json")

    def test_local_generation_backend_uses_injected_generator(self):
        calls = []

        def generate(prompt):
            calls.append(prompt)
            return "42"

        responder = LocalGenerationResponder(generate_fn=generate)
        result = evaluate_pair(self.request, self.request, self.contract, responder)
        self.assertEqual(result["pair_status"], "passed")
        self.assertTrue(result["original"]["answer_ok"])
        self.assertTrue(result["reduced"]["answer_ok"])
        self.assertEqual(len(calls), 2)
        self.assertNotIn("answer", result["original"])
        self.assertIn("local-generation:Qwen/Qwen3-0.6B", result["backend"])

    def test_local_generation_backend_detects_answer_mismatch(self):
        def generate(prompt):
            return "42" if "limit=42" in prompt else "unknown"

        responder = LocalGenerationResponder(generate_fn=generate)
        reduced = self.request.replace(b"limit=42", b"")
        result = evaluate_pair(self.request, reduced, self.contract, responder)
        self.assertEqual(result["pair_status"], "failed")
        self.assertTrue(result["answer_regression"])
        self.assertTrue(result["original"]["answer_ok"])
        self.assertFalse(result["reduced"]["answer_ok"])


if __name__ == "__main__":
    unittest.main()
