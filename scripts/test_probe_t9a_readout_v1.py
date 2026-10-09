#!/usr/bin/env python3
"""Tests for probe_t9a_readout_v1.

These exercise the dependency-lazy, model-free helpers of the staged probe:
option mapping, permutation ordering, probability validation, permutation
invariance/negative controls, survey baselines, and a method review of the
frozen protocol/script. No torch, no network, no checkpoint mutation.
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

import probe_t9a_readout_v1 as probe

HERE = Path(__file__).resolve().parent
SURVEY_PATH = HERE.parent / "research/skill_abstention_survey_v1.json"
PROTOCOL_PATH = HERE.parent / "research/t9a_readout_protocol_v1.json"
PROBE_SOURCE = Path(probe.__file__).read_text(encoding="utf-8")

BOOL_Q = {"type": "boolean", "instructions": "p", "criteria": {}}
SCORE2_Q = {"type": "score", "instructions": "rate", "criteria": ["low", "high"]}
SCORE4_Q = {"type": "score", "instructions": "rate4", "criteria": ["s0", "s1", "s2", "s3"]}
CHOICE2_Q = {"type": "choice", "instructions": "pick", "criteria": {"x": "X", "y": "Y"}}
CHOICE3_Q = {"type": "choice", "instructions": "pick3", "criteria": {"a": "A", "b": "B", "c": "C"}}


def load_survey():
    return json.loads(SURVEY_PATH.read_text(encoding="utf-8"))


def load_protocol():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def prob_vector(question, order, selected_key, confidence):
    options = probe.options_for(question)
    other = (1.0 - confidence) / (len(options) - 1) if len(options) > 1 else 0.0
    return [confidence if options[i][0] == selected_key else other for i in order]


def make_observations(question, selected_keys, arm="base_letter", confidence=0.95):
    options = probe.options_for(question)
    orders = probe.orders_for(question, arm)
    if isinstance(selected_keys, str):
        selected_keys = [selected_keys] * len(orders)
    return [probe.observation(options, order, prob_vector(question, order, key, confidence))
            for order, key in zip(orders, selected_keys)]


def make_row(state_id, qid, question, selected_key="true", arm="base_letter", confidence=0.95):
    return probe.readout_row({"id": state_id}, qid, question,
                             make_observations(question, selected_key, arm, confidence), arm)


def make_slot_fixed_row(state_id, qid, question, confidence=0.95):
    """Always argmax slot A, regardless of which semantic option that slot holds."""
    options = probe.options_for(question)
    orders = probe.orders_for(question, "base_letter")
    probs = [confidence] + [(1.0 - confidence) / (len(options) - 1)] * (len(options) - 1)
    obs = [probe.observation(options, order, probs) for order in orders]
    return probe.readout_row({"id": state_id}, qid, question, obs, "base_letter")


def make_conf_row(state_id, qid, question, selected_key, confidences, arm="base_letter"):
    options = probe.options_for(question)
    orders = probe.orders_for(question, arm)
    obs = [probe.observation(options, order, prob_vector(question, order, selected_key, c))
           for order, c in zip(orders, confidences)]
    return probe.readout_row({"id": state_id}, qid, question, obs, arm)


class TestOptionsFor(unittest.TestCase):
    def test_boolean_identity_mapping(self):
        self.assertEqual(probe.options_for(BOOL_Q),
                         [("true", "The proposition is true."),
                          ("false", "The proposition is false.")])

    def test_choice_identity_mapping_preserves_order_and_text(self):
        question = {"type": "choice", "instructions": "pick",
                    "criteria": {"beta": "second", "alpha": "first"}}
        self.assertEqual(probe.options_for(question),
                         [("beta", "beta: second"), ("alpha", "alpha: first")])

    def test_score_identity_mapping_is_ordinal_zero_based(self):
        self.assertEqual(probe.options_for(SCORE2_Q),
                         [("0", "low"), ("1", "high")])
        self.assertEqual([key for key, _ in probe.options_for(SCORE4_Q)],
                         ["0", "1", "2", "3"])

    def test_keys_are_unique(self):
        for question in (BOOL_Q, SCORE4_Q, CHOICE3_Q):
            keys = [key for key, _ in probe.options_for(question)]
            self.assertEqual(len(keys), len(set(keys)))

    def test_unsupported_type_rejected(self):
        with self.assertRaises(ValueError):
            probe.options_for({"type": "likert", "criteria": ["a", "b"]})

    def test_option_count_bounds(self):
        for criteria in ({"a": "A"}, {"a": "A", "b": "B", "c": "C", "d": "D", "e": "E"}):
            with self.assertRaises(ValueError):
                probe.options_for({"type": "choice", "criteria": criteria})
        for criteria in (["only"], ["0", "1", "2", "3", "4"]):
            with self.assertRaises(ValueError):
                probe.options_for({"type": "score", "criteria": criteria})


class TestOrdersFor(unittest.TestCase):
    def test_all_permutation_counts(self):
        self.assertEqual(len(probe.orders_for(CHOICE2_Q, "base_letter")), 2)
        self.assertEqual(len(probe.orders_for(CHOICE3_Q, "base_letter")), 6)
        self.assertEqual(len(probe.orders_for(SCORE4_Q, "base_letter")), 24)
        self.assertEqual(len(probe.orders_for(BOOL_Q, "base_letter")), 2)

    def test_identity_permutation_is_first(self):
        for question in (BOOL_Q, SCORE4_Q, CHOICE3_Q):
            for arm in ("base_letter", "finetuned_letter", "trained_head"):
                orders = probe.orders_for(question, arm)
                self.assertEqual(orders[0], tuple(range(len(probe.options_for(question)))), arm)

    def test_every_order_is_a_unique_valid_permutation(self):
        orders = probe.orders_for(SCORE4_Q, "base_letter")
        size = len(probe.options_for(SCORE4_Q))
        self.assertEqual(len(set(orders)), math.factorial(size))
        for order in orders:
            self.assertEqual(sorted(order), list(range(size)))

    def test_trained_head_boolean_and_score_permutation_not_applicable(self):
        for question in (BOOL_Q, SCORE2_Q, SCORE4_Q):
            orders = probe.orders_for(question, "trained_head")
            self.assertEqual(orders, [tuple(range(len(probe.options_for(question))))])

    def test_trained_head_choice_does_permute(self):
        self.assertEqual(len(probe.orders_for(CHOICE3_Q, "trained_head")), 6)

    def test_letter_arms_permute_boolean_and_score(self):
        self.assertEqual(len(probe.orders_for(BOOL_Q, "finetuned_letter")), 2)
        self.assertEqual(len(probe.orders_for(SCORE4_Q, "finetuned_letter")), 24)


class TestObservationMapping(unittest.TestCase):
    def test_identity_order_maps_probabilities_to_keys(self):
        options = probe.options_for(CHOICE2_Q)
        obs = probe.observation(options, (0, 1), [0.8, 0.2])
        self.assertEqual(obs["order"], ["x", "y"])
        self.assertEqual(obs["probabilities"], {"x": 0.8, "y": 0.2})
        self.assertEqual(obs["selected_id"], "x")
        self.assertEqual(obs["selected_slot"], "A")

    def test_permuted_order_reassigns_slots_not_semantics(self):
        options = probe.options_for(CHOICE2_Q)
        obs = probe.observation(options, (1, 0), [0.9, 0.1])
        # Slot A now holds option y; the mapping follows the order vector.
        self.assertEqual(obs["order"], ["y", "x"])
        self.assertEqual(obs["probabilities"], {"y": 0.9, "x": 0.1})
        self.assertEqual(obs["selected_id"], "y")
        self.assertEqual(obs["selected_slot"], "A")

    def test_derived_fields(self):
        options = probe.options_for(SCORE4_Q)
        obs = probe.observation(options, (0, 1, 2, 3), [0.1, 0.2, 0.6, 0.1], letter_mass=0.42)
        self.assertEqual(obs["selected_id"], "2")
        self.assertEqual(obs["selected_slot"], "C")
        self.assertAlmostEqual(obs["confidence"], 0.6)
        self.assertAlmostEqual(obs["top_margin"], 0.4)
        self.assertAlmostEqual(obs["letter_vocabulary_mass"], 0.42)


class TestProbabilityValidation(unittest.TestCase):
    OPTIONS = [("a", "a"), ("b", "b")]

    def _obs(self, probs, order=(0, 1), letter_mass=None):
        return probe.observation(self.OPTIONS, order, probs, letter_mass)

    def test_valid_probabilities_accepted(self):
        obs = self._obs([0.5, 0.5])
        self.assertEqual(obs["selected_slot"], "A")
        self.assertAlmostEqual(self._obs([0.25, 0.75])["confidence"], 0.75)

    def test_wrong_length_rejected(self):
        with self.assertRaises(ValueError):
            self._obs([1.0])
        with self.assertRaises(ValueError):
            self._obs([0.5, 0.3, 0.2])

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            self._obs([float("nan"), 0.0])

    def test_infinity_rejected(self):
        with self.assertRaises(ValueError):
            self._obs([float("inf"), 0.0])
        with self.assertRaises(ValueError):
            self._obs([float("-inf"), 1.0])

    def test_out_of_unit_range_rejected(self):
        with self.assertRaises(ValueError):
            self._obs([-0.1, 1.1])
        with self.assertRaises(ValueError):
            self._obs([1.0, 0.5])

    def test_sum_must_equal_one(self):
        with self.assertRaisesRegex(ValueError, "sum to one"):
            self._obs([0.5, 0.4])
        with self.assertRaisesRegex(ValueError, "sum to one"):
            self._obs([0.9, 0.0])

    def test_sum_within_tolerance_accepted(self):
        self._obs([1.0, 1e-6])
        with self.assertRaisesRegex(ValueError, "sum to one"):
            self._obs([1.0, 1e-4])

    def test_non_permutation_order_rejected(self):
        with self.assertRaisesRegex(ValueError, "Not a permutation"):
            self._obs([0.5, 0.5], order=(0, 0))

    def test_letter_mass_validation(self):
        for bad in (float("nan"), float("inf"), -0.1, 1.1):
            with self.assertRaises(ValueError):
                self._obs([0.5, 0.5], letter_mass=bad)
        for good in (0.0, 0.5, 1.0):
            self.assertAlmostEqual(self._obs([0.5, 0.5], letter_mass=good)["letter_vocabulary_mass"], good)


class TestQuestionSummaryControls(unittest.TestCase):
    def test_negative_always_letter_a_is_slot_stable_but_not_semantic(self):
        order = probe.orders_for(CHOICE3_Q, "base_letter")
        row = make_slot_fixed_row("s", "q", CHOICE3_Q)
        summary = probe.question_summary(row["observations"], True)
        self.assertTrue(summary["slot_stable"])
        self.assertFalse(summary["semantic_stable"])
        self.assertEqual(summary["permutations"], len(order))

    def test_positive_semantic_constant_is_semantic_stable_but_not_slot_stable(self):
        row = make_row("s", "q", CHOICE3_Q, selected_key="b")
        summary = probe.question_summary(row["observations"], True)
        self.assertTrue(summary["semantic_stable"])
        self.assertFalse(summary["slot_stable"])

    def test_head_boolean_and_score_control_not_applicable(self):
        row_bool = make_row("s", "q", BOOL_Q, "true", arm="trained_head")
        row_score = make_row("s", "q", SCORE4_Q, "0", arm="trained_head")
        for row in (row_bool, row_score):
            self.assertFalse(row["control"]["permutation_applicable"])
            self.assertIsNone(row["control"]["semantic_stable"])
            self.assertIsNone(row["control"]["slot_stable"])
            self.assertEqual(row["control"]["permutations"], 1)

    def test_head_choice_control_applicable(self):
        row = make_row("s", "q", CHOICE3_Q, "a", arm="trained_head")
        self.assertTrue(row["control"]["permutation_applicable"])


class TestSummarizeGate(unittest.TestCase):
    def test_negative_control_fails_permutation_gate(self):
        result = probe.summarize([make_slot_fixed_row("s", "q", CHOICE3_Q)], {}, 0.9)
        self.assertEqual(result["permutation_eligible"], 1)
        self.assertEqual(result["semantic_stable"], 0)
        self.assertEqual(result["slot_stable"], 1)
        self.assertFalse(result["drop_in_permutation_gate"])

    def test_positive_control_passes_permutation_gate(self):
        result = probe.summarize([make_row("s", "q", CHOICE3_Q, "b")], {}, 0.9)
        self.assertEqual(result["semantic_stable"], 1)
        self.assertTrue(result["drop_in_permutation_gate"])

    def test_invariance_is_not_accuracy(self):
        # Semantically stable but always picks the wrong option: gate passes, accuracy is zero.
        wrong = probe.summarize([make_row("s", "q", CHOICE3_Q, "a")], {"q": "c"}, 0.9)
        self.assertTrue(wrong["drop_in_permutation_gate"])
        self.assertEqual(wrong["declared_6"]["original_correct"], 0)
        self.assertEqual(wrong["declared_6"]["order_mean_accuracy"], 0.0)
        # The matching semantic constant is both stable and correct.
        right = probe.summarize([make_row("s", "q", CHOICE3_Q, "c")], {"q": "c"}, 0.9)
        self.assertTrue(right["drop_in_permutation_gate"])
        self.assertEqual(right["declared_6"]["original_correct"], 1)
        self.assertEqual(right["declared_6"]["order_mean_accuracy"], 1.0)

    def test_all_order_coverage(self):
        rows = [make_conf_row("s", "q1", BOOL_Q, "true", [0.95, 0.95]),
                make_conf_row("s", "q2", BOOL_Q, "true", [0.95, 0.5])]
        result = probe.summarize(rows, {}, 0.9)
        self.assertEqual(result["original_answered"], 2)
        self.assertEqual(result["answered_in_every_order"], 1)

    def test_question_weighted_order_accuracy(self):
        # A 2-option choice (2 orders, always right) and a 4-option score (24 orders, half right).
        row_a = make_row("s", "qa", CHOICE2_Q, "x")
        orders_b = probe.orders_for(SCORE4_Q, "base_letter")
        keys_b = ["0" if i % 2 == 0 else "1" for i in range(len(orders_b))]
        row_b = probe.readout_row({"id": "s"}, "qb", SCORE4_Q,
                                  make_observations(SCORE4_Q, keys_b), "base_letter")
        result = probe.summarize([row_a, row_b], {"qa": "x", "qb": "0"}, 0.9)
        weighted = result["declared_6"]["order_mean_accuracy"]
        self.assertAlmostEqual(weighted, 0.75)
        permutation_weighted = (2 * 1.0 + 24 * 0.5) / 26
        self.assertNotAlmostEqual(weighted, permutation_weighted, places=2)


class TestSurveyBaselines(unittest.TestCase):
    LABELS = {"b0_first", "blocked", "needs_new_test", "safe_to_drop", "check_secrets", "token_savings"}

    def _rows(self, force_wrong=()):
        survey = load_survey()
        labels = survey["expected_answers"]
        rows = []
        for state in survey["request"]["states"]:
            for qid, question in state["questions"].items():
                if qid in labels:
                    key = labels[qid]
                    if qid in force_wrong:
                        key = "false" if key == "true" else "true"
                else:
                    key = probe.options_for(question)[0][0]
                rows.append(make_row(state["id"], qid, question, key))
        return survey, labels, rows

    def test_staged_survey_has_six_declared_labels(self):
        survey, labels, rows = self._rows()
        self.assertEqual(len(rows), 13)
        self.assertEqual(set(labels), self.LABELS)
        self.assertTrue(all(value in {"true", "false"} for value in labels.values()))

    def test_declared_six_and_sensitivity_four(self):
        _, labels, rows = self._rows()
        result = probe.summarize(rows, labels, load_protocol()["threshold"])
        declared = result["declared_6"]
        self.assertEqual(declared["n"], 6)
        self.assertEqual(declared["original_correct"], 6)
        self.assertEqual(declared["original_covered"], 6)
        self.assertEqual(declared["original_covered_correct"], 6)
        self.assertEqual(declared["all_orders_correct"], 6)
        self.assertAlmostEqual(declared["order_mean_accuracy"], 1.0)
        sensitivity = result["sensitivity_4"]
        self.assertEqual(sensitivity["n"], 4)
        self.assertEqual(sensitivity["original_correct"], 4)
        self.assertEqual(sensitivity["all_orders_correct"], 4)

    def test_constant_baselines_use_staged_labels(self):
        _, labels, rows = self._rows()
        result = probe.summarize(rows, labels, load_protocol()["threshold"])
        self.assertEqual(result["declared_6"]["constant_true_correct"], 4)
        self.assertEqual(result["declared_6"]["constant_false_correct"], 2)
        self.assertEqual(result["sensitivity_4"]["constant_true_correct"], 2)
        self.assertEqual(result["sensitivity_4"]["constant_false_correct"], 2)
        for name in ("declared_6", "sensitivity_4"):
            block = result[name]
            self.assertEqual(block["constant_true_correct"] + block["constant_false_correct"], block["n"])

    def test_sensitivity_four_excludes_two_labels(self):
        _, labels, rows = self._rows(force_wrong=("check_secrets", "needs_new_test"))
        result = probe.summarize(rows, labels, load_protocol()["threshold"])
        self.assertEqual(result["declared_6"]["n"], 6)
        self.assertEqual(result["declared_6"]["original_correct"], 4)
        # The two excluded booleans must not drag the four-label sensitivity figure.
        self.assertEqual(result["sensitivity_4"]["n"], 4)
        self.assertEqual(result["sensitivity_4"]["original_correct"], 4)


class TestMethodReview(unittest.TestCase):
    def test_protocol_is_three_local_mps_fp32_arms(self):
        protocol = load_protocol()
        self.assertEqual(protocol["schema_version"], "nanojev-t9a-readout-protocol-v1")
        self.assertEqual(protocol["arms"], ["base_letter", "finetuned_letter", "trained_head"])
        self.assertEqual(protocol["device"], "mps")
        self.assertEqual(protocol["dtype"], "float32")
        self.assertEqual(protocol["temperature"], 1.0)
        self.assertEqual(protocol["enable_thinking"], False)
        self.assertEqual(protocol["autoregressive_decode_steps"], 0)
        self.assertTrue(protocol["base_revision"])

    def test_survey_sha_is_frozen(self):
        self.assertEqual(load_protocol()["survey_sha256"], probe.sha256(SURVEY_PATH))

    def test_necessary_only_gate_and_no_promotion(self):
        protocol = load_protocol()
        self.assertIn("necessary, not sufficient", protocol["decision_rule"])
        self.assertIn("tuning", protocol["decision_rule"])
        self.assertIn("promotion", protocol["decision_rule"])
        self.assertIn("not a fresh holdout", protocol["purpose"])
        # The head boolean/score permutation test is explicitly not applicable.
        self.assertIn("not applicable", protocol["permutations"])

    def test_script_is_local_strict_tied_and_untuned(self):
        source = PROBE_SOURCE
        self.assertIn("local_files_only=True", source)
        self.assertIn("HF_HUB_OFFLINE", source)
        self.assertIn("strict=True", source)
        self.assertIn("tie_word_embeddings", source)
        self.assertIn('body["lm_head.weight"] = body["model.embed_tokens.weight"]', source)
        self.assertIn("tie_weights()", source)
        self.assertIn("data_ptr()", source)
        self.assertIn('protocol["temperature"] != 1 or protocol["dtype"] != "float32"', source)
        self.assertIn('open("x"', source)

    def test_letter_mass_covers_full_vocabulary(self):
        self.assertIn("logits_to_keep=1", PROBE_SOURCE)
        self.assertIn("torch.logsumexp(logits, 0)", PROBE_SOURCE)

    def test_probe_import_does_not_require_torch(self):
        self.assertFalse(hasattr(probe, "torch"))


class TestWriteNew(unittest.TestCase):
    def test_loading_diagnostics_sets_are_json_serializable(self):
        result = probe.checked_loading_info({"missing_keys": set(), "unexpected_keys": set(),
                                             "mismatched_keys": [], "error_msgs": []})
        self.assertEqual(json.loads(json.dumps(result))["missing_keys"], [])

    def test_loading_diagnostics_reject_missing_weights(self):
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            with self.assertRaises(ValueError):
                probe.checked_loading_info({key: {"bad_weight"}})

    def test_write_new_creates_then_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            probe.write_new(path, {"a": 1})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"a": 1})
            with self.assertRaises(FileExistsError):
                probe.write_new(path, {"a": 2})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"a": 1})

    def test_write_new_refuses_non_finite_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            with self.assertRaises(ValueError):
                probe.write_new(path, {"x": float("nan")})
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
