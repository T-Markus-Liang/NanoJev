import json
from types import SimpleNamespace
import unittest
from pathlib import Path

import benchmark_filter_value_v1 as value_runner
from benchmark_filter_value_v1 import (
    DEFAULT_MANIFEST, DEFAULT_THRESHOLD_POLICY, assert_review_thresholds,
    build_arm_scorer, evaluate_arm, evaluate_dedup_case, load_threshold_policy,
)
from build_tool_history_fixtures_v1 import deterministic_scorer

V2_MANIFEST = Path(__file__).resolve().parent.parent / "research" / "context_filter_value_fixture_manifest_v2.json"
V3_MANIFEST = Path(__file__).resolve().parent.parent / "research" / "context_filter_value_fixture_manifest_v3.json"


class FilterValueFixtureTest(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))

    def test_manifest_is_self_authored_and_covers_adverse_cases(self):
        self.assertEqual(self.manifest["schema_version"],
                         "nanojev-context-filter-value-fixture-manifest-v1")
        self.assertEqual(len(self.manifest["cases"]), 8)
        kinds = {case["kind"] for case in self.manifest["cases"]}
        self.assertIn("required_evidence_marked_eligible", kinds)
        self.assertIn("ambiguous_history", kinds)
        self.assertIn("tool_result_plus_irrelevant_note", kinds)
        self.assertFalse(self.manifest["provenance"]["contains_real_credentials"])

    def test_deterministic_stub_is_detected_as_unsafe(self):
        result = evaluate_arm(self.manifest, "stub", scorer=deterministic_scorer,
                              threshold=0.99)
        self.assertEqual(result["totals"]["unsafe_removals"], 1)
        self.assertGreater(result["totals"]["required_string_failures"], 0)
        case = next(case for case in result["cases"]
                    if case["case_id"] == "required_evidence_marked_eligible")
        self.assertTrue(case["unsafe_removal"])
        self.assertEqual(case["required_strings_present"], 0)
        self.assertTrue(case["provider_pair"]["original"]["answer_ok"])
        self.assertFalse(case["provider_pair"]["reduced"]["answer_ok"])
        self.assertTrue(case["provider_pair"]["answer_regression"])

    def test_safe_dedup_removes_only_the_later_exact_duplicate(self):
        case = next(case for case in self.manifest["cases"]
                    if case["case_id"] == "exact_duplicate")
        raw, reduced, receipt, round_trip = evaluate_dedup_case(case)
        drops = [segment["pointer"] for segment in receipt["segments"]
                 if segment["suggestion"] == "drop"]
        self.assertEqual(drops, ["/messages/3/content"])
        self.assertLess(len(reduced), len(raw))
        self.assertTrue(round_trip["ok"])

    def test_control_never_removes_and_marks_no_unsafe_cases(self):
        result = evaluate_arm(self.manifest, "control")
        self.assertEqual(result["totals"]["removed_bytes"], 0)
        self.assertEqual(result["totals"]["unsafe_removals"], 0)
        self.assertEqual(result["totals"]["required_string_failures"], 0)
        self.assertEqual(result["totals"]["original_answer_ok"], 8)
        self.assertEqual(result["totals"]["reduced_answer_ok"], 8)
        self.assertEqual(result["totals"]["paired_answer_regressions"], 0)

    def test_v2_manifest_expands_families_and_bypass_cases(self):
        manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"],
                         "nanojev-context-filter-value-fixture-manifest-v2")
        self.assertEqual(manifest["case_count"], 25)
        kinds = {case["kind"] for case in manifest["cases"]}
        self.assertIn("multilingual_clear_irrelevant", kinds)
        self.assertIn("policy_and_credential_placeholder", kinds)
        self.assertIn("nonrepeatable_tool_result", kinds)
        self.assertIn("retrieval_distractor_cited_evidence", kinds)
        self.assertIn("unsupported_image_part_bypass", kinds)
        self.assertFalse(manifest["provenance"]["contains_real_credentials"])
        self.assertFalse(manifest["provenance"]["contains_provider_outputs"])

    def test_v3_manifest_is_new_heldout_cohort(self):
        manifest = json.loads(V3_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"],
                         "nanojev-context-filter-value-fixture-manifest-v3")
        self.assertEqual(manifest["case_count"], 12)
        self.assertTrue(all(case["case_id"].startswith("heldout_")
                            for case in manifest["cases"]))
        self.assertEqual(manifest["provenance"]["split"], "heldout_replay")
        bypass = [case for case in manifest["cases"]
                  if case["expected_gate"]["status"] == "bypass"]
        self.assertEqual(len(bypass), 2)
        self.assertFalse(manifest["provenance"]["contains_benchmark_rows"])
        self.assertFalse(manifest["provenance"]["contains_provider_outputs"])

    def test_v2_stub_detection_and_bypass_paths(self):
        manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
        result = evaluate_arm(manifest, "stub", scorer=deterministic_scorer,
                              threshold=0.99)
        self.assertEqual(result["totals"]["unsafe_removals"], 1)
        self.assertEqual(result["totals"]["paired_answer_regressions"], 1)
        reasons = {case["case_id"]: case["reason"] for case in result["cases"]}
        self.assertEqual(reasons["pending_tool_call_bypass"], "unresolved_tool_link")
        self.assertEqual(reasons["orphan_tool_result_bypass"], "unresolved_tool_link")
        self.assertEqual(reasons["unsupported_image_part_bypass"], "unsupported_content")
        self.assertEqual(reasons["unsupported_server_context_bypass"], "unresolved_server_context")
        self.assertEqual(reasons["unsupported_envelope_field_bypass"], "unsupported_envelope_fields")
        checkpoints = {case["case_id"]: case.get("scorer_checkpoint") for case in result["cases"]}
        self.assertIsNone(checkpoints["pending_tool_call_bypass"])
        self.assertIsNone(checkpoints["unsupported_image_part_bypass"])

    def test_threshold_policy_allows_declared_diagnostic_points(self):
        policy = load_threshold_policy(DEFAULT_THRESHOLD_POLICY)
        assert_review_thresholds(policy, 0.90, 0.95, ["systemone"])
        assert_review_thresholds(policy, 0.99, 0.95, ["cascade"])

    def test_threshold_policy_rejects_undeclared_review_threshold(self):
        policy = load_threshold_policy(DEFAULT_THRESHOLD_POLICY)
        with self.assertRaisesRegex(ValueError, "undeclared diagnostic threshold"):
            assert_review_thresholds(policy, 0.80, 0.95, ["systemone"])
        with self.assertRaisesRegex(ValueError, "requires --threshold-policy"):
            assert_review_thresholds(None, 0.90, 0.95, ["systemone"])

    def test_threshold_policy_rejects_undeclared_cascade_threshold(self):
        policy = load_threshold_policy(DEFAULT_THRESHOLD_POLICY)
        assert_review_thresholds(policy, 0.90, 0.80, ["cascade"])
        with self.assertRaisesRegex(ValueError, "cascade fast-path threshold"):
            assert_review_thresholds(policy, 0.90, 0.75, ["cascade"])

    def test_downstream_backend_selection_is_explicit(self):
        args = SimpleNamespace(downstream_backend="local-generation",
                               downstream_model="local-model",
                               downstream_revision="rev",
                               downstream_device="cpu",
                               downstream_max_new_tokens=8,
                               downstream_model_path=None)
        original = value_runner.LocalGenerationResponder
        try:
            value_runner.LocalGenerationResponder = lambda *args, **kwargs: ("local", args)
            self.assertEqual(
                value_runner.build_downstream_responder(args, None),
                ("local", ("local-model", "rev", "cpu", None, 8)))
        finally:
            value_runner.LocalGenerationResponder = original
        args.downstream_backend = "deterministic"
        self.assertIsInstance(value_runner.build_downstream_responder(args, None),
                              value_runner.DeterministicEvidenceResponder)
        args.downstream_backend = "other"
        with self.assertRaisesRegex(ValueError, "unknown downstream backend"):
            value_runner.build_downstream_responder(args, None)

    def test_fast_scorer_selection_is_explicit(self):
        args = SimpleNamespace(reflex_model="reflex-model", reflex_revision="rev",
                               device="mps", semif_model="semif-model",
                               semif_revision="rev2", semif_mlx_bits=4,
                               decider_model="decider-model", decider_revision="rev3",
                               fast_systemone_url="http://127.0.0.1:1",
                               fast_systemone_model="fast-model")
        original_reflex = value_runner.ReflexContextScorer
        original_semif = value_runner.SemIfContextScorer
        original_decider = value_runner.DeciderContextScorer
        original_systemone = value_runner.SystemOneHTTPScorer
        try:
            value_runner.ReflexContextScorer = lambda *args, **kwargs: "reflex-scorer"
            value_runner.SemIfContextScorer = lambda *args, **kwargs: "semif-scorer"
            value_runner.DeciderContextScorer = lambda *args, **kwargs: "decider-scorer"
            value_runner.SystemOneHTTPScorer = lambda *args, **kwargs: "systemone-scorer"
            self.assertEqual(value_runner.build_fast_scorer("reflex", args), "reflex-scorer")
            self.assertEqual(value_runner.build_fast_scorer("semif", args), "semif-scorer")
            self.assertEqual(value_runner.build_fast_scorer("decider", args), "decider-scorer")
            self.assertEqual(value_runner.build_fast_scorer("systemone", args), "systemone-scorer")
            with self.assertRaisesRegex(ValueError, "unknown fast scorer"):
                value_runner.build_fast_scorer("other", args)
        finally:
            value_runner.ReflexContextScorer = original_reflex
            value_runner.SemIfContextScorer = original_semif
            value_runner.DeciderContextScorer = original_decider
            value_runner.SystemOneHTTPScorer = original_systemone


if __name__ == "__main__":
    unittest.main()
