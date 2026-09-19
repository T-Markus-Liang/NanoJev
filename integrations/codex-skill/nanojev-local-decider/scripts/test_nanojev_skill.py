#!/usr/bin/env python3
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
import json
from unittest.mock import patch
from types import SimpleNamespace


SCRIPT = Path(__file__).with_name("nanojev_skill.py")
spec = importlib.util.spec_from_file_location("nanojev_skill", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class NanoJevSkillTest(unittest.TestCase):
    def test_remote_url_rejected_before_network(self):
        for url in ("https://example.com", "http://127.0.0.1.evil.test", "file:///tmp/input",
                    "http://user:secret@127.0.0.1:8765", "http://127.0.0.1:8765?token=x"):
            with self.subTest(url=url), patch.object(module.urllib.request, "build_opener") as opener:
                with self.assertRaises(ValueError): module.json_request(url)
                opener.assert_not_called()

    def test_redirect_cannot_escape_loopback(self):
        with self.assertRaises(RuntimeError):
            module.NoRedirect().redirect_request(None, None, 302, "Found", {}, "https://example.com")

    def test_bad_or_missing_probabilities_fail_closed(self):
        for probs in ({}, {"x": float("nan")}, {"x": float("inf")}, {"x": True}, {"x": 0.3}):
            with self.assertRaises(ValueError): module.confidence({"probabilities": probs})
        with self.assertRaises(ValueError):
            module.apply_response_compatibility({"states": []}, {("s", "q"): {}})

    def test_lifecycle_builds_local_advisory_for_each_phase(self):
        for stage in module.STAGES:
            args = SimpleNamespace(stage=stage, state="Tests have not run.",
                candidates='{"test":"Run unit tests", "skip":"Skip checks"}', abstain_below=0.9)
            with patch.object(module, "command_decide", return_value=0) as decide:
                self.assertEqual(module.command_lifecycle(args), 0)
                decide.assert_called_once_with(args)
            payload = json.loads(args.json)
            self.assertEqual(args.task_tag, stage)
            self.assertEqual(payload["states"][0]["questions"]["next_check"]["abstain_below"], 0.9)

    def test_wrong_checkpoint_and_remote_calls_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(input=None, json='{"states":[{"id":"s","state":"facts", "questions":{"q":{"type":"boolean","instructions":"True?"}}}]}',
                url="http://127.0.0.1:8876", project_root=Path(directory), checkpoint=Path(directory),
                runtime_dir=Path(directory), timeout=2, source="test", task_tag="test", log=Path(directory)/"usage.jsonl")
            for result in ({"execution":{"network_model_calls":1}},
                           {"execution":{"network_model_calls":0},"checkpoint":{"directory":"/wrong"}}):
                with patch.object(module,"ensure_service",return_value={}), patch.object(module,"json_request",return_value=result):
                    with self.assertRaises(RuntimeError): module.command_decide(args)
                self.assertFalse(args.log.exists())

    def test_noul_and_abstain_gate_are_normalized(self):
        payload = {"states": [{"id": "s1", "state": {"x": 1}, "questions": {
            "safe": {"type": "noul", "instructions": "Is this safe?", "abstain_below": 0.8},
        }}]}
        normalized, gates = module.normalize_payload(payload)
        self.assertEqual(normalized["states"][0]["questions"]["safe"]["type"], "boolean")
        self.assertNotIn("abstain_below", normalized["states"][0]["questions"]["safe"])
        self.assertEqual(gates[("s1", "safe")]["type"], "noul")

    def test_response_maps_noul_and_abstains(self):
        result = {"states": [{"id": "s1", "answers": {"safe": {
            "type": "boolean", "probabilities": {"false": 0.55, "true": 0.45}, "p_true": 0.45,
        }}}]}
        output = module.apply_response_compatibility(result, {("s1", "safe"): {"type": "noul", "abstain_below": 0.8}})
        answer = output["states"][0]["answers"]["safe"]
        self.assertEqual(answer["type"], "noul")
        self.assertTrue(answer["abstained"])
        self.assertIsNone(answer["value"])
        self.assertEqual(output["decision_summary"]["abstained"], 1)

    def test_log_summary_joins_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "usage.jsonl"
            event = {"event_type": "decision", "event_id": "d1", "request": {"question_count": 1, "question_types": ["choice"]}, "result": {"latency_ms": 10, "confidence_mean": 0.9, "abstained_count": 0}}
            module.append_event(event, path)
            module.append_event({"event_type": "feedback", "decision_event_id": "d1", "label": "correct"}, path)
            summary = module.summarize_log(path)
        self.assertEqual(summary["decision_events"], 1)
        self.assertEqual(summary["feedback_counts"], {"correct": 1})
        self.assertEqual(summary["feedback_coverage"], 1.0)

    # ---- domain/scope guard (nanojev-scope-guard-v1) -----------------------------------------

    def _request(self, instructions, question_type="boolean", criteria=None):
        question = {"type": question_type, "instructions": instructions, "abstain_below": 0.9}
        if criteria is not None:
            question["criteria"] = criteria
        return {"states": [{"id": "s", "state": "Release candidate built.", "questions": {"q": question}}]}

    def _gate_for(self, instructions, question_type="boolean", criteria=None):
        _, gates = module.normalize_payload(self._request(instructions, question_type, criteria))
        return gates[("s", "q")]

    def test_scope_guard_fires_for_authorization_style_question(self):
        gate = self._gate_for("Is it safe to remove production context without a validated gate?")
        self.assertTrue(gate["scope"]["out_of_scope"])
        self.assertEqual(gate["scope"]["reason"], module.OUT_OF_SCOPE_REASON)
        self.assertIn("safety_judgment", gate["scope"]["patterns"])
        self.assertIn("gate_or_approval_bypass", gate["scope"]["patterns"])
        self.assertIn("context_or_artifact_removal", gate["scope"]["patterns"])

    def test_scope_guard_covers_every_documented_trigger_family(self):
        cases = (
            "Is it safe to drop archived context?",
            "Should we deploy this build to production?",
            "Should we approve this commit?",
            "Does this change need a test?",
            "Can we replace the approval gate with a code comment?",
            "Is it OK to bypass review for this patch?",
            "Should we commit without running the checks?",
        )
        for instructions in cases:
            with self.subTest(instructions=instructions):
                self.assertTrue(self._gate_for(instructions)["scope"]["out_of_scope"])

    def test_scope_guard_leaves_in_domain_bounded_question_advisory(self):
        criteria = {"url": "Run the URL guard module first", "redirect": "Run the redirect module first"}
        gate = self._gate_for("Which module should run first?", "choice", criteria)
        self.assertFalse(gate["scope"]["out_of_scope"])
        self.assertIsNone(gate["scope"]["reason"])

        result = {"states": [{"id": "s", "answers": {"q": {
            "type": "choice", "choice": "url", "probabilities": {"url": 0.95, "redirect": 0.05},
        }}}]}
        output = module.apply_response_compatibility(result, {("s", "q"): gate})
        answer = output["states"][0]["answers"]["q"]
        self.assertEqual(answer["status"], module.ANSWER_STATUS_ADVISORY)
        self.assertFalse(answer["out_of_scope"])
        self.assertEqual(answer["choice"], "url")
        self.assertFalse(answer["abstained"])
        self.assertFalse(output["scope_assessment"]["out_of_scope"])

    def test_out_of_scope_keeps_probabilities_and_never_presents_the_answer(self):
        gate = self._gate_for("Should we deploy this build without a validated gate?")
        result = {"states": [{"id": "s", "answers": {"q": {
            "type": "boolean", "value": True, "p_true": 0.99,
            "probabilities": {"false": 0.01, "true": 0.99},
        }}}]}
        output = module.apply_response_compatibility(result, {("s", "q"): gate})
        answer = output["states"][0]["answers"]["q"]
        # Presented status is suppressed even though raw confidence clears the 0.9 gate.
        self.assertEqual(answer["status"], module.ANSWER_STATUS_OUT_OF_SCOPE)
        self.assertTrue(answer["out_of_scope"])
        self.assertIsNone(answer["value"])
        self.assertEqual(answer["suggested_value"], True)
        self.assertFalse(answer["abstained"])
        # Measurement is untouched.
        self.assertEqual(answer["probabilities"], {"false": 0.01, "true": 0.99})
        self.assertAlmostEqual(answer["confidence"], 0.99)
        self.assertFalse(answer["authorizes_execution"])
        self.assertTrue(output["scope_assessment"]["out_of_scope"])
        self.assertEqual(output["decision_summary"]["out_of_scope"], 1)
        self.assertEqual(output["decision_summary"]["answered_advisory"], 0)
        self.assertFalse(output["authorizes_execution"])

    def test_scope_guard_is_deterministic(self):
        request = self._request("Should we commit this change without a test?")
        first = module.normalize_payload(request)[1][("s", "q")]["scope"]
        second = module.normalize_payload(request)[1][("s", "q")]["scope"]
        self.assertEqual(first, second)

    def test_lifecycle_authorization_state_is_flagged_in_scope_guard(self):
        args = SimpleNamespace(
            stage="deployment",
            state="Build passed. Is it safe to deploy without a validated gate?",
            candidates='{"gate":"Run the readiness gate","now":"Deploy now"}', abstain_below=0.9)
        with patch.object(module, "command_decide", return_value=0):
            self.assertEqual(module.command_lifecycle(args), 0)
        _, gates = module.normalize_payload(json.loads(args.json))
        self.assertTrue(gates[("deployment", "next_check")]["scope"]["out_of_scope"])

        plain = SimpleNamespace(
            stage="testing", state="Tests have not run.",
            candidates='{"unit":"Run unit tests","benchmark":"Benchmark latency"}', abstain_below=0.9)
        with patch.object(module, "command_decide", return_value=0):
            self.assertEqual(module.command_lifecycle(plain), 0)
        _, gates = module.normalize_payload(json.loads(plain.json))
        self.assertFalse(gates[("testing", "next_check")]["scope"]["out_of_scope"])

    def test_decide_output_surfaces_operating_envelope_and_scope(self):
        request = json.dumps(self._request(
            "Is it safe to remove production context without a validated gate?"))
        response = {"execution": {"network_model_calls": 0, "device": "cpu", "precision": "fp32"},
                    "checkpoint": {}, "states": [{"id": "s", "answers": {"q": {
                        "type": "boolean", "value": True, "p_true": 0.99,
                        "probabilities": {"false": 0.01, "true": 0.99}}}}]}
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                input=None, json=request, url="http://127.0.0.1:8876", project_root=Path(directory),
                checkpoint=Path(directory), runtime_dir=Path(directory), timeout=2, source="test",
                task_tag="test", log=Path(directory) / "usage.jsonl")
            response["checkpoint"] = {"directory": str(Path(directory).resolve())}
            buffer = io.StringIO()
            with patch.object(module, "ensure_service", return_value={}), \
                    patch.object(module, "json_request", return_value=response), \
                    contextlib.redirect_stdout(buffer):
                self.assertEqual(module.command_decide(args), 0)
            output = json.loads(buffer.getvalue())
            record = json.loads(args.log.read_text(encoding="utf-8").strip())

        envelope = output["operating_envelope"]
        self.assertEqual(envelope["checkpoint_default_abstain_threshold"], 0.9)
        self.assertIn("abstains on essentially all out-of-domain engineering questions",
                      envelope["measured_out_of_domain_behaviour"])
        self.assertIn("13/13", envelope["measured_at_default_threshold"])
        self.assertIn("docs/NANOJEV_SKILL_READINESS_V1.md", envelope["evidence"])
        self.assertFalse(envelope["authorizes_execution"])
        self.assertFalse(output["authorizes_execution"])
        self.assertEqual(output["scope_assessment"]["out_of_scope_count"], 1)
        self.assertFalse(record["authorizes_execution"])
        self.assertEqual(record["result"]["out_of_scope_count"], 1)
        self.assertIn("docs/NANOJEV_SKILL_READINESS_V1.md", record["operating_envelope"]["evidence"])

    def test_summary_surfaces_envelope_and_out_of_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "usage.jsonl"
            module.append_event({
                "event_type": "decision", "event_id": "d1",
                "request": {"question_count": 2, "question_types": ["boolean", "choice"]},
                "result": {"latency_ms": 10, "confidence_mean": 0.9, "abstained_count": 1,
                           "out_of_scope_count": 1},
            }, path)
            summary = module.summarize_log(path)
        self.assertEqual(summary["out_of_scope_questions"], 1)
        self.assertEqual(summary["out_of_scope_rate"], 0.5)
        self.assertEqual(summary["operating_envelope"]["checkpoint_default_abstain_threshold"], 0.9)
        self.assertFalse(summary["authorizes_execution"])

    def test_default_threshold_is_unchanged_and_documented(self):
        self.assertEqual(module.DEFAULT_ABSTAIN_THRESHOLD, 0.9)
        self.assertEqual(module.operating_envelope()["checkpoint_default_abstain_threshold"], 0.9)
        args = module.build_parser().parse_args([
            "lifecycle", "--stage", "testing", "--state", "facts",
            "--candidates", '{"a":"x","b":"y"}'])
        self.assertEqual(args.abstain_below, 0.9)


if __name__ == "__main__":
    unittest.main()
