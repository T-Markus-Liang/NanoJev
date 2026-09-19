#!/usr/bin/env python3
import importlib.util
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


if __name__ == "__main__":
    unittest.main()
