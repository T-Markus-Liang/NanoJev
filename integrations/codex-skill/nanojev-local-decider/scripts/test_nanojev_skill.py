#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("nanojev_skill.py")
spec = importlib.util.spec_from_file_location("nanojev_skill", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class NanoJevSkillTest(unittest.TestCase):
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
