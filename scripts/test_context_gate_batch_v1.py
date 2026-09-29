import json
import unittest
import uuid
from unittest.mock import patch

import context_gate_v1 as gate

from context_gate_v1 import MAX_SCORED, parse_segments, scoring_payload, shadow_request


def history_body(n, wire="openai_chat", control=True, history="history"):
    messages = [{"role": "assistant", "content": f"{history} {index}"} for index in range(n)]
    messages.append({"role": "user", "content": "Find the warehouse status for SKU 123."})
    if wire == "openai_responses":
        body = {"model": "test-only", "input": messages}
        if control:
            body["instructions"] = "Never remove constraints"
        return body
    if wire == "anthropic_messages":
        body = {"model": "test-only", "messages": messages, "max_tokens": 32}
        if control:
            body["system"] = "Never remove constraints"
        return body
    body = {"model": "test-only", "messages": messages}
    if control:
        body["messages"] = [{"role": "system", "content": "Never remove constraints"}] + messages
    return body


def encoded(body):
    return (" \n" + json.dumps(body, ensure_ascii=False, indent=2) + "\n").encode()


def candidate_segments(raw, wire="openai_chat"):
    return [segment for segment in parse_segments(raw, wire)
            if segment.role == "assistant" and segment.text is not None]


def notes_for(segments, extra=None):
    notes = {segment.pointer: {"eligible": True} for segment in segments}
    if extra:
        for pointer, note in extra.items():
            notes[pointer] = {**notes.get(pointer, {}), **note}
    return {"segments": notes}


def responder(scores=None, default=0.999, checkpoints=None, event_ids=None, failures=None):
    """Deterministic scorer keyed by candidate_pointer; failures maps call index to mode."""
    calls = []
    failures = failures or {}

    def scorer(payload):
        index = len(calls)
        calls.append(payload)
        action = failures.get(index)
        if action == "throw":
            raise RuntimeError("SECRET_ERROR_TEXT")
        states = payload["states"]
        out = []
        for state in states:
            pointers = json.loads(state["state"])["candidate_pointers"]
            answers = {}
            for ci, pointer in enumerate(pointers):
                probability = default if scores is None else scores.get(pointer, default)
                answers[f"irrelevant_{ci}"] = {"type": "boolean", "probabilities": {
                    "false": 1 - probability, "true": probability}}
            out.append({"id": state["id"], "answers": answers})
        if action == "malformed":
            out = out[:-1]
        result = {"states": out, "checkpoint": checkpoints[index] if checkpoints else {"model": "stub"}}
        if event_ids:
            result["context_gate_usage_event_id"] = event_ids[index]
        return result

    scorer.calls = calls
    return scorer


class BatchGateTest(unittest.TestCase):
    def test_boundaries_batch_cap_and_full_judge_context(self):
        for n in (1, 32, 33, 64, 65, 127):
            with self.subTest(n=n):
                raw = encoded(history_body(n, control=False))
                segments = parse_segments(raw, "openai_chat")
                candidates = candidate_segments(raw)
                expected_batches = (n + MAX_SCORED - 1) // MAX_SCORED
                scorer = responder()
                out, receipt = shadow_request(raw, "openai_chat", notes_for(candidates), scorer)
                self.assertIs(out, raw)
                self.assertEqual(len(scorer.calls), expected_batches)
                self.assertEqual(len(receipt["batches"]), expected_batches)
                for call, meta in zip(scorer.calls, receipt["batches"]):
                    self.assertEqual(len(call["states"]), 1)
                    self.assertEqual(meta["batch_count"], expected_batches)
                    self.assertEqual(meta["candidate_count"],
                                     len(call["states"][0]["questions"]))
                    self.assertEqual(meta["status"], "scored")
                    self.assertEqual([s["id"] for s in call["states"]], ["batch"])
                    rendered = json.loads(call["states"][0]["state"])["conversation"]
                    self.assertEqual(len(rendered), len(segments))
                self.assertEqual(receipt["status"], "scored")
                self.assertEqual(receipt["reason"], "shadow_only")
                self.assertEqual(sum(1 for s in receipt["segments"] if s["suggestion"] == "drop"), n)

    def test_three_wire_formats_and_single_batch_payload_equivalence(self):
        for wire in ("openai_chat", "openai_responses", "anthropic_messages"):
            with self.subTest(wire=wire):
                raw = encoded(history_body(32, wire=wire))
                segments = parse_segments(raw, wire)
                candidates = candidate_segments(raw, wire)
                scorer = responder()
                shadow_request(raw, wire, notes_for(candidates), scorer)
                self.assertEqual(len(scorer.calls), 1)
                self.assertEqual(scorer.calls[0], scoring_payload(segments, candidates))
                self.assertEqual([s["id"] for s in scorer.calls[0]["states"]], ["batch"])
                self.assertEqual(len(scorer.calls[0]["states"][0]["questions"]), 32)
                raw33 = encoded(history_body(33, wire=wire))
                split = responder()
                shadow_request(raw33, wire, notes_for(candidate_segments(raw33, wire)), split)
                self.assertEqual(len(split.calls), 2)
                self.assertEqual([len(c["states"]) for c in split.calls], [1, 1])
                self.assertEqual([len(c["states"][0]["questions"]) for c in split.calls],
                                 [32, 1])

    def test_mapping_and_original_order_not_lexicographic(self):
        n = 33
        raw = encoded(history_body(n, control=False))
        candidates = candidate_segments(raw)
        pointers = [segment.pointer for segment in candidates]
        self.assertNotEqual(pointers, sorted(pointers))
        scores = {pointer: round(0.9901 + index * 0.0001, 6) for index, pointer in enumerate(pointers)}
        _, receipt = shadow_request(raw, "openai_chat", notes_for(candidates), responder(scores=scores))
        self.assertEqual([s["pointer"] for s in receipt["segments"] if s["suggestion"] == "drop"], pointers)
        by_pointer = {s["pointer"]: s["p_irrelevant"] for s in receipt["segments"]}
        for pointer, value in scores.items():
            self.assertAlmostEqual(by_pointer[pointer], value)

    def test_shuffled_result_order_still_maps_by_id(self):
        n = 40
        raw = encoded(history_body(n, control=False))
        candidates = candidate_segments(raw)

        def scorer(payload):
            states = payload["states"][::-1]
            return {"checkpoint": {"model": "stub"}, "states": [
                {"id": state["id"], "answers": {
                    f"irrelevant_{i}": {"type": "boolean", "probabilities": {
                        "false": 0.005, "true": 0.995}}
                    for i in range(len(json.loads(state["state"])["candidate_pointers"]))}}
                for state in states]}

        _, receipt = shadow_request(raw, "openai_chat", notes_for(candidates), scorer)
        by_pointer = {s["pointer"]: s for s in receipt["segments"]}
        self.assertTrue(all(by_pointer[p.pointer]["suggestion"] == "drop" for p in candidates))
        self.assertTrue(all(abs(by_pointer[p.pointer]["p_irrelevant"] - 0.995) < 1e-9 for p in candidates))

    def test_multi_batch_scores_match_single_large_batch_values(self):
        n = 65
        raw = encoded(history_body(n, control=False))
        candidates = candidate_segments(raw)
        scores = {segment.pointer: (0.001 if index % 2 else 0.995)
                  for index, segment in enumerate(candidates)}
        _, receipt = shadow_request(raw, "openai_chat", notes_for(candidates), responder(scores=scores))
        with patch.object(gate, "MAX_SCORED", 128):
            _, reference_receipt = shadow_request(raw, "openai_chat", notes_for(candidates),
                                                  responder(scores=scores))
        self.assertEqual(gate.serialized(receipt["segments"]),
                         gate.serialized(reference_receipt["segments"]))
        reference = scoring_payload(parse_segments(raw, "openai_chat"), candidates)
        reference_pointers = json.loads(reference["states"][0]["state"])["candidate_pointers"]
        self.assertEqual(reference_pointers, [segment.pointer for segment in candidates])
        by_pointer = {s["pointer"]: s["p_irrelevant"] for s in receipt["segments"]}
        for pointer, value in scores.items():
            self.assertAlmostEqual(by_pointer[pointer], value)

    def test_first_middle_final_failed_batch_retains_only_that_batch(self):
        n = 65
        raw = encoded(history_body(n, control=False))
        candidates = candidate_segments(raw)
        notes = notes_for(candidates)
        for position, action in ((0, "throw"), (1, "malformed"), (2, "throw"), (0, "malformed")):
            with self.subTest(position=position, action=action):
                _, receipt = shadow_request(raw, "openai_chat", notes,
                                            responder(failures={position: action}))
                self.assertEqual(receipt["status"], "scored")
                self.assertEqual(receipt["reason"], "partial_batch_fallback")
                failed = [m for m in receipt["batches"] if m["status"] == "failed"]
                self.assertEqual([m["batch_index"] for m in failed], [position])
                failed_pointers = {s.pointer for s in candidates[position * MAX_SCORED:(position + 1) * MAX_SCORED]}
                retained = {s["pointer"] for s in receipt["segments"]
                            if s["role"] == "assistant" and s["suggestion"] == "retain"}
                dropped = {s["pointer"] for s in receipt["segments"]
                           if s["role"] == "assistant" and s["suggestion"] == "drop"}
                self.assertEqual(retained, failed_pointers)
                self.assertEqual(dropped, {s.pointer for s in candidates} - failed_pointers)
                for segment in receipt["segments"]:
                    if segment["pointer"] in failed_pointers:
                        self.assertEqual(segment["reason"], "batch_fallback")
                        self.assertIsNone(segment["p_irrelevant"])

    def test_all_batches_failed_whole_request_bypass(self):
        n = 65
        raw = encoded(history_body(n, control=False))
        candidates = candidate_segments(raw)
        out, receipt = shadow_request(raw, "openai_chat", notes_for(candidates),
                                      responder(failures={0: "throw", 1: "throw", 2: "throw"}))
        self.assertIs(out, raw)
        self.assertEqual(receipt["status"], "bypass")
        self.assertEqual(receipt["reason"], "all_batches_failed")
        self.assertEqual(receipt["actual_removed_segments"], 0)
        self.assertTrue(all(segment["suggestion"] == "retain" for segment in receipt["segments"]))
        self.assertTrue(all(segment["reason"] == "batch_fallback"
                            for segment in receipt["segments"] if segment["role"] == "assistant"))
        self.assertEqual(receipt["scorer_event_ids"], [])

    def test_uncertain_score_retains_that_segment_only(self):
        n = 33
        raw = encoded(history_body(n, control=False))
        candidates = candidate_segments(raw)
        scores = {candidates[0].pointer: 0.5}
        _, receipt = shadow_request(raw, "openai_chat", notes_for(candidates), responder(scores=scores))
        self.assertEqual(receipt["status"], "scored")
        by_pointer = {s["pointer"]: s for s in receipt["segments"]}
        self.assertEqual(by_pointer[candidates[0].pointer]["suggestion"], "retain")
        self.assertEqual(by_pointer[candidates[0].pointer]["reason"], "uncertain_score")
        self.assertTrue(all(by_pointer[c.pointer]["suggestion"] == "drop"
                            for c in candidates[1:]))

    def test_cross_batch_dependencies_and_cycles_close_globally(self):
        n = 65
        raw = encoded(history_body(n, control=False))
        candidates = candidate_segments(raw)
        first, middle = candidates[0].pointer, candidates[40].pointer
        notes = notes_for(candidates, {first: {"depends_on": [middle]},
                                       middle: {"depends_on": [first]}})
        _, receipt = shadow_request(raw, "openai_chat", notes,
                                    responder(scores={first: 0.001, middle: 0.999}))
        by_pointer = {s["pointer"]: s for s in receipt["segments"]}
        self.assertEqual(receipt["status"], "scored")
        self.assertEqual(by_pointer[first]["suggestion"], "retain")
        self.assertEqual(by_pointer[middle]["suggestion"], "retain")
        self.assertEqual(by_pointer[middle]["reason"], "required_dependency")

        last = candidates[60].pointer
        failed_origin = dict(notes)
        failed_origin["segments"] = {**notes["segments"], first: {"eligible": True, "depends_on": [last]}}
        _, receipt = shadow_request(raw, "openai_chat", failed_origin, responder(failures={0: "throw"}))
        by_pointer = {s["pointer"]: s for s in receipt["segments"]}
        self.assertEqual(receipt["reason"], "partial_batch_fallback")
        self.assertEqual(by_pointer[first]["reason"], "batch_fallback")
        self.assertEqual(by_pointer[last]["suggestion"], "retain")
        self.assertEqual(by_pointer[last]["reason"], "required_dependency")

    def test_identity_mismatch_bypasses_whole_request(self):
        n = 33
        raw = encoded(history_body(n, control=False))
        candidates = candidate_segments(raw)
        _, receipt = shadow_request(raw, "openai_chat", notes_for(candidates),
                                    responder(checkpoints=[{"model": "a"}, {"model": "b"}]))
        self.assertEqual(receipt["status"], "bypass")
        self.assertEqual(receipt["reason"], "scorer_identity_mismatch")
        self.assertIsNone(receipt["model_fingerprint"])
        self.assertTrue(all(segment["suggestion"] == "retain" for segment in receipt["segments"]))
        _, consistent = shadow_request(raw, "openai_chat", notes_for(candidates),
                                       responder(checkpoints=[{"model": "a"}, {"model": "a"}]))
        self.assertEqual(consistent["status"], "scored")
        self.assertIsNotNone(consistent["model_fingerprint"])

    def test_uuid_sanitization_privacy_and_legacy_event_id(self):
        first_id, second_id = str(uuid.uuid4()), str(uuid.uuid4())
        raw = encoded(history_body(65, control=False, history="SECRET_HISTORY"))
        candidates = candidate_segments(raw)
        scorer = responder(event_ids=[first_id, second_id, "not-a-uuid"])
        _, receipt = shadow_request(raw, "openai_chat", notes_for(candidates), scorer)
        self.assertEqual(receipt["scorer_event_ids"], [first_id, second_id])
        self.assertEqual([m["scorer_event_id"] for m in receipt["batches"]],
                         [first_id, second_id, None])
        self.assertNotIn("SECRET_HISTORY", json.dumps(receipt))
        self.assertNotIn("SECRET_ERROR_TEXT", json.dumps(receipt))

        single_raw = encoded(history_body(16, control=False))
        single = responder(event_ids=[first_id])
        _, single_receipt = shadow_request(single_raw, "openai_chat",
                                           notes_for(candidate_segments(single_raw)), single)
        self.assertEqual(single_receipt["scorer_event_id"], first_id)
        self.assertEqual(single_receipt["scorer_event_ids"], [first_id])

    def test_raw_bytes_unchanged_and_no_actual_removal(self):
        raw = encoded(history_body(65, control=False))
        candidates = candidate_segments(raw)
        scorer = responder(scores={segment.pointer: 0.001 for segment in candidates})
        output, receipt = shadow_request(raw, "openai_chat", notes_for(candidates), scorer)
        self.assertIs(output, raw)
        self.assertEqual(receipt["request_sha256"], receipt["forwarded_sha256"])
        self.assertEqual(receipt["actual_removed_tokens"], 0)
        self.assertEqual(receipt["actual_removed_segments"], 0)
        self.assertNotIn("warehouse", json.dumps(receipt))
        self.assertNotIn("history", json.dumps(receipt))

    def test_batch_metadata_and_scorer_errors_do_not_leak(self):
        raw = encoded(history_body(33, control=False))
        candidates = candidate_segments(raw)
        _, receipt = shadow_request(raw, "openai_chat", notes_for(candidates),
                                    responder(failures={0: "throw"}))
        self.assertEqual(receipt["batches"][0]["reason"], "scorer_error")
        self.assertEqual(receipt["batches"][1]["reason"], "shadow_only")
        self.assertEqual(receipt["batches"][0]["status"], "failed")
        self.assertNotIn("SECRET", json.dumps(receipt))

    def test_malformed_types_are_batch_local_and_keep_usage_link(self):
        raw = encoded(history_body(33, control=False))
        candidates = candidate_segments(raw)
        event_id = str(uuid.uuid4())
        for bad_id in ([], {}, None):
            with self.subTest(bad_id=bad_id):
                calls = []

                def scorer(payload):
                    result = responder()(payload)
                    calls.append(payload)
                    if len(calls) == 1:
                        result["states"][0]["id"] = bad_id
                        result["context_gate_usage_event_id"] = event_id
                    return result

                _, receipt = shadow_request(raw, "openai_chat", notes_for(candidates), scorer)
                self.assertEqual(receipt["reason"], "partial_batch_fallback")
                self.assertEqual(receipt["scorer_event_ids"], [event_id])
                self.assertEqual(receipt["batches"][0]["reason"], "invalid_score_response")
                self.assertEqual([s["pointer"] for s in receipt["segments"] if s["suggestion"] == "drop"],
                                 [candidates[32].pointer])

    def test_segment_limit_still_bypasses_without_scoring(self):
        raw = encoded(history_body(128, control=False))  # 128 candidates + user exceeds 128 total
        scorer = responder()
        output, receipt = shadow_request(raw, "openai_chat", scorer=scorer)
        self.assertIs(output, raw)
        self.assertEqual(receipt["reason"], "segment_budget_exceeded")
        self.assertEqual(scorer.calls, [])


if __name__ == "__main__":
    unittest.main()
