#!/usr/bin/env python3
"""Tests for the deterministic, model-free safe deduplication arm.

Run with:

    .venv/bin/python -m unittest discover -s scripts -p 'test_safe_dedup_v1.py'

No model, no network, no provider, no socket. Every case is a deterministic in-process
construction or a repository fixture read from disk.
"""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from context_gate_v1 import fingerprint, parse_segments  # noqa: E402
from context_restore_v1 import (  # noqa: E402
    DROP_POINTER, RestoreError, SCHEMA_VERSION as RESTORE_SCHEMA, build_restore_manifest,
    canonical_bytes, restore_request, verify_round_trip,
)
from main_model_gateway_v1 import (  # noqa: E402
    ContextGateGateway, GatewayConfig, RESTORE_MANIFEST_HEADER, Reply, removal_plan,
)
from safe_dedup_v1 import (  # noqa: E402
    DECISION_DROP, DECISION_RETAIN, DROP_REASON, NORMALIZATION_ID, REPO_ROOT,
    build_safe_reduction, measure, plan_safe_dedup, self_test,
)


def body(messages, wire="openai_chat", extra=None):
    payload = {"model": "synthetic-main-model", "messages": messages}
    if extra:
        payload.update(extra)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def eligible(raw, wire_format, pointers=None):
    segments = parse_segments(raw, wire_format)
    if pointers is None:
        pointers = [segment.pointer for segment in segments]
    return {"segments": {pointer: {"eligible": True} for pointer in pointers}}


# --------------------------------------------------------------------------------------
# Deterministic bodies
# --------------------------------------------------------------------------------------

DUPLICATE_STATUS = body([
    {"role": "system", "content": "Synthetic harness."},
    {"role": "assistant", "content": "STATUS: build green"},
    {"role": "assistant", "content": "STATUS: build green"},
    {"role": "assistant", "content": "STATUS: build green"},
    {"role": "assistant", "content": "STATUS: build green"},
    {"role": "assistant", "content": "STATUS: build green"},
    {"role": "user", "content": "Summarize the current status."},
])

DUPLICATE_TOOL_RESULT = body([
    {"role": "system", "content": "Synthetic harness."},
    {"role": "assistant", "content": "First read."},
    {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function",
                                          "function": {"name": "read", "arguments": "{}"}}]},
    {"role": "tool", "content": "identical tool output", "tool_call_id": "call-1"},
    {"role": "assistant", "content": "Second read."},
    {"role": "assistant", "tool_calls": [{"id": "call-2", "type": "function",
                                          "function": {"name": "read", "arguments": "{}"}}]},
    {"role": "tool", "content": "identical tool output", "tool_call_id": "call-2"},
    {"role": "user", "content": "Compare the reads."},
])

DUPLICATE_USER = body([
    {"role": "system", "content": "Synthetic harness."},
    {"role": "user", "content": "Please continue."},
    {"role": "assistant", "content": "Continuing."},
    {"role": "user", "content": "Please continue."},
])

DUPLICATE_SYSTEM = body([
    {"role": "system", "content": "identical system instruction"},
    {"role": "system", "content": "identical system instruction"},
    {"role": "assistant", "content": "Working."},
    {"role": "user", "content": "Go on."},
])

DUPLICATE_UNDECLARED = body([
    {"role": "system", "content": "Synthetic harness."},
    {"role": "assistant", "content": "Undeclared duplicate."},
    {"role": "assistant", "content": "Undeclared duplicate."},
    {"role": "user", "content": "Go on."},
])

DUPLICATE_SAFETY_FLAG = body([
    {"role": "system", "content": "Synthetic harness."},
    {"role": "assistant", "content": "Safety-critical duplicate."},
    {"role": "assistant", "content": "Safety-critical duplicate."},
    {"role": "user", "content": "Go on."},
])

DUPLICATE_WITH_DEPENDENCY = body([
    {"role": "system", "content": "Synthetic harness."},
    {"role": "assistant", "content": "Dependent duplicate."},
    {"role": "assistant", "content": "Dependent duplicate."},
    {"role": "user", "content": "Use the earlier statement."},
])

DUPLICATE_OUTPUT_TEXT = json.dumps({
    "model": "synthetic-main-model",
    "input": [
        {"role": "assistant", "content": [{"type": "output_text", "text": "dup answer"}]},
        {"role": "assistant", "content": [{"type": "output_text", "text": "dup answer"}]},
        {"role": "user", "content": "Go on."},
    ],
}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

DUPLICATE_CACHE_CONTROL = json.dumps({
    "model": "synthetic-main-model",
    "system": "Synthetic harness.",
    "messages": [
        {"role": "assistant", "content": [{"type": "text", "text": "cached duplicate"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "cached duplicate",
                                           "cache_control": {"type": "ephemeral"}}]},
        {"role": "user", "content": "Go on."},
    ],
}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

NEAR_MISSES = body([
    {"role": "system", "content": "Synthetic harness."},
    {"role": "assistant", "content": "Alpha beta"},
    {"role": "assistant", "content": "alpha beta"},
    {"role": "assistant", "content": "Alpha  beta"},
    {"role": "assistant", "content": "Alpha beta "},
    {"role": "user", "content": "Go on."},
])


def _scorer(payload, value=0.999):
    """Deterministic test stub standing in for a model. Never a model call."""
    return {"checkpoint": {"stub": "safe-dedup-test"}, "states": [
        {"id": state["id"], "answers": {"irrelevant": {
            "type": "boolean", "probabilities": {"false": 1 - value, "true": value}}}}
        for state in payload["states"]]}


class StubGateway(ContextGateGateway):
    """The real gateway with only its transport replaced; no socket is opened."""

    def __init__(self, config):
        super().__init__(config)
        self.forwarded = None
        self.forward_count = 0

    def _forward(self, method, path, headers, raw):
        self.forwarded = raw
        self.forward_count += 1
        return Reply(200, [("Content-Type", "application/json")],
                     b'{"usage":{"input_tokens":9,"output_tokens":2}}')


# --------------------------------------------------------------------------------------
# 1. Determinism
# --------------------------------------------------------------------------------------

class DeterminismTest(unittest.TestCase):
    def test_plan_is_identical_across_repeated_calls(self):
        sidecar = eligible(DUPLICATE_STATUS, "openai_chat")
        first = plan_safe_dedup(DUPLICATE_STATUS, "openai_chat", sidecar)
        second = plan_safe_dedup(DUPLICATE_STATUS, "openai_chat", sidecar)
        self.assertEqual(json.dumps(first.as_dict(), sort_keys=True),
                         json.dumps(second.as_dict(), sort_keys=True))
        self.assertEqual(first.drop_pointers, ("/messages/2/content", "/messages/3/content",
                                               "/messages/4/content", "/messages/5/content"))

    def test_measurement_is_identical_across_repeated_calls(self):
        sidecar = eligible(DUPLICATE_STATUS, "openai_chat")
        first = measure_single(DUPLICATE_STATUS, "openai_chat", sidecar)
        second = measure_single(DUPLICATE_STATUS, "openai_chat", sidecar)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))

    def test_declared_normalization_is_identity(self):
        self.assertEqual(NORMALIZATION_ID, "identity-v1-exact-codepoint-equality")
        plan = plan_safe_dedup(NEAR_MISSES, "openai_chat", eligible(NEAR_MISSES, "openai_chat"))
        self.assertEqual(plan.drop_pointers, ())
        self.assertEqual(plan.removed, 0)


# --------------------------------------------------------------------------------------
# 2. Exact-duplicate-only removal
# --------------------------------------------------------------------------------------

class ExactDuplicateTest(unittest.TestCase):
    def test_keeps_first_and_removes_only_later_exact_duplicates(self):
        sidecar = eligible(DUPLICATE_STATUS, "openai_chat")
        plan = plan_safe_dedup(DUPLICATE_STATUS, "openai_chat", sidecar)
        self.assertEqual(plan.drop_pointers, ("/messages/2/content", "/messages/3/content",
                                              "/messages/4/content", "/messages/5/content"))
        first = [d for d in plan.decisions if d.pointer == "/messages/1/content"][0]
        self.assertEqual(first.decision, DECISION_RETAIN)
        self.assertEqual(first.reason, "retained_first_occurrence")
        for decision in plan.decisions:
            if decision.decision == DECISION_DROP:
                self.assertEqual(decision.reason, DROP_REASON)
                self.assertEqual(decision.retained_pointer, "/messages/1/content")

    def test_novel_text_is_never_removed(self):
        raw = body([
            {"role": "system", "content": "Synthetic harness."},
            {"role": "assistant", "content": "unique one"},
            {"role": "assistant", "content": "unique two"},
            {"role": "user", "content": "Go on."},
        ])
        plan = plan_safe_dedup(raw, "openai_chat", eligible(raw, "openai_chat"))
        self.assertEqual(plan.drop_pointers, ())

    def test_only_arm_eligible_segments_are_considered(self):
        plan = plan_safe_dedup(DUPLICATE_UNDECLARED, "openai_chat", {"segments": {}})
        self.assertEqual(plan.arm_eligible_pointers, ())
        self.assertEqual(plan.drop_pointers, ())
        reasons = {d.reason for d in plan.decisions}
        self.assertIn("not_explicitly_eligible", reasons)

    def test_both_whole_message_and_text_part_duplicates_are_detected(self):
        raw = body([
            {"role": "system", "content": "Synthetic harness."},
            {"role": "assistant", "content": "repeated text"},
            {"role": "assistant", "content": [{"type": "text", "text": "repeated text"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "repeated text"}]},
            {"role": "user", "content": "Go on."},
        ])
        plan = plan_safe_dedup(raw, "openai_chat", eligible(raw, "openai_chat"))
        # The whole-message copy and the two text parts are different kinds, so each kind
        # keeps its own first occurrence and only the second text part is removed.
        self.assertEqual(plan.drop_pointers, ("/messages/3/content/0/text",))

    def test_bypass_plan_removes_nothing(self):
        raw = body([
            {"role": "system", "content": "Synthetic harness."},
            {"role": "assistant", "tool_calls": [{"id": None, "type": "function",
                                                  "function": {"name": "x", "arguments": "{}"}}]},
            {"role": "user", "content": "Go on."},
        ])
        plan = plan_safe_dedup(raw, "openai_chat", {"segments": {}})
        self.assertEqual(plan.parse_status, "bypass")
        self.assertEqual(plan.drop_pointers, ())
        self.assertEqual(plan.segments_examined, 0)

    def test_unsupported_wire_format_removes_nothing(self):
        plan = plan_safe_dedup(DUPLICATE_STATUS, "not_a_format", None)
        self.assertEqual(plan.parse_status, "bypass")
        self.assertEqual(plan.drop_pointers, ())


# --------------------------------------------------------------------------------------
# 3. Protected segments are never removed
# --------------------------------------------------------------------------------------

class ProtectedSegmentTest(unittest.TestCase):
    def _assert_no_removal(self, raw, wire_format, sidecar, expected_reason=None):
        plan = plan_safe_dedup(raw, wire_format, sidecar)
        self.assertEqual(plan.drop_pointers, ())
        self.assertEqual(plan.removed, 0)
        if expected_reason is not None:
            self.assertIn(expected_reason, {d.reason for d in plan.decisions})
        for decision in plan.decisions:
            if decision.pointer in plan.protected_pointers:
                self.assertEqual(decision.decision, DECISION_RETAIN)
        return plan

    def test_duplicated_tool_result_is_protected(self):
        self._assert_no_removal(DUPLICATE_TOOL_RESULT, "openai_chat",
                                eligible(DUPLICATE_TOOL_RESULT, "openai_chat"),
                                "protected_structure")

    def test_duplicated_user_message_is_protected(self):
        self._assert_no_removal(DUPLICATE_USER, "openai_chat",
                                eligible(DUPLICATE_USER, "openai_chat"), "protected_structure")

    def test_duplicated_system_message_is_protected(self):
        self._assert_no_removal(DUPLICATE_SYSTEM, "openai_chat",
                                eligible(DUPLICATE_SYSTEM, "openai_chat"), "protected_structure")

    def test_caller_protection_flag_retains_a_duplicate(self):
        sidecar = eligible(DUPLICATE_SAFETY_FLAG, "openai_chat")
        sidecar["segments"]["/messages/2/content"]["safety"] = True
        self._assert_no_removal(DUPLICATE_SAFETY_FLAG, "openai_chat", sidecar, "caller_protected")

    def test_undeclared_segment_is_not_explicitly_eligible(self):
        self._assert_no_removal(DUPLICATE_UNDECLARED, "openai_chat", {"segments": {}},
                                "not_explicitly_eligible")

    def test_dependency_closure_retains_a_duplicate(self):
        sidecar = {"segments": {
            "/messages/1/content": {"eligible": True},
            "/messages/2/content": {"eligible": True},
            "/messages/3/content": {"depends_on": ["/messages/2/content"]}}}
        plan = self._assert_no_removal(DUPLICATE_WITH_DEPENDENCY, "openai_chat", sidecar,
                                       "required_dependency")
        duplicate = [d for d in plan.decisions if d.pointer == "/messages/2/content"][0]
        self.assertEqual(duplicate.reason, "required_dependency")

    def test_no_sidecar_means_no_removal(self):
        plan = plan_safe_dedup(DUPLICATE_STATUS, "openai_chat", None)
        self.assertEqual(plan.drop_pointers, ())

    def test_irreversible_output_text_duplicate_is_retained(self):
        sidecar = eligible(DUPLICATE_OUTPUT_TEXT, "openai_responses")
        plan = plan_safe_dedup(DUPLICATE_OUTPUT_TEXT, "openai_responses", sidecar)
        self.assertEqual(plan.drop_pointers, ())
        self.assertIn("retain_not_reversible", {d.reason for d in plan.decisions})

    def test_irreversible_cache_control_duplicate_is_retained(self):
        sidecar = eligible(DUPLICATE_CACHE_CONTROL, "anthropic_messages")
        plan = plan_safe_dedup(DUPLICATE_CACHE_CONTROL, "anthropic_messages", sidecar)
        self.assertEqual(plan.drop_pointers, ())
        self.assertIn("retain_not_reversible", {d.reason for d in plan.decisions})

    def test_a4_fixtures_have_zero_protected_deletions(self):
        receipt = ReceiptFixture.receipt()
        aggregate = receipt["fixtures"]["a4_tool_history"]["aggregate"]
        self.assertEqual(aggregate["protected_segment_deletions"], 0)
        self.assertEqual(aggregate["tool_linked_segment_deletions"], 0)
        # Measured fact, pinned as a regression tripwire: the frozen A4 bodies contain no
        # exact duplicate of an arm-eligible segment, so the arm removes nothing there.
        self.assertEqual(aggregate["segments_removed"], 0)
        self.assertEqual(aggregate["cases_examined"], 11)


# --------------------------------------------------------------------------------------
# 4. Byte-identical restore round-trip
# --------------------------------------------------------------------------------------

class RestoreRoundTripTest(unittest.TestCase):
    def test_every_removal_restores_byte_identically(self):
        sidecar = eligible(DUPLICATE_STATUS, "openai_chat")
        plan = plan_safe_dedup(DUPLICATE_STATUS, "openai_chat", sidecar)
        reduced, manifest, removed, verify = build_safe_reduction(
            DUPLICATE_STATUS, "openai_chat", plan.drop_pointers)
        restored = restore_request(reduced, manifest, removed)
        self.assertEqual(restored, canonical_bytes(json.loads(DUPLICATE_STATUS)))
        self.assertEqual(restored, DUPLICATE_STATUS)  # canonical input -> byte-identical
        self.assertTrue(verify["restored_bytes_identical"])
        self.assertTrue(verify["reduced_request_sha256_matches"])
        self.assertEqual(verify["dropped_segment_count"], len(plan.drop_pointers))
        self.assertTrue(plan.drop_pointers)
        # The surviving first copy is still present in the reduced request.
        self.assertIn(b"STATUS: build green", reduced)

    def test_round_trip_holds_for_all_wire_formats(self):
        cases = [
            ("openai_chat", body([
                {"role": "system", "content": "s"},
                {"role": "assistant", "content": "same text"},
                {"role": "assistant", "content": "same text"},
                {"role": "user", "content": "go"}])),
            ("anthropic_messages", json.dumps({
                "model": "m", "system": "s", "messages": [
                    {"role": "assistant", "content": [{"type": "text", "text": "same text"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "same text"}]},
                    {"role": "user", "content": "go"}]},
                ensure_ascii=False, separators=(",", ":")).encode("utf-8")),
            ("openai_responses", json.dumps({
                "model": "m", "instructions": "s", "input": [
                    {"role": "assistant", "content": "same text"},
                    {"role": "assistant", "content": "same text"},
                    {"role": "user", "content": "go"}]},
                ensure_ascii=False, separators=(",", ":")).encode("utf-8")),
        ]
        for wire_format, raw in cases:
            with self.subTest(wire_format=wire_format):
                sidecar = eligible(raw, wire_format)
                plan = plan_safe_dedup(raw, wire_format, sidecar)
                self.assertEqual(plan.removed, 1)
                reduced, manifest, removed, verify = build_safe_reduction(
                    raw, wire_format, plan.drop_pointers)
                self.assertEqual(restore_request(reduced, manifest, removed),
                                 canonical_bytes(json.loads(raw)))
                self.assertTrue(verify["restored_bytes_identical"])

    def test_non_canonical_original_restores_to_its_canonical_form(self):
        pretty = (json.dumps(json.loads(DUPLICATE_STATUS), ensure_ascii=False, indent=2)
                  + "\n").encode("utf-8")
        sidecar = eligible(pretty, "openai_chat")
        plan = plan_safe_dedup(pretty, "openai_chat", sidecar)
        self.assertEqual(plan.removed, 4)
        reduced, manifest, removed, verify = build_safe_reduction(
            pretty, "openai_chat", plan.drop_pointers)
        self.assertEqual(restore_request(reduced, manifest, removed),
                         canonical_bytes(json.loads(pretty)))
        self.assertFalse(verify["restored_bytes_identical"])
        self.assertFalse(manifest.canonical_json)

    def test_receipt_reports_a_100_percent_round_trip_rate_where_removals_exist(self):
        summary = ReceiptFixture.receipt()["round_trip_summary"]
        self.assertGreater(summary["reduced_requests_round_tripped"], 0)
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertEqual(summary["reduced_requests_round_tripped"],
                         summary["reduced_requests_restored_byte_identically"])


# --------------------------------------------------------------------------------------
# 5. Fail-open / fail-closed on tampering and mismatch
# --------------------------------------------------------------------------------------

class FailOpenTest(unittest.TestCase):
    def setUp(self):
        sidecar = eligible(DUPLICATE_STATUS, "openai_chat")
        plan = plan_safe_dedup(DUPLICATE_STATUS, "openai_chat", sidecar)
        self.plan = plan
        (self.reduced, self.manifest, self.removed,
         self.verify) = build_safe_reduction(DUPLICATE_STATUS, "openai_chat", plan.drop_pointers)

    def assertRejected(self, callable_):
        with self.assertRaises(RestoreError):
            callable_()

    def test_tampered_manifest_hash_is_rejected(self):
        tampered = self.manifest.as_dict()
        tampered["restored_bytes_sha256"] = "0" * 64
        self.assertRejected(lambda: restore_request(self.reduced, tampered, self.removed))

    def test_tampered_manifest_count_is_rejected(self):
        tampered = self.manifest.as_dict()
        tampered["dropped_segment_count"] = 99
        self.assertRejected(lambda: restore_request(self.reduced, tampered, self.removed))

    def test_tampered_reduced_bytes_are_rejected(self):
        tampered = self.reduced[:-1] + b" "
        self.assertRejected(lambda: restore_request(tampered, self.manifest, self.removed))

    def test_tampered_removed_segment_is_rejected(self):
        pointer = self.manifest.records[0].pointer
        segments = dict(self.removed)
        segments[pointer] = "a different text"
        self.assertRejected(lambda: restore_request(self.reduced, self.manifest, segments))

    def test_missing_removed_segment_is_rejected(self):
        pointer = self.manifest.records[0].pointer
        segments = dict(self.removed)
        segments.pop(pointer)
        self.assertRejected(lambda: restore_request(self.reduced, self.manifest, segments))

    def test_extra_removed_segment_is_rejected(self):
        segments = {**self.removed, "/messages/99/content": "extra"}
        self.assertRejected(lambda: restore_request(self.reduced, self.manifest, segments))

    def test_arm_retains_a_candidate_whose_restore_fails(self):
        # An openai_responses output_text part cannot be reconstructed byte-identically by
        # the restore module, so the arm must retain it rather than send an unreversible
        # reduction.
        sidecar = eligible(DUPLICATE_OUTPUT_TEXT, "openai_responses")
        plan = plan_safe_dedup(DUPLICATE_OUTPUT_TEXT, "openai_responses", sidecar)
        self.assertEqual(plan.drop_pointers, ())
        self.assertIn("retain_not_reversible", {d.reason for d in plan.decisions})

    def test_existing_gateway_refuses_the_arm_drop_reason(self):
        receipt = {"segments": [
            {"pointer": pointer, "role": "assistant", "suggestion": "drop", "reason": DROP_REASON}
            for pointer in self.plan.drop_pointers]}
        drops, error = removal_plan(receipt)
        self.assertEqual(drops, [])
        self.assertEqual(error, "protected_segment_in_removal_set")

    def test_gateway_fails_open_on_an_irreversible_reduction(self):
        # The gateway's own active path, with only its transport stubbed. The reduction is
        # well-formed and passes the gateway's plan check, but the restore round-trip fails,
        # so the gateway must forward the ORIGINAL bytes with no restore header.
        raw = json.dumps({
            "model": "m",
            "input": [
                {"role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
                {"role": "user", "content": "go"}]},
            ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        sidecar = {"segments": {"/input/0/content/0/text": {"eligible": True}}}
        with tempfile.TemporaryDirectory() as directory:
            receipt_log = Path(directory) / "receipts.jsonl"
            gateway = StubGateway(GatewayConfig(
                upstream_base_url="http://127.0.0.1:9", mode="active", scorer=_scorer,
                sidecar=sidecar, receipt_log=receipt_log))
            reply = gateway.handle("POST", "/v1/responses", [], raw)
            receipt = json.loads(receipt_log.read_text(encoding="utf-8").strip())
        self.assertEqual(reply.status, 200)
        self.assertEqual(receipt["forward_reason"], "reduction_error")
        self.assertTrue(receipt["forwarded_unchanged"])
        self.assertEqual(receipt["forwarded_sha256"], receipt["request_sha256"])
        self.assertEqual(gateway.forwarded, raw)
        self.assertFalse(any(name.lower() == RESTORE_MANIFEST_HEADER for name, _ in reply.headers))
        self.assertEqual(receipt["token_accounting"]["savings"]["claim"], "none")
        self.assertEqual(receipt["token_accounting"]["savings"]["basis"], "no_reduction_sent")

    def test_gateway_still_reduces_a_reversible_model_reason_plan(self):
        # Control: the same gateway path DOES apply and describe a reduction whose segment
        # the restore module can reconstruct. This proves the fail-open above is specific
        # to the mismatch, not a blanket refusal.
        raw = json.dumps({
            "model": "m", "system": "s",
            "messages": [
                {"role": "assistant", "content": [{"type": "text", "text": "redundant note"}]},
                {"role": "user", "content": "go"}]},
            ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        sidecar = {"segments": {"/messages/0/content/0/text": {"eligible": True}}}
        with tempfile.TemporaryDirectory() as directory:
            receipt_log = Path(directory) / "receipts.jsonl"
            gateway = StubGateway(GatewayConfig(
                upstream_base_url="http://127.0.0.1:9", mode="active", scorer=_scorer,
                sidecar=sidecar, receipt_log=receipt_log))
            reply = gateway.handle("POST", "/v1/messages", [], raw)
            receipt = json.loads(receipt_log.read_text(encoding="utf-8").strip())
            headers = dict((name.lower(), value) for name, value in reply.headers)
        self.assertEqual(receipt["forward_reason"], "active_reduced")
        self.assertFalse(receipt["forwarded_unchanged"])
        self.assertIn(RESTORE_MANIFEST_HEADER, headers)
        manifest = json.loads(headers[RESTORE_MANIFEST_HEADER])
        self.assertEqual(manifest["schema_version"], RESTORE_SCHEMA)
        self.assertTrue(manifest["applied"])
        self.assertEqual(manifest["dropped_segment_count"], 1)
        self.assertEqual(fingerprint(gateway.forwarded), receipt["forwarded_sha256"])
        self.assertNotEqual(gateway.forwarded, raw)

    def test_self_test_reports_ok(self):
        report = self_test()
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["passed"], report["total"])


# --------------------------------------------------------------------------------------
# 6. Contract validity
# --------------------------------------------------------------------------------------

class ContractValidityTest(unittest.TestCase):
    def setUp(self):
        sidecar = eligible(DUPLICATE_STATUS, "openai_chat")
        plan = plan_safe_dedup(DUPLICATE_STATUS, "openai_chat", sidecar)
        (self.reduced, self.manifest, self.removed,
         self.verify) = build_safe_reduction(DUPLICATE_STATUS, "openai_chat", plan.drop_pointers)

    def test_manifest_has_exactly_the_declared_fields(self):
        data = self.manifest.as_dict()
        self.assertEqual(set(data), {
            "schema_version", "wire_format", "message_key", "canonical_json",
            "original_request_sha256", "restored_bytes_sha256", "reduced_request_sha256",
            "applied", "dropped_segment_count", "records"})
        self.assertEqual(self.manifest.schema_version, RESTORE_SCHEMA)
        self.assertTrue(self.manifest.applied)
        self.assertEqual(self.manifest.dropped_segment_count, len(self.manifest.records))
        self.assertEqual(len(self.manifest.records), len(self.removed))

    def test_every_record_pointer_is_in_the_restore_grammar(self):
        for record in self.manifest.records:
            self.assertIsNotNone(DROP_POINTER.match(record.pointer))
            self.assertIn(record.pointer, self.removed)

    def test_reduced_request_still_parses_with_the_core(self):
        self.assertTrue(parse_segments(self.reduced, "openai_chat"))

    def test_plan_and_receipt_are_content_free(self):
        plan = plan_safe_dedup(DUPLICATE_STATUS, "openai_chat",
                               eligible(DUPLICATE_STATUS, "openai_chat"))
        rendered = json.dumps(plan.as_dict(), ensure_ascii=False)
        self.assertNotIn("STATUS: build green", rendered)
        self.assertNotIn("Synthetic harness", rendered)
        receipt = json.dumps(ReceiptFixture.receipt(), ensure_ascii=False)
        self.assertNotIn("STATUS: build green", receipt)
        self.assertNotIn("identical tool output", receipt)

    def test_normalization_and_reason_are_declared(self):
        data = plan_safe_dedup(DUPLICATE_STATUS, "openai_chat",
                               eligible(DUPLICATE_STATUS, "openai_chat")).as_dict()
        self.assertEqual(data["normalization_id"], NORMALIZATION_ID)
        for decision in data["decisions"]:
            if decision["decision"] == DECISION_DROP:
                self.assertEqual(decision["reason"], DROP_REASON)

    def test_measurement_receipt_is_deterministic_and_complete(self):
        first = json.dumps(ReceiptFixture.receipt(), sort_keys=True)
        second = json.dumps(measure(), sort_keys=True)
        self.assertEqual(first, second)
        receipt = ReceiptFixture.receipt()
        self.assertEqual(receipt["no_model_calls"], True)
        self.assertEqual(receipt["no_network_calls"], True)
        self.assertEqual(receipt["provider_calls_made"], 0)
        self.assertEqual(receipt["bytes_sent_to_provider"], 0)
        self.assertEqual(receipt["accounting"]["net_sent_token_saving"], 0)
        self.assertEqual(receipt["accounting"]["gateway_would_report"]["claim"], "none")
        self.assertIsNone(receipt["accounting"]["actual_provider_savings"])


class ReceiptFixture:
    """One shared, deterministic measurement for the whole test module."""

    _receipt = None

    @classmethod
    def receipt(cls):
        if cls._receipt is None:
            cls._receipt = measure()
        return cls._receipt


def measure_single(raw, wire_format, sidecar):
    from safe_dedup_v1 import load_tokenizer, measure_request
    if not hasattr(measure_single, "_tokenizer"):
        measure_single._tokenizer = load_tokenizer()
    return measure_request(raw, wire_format, sidecar, measure_single._tokenizer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
