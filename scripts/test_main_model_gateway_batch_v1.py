"""T8 loopback integration: batching, failure retention and bounded restore headers."""
import json
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import main_model_gateway_v1 as gateway
from context_restore_v1 import canonical_bytes, restore_request
from scorer_adapters_v1 import FailureRecordingScorer
from test_main_model_gateway_v1 import (
    GatewayTestCase, http_post_with_headers, read_receipts, score_result,
)


def many_request(count=65, wire="openai_chat"):
    key = "input" if wire == "openai_responses" else "messages"
    body = {"model": "synthetic-only", key: [
        {"role": "assistant", "content": f"Synthetic archived note {i}."}
        for i in range(count)] + [{"role": "user", "content": "Keep my final request intact."}]}
    if wire == "anthropic_messages":
        body["max_tokens"] = 64
    notes = {f"/{key}/{i}/content": {"eligible": True} for i in range(count)}
    return body, {"segments": notes}


class RestoreHeaderBudgetTests(unittest.TestCase):
    def test_full_header_line_exact_boundary_and_one_byte_over(self):
        manifest = SimpleNamespace(to_json=lambda: '{"content_free":true}')
        value = manifest.to_json()
        length = len(f"{gateway.RESTORE_MANIFEST_HEADER}: {value}\r\n".encode("ascii"))
        with patch.object(gateway, "MAX_RESTORE_HEADER_BYTES", length):
            self.assertEqual(gateway.bounded_restore_header(manifest), value)
        with patch.object(gateway, "MAX_RESTORE_HEADER_BYTES", length - 1):
            with self.assertRaises(gateway.RestoreHeaderTooLarge):
                gateway.bounded_restore_header(manifest)

    def test_success_does_not_erase_earlier_batch_failure(self):
        attempts = iter([TimeoutError("PRIVATE"), {}, RuntimeError("PRIVATE"), {}])

        def scorer(payload):
            value = next(attempts)
            if isinstance(value, Exception):
                raise value
            return value

        recorder = FailureRecordingScorer(scorer)
        with self.assertRaises(TimeoutError):
            recorder({})
        recorder({})
        self.assertEqual(recorder.last_failure, "timeout")
        with self.assertRaises(RuntimeError):
            recorder({})
        recorder({})
        self.assertEqual(recorder.last_failure, "exception")
        self.assertIsNone(FailureRecordingScorer(scorer).last_failure)


class BatchGatewayTests(GatewayTestCase):
    def test_shadow_scores_65_and_forwards_exact_original_bytes(self):
        body, sidecar = many_request()
        raw = (" \n" + json.dumps(body, indent=2) + "\n").encode()
        calls = []

        def scorer(payload):
            calls.append(len(payload["states"][0]["questions"]))
            return score_result(payload)

        config, host, port = self.start_gateway(scorer=scorer, sidecar=sidecar)
        status, _, headers = http_post_with_headers(host, port, "/v1/chat/completions", raw)
        self.assertEqual(status, 200)
        self.assertEqual(calls, [32, 32, 1])
        self.assertEqual(self.upstream.requests[-1]["body"], raw)
        receipt = read_receipts(config.receipt_log)[-1]
        self.assertEqual(len(receipt["removal_plan"]["proposed_pointers"]), 65)
        self.assertFalse(receipt["removal_plan"]["applied"])
        self.assertNotIn(gateway.RESTORE_MANIFEST_HEADER, headers)
        self.assertNotIn("Synthetic archived note", json.dumps(receipt))

    def test_large_restore_header_fails_open_before_forwarding(self):
        body, sidecar = many_request()
        raw = canonical_bytes(body)
        config, host, port = self.start_gateway(mode="active", scorer=score_result, sidecar=sidecar)
        status, _, headers = http_post_with_headers(host, port, "/v1/chat/completions", raw)
        self.assertEqual(status, 200)
        self.assertEqual(self.upstream.requests[-1]["body"], raw)
        self.assertNotIn(gateway.RESTORE_MANIFEST_HEADER, headers)
        receipt = read_receipts(config.receipt_log)[-1]
        self.assertEqual(receipt["forward_reason"], "restore_manifest_header_too_large")
        self.assertEqual(len(receipt["removal_plan"]["proposed_pointers"]), 65)
        self.assertEqual(receipt["removal_plan"]["applied_pointers"], [])
        self.assertFalse(receipt["removal_plan"]["applied"])
        self.assertTrue(receipt["forwarded_unchanged"])

    def test_small_cross_batch_plan_roundtrips_in_all_three_formats(self):
        paths = {"openai_chat": "/v1/chat/completions", "openai_responses": "/v1/responses",
                 "anthropic_messages": "/v1/messages"}
        for wire, path in paths.items():
            with self.subTest(wire=wire):
                body, sidecar = many_request(33, wire)
                raw = canonical_bytes(body)
                key = "input" if wire == "openai_responses" else "messages"
                drops = [f"/{key}/0/content", f"/{key}/32/content"]

                def scorer(payload):
                    pointers = json.loads(payload["states"][0]["state"])["candidate_pointers"]
                    probabilities = [0.999 if p in drops else 0.001 for p in pointers]
                    return score_result(payload, probabilities)

                config, host, port = self.start_gateway(mode="active", scorer=scorer, sidecar=sidecar)
                status, _, headers = http_post_with_headers(host, port, path, raw)
                self.assertEqual(status, 200)
                receipt = read_receipts(config.receipt_log)[-1]
                self.assertEqual(receipt["removal_plan"]["applied_pointers"], drops)
                value = headers[gateway.RESTORE_MANIFEST_HEADER]
                self.assertLessEqual(len(f"{gateway.RESTORE_MANIFEST_HEADER}: {value}\r\n".encode()),
                                     gateway.MAX_RESTORE_HEADER_BYTES)
                removed = {drops[0]: body[key][0]["content"], drops[1]: body[key][32]["content"]}
                self.assertEqual(restore_request(self.upstream.requests[-1]["body"],
                                                json.loads(value), removed), raw)

    def test_failed_first_batch_retained_and_later_success_recorded(self):
        body, sidecar = many_request(33)
        raw = canonical_bytes(body)
        calls = []

        def scorer(payload):
            calls.append(payload)
            if len(calls) == 1:
                raise RuntimeError("PRIVATE ERROR TEXT")
            return score_result(payload)

        config, host, port = self.start_gateway(mode="active", scorer=scorer, sidecar=sidecar)
        status, _, headers = http_post_with_headers(host, port, "/v1/chat/completions", raw)
        self.assertEqual(status, 200)
        receipt = read_receipts(config.receipt_log)[-1]
        self.assertEqual(receipt["scorer_failure_kind"], "exception")
        self.assertEqual(receipt["removal_plan"]["applied_pointers"], ["/messages/32/content"])
        sent = json.loads(self.upstream.requests[-1]["body"])
        self.assertEqual(sent["messages"][:32], body["messages"][:32])
        self.assertEqual(receipt["gate_receipt"]["reason"], "partial_batch_fallback")
        self.assertNotIn("PRIVATE", json.dumps(receipt))
        self.assertIn(gateway.RESTORE_MANIFEST_HEADER, headers)

    def test_timeout_batch_retained_and_later_success_does_not_hide_timeout(self):
        body, sidecar = many_request(33)
        raw = canonical_bytes(body)
        release = threading.Event()
        self.addCleanup(release.set)
        calls = []

        def scorer(payload):
            calls.append(payload)
            if len(calls) == 1:
                release.wait(1)
            return score_result(payload)

        config, host, port = self.start_gateway(mode="active", scorer=scorer, sidecar=sidecar,
                                                score_timeout=0.05)
        status, _, _ = http_post_with_headers(host, port, "/v1/chat/completions", raw)
        release.set()
        self.assertEqual(status, 200)
        receipt = read_receipts(config.receipt_log)[-1]
        self.assertEqual(receipt["scorer_failure_kind"], "timeout")
        self.assertEqual(receipt["removal_plan"]["applied_pointers"], ["/messages/32/content"])

    def test_failed_batch_dependency_protects_later_batch(self):
        body, sidecar = many_request(33)
        raw = canonical_bytes(body)
        sidecar["segments"]["/messages/0/content"]["depends_on"] = ["/messages/32/content"]
        calls = []

        def scorer(payload):
            calls.append(payload)
            if len(calls) == 1:
                return {"states": []}  # malformed, not partially accepted
            return score_result(payload)

        config, host, port = self.start_gateway(mode="active", scorer=scorer, sidecar=sidecar)
        _, _, headers = http_post_with_headers(host, port, "/v1/chat/completions", raw)
        self.assertEqual(self.upstream.requests[-1]["body"], raw)
        self.assertNotIn(gateway.RESTORE_MANIFEST_HEADER, headers)
        receipt = read_receipts(config.receipt_log)[-1]
        last = next(s for s in receipt["gate_receipt"]["segments"] if s["pointer"] == "/messages/32/content")
        self.assertEqual(last["reason"], "required_dependency")
        self.assertEqual(receipt["removal_plan"]["proposed_pointers"], [])


if __name__ == "__main__":
    unittest.main()
