#!/usr/bin/env python3
"""Gateway tests. A loopback fake upstream is started inside each test.

No real provider is contacted, no API key is required, and no external network is used.
Run with::

    .venv/bin/python -m unittest discover -s scripts -p 'test_main_model_gateway*.py'
"""

import base64
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import main_model_gateway_v1 as gateway_module
from main_model_gateway_v1 import (
    BASELINE_HEADER, KILL_SWITCH_ENV, RESTORE_MANIFEST_HEADER, SIDECAR_HEADER, GatewayConfig,
    ContextGateGateway, build_reduced_request, make_server, removal_plan, _kill_switch_from_env,
)
from context_restore_v1 import (
    RestoreError, build_restore_manifest, canonical_bytes, restore_request, verify_round_trip,
)
from scorer_adapters_v1 import (
    DeadlineScorer, FailureRecordingScorer, InProcessScorer, LayaEncoderScorerAdapter,
    NanoJevHTTPScorer, ScorerError, ScorerTimeout, build_scorer,
)


SYSTEM_TEXT = "Preserve all user constraints and never drop instructions."
USER_TEXT = "Find the warehouse status for SKU 123 and cite the source."
HISTORY_TEXT = "An unrelated archived weather note from a previous task."
SECOND_HISTORY = "A second unrelated historical aside about lunch."


def score_result(payload, probabilities=None):
    probabilities = probabilities or [0.999] * len(payload["states"])
    return {"checkpoint": {"model": "synthetic-scoring-stub"}, "states": [
        {"id": state["id"], "answers": {"irrelevant": {"type": "boolean",
                                                       "probabilities": {"false": 1 - p, "true": p}}}}
        for state, p in zip(payload["states"], probabilities)]}


def request(history=HISTORY_TEXT, wire="openai_chat"):
    messages = [{"role": "assistant", "content": history}, {"role": "user", "content": USER_TEXT}]
    if wire == "anthropic_messages":
        return {"model": "synthetic-main-model", "system": SYSTEM_TEXT, "messages": messages, "max_tokens": 64}
    return {"model": "synthetic-main-model",
            "messages": [{"role": "system", "content": SYSTEM_TEXT}] + messages}


def two_eligible_request():
    body = request()
    # system(0), assistant(1), assistant(2), user(3)
    body["messages"].insert(2, {"role": "assistant", "content": SECOND_HISTORY})
    return body


def eligible_pointer(wire="openai_chat"):
    if wire == "anthropic_messages":
        return "/messages/0/content"
    if wire == "openai_responses":
        return "/input/0/content"
    return "/messages/1/content"


def eligible_sidecar(pointer="/messages/1/content"):
    """Integration-supplied eligibility. The request body cannot grant this itself."""
    return {"segments": {pointer: {"eligible": True}}}


def encoded(body):
    return json.dumps(body, ensure_ascii=False, indent=2).encode("utf-8")


def word_counter(text):
    return len(text.split())


class FakeUpstream:
    """A local http.server that records raw request bytes and returns a fixed shape."""

    def __init__(self, status=200, body=None):
        self.status = status
        self.body = body
        self.requests = []
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _handle(self):
                length = int(self.headers.get("Content-Length", 0) or 0)
                raw = self.rfile.read(length) if length else b""
                upstream.requests.append({"method": self.command, "path": self.path,
                                          "headers": dict(self.headers.items()), "body": raw})
                if upstream.body is not None:
                    reply_status, reply_body = upstream.status, upstream.body
                else:
                    words = len(raw.split())
                    reply_status = upstream.status
                    reply_body = json.dumps({
                        "id": "fake-upstream", "object": "chat.completion",
                        "usage": {"prompt_tokens": words, "completion_tokens": 3, "total_tokens": words + 3},
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
                    }).encode("utf-8")
                self.send_response(reply_status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(reply_body)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(reply_body)

            do_POST = _handle
            do_GET = _handle

            def log_message(self, *args):
                return

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class FakeNanoJevService:
    """A loopback stand-in for the existing NanoJev ``/api/evaluate`` contract."""

    def __init__(self):
        service = self
        self.calls = []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0) or 0)
                payload = json.loads(self.rfile.read(length))
                service.calls.append(self.path)
                body = json.dumps(score_result(payload)).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                return

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def http_post(host, port, path, raw, headers=None):
    connection = HTTPConnection(host, port, timeout=10)
    try:
        connection.request("POST", path, body=raw,
                           headers={"Content-Type": "application/json", **(headers or {})})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def http_post_with_headers(host, port, path, raw, headers=None):
    """Same as ``http_post`` but also returns the response headers as a lowercased dict."""
    connection = HTTPConnection(host, port, timeout=10)
    try:
        connection.request("POST", path, body=raw,
                           headers={"Content-Type": "application/json", **(headers or {})})
        response = connection.getresponse()
        body = response.read()
        reply_headers = {name.lower(): value for name, value in response.getheaders()}
        return response.status, body, reply_headers
    finally:
        connection.close()


def http_get(host, port, path):
    connection = HTTPConnection(host, port, timeout=10)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def read_receipts(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


class GatewayTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="nanojev-gateway-", dir="/tmp"))
        self.upstream = FakeUpstream()
        self.addCleanup(self.upstream.close)
        self.servers = []

    def start_gateway(self, mode="shadow", scorer=None, kill_switch=False, sidecar=None,
                      receipt_log=None, score_timeout=5.0, token_counter=word_counter,
                      tokenizer_id="test-word-counter", upstream_url=None):
        config = GatewayConfig(
            upstream_base_url=upstream_url or self.upstream.base_url,
            listen_host="127.0.0.1", listen_port=0,
            mode=mode, kill_switch=kill_switch, scorer=scorer, sidecar=sidecar or {},
            receipt_log=Path(receipt_log) if receipt_log else (self.tmp / "receipts.jsonl"),
            score_timeout=score_timeout, token_counter=token_counter, tokenizer_id=tokenizer_id)
        gateway = ContextGateGateway(config)
        server = make_server(gateway)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.servers.append((server, thread))
        self.addCleanup(self._stop, server, thread)
        return config, server.server_address[0], server.server_address[1]

    @staticmethod
    def _stop(server, thread):
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class ShadowModeTest(GatewayTestCase):
    def test_shadow_forwards_original_bytes_byte_identically_and_only_estimates(self):
        raw = encoded(request())
        config, host, port = self.start_gateway(mode="shadow", scorer=score_result, sidecar=eligible_sidecar())
        status, body = http_post(host, port, "/v1/chat/completions", raw)
        self.assertEqual(status, 200)
        self.assertEqual(self.upstream.requests[0]["body"], raw)
        receipt = read_receipts(config.receipt_log)[-1]
        self.assertEqual(receipt["mode"], "shadow")
        self.assertTrue(receipt["forwarded_unchanged"])
        self.assertEqual(receipt["forward_reason"], "shadow_mode")
        self.assertEqual(receipt["request_sha256"], receipt["forwarded_sha256"])
        self.assertEqual(receipt["gate_receipt"]["status"], "scored")
        self.assertEqual(receipt["removal_plan"]["proposed_pointers"], ["/messages/1/content"])
        self.assertFalse(receipt["removal_plan"]["applied"])
        # Shadow mode reports an ESTIMATE labelled with the tokenizer identity, never actual savings.
        savings = receipt["token_accounting"]["savings"]
        self.assertEqual(savings["claim"], "estimate")
        self.assertEqual(savings["basis"], "shadow_estimate_only")
        self.assertGreater(savings["tokens"], 0)
        self.assertEqual(receipt["token_accounting"]["estimate"]["tokenizer_id"], "test-word-counter")
        self.assertEqual(receipt["token_accounting"]["provider_reported"]["prompt_tokens"], len(raw.split()))

    def test_default_config_mode_is_shadow(self):
        config = GatewayConfig(upstream_base_url="http://127.0.0.1:1")
        self.assertEqual(config.mode, "shadow")
        self.assertFalse(config.kill_switch)


class ActiveModeTest(GatewayTestCase):
    def test_active_forwards_reduced_bytes_and_reports_provider_savings(self):
        raw = encoded(request())
        config, host, port = self.start_gateway(mode="active", scorer=score_result,
                                                sidecar=eligible_sidecar())
        status, _ = http_post(host, port, "/v1/chat/completions", raw,
                              headers={BASELINE_HEADER: "500"})
        self.assertEqual(status, 200)
        sent = self.upstream.requests[0]["body"]
        self.assertNotEqual(sent, raw)
        reduced = json.loads(sent)
        self.assertEqual([message["role"] for message in reduced["messages"]], ["system", "user"])
        self.assertEqual(reduced["messages"][1]["content"], USER_TEXT)
        self.assertEqual(reduced["messages"][0]["content"], SYSTEM_TEXT)
        # The caller's internal accounting header is stripped before forwarding.
        self.assertNotIn(BASELINE_HEADER, {name.lower() for name in self.upstream.requests[0]["headers"]})
        receipt = read_receipts(config.receipt_log)[-1]
        self.assertEqual(receipt["mode"], "active")
        self.assertFalse(receipt["forwarded_unchanged"])
        self.assertEqual(receipt["forward_reason"], "active_reduced")
        self.assertTrue(receipt["removal_plan"]["applied"])
        self.assertEqual(receipt["removal_plan"]["applied_pointers"], ["/messages/1/content"])
        provider = receipt["token_accounting"]["provider_reported"]
        self.assertEqual(provider["prompt_tokens"], len(sent.split()))
        savings = receipt["token_accounting"]["savings"]
        self.assertEqual(savings["claim"], "actual")
        self.assertEqual(savings["basis"], "provider_paired_baseline")
        self.assertEqual(savings["tokens"], 500 - len(sent.split()))

    def test_active_without_paired_baseline_reports_estimate_not_actual(self):
        raw = encoded(request())
        config, host, port = self.start_gateway(mode="active", scorer=score_result,
                                                sidecar=eligible_sidecar())
        http_post(host, port, "/v1/chat/completions", raw)
        receipt = read_receipts(config.receipt_log)[-1]
        savings = receipt["token_accounting"]["savings"]
        self.assertNotEqual(receipt["forwarded_unchanged"], True)
        self.assertEqual(savings["claim"], "estimate")
        self.assertEqual(savings["basis"], "provider_usage_observed_no_paired_baseline")

    def test_anthropic_wire_format_reduces_eligible_assistant_text(self):
        body = {"model": "synthetic-main-model", "system": SYSTEM_TEXT,
                "messages": [{"role": "assistant", "content": [{"type": "text", "text": HISTORY_TEXT}]},
                             {"role": "user", "content": USER_TEXT}],
                "max_tokens": 64}
        raw = encoded(body)
        sidecar = {"segments": {"/messages/0/content/0/text": {"eligible": True}}}
        config, host, port = self.start_gateway(mode="active", scorer=score_result, sidecar=sidecar)
        status, _ = http_post(host, port, "/v1/messages", raw)
        self.assertEqual(status, 200)
        sent = self.upstream.requests[0]["body"]
        self.assertNotEqual(sent, raw)
        reduced = json.loads(sent)
        self.assertEqual([message["role"] for message in reduced["messages"]], ["user"])
        receipt = read_receipts(config.receipt_log)[-1]
        self.assertEqual(receipt["wire_format"], "anthropic_messages")
        self.assertEqual(receipt["removal_plan"]["applied_pointers"], ["/messages/0/content/0/text"])

    def test_sidecar_header_supplies_eligibility_and_is_stripped(self):
        raw = encoded(request())
        header = base64.b64encode(json.dumps(eligible_sidecar()).encode("utf-8")).decode("ascii")
        config, host, port = self.start_gateway(mode="active", scorer=score_result)
        status, _ = http_post(host, port, "/v1/chat/completions", raw,
                              headers={SIDECAR_HEADER: header})
        self.assertEqual(status, 200)
        self.assertNotEqual(self.upstream.requests[0]["body"], raw)
        self.assertNotIn(SIDECAR_HEADER, {name.lower() for name in self.upstream.requests[0]["headers"]})
        receipt = read_receipts(config.receipt_log)[-1]
        self.assertEqual(receipt["removal_plan"]["applied_pointers"], ["/messages/1/content"])

    def test_malformed_sidecar_header_fails_open(self):
        raw = encoded(request())
        config, host, port = self.start_gateway(mode="active", scorer=score_result)
        status, _ = http_post(host, port, "/v1/chat/completions", raw,
                              headers={SIDECAR_HEADER: "!!!not-base64!!!"})
        self.assertEqual(status, 200)
        self.assertEqual(self.upstream.requests[0]["body"], raw)
        receipt = read_receipts(config.receipt_log)[-1]
        self.assertEqual(receipt["gate_receipt"]["reason"], "caller_bypass")
        self.assertTrue(receipt["forwarded_unchanged"])


class FailOpenTest(GatewayTestCase):
    def test_protected_segment_request_is_never_reduced(self):
        raw = encoded(request())
        # The only candidate is caller-pinned, so the whole request must stay intact.
        sidecar = {"segments": {"/messages/1/content": {"eligible": True, "pinned": True}}}
        config, host, port = self.start_gateway(mode="active", scorer=score_result, sidecar=sidecar)
        status, _ = http_post(host, port, "/v1/chat/completions", raw)
        self.assertEqual(status, 200)
        self.assertEqual(self.upstream.requests[0]["body"], raw)
        receipt = read_receipts(config.receipt_log)[-1]
        self.assertTrue(receipt["forwarded_unchanged"])
        self.assertEqual(receipt["gate_receipt"]["reason"], "no_eligible_segments")
        self.assertEqual(receipt["removal_plan"]["proposed_pointers"], [])
        self.assertFalse(receipt["removal_plan"]["applied"])

    def test_protected_drop_in_gate_plan_is_refused_defensively(self):
        raw = encoded(request())

        def fake_shadow(request_bytes, wire_format, sidecar=None, scorer=None, **kwargs):
            receipt = {"schema_version": "nanojev-context-shadow-v1", "status": "scored",
                       "reason": "shadow_only", "latency_ms": 1.0, "segments": [
                           {"pointer": "/messages/1/content", "role": "assistant", "sha256": "a",
                            "suggestion": "drop", "reason": "required_dependency", "p_irrelevant": 0.999},
                           {"pointer": "/messages/0/content", "role": "system", "sha256": "b",
                            "suggestion": "drop", "reason": "protected_structure", "p_irrelevant": 0.999}],
                       "token_counts": {"scope": "unavailable", "proposed_text_tokens": None}}
            return request_bytes, receipt

        config, host, port = self.start_gateway(mode="active", scorer=score_result)
        with patch.object(gateway_module, "shadow_request", side_effect=fake_shadow):
            status, _ = http_post(host, port, "/v1/chat/completions", raw)
        self.assertEqual(status, 200)
        self.assertEqual(self.upstream.requests[0]["body"], raw)
        receipt = read_receipts(config.receipt_log)[-1]
        self.assertEqual(receipt["forward_reason"], "protected_segment_in_removal_set")
        self.assertTrue(receipt["forwarded_unchanged"])
        self.assertFalse(receipt["removal_plan"]["applied"])

    def test_removal_plan_helper_rejects_every_non_eligible_drop(self):
        for segment in ({"pointer": "/messages/0/content", "role": "system", "suggestion": "drop",
                         "reason": "protected_structure"},
                        {"pointer": "/messages/1/content", "role": "assistant", "suggestion": "drop",
                         "reason": "caller_protected"},
                        {"pointer": "/messages/1/content", "role": "assistant", "suggestion": "drop",
                         "reason": "required_dependency"},
                        {"pointer": "/messages/1/content", "role": "user", "suggestion": "drop",
                         "reason": "high_irrelevance_score"}):
            with self.subTest(segment=segment):
                drops, error = removal_plan({"segments": [segment]})
                self.assertEqual(drops, [])
                self.assertEqual(error, "protected_segment_in_removal_set")

    def test_scorer_exception_fails_open_and_does_not_leak_text(self):
        secret = "UNIQUE_PRIVATE_PROMPT_TEXT_42"

        def exploding_scorer(payload):
            raise RuntimeError(secret)

        raw = encoded(request(history=f"old context {secret}"))
        config, host, port = self.start_gateway(mode="active", scorer=exploding_scorer,
                                                sidecar=eligible_sidecar())
        status, _ = http_post(host, port, "/v1/chat/completions", raw)
        self.assertEqual(status, 200)
        self.assertEqual(self.upstream.requests[0]["body"], raw)
        receipt = read_receipts(config.receipt_log)[-1]
        self.assertEqual(receipt["gate_receipt"]["reason"], "scorer_error")
        self.assertEqual(receipt["scorer_failure_kind"], "exception")
        self.assertTrue(receipt["forwarded_unchanged"])
        self.assertNotIn(secret, config.receipt_log.read_text(encoding="utf-8"))

    def test_scorer_timeout_fails_open(self):
        def slow_scorer(payload):
            time.sleep(1.0)
            return score_result(payload)

        raw = encoded(request())
        config, host, port = self.start_gateway(mode="active", scorer=slow_scorer, score_timeout=0.05,
                                                sidecar=eligible_sidecar())
        status, _ = http_post(host, port, "/v1/chat/completions", raw)
        self.assertEqual(status, 200)
        self.assertEqual(self.upstream.requests[0]["body"], raw)
        receipt = read_receipts(config.receipt_log)[-1]
        self.assertEqual(receipt["gate_receipt"]["reason"], "scorer_error")
        self.assertEqual(receipt["scorer_failure_kind"], "timeout")
        self.assertTrue(receipt["forwarded_unchanged"])
        self.assertFalse(receipt["removal_plan"]["applied"])

    def test_absent_scorer_fails_open(self):
        raw = encoded(request())
        config, host, port = self.start_gateway(mode="active", scorer=None, sidecar=eligible_sidecar())
        status, _ = http_post(host, port, "/v1/chat/completions", raw)
        self.assertEqual(status, 200)
        self.assertEqual(self.upstream.requests[0]["body"], raw)
        receipt = read_receipts(config.receipt_log)[-1]
        self.assertEqual(receipt["gate_receipt"]["reason"], "scorer_unavailable")

    def test_malformed_partial_and_nonfinite_scores_fail_open(self):
        body = two_eligible_request()
        raw = encoded(body)
        sidecar = {"segments": {"/messages/1/content": {"eligible": True},
                                "/messages/2/content": {"eligible": True}}}

        class SwitchScorer:
            def __init__(self):
                self.result = None

            def __call__(self, payload):
                return self.result

        holder = SwitchScorer()
        config, host, port = self.start_gateway(mode="active", scorer=holder, sidecar=sidecar)
        payload_stub = {"states": [{"id": "segment_0"}, {"id": "segment_1"}]}
        one_state = score_result(payload_stub)
        one_state["states"] = one_state["states"][:1]
        partial_envelope = {"states": [{"id": "segment_0",
                                        "answers": {"irrelevant": {"type": "boolean",
                                                                   "probabilities": {"false": 0.001, "true": 0.999}}}}]}
        nan_result = score_result(payload_stub)
        nan_result["states"][0]["answers"]["irrelevant"]["probabilities"]["true"] = float("nan")
        nonunit = score_result(payload_stub)
        nonunit["states"][0]["answers"]["irrelevant"]["probabilities"] = {"false": 0.5, "true": 0.7}
        duplicate = score_result(payload_stub)
        duplicate["states"] = [duplicate["states"][0], duplicate["states"][0]]
        bads = [None, {}, {"states": []}, partial_envelope, one_state, nan_result, nonunit, duplicate]
        for bad in bads:
            with self.subTest(result=bad):
                holder.result = bad
                status, _ = http_post(host, port, "/v1/chat/completions", raw)
                self.assertEqual(status, 200)
                self.assertEqual(self.upstream.requests[-1]["body"], raw)
                receipt = read_receipts(config.receipt_log)[-1]
                self.assertEqual(receipt["gate_receipt"]["reason"], "invalid_score_response")
                self.assertFalse(receipt["removal_plan"]["applied"])
                self.assertTrue(receipt["forwarded_unchanged"])

    def test_uncertain_score_fails_open(self):
        body = two_eligible_request()
        raw = encoded(body)
        sidecar = {"segments": {"/messages/1/content": {"eligible": True},
                                "/messages/2/content": {"eligible": True}}}
        config, host, port = self.start_gateway(
            mode="active", sidecar=sidecar,
            scorer=lambda payload: score_result(payload, [0.999, 0.5]))
        status, _ = http_post(host, port, "/v1/chat/completions", raw)
        self.assertEqual(status, 200)
        self.assertEqual(self.upstream.requests[0]["body"], raw)
        receipt = read_receipts(config.receipt_log)[-1]
        self.assertEqual(receipt["gate_receipt"]["reason"], "uncertain_score")
        self.assertTrue(receipt["forwarded_unchanged"])

    def test_malformed_reduction_plan_fails_open(self):
        raw = encoded(request())
        with self.assertRaises(ValueError):
            build_reduced_request(raw, "openai_chat", ["/messages/1/role"])
        with self.assertRaises(ValueError):
            build_reduced_request(raw, "openai_chat", ["/messages/99/content"])
        with self.assertRaises(ValueError):
            # Both the assistant history and the user intent would disappear.
            build_reduced_request(encoded(request()), "openai_chat",
                                  ["/messages/1/content", "/messages/2/content"])


class KillSwitchTest(GatewayTestCase):
    def test_kill_switch_forwards_everything_unchanged_without_scoring(self):
        raw = encoded(request())

        def scorer_must_not_run(payload):
            raise AssertionError("kill switch must not score")

        config, host, port = self.start_gateway(mode="active", scorer=scorer_must_not_run, kill_switch=True)
        status, _ = http_post(host, port, "/v1/chat/completions", raw)
        self.assertEqual(status, 200)
        self.assertEqual(self.upstream.requests[0]["body"], raw)
        receipt = read_receipts(config.receipt_log)[-1]
        self.assertEqual(receipt["forward_reason"], "kill_switch")
        self.assertTrue(receipt["kill_switch"])
        self.assertTrue(receipt["forwarded_unchanged"])
        self.assertIsNone(receipt["gate_receipt"])
        self.assertEqual(receipt["token_accounting"]["savings"]["claim"], "none")

    def test_kill_switch_environment_variable_is_deterministic(self):
        with patch.dict(os.environ, {KILL_SWITCH_ENV: "1"}):
            self.assertTrue(_kill_switch_from_env())
        with patch.dict(os.environ, {KILL_SWITCH_ENV: "off"}):
            self.assertFalse(_kill_switch_from_env())
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_kill_switch_from_env())


class BypassTest(GatewayTestCase):
    def test_unsupported_wire_format_forwards_original_bytes(self):
        raw = json.dumps({"input": "embed this", "model": "text-embedding"}).encode("utf-8")
        config, host, port = self.start_gateway(mode="active", scorer=score_result)
        status, _ = http_post(host, port, "/v1/embeddings", raw)
        self.assertEqual(status, 200)
        self.assertEqual(self.upstream.requests[0]["body"], raw)
        receipt = read_receipts(config.receipt_log)[-1]
        self.assertIsNone(receipt["wire_format"])
        self.assertEqual(receipt["forward_reason"], "unsupported_wire_format")
        self.assertTrue(receipt["forwarded_unchanged"])

    def test_non_post_method_forwards_original_bytes(self):
        config, host, port = self.start_gateway(mode="active", scorer=score_result)
        status, _ = http_get(host, port, "/v1/chat/completions")
        self.assertEqual(status, 200)
        receipt = read_receipts(config.receipt_log)[-1]
        self.assertEqual(receipt["forward_reason"], "unsupported_method")
        self.assertEqual(self.upstream.requests[0]["method"], "GET")

    def test_openai_responses_format_is_supported(self):
        body = {"model": "synthetic-main-model", "instructions": SYSTEM_TEXT,
                "input": [{"role": "assistant", "content": HISTORY_TEXT},
                          {"role": "user", "content": USER_TEXT}]}
        raw = encoded(body)
        sidecar = {"segments": {"/input/0/content": {"eligible": True}}}
        config, host, port = self.start_gateway(mode="active", scorer=score_result, sidecar=sidecar)
        status, _ = http_post(host, port, "/v1/responses", raw)
        self.assertEqual(status, 200)
        receipt = read_receipts(config.receipt_log)[-1]
        self.assertEqual(receipt["wire_format"], "openai_responses")
        self.assertEqual(receipt["removal_plan"]["applied_pointers"], ["/input/0/content"])


class RestoreManifestHeaderTest(GatewayTestCase):
    """Reversible filtering: the restore manifest is emitted only for an applied reduction.

    The header describes how the caller reconstructs the original request from the segments
    it retained. It must never carry raw prompt text, and every fail-open path must omit it.
    """

    def test_header_is_emitted_for_an_applied_reduction_and_proves_reversibility(self):
        body = request()
        raw = encoded(body)
        config, host, port = self.start_gateway(mode="active", scorer=score_result,
                                                sidecar=eligible_sidecar())
        status, _, reply_headers = http_post_with_headers(host, port, "/v1/chat/completions", raw)
        self.assertEqual(status, 200)
        sent = self.upstream.requests[0]["body"]
        self.assertNotEqual(sent, raw)
        manifest = json.loads(reply_headers[RESTORE_MANIFEST_HEADER])
        self.assertEqual(manifest["schema_version"], "nanojev-context-restore-v1")
        self.assertTrue(manifest["applied"])
        self.assertEqual(manifest["wire_format"], "openai_chat")
        self.assertEqual(manifest["dropped_segment_count"], 1)
        self.assertEqual([record["pointer"] for record in manifest["records"]],
                         ["/messages/1/content"])
        # The header itself is content-free.
        self.assertNotIn(HISTORY_TEXT, reply_headers[RESTORE_MANIFEST_HEADER])
        self.assertNotIn(USER_TEXT, reply_headers[RESTORE_MANIFEST_HEADER])
        self.assertNotIn("archived weather note", reply_headers[RESTORE_MANIFEST_HEADER])
        # The caller's original body is indented, so the manifest flags that restoration is
        # canonical-equivalent, and the reconstruction reproduces the canonical original.
        self.assertFalse(manifest["canonical_json"])
        self.assertNotEqual(manifest["original_request_sha256"], manifest["restored_bytes_sha256"])
        restored = restore_request(sent, manifest, {"/messages/1/content": HISTORY_TEXT})
        self.assertEqual(restored, canonical_bytes(body))
        result = verify_round_trip(raw, sent, manifest, {"/messages/1/content": HISTORY_TEXT})
        self.assertFalse(result["restored_bytes_identical"])
        self.assertTrue(result["original_request_sha256_matches"])
        self.assertTrue(result["reduced_request_sha256_matches"])

    def test_irreversible_plan_fails_open_instead_of_being_sent(self):
        """Regression for the reviewer-found contract violation.

        Building a manifest is not proof of reversibility: an emptied mid-list message
        followed by another list-content message still cannot be reconstructed. Before the
        guard existed the gateway forwarded that reduction anyway and handed the caller an
        unusable manifest, silently breaking the reversible-filtering contract. The gateway
        must instead forward the ORIGINAL bytes and emit no header.
        """
        body = {"model": "synthetic-main-model", "messages": [
            {"role": "user", "content": USER_TEXT},
            {"role": "assistant", "content": [{"type": "text", "text": HISTORY_TEXT}]},
            {"role": "assistant", "content": [{"type": "text", "text": SECOND_HISTORY}]}]}
        raw = encoded(body)
        config, host, port = self.start_gateway(
            mode="active", scorer=score_result,
            sidecar={"segments": {"/messages/1/content/0/text": {"eligible": True}}})
        status, _, reply_headers = http_post_with_headers(host, port, "/v1/messages", raw)
        self.assertEqual(status, 200)
        self.assertEqual(self.upstream.requests[0]["body"], raw)  # original bytes forwarded
        self.assertNotIn(RESTORE_MANIFEST_HEADER, reply_headers)  # no unusable manifest
        receipt = read_receipts(config.receipt_log)[-1]
        self.assertEqual(receipt["forward_reason"], "reduction_error")
        self.assertTrue(receipt["forwarded_unchanged"])
        self.assertFalse(receipt["removal_plan"]["applied"])

    def test_header_is_emitted_for_partial_content_reduction(self):
        body = {"model": "synthetic-main-model", "messages": [
            {"role": "system", "content": SYSTEM_TEXT},
            {"role": "assistant", "content": [
                {"type": "text", "text": HISTORY_TEXT},
                {"type": "text", "text": SECOND_HISTORY}]},
            {"role": "user", "content": USER_TEXT}]}
        raw = encoded(body)
        sidecar = {"segments": {"/messages/1/content/0/text": {"eligible": True}}}
        config, host, port = self.start_gateway(mode="active", scorer=score_result, sidecar=sidecar)
        status, _, reply_headers = http_post_with_headers(host, port, "/v1/chat/completions", raw)
        self.assertEqual(status, 200)
        sent = self.upstream.requests[0]["body"]
        manifest = json.loads(reply_headers[RESTORE_MANIFEST_HEADER])
        self.assertEqual(manifest["records"][0]["kind"], "text_part")
        self.assertEqual(restore_request(sent, manifest, {"/messages/1/content/0/text": HISTORY_TEXT}),
                         canonical_bytes(body))

    def test_header_manifest_is_the_same_manifest_the_gateway_could_have_emitted(self):
        body = request()
        raw = encoded(body)
        config, host, port = self.start_gateway(mode="active", scorer=score_result,
                                                sidecar=eligible_sidecar())
        status, _, reply_headers = http_post_with_headers(host, port, "/v1/chat/completions", raw)
        self.assertEqual(status, 200)
        sent = self.upstream.requests[0]["body"]
        # Independently rebuilt from the original bytes and the applied pointers, the
        # manifest is exactly the one the gateway returned.
        expected = build_restore_manifest(raw, "openai_chat", ["/messages/1/content"])
        self.assertEqual(json.loads(reply_headers[RESTORE_MANIFEST_HEADER]), expected.as_dict())
        self.assertEqual(restore_request(sent, expected, {"/messages/1/content": HISTORY_TEXT}),
                         canonical_bytes(body))

    def test_canonical_request_round_trips_byte_identically_through_the_gateway(self):
        body = request()
        raw = canonical_bytes(body)
        config, host, port = self.start_gateway(mode="active", scorer=score_result,
                                                sidecar=eligible_sidecar())
        status, _, reply_headers = http_post_with_headers(host, port, "/v1/chat/completions", raw)
        self.assertEqual(status, 200)
        sent = self.upstream.requests[0]["body"]
        manifest = json.loads(reply_headers[RESTORE_MANIFEST_HEADER])
        self.assertTrue(manifest["canonical_json"])
        self.assertEqual(manifest["original_request_sha256"], manifest["restored_bytes_sha256"])
        restored = restore_request(sent, manifest, {"/messages/1/content": HISTORY_TEXT})
        self.assertEqual(restored, raw)
        result = verify_round_trip(raw, sent, manifest, {"/messages/1/content": HISTORY_TEXT})
        self.assertTrue(result["restored_bytes_identical"])

    def test_header_is_never_sent_when_no_reduction_is_applied(self):
        raw = encoded(request())
        # Shadow mode: a plan is proposed but nothing is removed.
        config, host, port = self.start_gateway(mode="shadow", scorer=score_result,
                                                sidecar=eligible_sidecar())
        status, _, shadow_headers = http_post_with_headers(host, port, "/v1/chat/completions", raw)
        self.assertEqual(status, 200)
        self.assertNotIn(RESTORE_MANIFEST_HEADER, shadow_headers)
        receipt = read_receipts(config.receipt_log)[-1]
        self.assertEqual(receipt["forward_reason"], "shadow_mode")

    def test_header_is_not_sent_for_kill_switch_or_unsupported_paths(self):
        raw = encoded(request())
        config, host, port = self.start_gateway(mode="active", scorer=score_result,
                                                sidecar=eligible_sidecar(), kill_switch=True)
        status, _, headers = http_post_with_headers(host, port, "/v1/chat/completions", raw)
        self.assertEqual(status, 200)
        self.assertNotIn(RESTORE_MANIFEST_HEADER, headers)
        self.assertEqual(self.upstream.requests[0]["body"], raw)
        # An unsupported wire format bypasses the gate entirely.
        config2, host2, port2 = self.start_gateway(mode="active", scorer=score_result,
                                                   sidecar=eligible_sidecar())
        status2, _, headers2 = http_post_with_headers(
            host2, port2, "/v1/embeddings", json.dumps({"input": "x"}).encode("utf-8"))
        self.assertEqual(status2, 200)
        self.assertNotIn(RESTORE_MANIFEST_HEADER, headers2)

    def test_header_is_not_sent_on_every_fail_open_path(self):
        raw = encoded(request())

        def exploding_scorer(payload):
            raise RuntimeError("boom")

        def slow_scorer(payload):
            time.sleep(1.0)
            return score_result(payload)

        def fake_shadow(request_bytes, wire_format, sidecar=None, scorer=None, **kwargs):
            return request_bytes, {"schema_version": "nanojev-context-shadow-v1", "status": "scored",
                                   "reason": "shadow_only", "latency_ms": 1.0, "segments": [
                                       {"pointer": "/messages/1/content", "role": "assistant",
                                        "sha256": "a", "suggestion": "drop",
                                        "reason": "required_dependency", "p_irrelevant": 0.999}],
                                   "token_counts": {"scope": "unavailable",
                                                    "proposed_text_tokens": None}}

        pinned_sidecar = {"segments": {"/messages/1/content": {"eligible": True, "pinned": True}}}
        cases = [
            ("protected_segment", dict(scorer=score_result, sidecar=pinned_sidecar)),
            ("scorer_exception", dict(scorer=exploding_scorer, sidecar=eligible_sidecar())),
            ("scorer_timeout", dict(scorer=slow_scorer, sidecar=eligible_sidecar(),
                                    score_timeout=0.05)),
            ("absent_scorer", dict(scorer=None, sidecar=eligible_sidecar())),
        ]
        for name, kwargs in cases:
            with self.subTest(case=name):
                config, host, port = self.start_gateway(mode="active", **kwargs)
                status, _, headers = http_post_with_headers(host, port, "/v1/chat/completions", raw)
                self.assertEqual(status, 200)
                self.assertNotIn(RESTORE_MANIFEST_HEADER, headers)
                self.assertEqual(self.upstream.requests[-1]["body"], raw)
                receipt = read_receipts(config.receipt_log)[-1]
                self.assertTrue(receipt["forwarded_unchanged"])
        # A protected drop inside an otherwise-scored plan is refused defensively.
        config, host, port = self.start_gateway(mode="active", scorer=score_result)
        with patch.object(gateway_module, "shadow_request", side_effect=fake_shadow):
            status, _, headers = http_post_with_headers(host, port, "/v1/chat/completions", raw)
        self.assertEqual(status, 200)
        self.assertNotIn(RESTORE_MANIFEST_HEADER, headers)
        self.assertEqual(self.upstream.requests[-1]["body"], raw)
        receipt = read_receipts(config.receipt_log)[-1]
        self.assertEqual(receipt["forward_reason"], "protected_segment_in_removal_set")

    def test_header_is_never_forwarded_upstream(self):
        raw = encoded(request())
        config, host, port = self.start_gateway(mode="active", scorer=score_result,
                                                sidecar=eligible_sidecar())
        status, _, headers = http_post_with_headers(host, port, "/v1/chat/completions", raw)
        self.assertEqual(status, 200)
        self.assertIn(RESTORE_MANIFEST_HEADER, headers)
        upstream_headers = {name.lower() for name in self.upstream.requests[0]["headers"]}
        self.assertNotIn(RESTORE_MANIFEST_HEADER, upstream_headers)

    def test_header_is_not_written_into_the_receipt(self):
        raw = encoded(request(history=f"archived note {HISTORY_TEXT}"))
        config, host, port = self.start_gateway(mode="active", scorer=score_result,
                                                sidecar=eligible_sidecar())
        status, _, headers = http_post_with_headers(host, port, "/v1/chat/completions", raw)
        self.assertEqual(status, 200)
        text = config.receipt_log.read_text(encoding="utf-8")
        self.assertNotIn(HISTORY_TEXT, text)
        self.assertNotIn(RESTORE_MANIFEST_HEADER, text)

    def test_malformed_reduction_manifest_fails_open_and_emits_no_header(self):
        raw = encoded(request())
        config, host, port = self.start_gateway(mode="active", scorer=score_result,
                                                sidecar=eligible_sidecar())
        with patch.object(gateway_module, "build_restore_manifest",
                          side_effect=RestoreError("manifest_unavailable")):
            status, _, headers = http_post_with_headers(host, port, "/v1/chat/completions", raw)
        self.assertEqual(status, 200)
        self.assertNotIn(RESTORE_MANIFEST_HEADER, headers)
        self.assertEqual(self.upstream.requests[0]["body"], raw)
        receipt = read_receipts(config.receipt_log)[-1]
        self.assertEqual(receipt["forward_reason"], "reduction_error")
        self.assertTrue(receipt["forwarded_unchanged"])


class ReceiptPrivacyTest(GatewayTestCase):
    def test_receipts_never_contain_prompt_text_tool_output_or_credentials(self):
        secret = "UNIQUE_PRIVATE_TEXT_7f3a"
        credential = "Bearer UNIQUE_CREDENTIAL_9d1c"
        raw = encoded(request(history=f"archived note {secret}"))
        config, host, port = self.start_gateway(mode="active", scorer=score_result,
                                                sidecar=eligible_sidecar())
        http_post(host, port, "/v1/chat/completions", raw,
                  headers={"Authorization": credential, BASELINE_HEADER: "500"})
        text = config.receipt_log.read_text(encoding="utf-8")
        self.assertNotIn(secret, text)
        self.assertNotIn("UNIQUE_CREDENTIAL", text)
        self.assertNotIn("Authorization", text)
        receipts = read_receipts(config.receipt_log)
        self.assertEqual(len(receipts), 1)
        receipt = receipts[0]
        self.assertEqual(receipt["schema_version"], "nanojev-main-model-gateway-v1")
        self.assertIn("gate_receipt", receipt)
        self.assertIn("token_accounting", receipt)
        self.assertIn("removal_plan", receipt)
        self.assertIn("upstream", receipt)
        self.assertIsInstance(receipt["gate_latency_ms"], (int, float))
        self.assertIsInstance(receipt["total_latency_ms"], (int, float))
        for segment in receipt["gate_receipt"]["segments"]:
            self.assertIn("sha256", segment)
            self.assertNotIn("text", segment)


class UpstreamPassThroughTest(GatewayTestCase):
    def test_upstream_error_status_is_passed_through_not_rewritten(self):
        self.upstream.close()
        self.upstream = FakeUpstream(status=503, body=b'{"error":{"type":"upstream_busy","message":"try later"}}')
        self.addCleanup(self.upstream.close)
        raw = encoded(request())
        config, host, port = self.start_gateway(mode="shadow", scorer=score_result)
        status, body = http_post(host, port, "/v1/chat/completions", raw)
        self.assertEqual(status, 503)
        self.assertEqual(body, b'{"error":{"type":"upstream_busy","message":"try later"}}')
        receipt = read_receipts(config.receipt_log)[-1]
        self.assertEqual(receipt["upstream"]["status"], 503)
        self.assertEqual(self.upstream.requests[0]["body"], raw)

    def test_upstream_unreachable_reports_502_without_leaking(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]
        probe.close()  # nothing listens here now
        raw = encoded(request())
        config, host, port = self.start_gateway(mode="shadow", scorer=score_result,
                                                upstream_url=f"http://127.0.0.1:{dead_port}")
        status, body = http_post(host, port, "/v1/chat/completions", raw)
        self.assertEqual(status, 502)
        self.assertIn(b"nanojev_upstream_unreachable", body)
        receipt = read_receipts(config.receipt_log)[-1]
        self.assertEqual(receipt["upstream"]["error"], "unreachable")


class ConfigValidationTest(unittest.TestCase):
    def test_invalid_configurations_are_rejected(self):
        for kwargs in ({"mode": "always"}, {"listen_host": "0.0.0.0"}, {"listen_host": "example.org"},
                       {"listen_port": -1}, {"score_timeout": 0}, {"score_timeout": float("inf")},
                       {"threshold": 0.5}, {"upstream_base_url": "ftp://example.org"},
                       {"upstream_base_url": "https://user:pass@example.org"},
                       {"upstream_base_url": "https://api.example.org/v1"},
                       {"max_body_bytes": 0}):
            with self.subTest(kwargs=kwargs):
                base = {"upstream_base_url": "http://127.0.0.1:1"}
                base.update(kwargs)
                with self.assertRaises(ValueError):
                    GatewayConfig(**base)

    def test_valid_loopback_and_remote_https_origins_are_accepted(self):
        self.assertEqual(GatewayConfig(upstream_base_url="https://api.openai.com").mode, "shadow")
        self.assertEqual(GatewayConfig(upstream_base_url="http://127.0.0.1:9", listen_host="::1").listen_host, "::1")


class ScorerAdapterTest(unittest.TestCase):
    def test_in_process_and_build_scorer_shapes(self):
        scorer = InProcessScorer(score_result)
        payload = {"states": [{"id": "segment_0"}]}
        self.assertEqual(len(scorer(payload)["states"]), 1)
        self.assertIsNone(build_scorer("none"))
        self.assertIsInstance(build_scorer("http", url="http://127.0.0.1:8765"), NanoJevHTTPScorer)
        self.assertIsInstance(build_scorer("inprocess", inprocess=score_result), InProcessScorer)
        with self.assertRaises(ValueError):
            build_scorer("unknown")

    def test_deadline_scorer_and_failure_recorder(self):
        with self.assertRaises(ScorerTimeout):
            DeadlineScorer(lambda payload: time.sleep(1.0), 0.05)({"states": []})
        recorder = FailureRecordingScorer(DeadlineScorer(lambda payload: time.sleep(1.0), 0.05))
        with self.assertRaises(ScorerTimeout):
            recorder({"states": []})
        self.assertEqual(recorder.last_failure, "timeout")
        boom = FailureRecordingScorer(lambda payload: (_ for _ in ()).throw(RuntimeError("SECRET")))
        with self.assertRaises(RuntimeError):
            boom({"states": []})
        self.assertEqual(boom.last_failure, "exception")

    def test_nanojev_http_scorer_uses_existing_api_evaluate_contract(self):
        service = FakeNanoJevService()
        self.addCleanup(service.close)
        scorer = NanoJevHTTPScorer(service.base_url, timeout=2.0)
        result = scorer({"states": [{"id": "segment_0"}]})
        self.assertEqual(len(result["states"]), 1)
        self.assertEqual(service.calls, ["/api/evaluate"])

    def test_laya_adapter_is_documented_but_not_installed_or_enabled(self):
        adapter = LayaEncoderScorerAdapter()
        self.assertFalse(adapter.available)
        with self.assertRaises(ScorerError):
            adapter({"states": [{"id": "segment_0"}]})
        # Importing the adapter must never import, install, or download laya.
        self.assertNotIn("laya", sys.modules)
        with self.assertRaises(ValueError):
            LayaEncoderScorerAdapter(url="http://example.org")


class CliEntryPointTest(unittest.TestCase):
    """Regression: the CLI must actually start.

    ``main()`` previously passed a ``GatewayConfig`` to ``make_server()``, which
    expects a ``ContextGateGateway``, so the documented command line crashed with
    ``AttributeError: 'GatewayConfig' object has no attribute 'config'`` while every
    library-level test still passed. This test exercises the real entry point.
    """

    def test_cli_prints_banner_and_keeps_serving(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            # stderr goes to a file, never a pipe: reading a live pipe would block.
            err_path = Path(tmp) / "stderr.txt"
            with err_path.open("w+", encoding="utf-8") as err_file:
                proc = subprocess.Popen(
                    [sys.executable, str(repo / "scripts" / "main_model_gateway_v1.py"),
                     "--upstream", "http://127.0.0.1:1", "--listen-port", "0",
                     "--receipt-log", str(Path(tmp) / "receipts.jsonl")],
                    cwd=repo, stdout=subprocess.PIPE, stderr=err_file, text=True)
                try:
                    line = proc.stdout.readline()
                    if not line.strip():
                        proc.wait(timeout=10)
                        err_file.seek(0)
                        self.fail(f"CLI produced no banner; stderr={err_file.read()!r}")
                    banner = json.loads(line)
                    self.assertIn("listen", banner)
                    self.assertEqual(banner["mode"], "shadow")  # shadow is the default
                    self.assertEqual(banner["scorer"], "none")
                    self.assertIsNone(proc.poll(), "CLI exited immediately after startup")
                finally:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:  # pragma: no cover
                        proc.kill()
                        proc.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
