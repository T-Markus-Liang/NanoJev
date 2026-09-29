#!/usr/bin/env python3
"""Phase0 loopback active-mode dry-run.

Runs the gateway only against a loopback fake upstream. No provider is contacted.
The emitted receipt contains hashes, statuses, counts, and applied pointers only.
"""

import argparse
import base64
import hashlib
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time

from context_restore_v1 import restore_request, verify_round_trip
import main_model_gateway_v1 as gateway_module
from main_model_gateway_v1 import (
    BASELINE_HEADER, RESTORE_MANIFEST_HEADER, SIDECAR_HEADER, GatewayConfig,
    ContextGateGateway, make_server,
)
from scorer_adapters_v1 import DeadlineScorer, InProcessScorer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "active_mode_phase0_loopback_v1.json"

SYSTEM_TEXT = "Preserve instructions and answer from recorded evidence."
EVIDENCE_TEXT = "The latest queue depth is 42 jobs."
NOISE_TEXT = "Archived note about picnic blankets, reusable bottles, and weather forecasts."
USER_TEXT = "What is the latest queue depth?"


def request_path(wire="openai_chat"):
    return {"openai_chat": "/v1/chat/completions",
            "anthropic_messages": "/v1/messages",
            "openai_responses": "/v1/responses"}[wire]


def drop_pointer(wire="openai_chat"):
    return {"openai_chat": "/messages/2/content",
            "anthropic_messages": "/messages/1/content/0/text",
            "openai_responses": "/input/1/content"}[wire]


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_text(data):
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def request_body(wire="openai_chat"):
    if wire == "anthropic_messages":
        return {
            "model": "phase0-fake-upstream",
            "system": SYSTEM_TEXT,
            "max_tokens": 64,
            "messages": [
                {"role": "assistant",
                 "content": [{"type": "text", "text": EVIDENCE_TEXT}]},
                {"role": "assistant",
                 "content": [{"type": "text", "text": NOISE_TEXT}]},
                {"role": "user", "content": USER_TEXT},
            ],
        }
    if wire == "openai_responses":
        return {
            "model": "phase0-fake-upstream",
            "instructions": SYSTEM_TEXT,
            "input": [
                {"role": "assistant", "content": EVIDENCE_TEXT},
                {"role": "assistant", "content": NOISE_TEXT},
                {"role": "user", "content": USER_TEXT},
            ],
        }
    return {
        "model": "phase0-fake-upstream",
        "messages": [
            {"role": "system", "content": SYSTEM_TEXT},
            {"role": "assistant", "content": EVIDENCE_TEXT},
            {"role": "assistant", "content": NOISE_TEXT},
            {"role": "user", "content": USER_TEXT},
        ],
    }


def encoded_request(wire="openai_chat"):
    return json.dumps(request_body(wire), ensure_ascii=False, indent=2).encode("utf-8")


def deterministic_scorer(payload):
    return {"checkpoint": {"model": "phase0-deterministic-scorer"},
            "states": [{"id": state["id"],
                        "answers": {name: {"type": "boolean",
                                           "probabilities": {"false": 0.001,
                                                             "true": 0.999}}
                                    for name in (state.get("questions") or {})}}
                       for state in payload["states"]]}


def failing_scorer(_payload):
    raise RuntimeError("phase0 synthetic scorer failure")


def uncertain_scorer(payload):
    return {"checkpoint": {"model": "phase0-uncertain"},
            "states": [{"id": state["id"],
                        "answers": {name: {"type": "boolean",
                                           "probabilities": {"false": 0.5,
                                                             "true": 0.5}}
                                    for name in (state.get("questions") or {})}}
                       for state in payload["states"]]}


def invalid_scorer(_payload):
    return {"checkpoint": {"model": "phase0-invalid"},
            "states": [{"id": "unknown",
                        "answers": {"irrelevant": {"type": "boolean",
                                                   "probabilities": {"false": 0.001,
                                                                     "true": 0.999}}}}]}


def slow_scorer(payload):
    time.sleep(0.2)
    return deterministic_scorer(payload)


def dependency_scorer(payload):
    questions = payload["states"][0].get("questions") or {}
    if len(questions) != 2:
        raise ValueError("dependency scorer expects two candidates")
    values = [0.999, 0.001]
    return {"checkpoint": {"model": "phase0-dependency"},
            "states": [{"id": payload["states"][0]["id"],
                        "answers": {name: {"type": "boolean",
                                           "probabilities": {"false": 1 - value,
                                                             "true": value}}
                                    for name, value in zip(questions, values)}}]}


class FakeUpstream:
    def __init__(self):
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0) or 0)
                raw = self.rfile.read(length)
                owner.requests.append({"headers": dict(self.headers.items()), "body": raw})
                words = len(raw.split())
                body = json.dumps({
                    "id": "phase0-fake-upstream",
                    "object": "chat.completion",
                    "usage": {"prompt_tokens": words, "completion_tokens": 2,
                              "total_tokens": words + 2},
                    "choices": [{"index": 0,
                                  "message": {"role": "assistant", "content": "ok"}}],
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
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
        return response.status, response.read(), {
            name.lower(): value for name, value in response.getheaders()}
    finally:
        connection.close()


def read_receipt(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return json.loads(lines[-1])


def run_case(name, mode="active", wire="openai_chat", kill_switch=False,
             scorer=deterministic_scorer, sidecar=None, headers=None,
             path=None, raw=None, max_body_bytes=None,
             restore_header_limit=None):
    raw = raw if raw is not None else encoded_request(wire)
    path = path or request_path(wire)
    upstream = FakeUpstream()
    gateway_server = None
    old_restore_limit = gateway_module.MAX_RESTORE_HEADER_BYTES
    if restore_header_limit is not None:
        gateway_module.MAX_RESTORE_HEADER_BYTES = restore_header_limit
    try:
        with tempfile.TemporaryDirectory() as tempdir:
            receipt_log = Path(tempdir) / "receipts.jsonl"
            config_kwargs = {}
            if max_body_bytes is not None:
                config_kwargs["max_body_bytes"] = max_body_bytes
            config = GatewayConfig(
                upstream_base_url=upstream.base_url,
                listen_host="127.0.0.1",
                listen_port=0,
                mode=mode,
                kill_switch=kill_switch,
                receipt_log=receipt_log,
                scorer=InProcessScorer(scorer) if scorer is not None else None,
                sidecar=sidecar or {},
                threshold=0.90,
                **config_kwargs,
            )
            gateway_server = make_server(ContextGateGateway(config))
            gateway_thread = threading.Thread(target=gateway_server.serve_forever, daemon=True)
            gateway_thread.start()
            host, port = gateway_server.server_address[:2]
            status, _body, reply_headers = http_post(host, port, path, raw,
                                                    headers=headers)
            sent = upstream.requests[-1]["body"] if upstream.requests else None
            upstream_headers = (upstream.requests[-1].get("headers") or {}
                                if upstream.requests else {})
            receipt = read_receipt(receipt_log) if receipt_log.exists() else {}
            manifest_header = reply_headers.get(RESTORE_MANIFEST_HEADER)
            result = {
                "case_id": name,
                "status": "pass",
                "upstream_status": status,
                "upstream_request_count": len(upstream.requests),
                "upstream_internal_headers": [
                    name.lower() for name in upstream_headers
                    if name.lower().startswith("x-nanojev-")
                ],
                "request_sha256": sha256_bytes(raw),
                "forwarded_sha256": sha256_bytes(sent) if sent is not None else None,
                "forwarded_unchanged": sent == raw if sent is not None else None,
                "restore_header_present": manifest_header is not None,
                "forward_reason": receipt.get("forward_reason"),
                "wire_format": receipt.get("wire_format"),
                "savings_claim": ((receipt.get("token_accounting") or {})
                                  .get("savings") or {}).get("claim"),
                "savings_basis": ((receipt.get("token_accounting") or {})
                                  .get("savings") or {}).get("basis"),
                "proposed_pointers": (receipt.get("removal_plan") or {}).get("proposed_pointers"),
                "applied_pointers": (receipt.get("removal_plan") or {}).get("applied_pointers"),
                "gate_reason": (receipt.get("gate_receipt") or {}).get("reason"),
                "scorer_failure_kind": receipt.get("scorer_failure_kind"),
                "receipt_sha256": sha256_text(json.dumps(receipt, sort_keys=True)),
            }
            if manifest_header is not None:
                manifest = json.loads(manifest_header)
                result["manifest_records"] = manifest.get("dropped_segment_count")
                result["restore_round_trip"] = verify_round_trip(
                    raw, sent, manifest, {drop_pointer(wire): NOISE_TEXT})
                result["restore_header_sha256"] = sha256_text(manifest_header)
                result["restore_header_contains_raw_text"] = any(
                    text in manifest_header for text in (SYSTEM_TEXT, EVIDENCE_TEXT,
                                                           NOISE_TEXT, USER_TEXT))
            return result, sent, manifest_header
    finally:
        gateway_module.MAX_RESTORE_HEADER_BYTES = old_restore_limit
        if gateway_server is not None:
            gateway_server.shutdown()
            gateway_server.server_close()
        upstream.close()


def expect(condition, message, failures):
    if not condition:
        failures.append(message)


def expect_active_drop(case, sent, raw, manifest_header, wire, expected_roles, failures):
    pointer = drop_pointer(wire)
    reduced = json.loads(sent.decode("utf-8"))
    key = "input" if wire == "openai_responses" else "messages"
    expect(case["upstream_status"] == 200,
           f"{case['case_id']}: did not reach upstream", failures)
    expect(sent != raw, f"{case['case_id']}: forwarded original bytes", failures)
    expect([item["role"] for item in reduced[key]] == expected_roles,
           f"{case['case_id']}: wrong remaining roles", failures)
    expect(EVIDENCE_TEXT.encode() in sent,
           f"{case['case_id']}: lost evidence", failures)
    expect(NOISE_TEXT.encode() not in sent,
           f"{case['case_id']}: retained noise", failures)
    expect(manifest_header is not None,
           f"{case['case_id']}: lacks restore manifest", failures)
    expect(case["restore_round_trip"]["original_request_sha256_matches"] is True,
           f"{case['case_id']}: restore original hash mismatch", failures)
    expect(case["restore_round_trip"]["reduced_request_sha256_matches"] is True,
           f"{case['case_id']}: restore reduced hash mismatch", failures)
    expect(case["restore_header_contains_raw_text"] is False,
           f"{case['case_id']}: restore header leaked raw text", failures)
    expect(case["applied_pointers"] == [pointer],
           f"{case['case_id']}: applied pointer mismatch", failures)


def run_phase0():
    raw = encoded_request()
    failures = []
    cases = []

    eligible_sidecar = {"segments": {drop_pointer("openai_chat"): {"eligible": True}}}
    case, sent, manifest_header = run_case(
        "active_eligible_drop", mode="active", sidecar=eligible_sidecar)
    expect_active_drop(case, sent, raw, manifest_header, "openai_chat",
                       ["system", "assistant", "user"], failures)
    cases.append(case)

    case, sent, manifest_header = run_case(
        "provider_accounting_baseline", mode="active", sidecar=eligible_sidecar,
        headers={BASELINE_HEADER: "500"})
    expect_active_drop(case, sent, raw, manifest_header, "openai_chat",
                       ["system", "assistant", "user"], failures)
    expect(case["savings_claim"] == "actual",
           "provider baseline case did not record actual claim", failures)
    expect(case["savings_basis"] == "provider_paired_baseline",
           "provider baseline case basis mismatch", failures)
    cases.append(case)

    anthropic_raw = encoded_request("anthropic_messages")
    anthropic_sidecar = {"segments": {drop_pointer("anthropic_messages"): {"eligible": True}}}
    case, sent, manifest_header = run_case(
        "anthropic_eligible_drop", mode="active", wire="anthropic_messages",
        sidecar=anthropic_sidecar)
    expect_active_drop(case, sent, anthropic_raw, manifest_header,
                       "anthropic_messages", ["assistant", "user"], failures)
    expect(case["wire_format"] == "anthropic_messages",
           "anthropic case wire format mismatch", failures)
    cases.append(case)

    responses_raw = encoded_request("openai_responses")
    responses_sidecar = {"segments": {drop_pointer("openai_responses"): {"eligible": True}}}
    case, sent, manifest_header = run_case(
        "responses_eligible_drop", mode="active", wire="openai_responses",
        sidecar=responses_sidecar)
    expect_active_drop(case, sent, responses_raw, manifest_header,
                       "openai_responses", ["assistant", "user"], failures)
    expect(case["wire_format"] == "openai_responses",
           "responses case wire format mismatch", failures)
    cases.append(case)

    unsupported_raw = json.dumps({"model": "embedding-fake", "input": "embed this"}).encode()
    case, sent, manifest_header = run_case(
        "unsupported_embeddings", mode="active", path="/v1/embeddings",
        raw=unsupported_raw)
    expect(sent == unsupported_raw, "unsupported path modified bytes", failures)
    expect(manifest_header is None, "unsupported path emitted restore manifest", failures)
    expect(case["forward_reason"] == "unsupported_wire_format",
           "unsupported path reason mismatch", failures)
    cases.append(case)

    case, sent, manifest_header = run_case(
        "active_no_reduction", mode="active")
    expect(sent == raw, "no-reduction case modified upstream bytes", failures)
    expect(manifest_header is None, "no-reduction case emitted restore manifest", failures)
    expect(case["forward_reason"] == "active_no_reduction",
           "no-reduction case reason mismatch", failures)
    cases.append(case)

    case, sent, manifest_header = run_case(
        "shadow_control", mode="shadow", sidecar=eligible_sidecar)
    expect(sent == raw, "shadow control modified upstream bytes", failures)
    expect(manifest_header is None, "shadow control emitted restore manifest", failures)
    expect(case["forward_reason"] == "shadow_mode", "shadow control reason mismatch", failures)
    cases.append(case)

    case, sent, manifest_header = run_case(
        "kill_switch", mode="active", kill_switch=True, sidecar=eligible_sidecar)
    expect(sent == raw, "kill switch modified upstream bytes", failures)
    expect(manifest_header is None, "kill switch emitted restore manifest", failures)
    expect(case["forward_reason"] == "kill_switch", "kill switch reason mismatch", failures)
    cases.append(case)

    case, sent, manifest_header = run_case(
        "scorer_error", mode="active", scorer=failing_scorer, sidecar=eligible_sidecar)
    expect(sent == raw, "scorer error modified upstream bytes", failures)
    expect(manifest_header is None, "scorer error emitted restore manifest", failures)
    expect(case["gate_reason"] == "scorer_error", "scorer error gate reason mismatch", failures)
    expect(case["scorer_failure_kind"] == "exception",
           "scorer error kind mismatch", failures)
    cases.append(case)

    malformed = base64.b64encode(b"not-json").decode("ascii")
    case, sent, manifest_header = run_case(
        "malformed_sidecar", mode="active", headers={SIDECAR_HEADER: malformed})
    expect(sent == raw, "malformed sidecar modified upstream bytes", failures)
    expect(manifest_header is None, "malformed sidecar emitted restore manifest", failures)
    expect(case["gate_reason"] == "caller_bypass",
           "malformed sidecar gate reason mismatch", failures)
    cases.append(case)

    dependency_sidecar = {"segments": {
        "/messages/1/content": {"eligible": True},
        "/messages/2/content": {"eligible": True,
                                "depends_on": ["/messages/1/content"]},
    }}
    case, sent, manifest_header = run_case(
        "dependency_closure", mode="active", scorer=dependency_scorer,
        sidecar=dependency_sidecar)
    expect(sent == raw, "dependency closure modified upstream bytes", failures)
    expect(manifest_header is None, "dependency closure emitted restore manifest", failures)
    expect(case["forward_reason"] == "active_no_reduction",
           "dependency closure reason mismatch", failures)
    cases.append(case)

    case, sent, manifest_header = run_case(
        "uncertain_score", mode="active", scorer=uncertain_scorer,
        sidecar=eligible_sidecar)
    expect(sent == raw, "uncertain score modified upstream bytes", failures)
    expect(manifest_header is None, "uncertain score emitted restore manifest", failures)
    expect(case["gate_reason"] == "shadow_only" and
           case["forward_reason"] == "active_no_reduction",
           "uncertain score did not retain-and-forward", failures)
    cases.append(case)

    case, sent, manifest_header = run_case(
        "invalid_score_response", mode="active", scorer=invalid_scorer,
        sidecar=eligible_sidecar)
    expect(sent == raw, "invalid score modified upstream bytes", failures)
    expect(manifest_header is None, "invalid score emitted restore manifest", failures)
    expect(case["gate_reason"] == "invalid_score_response",
           "invalid score gate reason mismatch", failures)
    cases.append(case)

    case, sent, manifest_header = run_case(
        "scorer_timeout", mode="active",
        scorer=DeadlineScorer(slow_scorer, timeout=0.01),
        sidecar=eligible_sidecar)
    expect(sent == raw, "scorer timeout modified upstream bytes", failures)
    expect(manifest_header is None, "scorer timeout emitted restore manifest", failures)
    expect(case["gate_reason"] == "scorer_error",
           "scorer timeout gate reason mismatch", failures)
    expect(case["scorer_failure_kind"] == "timeout",
           "scorer timeout kind mismatch", failures)
    cases.append(case)

    case, sent, manifest_header = run_case(
        "restore_manifest_too_large", mode="active", sidecar=eligible_sidecar,
        restore_header_limit=1)
    expect(sent == raw, "oversized restore manifest modified bytes", failures)
    expect(manifest_header is None, "oversized restore manifest was emitted", failures)
    expect(case["forward_reason"] == "restore_manifest_header_too_large",
           "oversized restore manifest reason mismatch", failures)
    cases.append(case)

    case, sent, manifest_header = run_case(
        "oversized_request_body", mode="active", max_body_bytes=len(raw) - 1)
    expect(case["upstream_status"] == 413, "oversized request did not fail 413", failures)
    expect(case["upstream_request_count"] == 0,
           "oversized request reached upstream", failures)
    expect(manifest_header is None, "oversized request emitted restore manifest", failures)
    cases.append(case)

    sidecar_header = base64.b64encode(json.dumps(eligible_sidecar).encode()).decode("ascii")
    case, sent, manifest_header = run_case(
        "internal_header_stripping", mode="active",
        headers={SIDECAR_HEADER: sidecar_header, "X-NanoJev-Debug": "1"})
    expect_active_drop(case, sent, raw, manifest_header, "openai_chat",
                       ["system", "assistant", "user"], failures)
    expect(case["upstream_internal_headers"] == [],
           "internal X-NanoJev headers reached upstream", failures)
    cases.append(case)

    receipt = {
        "schema_version": "nanojev-active-mode-phase0-v1",
        "status": "phase0_pass" if not failures else "phase0_fail",
        "provider_calls": 0,
        "upstream": "loopback_fake_only",
        "cases": cases,
        "failures": failures,
        "ok": not failures,
    }
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    receipt = run_phase0()
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "sha256": sha256_text(encoded), "ok": receipt["ok"],
                      "status": receipt["status"], "cases": len(receipt["cases"])},
                     indent=2))
    if not receipt["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
