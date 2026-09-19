#!/usr/bin/env python3
"""Main-model gateway: run the byte-preserving context gate in front of a provider.

This is the forwarding piece the shadow library deliberately lacked. It exposes
OpenAI-compatible ``POST /v1/chat/completions`` and Anthropic-compatible
``POST /v1/messages`` endpoints (plus ``POST /v1/responses``, a format the existing
shadow core already supports) and forwards to a configured upstream base URL.

Safety posture:

* **Shadow mode is the default.** The original request bytes are forwarded unchanged and
  the upstream response is returned unmodified. The gate only records a hypothetical
  removal plan.
* **Active mode is explicit opt-in** (``--mode active`` / ``GatewayConfig(mode="active")``)
  and always fails open. Any gate exception, timeout, partial/malformed/nonfinite score,
  protected segment in the removal set, or reduction error forwards the ORIGINAL bytes.
* **Every applied reduction is reversible.** When a reduction is actually applied, the
  gateway builds a content-free restore manifest
  (:mod:`context_restore_v1`) and returns it to the caller in the
  ``x-nanojev-restore-manifest`` response header. The header carries pointers, wire
  format, and SHA-256 hashes only -- never prompt text -- and the caller, which retained
  the removed segments, can reconstruct the original request exactly. A reduction whose
  manifest cannot be built fails open instead of being sent.
* **A deterministic kill switch** (``--kill-switch`` or ``NANOJEV_GATEWAY_KILL_SWITCH``)
  forwards everything unchanged without scoring.
* All parsing and protection decisions come from :func:`context_gate_v1.shadow_request`.
  This module never re-implements pointer parsing or protection policy.

The gateway forwards caller headers (including provider credentials) opaquely to the
upstream and never reads, stores, or logs them. Receipts are content-free JSONL.
Standard library only. No real provider is contacted by the test suite.
"""

from dataclasses import dataclass, field
from http.client import HTTPConnection, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlsplit
import argparse
import base64
import binascii
import json
import math
import os
import re
import sys
import threading
import time
import uuid

from context_gate_v1 import fingerprint, parse_segments, serialized, shadow_request
from context_restore_v1 import build_restore_manifest, canonical_bytes, restore_request
from predict_toy_decisions import reject_nonfinite, unique_object
from scorer_adapters_v1 import (
    DeadlineScorer, FailureRecordingScorer, build_scorer,
)


SCHEMA_VERSION = "nanojev-main-model-gateway-v1"
DEFAULT_THRESHOLD = 0.99  # provisional diagnostic; not a calibration claim
DEFAULT_LISTEN_HOST = "127.0.0.1"
DEFAULT_LISTEN_PORT = 8790
MODE_SHADOW = "shadow"
MODE_ACTIVE = "active"
MODES = (MODE_SHADOW, MODE_ACTIVE)

WIRE_BY_PATH = {
    "/v1/chat/completions": "openai_chat",
    "/v1/messages": "anthropic_messages",
    "/v1/responses": "openai_responses",
}

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-length", "host",
}
INTERNAL_PREFIX = "x-nanojev-"
SIDECAR_HEADER = "x-nanojev-sidecar"
BASELINE_HEADER = "x-nanojev-baseline-provider-prompt-tokens"
# Emitted to the CALLER only when a reduction is actually applied. It describes how to
# reconstruct the original request and contains no raw prompt text: pointers, the wire
# format, and SHA-256 hashes only. The removed segment content always stays with the caller.
RESTORE_MANIFEST_HEADER = "x-nanojev-restore-manifest"
KILL_SWITCH_ENV = "NANOJEV_GATEWAY_KILL_SWITCH"

# Only an integration-supplied eligible assistant text segment may ever use this reason.
APPLIED_DROP_REASON = "high_irrelevance_score"
REDUCTION_POINTER = re.compile(r"^/(messages|input)/(\d+)/content(?:/(\d+)/text)?$")

WHITESPACE_TOKENIZER_ID = "whitespace-word-split-v1-not-a-provider-tokenizer"


def whitespace_word_counter(text):
    """Deterministic, dependency-free token *estimate*; explicitly not a BPE tokenizer."""
    return len(text.split()) if isinstance(text, str) else 0


class UpstreamError(RuntimeError):
    """The configured upstream could not be reached or returned an unusable transport."""


class ReductionError(ValueError):
    """The removal plan could not be applied safely; the gateway must fail open."""


@dataclass
class GatewayConfig:
    upstream_base_url: str
    listen_host: str = DEFAULT_LISTEN_HOST
    listen_port: int = DEFAULT_LISTEN_PORT
    mode: str = MODE_SHADOW
    kill_switch: bool = False
    score_timeout: float = 5.0
    upstream_timeout: float = 60.0
    receipt_log: Path | None = None
    scorer: object | None = None
    sidecar: dict = field(default_factory=dict)
    token_counter: object | None = None
    tokenizer_id: str | None = None
    threshold: float = DEFAULT_THRESHOLD
    max_body_bytes: int = 8_000_000
    max_upstream_bytes: int = 64_000_000

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError("mode must be 'shadow' or 'active'")
        try:
            if not ip_address(self.listen_host).is_loopback:
                raise ValueError("listen_host must be a loopback address")
        except ValueError as error:
            raise ValueError("listen_host must be a literal loopback IP address") from error
        if type(self.listen_port) is not int or not 0 <= self.listen_port <= 65535:
            raise ValueError("listen_port must be an integer in [0,65535]")
        for name in ("score_timeout", "upstream_timeout"):
            value = getattr(self, name)
            if type(value) not in {int, float} or not math.isfinite(value) or not 0 < value <= 3600:
                raise ValueError(f"{name} must be finite and in (0,3600] seconds")
        if type(self.threshold) not in {int, float} or not math.isfinite(self.threshold) or not 0.5 < self.threshold <= 1:
            raise ValueError("threshold must be finite and in (0.5,1]")
        if self.token_counter is not None and not callable(self.token_counter):
            raise ValueError("token_counter must be callable")
        parsed = urlsplit(self.upstream_base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
            raise ValueError("upstream_base_url must be a bare http(s) origin without credentials, path, query, or fragment")
        if type(self.max_body_bytes) is not int or self.max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be a positive integer")
        if type(self.max_upstream_bytes) is not int or self.max_upstream_bytes <= 0:
            raise ValueError("max_upstream_bytes must be a positive integer")


@dataclass
class Reply:
    status: int
    headers: list
    body: bytes


def get_header(headers, name):
    target = name.lower()
    for key, value in headers:
        if key.lower() == target:
            return value
    return None


def decode_sidecar(headers):
    """Trusted caller sidecar from an internal header; malformed input forces a bypass."""
    raw = get_header(headers, SIDECAR_HEADER)
    if raw is None:
        return None
    try:
        decoded = base64.b64decode(raw.encode("ascii"), validate=True)
        if len(decoded) > 64_000:
            raise ValueError("sidecar too large")
        value = json.loads(decoded.decode("utf-8"), object_pairs_hook=unique_object, parse_constant=reject_nonfinite)
        if not isinstance(value, dict):
            raise ValueError("sidecar must be an object")
        return value
    except (ValueError, UnicodeError, binascii.Error):
        # Fail open: an unreadable sidecar must never grant eligibility or drop protection.
        return {"bypass": True}


def merge_sidecar(base, override):
    if not base:
        return override or {}
    if not override:
        return base
    if base.get("bypass") or override.get("bypass"):
        return {"bypass": True}
    segments = {**base.get("segments", {}), **override.get("segments", {})}
    return {**base, **override, "segments": segments}


def decode_baseline_prompt_tokens(headers):
    """Caller-supplied provider-reported baseline for paired token accounting."""
    raw = get_header(headers, BASELINE_HEADER)
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def removal_plan(gate_receipt):
    """Return ``(drop_pointers, error)``. Any protected/unknown drop means fail open."""
    segments = gate_receipt.get("segments") if isinstance(gate_receipt, dict) else None
    if not isinstance(segments, list):
        return [], "malformed_receipt"
    drops = []
    for segment in segments:
        if not isinstance(segment, dict):
            return [], "malformed_receipt"
        if segment.get("suggestion") != "drop":
            continue
        if segment.get("role") != "assistant" or segment.get("reason") != APPLIED_DROP_REASON:
            return [], "protected_segment_in_removal_set"
        pointer = segment.get("pointer")
        if not isinstance(pointer, str):
            return [], "malformed_receipt"
        drops.append(pointer)
    return drops, None


def build_reduced_request(raw, wire_format, drop_pointers):
    """Apply ONLY the gate's drop pointers to a freshly parsed copy of the request.

    Every pointer must resolve to an eligible assistant text segment. The reduced body is
    re-validated with the core parser before it is allowed onto the wire.
    """
    if not drop_pointers:
        raise ReductionError("empty removal plan")
    body = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object, parse_constant=reject_nonfinite)
    key = "input" if wire_format == "openai_responses" else "messages"
    items = body.get(key)
    if not isinstance(items, list):
        raise ReductionError("missing message list")
    drop_messages, drop_parts = set(), {}
    for pointer in drop_pointers:
        match = REDUCTION_POINTER.match(pointer)
        if match is None or match.group(1) != key:
            raise ReductionError("unsupported_drop_pointer")
        index = int(match.group(2))
        if index >= len(items):
            raise ReductionError("drop_pointer_out_of_range")
        if match.group(3) is None:
            drop_messages.add(index)
        else:
            drop_parts.setdefault(index, set()).add(int(match.group(3)))
    reduced = []
    for index, item in enumerate(items):
        if index in drop_messages:
            continue
        part_indexes = drop_parts.get(index)
        if part_indexes:
            if not isinstance(item, dict) or not isinstance(item.get("content"), list):
                raise ReductionError("drop_pointer_target_not_a_list")
            kept = [part for position, part in enumerate(item["content"]) if position not in part_indexes]
            if not kept:
                continue  # an emptied message is removed whole
            item = {**item, "content": kept}
        reduced.append(item)
    if not reduced:
        raise ReductionError("reduced_request_would_be_empty")
    if not any(isinstance(item, dict) and item.get("role") == "user" for item in reduced):
        raise ReductionError("reduced_request_would_lose_user_intent")
    reduced_body = {**body, key: reduced}
    return json.dumps(reduced_body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def removed_segments_from_raw(raw, wire_format, drop_pointers):
    """Recover the values a removal plan takes out, from the caller's own original bytes.

    Used ONLY for the in-process reversibility self-check before a reduction is sent. The
    gateway does not persist, log, or transmit the result; the caller remains the sole
    holder of removed content.
    """
    body = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object, parse_constant=reject_nonfinite)
    key = "input" if wire_format == "openai_responses" else "messages"
    items = body.get(key)
    if not isinstance(items, list):
        raise ReductionError("missing message list")
    recovered = {}
    for pointer in drop_pointers:
        match = REDUCTION_POINTER.match(pointer)
        if match is None or match.group(1) != key:
            raise ReductionError("unsupported_drop_pointer")
        index = int(match.group(2))
        if index >= len(items) or not isinstance(items[index], dict):
            raise ReductionError("drop_pointer_out_of_range")
        part = match.group(3)
        if part is None:
            # Whole-message drop: a single-text message object is what the restorer accepts.
            recovered[pointer] = items[index]
        else:
            content = items[index].get("content")
            if not isinstance(content, list):
                raise ReductionError("drop_pointer_target_not_a_list")
            part_index = int(part)
            if part_index >= len(content) or not isinstance(content[part_index], dict):
                raise ReductionError("drop_pointer_out_of_range")
            text = content[part_index].get("text")
            if not isinstance(text, str):
                raise ReductionError("drop_pointer_target_not_text")
            recovered[pointer] = text
    return recovered


def provider_reported_usage(wire_format, body):
    """Extract provider-reported usage counts. Never returns any response text."""
    try:
        data = json.loads(body.decode("utf-8"), object_pairs_hook=unique_object, parse_constant=reject_nonfinite)
    except (ValueError, UnicodeError):
        return None
    if not isinstance(data, dict):
        return None
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    if wire_format == "anthropic_messages":
        prompt_field = "usage.input_tokens" if "input_tokens" in usage else None
        completion_field = "usage.output_tokens" if "output_tokens" in usage else None
    else:
        prompt_field = "usage.prompt_tokens" if "prompt_tokens" in usage else (
            "usage.input_tokens" if "input_tokens" in usage else None)
        completion_field = "usage.completion_tokens" if "completion_tokens" in usage else (
            "usage.output_tokens" if "output_tokens" in usage else None)
    prompt = usage.get(prompt_field.split(".")[-1]) if prompt_field else None
    completion = usage.get(completion_field.split(".")[-1]) if completion_field else None
    if type(prompt) is not int or prompt < 0:
        prompt, prompt_field = None, None
    if type(completion) is not int or completion < 0:
        completion, completion_field = None, None
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "prompt_field": prompt_field, "completion_field": completion_field}


class ContextGateGateway:
    """Wire the shadow core into an HTTP forwarding path. One instance serves one process."""

    def __init__(self, config):
        self.config = config
        self._receipt_lock = threading.Lock()

    # -- request handling -------------------------------------------------------------

    def handle(self, method, path, headers, raw):
        started = time.perf_counter()
        path_only = path.partition("?")[0]
        wire_format = WIRE_BY_PATH.get(path_only)
        kill = bool(self.config.kill_switch)
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "event_type": "context_gate_forward",
            "event_id": str(uuid.uuid4()),
            "mode": self.config.mode,
            "kill_switch": kill,
            "wire_format": wire_format,
            "request_method": method,
            "path": path_only,
            "request_sha256": fingerprint(raw),
            "forwarded_sha256": fingerprint(raw),
            "forwarded_unchanged": True,
            "forward_reason": None,
            "policy_threshold": self.config.threshold,
            "gate_receipt": None,
            "gate_latency_ms": None,
            "scorer_failure_kind": None,
            "removal_plan": {"proposed_pointers": [], "applied_pointers": [], "applied": False, "error": None},
            "upstream": {"attempted": False, "status": None, "error": None},
            "token_accounting": self._token_accounting_skeleton(),
        }

        fail_open_reason = None
        if method != "POST":
            fail_open_reason = "unsupported_method"
        elif wire_format is None:
            fail_open_reason = "unsupported_wire_format"
        elif kill:
            fail_open_reason = "kill_switch"

        forwarded = raw
        proposed = []
        recorder = None
        restore_manifest = None
        if fail_open_reason is None:
            sidecar = self.config.sidecar
            header_sidecar = decode_sidecar(headers)
            if header_sidecar is not None:
                sidecar = merge_sidecar(self.config.sidecar, header_sidecar)
            scorer = self.config.scorer
            if scorer is not None:
                recorder = FailureRecordingScorer(DeadlineScorer(scorer, self.config.score_timeout))
                scorer = recorder
            try:
                _, gate_receipt = shadow_request(
                    raw, wire_format, sidecar, scorer, threshold=self.config.threshold,
                    token_counter=self.config.token_counter, tokenizer_id=self.config.tokenizer_id)
            except Exception:  # noqa: BLE001 - any gate failure must fail open
                gate_receipt = None
                fail_open_reason = "gate_error"
            if gate_receipt is not None:
                receipt["gate_receipt"] = gate_receipt
                receipt["gate_latency_ms"] = gate_receipt.get("latency_ms")
                proposed, plan_error = removal_plan(gate_receipt)
                receipt["removal_plan"]["proposed_pointers"] = proposed
                receipt["removal_plan"]["error"] = plan_error
                if plan_error is not None and self.config.mode == MODE_ACTIVE:
                    fail_open_reason = plan_error
                elif self.config.mode == MODE_ACTIVE and proposed:
                    try:
                        candidate = build_reduced_request(raw, wire_format, proposed)
                        parse_segments(candidate, wire_format)  # core re-validation, not a reimplementation
                        # Reversibility is proven before the reduction is allowed onto the
                        # wire. Building a manifest is NOT sufficient: the manifest can be
                        # constructible while the reconstruction still fails, which would
                        # send a reduction the caller cannot undo. So the real round-trip
                        # is executed here against the caller's own original bytes, and any
                        # failure fails open. The recovered segments are used only for this
                        # in-process check and are never persisted or logged.
                        candidate_manifest = build_restore_manifest(raw, wire_format, proposed)
                        recovered = removed_segments_from_raw(raw, wire_format, proposed)
                        # restore_request returns the CANONICAL serialization, so compare it
                        # against the canonical form of the caller's original. A non-canonical
                        # client body (pretty-printed JSON, different key order) still restores
                        # correctly and must not be treated as a failure.
                        original_body = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                                                   parse_constant=reject_nonfinite)
                        if restore_request(candidate, candidate_manifest, recovered) != canonical_bytes(original_body):
                            raise ReductionError("restore_round_trip_mismatch")
                    except Exception:  # noqa: BLE001 - malformed or irreversible reduction must fail open
                        fail_open_reason = "reduction_error"
                    else:
                        forwarded = candidate
                        restore_manifest = candidate_manifest
            if recorder is not None and recorder.last_failure is not None:
                receipt["scorer_failure_kind"] = recorder.last_failure

        if fail_open_reason is not None:
            forwarded = raw
            receipt["forward_reason"] = fail_open_reason
            receipt["forwarded_unchanged"] = True
        elif forwarded is raw:
            receipt["forward_reason"] = "shadow_mode" if self.config.mode == MODE_SHADOW else "active_no_reduction"
            receipt["forwarded_unchanged"] = True
        else:
            receipt["forward_reason"] = "active_reduced"
            receipt["forwarded_unchanged"] = False
            receipt["removal_plan"]["applied"] = True
            receipt["removal_plan"]["applied_pointers"] = list(proposed)

        receipt["forwarded_sha256"] = fingerprint(forwarded)
        try:
            reply = self._forward(method, path, headers, forwarded)
        except UpstreamError:
            reply = Reply(502, [("Content-Type", "application/json")],
                          b'{"error":{"type":"nanojev_upstream_unreachable",'
                          b'"message":"gateway could not reach the configured upstream"}}')
            receipt["upstream"] = {"attempted": True, "status": None, "error": "unreachable"}
        else:
            receipt["upstream"] = {"attempted": True, "status": reply.status, "error": None}

        usage = provider_reported_usage(wire_format, reply.body) if wire_format else None
        receipt["token_accounting"] = self._token_accounting(
            raw, forwarded, receipt["gate_receipt"], receipt["forwarded_unchanged"] is False,
            usage, decode_baseline_prompt_tokens(headers))
        receipt["total_latency_ms"] = (time.perf_counter() - started) * 1000
        self._write_receipt(receipt)
        if restore_manifest is not None:
            # Additive response metadata for the caller that retained the removed segments.
            # The manifest is content-free, and it is never sent upstream.
            reply.headers = list(reply.headers) + [
                (RESTORE_MANIFEST_HEADER, restore_manifest.to_json())]
        return reply

    # -- transport --------------------------------------------------------------------

    def _forward(self, method, path, headers, raw):
        target = urlsplit(self.config.upstream_base_url)
        request_path = path if path.startswith("/") else "/" + path
        outgoing = []
        for name, value in headers:
            lower = name.lower()
            if lower in HOP_BY_HOP or lower.startswith(INTERNAL_PREFIX):
                continue
            outgoing.append((name, value))
        if target.scheme == "https":
            connection = HTTPSConnection(target.hostname, target.port or 443, timeout=self.config.upstream_timeout)
        else:
            connection = HTTPConnection(target.hostname, target.port or 80, timeout=self.config.upstream_timeout)
        try:
            # http.client never uses environment proxies and never follows redirects.
            connection.request(method, request_path, body=raw, headers=dict(outgoing))
            response = connection.getresponse()
            body = response.read(self.config.max_upstream_bytes + 1)
            if len(body) > self.config.max_upstream_bytes:
                raise UpstreamError("upstream response budget exceeded")
        except (OSError, ValueError) as error:
            raise UpstreamError("upstream transport failed") from error
        finally:
            connection.close()
        response_headers = [(name, value) for name, value in response.getheaders()
                            if name.lower() not in HOP_BY_HOP]
        return Reply(response.status, response_headers, body)

    # -- receipts ---------------------------------------------------------------------

    @staticmethod
    def _token_accounting_skeleton():
        return {
            "scope": "content_free_accounting",
            "estimate": {"tokenizer_id": None, "tokenizer_sha256": None,
                         "removed_segment_text_tokens": None,
                         "original_body_text_tokens": None, "reduced_body_text_tokens": None,
                         "note": "local_estimate_not_provider_billing"},
            "provider_reported": {"source": "upstream_response", "prompt_tokens": None,
                                  "completion_tokens": None, "usage_field": None},
            "savings": {"claim": "none", "tokens": None, "basis": "no_reduction_sent"},
        }

    def _token_accounting(self, raw, forwarded, gate_receipt, was_reduced, usage, baseline):
        accounting = self._token_accounting_skeleton()
        estimate = accounting["estimate"]
        token_counts = (gate_receipt or {}).get("token_counts") or {}
        removed = token_counts.get("proposed_text_tokens")
        estimate["removed_segment_text_tokens"] = removed if type(removed) is int and removed >= 0 else None
        estimate["tokenizer_sha256"] = token_counts.get("tokenizer_sha256")
        estimate["tokenizer_id"] = self.config.tokenizer_id
        counter = self.config.token_counter
        if counter is not None:
            estimate["original_body_text_tokens"] = self._count(counter, raw)
            if was_reduced:
                estimate["reduced_body_text_tokens"] = self._count(counter, forwarded)
        if usage is not None:
            accounting["provider_reported"] = {
                "source": "upstream_response",
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "usage_field": usage.get("prompt_field"),
            }
        provider_prompt = (usage or {}).get("prompt_tokens")
        savings = accounting["savings"]
        if not was_reduced and self.config.mode == MODE_SHADOW:
            # Shadow mode sends nothing reduced, so it may only ever report an ESTIMATE,
            # explicitly labelled with the tokenizer identity used.
            if removed is not None:
                savings.update(claim="estimate", tokens=removed, basis="shadow_estimate_only")
            else:
                savings.update(claim="none", tokens=None, basis="estimate_unavailable")
        elif not was_reduced:
            savings.update(claim="none", tokens=None, basis="no_reduction_sent")
        elif provider_prompt is not None and baseline is not None:
            # Both ends are provider-reported: the caller's measured baseline for the
            # unfiltered request minus the provider's count for the reduced request.
            savings.update(claim="actual", tokens=baseline - provider_prompt,
                           basis="provider_paired_baseline")
        elif removed is not None:
            savings.update(claim="estimate", tokens=removed,
                           basis="provider_usage_observed_no_paired_baseline" if provider_prompt is not None
                           else "provider_usage_unavailable")
        else:
            savings.update(claim="none", tokens=None,
                           basis="estimate_unavailable")
        return accounting

    @staticmethod
    def _count(counter, raw):
        try:
            value = counter(raw.decode("utf-8", "replace"))
        except Exception:  # noqa: BLE001 - an estimate failure must never break forwarding
            return None
        return value if type(value) is int and value >= 0 else None

    def _write_receipt(self, receipt):
        if self.config.receipt_log is None:
            return
        try:
            line = serialized(receipt) + "\n"
        except (ValueError, TypeError):
            return
        with self._receipt_lock:
            try:
                path = Path(self.config.receipt_log)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(line)
            except OSError:
                # Receipt persistence failure must not take the forwarding path down.
                return


class _GatewayRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "NanoJevGateway/1.0"

    def _run(self):
        gateway = self.server.gateway
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length) if raw_length is not None else 0
        except (TypeError, ValueError):
            self._send(Reply(400, [("Content-Type", "application/json")], b'{"error":{"type":"invalid_content_length"}}'))
            return
        if length < 0 or length > gateway.config.max_body_bytes:
            self._send(Reply(413, [("Content-Type", "application/json")], b'{"error":{"type":"request_too_large"}}'))
            return
        raw = self.rfile.read(length) if length else b""
        reply = gateway.handle(self.command, self.path, list(self.headers.items()), raw)
        self._send(reply)

    do_POST = _run
    do_GET = _run
    do_PUT = _run
    do_PATCH = _run
    do_DELETE = _run

    def _send(self, reply):
        self.send_response_only(reply.status)
        for name, value in reply.headers:
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(reply.body)))
        self.end_headers()
        if self.command != "HEAD" and reply.body:
            self.wfile.write(reply.body)

    def log_message(self, *args):  # never log request lines or headers
        return


def make_server(gateway):
    server = ThreadingHTTPServer((gateway.config.listen_host, gateway.config.listen_port), _GatewayRequestHandler)
    server.gateway = gateway
    return server


def _kill_switch_from_env():
    value = os.environ.get(KILL_SWITCH_ENV)
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upstream", required=True,
                        help="bare upstream origin, e.g. https://api.openai.com or http://127.0.0.1:8080")
    parser.add_argument("--listen-host", default=DEFAULT_LISTEN_HOST)
    parser.add_argument("--listen-port", type=int, default=DEFAULT_LISTEN_PORT)
    parser.add_argument("--mode", choices=MODES, default=MODE_SHADOW,
                        help="shadow (default, forwards original bytes) or active (explicit opt-in)")
    parser.add_argument("--kill-switch", action="store_true", help="forward everything unchanged; no scoring")
    parser.add_argument("--score-timeout", type=float, default=5.0)
    parser.add_argument("--upstream-timeout", type=float, default=60.0)
    parser.add_argument("--receipt-log", type=Path, required=True, help="JSONL path for content-free receipts")
    parser.add_argument("--scorer", choices=("none", "http", "laya", "inprocess"), default="none")
    parser.add_argument("--scorer-url", default=None)
    parser.add_argument("--estimate-tokenizer", choices=("none", "word"), default="word",
                        help="labelled local ESTIMATE only; never provider billing")
    args = parser.parse_args(argv)
    token_counter = whitespace_word_counter if args.estimate_tokenizer == "word" else None
    tokenizer_id = WHITESPACE_TOKENIZER_ID if token_counter else None
    config = GatewayConfig(
        upstream_base_url=args.upstream, listen_host=args.listen_host, listen_port=args.listen_port,
        mode=args.mode, kill_switch=args.kill_switch or _kill_switch_from_env(),
        score_timeout=args.score_timeout, upstream_timeout=args.upstream_timeout,
        receipt_log=args.receipt_log,
        scorer=build_scorer(args.scorer, url=args.scorer_url, timeout=args.score_timeout),
        token_counter=token_counter, tokenizer_id=tokenizer_id)
    server = make_server(ContextGateGateway(config))
    host, port = server.server_address[0], server.server_address[1]
    print(json.dumps({"schema_version": SCHEMA_VERSION, "listen": f"http://{host}:{port}",
                      "upstream": args.upstream, "mode": config.mode,
                      "kill_switch": config.kill_switch, "scorer": args.scorer,
                      "receipt_log": str(config.receipt_log)}, sort_keys=True))
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
