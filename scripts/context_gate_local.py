#!/usr/bin/env python3
"""Explicit local shadow gate CLI and logged loopback adapter; no provider reconfiguration."""

import argparse
import http.client
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit
import uuid

from context_gate_v1 import shadow_request, serialized
from predict_toy_decisions import read_json, unique_object, reject_nonfinite


HELPER_PATH = Path(__file__).resolve().parents[1] / "integrations/codex-skill/nanojev-local-decider/scripts/nanojev_skill.py"


def skill_helper():
    spec = importlib.util.spec_from_file_location("nanojev_context_skill", HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LoggedScorer:
    """Reuse the existing usage-v1 receipt. Always suppress optional raw debug payloads."""

    def __init__(self, predict, checkpoint, log_path):
        self.predict = predict
        self.checkpoint = Path(checkpoint)
        self.log_path = Path(log_path)
        self.helper = skill_helper()

    def __call__(self, payload):
        start = time.perf_counter()
        result = self.predict(payload)
        normalized, gates = self.helper.normalize_payload(payload)
        compatible = self.helper.apply_response_compatibility(result, gates)
        event_id = str(uuid.uuid4())
        record = self.helper.usage_record(normalized, compatible, "context-gate", "shadow-context",
                                          self.checkpoint, (time.perf_counter() - start) * 1000,
                                          event_id)
        record.pop("raw_payload", None)
        # The helper's identity is configured, not attestation of a running HTTP server's weights.
        record["checkpoint_identity_scope"] = "configured_local_checkpoint_not_server_attestation"
        serialized(record)  # Reject non-finite metrics before using the legacy JSONL writer.
        self.helper.append_event(record, self.log_path)
        return {**result, "context_gate_usage_event_id": event_id}


class LoopbackPredictor:
    def __init__(self, url, timeout=5.0):
        parsed = urlsplit(url)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"}
                or parsed.username is not None or parsed.password is not None
                or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
            raise ValueError("only literal loopback HTTP origins are allowed")
        if type(timeout) not in {int, float} or not math.isfinite(timeout) or not 0 < timeout <= 30:
            raise ValueError("socket timeout must be in (0,30] seconds")
        self.host, self.port, self.timeout = parsed.hostname, parsed.port or 80, timeout

    def __call__(self, payload):
        body = serialized(payload).encode("utf-8")
        if len(body) > 2_000_000:
            raise ValueError("local request budget exceeded")
        # HTTPConnection never uses environment proxies and never follows redirects.
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            connection.request("POST", "/api/evaluate", body=body, headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 200:
                raise ValueError("local scoring request failed")
            content = response.read(2_000_001)
            if len(content) > 2_000_000:
                raise ValueError("local response budget exceeded")
            return json.loads(content, object_pairs_hook=unique_object, parse_constant=reject_nonfinite)
        finally:
            connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--format", choices=("openai_chat", "openai_responses", "anthropic_messages"), required=True)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--checkpoint", type=Path, default=HELPER_PATH.parents[4] / "checkpoints/local_atomic_seed17/variants/local_atomic_seed17")
    parser.add_argument("--log", type=Path, required=True, help="Explicit local usage JSONL; no raw request text")
    parser.add_argument("--receipt-log", type=Path, required=True, help="Explicit shadow receipt JSONL")
    args = parser.parse_args()
    raw = args.input.read_bytes()
    scorer = LoggedScorer(LoopbackPredictor(args.url, args.timeout), args.checkpoint, args.log)
    forwarded, receipt = shadow_request(raw, args.format, read_json(args.sidecar) if args.sidecar else {}, scorer)
    assert forwarded == raw
    skill_helper().append_event(receipt, args.receipt_log)
    print(json.dumps(receipt, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
