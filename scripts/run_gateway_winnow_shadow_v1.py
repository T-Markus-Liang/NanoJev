#!/usr/bin/env python3
"""Gateway shadow-mode dry-run against the live Winnow-12B Q8 scorer.

Same loopback topology as Phase0, but the scorer is the real native
``winnow-server`` on ``127.0.0.1:8091`` instead of an in-process stub.
Shadow mode guarantees original bytes upstream; this proves the full
integration path end to end. No provider calls.
"""

import argparse
import base64
import hashlib
import json
import time
from pathlib import Path

from scorer_adapters_v1 import SystemOneHTTPScorer, build_scorer
from run_active_mode_phase0_v1 import (
    SIDECAR_HEADER, drop_pointer, encoded_request, run_case,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "gateway_winnow_shadow_v1.json"
WINNOW_URL = "http://127.0.0.1:8091"
WINNOW_MODEL = "Winnow-12B"
FAST_URL = "http://127.0.0.1:8092"
FAST_MODEL = "kev-latest"


def sha256_text(data):
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def expect(condition, message, failures):
    if not condition:
        failures.append(message)


class CountingScorer:
    def __init__(self, scorer):
        self.scorer = scorer
        self.calls = 0
        self.latencies = []

    def __call__(self, payload):
        self.calls += 1
        started = time.perf_counter()
        result = self.scorer(payload)
        self.latencies.append(round((time.perf_counter() - started) * 1000, 1))
        return result


def sidecar_for(wire):
    return {"segments": {drop_pointer(wire): {"eligible": True}}}


def run_winnow_shadow(scorer_url=WINNOW_URL, scorer_kind="systemone",
                      strong_url=None):
    failures, cases = [], []
    if scorer_kind == "cascade":
        inner = build_scorer("cascade", url=scorer_url, strong_url=strong_url,
                             timeout=30.0, model_id=FAST_MODEL,
                             strong_model_id=WINNOW_MODEL)
    else:
        inner = SystemOneHTTPScorer(scorer_url, timeout=30.0,
                                  model_id=WINNOW_MODEL)
    scorer = CountingScorer(inner)

    def shadow_case(name, wire="openai_chat", sidecar=None, **kwargs):
        raw = encoded_request(wire)
        case, sent, manifest_header = run_case(
            name, mode="shadow", wire=wire, scorer=scorer,
            sidecar=sidecar if sidecar is not None else sidecar_for(wire),
            **kwargs)
        expect(sent == raw, f"{name}: shadow changed bytes", failures)
        expect(manifest_header is None, f"{name}: restore manifest emitted", failures)
        expect(case["applied_pointers"] == [],
               f"{name}: shadow applied pointers", failures)
        return case, raw

    for wire in ("openai_chat", "anthropic_messages", "openai_responses"):
        case, _ = shadow_case(f"winnow_shadow_{wire}", wire=wire)
        expect(case["forward_reason"] == "shadow_mode",
               f"{wire}: forward reason mismatch", failures)
        expect(case["gate_reason"] == "shadow_only",
               f"{wire}: expected scored shadow receipt, got {case['gate_reason']}",
               failures)
        cases.append(case)

    case, _ = shadow_case("winnow_shadow_no_sidecar", sidecar={})
    expect(case["gate_reason"] == "no_eligible_segments",
           "no-sidecar gate reason mismatch", failures)
    cases.append(case)

    unsupported_raw = json.dumps({"model": "embedding-fake",
                                  "input": "embed this"}).encode()
    case, sent, manifest_header = run_case(
        "winnow_shadow_unsupported", mode="shadow", scorer=scorer,
        path="/v1/embeddings", raw=unsupported_raw)
    expect(sent == unsupported_raw, "unsupported path changed bytes", failures)
    expect(case["forward_reason"] == "unsupported_wire_format",
           "unsupported path reason mismatch", failures)
    cases.append(case)

    calls_before = scorer.calls
    case, _ = shadow_case("winnow_shadow_kill_switch", kill_switch=True)
    expect(case["forward_reason"] == "kill_switch",
           "kill switch reason mismatch", failures)
    expect(scorer.calls == calls_before,
           "kill switch still called the scorer", failures)
    cases.append(case)

    header = base64.b64encode(json.dumps(sidecar_for("openai_chat")).encode()).decode("ascii")
    case, sent, manifest_header = run_case(
        "winnow_shadow_internal_headers", mode="shadow", scorer=scorer,
        headers={SIDECAR_HEADER: header, "X-NanoJev-Debug": "1"})
    expect(sent == encoded_request("openai_chat"),
           "internal-header case changed bytes", failures)
    expect(case["upstream_internal_headers"] == [],
           "internal headers reached upstream", failures)
    cases.append(case)

    proposals = {case["case_id"]: case.get("proposed_pointers") or []
                 for case in cases}
    scored_cases = [case["case_id"] for case in cases
                    if case["gate_reason"] == "shadow_only"]
    return {
        "schema_version": "nanojev-gateway-winnow-shadow-v1",
        "status": "winnow_shadow_pass" if not failures else "winnow_shadow_fail",
        "provider_calls": 0,
        "scorer": {"kind": scorer_kind, "url": scorer_url,
                   "strong_url": strong_url if scorer_kind == "cascade" else None,
                   "model": FAST_MODEL if scorer_kind == "cascade" else WINNOW_MODEL,
                   "strong_model": WINNOW_MODEL if scorer_kind == "cascade" else None},
        "scorer_calls": scorer.calls,
        "scorer_latency_ms": {"min": min(scorer.latencies) if scorer.latencies else None,
                              "max": max(scorer.latencies) if scorer.latencies else None,
                              "mean": round(sum(scorer.latencies) / len(scorer.latencies), 1)
                              if scorer.latencies else None},
        "cases": cases,
        "proposed_pointers": proposals,
        "scored_cases": scored_cases,
        "failures": failures,
        "ok": not failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer-url", default=WINNOW_URL,
                        help="scorer URL; for --scorer-kind cascade this is the FAST path")
    parser.add_argument("--scorer-kind", choices=("systemone", "cascade"),
                        default="systemone")
    parser.add_argument("--strong-url", default=WINNOW_URL,
                        help="cascade only: strong scorer URL")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    receipt = run_winnow_shadow(args.scorer_url, args.scorer_kind,
                                args.strong_url)
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "sha256": sha256_text(encoded), "ok": receipt["ok"],
                      "status": receipt["status"],
                      "scorer_calls": receipt["scorer_calls"]}, indent=2))
    if not receipt["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
