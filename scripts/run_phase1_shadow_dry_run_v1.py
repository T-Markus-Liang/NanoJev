#!/usr/bin/env python3
"""Phase1 shadow-measurement dry-run using loopback upstream only.

Runs the gateway in shadow mode across supported wire formats and denial paths.
No provider is contacted and no reduced bytes are sent.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path

from run_active_mode_phase0_v1 import (
    SIDECAR_HEADER, drop_pointer, encoded_request, run_case,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "phase1_shadow_dry_run_v1.json"


def sha256_text(data):
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def expect(condition, message, failures):
    if not condition:
        failures.append(message)


def expect_shadow(case, sent, raw, manifest_header, expected_pointer=None,
                  failures=None):
    failures = failures if failures is not None else []
    expect(case["upstream_status"] == 200,
           f"{case['case_id']}: did not reach upstream", failures)
    expect(sent == raw, f"{case['case_id']}: shadow mode changed bytes", failures)
    expect(manifest_header is None,
           f"{case['case_id']}: shadow mode emitted restore manifest", failures)
    expect(case["forward_reason"] == "shadow_mode",
           f"{case['case_id']}: forward reason mismatch", failures)
    expect(case["applied_pointers"] == [],
           f"{case['case_id']}: shadow mode applied pointers", failures)
    if expected_pointer is not None:
        expect(expected_pointer in (case.get("proposed_pointers") or []),
               f"{case['case_id']}: expected shadow proposal missing", failures)


def run_phase1_dry_run():
    failures = []
    cases = []

    openai_raw = encoded_request("openai_chat")
    openai_sidecar = {"segments": {drop_pointer("openai_chat"): {"eligible": True}}}
    case, sent, manifest_header = run_case(
        "shadow_openai_eligible_plan", mode="shadow", sidecar=openai_sidecar)
    expect_shadow(case, sent, openai_raw, manifest_header,
                  drop_pointer("openai_chat"), failures)
    cases.append(case)

    anthropic_raw = encoded_request("anthropic_messages")
    anthropic_sidecar = {"segments": {drop_pointer("anthropic_messages"): {"eligible": True}}}
    case, sent, manifest_header = run_case(
        "shadow_anthropic_eligible_plan", mode="shadow",
        wire="anthropic_messages", sidecar=anthropic_sidecar)
    expect_shadow(case, sent, anthropic_raw, manifest_header,
                  drop_pointer("anthropic_messages"), failures)
    expect(case["wire_format"] == "anthropic_messages",
           "anthropic shadow wire format mismatch", failures)
    cases.append(case)

    responses_raw = encoded_request("openai_responses")
    responses_sidecar = {"segments": {drop_pointer("openai_responses"): {"eligible": True}}}
    case, sent, manifest_header = run_case(
        "shadow_responses_eligible_plan", mode="shadow",
        wire="openai_responses", sidecar=responses_sidecar)
    expect_shadow(case, sent, responses_raw, manifest_header,
                  drop_pointer("openai_responses"), failures)
    expect(case["wire_format"] == "openai_responses",
           "responses shadow wire format mismatch", failures)
    cases.append(case)

    case, sent, manifest_header = run_case(
        "shadow_no_sidecar", mode="shadow")
    expect_shadow(case, sent, openai_raw, manifest_header, failures=failures)
    expect(case["gate_reason"] == "no_eligible_segments",
           "no-sidecar shadow gate reason mismatch", failures)
    cases.append(case)

    header_sidecar = base64.b64encode(json.dumps(openai_sidecar).encode()).decode("ascii")
    case, sent, manifest_header = run_case(
        "shadow_internal_headers", mode="shadow",
        headers={SIDECAR_HEADER: header_sidecar, "X-NanoJev-Debug": "1"})
    expect_shadow(case, sent, openai_raw, manifest_header,
                  drop_pointer("openai_chat"), failures)
    expect(case["upstream_internal_headers"] == [],
           "internal headers reached upstream in shadow mode", failures)
    cases.append(case)

    unsupported_raw = json.dumps({"model": "embedding-fake", "input": "embed this"}).encode()
    case, sent, manifest_header = run_case(
        "shadow_unsupported_embeddings", mode="shadow",
        path="/v1/embeddings", raw=unsupported_raw)
    expect(sent == unsupported_raw, "unsupported shadow path changed bytes", failures)
    expect(manifest_header is None, "unsupported path emitted manifest", failures)
    expect(case["forward_reason"] == "unsupported_wire_format",
           "unsupported shadow path reason mismatch", failures)
    cases.append(case)

    case, sent, manifest_header = run_case(
        "shadow_kill_switch", mode="shadow", kill_switch=True,
        sidecar=openai_sidecar)
    expect(sent == openai_raw, "kill switch changed bytes", failures)
    expect(manifest_header is None, "kill switch emitted manifest", failures)
    expect(case["forward_reason"] == "kill_switch",
           "kill switch reason mismatch", failures)
    cases.append(case)

    case, sent, manifest_header = run_case(
        "shadow_scorer_error", mode="shadow", scorer=lambda _payload: (_ for _ in ()).throw(RuntimeError("dry-run scorer error")),
        sidecar=openai_sidecar)
    expect(sent == openai_raw, "shadow scorer error changed bytes", failures)
    expect(manifest_header is None, "shadow scorer error emitted manifest", failures)
    expect(case["forward_reason"] == "shadow_mode",
           "shadow scorer error reason mismatch", failures)
    expect(case["scorer_failure_kind"] == "exception",
           "shadow scorer error kind mismatch", failures)
    cases.append(case)

    return {
        "schema_version": "nanojev-phase1-shadow-dry-run-v1",
        "status": "phase1_shadow_dry_run_pass" if not failures else "phase1_shadow_dry_run_fail",
        "provider_calls": 0,
        "active_filtering_applied": False,
        "upstream": "loopback_fake_only",
        "cases": cases,
        "failures": failures,
        "ok": not failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    receipt = run_phase1_dry_run()
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
