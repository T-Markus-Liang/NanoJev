#!/usr/bin/env python3
"""Active-mode loopback dry-run with the LIVE cascade scorer.

Phase0 proves the apply path with deterministic in-process scorers; this
runner exercises the same apply path (reduction + restore manifest +
round-trip verification) against the real Kev-4B → Winnow-12B cascade on
loopback. Whether the live scorer proposes a drop is model-dependent, so
each eligible case is validated either way: a proposed drop must satisfy the
full apply-path invariants; a retained request must forward original bytes.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from scorer_adapters_v1 import build_scorer
from run_active_mode_phase0_v1 import (
    drop_pointer, encoded_request, expect, expect_active_drop, run_case,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "gateway_live_active_v1.json"
FAST_URL = "http://127.0.0.1:8092"
STRONG_URL = "http://127.0.0.1:8091"
FAST_MODEL = "kev-latest"
STRONG_MODEL = "Winnow-12B"


def sha256_text(data):
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


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


EXPECTED_ROLES = {
    "openai_chat": ["system", "assistant", "user"],
    "anthropic_messages": ["assistant", "user"],
    "openai_responses": ["assistant", "user"],
}


def run_live_active(fast_url=FAST_URL, strong_url=STRONG_URL):
    failures, cases, outcomes = [], [], {}
    scorer = CountingScorer(build_scorer(
        "cascade", url=fast_url, strong_url=strong_url, timeout=30.0,
        model_id=FAST_MODEL, strong_model_id=STRONG_MODEL))

    for wire in ("openai_chat", "anthropic_messages", "openai_responses"):
        raw = encoded_request(wire)
        case, sent, manifest_header = run_case(
            f"live_active_{wire}", mode="active", wire=wire, scorer=scorer,
            sidecar={"segments": {drop_pointer(wire): {"eligible": True}}})
        cases.append(case)
        if case["applied_pointers"]:
            expect_active_drop(case, sent, raw, manifest_header, wire,
                               EXPECTED_ROLES[wire], failures)
            outcomes[wire] = "dropped"
        else:
            expect(sent == raw,
                   f"{wire}: retained request changed upstream bytes", failures)
            expect(manifest_header is None,
                   f"{wire}: retained request emitted restore manifest", failures)
            expect(case["forward_reason"] in ("active_no_reduction",
                                              "below_min_reduction"),
                   f"{wire}: unexpected forward reason "
                   f"{case['forward_reason']}", failures)
            outcomes[wire] = "retained"

    case, sent, manifest_header = run_case(
        "live_active_no_sidecar", mode="active", scorer=scorer, sidecar={})
    expect(sent == encoded_request("openai_chat"),
           "no-sidecar case modified bytes", failures)
    expect(case["gate_reason"] == "no_eligible_segments",
           "no-sidecar gate reason mismatch", failures)
    cases.append(case)

    calls_before = scorer.calls
    case, sent, manifest_header = run_case(
        "live_active_kill_switch", mode="active", kill_switch=True,
        scorer=scorer,
        sidecar={"segments": {drop_pointer("openai_chat"): {"eligible": True}}})
    expect(sent == encoded_request("openai_chat"),
           "kill switch modified bytes", failures)
    expect(case["forward_reason"] == "kill_switch",
           "kill switch reason mismatch", failures)
    expect(scorer.calls == calls_before,
           "kill switch still called the scorer", failures)
    cases.append(case)

    return {
        "schema_version": "nanojev-gateway-live-active-v1",
        "status": "live_active_pass" if not failures else "live_active_fail",
        "provider_calls": 0,
        "scorer": {"kind": "cascade", "fast_url": fast_url,
                   "strong_url": strong_url, "fast_model": FAST_MODEL,
                   "strong_model": STRONG_MODEL},
        "scorer_calls": scorer.calls,
        "scorer_latency_ms": {"min": min(scorer.latencies) if scorer.latencies else None,
                              "max": max(scorer.latencies) if scorer.latencies else None,
                              "mean": round(sum(scorer.latencies) / len(scorer.latencies), 1)
                              if scorer.latencies else None},
        "outcomes": outcomes,
        "cases": cases,
        "failures": failures,
        "ok": not failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast-url", default=FAST_URL)
    parser.add_argument("--strong-url", default=STRONG_URL)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    receipt = run_live_active(args.fast_url, args.strong_url)
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "sha256": sha256_text(encoded), "ok": receipt["ok"],
                      "status": receipt["status"],
                      "outcomes": receipt["outcomes"]}, indent=2))
    if not receipt["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
