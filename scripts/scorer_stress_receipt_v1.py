#!/usr/bin/env python3
"""Deterministic scorer stress receipt for cascade routing.

Uses synthetic state IDs and fake scorer callables only. The receipt verifies that
fast-path errors, timeouts, and malformed outputs fall back to the strong scorer,
while strong-path failures propagate so the gate can fail open. No model is loaded
and no request text is stored.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scorer_adapters_v1 import CascadeScorer, ScorerError, ScorerTimeout

PAYLOAD = {"states": [
    {"id": "segment_0"},
    {"id": "segment_1"},
    {"id": "segment_2"},
]}


def answer_state(state_id, probability):
    return {"id": state_id,
            "answers": {"irrelevant": {"type": "boolean", "probabilities": {
                "false": 1.0 - probability, "true": probability}}}}


def valid_response(payload, probability):
    return {"checkpoint": {"adapter": "fake"},
            "states": [answer_state(state["id"], probability)
                       for state in payload["states"]]}


def run_cascade(name, fast, strong, expected):
    scorer = CascadeScorer(fast, strong, fast_threshold=0.95)
    failures = []
    try:
        result = scorer(PAYLOAD)
        checkpoint = result.get("checkpoint") or {}
        observed = {
            "fast_paths": checkpoint.get("fast_paths"),
            "strong_paths": checkpoint.get("strong_paths"),
            "fast_errors": checkpoint.get("fast_errors"),
        }
        for key, value in expected.items():
            if observed.get(key) != value:
                failures.append(f"{key}: expected={value!r} observed={observed.get(key)!r}")
        return {"scenario": name, "propagated": False, "observed": observed,
                "failures": failures}
    except Exception as error:
        propagated_type = type(error).__name__
        if expected.get("propagated") != propagated_type:
            failures.append(
                f"propagated: expected={expected.get('propagated')!r} observed={propagated_type!r}")
        return {"scenario": name, "propagated": True, "propagated_type": propagated_type,
                "failures": failures}


def build_receipt():
    scenarios = []

    def partial_fast(payload):
        state_id = payload["states"][0]["id"]
        if state_id == "segment_1":
            raise RuntimeError("synthetic fast failure")
        return valid_response(payload, 0.99 if state_id == "segment_0" else 0.5)

    scenarios.append(run_cascade(
        "partial_fast_error_falls_back",
        partial_fast,
        lambda payload: valid_response(payload, 0.99),
        {"fast_paths": ["segment_0"], "strong_paths": ["segment_1", "segment_2"],
         "fast_errors": 1},
    ))

    def malformed_fast(payload):
        state_id = payload["states"][0]["id"]
        if state_id == "segment_0":
            return {"checkpoint": {"adapter": "fake"}, "states": [{"id": state_id}]}
        return valid_response(payload, 0.5)

    scenarios.append(run_cascade(
        "malformed_fast_output_falls_back",
        malformed_fast,
        lambda payload: valid_response(payload, 0.99),
        {"fast_paths": [], "strong_paths": ["segment_0", "segment_1", "segment_2"],
         "fast_errors": 0},
    ))

    def timeout_fast(payload):
        raise ScorerTimeout("synthetic timeout")

    scenarios.append(run_cascade(
        "fast_timeout_falls_back",
        timeout_fast,
        lambda payload: valid_response(payload, 0.99),
        {"fast_paths": [], "strong_paths": ["segment_0", "segment_1", "segment_2"],
         "fast_errors": 3},
    ))

    scenarios.append(run_cascade(
        "strong_error_propagates_for_gate_fail_open",
        lambda payload: valid_response(payload, 0.5),
        lambda payload: (_ for _ in ()).throw(RuntimeError("synthetic strong failure")),
        {"propagated": "RuntimeError"},
    ))

    scenarios.append(run_cascade(
        "strong_malformed_propagates_for_gate_fail_open",
        lambda payload: valid_response(payload, 0.5),
        lambda payload: {"checkpoint": {"adapter": "fake"}, "states": []},
        {"propagated": "ScorerError"},
    ))
    failures = [failure for scenario in scenarios for failure in scenario["failures"]]
    return {"schema_version": "nanojev-scorer-stress-receipt-v1",
            "scope": "deterministic cascade stress only; no model load; no provider call",
            "scenarios": scenarios, "ok": not failures, "failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "scorer_stress_receipt_v1.json")
    args = parser.parse_args()
    receipt = build_receipt()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()),
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "ok": receipt["ok"], "failures": receipt["failures"]}, indent=2))
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
