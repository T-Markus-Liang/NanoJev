#!/usr/bin/env python3
"""Value-oriented shadow filtering benchmark V1.

This runner answers a different question from the A4 safety fixtures: when a trusted
sidecar marks a segment eligible, does the scorer propose removals that (a) keep every
required evidence string, (b) preserve dependency/tool structure, and (c) create real
hypothetical byte/token reduction?

The runner never sends reduced bytes to a provider. Model arms still run the ordinary
``shadow_request`` path, then apply the resulting removal plan only in-process through
``build_safe_reduction`` so every non-empty plan is proved byte-reversible.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_tool_history_fixtures_v1 import deterministic_scorer, encode_request
from context_gate_v1 import Bypass, parse_segments, shadow_request
from context_restore_v1 import canonical_bytes, build_restore_manifest, restore_request, verify_round_trip
from main_model_gateway_v1 import removal_plan, removed_segments_from_raw
from safe_dedup_v1 import build_safe_reduction, load_tokenizer, measure_request, plan_safe_dedup
from scorer_adapters_v1 import CascadeScorer, SystemOneHTTPScorer
from benchmark_context_shadow_systemone_v1 import DeciderContextScorer, ReflexContextScorer, SemIfContextScorer
from local_main_model_evaluator_v1 import (
    DeterministicEvidenceResponder, LocalGenerationResponder, evaluate_pair)


DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / "research" / "context_filter_value_fixture_manifest_v1.json"
DEFAULT_REFLEX_MODEL = Path(__file__).resolve().parent.parent / "external" / "models" / "Qwen3.5-4B"
DEFAULT_THRESHOLD_POLICY = Path(__file__).resolve().parent.parent / "research" / "context_filter_threshold_policy_v1.json"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_manifest(manifest):
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest must contain non-empty cases")
    seen = set()
    for index, case in enumerate(cases):
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or case_id in seen:
            raise ValueError(f"duplicate or missing case_id at index {index}")
        seen.add(case_id)
        wire_format = case.get("wire_format")
        raw = encode_request(case.get("body"))
        downstream = case.get("downstream") or {}
        if not isinstance(downstream.get("required_strings"), list):
            raise ValueError(f"{case_id}: downstream.required_strings must be a list")
        if not isinstance(downstream.get("expected_answer"), str):
            raise ValueError(f"{case_id}: downstream.expected_answer must be a string")
        expected_gate = case.get("expected_gate") or {}
        try:
            segments = parse_segments(raw, wire_format)
        except Bypass as bypass:
            reason = str(bypass)
            if expected_gate.get("status") != "bypass" or expected_gate.get("reason") != reason:
                raise ValueError(f"{case_id}: expected {expected_gate}, got bypass {reason}")
            continue
        if expected_gate.get("status") != "scored":
            raise ValueError(f"{case_id}: expected bypass but request parses")
        pointers = {segment.pointer for segment in segments}
        declared = {item.get("pointer") for item in case.get("segments", [])}
        if not declared or not declared <= pointers:
            raise ValueError(f"{case_id}: declared segment pointers are missing from parsed request")
        drops = [item for item in case.get("segments", [])
                 if item.get("expected_suggestion") == "drop"]
        if expected_gate.get("all_retain") is False and not drops:
            raise ValueError(f"{case_id}: expected a removable segment but none declared")
    return True


def load_threshold_policy(path):
    if path is None:
        return None
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    if policy.get("schema_version") != "nanojev-context-filter-threshold-policy-v1":
        raise ValueError("unsupported threshold policy schema")
    if not isinstance(policy.get("review_mode", {}).get("allowed_threshold_values"), list):
        raise ValueError("threshold policy lacks review_mode.allowed_threshold_values")
    return policy


def assert_review_thresholds(policy, threshold, cascade_threshold, arms):
    if policy is None:
        raise ValueError("review mode requires --threshold-policy")
    review = policy.get("review_mode") or {}
    allowed = {float(value) for value in review.get("allowed_threshold_values", [])}
    if float(threshold) not in allowed:
        raise ValueError(f"undeclared diagnostic threshold in review mode: {threshold}")
    if "cascade" in arms:
        allowed_cascade = {float(value) for value in
                           review.get("allowed_cascade_threshold_values", [])}
        if float(cascade_threshold) not in allowed_cascade:
            raise ValueError(
                f"undeclared cascade fast-path threshold in review mode: {cascade_threshold}")


def policy_receipt(path, policy):
    if policy is None:
        return None
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path),
            "schema_version": policy.get("schema_version"),
            "policy_id": policy.get("policy_id"),
            "status": policy.get("status")}


def count_tokens(counter, text):
    if counter is None:
        return None
    try:
        return int(counter(text))
    except Exception:
        return None


def text_of_request(raw):
    return raw.decode("utf-8")


def expected_maps(case):
    suggestions, statuses = {}, {}
    for item in case.get("segments", []):
        suggestions[item["pointer"]] = item.get("expected_suggestion")
        statuses[item["pointer"]] = item.get("expected_status")
    return suggestions, statuses


def compare_segments(receipt, case):
    expected_suggestion, expected_status = expected_maps(case)
    observed = {item["pointer"]: item for item in receipt.get("segments", [])}
    comparisons = []
    protected_drops, expected_match = 0, 0
    for pointer, expected in expected_suggestion.items():
        item = observed.get(pointer)
        actual = item.get("suggestion") if isinstance(item, dict) else None
        reason = item.get("reason") if isinstance(item, dict) else None
        eligible = reason not in {"protected_structure", "caller_protected", "required_dependency", "not_explicitly_eligible"}
        status = "eligible" if eligible else "protected"
        match = actual == expected
        expected_match += int(match)
        if actual == "drop" and status == "protected":
            protected_drops += 1
        comparisons.append({"pointer": pointer, "expected_status": expected_status.get(pointer),
                            "observed_status": status, "status_match": status == expected_status.get(pointer),
                            "expected_suggestion": expected, "observed_suggestion": actual,
                            "suggestion_match": match, "reason": reason,
                            "p_irrelevant": (item or {}).get("p_irrelevant")})
    return comparisons, expected_match, protected_drops


def apply_model_plan(raw, wire_format, receipt):
    pointers, error = removal_plan(receipt)
    if error is not None or not pointers:
        return raw, None, {}, {"attempted": False, "ok": None, "detail": error or "no_removal"}
    reduced, manifest, removed, verify = build_safe_reduction(raw, wire_format, tuple(pointers))
    detail = {"attempted": True,
              "ok": bool(verify["reconstructed_sha256"] == hashlib.sha256(canonical_bytes(json.loads(raw.decode("utf-8")))).hexdigest()
                         and verify["reduced_request_sha256_matches"]),
              "detail": "byte_identical_canonical_reconstruction",
              "manifest_sha256": hashlib.sha256(manifest.to_json().encode("utf-8")).hexdigest(),
              "dropped_segment_count": verify["dropped_segment_count"]}
    return reduced, manifest, removed, detail


class RecordingScorer:
    """Keep the scorer's content-free checkpoint and elapsed time."""

    def __init__(self, scorer):
        self.scorer = scorer
        self.last_checkpoint = None
        self.last_latency_ms = None

    def __call__(self, payload):
        started = time.perf_counter()
        try:
            result = self.scorer(payload)
        finally:
            self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        if isinstance(result, dict):
            self.last_checkpoint = result.get("checkpoint")
        return result


def percentile(values, p):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * p))))
    return values[index]


def evaluate_model_case(case, scorer, threshold):
    raw = encode_request(case["body"])
    wire_format = case["wire_format"]
    reduced, receipt = shadow_request(raw, wire_format, sidecar=case.get("sidecar"),
                                      scorer=scorer, threshold=threshold)
    if reduced != raw:
        raise AssertionError("shadow_request must return original bytes")
    reduced, manifest, removed, round_trip = apply_model_plan(raw, wire_format, receipt)
    return raw, reduced, receipt, round_trip


def evaluate_dedup_case(case):
    raw = encode_request(case["body"])
    wire_format = case["wire_format"]
    plan = plan_safe_dedup(raw, wire_format, case.get("sidecar"))
    if not plan.drop_pointers:
        return (raw, raw, {"schema_version": "nanojev-safe-dedup-shadow-v1", "status": plan.parse_status,
                           "reason": plan.bypass_reason or "no_removal",
                           "segments": [{"pointer": d.pointer, "role": d.role,
                                         "suggestion": "drop" if d.decision == "drop" else "retain",
                                         "reason": d.reason, "p_irrelevant": None}
                                        for d in plan.decisions]},
                {"attempted": False, "ok": None, "detail": "no_removal"})
    reduced, manifest, removed, verify = build_safe_reduction(raw, wire_format, tuple(plan.drop_pointers))
    return (raw, reduced, {"schema_version": "nanojev-safe-dedup-shadow-v1", "status": "scored",
                           "reason": "deterministic_exact_duplicate", "segments": [
                               {"pointer": d.pointer, "role": d.role,
                                "suggestion": "drop" if d.decision == "drop" else "retain",
                                "reason": d.reason, "p_irrelevant": None}
                               for d in plan.decisions]},
            {"attempted": True, "ok": bool(verify["reconstructed_sha256"] == hashlib.sha256(
                 canonical_bytes(json.loads(raw.decode("utf-8")))).hexdigest()
                 and verify["reduced_request_sha256_matches"]),
             "detail": "byte_identical_canonical_reconstruction",
             "manifest_sha256": hashlib.sha256(manifest.to_json().encode("utf-8")).hexdigest(),
             "dropped_segment_count": verify["dropped_segment_count"]})


def evaluate_control_case(case):
    raw = encode_request(case["body"])
    receipt = {"schema_version": "nanojev-filter-value-control-v1", "status": "control",
               "reason": "no_scoring", "segments": [
                   {"pointer": item["pointer"], "role": item["role"], "suggestion": "retain",
                    "reason": "control_retain", "p_irrelevant": None}
                   for item in case.get("segments", [])]}
    return raw, raw, receipt, {"attempted": False, "ok": None, "detail": "control_no_removal"}


def evaluate_arm(manifest, arm, scorer=None, threshold=0.99, tokenizer=None,
                 responder=None):
    counter, tokenizer_info = tokenizer if tokenizer else (None, {"status": "unavailable"})
    responder = responder or DeterministicEvidenceResponder(counter)
    scorer = RecordingScorer(scorer) if scorer is not None and not isinstance(scorer, RecordingScorer) else scorer
    cases = []
    scorer_latencies = []
    totals = {"request_bytes": 0, "reduced_bytes": 0, "removed_bytes": 0,
              "original_tokens": 0, "reduced_tokens": 0, "removed_tokens": 0,
              "token_cases": 0, "required_string_failures": 0,
              "unsafe_removals": 0, "expected_suggestion_matches": 0,
              "expected_suggestion_total": 0, "round_trip_ok": 0,
              "round_trip_attempted": 0, "fast_paths": 0, "strong_paths": 0,
              "fast_errors": 0, "original_answer_ok": 0,
              "reduced_answer_ok": 0, "paired_answer_regressions": 0}
    for case in manifest.get("cases", []):
        if arm == "control":
            raw, reduced, receipt, round_trip = evaluate_control_case(case)
        elif arm == "safe_dedup":
            raw, reduced, receipt, round_trip = evaluate_dedup_case(case)
        else:
            if isinstance(scorer, RecordingScorer):
                scorer.last_checkpoint = None
                scorer.last_latency_ms = None
            raw, reduced, receipt, round_trip = evaluate_model_case(case, scorer, threshold)
        comparisons, expected_match, protected_drops = compare_segments(receipt, case)
        downstream = case.get("downstream") or {}
        required = downstream.get("required_strings") or []
        reduced_text = text_of_request(reduced)
        missing_required = [hashlib.sha256(text.encode("utf-8")).hexdigest()
                            for text in required if text not in reduced_text]
        original_text = text_of_request(raw)
        original_tokens = count_tokens(counter, original_text)
        reduced_tokens = count_tokens(counter, reduced_text)
        provider_pair = evaluate_pair(raw, reduced, downstream, responder)
        original_provider = provider_pair["original"]
        reduced_provider = provider_pair["reduced"]
        paired_regression = provider_pair["answer_regression"]
        unsafe = bool(missing_required or protected_drops or paired_regression or
                      (round_trip.get("attempted") and not round_trip.get("ok")))
        case_result = {
            "case_id": case.get("case_id"), "kind": case.get("kind"),
            "request_bytes": len(raw), "reduced_bytes": len(reduced),
            "removed_bytes": len(raw) - len(reduced),
            "original_tokens": original_tokens, "reduced_tokens": reduced_tokens,
            "removed_tokens": (None if original_tokens is None or reduced_tokens is None
                               else original_tokens - reduced_tokens),
            "status": receipt.get("status"), "reason": receipt.get("reason"),
            "segments": comparisons, "expected_suggestion_matches": expected_match,
            "expected_suggestion_total": len(comparisons),
            "protected_drops": protected_drops,
            "missing_required_string_sha256": missing_required,
            "required_strings_total": len(required),
            "required_strings_present": len(required) - len(missing_required),
            "provider_pair": provider_pair,
            "round_trip": round_trip,
            "scorer_checkpoint": getattr(scorer, "last_checkpoint", None),
            "scorer_latency_ms": getattr(scorer, "last_latency_ms", None),
            "unsafe_removal": unsafe,
        }
        if case_result["scorer_latency_ms"] is not None:
            scorer_latencies.append(case_result["scorer_latency_ms"])
        cases.append(case_result)
        totals["request_bytes"] += len(raw)
        totals["reduced_bytes"] += len(reduced)
        totals["removed_bytes"] += len(raw) - len(reduced)
        totals["required_string_failures"] += len(missing_required)
        totals["unsafe_removals"] += int(unsafe)
        totals["original_answer_ok"] += int(original_provider["answer_ok"])
        totals["reduced_answer_ok"] += int(reduced_provider["answer_ok"])
        totals["paired_answer_regressions"] += int(paired_regression)
        totals["expected_suggestion_matches"] += expected_match
        totals["expected_suggestion_total"] += len(comparisons)
        totals["round_trip_attempted"] += int(bool(round_trip.get("attempted")))
        totals["round_trip_ok"] += int(bool(round_trip.get("ok")))
        checkpoint = case_result.get("scorer_checkpoint") or {}
        totals["fast_paths"] += len(checkpoint.get("fast_paths") or [])
        totals["strong_paths"] += len(checkpoint.get("strong_paths") or [])
        totals["fast_errors"] += int(checkpoint.get("fast_errors") or 0)
        if original_tokens is not None and reduced_tokens is not None:
            totals["token_cases"] += 1
            totals["original_tokens"] += original_tokens
            totals["reduced_tokens"] += reduced_tokens
            totals["removed_tokens"] += original_tokens - reduced_tokens
    totals["byte_reduction_percent"] = (None if totals["request_bytes"] == 0 else
        100.0 * totals["removed_bytes"] / totals["request_bytes"])
    totals["token_reduction_percent"] = (None if totals["original_tokens"] == 0 else
        100.0 * totals["removed_tokens"] / totals["original_tokens"])
    route_total = totals["fast_paths"] + totals["strong_paths"]
    totals["fast_route_share"] = None if route_total == 0 else totals["fast_paths"] / route_total
    totals["strong_route_share"] = None if route_total == 0 else totals["strong_paths"] / route_total
    totals["scorer_calls"] = len(scorer_latencies)
    totals["scorer_latency_ms_p50"] = percentile(scorer_latencies, 0.5)
    totals["scorer_latency_ms_p95"] = percentile(scorer_latencies, 0.95)
    return {"arm": arm, "threshold": threshold, "cases": cases, "totals": totals,
            "tokenizer": tokenizer_info}


def build_fast_scorer(name, args):
    if name == "reflex":
        return ReflexContextScorer(str(args.reflex_model), device=args.device,
                                   revision=args.reflex_revision)
    if name == "semif":
        return SemIfContextScorer(str(args.semif_model),
                                  revision=args.semif_revision,
                                  mlx_bits=args.semif_mlx_bits)
    if name == "decider":
        return DeciderContextScorer(str(args.decider_model), device=args.device,
                                    revision=args.decider_revision)
    if name == "systemone":
        return SystemOneHTTPScorer(args.fast_systemone_url, endpoint="/v1/systemone",
                                   model_id=args.fast_systemone_model)
    raise ValueError(f"unknown fast scorer {name!r}")


def build_arm_scorer(arm, args):
    if arm in {"control", "safe_dedup", "stub"}:
        return deterministic_scorer if arm == "stub" else None
    if arm == "systemone":
        return SystemOneHTTPScorer(args.systemone_url, endpoint="/v1/systemone",
                                   model_id=args.systemone_model)
    if arm == "reflex":
        return build_fast_scorer("reflex", args)
    if arm == "semif":
        return build_fast_scorer("semif", args)
    if arm == "decider":
        return build_fast_scorer("decider", args)
    if arm == "cascade":
        fast = build_fast_scorer(args.fast_scorer, args)
        strong = SystemOneHTTPScorer(args.systemone_url, endpoint="/v1/systemone",
                                     model_id=args.systemone_model)
        return CascadeScorer(fast, strong, fast_threshold=args.cascade_threshold)
    raise ValueError(f"unknown arm {arm!r}")


def build_downstream_responder(args, tokenizer):
    counter = tokenizer[0] if tokenizer else None
    if args.downstream_backend == "deterministic":
        return DeterministicEvidenceResponder(counter)
    if args.downstream_backend == "local-generation":
        return LocalGenerationResponder(
            args.downstream_model, args.downstream_revision,
            args.downstream_device, counter, args.downstream_max_new_tokens,
            model_path=args.downstream_model_path)
    raise ValueError(f"unknown downstream backend {args.downstream_backend!r}")


def run(args):
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    validate_manifest(manifest)
    threshold_policy = load_threshold_policy(args.threshold_policy)
    arms = ["control", "safe_dedup", "stub"] if args.self_test else args.arms
    if args.review_mode:
        assert_review_thresholds(threshold_policy, args.threshold,
                                 args.cascade_threshold, arms)
    tokenizer = load_tokenizer(args.tokenizer)
    responder = build_downstream_responder(args, tokenizer)
    results = []
    for arm in arms:
        scorer = build_arm_scorer(arm, args)
        threshold = args.stub_threshold if arm == "stub" else args.threshold
        results.append(evaluate_arm(manifest, arm, scorer=scorer,
                                    threshold=threshold, tokenizer=tokenizer,
                                    responder=responder))
    output = {
        "schema_version": "nanojev-filter-value-measurement-v1",
        "scope": "synthetic fixture evaluation; offline reduction simulation only; no provider call; no active filtering authorization",
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "threshold_policy": policy_receipt(args.threshold_policy, threshold_policy),
        "review_mode": bool(args.review_mode),
        "self_test": bool(args.self_test),
        "results": results,
    }
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2,
                                          allow_nan=False) + "\n", encoding="utf-8")
    return {"output": str(Path(args.output).resolve()), "sha256": sha256_file(args.output),
            "arms": [item["arm"] for item in results],
            "unsafe_removals": {item["arm"]: item["totals"]["unsafe_removals"] for item in results},
            "removed_bytes": {item["arm"]: item["totals"]["removed_bytes"] for item in results}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--threshold-policy", type=Path, default=None)
    parser.add_argument("--review-mode", action="store_true",
                        help="require a threshold policy and fail closed on undeclared thresholds")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=["control", "safe_dedup", "systemone", "reflex", "semif", "decider", "cascade", "stub"],
                        default=["control", "safe_dedup"])
    parser.add_argument("--fast-scorer", choices=["reflex", "semif", "decider", "systemone"], default="reflex")
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--stub-threshold", type=float, default=0.99)
    parser.add_argument("--systemone-url", default="http://127.0.0.1:8091")
    parser.add_argument("--systemone-model", default="Winnow-12B")
    parser.add_argument("--reflex-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--reflex-revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    parser.add_argument("--semif-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--semif-revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    parser.add_argument("--semif-mlx-bits", type=int, choices=[4, 8], default=4)
    parser.add_argument("--decider-model", default="Mapika/decider-0.8b")
    parser.add_argument("--decider-revision", default="1ea54127d3bd52f6d753d9257b32a6380b873907")
    parser.add_argument("--fast-systemone-url")
    parser.add_argument("--fast-systemone-model")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--cascade-threshold", type=float, default=0.95)
    parser.add_argument("--downstream-backend", choices=["deterministic", "local-generation"],
                        default="deterministic")
    parser.add_argument("--downstream-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--downstream-model-path", type=Path,
                        help="Optional local model directory; bypasses HF cache lookup")
    parser.add_argument("--downstream-revision", default="c1899de289a04d12100db370d81485cdf75e47ca")
    parser.add_argument("--downstream-device", default="mps")
    parser.add_argument("--downstream-max-new-tokens", type=int, default=32)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
