#!/usr/bin/env python3
"""Run deterministic shadow evaluation over a real-context holdout manifest.

Reads local raw request files referenced by a content-free manifest and emits a
content-free receipt. No provider calls, no active filtering, no raw text output.
"""

import argparse
import hashlib
import json
from pathlib import Path

from context_gate_v1 import shadow_request
from context_restore_v1 import (
    build_restore_manifest, canonical_bytes, restore_request, verify_round_trip,
)
from main_model_gateway_v1 import build_reduced_request, removed_segments_from_raw
from real_context_holdout_manifest_v1 import validate_manifest
from scorer_adapters_v1 import InProcessScorer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTOCOL = ROOT / "research" / "real_context_holdout_protocol_v1.json"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def deterministic_scorer(payload):
    return {"checkpoint": {"model": "real-context-holdout-deterministic-scorer"},
            "states": [{"id": state["id"],
                        "answers": {name: {"type": "boolean",
                                           "probabilities": {"false": 0.001,
                                                             "true": 0.999}}
                                    for name in (state.get("questions") or {})}}
                       for state in payload["states"]]}


def _sidecar(case):
    if case["expected_gate_status"] == "bypass":
        return {"bypass": True}
    segments = {}
    for pointer in case.get("protected_pointers") or []:
        segments[pointer] = {"pinned": True}
    for pointer in case.get("dependency_pointers") or []:
        segments.setdefault(pointer, {})["pinned"] = True
    for pointer in case.get("eligible_candidate_pointers") or []:
        segments.setdefault(pointer, {})["eligible"] = True
    return {"segments": segments}


def _case_status(case, receipt):
    expected = case["expected_gate_status"]
    if expected == "protected_only":
        return receipt.get("status") == "bypass" and receipt.get("reason") == "no_eligible_segments"
    return receipt.get("status") == expected


def evaluate_case(case, raw, scorer, threshold):
    wire_format = case["wire_format"]
    _, receipt = shadow_request(raw, wire_format, _sidecar(case), scorer,
                                threshold=threshold)
    segments = receipt.get("segments") or []
    suggested = [segment["pointer"] for segment in segments
                 if segment.get("suggestion") == "drop"]
    protected = set(case.get("protected_pointers") or [])
    dependencies = set(case.get("dependency_pointers") or [])
    required = set((case.get("downstream") or {}).get("required_evidence_pointers") or [])
    eligible = set(case.get("eligible_candidate_pointers") or [])
    suggested_set = set(suggested)
    protected_drops = sorted(suggested_set & (protected | dependencies | required))
    unexpected_drops = sorted(suggested_set - eligible)
    expected = case.get("expected_suggestions") or []
    expected_mismatch = bool(expected) and set(expected) != suggested_set

    reduced_sha256 = None
    restore = None
    restore_error = None
    if suggested:
        try:
            reduced = build_reduced_request(raw, wire_format, suggested)
            manifest = build_restore_manifest(raw, wire_format, suggested)
            recovered = removed_segments_from_raw(raw, wire_format, suggested)
            if restore_request(reduced, manifest, recovered) != canonical_bytes(json.loads(raw)):
                raise ValueError("restore_round_trip_mismatch")
            result = verify_round_trip(raw, reduced, manifest, recovered)
            if not result["original_request_sha256_matches"] or not result["reduced_request_sha256_matches"]:
                raise ValueError("restore_manifest_hash_mismatch")
            reduced_sha256 = sha256_bytes(reduced)
            restore = {"attempted": True, "ok": True}
        except Exception:
            restore = {"attempted": True, "ok": False}
            restore_error = "restore_round_trip_failed"
    else:
        restore = {"attempted": False, "ok": None}

    unsafe = bool(protected_drops or unexpected_drops or restore_error or expected_mismatch)
    return {
        "case_id": case["case_id"],
        "request_sha256": case["request_sha256"],
        "body_bytes": case["body_bytes"],
        "wire_format": wire_format,
        "expected_gate_status": case["expected_gate_status"],
        "gate_status": receipt.get("status"),
        "gate_reason": receipt.get("reason"),
        "gate_status_matches_expected": _case_status(case, receipt),
        "suggested_pointers": suggested,
        "suggested_pointer_count": len(suggested),
        "protected_drop_pointers": protected_drops,
        "unexpected_drop_pointers": unexpected_drops,
        "expected_suggestion_mismatch": expected_mismatch,
        "reduced_sha256": reduced_sha256,
        "restore": restore,
        "restore_error": restore_error,
        "unsafe": unsafe,
    }


def evaluate_manifest(manifest_path, case_dir=None, protocol_path=DEFAULT_PROTOCOL,
                      threshold=0.90):
    manifest_path = Path(manifest_path)
    case_dir = Path(case_dir) if case_dir else manifest_path.parent / "cases"
    manifest_receipt = validate_manifest(manifest_path, protocol_path)
    failures = []
    cases_out = []
    if not manifest_receipt["ok"]:
        failures.extend(f"manifest: {failure}" for failure in manifest_receipt["failures"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cases = manifest.get("cases") or []
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cases = manifest.get("cases") or []
    scorer = InProcessScorer(deterministic_scorer)
    for case in cases:
        request_rel = case.get("request_file")
        try:
            raw_path = (case_dir / request_rel).resolve()
            if case_dir.resolve() not in raw_path.parents:
                raise ValueError("request_file escapes case_dir")
            raw = raw_path.read_bytes()
            if sha256_bytes(raw) != case.get("request_sha256"):
                raise ValueError("request_sha256 mismatch")
            result = evaluate_case(case, raw, scorer, threshold)
            cases_out.append(result)
            if result["unsafe"]:
                failures.append(f"{case['case_id']}: unsafe")
        except Exception as error:
            failures.append(f"{case.get('case_id')}: {error}")
            cases_out.append({"case_id": case.get("case_id"),
                              "request_sha256": case.get("request_sha256"),
                              "unsafe": True,
                              "error": "case_evaluation_failed"})
    coverage = (manifest.get("coverage") or {})
    return {
        "schema_version": "nanojev-real-context-holdout-shadow-v1",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
        "manifest_validation_ok": manifest_receipt["ok"],
        "case_dir": str(case_dir.resolve()),
        "threshold": threshold,
        "provider_calls": 0,
        "active_filtering_applied": False,
        "case_count": len(cases_out),
        "unsafe_count": sum(1 for case in cases_out if case.get("unsafe")),
        "gate_status_mismatch_count": sum(
            1 for case in cases_out if case.get("gate_status_matches_expected") is False),
        "ready_for_evaluation": coverage.get("ready_for_evaluation"),
        "cases": cases_out,
        "failures": failures,
        "status": "shadow_eval_pass" if not failures else "shadow_eval_fail",
        "ok": not failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = evaluate_manifest(args.manifest, args.case_dir, args.protocol,
                                args.threshold)
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
        print(json.dumps({"output": str(args.output),
                          "sha256": sha256_bytes(encoded.encode()),
                          "ok": receipt["ok"], "status": receipt["status"],
                          "cases": receipt["case_count"]}, indent=2))
    else:
        print(encoded, end="")
    if not receipt["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
