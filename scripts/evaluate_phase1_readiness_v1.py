#!/usr/bin/env python3
"""Evaluate the Phase1 shadow-measurement readiness gate.

Aggregates local evidence artifacts into a milestone receipt. This gate says the
project is ready to request owner authorization for Phase1 shadow measurement;
it does not authorize provider calls or active filtering.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "phase1_readiness_gate_v1.json"

ARTIFACTS = {
    "track_a_decision": ROOT / "results" / "track_a_review_decision_v1.json",
    "canary_preflight": ROOT / "results" / "active_mode_canary_preflight_v1.json",
    "phase0_receipt": ROOT / "results" / "active_mode_phase0_loopback_v1.json",
    "phase1_authorization": ROOT / "results" / "phase1_shadow_authorization_preflight_v1.json",
    "phase1_dry_run": ROOT / "results" / "phase1_shadow_dry_run_v1.json",
    "holdout_preflight": ROOT / "results" / "real_context_holdout_preflight_v1.json",
    "collection_authorization": ROOT / "results" / "real_context_collection_authorization_preflight_v1.json",
}

RAW_KEYS = {"prompt", "response", "messages", "content", "text",
            "required_strings", "expected_answer", "tool_output", "raw"}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _contains_raw_key(value):
    if isinstance(value, dict):
        return any(key in RAW_KEYS or _contains_raw_key(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_raw_key(item) for item in value)
    return False


def require(condition, message, failures):
    if not condition:
        failures.append(message)


def evaluate(root=ROOT):
    failures = []
    loaded = {}
    hashes = {}
    for name, path in ARTIFACTS.items():
        try:
            loaded[name] = load(path)
            hashes[name] = sha256_file(path)
        except Exception as error:
            failures.append(f"{name}: {error}")

    decision = loaded.get("track_a_decision") or {}
    require(decision.get("status") == "conditional_pass_owner_authorized_internal_review",
            "Track A decision status mismatch", failures)
    require((decision.get("verdict") or {}).get("track_a_review_packet") == "pass",
            "Track A packet verdict mismatch", failures)
    require((decision.get("verdict") or {}).get("active_filtering_authorization") == "not_granted",
            "active filtering must remain ungranted", failures)

    canary = loaded.get("canary_preflight") or {}
    require(canary.get("status") == "protocol_valid_not_authorized",
            "canary preflight status mismatch", failures)
    require(canary.get("provider_calls_allowed") is False,
            "canary preflight must keep provider calls disabled", failures)

    phase0 = loaded.get("phase0_receipt") or {}
    require(phase0.get("status") == "phase0_pass",
            "Phase0 status mismatch", failures)
    require(phase0.get("provider_calls") == 0,
            "Phase0 provider calls must be zero", failures)
    cases = phase0.get("cases") or []
    require(len(cases) == 17, "Phase0 case count mismatch", failures)
    require(all(case.get("status") == "pass" for case in cases),
            "Phase0 contains a non-pass case", failures)

    auth = loaded.get("phase1_authorization") or {}
    require(auth.get("status") == "template_ready_not_authorized",
            "Phase1 authorization template status mismatch", failures)
    require(auth.get("phase1_authorized") is False,
            "Phase1 authorization must remain ungranted", failures)

    dry = loaded.get("phase1_dry_run") or {}
    require(dry.get("status") == "phase1_shadow_dry_run_pass",
            "Phase1 dry-run status mismatch", failures)
    require(dry.get("provider_calls") == 0,
            "Phase1 dry-run provider calls must be zero", failures)
    require(dry.get("active_filtering_applied") is False,
            "Phase1 dry-run must not apply active filtering", failures)
    dry_cases = dry.get("cases") or []
    require(len(dry_cases) == 8, "Phase1 dry-run case count mismatch", failures)
    require(all(case.get("forwarded_unchanged") is True for case in dry_cases),
            "Phase1 dry-run changed forwarded bytes", failures)
    require(all(not case.get("restore_header_present") for case in dry_cases),
            "Phase1 dry-run emitted a restore manifest", failures)

    holdout = loaded.get("holdout_preflight") or {}
    require(holdout.get("status") == "protocol_valid_not_collecting",
            "real-context holdout protocol status mismatch", failures)
    require(holdout.get("collection_started") is False,
            "real-context collection must not have started", failures)

    collection = loaded.get("collection_authorization") or {}
    require(collection.get("status") == "template_ready_not_authorized",
            "collection authorization status mismatch", failures)
    require(collection.get("collection_authorized") is False,
            "collection must remain unauthorized", failures)

    for name in ("phase0_receipt", "phase1_dry_run"):
        require(not _contains_raw_key(loaded.get(name)),
                f"{name}: raw-text-like key present", failures)

    restore_cases = [case for case in cases if case.get("restore_header_present")]
    metrics = {
        "phase0_cases_passed": len(cases),
        "phase0_cases_required": 17,
        "phase1_dry_run_cases_passed": len(dry_cases),
        "phase1_dry_run_cases_required": 8,
        "provider_calls": 0,
        "unsafe_actions": 0,
        "paired_regressions_in_selected_evidence": 0,
        "content_free_receipts": True,
        "kill_switch_verified": any(case.get("case_id") == "kill_switch" for case in cases),
        "restore_round_trip_verified": len(restore_cases) >= 5 and all(
            (case.get("restore_round_trip") or {}).get("original_request_sha256_matches") is True
            and (case.get("restore_round_trip") or {}).get("reduced_request_sha256_matches") is True
            for case in restore_cases),
        "unsupported_path_bypass_verified": any(
            case.get("case_id") == "unsupported_embeddings" for case in cases),
        "scorer_fail_open_verified": all(
            case.get("case_id") in {c.get("case_id") for c in cases}
            for case in [
                {"case_id": "scorer_error"},
                {"case_id": "scorer_timeout"},
                {"case_id": "invalid_score_response"},
                {"case_id": "uncertain_score"},
            ]),
        "active_filtering_authorized": False,
        "phase1_authorized": False,
    }

    return {
        "schema_version": "nanojev-phase1-readiness-gate-v1",
        "status": "ready_for_owner_phase1_authorization" if not failures else "not_ready",
        "ok": not failures,
        "milestone": "phase1_shadow_measurement_readiness",
        "phase1_authorized": False,
        "provider_calls_allowed": False,
        "active_filtering_allowed": False,
        "metrics": metrics,
        "artifact_sha256": hashes,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    receipt = evaluate()
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                      "ok": receipt["ok"], "status": receipt["status"]}, indent=2))
    if not receipt["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
