#!/usr/bin/env python3
"""Validate the Phase1 shadow authorization template.

This artifact defines what owner authorization must supply before any future
shadow measurement. The template itself grants nothing and calls no provider.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUTH = ROOT / "research" / "phase1_shadow_authorization_v1.json"
SCHEMA = "nanojev-phase1-shadow-authorization-v1"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message, failures):
    if not condition:
        failures.append(message)


def validate_authorization(path=DEFAULT_AUTH, root=ROOT):
    path = Path(path)
    root = Path(root)
    failures = []
    auth = json.loads(path.read_text(encoding="utf-8"))
    require(auth.get("schema_version") == SCHEMA,
            "unsupported Phase1 authorization schema", failures)
    require(auth.get("status") == "template_ready_not_authorized",
            "Phase1 authorization status mismatch", failures)

    grant = auth.get("authorization") or {}
    require(grant.get("granted") is False,
            "Phase1 authorization must remain ungranted", failures)
    for field in ("granted_by", "granted_at", "scope", "expires_at", "upstream"):
        require(grant.get(field) is None,
                f"authorization.{field} must be null until owner authorization", failures)
    require(type(grant.get("max_requests")) is int and
            1 <= grant.get("max_requests") <= 25,
            "authorization.max_requests must be bounded <=25", failures)

    contract = auth.get("measurement_contract") or {}
    require(contract.get("mode") == "shadow", "Phase1 mode must be shadow", failures)
    for field in ("measurement_only", "actual_savings_requires_paired_provider_usage",
                  "local_token_estimates_are_not_billing"):
        require(contract.get(field) is True,
                f"measurement_contract.{field} must be true", failures)
    for field in ("reduced_bytes_may_be_sent", "provider_calls_allowed_in_template",
                  "active_filtering_allowed"):
        require(contract.get(field) is False,
                f"measurement_contract.{field} must be false", failures)

    metrics = auth.get("milestone_metrics") or {}
    require(metrics.get("phase0_cases_required") == 17,
            "Phase0 required case count mismatch", failures)
    require(metrics.get("phase0_status_required") == "phase0_pass",
            "Phase0 status requirement mismatch", failures)
    require(metrics.get("phase1_dry_run_cases_required") == 8,
            "Phase1 dry-run required case count mismatch", failures)
    require(metrics.get("phase1_dry_run_status_required") == "phase1_shadow_dry_run_pass",
            "Phase1 dry-run status requirement mismatch", failures)
    require(metrics.get("provider_calls_required") == 0,
            "provider calls must remain zero", failures)
    for field in ("unsafe_count_required", "paired_regressions_required"):
        require(metrics.get(field) == 0, f"{field} must be zero", failures)
    for field in ("content_free_receipts_required", "kill_switch_required",
                  "restore_round_trip_required", "unsupported_paths_must_bypass",
                  "scorer_failures_must_fail_open"):
        require(metrics.get(field) is True, f"{field} must be true", failures)

    pre = auth.get("preconditions") or {}
    require(pre.get("track_a_required_status") ==
            "conditional_pass_owner_authorized_internal_review",
            "Track A required status mismatch", failures)
    require(pre.get("phase0_required_case_count") == 17,
            "Phase0 precondition case count mismatch", failures)
    for field in ("track_a_decision", "canary_protocol", "canary_preflight",
                  "phase0_receipt", "phase1_dry_run_receipt",
                  "real_context_protocol", "collection_authorization"):
        artifact = pre.get(field)
        require(isinstance(artifact, str) and (root / artifact).exists(),
                f"missing precondition artifact: {field}", failures)

    required = set(auth.get("required_owner_fields_before_phase1") or [])
    for field in ("authorization.granted=true", "authorization.granted_by",
                  "authorization.granted_at", "authorization.scope",
                  "authorization.expires_at", "authorization.upstream"):
        require(field in required, f"missing owner field: {field}", failures)

    checklist = auth.get("operator_checklist") or []
    ids = {item.get("id") for item in checklist if isinstance(item, dict)}
    for item_id in ("scope_named", "measurement_only_confirmed",
                    "accounting_confirmed", "privacy_confirmed",
                    "stop_conditions_confirmed"):
        require(item_id in ids, f"missing checklist item: {item_id}", failures)
    for item in checklist:
        require(item.get("required") is True,
                f"checklist {item.get('id')} must be required", failures)
        require(item.get("completed") is False,
                f"checklist {item.get('id')} must be incomplete", failures)

    forbidden = set(auth.get("forbidden") or [])
    for phrase in ("sending reduced bytes", "active filtering",
                   "claiming provider billing savings without paired provider usage",
                   "logging raw prompt or response text",
                   "using scorer outputs as labels",
                   "using measurement data for training or calibration"):
        require(phrase in forbidden, f"missing forbidden action: {phrase}", failures)

    stops = set(auth.get("stop_conditions") or [])
    for phrase in ("any request would send reduced bytes",
                   "any protected or required evidence pointer is proposed for removal",
                   "any restore round-trip fails",
                   "receipt contains raw prompt, response, credential, or tool-output text",
                   "owner revokes scope or requests stop"):
        require(phrase in stops, f"missing stop condition: {phrase}", failures)

    return {
        "schema_version": "nanojev-phase1-shadow-authorization-check-v1",
        "authorization_path": str(path.resolve()),
        "authorization_sha256": sha256_file(path),
        "status": "template_ready_not_authorized" if not failures else "template_invalid",
        "phase1_authorized": False,
        "provider_calls_allowed": False,
        "active_filtering_allowed": False,
        "failures": failures,
        "ok": not failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = validate_authorization(args.authorization)
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
        print(json.dumps({"output": str(args.output),
                          "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                          "ok": receipt["ok"], "status": receipt["status"]}, indent=2))
    else:
        print(encoded, end="")
    if not receipt["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
