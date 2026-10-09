#!/usr/bin/env python3
"""Validate the real-context holdout intake protocol without collecting data."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTOCOL = ROOT / "research" / "real_context_holdout_protocol_v1.json"
SCHEMA = "nanojev-real-context-holdout-protocol-v1"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message, failures):
    if not condition:
        failures.append(message)


def validate_protocol(path=DEFAULT_PROTOCOL, root=ROOT):
    path = Path(path)
    root = Path(root)
    failures = []
    protocol = json.loads(path.read_text(encoding="utf-8"))
    require(protocol.get("schema_version") == SCHEMA,
            "unsupported real-context holdout schema", failures)
    require(protocol.get("status") == "protocol_valid_not_collecting",
            "protocol status must remain protocol_valid_not_collecting", failures)

    collection = protocol.get("collection") or {}
    require(collection.get("started") is False,
            "collection must not have started", failures)
    require(collection.get("requires_owner_authorization") is True,
            "collection must require owner authorization", failures)
    require(collection.get("owner_authorization_granted") is False,
            "protocol cannot self-grant collection authorization", failures)
    require(collection.get("raw_files_expected") == 0,
            "protocol v1 expects zero raw files", failures)
    require(type(collection.get("minimum_cases_when_collected")) is int and
            collection.get("minimum_cases_when_collected") >= 30,
            "minimum collected cases must be >=30", failures)
    require(type(collection.get("target_cases_when_collected")) is int and
            collection.get("target_cases_when_collected") >=
            collection.get("minimum_cases_when_collected"),
            "target cases must be >= minimum", failures)
    data_root = root / collection.get("data_root", "")
    require(str(collection.get("data_root", "")).startswith("data/"),
            "raw holdout root must be under ignored data/", failures)
    if data_root.exists():
        require(not any(p.is_file() for p in data_root.rglob("*")),
                "collection.started=false but raw holdout files exist", failures)
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    require("data/*" in gitignore.splitlines(),
            ".gitignore must keep data/* local-only", failures)

    allowed = set(protocol.get("allowed_sources") or [])
    forbidden = set(protocol.get("forbidden_sources") or [])
    for phrase in ("owner_selected_local_requests",
                   "owner_selected_sanitized_tool_traces"):
        require(phrase in allowed, f"missing allowed source: {phrase}", failures)
    for phrase in ("jevbench_or_other_benchmark_rows",
                   "provider_generated_outputs_as_labels",
                   "real_credentials_or_tokens",
                   "requests_containing_unredactable_sensitive_content"):
        require(phrase in forbidden, f"missing forbidden source: {phrase}", failures)

    contract = protocol.get("case_contract") or {}
    required = set(contract.get("required_fields") or [])
    for field in ("case_id", "wire_format", "source_type", "source_hash",
                  "request_sha256", "expected_gate_status", "protected_pointers",
                  "downstream"):
        require(field in required, f"missing required case field: {field}", failures)
    require(set(contract.get("wire_formats") or []) ==
            {"openai_chat", "anthropic_messages", "openai_responses"},
            "wire-format contract mismatch", failures)
    require(set(contract.get("expected_gate_statuses") or []) ==
            {"scored", "bypass", "protected_only"},
            "gate-status contract mismatch", failures)

    labeling = protocol.get("labeling") or {}
    for field in ("labels_are_manual", "scorer_outputs_must_not_be_labels",
                  "protected_pointers_manual_required",
                  "required_evidence_must_be_declared",
                  "downstream_contract_required_for_scored_cases"):
        require(labeling.get(field) is True,
                f"labeling.{field} must be true", failures)
    require(labeling.get("conflict_resolution") == "retain uncertain context",
            "uncertain labels must retain context", failures)

    coverage = protocol.get("coverage_requirements") or {}
    expected_minima = {
        "minimum_cases": 30,
        "minimum_distinct_families": 8,
        "minimum_bypass_cases": 3,
        "minimum_tool_linked_cases": 3,
        "minimum_multilingual_cases": 2,
        "minimum_protected_only_cases": 2,
        "minimum_long_tail_cases": 2,
        "minimum_mixed_part_cases": 1,
        "minimum_mutable_tool_result_cases": 1,
        "minimum_nonrepeatable_tool_result_cases": 1,
    }
    for field, minimum in expected_minima.items():
        require(type(coverage.get(field)) is int and coverage.get(field) >= minimum,
                f"coverage.{field} must be >= {minimum}", failures)

    isolation = protocol.get("isolation") or {}
    for field in ("exclude_v2_v3_cases", "evaluation_only",
                  "training_calibration_forbidden"):
        require(isolation.get(field) is True,
                f"isolation.{field} must be true", failures)
    require("request_sha256" in str(isolation.get("duplicate_detection")),
            "duplicate detection must use request_sha256", failures)
    require("manual review" in str(isolation.get("near_duplicate_policy")),
            "near duplicates must require manual review", failures)

    privacy = protocol.get("privacy") or {}
    for field in ("raw_files_local_only", "raw_root_must_be_gitignored",
                  "public_manifest_content_free", "credential_scan_required",
                  "unredactable_cases_must_be_dropped",
                  "receipts_must_not_store_raw_prompt_or_response"):
        require(privacy.get(field) is True,
                f"privacy.{field} must be true", failures)
    manifest_fields = set(privacy.get("manifest_may_store_only") or [])
    require("hashes" in manifest_fields and "counts" in manifest_fields,
            "manifest must be limited to hashes/counts/status-like fields", failures)

    order = protocol.get("evaluation_order") or []
    require(order == [
        "manifest_schema_and_hash_validation",
        "privacy_and_isolation_checks",
        "deterministic_shadow_evaluation",
        "pinned_local_generation_pair_evaluation",
        "optional_phase1_shadow_provider_measurement_after_owner_authorization",
        "optional_phase2_paired_active_canary_after_owner_authorization",
    ], "evaluation order mismatch", failures)
    require(protocol.get("provider_calls_allowed") is False,
            "protocol must not allow provider calls", failures)
    require(protocol.get("active_filtering_allowed") is False,
            "protocol must not allow active filtering", failures)

    stops = set(protocol.get("stop_conditions") or [])
    for phrase in (
        "raw context appears outside the ignored data root",
        "credential scan fails",
        "request_sha256 overlaps V2/V3 or another holdout case",
        "provider call or active filtering is attempted before explicit owner authorization",
    ):
        require(phrase in stops, f"missing stop condition: {phrase}", failures)

    artifacts = protocol.get("review_artifacts") or {}
    require(str(artifacts.get("future_manifest", "")).startswith(
        str(collection.get("data_root"))),
            "future manifest must live under the raw data root", failures)
    require(str(artifacts.get("future_preflight_receipt", "")).startswith("results/"),
            "future preflight receipt must be under results/", failures)
    require(str(artifacts.get("future_value_receipt", "")).startswith("results/"),
            "future value receipt must be under results/", failures)

    acceptance = protocol.get("acceptance") or {}
    require(acceptance.get("protocol_status_required") == "protocol_valid_not_collecting",
            "acceptance protocol status mismatch", failures)
    for field in ("collection_preflight_required_before_raw_files",
                  "provider_phase_requires_new_owner_scope",
                  "active_canary_requires_new_owner_scope"):
        require(acceptance.get(field) is True,
                f"acceptance.{field} must be true", failures)

    return {
        "schema_version": "nanojev-real-context-holdout-preflight-v1",
        "protocol_path": str(path.resolve()),
        "protocol_sha256": sha256_file(path),
        "status": "protocol_valid_not_collecting" if not failures else "protocol_invalid",
        "collection_started": False,
        "provider_calls_allowed": False,
        "active_filtering_allowed": False,
        "failures": failures,
        "ok": not failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = validate_protocol(args.protocol)
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
