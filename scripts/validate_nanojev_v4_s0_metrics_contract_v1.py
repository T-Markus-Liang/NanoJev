#!/usr/bin/env python3
"""Read-only, fail-closed validator for the V4-S0 metrics contract.

The validator checks only the machine-readable pre-registration contract. It never
loads a model, artifact, tokenizer or dataset, never accesses the network, and
never authorizes measurement, training, quantization, promotion or deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


SCHEMA_VERSION = "nanojev-v4-s0-metrics-contract-v1"
REPORT_VERSION = "nanojev-v4-s0-metrics-report-v1"
TARGET_STATUS = "candidate_targets_pending_owner_reviewer"
REQUIRED_CANDIDATES = (
    "fp32_baseline",
    "fp16_cast",
    "int8_weight_only",
    "int4_weight_only",
    "distilled_student",
)
EXPECTED_KINDS = {
    "fp32_baseline": "baseline",
    "fp16_cast": "cast",
    "int8_weight_only": "weight_only_quantization",
    "int4_weight_only": "weight_only_quantization",
    "distilled_student": "distillation",
}
REQUIRED_SCOPES = {
    "model_compute": ("p50_ms", "p95_ms", "p99_ms"),
    "paper_decision_e2e": ("p50_ms", "p95_ms", "p99_ms"),
    "local_serving": ("cold_p95_ms", "cold_p99_ms", "warm_p50_ms", "warm_p95_ms", "warm_p99_ms"),
}
REQUIRED_TARGET_KEYS = {
    "v4_m1_research": (
        "package_bytes_max",
        "peak_memory_bytes_max",
        "warm_single_decision_p95_ms_max",
        "cold_start_p95_ms_max",
    ),
    "v4_m2_release": (
        "package_bytes_max",
        "peak_memory_bytes_max",
        "warm_single_decision_p95_ms_max",
        "cold_start_p95_ms_max",
    ),
}
REQUIRED_QUALITY_METRICS = (
    "accuracy",
    "nll",
    "brier",
    "ece",
    "coverage",
    "selective_risk",
)
REQUIRED_COST_FIELDS = (
    "wall_time_ms",
    "device",
    "energy_proxy",
    "load_time_ms",
    "retry_count",
    "tool_reexecution_count",
)
REQUIRED_RECEIPT_FIELDS = (
    "candidate_id",
    "paired_with",
    "workload_manifest_sha256",
    "hardware_profile_sha256",
    "runtime_manifest_sha256",
    "artifact_sha256",
    "tokenizer_sha256",
    "package_bytes",
    "peak_memory_bytes",
    "latency_by_scope",
    "quality_metrics",
    "calibration_metrics",
    "ood_metrics",
    "protected_error_count",
    "local_cost_proxy",
    "reliability_checks",
    "network_model_calls",
    "deterministic",
    "model_loaded",
    "quantization_performed",
    "training_performed",
    "deployment_performed",
    "measurement_authorized",
    "receipt_sha256",
)
AUTHORIZATION_KEYS = {
    "measurement_authorized",
    "training_authorized",
    "deployment_authorized",
    "production_pruning_authorized",
    "active_pruning_authorized",
    "promotion_authorized",
    "authorizes_execution",
    "authorizes_training",
    "authorizes_deployment",
    "model_weights_modified",
    "checkpoint_written",
    "network_allowed",
}


def _add(violations, code, path, message):
    violations.append({"code": code, "path": path, "message": message})


def _is_string(value):
    return isinstance(value, str) and bool(value.strip())


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require_string(obj, key, violations, path):
    if key not in obj:
        _add(violations, "missing_field", f"{path}.{key}", "required non-empty string is absent")
        return None
    if not _is_string(obj[key]):
        _add(violations, "malformed_field", f"{path}.{key}", "must be a non-empty string")
        return None
    return obj[key]


def _require_bool(obj, key, violations, path, expected=True, code="malformed_field"):
    if key not in obj:
        _add(violations, "missing_field", f"{path}.{key}", "required boolean is absent")
        return None
    if type(obj[key]) is not bool or obj[key] is not expected:
        _add(violations, code, f"{path}.{key}", f"must be exactly {str(expected).lower()}")
        return None
    return obj[key]


def _require_object(obj, key, violations, path):
    if key not in obj:
        _add(violations, "missing_field", f"{path}.{key}", "required object is absent")
        return None
    if not isinstance(obj[key], dict):
        _add(violations, "malformed_field", f"{path}.{key}", "must be a JSON object")
        return None
    return obj[key]


def _require_list(obj, key, violations, path):
    if key not in obj:
        _add(violations, "missing_field", f"{path}.{key}", "required list is absent")
        return None
    if not isinstance(obj[key], list):
        _add(violations, "malformed_field", f"{path}.{key}", "must be a JSON list")
        return None
    return obj[key]


def _require_number(obj, key, violations, path, *, minimum=None, maximum=None):
    if key not in obj:
        _add(violations, "missing_field", f"{path}.{key}", "required finite number is absent")
        return None
    value = obj[key]
    if not _is_number(value):
        _add(violations, "malformed_field", f"{path}.{key}", "must be a finite number")
        return None
    if minimum is not None and value < minimum:
        _add(violations, "threshold_contract", f"{path}.{key}", f"must be >= {minimum}")
    if maximum is not None and value > maximum:
        _add(violations, "threshold_contract", f"{path}.{key}", f"must be <= {maximum}")
    return value


def _unique_strings(value):
    return isinstance(value, list) and bool(value) and all(_is_string(item) for item in value) and len(value) == len(set(value))


def _scan_authorization(node, path, violations):
    if isinstance(node, dict):
        for key, value in node.items():
            lowered = key.lower()
            auth_like = key in AUTHORIZATION_KEYS or lowered.startswith("authorizes_") or lowered.endswith("_authorized")
            if auth_like and value is not False:
                _add(violations, "authorization_flag", f"{path}.{key}", "authorization-like value must be false in preflight")
            _scan_authorization(value, f"{path}.{key}", violations)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _scan_authorization(value, f"{path}[{index}]", violations)


def _validate_candidates(protocol, violations):
    candidates = _require_list(protocol, "candidate_ladder", violations, "$")
    if candidates is None:
        return []
    seen = set()
    present = []
    for index, candidate in enumerate(candidates):
        path = f"$.candidate_ladder[{index}]"
        if not isinstance(candidate, dict):
            _add(violations, "malformed_field", path, "candidate must be an object")
            continue
        candidate_id = _require_string(candidate, "candidate_id", violations, path)
        if candidate_id is None:
            continue
        if candidate_id in seen:
            _add(violations, "duplicate_candidate", f"{path}.candidate_id", "candidate id is duplicated")
        seen.add(candidate_id)
        present.append(candidate_id)
        if candidate_id not in REQUIRED_CANDIDATES:
            _add(violations, "unknown_candidate", f"{path}.candidate_id", "candidate is outside the fixed ladder")
        kind = _require_string(candidate, "kind", violations, path)
        _require_string(candidate, "representation", violations, path)
        parent = candidate.get("parent_candidate_id")
        paired = _require_string(candidate, "paired_with", violations, path)
        _require_bool(candidate, "requires_training", violations, path, False if candidate_id != "distilled_student" else True, "candidate_contract")
        _require_bool(candidate, "requires_quantization", violations, path, False if candidate_id in ("fp32_baseline", "distilled_student") else True, "candidate_contract")
        if kind != EXPECTED_KINDS.get(candidate_id):
            _add(violations, "candidate_contract", f"{path}.kind", "candidate kind does not match the fixed ladder")
        if candidate_id == "fp32_baseline":
            if parent is not None:
                _add(violations, "candidate_contract", f"{path}.parent_candidate_id", "baseline parent must be null")
            if paired != "fp32_baseline":
                _add(violations, "candidate_contract", f"{path}.paired_with", "baseline must pair with itself")
        else:
            if parent != "fp32_baseline":
                _add(violations, "candidate_contract", f"{path}.parent_candidate_id", "candidate parent must be fp32_baseline")
            if paired != "fp32_baseline":
                _add(violations, "candidate_contract", f"{path}.paired_with", "candidate must pair with fp32_baseline")
        _require_string(candidate, "description", violations, path)
    missing = [candidate for candidate in REQUIRED_CANDIDATES if candidate not in seen]
    if missing:
        _add(violations, "missing_candidate", "$.candidate_ladder", f"missing candidates: {', '.join(missing)}")
    if len(candidates) != len(REQUIRED_CANDIDATES) or len(seen) != len(REQUIRED_CANDIDATES):
        _add(violations, "candidate_count", "$.candidate_ladder", "candidate ladder must contain exactly five unique candidates")
    return present


def _validate_hardware_workload(protocol, violations):
    contract = _require_object(protocol, "hardware_workload_contract", violations, "$")
    if contract is None:
        return
    for key in ("owner_freeze_required", "workload_manifest_required", "same_device", "same_runtime", "same_tokenizer", "same_input_order", "same_frozen_splits"):
        _require_bool(contract, key, violations, "$.hardware_workload_contract", True, "workload_contract")
    scopes = _require_list(contract, "scopes", violations, "$.hardware_workload_contract")
    if scopes is None:
        return
    seen = set()
    for index, scope in enumerate(scopes):
        path = f"$.hardware_workload_contract.scopes[{index}]"
        if not isinstance(scope, dict):
            _add(violations, "malformed_field", path, "scope must be an object")
            continue
        scope_id = _require_string(scope, "scope_id", violations, path)
        _require_string(scope, "description", violations, path)
        fields = scope.get("required_latency_fields")
        if not _unique_strings(fields):
            _add(violations, "scope_contract", f"{path}.required_latency_fields", "must be a unique non-empty string list")
        if scope_id in seen:
            _add(violations, "scope_contract", f"{path}.scope_id", "scope id is duplicated")
        seen.add(scope_id)
        if scope_id not in REQUIRED_SCOPES:
            _add(violations, "scope_contract", f"{path}.scope_id", "scope is outside the fixed V4 scope set")
        elif tuple(fields or ()) != REQUIRED_SCOPES[scope_id]:
            _add(violations, "scope_contract", f"{path}.required_latency_fields", "latency fields do not match the fixed scope")
    if seen != set(REQUIRED_SCOPES):
        _add(violations, "scope_contract", "$.hardware_workload_contract.scopes", "must contain exactly the three fixed scopes")


def _validate_targets(protocol, violations):
    profiles = _require_object(protocol, "target_profiles", violations, "$")
    if profiles is None:
        return
    for profile, keys in REQUIRED_TARGET_KEYS.items():
        target = _require_object(profiles, profile, violations, "$.target_profiles")
        if target is None:
            continue
        for scope_key in ("warm_single_decision_scope_id", "cold_start_scope_id"):
            if target.get(scope_key) != "local_serving":
                _add(violations, "scope_binding", f"$.target_profiles.{profile}.{scope_key}", "V4 single-decision targets must bind to local_serving")
        for key in keys:
            value = _require_number(target, key, violations, f"$.target_profiles.{profile}", minimum=0)
            if value is not None and value == 0:
                _add(violations, "threshold_contract", f"$.target_profiles.{profile}.{key}", "target must be positive")
    _require_bool(profiles, "p99_required_but_not_yet_thresholded", violations, "$.target_profiles", True, "threshold_contract")
    m1 = profiles.get("v4_m1_research", {})
    m2 = profiles.get("v4_m2_release", {})
    for m1_key, m2_key in (("package_bytes_max", "package_bytes_max"), ("peak_memory_bytes_max", "peak_memory_bytes_max"), ("warm_single_decision_p95_ms_max", "warm_single_decision_p95_ms_max"), ("cold_start_p95_ms_max", "cold_start_p95_ms_max")):
        if _is_number(m1.get(m1_key)) and _is_number(m2.get(m2_key)) and m2[m2_key] >= m1[m1_key]:
            _add(violations, "threshold_contract", f"$.target_profiles.v4_m2_release.{m2_key}", "release target must be stricter than research target")


def _validate_quality(protocol, violations):
    quality = _require_object(protocol, "quality_contract", violations, "$")
    if quality is None:
        return
    if quality.get("paired_with") != "fp32_baseline":
        _add(violations, "quality_contract", "$.quality_contract.paired_with", "quality must pair with fp32_baseline")
    _require_number(quality, "ci_level", violations, "$.quality_contract", minimum=0.9, maximum=0.999)
    _require_number(quality, "accuracy_noninferiority_margin_pp", violations, "$.quality_contract", minimum=0, maximum=2.0)
    _require_bool(quality, "task_family_worst_case_noninferior", violations, "$.quality_contract", True, "quality_contract")
    metrics = quality.get("required_metrics")
    if not _unique_strings(metrics) or tuple(metrics) != REQUIRED_QUALITY_METRICS:
        _add(violations, "quality_contract", "$.quality_contract.required_metrics", "quality metrics must match the fixed ordered set")
    _require_number(quality, "protected_error_count_max", violations, "$.quality_contract", minimum=0, maximum=0)
    _require_number(quality, "probability_normalization_error_max", violations, "$.quality_contract", minimum=0, maximum=1e-9)
    for key in ("ood_abstain_regression_allowed", "permutation_instability_allowed", "test_ood_fit_allowed"):
        _require_bool(quality, key, violations, "$.quality_contract", False, "quality_contract")


def _validate_cost(protocol, violations):
    cost = _require_object(protocol, "local_cost_contract", violations, "$")
    if cost is None:
        return
    if cost.get("paired_with") != "fp32_baseline":
        _add(violations, "cost_contract", "$.local_cost_contract.paired_with", "cost must pair with fp32_baseline")
    _require_number(cost, "ci_level", violations, "$.local_cost_contract", minimum=0.9, maximum=0.999)
    fields = cost.get("required_fields")
    if not _unique_strings(fields) or tuple(fields) != REQUIRED_COST_FIELDS:
        _add(violations, "cost_contract", "$.local_cost_contract.required_fields", "cost fields must match the fixed ordered set")
    for key in ("absolute_currency_allowed", "cloud_cost_claim_allowed"):
        _require_bool(cost, key, violations, "$.local_cost_contract", False, "cost_contract")
    _require_bool(cost, "improvement_required_for_release", violations, "$.local_cost_contract", True, "cost_contract")


def _validate_reliability(protocol, violations):
    reliability = _require_object(protocol, "reliability_contract", violations, "$")
    if reliability is None:
        return
    for key in ("deterministic_required", "offline_install_required", "corrupted_artifact_reject_required", "rollback_required", "kill_switch_required"):
        _require_bool(reliability, key, violations, "$.reliability_contract", True, "reliability_contract")
    if reliability.get("network_model_calls_max") != 0:
        _add(violations, "reliability_contract", "$.reliability_contract.network_model_calls_max", "network model calls max must be zero")
    _require_bool(reliability, "active_pruning_default", violations, "$.reliability_contract", False, "reliability_contract")
    _require_bool(reliability, "scope_guard_disable_allowed", violations, "$.reliability_contract", False, "reliability_contract")


def _validate_receipt_fields(protocol, violations):
    fields = _require_list(protocol, "required_receipt_fields", violations, "$")
    if not _unique_strings(fields) or tuple(fields or ()) != REQUIRED_RECEIPT_FIELDS:
        _add(violations, "receipt_contract", "$.required_receipt_fields", "receipt fields must match the fixed ordered set")


def validate_protocol(protocol, *, source="<memory>"):
    violations = []
    if not isinstance(protocol, dict):
        violations.append({"code": "malformed_root", "path": "$", "message": "protocol root must be an object"})
        return _report(source, violations, [])
    if protocol.get("schema_version") != SCHEMA_VERSION:
        _add(violations, "schema_version", "$.schema_version", f"must equal {SCHEMA_VERSION}")
    for key in ("protocol_id", "status", "purpose", "mode", "target_status", "freeze_scope", "approval_mode", "protocol_hash_algorithm", "decision_rule"):
        _require_string(protocol, key, violations, "$")
    if protocol.get("status") != "frozen_research_contract":
        _add(violations, "status_contract", "$.status", "must be frozen_research_contract")
    if protocol.get("mode") != "read_only_preflight":
        _add(violations, "mode_contract", "$.mode", "must be read_only_preflight")
    if protocol.get("target_status") != TARGET_STATUS:
        _add(violations, "target_status", "$.target_status", f"must be {TARGET_STATUS}")
    if protocol.get("freeze_scope") != "schema_and_candidate_ladder_only":
        _add(violations, "freeze_scope", "$.freeze_scope", "must state that only schema and candidate ladder are frozen")
    if protocol.get("approval_mode") != "pre_approval_only":
        _add(violations, "approval_mode", "$.approval_mode", "must state pre_approval_only")
    if protocol.get("protocol_hash_algorithm") != "sha256_canonical_json":
        _add(violations, "hash_contract", "$.protocol_hash_algorithm", "must state sha256_canonical_json")
    for key in ("frozen_before_measurement", "read_only"):
        _require_bool(protocol, key, violations, "$")
    if protocol.get("network_model_calls") != 0:
        _add(violations, "network_boundary", "$.network_model_calls", "must be integer zero")
    for key in ("model_loaded", "artifact_generation_performed", "quantization_performed", "training_performed", "deployment_performed", "owner_approved", "independent_reviewer_approved"):
        _require_bool(protocol, key, violations, "$", False, "execution_boundary")
    if protocol.get("baseline_candidate_id") != "fp32_baseline":
        _add(violations, "baseline_candidate", "$.baseline_candidate_id", "must be fp32_baseline")
    _validate_candidates(protocol, violations)
    _validate_hardware_workload(protocol, violations)
    _validate_targets(protocol, violations)
    _validate_quality(protocol, violations)
    _validate_cost(protocol, violations)
    _validate_reliability(protocol, violations)
    _validate_receipt_fields(protocol, violations)
    _require_object(protocol, "boundaries", violations, "$")
    if isinstance(protocol.get("boundaries"), dict):
        for key in AUTHORIZATION_KEYS:
            _require_bool(protocol["boundaries"], key, violations, "$.boundaries", False, "authorization_flag")
        _require_bool(protocol["boundaries"], "shadow_only", violations, "$.boundaries", True, "authorization_flag")
    _scan_authorization(protocol, "$", violations)
    return _report(source, violations, protocol.get("candidate_ladder", []))


def _report(source, violations, candidates):
    reasons = []
    for violation in violations:
        if violation["code"] not in reasons:
            reasons.append(violation["code"])
    return {
        "report_version": REPORT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "source": str(source),
        "protocol_sha256": None,
        "valid": not violations,
        "status": "metrics_contract_blocked" if violations else "metrics_contract_valid_not_authorized",
        "violations": violations,
        "block_reasons": reasons,
        "counts": {"candidates": len(candidates) if isinstance(candidates, list) else 0, "scopes": len(REQUIRED_SCOPES), "required_receipt_fields": len(REQUIRED_RECEIPT_FIELDS)},
        "measurement_authorized": False,
        "training_authorized": False,
        "deployment_authorized": False,
        "quantization_performed": False,
        "training_performed": False,
        "deployment_performed": False,
        "model_loaded": False,
        "network_model_calls": 0,
    }


def load_protocol(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def validate_file(path):
    try:
        protocol = load_protocol(path)
    except (OSError, ValueError) as error:
        return _report(path, [{"code": "protocol_unreadable", "path": str(path), "message": str(error)}], [])
    report = validate_protocol(protocol, source=path)
    if isinstance(protocol, dict):
        canonical = json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        report["protocol_sha256"] = hashlib.sha256(canonical).hexdigest()
    return report


def _render(report):
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = validate_file(args.protocol)
    if args.output is not None:
        try:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(_render(report))
        except OSError as error:
            report = _report(args.protocol, [{"code": "report_output", "path": str(args.output), "message": str(error)}], [])
    sys.stdout.write(_render(report))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
