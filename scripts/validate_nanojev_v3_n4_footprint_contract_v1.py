#!/usr/bin/env python3
"""Read-only, fail-closed validator for the V3 N4 footprint contract.

The validator never loads a model, artifact, tokenizer or dataset and never accesses
the network. A valid report is a contract/preflight result only; it never authorizes
quantization, training, promotion or deployment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA_VERSION = "nanojev-v3-n4-footprint-contract-v1"
REPORT_VERSION = "nanojev-v3-n4-footprint-report-v1"
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
REQUIRED_SPLITS = ("train", "dev", "calibration", "test", "ood")
FORBIDDEN_EVAL_USAGE = frozenset({"fit", "select", "calibrate", "tune", "train", "promote"})
REQUIRED_BOUNDARIES = (
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
)
REQUIRED_METRICS = (
    "artifact_sha256", "tokenizer_sha256", "protocol_sha256", "parameter_count",
    "weight_bytes", "package_bytes", "peak_memory_bytes", "load_time_ms",
    "cold_latency_ms", "warm_p50_ms", "warm_p95_ms", "warm_p99_ms",
    "quality_metrics", "calibration_metrics", "ood_metrics", "protected_error_count",
    "probability_normalization_error", "deterministic", "network_model_calls",
    "model_loaded", "quantization_performed", "training_performed", "deployment_authorized",
)
REQUIRED_RECEIPT_FIELDS = REQUIRED_METRICS + (
    "candidate_id", "representation", "parent_candidate_id", "paired_with", "receipt_sha256",
)
AUTHORIZATION_SUFFIXES = ("_authorized", "_authorization")


def _add(violations, code, path, message):
    violations.append({"code": code, "path": path, "message": message})


def _is_string(value):
    return isinstance(value, str) and bool(value.strip())


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


def _require_list(obj, key, violations, path):
    if key not in obj:
        _add(violations, "missing_field", f"{path}.{key}", "required list is absent")
        return None
    if not isinstance(obj[key], list):
        _add(violations, "malformed_field", f"{path}.{key}", "must be a JSON list")
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


def _string_list(value):
    return isinstance(value, list) and bool(value) and all(_is_string(item) for item in value)


def _unique_strings(value):
    return _string_list(value) and len(value) == len(set(value))


def _scan_authorization(node, path, violations):
    if isinstance(node, dict):
        for key, value in node.items():
            lowered = key.lower()
            is_auth = (
                lowered.startswith("authorizes_")
                or lowered.endswith(AUTHORIZATION_SUFFIXES)
                or key in REQUIRED_BOUNDARIES
            )
            if is_auth and value is not False:
                _add(violations, "authorization_flag", f"{path}.{key}", "authorization-like value must be false")
            _scan_authorization(value, f"{path}.{key}", violations)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _scan_authorization(value, f"{path}[{index}]", violations)


def _validate_candidates(protocol, violations):
    candidates = _require_list(protocol, "candidate_ladder", violations, "$")
    present = []
    if candidates is None:
        return present
    seen = set()
    for index, candidate in enumerate(candidates):
        path = f"$.candidate_ladder[{index}]"
        if not isinstance(candidate, dict):
            _add(violations, "malformed_field", path, "candidate must be an object")
            continue
        candidate_id = _require_string(candidate, "candidate_id", violations, path)
        if candidate_id is None:
            continue
        if candidate_id in seen:
            _add(violations, "duplicate_candidate", f"{path}.candidate_id", f"duplicate candidate {candidate_id!r}")
        else:
            present.append(candidate_id)
        seen.add(candidate_id)
        if candidate_id not in REQUIRED_CANDIDATES:
            _add(violations, "unknown_candidate", f"{path}.candidate_id", f"unknown candidate {candidate_id!r}")
        for key in ("kind", "representation", "artifact_format", "paired_with", "description", "calibration_split"):
            _require_string(candidate, key, violations, path)
        if candidate.get("kind") != EXPECTED_KINDS.get(candidate_id):
            _add(violations, "candidate_kind", f"{path}.kind", "candidate kind does not match its fixed ladder role")
        parent = candidate.get("parent_candidate_id")
        if candidate_id == "fp32_baseline":
            if parent is not None:
                _add(violations, "baseline_parent", f"{path}.parent_candidate_id", "baseline parent must be null")
            if candidate.get("paired_with") != "fp32_baseline":
                _add(violations, "baseline_pairing", f"{path}.paired_with", "baseline must pair with itself")
        else:
            if not _is_string(parent):
                _add(violations, "missing_parent", f"{path}.parent_candidate_id", "non-baseline needs a parent")
            elif parent == candidate_id:
                _add(violations, "self_parent", f"{path}.parent_candidate_id", "candidate cannot parent itself")
            if candidate.get("paired_with") != "fp32_baseline":
                _add(violations, "unpaired_candidate", f"{path}.paired_with", "candidate must pair with fp32_baseline")
        for key in ("requires_training", "requires_quantization"):
            if type(candidate.get(key)) is not bool:
                _add(violations, "malformed_field", f"{path}.{key}", "must be boolean")
        fit_split = candidate.get("fit_split")
        if fit_split is not None and not _is_string(fit_split):
            _add(violations, "malformed_field", f"{path}.fit_split", "must be null or a non-empty string")
        if fit_split is not None and fit_split not in ("train", "dev"):
            _add(violations, "split_contract", f"{path}.fit_split", "fit split must be train, dev, or null")
        if candidate.get("calibration_split") != "calibration":
            _add(violations, "split_contract", f"{path}.calibration_split", "calibration split must be calibration")
        evaluation = candidate.get("evaluation_splits")
        if not _unique_strings(evaluation) or set(evaluation) != {"test", "ood"}:
            _add(violations, "malformed_field", f"{path}.evaluation_splits", "must be exactly ['test', 'ood'] without duplicates")
        if parent is not None and parent not in REQUIRED_CANDIDATES:
            _add(violations, "unknown_parent", f"{path}.parent_candidate_id", "parent must be a declared candidate")
    missing = [item for item in REQUIRED_CANDIDATES if item not in seen]
    if missing:
        _add(violations, "missing_candidate", "$.candidate_ladder", f"missing candidates: {', '.join(missing)}")
    if len(present) != len(REQUIRED_CANDIDATES):
        _add(violations, "candidate_count", "$.candidate_ladder", "candidate ladder must contain exactly five unique required candidates")
    return present


def _validate_pairing(protocol, violations):
    pairing = _require_object(protocol, "pairing", violations, "$")
    if pairing is None:
        return
    for key in ("same_workload_manifest", "same_tokenizer", "same_input_order", "same_n3_controls", "paired_with_baseline", "candidate_receipts_are_content_free"):
        _require_bool(pairing, key, violations, "$.pairing", True, "pairing_invariant")


def _validate_splits(protocol, violations):
    roles = _require_object(protocol, "split_roles", violations, "$")
    if roles is None:
        return
    extra = sorted(set(roles) - set(REQUIRED_SPLITS))
    if extra:
        _add(violations, "unknown_split", "$.split_roles", f"unexpected split roles: {', '.join(extra)}")
    for name in REQUIRED_SPLITS:
        path = f"$.split_roles.{name}"
        role = roles.get(name)
        if not isinstance(role, dict):
            _add(violations, "missing_split", path, "required split role must be an object")
            continue
        _require_bool(role, "frozen", violations, path, True, "unfrozen_split")
        usable = _require_list(role, "usable_for", violations, path)
        if usable is not None:
            if not _unique_strings(usable):
                _add(violations, "malformed_field", f"{path}.usable_for", "must be a non-empty unique string list")
            elif name in ("test", "ood"):
                forbidden = sorted(set(usable) & FORBIDDEN_EVAL_USAGE)
                if forbidden:
                    _add(violations, "unfrozen_split", f"{path}.usable_for", f"test/OOD cannot be used for {', '.join(forbidden)}")


def _validate_fields(protocol, violations):
    metrics = _require_list(protocol, "required_metrics", violations, "$")
    if metrics is not None:
        if not _unique_strings(metrics) or set(metrics) != set(REQUIRED_METRICS):
            _add(violations, "metric_contract", "$.required_metrics", "required metrics must exactly match the N4 metric set")
    fields = _require_list(protocol, "receipt_fields", violations, "$")
    if fields is not None:
        if not _unique_strings(fields) or set(fields) != set(REQUIRED_RECEIPT_FIELDS):
            _add(violations, "receipt_contract", "$.receipt_fields", "receipt fields must exactly match the N4 field set")


def _validate_quality_gates(protocol, violations):
    gates = _require_object(protocol, "quality_gates", violations, "$")
    if gates is None:
        return
    if type(gates.get("protected_error_count_max")) is not int or gates.get("protected_error_count_max") != 0:
        _add(violations, "quality_gate", "$.quality_gates.protected_error_count_max", "must be integer zero")
    tolerance = gates.get("probability_normalization_error_max")
    if not isinstance(tolerance, (int, float)) or not 0 <= tolerance <= 1e-9:
        _add(violations, "quality_gate", "$.quality_gates.probability_normalization_error_max", "must be in [0, 1e-9]")
    for key in ("ood_confident_answer_regression_allowed", "restore_fail_open_regression_allowed", "quality_delta_must_be_paired"):
        _require_bool(gates, key, violations, "$.quality_gates", False if key != "quality_delta_must_be_paired" else True, "quality_gate")


def _validate_boundaries(protocol, violations):
    boundaries = _require_object(protocol, "boundaries", violations, "$")
    if boundaries is None:
        return
    for key in REQUIRED_BOUNDARIES:
        _require_bool(boundaries, key, violations, "$.boundaries", False, "authorization_flag")
    _require_bool(boundaries, "shadow_only", violations, "$.boundaries", True, "shadow_boundary")


def validate_protocol(protocol, *, source="<memory>"):
    violations = []
    if not isinstance(protocol, dict):
        violations.append({"code": "malformed_root", "path": "$", "message": "protocol root must be an object"})
        return _report(protocol, source, violations, 0, [])
    if protocol.get("schema_version") != SCHEMA_VERSION:
        _add(violations, "schema_version", "$.schema_version", f"must equal {SCHEMA_VERSION}")
    for key in ("protocol_id", "status", "purpose", "mode", "decision_rule"):
        _require_string(protocol, key, violations, "$")
    limitations = _require_list(protocol, "limitations", violations, "$")
    if limitations is not None and not _unique_strings(limitations):
        _add(violations, "malformed_field", "$.limitations", "must be a non-empty unique string list")
    _require_bool(protocol, "frozen_before_measurement", violations, "$")
    _require_bool(protocol, "read_only", violations, "$")
    if type(protocol.get("network_model_calls")) is not int or protocol.get("network_model_calls") != 0:
        _add(violations, "network_boundary", "$.network_model_calls", "must be integer zero")
    for key in ("model_loaded", "artifact_generation_performed", "quantization_performed", "training_performed", "deployment_performed"):
        _require_bool(protocol, key, violations, "$", False, "execution_boundary")
    baseline = protocol.get("baseline_candidate_id")
    if baseline != "fp32_baseline":
        _add(violations, "baseline_candidate", "$.baseline_candidate_id", "must be fp32_baseline")
    candidates = _validate_candidates(protocol, violations)
    _validate_pairing(protocol, violations)
    _validate_splits(protocol, violations)
    _validate_fields(protocol, violations)
    _validate_quality_gates(protocol, violations)
    _validate_boundaries(protocol, violations)
    _scan_authorization(protocol, "$", violations)
    return _report(protocol, source, violations, len(candidates), candidates)


def _report(protocol, source, violations, candidate_count, candidates):
    blocked = bool(violations)
    unique_reasons = []
    for violation in violations:
        if violation["code"] not in unique_reasons:
            unique_reasons.append(violation["code"])
    return {
        "report_version": REPORT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "source": str(source),
        "valid": not blocked,
        "status": "footprint_contract_blocked" if blocked else "footprint_contract_valid_not_authorized",
        "violations": violations,
        "block_reasons": unique_reasons,
        "counts": {"candidates": candidate_count, "required_metrics": len(REQUIRED_METRICS), "receipt_fields": len(REQUIRED_RECEIPT_FIELDS)},
        "candidates_present": candidates,
        "training_authorized": False,
        "deployment_authorized": False,
        "production_pruning_authorized": False,
        "promotion_authorized": False,
        "artifact_generation_performed": False,
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
        return _report({}, path, [{"code": "protocol_unreadable", "path": str(path), "message": str(error)}], 0, [])
    return validate_protocol(protocol, source=path)


def _render(report):
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = validate_file(args.protocol)
    try:
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(_render(report))
    except OSError as error:
        report = _report({}, args.protocol, [{"code": "report_output", "path": str(args.output), "message": str(error)}], 0, [])
    sys.stdout.write(_render(report))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
