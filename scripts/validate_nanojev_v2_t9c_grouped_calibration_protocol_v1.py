#!/usr/bin/env python3
"""Read-only, fail-closed validator for the V2 T9c grouped-calibration protocol."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA_VERSION = "nanojev-v2-t9c-grouped-calibration-protocol-v1"
REPORT_VERSION = "nanojev-v2-t9c-grouped-calibration-report-v1"
REQUIRED_SPLITS = ("train", "dev", "calibration", "test", "ood")
EXPECTED_USABLE_FOR = {
    "train": ["model_fit_only"],
    "dev": ["protocol_diagnostics"],
    "calibration": ["fit_grouped_temperature"],
    "test": ["evaluate"],
    "ood": ["evaluate", "abstain"],
}
FORBIDDEN_EVAL_USAGE = frozenset({"fit", "select", "tune", "calibrate", "model_fit_only"})
REQUIRED_CONTROLS = (
    "calibration_hash_and_split_guard",
    "heldout_no_fit",
    "group_recomputed",
    "missing_group_baseline_fallback",
    "monotone_ordering_disclosure",
    "protected_and_ood_reported",
)
AUTHORIZATION_KEYS = frozenset(
    {
        "training_authorized",
        "deployment_authorized",
        "production_pruning_authorized",
        "active_pruning_authorized",
        "promotion_authorized",
        "authorizes_execution",
        "authorizes_training",
        "authorizes_deployment",
    }
)


def _add(violations, code, path, message):
    violations.append({"code": code, "path": path, "message": message})


def _string(value):
    return isinstance(value, str) and bool(value.strip())


def _required_string(obj, key, violations, path):
    if key not in obj:
        _add(violations, "missing_field", f"{path}.{key}", "required non-empty string is absent")
        return None
    value = obj[key]
    if not _string(value):
        _add(violations, "malformed_field", f"{path}.{key}", "must be a non-empty string")
        return None
    return value


def _required_bool(obj, key, violations, path, expected=True):
    if key not in obj:
        _add(violations, "missing_field", f"{path}.{key}", "required boolean is absent")
        return None
    value = obj[key]
    if type(value) is not bool or value is not expected:
        _add(violations, "malformed_field", f"{path}.{key}", f"must be exactly {str(expected).lower()}")
        return None
    return value


def _required_object(obj, key, violations, path):
    if key not in obj:
        _add(violations, "missing_field", f"{path}.{key}", "required object is absent")
        return None
    value = obj[key]
    if not isinstance(value, dict):
        _add(violations, "malformed_field", f"{path}.{key}", "must be an object")
        return None
    return value


def _required_list(obj, key, violations, path):
    if key not in obj:
        _add(violations, "missing_field", f"{path}.{key}", "required list is absent")
        return None
    value = obj[key]
    if not isinstance(value, list):
        _add(violations, "malformed_field", f"{path}.{key}", "must be a list")
        return None
    return value


def _scan_auth(node, path, violations):
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}"
            lowered = key.lower()
            if key in AUTHORIZATION_KEYS or lowered.endswith("_authorized") or lowered.startswith("authorizes_"):
                if type(value) is not bool or value is not False:
                    _add(violations, "authorization_flag", child, "authorization-like values must be exactly false")
            _scan_auth(value, child, violations)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _scan_auth(value, f"{path}[{index}]", violations)


def _validate_splits(protocol, violations):
    roles = _required_object(protocol, "split_roles", violations, "$")
    if roles is None:
        return
    extra = sorted(set(roles) - set(REQUIRED_SPLITS))
    if extra:
        _add(violations, "unknown_split", "$.split_roles", f"unexpected split roles: {extra}")
    for split in REQUIRED_SPLITS:
        path = f"$.split_roles.{split}"
        role = roles.get(split)
        if not isinstance(role, dict):
            _add(violations, "missing_split", path, "split role must be an object")
            continue
        _required_bool(role, "frozen", violations, path, True)
        usable = role.get("usable_for")
        if not isinstance(usable, list) or not usable or not all(_string(item) for item in usable):
            _add(violations, "malformed_field", f"{path}.usable_for", "must be a non-empty list of strings")
            continue
        if usable != EXPECTED_USABLE_FOR[split]:
            code = "wrong_fit_split" if "fit_grouped_temperature" in usable else "wrong_split_usage"
            _add(violations, code, f"{path}.usable_for", f"must be exactly {EXPECTED_USABLE_FOR[split]}")
        if split in {"test", "ood"}:
            bad = sorted(
                item for item in usable
                if item in FORBIDDEN_EVAL_USAGE
                or item == "fit_grouped_temperature"
                or item.startswith(("fit", "select", "tune", "calibr"))
            )
            if bad:
                _add(violations, "heldout_fit", f"{path}.usable_for", f"test/OOD cannot be used for {bad}")


def _validate_grouping(protocol, violations):
    grouping = _required_object(protocol, "grouping", violations, "$")
    if grouping is None:
        return
    keys = grouping.get("keys")
    if keys != ["question_type", "option_count"]:
        _add(violations, "wrong_group_keys", "$.grouping.keys", "grouping keys must be question_type and option_count")
    types = grouping.get("question_types")
    if not isinstance(types, list) or not types or not all(_string(item) for item in types):
        _add(violations, "malformed_field", "$.grouping.question_types", "must be a non-empty string list")
    counts = grouping.get("option_counts")
    if counts != [2, 3, 4]:
        _add(violations, "wrong_option_counts", "$.grouping.option_counts", "must be exactly [2, 3, 4]")
    for key in ("minimum_calibration_questions", "minimum_calibration_states"):
        value = grouping.get(key)
        expected = 32 if key == "minimum_calibration_questions" else 8
        if type(value) is not int or value != expected:
            _add(violations, "wrong_minimum", f"$.grouping.{key}", f"must be exactly {expected}")
    if grouping.get("missing_group_policy") != "report_unestimated_and_keep_baseline":
        _add(violations, "unsafe_missing_group_policy", "$.grouping.missing_group_policy", "missing groups must keep baseline")
    if grouping.get("group_identity_source") != "row.question_type plus len(row.options), recomputed from frozen input":
        _add(violations, "unrecomputed_group", "$.grouping.group_identity_source", "group identity must be recomputed from input")


def _validate_objectives(protocol, violations):
    objectives = _required_object(protocol, "objectives", violations, "$")
    if objectives is None:
        return
    if objectives.get("primary") != "nll":
        _add(violations, "wrong_primary_objective", "$.objectives.primary", "primary objective must be nll")
    secondary = objectives.get("secondary")
    if not isinstance(secondary, list) or not {"brier", "ece_fixed_10", "coverage_selective_risk"}.issubset(secondary):
        _add(violations, "incomplete_objectives", "$.objectives.secondary", "required calibration metrics are missing")
    search = _required_object(objectives, "search", violations, "$.objectives")
    if search is not None:
        if search.get("temperature_min") != 0.05 or search.get("temperature_max") != 20.0:
            _add(violations, "wrong_search_range", "$.objectives.search", "temperature range must be 0.05..20.0")
        if type(search.get("grid_points")) is not int or search.get("grid_points") != 4000:
            _add(violations, "wrong_search_grid", "$.objectives.search.grid_points", "grid_points must be 4000")
        if type(search.get("refine_steps")) is not int or search.get("refine_steps") != 200:
            _add(violations, "wrong_refine_steps", "$.objectives.search.refine_steps", "refine_steps must be 200")
        _required_bool(search, "include_baseline_temperature", violations, "$.objectives.search", True)
    thresholds = objectives.get("thresholds")
    if thresholds != [0.5, 0.7, 0.9]:
        _add(violations, "wrong_thresholds", "$.objectives.thresholds", "thresholds must remain [0.5, 0.7, 0.9]")


def _validate_boundaries(protocol, violations):
    boundaries = _required_object(protocol, "boundaries", violations, "$")
    if boundaries is None:
        return
    for key, value in boundaries.items():
        if key.endswith("authorized") or key.startswith("authorizes_") or key in {"model_weights_modified", "checkpoint_written", "network_allowed"}:
            if type(value) is not bool or value is not False:
                _add(violations, "authorization_flag", f"$.boundaries.{key}", "boundary must be exactly false")
    if boundaries.get("shadow_only") is not True:
        _add(violations, "shadow_required", "$.boundaries.shadow_only", "protocol must remain shadow-only")


def validate_protocol(protocol):
    violations = []
    if not isinstance(protocol, dict):
        return _report(False, [{"code": "malformed_root", "path": "$", "message": "protocol must be an object"}], protocol)
    _scan_auth(protocol, "$", violations)
    if protocol.get("schema_version") != SCHEMA_VERSION:
        _add(violations, "wrong_schema_version", "$.schema_version", f"must be {SCHEMA_VERSION!r}")
    _required_string(protocol, "protocol_id", violations, "$")
    if protocol.get("status") != "frozen_research_protocol":
        _add(violations, "wrong_status", "$.status", "must be frozen_research_protocol")
    if protocol.get("mode") != "read_only_measurement":
        _add(violations, "wrong_mode", "$.mode", "must be read_only_measurement")
    _required_bool(protocol, "frozen_before_inference", violations, "$")
    _required_bool(protocol, "read_only", violations, "$")
    if type(protocol.get("network_model_calls")) is not int or protocol.get("network_model_calls") != 0:
        _add(violations, "network_not_zero", "$.network_model_calls", "must be integer zero")
    model = _required_object(protocol, "model", violations, "$")
    if model is not None:
        _required_string(model, "checkpoint_id", violations, "$.model")
        if model.get("baseline_temperature") != 1.0:
            _add(violations, "wrong_baseline_temperature", "$.model.baseline_temperature", "must be exactly 1.0")
        for key in ("service_files_may_change", "candidate_checkpoint_write"):
            _required_bool(model, key, violations, "$.model", False)
    _validate_splits(protocol, violations)
    _validate_grouping(protocol, violations)
    _validate_objectives(protocol, violations)
    uncertainty = _required_object(protocol, "uncertainty", violations, "$")
    if uncertainty is not None:
        if uncertainty.get("resampling_unit") != "state_id":
            _add(violations, "wrong_resampling_unit", "$.uncertainty.resampling_unit", "must be state_id")
        if uncertainty.get("bootstrap_replicates") != 2000:
            _add(violations, "wrong_bootstrap_replicates", "$.uncertainty.bootstrap_replicates", "must be 2000")
        if uncertainty.get("confidence_level") != 0.95:
            _add(violations, "wrong_confidence_level", "$.uncertainty.confidence_level", "must be 0.95")
        if uncertainty.get("paired_against") != "temperature_1.0_same_row_and_same_group":
            _add(violations, "unpaired_baseline", "$.uncertainty.paired_against", "must pair each row/group with T=1.0")
    controls = _required_list(protocol, "controls", violations, "$")
    if controls is not None:
        ids = []
        for index, control in enumerate(controls):
            path = f"$.controls[{index}]"
            if not isinstance(control, dict):
                _add(violations, "malformed_control", path, "control must be an object")
                continue
            control_id = _required_string(control, "control_id", violations, path)
            _required_bool(control, "required", violations, path, True)
            _required_string(control, "description", violations, path)
            if control_id is not None:
                ids.append(control_id)
        if len(ids) != len(set(ids)):
            _add(violations, "duplicate_control", "$.controls", "control IDs must be unique")
        missing = sorted(set(REQUIRED_CONTROLS) - set(ids))
        if missing:
            _add(violations, "missing_control", "$.controls", f"missing controls: {missing}")
    acceptance = _required_object(protocol, "acceptance", violations, "$")
    if acceptance is not None:
        for key in ("service_temperature_change", "checkpoint_promotion", "ood_threshold_tuning", "no_global_winner", "review_required_before_any_serving_change"):
            _required_bool(acceptance, key, violations, "$.acceptance", False if key != "no_global_winner" and key != "review_required_before_any_serving_change" else True)
        for key in ("noninferiority_margin_must_be_preregistered", "confidence_level_must_be_preregistered"):
            _required_bool(acceptance, key, violations, "$.acceptance", True)
        if acceptance.get("protected_confident_error_limit") != 0:
            _add(violations, "protected_error_limit", "$.acceptance.protected_confident_error_limit", "must be zero")
    fields = _required_list(protocol, "receipt_fields", violations, "$")
    if fields is not None and len(fields) != len(set(fields)):
        _add(violations, "duplicate_receipt_field", "$.receipt_fields", "receipt fields must be unique")
    _validate_boundaries(protocol, violations)
    return _report(not violations, violations, protocol)


def _report(valid, violations, protocol):
    return {
        "schema_version": REPORT_VERSION,
        "protocol_id": protocol.get("protocol_id") if isinstance(protocol, dict) else None,
        "status": "protocol_valid_not_authorized" if valid else "protocol_blocked",
        "valid": bool(valid),
        "violations": violations,
        "block_reasons": sorted({item["code"] for item in violations}),
        "network_model_calls": 0,
        "model_loaded": False,
        "measurement_authorized": False,
        "training_authorized": False,
        "deployment_authorized": False,
        "production_pruning_authorized": False,
        "promotion_authorized": False,
        "authorizes_execution": False,
    }


def load_protocol(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"protocol_id": None, "_load_error": str(exc)}


def validate_file(path: Path):
    protocol = load_protocol(path)
    report = validate_protocol(protocol)
    if "_load_error" in protocol:
        report["valid"] = False
        report["status"] = "protocol_blocked"
        report["block_reasons"] = ["protocol_read_error"]
        report["violations"] = [{"code": "protocol_read_error", "path": str(path), "message": protocol["_load_error"]}]
    return report


def _write_exclusive(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report = validate_file(Path(args.protocol))
    output = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    try:
        if args.output:
            _write_exclusive(Path(args.output), report)
    except OSError as exc:
        report = _report(False, [{"code": "output_error", "path": str(args.output), "message": str(exc)}], report)
        output = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        print(output, end="")
        return 2
    print(output, end="")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
