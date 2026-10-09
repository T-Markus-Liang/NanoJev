#!/usr/bin/env python3
"""Read-only, fail-closed validator for the NanoJev V3 N3 readout protocol.

Standard library only. This validator reads exactly one JSON protocol file. It never
loads a model, checkpoint or dataset, never imports a third-party package and never
accesses the network. A clean report is a schema/isolation statement only: it always
declares ``training_authorized=false`` and ``deployment_authorized=false`` regardless
of the protocol contents.

CLI:
    python3 scripts/validate_nanojev_v3_n3_readout_protocol_v1.py \
        --protocol research/nanojev_v3_n3_readout_protocol_v1.json [--output report.json]

Exit codes:
    0  protocol is structurally valid (still never an authorization)
    2  protocol is missing, malformed, blocked, or the report path already exists
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA_VERSION = "nanojev-v3-n3-readout-protocol-v1"
REPORT_VERSION = "nanojev-v3-n3-readout-protocol-report-v1"

REQUIRED_ARMS = ("trained_head", "schema_readout", "encoder_router")
REQUIRED_SPLIT_ROLES = ("train", "dev", "calibration", "test", "ood")
REQUIRED_CONTROLS = (
    "option_permutation",
    "label_permutation",
    "boolean_coverage",
    "score_coverage",
    "ood_cases",
    "protected_cases",
    "constant_baselines",
)
REQUIRED_CONSTANT_BASELINES = ("constant_true", "constant_false")
REQUIRED_RECEIPT_FIELDS = (
    "arm_id",
    "arm_role",
    "split",
    "case_id",
    "question_type",
    "option_order",
    "option_permutation_id",
    "label_permutation_id",
    "label_assignment",
    "predicted_label",
    "probabilities",
    "confidence",
    "coverage",
    "correct",
    "constant_true_correct",
    "constant_false_correct",
    "control_id",
    "protected_case",
    "latency_ms",
    "memory_bytes",
    "receipt_sha256",
    "network_model_calls",
    "training_authorized",
    "deployment_authorized",
)
# test/OOD roles must not be usable for any fitting, selection or calibration step.
FORBIDDEN_EVAL_USAGE = frozenset({"fit", "select", "calibrate", "tune", "train", "promote"})
# Any of these keys carrying a truthy value is an authorization-like flag and blocks.
AUTHORIZATION_KEYS = frozenset(
    {
        "training_authorized",
        "deployment_authorized",
        "production_pruning_authorized",
        "active_pruning_authorized",
        "promotion_authorized",
        "execution_authorized",
        "deploy_authorized",
        "authorizes_execution",
        "authorizes_training",
        "authorizes_deployment",
        "authorizes_production_pruning",
    }
)


def _add(violations, code, path, message):
    violations.append({"code": code, "path": path, "message": message})


def _is_string(value):
    return isinstance(value, str) and bool(value.strip())


def _require_string(obj, key, violations, path):
    if key not in obj:
        _add(violations, "missing_field", f"{path}.{key}", "required string is absent")
        return None
    value = obj[key]
    if not _is_string(value):
        _add(violations, "malformed_field", f"{path}.{key}", "must be a non-empty string")
        return None
    return value


def _require_bool(obj, key, violations, path, expected=True, code="malformed_field"):
    if key not in obj:
        _add(violations, "missing_field", f"{path}.{key}", "required boolean is absent")
        return None
    value = obj[key]
    if value is not expected:
        _add(violations, code, f"{path}.{key}", f"must be exactly {str(expected).lower()}")
        return None
    return value


def _require_object(obj, key, violations, path):
    if key not in obj:
        _add(violations, "missing_field", f"{path}.{key}", "required object is absent")
        return None
    value = obj[key]
    if not isinstance(value, dict):
        _add(violations, "malformed_field", f"{path}.{key}", "must be a JSON object")
        return None
    return value


def _require_list(obj, key, violations, path):
    if key not in obj:
        _add(violations, "missing_field", f"{path}.{key}", "required list is absent")
        return None
    value = obj[key]
    if not isinstance(value, list):
        _add(violations, "malformed_field", f"{path}.{key}", "must be a JSON list")
        return None
    return value


def _string_list(value):
    return isinstance(value, list) and bool(value) and all(_is_string(item) for item in value)


def _scan_authorization(node, path, violations):
    """Reject any authorization-like key whose value is not exactly false."""
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}"
            lowered = key.lower()
            is_authorization = (
                key in AUTHORIZATION_KEYS
                or lowered.endswith("_authorized")
                or lowered.endswith("_authorization")
                or lowered.startswith("authorizes_")
            )
            if is_authorization and value is not False:
                _add(
                    violations,
                    "authorization_flag",
                    child,
                    "authorization-like flags must be exactly false in a read-only protocol",
                )
            _scan_authorization(value, child, violations)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _scan_authorization(item, f"{path}[{index}]", violations)


def _validate_arms(protocol, violations):
    arms = _require_list(protocol, "paired_arms", violations, "$")
    present = []
    if arms is None:
        return present
    seen = set()
    for index, arm in enumerate(arms):
        path = f"$.paired_arms[{index}]"
        if not isinstance(arm, dict):
            _add(violations, "malformed_field", path, "arm must be an object")
            continue
        arm_id = _require_string(arm, "arm_id", violations, path)
        if arm_id is None:
            continue
        if arm_id in seen:
            _add(violations, "duplicate_arm", f"{path}.arm_id", f"duplicate arm {arm_id!r}")
        else:
            present.append(arm_id)
        seen.add(arm_id)
        if arm_id not in REQUIRED_ARMS:
            _add(violations, "unknown_arm", f"{path}.arm_id", f"{arm_id!r} is not a declared paired arm")
        for key in ("role", "description", "input_serialization", "readout"):
            _require_string(arm, key, violations, path)
        _require_bool(arm, "paired", violations, path, True, "unpaired_arm")
    missing = [arm for arm in REQUIRED_ARMS if arm not in seen]
    if missing:
        _add(violations, "missing_arm", "$.paired_arms", f"missing arms: {', '.join(missing)}")

    pairing = _require_object(protocol, "pairing", violations, "$")
    if pairing is not None:
        for key in ("paired", "same_frozen_splits", "same_controls", "same_receipt_fields"):
            _require_bool(pairing, key, violations, "$.pairing", True, "unpaired_arm")
    return present


def _validate_split_roles(protocol, violations):
    roles = _require_object(protocol, "split_roles", violations, "$")
    present = []
    if roles is None:
        return present
    for name in REQUIRED_SPLIT_ROLES:
        path = f"$.split_roles.{name}"
        if name not in roles:
            _add(violations, "missing_split_role", path, "required split role is absent")
            continue
        role = roles[name]
        if not isinstance(role, dict):
            _add(violations, "malformed_field", path, "split role must be an object")
            continue
        present.append(name)
        _require_string(role, "description", violations, path)
        _require_bool(role, "frozen", violations, path, True, "unfrozen_split")
        usable = _require_list(role, "usable_for", violations, path)
        if usable is not None:
            if not _string_list(usable):
                _add(
                    violations,
                    "malformed_field",
                    f"{path}.usable_for",
                    "must be a non-empty list of strings",
                )
            elif name in ("test", "ood"):
                bad = sorted({item for item in usable if item in FORBIDDEN_EVAL_USAGE})
                if bad:
                    _add(
                        violations,
                        "unfrozen_split",
                        f"{path}.usable_for",
                        f"test/OOD must not be usable for {', '.join(bad)}",
                    )
    extra = sorted(set(roles) - set(REQUIRED_SPLIT_ROLES))
    if extra:
        _add(violations, "unknown_split_role", "$.split_roles", f"unexpected split roles: {', '.join(extra)}")
    return present


def _validate_controls(protocol, violations):
    controls = _require_list(protocol, "controls", violations, "$")
    present = []
    if controls is None:
        return present
    seen = set()
    for index, control in enumerate(controls):
        path = f"$.controls[{index}]"
        if not isinstance(control, dict):
            _add(violations, "malformed_field", path, "control must be an object")
            continue
        control_id = _require_string(control, "control_id", violations, path)
        if control_id is None:
            continue
        if control_id in seen:
            _add(violations, "duplicate_control", f"{path}.control_id", f"duplicate control {control_id!r}")
        else:
            present.append(control_id)
        seen.add(control_id)
        _require_string(control, "kind", violations, path)
        _require_string(control, "description", violations, path)
        _require_bool(control, "required", violations, path, True, "malformed_field")
        applies = _require_list(control, "applies_to", violations, path)
        if applies is not None:
            if not _string_list(applies):
                _add(violations, "malformed_field", f"{path}.applies_to", "must be a non-empty list of strings")
            else:
                unknown = sorted(set(applies) - set(REQUIRED_ARMS))
                if unknown:
                    _add(violations, "unknown_arm", f"{path}.applies_to", f"unknown arms: {', '.join(unknown)}")
                missing_arms = sorted(set(REQUIRED_ARMS) - set(applies))
                if missing_arms:
                    _add(
                        violations,
                        "incomplete_control",
                        f"{path}.applies_to",
                        f"control must apply to every paired arm; missing: {', '.join(missing_arms)}",
                    )
    missing = [control for control in REQUIRED_CONTROLS if control not in seen]
    if missing:
        _add(violations, "missing_control", "$.controls", f"missing controls: {', '.join(missing)}")
    return present


def _validate_option_permutation(protocol, violations):
    control = _require_object(protocol, "option_permutation", violations, "$")
    if control is None:
        return
    path = "$.option_permutation"
    _require_bool(control, "enabled", violations, path, True, "disabled_control")
    _require_bool(control, "identity_first", violations, path, True)
    _require_bool(control, "all_permutations", violations, path, True)
    _require_bool(control, "semantic_ids_fixed", violations, path, True)
    types = _require_list(control, "applies_to_question_types", violations, path)
    if types is not None:
        if not _string_list(types):
            _add(
                violations,
                "malformed_field",
                f"{path}.applies_to_question_types",
                "must be a non-empty list of strings",
            )
        else:
            missing = [name for name in ("boolean", "score", "choice") if name not in types]
            if missing:
                _add(
                    violations,
                    "incomplete_coverage",
                    f"{path}.applies_to_question_types",
                    f"missing question types: {', '.join(missing)}",
                )
    max_options = control.get("max_options")
    if not isinstance(max_options, int) or isinstance(max_options, bool) or not 2 <= max_options <= 4:
        _add(violations, "malformed_field", f"{path}.max_options", "must be an integer from 2 through 4")


def _validate_label_permutation(protocol, violations):
    control = _require_object(protocol, "label_permutation", violations, "$")
    if control is None:
        return
    path = "$.label_permutation"
    _require_bool(control, "enabled", violations, path, True, "disabled_control")
    _require_bool(control, "preserve_option_set", violations, path, True)
    _require_bool(control, "preserve_label_multiset", violations, path, True)


def _validate_coverage(protocol, violations):
    coverage = _require_object(protocol, "coverage", violations, "$")
    if coverage is None:
        return
    for name, minimum in (("boolean", 2), ("score", 2)):
        path = f"$.coverage.{name}"
        if name not in coverage:
            _add(violations, "missing_field", path, "required coverage block is absent")
            continue
        block = coverage[name]
        if not isinstance(block, dict):
            _add(violations, "malformed_field", path, "coverage block must be an object")
            continue
        _require_bool(block, "required", violations, path, True, "incomplete_coverage")
        counts = _require_list(block, "option_counts", violations, path)
        if counts is not None:
            valid = bool(counts) and all(
                isinstance(count, int) and not isinstance(count, bool)
                and minimum <= count <= 4
                for count in counts
            )
            if not valid:
                _add(
                    violations,
                    "malformed_field",
                    f"{path}.option_counts",
                    f"must be a non-empty list of integers from {minimum} through 4",
                )
        if name == "boolean" and counts is not None and counts != [2]:
            _add(violations, "incomplete_coverage", f"{path}.option_counts", "Boolean coverage must declare exactly [2]")


def _validate_ood(protocol, violations):
    ood = _require_object(protocol, "ood", violations, "$")
    if ood is None:
        return
    path = "$.ood"
    _require_bool(ood, "required", violations, path, True, "missing_ood")
    split = _require_string(ood, "split", violations, path)
    if split is not None and split != "ood":
        _add(violations, "missing_ood", f"{path}.split", "OOD cases must come from the ood split")
    _require_bool(ood, "abstain_required", violations, path, True, "missing_ood")
    _require_bool(ood, "no_threshold_tuning_on_ood", violations, path, True)


def _validate_protected_cases(protocol, violations):
    cases = _require_list(protocol, "protected_cases", violations, "$")
    present = []
    if cases is None:
        return present
    if not cases:
        _add(violations, "missing_protected_case", "$.protected_cases", "at least one protected case is required")
        return present
    seen = set()
    for index, case in enumerate(cases):
        path = f"$.protected_cases[{index}]"
        if not isinstance(case, dict):
            _add(violations, "malformed_field", path, "protected case must be an object")
            continue
        case_id = _require_string(case, "case_id", violations, path)
        if case_id is not None:
            if case_id in seen:
                _add(violations, "duplicate_protected_case", f"{path}.case_id", f"duplicate case {case_id!r}")
            else:
                present.append(case_id)
            seen.add(case_id)
        _require_string(case, "reason", violations, path)
        split = _require_string(case, "split", violations, path)
        if split is not None and split not in ("test", "ood"):
            _add(violations, "missing_protected_case", f"{path}.split", "protected cases must belong to test or ood")
    return present


def _validate_constant_baselines(protocol, violations):
    baselines = _require_list(protocol, "constant_baselines", violations, "$")
    if baselines is None:
        return []
    if not _string_list(baselines):
        _add(violations, "malformed_field", "$.constant_baselines", "must be a non-empty list of strings")
        return []
    if len(set(baselines)) != len(baselines):
        _add(violations, "duplicate_control", "$.constant_baselines", "constant baselines must be unique")
    missing = [name for name in REQUIRED_CONSTANT_BASELINES if name not in baselines]
    if missing:
        _add(violations, "missing_control", "$.constant_baselines", f"missing constant baselines: {', '.join(missing)}")
    return list(baselines)


def _validate_receipt_fields(protocol, violations):
    fields = _require_list(protocol, "paired_receipt_fields", violations, "$")
    if fields is None:
        return []
    if not _string_list(fields):
        _add(violations, "malformed_field", "$.paired_receipt_fields", "must be a non-empty list of strings")
        return []
    if len(set(fields)) != len(fields):
        _add(violations, "duplicate_receipt_field", "$.paired_receipt_fields", "receipt fields must be unique")
    missing = [field for field in REQUIRED_RECEIPT_FIELDS if field not in fields]
    if missing:
        _add(violations, "missing_receipt_field", "$.paired_receipt_fields", f"missing receipt fields: {', '.join(missing)}")
    return list(fields)


def _validate_boundaries(protocol, violations):
    boundaries = _require_object(protocol, "boundaries", violations, "$")
    if boundaries is None:
        return
    path = "$.boundaries"
    for key in (
        "training_authorized",
        "deployment_authorized",
        "production_pruning_authorized",
        "active_pruning_authorized",
        "promotion_authorized",
        "authorizes_execution",
        "authorizes_training",
        "authorizes_deployment",
    ):
        _require_bool(boundaries, key, violations, path, False, "authorization_flag")
    for key in ("model_weights_modified", "checkpoint_written", "network_allowed"):
        _require_bool(boundaries, key, violations, path, False, "authorization_flag")


def _report(source, violations, arms, split_roles, controls, counts):
    codes = sorted({violation["code"] for violation in violations})
    valid = not violations
    return {
        "report_version": REPORT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "source": str(source),
        "valid": valid,
        "status": "protocol_valid_not_authorized" if valid else "protocol_blocked",
        "violations": violations,
        "block_reasons": codes,
        "counts": counts,
        "arms_present": arms,
        "split_roles_present": split_roles,
        "controls_present": controls,
        "training_authorized": False,
        "deployment_authorized": False,
        "production_pruning_authorized": False,
        "training_performed": False,
        "deployment_performed": False,
        "model_loaded": False,
        "network_model_calls": 0,
    }


def validate_protocol(protocol, source="<memory>"):
    """Validate a parsed protocol object and return a machine-readable report."""
    violations = []
    if not isinstance(protocol, dict):
        _add(violations, "malformed_protocol", "$", "protocol root must be a JSON object")
        empty = {"arms": 0, "split_roles": 0, "controls": 0, "protected_cases": 0, "receipt_fields": 0, "constant_baselines": 0}
        return _report(source, violations, [], [], [], empty)

    _scan_authorization(protocol, "$", violations)

    schema_version = _require_string(protocol, "schema_version", violations, "$")
    if schema_version is not None and schema_version != SCHEMA_VERSION:
        _add(violations, "schema_version_mismatch", "$.schema_version", f"expected {SCHEMA_VERSION!r}")

    for key in ("protocol_id", "purpose", "status", "mode", "decision_rule"):
        _require_string(protocol, key, violations, "$")

    _require_bool(protocol, "frozen_before_inference", violations, "$", True, "not_frozen")
    _require_bool(protocol, "read_only", violations, "$", True, "not_read_only")

    network = protocol.get("network_model_calls", "<missing>")
    if network == "<missing>":
        _add(violations, "missing_field", "$.network_model_calls", "required field is absent")
    elif network != 0:
        _add(violations, "network_enabled", "$.network_model_calls", "must be exactly 0")

    arms = _validate_arms(protocol, violations)
    split_roles = _validate_split_roles(protocol, violations)
    controls = _validate_controls(protocol, violations)
    _validate_option_permutation(protocol, violations)
    _validate_label_permutation(protocol, violations)
    _validate_coverage(protocol, violations)
    _validate_ood(protocol, violations)
    protected = _validate_protected_cases(protocol, violations)
    baselines = _validate_constant_baselines(protocol, violations)
    receipt_fields = _validate_receipt_fields(protocol, violations)
    _validate_boundaries(protocol, violations)

    limitations = _require_list(protocol, "limitations", violations, "$")
    if limitations is not None and not _string_list(limitations):
        _add(violations, "malformed_field", "$.limitations", "must be a non-empty list of strings")

    counts = {
        "arms": len(arms),
        "split_roles": len(split_roles),
        "controls": len(controls),
        "protected_cases": len(protected),
        "receipt_fields": len(receipt_fields),
        "constant_baselines": len(baselines),
    }
    return _report(source, violations, arms, split_roles, controls, counts)


def load_protocol(path):
    """Read and parse exactly one JSON protocol file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_file(path):
    """Load and validate a protocol file, converting read/parse errors to a blocked report."""
    try:
        protocol = load_protocol(path)
    except (OSError, ValueError) as exc:
        violations = [{"code": "unreadable_protocol", "path": "$", "message": str(exc)}]
        empty = {"arms": 0, "split_roles": 0, "controls": 0, "protected_cases": 0, "receipt_fields": 0, "constant_baselines": 0}
        return _report(path, violations, [], [], [], empty)
    return validate_protocol(protocol, source=path)


def _render(report):
    return json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "research/nanojev_v3_n3_readout_protocol_v1.json",
        help="Path to the read-only N3 readout protocol JSON.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional exclusive report path; never overwritten.")
    args = parser.parse_args(argv)

    report = validate_file(args.protocol)
    if args.output is not None:
        try:
            with Path(args.output).open("x", encoding="utf-8") as stream:
                stream.write(_render(report))
        except FileExistsError:
            print(f"refusing to overwrite existing report: {args.output}", file=sys.stderr)
            return 2
    sys.stdout.write(_render(report))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
