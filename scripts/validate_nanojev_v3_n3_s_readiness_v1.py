#!/usr/bin/env python3
"""Read-only, fail-closed readiness validator for the V3 N3-S measurement gate.

The validator checks pinned JSON receipts and review declarations only.  It does
not load a model, artifact, tokenizer or dataset, never accesses the network,
and never turns readiness into authorization for inference, training, deployment
or promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


SCHEMA_VERSION = "nanojev-v3-n3-s-readiness-v1"
REPORT_VERSION = "nanojev-v3-n3-s-readiness-report-v1"
# Minimum producer contracts live in code, not in caller-editable expectations.
# In particular, N2 has no `valid`, `model_loaded` or `network_model_calls` field.
EVIDENCE_CONTRACTS = {
    "n2_domain_pack_preflight": {
        "kind": "domain_pack_preflight",
        "expected_status": "preflight_passed_not_training_authorized",
        "required_values": {
            "schema_version": "nanojev-v3-domain-pack-preflight-v1",
            "input_files_changed": False,
            "manifest_errors": [],
            "violations": [],
            "block_reasons": [],
            "training_authorized": False,
            "training_performed": False,
            "merged_rows_written": 0,
        },
    },
    "n3_protocol_preflight": {
        "kind": "readout_protocol_preflight",
        "expected_status": "protocol_valid_not_authorized",
        "required_values": {
            "report_version": "nanojev-v3-n3-readout-protocol-report-v1",
            "schema_version": "nanojev-v3-n3-readout-protocol-v1",
            "valid": True,
            "training_authorized": False,
            "deployment_authorized": False,
            "production_pruning_authorized": False,
            "training_performed": False,
            "deployment_performed": False,
            "model_loaded": False,
            "network_model_calls": 0,
        },
    },
    "n3_synthetic_controls": {
        "kind": "synthetic_controls",
        "expected_status": "synthetic_controls_passed_not_model_evidence",
        "required_values": {
            "report_version": "nanojev-v3-n3-synthetic-report-v1",
            "protocol_schema_version": "nanojev-v3-n3-readout-protocol-v1",
            "synthetic_only": True,
            "training_authorized": False,
            "deployment_authorized": False,
            "production_pruning_authorized": False,
            "training_performed": False,
            "deployment_performed": False,
            "model_loaded": False,
            "network_model_calls": 0,
        },
    },
}
REQUIRED_EVIDENCE_IDS = tuple(EVIDENCE_CONTRACTS)
N2_SPLITS = ("train", "dev", "calibration", "test", "ood")
REQUIRED_REVIEW_IDS = (
    "n2_domain_pack_independent_review",
    "n3_protocol_and_controls_independent_review",
    "labels_holdout_and_protected_cases_review",
    "data_license_and_provenance_review",
)
REVIEW_STATUSES = frozenset({"pending", "approved", "rejected", "not_recorded"})
BOUNDARY_EXPECTATIONS = {
    "read_only": True,
    "network_allowed": False,
    "model_loaded": False,
    "measurement_performed": False,
    "training_authorized": False,
    "deployment_authorized": False,
    "production_pruning_authorized": False,
    "promotion_authorized": False,
    "real_n3_measurement_authorized": False,
    "authorizes_execution": False,
}
AUTHORIZATION_SUFFIXES = ("_authorized", "_authorization")


def _add(violations, code, path, message):
    violations.append({"code": code, "path": path, "message": message})


def _is_string(value):
    return isinstance(value, str) and bool(value.strip())


def _strict_equal(actual, expected):
    """JSON equality without Python's bool/int/float coercion."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _strict_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(text):
    return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_object)


def _stable_source(source, root):
    """Return a report source marker that is stable across checkout paths."""
    if not isinstance(source, (str, Path)):
        return str(source)
    try:
        candidate = Path(source)
        if candidate.is_absolute():
            return str(candidate.resolve().relative_to(Path(root).resolve()))
    except (OSError, RuntimeError, ValueError):
        pass
    return str(source)


def _require_string(obj, key, violations, path):
    value = obj.get(key)
    if not _is_string(value):
        _add(violations, "missing_or_malformed_field", f"{path}.{key}", "must be a non-empty string")
        return None
    return value


def _require_bool(obj, key, expected, violations, path):
    value = obj.get(key)
    if type(value) is not bool or value is not expected:
        _add(violations, "boundary_violation", f"{path}.{key}", f"must be exactly {str(expected).lower()}")
        return None
    return value


def _scan_authorization(node, path, violations):
    if isinstance(node, dict):
        for key, value in node.items():
            if not isinstance(key, str):
                _add(violations, "malformed_key", path, "JSON object keys must be strings")
                continue
            lowered = key.lower()
            is_auth = (
                lowered.startswith("authorizes_")
                or lowered.endswith(AUTHORIZATION_SUFFIXES)
                or lowered.endswith("_performed")
                or lowered in {name.lower() for name in BOUNDARY_EXPECTATIONS}
            )
            if is_auth and value is not False and not (key == "read_only" and value is True):
                _add(violations, "authorization_flag", f"{path}.{key}", "authorization-like values must be false")
            if lowered == "network_model_calls" and not _strict_equal(value, 0):
                _add(violations, "network_activity", f"{path}.{key}", "must be integer zero")
            _scan_authorization(value, f"{path}.{key}", violations)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _scan_authorization(value, f"{path}[{index}]", violations)


def _safe_relative_path(value, root, violations, path):
    if not _is_string(value):
        _add(violations, "unsafe_evidence_path", path, "evidence path must be a non-empty relative string")
        return None
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        _add(violations, "unsafe_evidence_path", path, "evidence path must not be absolute or escape the workspace root")
        return None
    try:
        resolved = (root / candidate).resolve()
        resolved.relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        _add(violations, "unsafe_evidence_path", path, "evidence path resolves outside the workspace root")
        return None
    return resolved


def _validate_evidence_contract(protocol, root, violations):
    entries = protocol.get("required_evidence")
    if not isinstance(entries, list):
        _add(violations, "missing_or_malformed_field", "$.required_evidence", "must be a list")
        return [], []
    seen = set()
    entry_by_id = {}
    for index, entry in enumerate(entries):
        path = f"$.required_evidence[{index}]"
        if not isinstance(entry, dict):
            _add(violations, "malformed_evidence_declaration", path, "must be an object")
            continue
        evidence_id = _require_string(entry, "evidence_id", violations, path)
        _require_string(entry, "kind", violations, path)
        _require_string(entry, "path", violations, path)
        _require_string(entry, "expected_status", violations, path)
        expected_hash = entry.get("expected_sha256")
        if not _is_string(expected_hash) or len(expected_hash) != 64:
            _add(violations, "missing_pinned_hash", f"{path}.expected_sha256", "must be a 64-character SHA-256")
        elif re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
            _add(violations, "malformed_pinned_hash", f"{path}.expected_sha256", "must be lowercase hexadecimal SHA-256")
        values = entry.get("required_values")
        if not isinstance(values, dict) or not values:
            _add(violations, "malformed_evidence_declaration", f"{path}.required_values", "must be a non-empty object")
        if evidence_id is None:
            continue
        frozen = EVIDENCE_CONTRACTS.get(evidence_id)
        if frozen is not None:
            for key, expected in frozen.items():
                if not _strict_equal(entry.get(key), expected):
                    _add(violations, "evidence_contract_mismatch", f"{path}.{key}", "must match the frozen producer contract")
        if evidence_id in seen:
            _add(violations, "duplicate_evidence", f"{path}.evidence_id", f"duplicate evidence id {evidence_id!r}")
        seen.add(evidence_id)
        entry_by_id[evidence_id] = entry
    missing = [item for item in REQUIRED_EVIDENCE_IDS if item not in seen]
    unknown = sorted(seen - set(REQUIRED_EVIDENCE_IDS))
    if missing:
        _add(violations, "missing_evidence_declaration", "$.required_evidence", f"missing: {', '.join(missing)}")
    if unknown:
        _add(violations, "unknown_evidence_declaration", "$.required_evidence", f"unknown: {', '.join(unknown)}")
    if len(entries) != len(REQUIRED_EVIDENCE_IDS) or len(seen) != len(REQUIRED_EVIDENCE_IDS):
        _add(violations, "evidence_count", "$.required_evidence", "must contain exactly the three frozen evidence declarations")

    results = []
    for evidence_id in REQUIRED_EVIDENCE_IDS:
        entry = entry_by_id.get(evidence_id)
        if entry is None:
            results.append({"evidence_id": evidence_id, "present": False, "status": "missing_declaration"})
            continue
        item_path = f"$.required_evidence[{evidence_id}]"
        evidence_path = _safe_relative_path(entry.get("path"), root, violations, f"{item_path}.path")
        result = {
            "evidence_id": evidence_id,
            "kind": entry.get("kind"),
            "path": entry.get("path"),
            "present": False,
            "sha256": None,
            "status": None,
            "checks": {},
        }
        if evidence_path is None:
            _add(violations, "missing_evidence", f"{item_path}.path", "pinned evidence file is missing or not a regular file")
            results.append(result)
            continue
        try:
            if not evidence_path.is_file():
                _add(violations, "missing_evidence", f"{item_path}.path", "pinned evidence file is missing or not a regular file")
                results.append(result)
                continue
            raw = evidence_path.read_bytes()
        except (OSError, ValueError) as error:
            _add(violations, "evidence_unreadable", f"{item_path}.path", f"cannot read pinned evidence: {error}")
            results.append(result)
            continue
        result["present"] = True
        actual_hash = hashlib.sha256(raw).hexdigest()
        result["sha256"] = actual_hash
        expected_hash = entry.get("expected_sha256")
        if actual_hash != expected_hash:
            _add(violations, "evidence_hash_mismatch", f"{item_path}.expected_sha256", "evidence bytes do not match the pinned hash")
        try:
            evidence = _load_json(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            _add(violations, "malformed_evidence", f"{item_path}.path", f"evidence is not valid UTF-8 JSON: {error}")
            results.append(result)
            continue
        if not isinstance(evidence, dict):
            _add(violations, "malformed_evidence", f"{item_path}.path", "evidence root must be an object")
            results.append(result)
            continue
        actual_status = evidence.get("status")
        result["status"] = actual_status
        expected_status = EVIDENCE_CONTRACTS[evidence_id]["expected_status"]
        if actual_status != expected_status:
            _add(violations, "evidence_status", f"{item_path}.expected_status", f"expected {expected_status!r}, got {actual_status!r}")
        required_values = EVIDENCE_CONTRACTS[evidence_id]["required_values"]
        for key, expected in required_values.items():
            actual = evidence.get(key)
            result["checks"][key] = key in evidence and _strict_equal(actual, expected)
            if not result["checks"][key]:
                _add(violations, "evidence_field", f"{item_path}.{key}", f"expected {expected!r}, got {actual!r}")
        if evidence_id == "n2_domain_pack_preflight":
            hashes = evidence.get("source_hashes")
            source_paths = evidence.get("source_paths")
            split_paths = source_paths.get("splits") if isinstance(source_paths, dict) else None
            source_names = [source_paths.get("manifest")] if isinstance(source_paths, dict) else []
            if isinstance(split_paths, dict):
                source_names.extend(split_paths.get(split) for split in N2_SPLITS)
            paths_valid = (
                isinstance(source_paths, dict)
                and source_paths.get("manifest") == "manifest.json"
                and isinstance(split_paths, dict)
                and set(split_paths) == set(N2_SPLITS)
                and len(split_paths) == len(N2_SPLITS)
                and all(
                    _is_string(value) and not Path(value).is_absolute()
                    and ".." not in Path(value).parts
                    for value in source_names
                )
                and len(set(source_names)) == 6
            )
            hashes_valid = (
                isinstance(hashes, dict) and len(hashes) == 6
                and paths_valid and set(hashes) == set(source_names)
                and all(_is_string(name) and isinstance(value, str)
                        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
                        for name, value in hashes.items())
            )
            hashes_after = evidence.get("source_hashes_after")
            hashes_unchanged = hashes_valid and _strict_equal(hashes, hashes_after)
            result["checks"]["source_hashes_unchanged"] = bool(hashes_unchanged)
            if not hashes_unchanged:
                _add(violations, "n2_source_hashes", item_path, "manifest/source_paths and five split hashes must be present, valid and unchanged")
        evidence_violations = []
        _scan_authorization(evidence, f"{item_path}.report", evidence_violations)
        for violation in evidence_violations:
            _add(violations, "evidence_authorization_flag", violation["path"], violation["message"])
        results.append(result)
    return results, entry_by_id


def _validate_reviews(protocol, violations):
    reviews = protocol.get("review_requirements")
    if not isinstance(reviews, list):
        _add(violations, "missing_or_malformed_field", "$.review_requirements", "must be a list")
        return []
    seen = set()
    declarations = {}
    for index, review in enumerate(reviews):
        path = f"$.review_requirements[{index}]"
        if not isinstance(review, dict):
            _add(violations, "malformed_review_declaration", path, "must be an object")
            continue
        review_id = _require_string(review, "review_id", violations, path)
        status = _require_string(review, "status", violations, path)
        if status is not None and status not in REVIEW_STATUSES:
            _add(violations, "review_status", f"{path}.status", f"must be one of {sorted(REVIEW_STATUSES)}")
        if type(review.get("required")) is not bool or review.get("required") is not True:
            _add(violations, "review_required", f"{path}.required", "must be true")
        if review_id is None:
            continue
        if review_id in seen:
            _add(violations, "duplicate_review", f"{path}.review_id", f"duplicate review id {review_id!r}")
        seen.add(review_id)
        declarations[review_id] = review
    missing = [item for item in REQUIRED_REVIEW_IDS if item not in seen]
    unknown = sorted(seen - set(REQUIRED_REVIEW_IDS))
    if missing:
        _add(violations, "missing_review_declaration", "$.review_requirements", f"missing: {', '.join(missing)}")
    if unknown:
        _add(violations, "unknown_review_declaration", "$.review_requirements", f"unknown: {', '.join(unknown)}")
    if len(reviews) != len(REQUIRED_REVIEW_IDS) or len(seen) != len(REQUIRED_REVIEW_IDS):
        _add(violations, "review_count", "$.review_requirements", "must contain exactly the four frozen review declarations")
    results = []
    for review_id in REQUIRED_REVIEW_IDS:
        review = declarations.get(review_id)
        if review is None:
            results.append({"review_id": review_id, "status": "missing_declaration", "approved": False})
            continue
        status = review.get("status")
        approved = status == "approved"
        if not approved:
            _add(violations, "review_pending", f"$.review_requirements[{review_id}].status", "required independent review is not approved")
        results.append({"review_id": review_id, "status": status, "approved": approved})
    return results


def _report(protocol, source, root, violations, evidence_results, review_results):
    unique_reasons = []
    for violation in violations:
        if violation["code"] not in unique_reasons:
            unique_reasons.append(violation["code"])
    contract_valid = not violations
    ready = contract_valid and all(item["approved"] for item in review_results) and all(
        item["present"] and item["status"] is not None for item in evidence_results
    )
    return {
        "report_version": REPORT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "source": _stable_source(source, root),
        "workspace_root": ".",
        "contract_valid": contract_valid,
        "valid": contract_valid,
        "ready": ready,
        "status": "n3_s_readiness_passed_not_authorized" if ready else "n3_s_readiness_blocked",
        "violations": violations,
        "block_reasons": unique_reasons,
        "evidence": evidence_results,
        "reviews": review_results,
        "review_declarations_only": True,
        "authorizes_execution": False,
        "measurement_authorized": False,
        "real_n3_measurement_authorized": False,
        "training_authorized": False,
        "deployment_authorized": False,
        "production_pruning_authorized": False,
        "promotion_authorized": False,
        "model_loaded": False,
        "network_model_calls": 0,
        "measurement_performed": False,
        "training_performed": False,
        "deployment_performed": False,
    }


def validate_protocol(protocol, *, source="<memory>", root=None):
    violations = []
    root_path = Path(root or ".").resolve()
    if not isinstance(protocol, dict):
        _add(violations, "malformed_root", "$", "protocol root must be an object")
        return _report(protocol, source, root_path, violations, [], [])
    if protocol.get("schema_version") != SCHEMA_VERSION:
        _add(violations, "schema_version", "$.schema_version", f"must equal {SCHEMA_VERSION}")
    for key in ("protocol_id", "status", "purpose", "mode", "decision_rule"):
        _require_string(protocol, key, violations, "$")
    if protocol.get("status") != "frozen_readiness_contract":
        _add(violations, "contract_status", "$.status", "must equal frozen_readiness_contract")
    if protocol.get("mode") != "read_only_preflight":
        _add(violations, "contract_mode", "$.mode", "must equal read_only_preflight")
    if protocol.get("workspace_root") != ".":
        _add(violations, "workspace_root", "$.workspace_root", "must be the current workspace root marker '.'")
    limitations = protocol.get("limitations")
    if not isinstance(limitations, list) or not limitations or not all(_is_string(item) for item in limitations):
        _add(violations, "limitations", "$.limitations", "must be a non-empty string list")
    boundaries = protocol.get("boundaries")
    if not isinstance(boundaries, dict):
        _add(violations, "missing_or_malformed_field", "$.boundaries", "must be an object")
    else:
        for key, expected in BOUNDARY_EXPECTATIONS.items():
            _require_bool(boundaries, key, expected, violations, "$.boundaries")
    _scan_authorization(protocol, "$", violations)
    evidence_results, _ = _validate_evidence_contract(protocol, root_path, violations)
    review_results = _validate_reviews(protocol, violations)
    return _report(protocol, source, root_path, violations, evidence_results, review_results)


def load_protocol(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return _load_json(stream.read())


def validate_file(path, *, root=None):
    path = Path(path)
    root_path = Path(root).resolve() if root is not None else path.resolve().parent.parent
    try:
        protocol = load_protocol(path)
    except (OSError, ValueError) as error:
        return _report({}, path, root_path, [{"code": "protocol_unreadable", "path": str(path), "message": str(error)}], [], [])
    return validate_protocol(protocol, source=path, root=root_path)


def _render(report):
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = validate_file(args.protocol, root=args.root)
    try:
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(_render(report))
    except OSError as error:
        report = _report(
            {},
            args.protocol,
            args.root or args.protocol.resolve().parent.parent,
            [{"code": "report_output", "path": str(args.output), "message": str(error)}],
            [],
            [],
        )
    sys.stdout.write(_render(report))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
