#!/usr/bin/env python3
"""Build or validate a content-free real-context holdout manifest.

Input case metadata may reference a local raw request file, but the emitted
manifest contains only hashes, counts, pointers, and labels. No provider calls,
no active filtering, and no raw prompt/response copying are performed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTOCOL = ROOT / "research" / "real_context_holdout_protocol_v1.json"
MANIFEST_SCHEMA = "nanojev-real-context-holdout-manifest-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_METADATA_KEYS = {
    "request", "messages", "input", "content", "prompt", "response",
    "required_strings", "expected_answer", "raw", "text",
}
SECRET_PATTERNS = [
    re.compile(r"apikey_[A-Za-z0-9_-]{8,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{12,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]


class ManifestError(ValueError):
    pass


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def load_protocol(path=DEFAULT_PROTOCOL):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _contains_forbidden_key(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_METADATA_KEYS:
                return key
            found = _contains_forbidden_key(child)
            if found:
                return f"{key}.{found}"
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _contains_forbidden_key(child)
            if found:
                return f"[{index}].{found}"
    return None


def _list_of_strings(value, field, required=False):
    if value is None:
        if required:
            raise ManifestError(f"{field} is required")
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ManifestError(f"{field} must be a list of strings")
    return value


def _hash_list(value, field, required=False):
    if value is None:
        if required:
            raise ManifestError(f"{field} is required")
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) or not HEX64.match(item)
                                          for item in value):
        raise ManifestError(f"{field} must be a list of SHA-256 hex strings")
    return value


def _safe_relative(path_value, field):
    if not isinstance(path_value, str) or not path_value:
        raise ManifestError(f"{field} must be a non-empty relative path")
    path = Path(path_value)
    if path.is_absolute() or ".." in path.parts:
        raise ManifestError(f"{field} must stay inside the holdout directory")
    return path


def _credential_scan(raw):
    try:
        text = raw.decode("utf-8", errors="ignore")
    except AttributeError:
        return "binary_request_bytes"
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None


def _validate_case_metadata(case, protocol, request_path, raw):
    contract = protocol["case_contract"]
    allowed_sources = set(protocol["allowed_sources"])
    wire_formats = set(contract["wire_formats"])
    statuses = set(contract["expected_gate_statuses"])

    for field in ("case_id", "wire_format", "source_type", "source_hash",
                  "request_file", "captured_or_redacted_at", "expected_gate_status",
                  "protected_pointers", "downstream", "family", "language"):
        if field not in case:
            raise ManifestError(f"missing case field: {field}")

    forbidden = _contains_forbidden_key(case)
    if forbidden:
        raise ManifestError(f"raw-text-like metadata key is forbidden: {forbidden}")

    if not isinstance(case["case_id"], str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,63}",
                                                              case["case_id"]):
        raise ManifestError("case_id must be a bounded slug")
    if case["wire_format"] not in wire_formats:
        raise ManifestError("wire_format is not in the protocol")
    if case["source_type"] not in allowed_sources:
        raise ManifestError("source_type is not allowed")
    if not isinstance(case["source_hash"], str) or not HEX64.match(case["source_hash"]):
        raise ManifestError("source_hash must be a SHA-256 hex string")
    if case["expected_gate_status"] not in statuses:
        raise ManifestError("expected_gate_status is not in the protocol")

    protected = _list_of_strings(case["protected_pointers"], "protected_pointers", required=True)
    eligible = _list_of_strings(case.get("eligible_candidate_pointers"),
                                "eligible_candidate_pointers")
    dependencies = _list_of_strings(case.get("dependency_pointers"), "dependency_pointers")
    suggestions = _list_of_strings(case.get("expected_suggestions"), "expected_suggestions")
    if set(protected) & set(eligible):
        raise ManifestError("a pointer cannot be both protected and eligible")

    downstream = case["downstream"]
    if not isinstance(downstream, dict):
        raise ManifestError("downstream must be an object")
    required_pointers = _list_of_strings(downstream.get("required_evidence_pointers"),
                                         "downstream.required_evidence_pointers")
    required_hashes = _hash_list(downstream.get("required_evidence_sha256"),
                                 "downstream.required_evidence_sha256")
    expected_answer_hash = downstream.get("expected_answer_sha256")
    if expected_answer_hash is not None and not HEX64.match(str(expected_answer_hash)):
        raise ManifestError("expected_answer_sha256 must be a SHA-256 hex string")
    if case["expected_gate_status"] == "scored" and not (
            required_pointers or required_hashes or expected_answer_hash):
        raise ManifestError("scored cases require a declared downstream contract")

    labels = case.get("labels")
    if not isinstance(labels, dict):
        raise ManifestError("labels must be an object")
    if not isinstance(labels.get("author"), str) or not labels["author"]:
        raise ManifestError("labels.author is required")
    if labels.get("confidence") not in {"high", "medium", "low"}:
        raise ManifestError("labels.confidence must be high, medium, or low")
    if labels.get("confidence") == "low" and case["expected_gate_status"] == "scored":
        raise ManifestError("low-confidence scored cases must be retained or excluded")

    if not isinstance(case["family"], str) or not case["family"]:
        raise ManifestError("family must be a non-empty string")
    if not isinstance(case["language"], str) or not case["language"]:
        raise ManifestError("language must be a non-empty string")
    tool_linked = bool(case.get("tool_linked", False))
    long_tail = bool(case.get("long_tail", False))
    mixed_part = bool(case.get("mixed_part", False))
    mutability = case.get("tool_result_mutability", "none")
    if mutability not in {"none", "mutable", "nonrepeatable"}:
        raise ManifestError("tool_result_mutability must be none, mutable, or nonrepeatable")

    try:
        json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ManifestError("request_file must be UTF-8 JSON") from error
    secret = _credential_scan(raw)
    if secret is not None:
        raise ManifestError(f"credential-like pattern in request_file: {secret}")

    return {
        "case_id": case["case_id"],
        "family": case["family"],
        "language": case["language"],
        "wire_format": case["wire_format"],
        "source_type": case["source_type"],
        "source_hash": case["source_hash"],
        "request_file": str(request_path),
        "request_sha256": sha256_bytes(raw),
        "body_bytes": len(raw),
        "captured_or_redacted_at": case["captured_or_redacted_at"],
        "expected_gate_status": case["expected_gate_status"],
        "protected_pointers": protected,
        "eligible_candidate_pointers": eligible,
        "dependency_pointers": dependencies,
        "expected_suggestions": suggestions,
        "tool_linked": tool_linked,
        "long_tail": long_tail,
        "mixed_part": mixed_part,
        "tool_result_mutability": mutability,
        "downstream": {
            "required_evidence_pointers": required_pointers,
            "required_evidence_sha256": required_hashes,
            "expected_answer_sha256": expected_answer_hash,
        },
        "labels": {
            "author": labels["author"],
            "confidence": labels["confidence"],
            "notes_sha256": labels.get("notes_sha256"),
        },
    }


def coverage_report(cases, protocol):
    coverage = protocol["coverage_requirements"]
    families = {case["family"] for case in cases}
    languages = {case["language"] for case in cases}
    report = {
        "case_count": len(cases),
        "distinct_families": len(families),
        "bypass_cases": sum(case["expected_gate_status"] == "bypass" for case in cases),
        "tool_linked_cases": sum(case["tool_linked"] for case in cases),
        "multilingual_cases": sum(case["language"].lower() not in {"en", "english"}
                                  for case in cases),
        "protected_only_cases": sum(case["expected_gate_status"] == "protected_only"
                                    for case in cases),
        "long_tail_cases": sum(case["long_tail"] for case in cases),
        "mixed_part_cases": sum(case["mixed_part"] for case in cases),
        "mutable_tool_result_cases": sum(case["tool_result_mutability"] == "mutable"
                                         for case in cases),
        "nonrepeatable_tool_result_cases": sum(
            case["tool_result_mutability"] == "nonrepeatable" for case in cases),
        "languages": sorted(languages),
    }
    report["ready_for_evaluation"] = all([
        report["case_count"] >= coverage["minimum_cases"],
        report["distinct_families"] >= coverage["minimum_distinct_families"],
        report["bypass_cases"] >= coverage["minimum_bypass_cases"],
        report["tool_linked_cases"] >= coverage["minimum_tool_linked_cases"],
        report["multilingual_cases"] >= coverage["minimum_multilingual_cases"],
        report["protected_only_cases"] >= coverage["minimum_protected_only_cases"],
        report["long_tail_cases"] >= coverage["minimum_long_tail_cases"],
        report["mixed_part_cases"] >= coverage["minimum_mixed_part_cases"],
        report["mutable_tool_result_cases"] >= coverage["minimum_mutable_tool_result_cases"],
        report["nonrepeatable_tool_result_cases"] >=
        coverage["minimum_nonrepeatable_tool_result_cases"],
    ])
    return report


def build_manifest(case_dir, protocol_path=DEFAULT_PROTOCOL, data_root=None):
    case_dir = Path(case_dir)
    protocol = load_protocol(protocol_path)
    data_root = Path(data_root or protocol["collection"]["data_root"])
    cases = []
    seen_ids = set()
    seen_hashes = set()
    for path in sorted(case_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        case = json.loads(path.read_text(encoding="utf-8"))
        request_rel = _safe_relative(case.get("request_file"), "request_file")
        request_path = (path.parent / request_rel).resolve()
        if path.parent.resolve() not in request_path.parents and request_path != path.parent.resolve():
            raise ManifestError("request_file must stay inside the case directory")
        if not request_path.is_file():
            raise ManifestError(f"request_file missing: {request_rel}")
        raw = request_path.read_bytes()
        item = _validate_case_metadata(case, protocol, request_rel, raw)
        if item["case_id"] in seen_ids:
            raise ManifestError(f"duplicate case_id: {item['case_id']}")
        if item["request_sha256"] in seen_hashes:
            raise ManifestError(f"duplicate request_sha256: {item['request_sha256']}")
        seen_ids.add(item["case_id"])
        seen_hashes.add(item["request_sha256"])
        cases.append(item)
    cases.sort(key=lambda item: item["case_id"])
    return {
        "schema_version": MANIFEST_SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(protocol_path),
        "data_root": str(data_root),
        "content_free": True,
        "case_count": len(cases),
        "cases": cases,
        "coverage": coverage_report(cases, protocol),
        "collection": {
            "evaluation_only": True,
            "provider_calls_allowed": False,
            "active_filtering_allowed": False,
        },
    }


def validate_manifest(path, protocol_path=DEFAULT_PROTOCOL):
    protocol = load_protocol(protocol_path)
    failures = []
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return {"ok": False, "failures": [f"manifest unreadable: {error}"]}
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        failures.append("manifest schema_version mismatch")
    if manifest.get("protocol_id") != protocol.get("protocol_id"):
        failures.append("manifest protocol_id mismatch")
    if manifest.get("protocol_sha256") != sha256_file(protocol_path):
        failures.append("manifest protocol_sha256 mismatch")
    if manifest.get("content_free") is not True:
        failures.append("manifest must be content-free")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        failures.append("cases must be a list")
        cases = []
    if manifest.get("case_count") != len(cases):
        failures.append("case_count mismatch")
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            failures.append("case must be an object")
            continue
        forbidden = _contains_forbidden_key(case)
        if forbidden:
            failures.append(f"{case.get('case_id')}: forbidden raw key {forbidden}")
        request_hash = case.get("request_sha256")
        if not isinstance(request_hash, str) or not HEX64.match(request_hash):
            failures.append(f"{case.get('case_id')}: invalid request_sha256")
        elif request_hash in seen:
            failures.append(f"{case.get('case_id')}: duplicate request_sha256")
        else:
            seen.add(request_hash)
        if set(case.get("protected_pointers") or []) & set(
                case.get("eligible_candidate_pointers") or []):
            failures.append(f"{case.get('case_id')}: protected/eligible pointer overlap")
        downstream = case.get("downstream") or {}
        if case.get("expected_gate_status") == "scored" and not (
                downstream.get("required_evidence_pointers") or
                downstream.get("required_evidence_sha256") or
                downstream.get("expected_answer_sha256")):
            failures.append(f"{case.get('case_id')}: scored case lacks downstream contract")
    expected_coverage = coverage_report(cases, protocol)
    if manifest.get("coverage") != expected_coverage:
        failures.append("coverage report mismatch")
    collection = manifest.get("collection") or {}
    if collection.get("evaluation_only") is not True:
        failures.append("collection.evaluation_only must be true")
    if collection.get("provider_calls_allowed") is not False:
        failures.append("collection.provider_calls_allowed must be false")
    if collection.get("active_filtering_allowed") is not False:
        failures.append("collection.active_filtering_allowed must be false")
    return {
        "schema_version": "nanojev-real-context-holdout-manifest-check-v1",
        "manifest_path": str(Path(path).resolve()),
        "manifest_sha256": sha256_file(path),
        "case_count": len(cases),
        "ready_for_evaluation": expected_coverage["ready_for_evaluation"],
        "provider_calls_allowed": False,
        "active_filtering_allowed": False,
        "failures": failures,
        "ok": not failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build a content-free manifest from local case metadata")
    build.add_argument("--case-dir", type=Path, required=True)
    build.add_argument("--data-root", type=Path, default=None)
    build.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    build.add_argument("--output", type=Path, required=True)
    validate = sub.add_parser("validate", help="validate an existing manifest")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    validate.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "build":
        if args.output.exists() and args.output.stat().st_size:
            raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
        manifest = build_manifest(args.case_dir, args.protocol, args.data_root)
        encoded = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(json.dumps({"output": str(args.output), "sha256": sha256_text(encoded),
                          "case_count": manifest["case_count"],
                          "ready_for_evaluation": manifest["coverage"]["ready_for_evaluation"]},
                         indent=2))
        return

    receipt = validate_manifest(args.manifest, args.protocol)
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    if not receipt["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
