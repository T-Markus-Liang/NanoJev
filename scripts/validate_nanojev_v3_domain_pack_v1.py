#!/usr/bin/env python3
"""NanoJev V3 N2 read-only domain-pack preflight validator (contract v1).

This is a read-only, standard-library-only gate over a versioned JSON manifest plus
five split JSONL inputs (train / dev / calibration / test / ood). It records input
hashes before and after validation, fails closed on schema or isolation defects, and
never writes a merged row. A clean run still reports ``training_authorized=false``;
no optimizer is started and no checkpoint is touched.

Exit codes:
    0  clean preflight (status ``preflight_passed_not_training_authorized``)
    2  blocked report (violations) or hard read/write error

The only file this tool may create is its optional exclusive report path; an existing
report is never overwritten and no path is written under the pack root.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

SCHEMA = "nanojev-v3-domain-pack-v1"
REPORT_SCHEMA = "nanojev-v3-domain-pack-preflight-v1"
REQUIRED_SPLITS = ("train", "dev", "calibration", "test", "ood")
HELDOUT_SPLITS = ("test", "ood")


class ContractError(Exception):
    """Raised for malformed inputs that must fail closed."""


def canonical(value):
    """Deterministic JSON: dict key order normalized, list order preserved."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(root, paths):
    """Map each path to its sha256 (or None when absent), rooted at ``root``."""
    result = {}
    for path in paths:
        relative = str(path.relative_to(root))
        result[relative] = file_hash(path) if path.is_file() else None
    return result


def visible_input(record):
    """Only the canonical model-visible input: state + question type/instructions/criteria.

    Record id, split, source/lineage ids, provenance and gold/target are excluded
    because they never enter the model input. Dict key order is normalized; an
    ordinal Score criteria list keeps its order.
    """
    question = record.get("question")
    if not isinstance(question, dict):
        question = {}
    return {
        "state": record.get("state"),
        "type": question.get("type"),
        "instructions": question.get("instructions"),
        "criteria": question.get("criteria", {}),
    }


def provenance_is_evaluation_derived(alias):
    """Recognize hyphen/underscore/space variants of evaluation-derived sources."""
    normalized = re.sub(r"[-_\s]+", " ", str(alias).lower())
    return any(marker in normalized for marker in (
        "abstention survey", "workflow challenge", "workflow v2 evaluation",
        "context relevance test", "context relevance ood"))


def load_json_object(path):
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, ValueError) as error:
        raise ContractError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(data, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return data


def load_jsonl(path):
    """Return (records, errors); malformed lines are recorded, not silently dropped."""
    records, errors = [], []
    try:
        stream = Path(path).open("r", encoding="utf-8")
    except OSError as error:
        raise ContractError(f"cannot read {path}: {error}") from error
    with stream:
        for number, line in enumerate(stream, 1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except ValueError as error:
                errors.append(f"line {number}: invalid JSON ({error})")
                continue
            if not isinstance(value, dict):
                errors.append(f"line {number}: record must be a JSON object")
                continue
            records.append(value)
    return records, errors


def record_schema_violations(record):
    """Return schema violations that are independent of split isolation."""
    violations = []
    if "state" not in record or record.get("state") is None:
        violations.append("missing_visible_input")
    question = record.get("question")
    if not isinstance(question, dict):
        violations.append("missing_visible_input")
    else:
        for field in ("type", "instructions", "criteria"):
            if field not in question:
                violations.append("malformed_question")
        if "type" in question and (not isinstance(question["type"], str)
                                    or not question["type"].strip()):
            violations.append("malformed_question")
        if "instructions" in question and (not isinstance(question["instructions"], str)
                                            or not question["instructions"].strip()):
            violations.append("malformed_question")
    if "target" not in record:
        violations.append("missing_target")
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        violations.append("missing_provenance")
    else:
        alias = provenance.get("source_alias")
        derived = provenance.get("derived_from_evaluation_corpus")
        if not isinstance(alias, str) or not alias.strip() or not isinstance(derived, bool):
            violations.append("malformed_provenance")
    return violations


def _base_report(root):
    return {
        "schema_version": REPORT_SCHEMA,
        # Do not embed a checkout-specific absolute path in a report that may be
        # pinned by hash.  A valid manifest replaces this with pack_id@version;
        # malformed/missing manifests remain explicitly unresolved.
        "pack": "unresolved",
        "source_paths": {"manifest": "manifest.json", "splits": {}},
        "source_hashes": {},
        "source_hashes_after": {},
        "manifest_errors": [],
        "heldout": None,
        "protected_cases": {"declared": 0, "resolved": 0},
        "counts_by_split": {},
        "record_count": 0,
        "violations": [],
        "block_reasons": [],
        "training_authorized": False,
        "training_performed": False,
        "merged_rows_written": 0,
        "input_files_changed": False,
        "status": "blocked",
    }


def _finalize(report):
    report["block_reasons"] = sorted({entry["code"] for entry in report["violations"]})
    report["status"] = ("blocked" if report["block_reasons"]
                        else "preflight_passed_not_training_authorized")
    report["training_authorized"] = False
    report["training_performed"] = False
    report["merged_rows_written"] = 0
    return report


def validate_pack(pack_root):
    root = Path(pack_root)
    if not root.is_dir():
        report = _base_report(root)
        report["violations"].append({"code": "pack_root_missing", "detail": str(root)})
        return _finalize(report)
    root = root.resolve()
    report = _base_report(root)
    violations = report["violations"]

    def add(code, detail):
        violations.append({"code": code, "detail": detail})

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        add("manifest_missing", str(manifest_path))
        return _finalize(report)
    if not manifest_path.resolve().is_relative_to(root):
        add("manifest_path_outside_pack", str(manifest_path))
        return _finalize(report)

    try:
        manifest = load_json_object(manifest_path)
    except ContractError as error:
        report["manifest_errors"].append(str(error))
        add("manifest_malformed", str(error))
        report["source_hashes"] = snapshot(root, [manifest_path])
        report["source_hashes_after"] = dict(report["source_hashes"])
        return _finalize(report)

    pack_id = manifest.get("pack_id")
    pack_version = manifest.get("pack_version")
    if (isinstance(pack_id, str) and pack_id.strip()
            and isinstance(pack_version, str) and pack_version.strip()):
        report["pack"] = f"{pack_id}@{pack_version}"

    if manifest.get("schema_version") != SCHEMA:
        add("schema_version_mismatch",
            f"expected {SCHEMA!r}, got {manifest.get('schema_version')!r}")
    for field in ("pack_id", "pack_version"):
        value = manifest.get(field)
        if not isinstance(value, str) or not value.strip():
            add("missing_manifest_field", field)

    heldout = manifest.get("heldout")
    if (not isinstance(heldout, dict)
            or not str(heldout.get("identity", "")).strip()
            or not str(heldout.get("description", "")).strip()):
        add("missing_heldout_identity", "manifest.heldout.identity/description required")
    else:
        report["heldout"] = {
            "identity": heldout["identity"],
            "description": heldout["description"],
        }

    declared_splits = manifest.get("splits")
    if not isinstance(declared_splits, dict):
        add("missing_manifest_field", "splits")
        declared_splits = {}

    split_files = {}
    for split in REQUIRED_SPLITS:
        relative = declared_splits.get(split)
        if not isinstance(relative, str) or not relative.strip():
            add("missing_split", split)
            continue
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root):
            add("split_path_outside_pack", f"{split}: {relative}")
            continue
        split_files[split] = candidate

    report["source_paths"] = {
        "manifest": str(manifest_path.relative_to(root)),
        "splits": {
            split: str(path.relative_to(root))
            for split, path in split_files.items()
        },
    }

    paths = [manifest_path] + list(split_files.values())
    hashes_before = snapshot(root, paths)
    report["source_hashes"] = dict(hashes_before)

    all_records = []
    records_by_id = {}
    id_splits = {}
    for split in REQUIRED_SPLITS:
        path = split_files.get(split)
        if path is None:
            continue
        if not path.is_file():
            add("split_file_missing", split)
            continue
        try:
            records, errors = load_jsonl(path)
        except ContractError as error:
            add("split_malformed", f"{split}: {error}")
            continue
        for error in errors:
            add("malformed_record", f"{split}: {error}")
        report["counts_by_split"][split] = len(records)
        if not records:
            add("empty_split", split)
        for record in records:
            all_records.append((split, record))

    source_splits = defaultdict(set)
    lineage_splits = defaultdict(set)
    input_splits = defaultdict(set)
    input_refs = defaultdict(list)

    for split, record in all_records:
        rid = record.get("id")
        if not isinstance(rid, str) or not rid.strip():
            add("missing_record_id", f"{split}: record without id")
        else:
            if rid in id_splits:
                add("duplicate_record_id", rid)
            else:
                id_splits[rid] = split
            records_by_id[rid] = record

        declared_split = record.get("split")
        if declared_split is not None and declared_split != split:
            add("record_split_mismatch", f"{rid}: {declared_split!r} != {split!r}")

        source = record.get("source_group_id")
        if isinstance(source, str) and source.strip():
            source_splits[source].add(split)
        else:
            add("missing_source_group_id", str(rid))

        lineage = record.get("lineage_id")
        if isinstance(lineage, str) and lineage.strip():
            lineage_splits[lineage].add(split)
        else:
            add("missing_lineage_id", str(rid))

        for code in record_schema_violations(record):
            add(code, str(rid))

        provenance = record.get("provenance")
        if not isinstance(provenance, dict):
            add("missing_provenance", str(rid))
        else:
            alias = provenance.get("source_alias", "")
            if (provenance.get("derived_from_evaluation_corpus") is True
                    or provenance_is_evaluation_derived(alias)):
                add("evaluation_derived_provenance", f"{rid}:{alias}")

        fingerprint = digest(visible_input(record))
        input_splits[fingerprint].add(split)
        input_refs[fingerprint].append({"id": rid, "split": split})

    report["record_count"] = len(all_records)

    for source, splits in sorted(source_splits.items()):
        if len(splits) > 1:
            add("source_group_id_crosses_splits", f"{source}: {sorted(splits)}")
    for lineage, splits in sorted(lineage_splits.items()):
        if len(splits) > 1:
            add("lineage_id_crosses_splits", f"{lineage}: {sorted(splits)}")
    for fingerprint, splits in sorted(input_splits.items()):
        if len(splits) > 1:
            add("canonical_input_crosses_splits", {
                "input_sha256": fingerprint,
                "splits": sorted(splits),
                "records": sorted(ref["id"] for ref in input_refs[fingerprint]),
            })

    declared_protected = manifest.get("protected_cases")
    if not isinstance(declared_protected, list) or not declared_protected:
        add("missing_protected_case_declarations",
            "manifest.protected_cases must be a non-empty list")
    else:
        resolved = 0
        for entry in declared_protected:
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) \
                    or not entry["id"].strip():
                add("missing_protected_case_declarations", f"invalid declaration: {entry}")
                continue
            pid = entry["id"]
            declared_split = entry.get("split")
            reason = entry.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                add("malformed_protected_case_declaration", pid)
            if declared_split not in HELDOUT_SPLITS:
                add("protected_case_declared_for_wrong_split", f"{pid}:{declared_split}")
                continue
            if id_splits.get(pid) != declared_split:
                add("protected_case_absent_from_test_or_ood",
                    f"{pid}: declared {declared_split}, found {id_splits.get(pid)}")
            else:
                resolved += 1
        report["protected_cases"] = {
            "declared": len(declared_protected),
            "resolved": resolved,
        }

    hashes_after = snapshot(root, paths)
    report["source_hashes_after"] = dict(hashes_after)
    report["input_files_changed"] = hashes_after != hashes_before
    if report["input_files_changed"]:
        add("input_changed_during_validation",
            "pack bytes differ before/after the read-only preflight")

    return _finalize(report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True,
                        help="domain-pack root containing manifest.json and split JSONL")
    parser.add_argument("--output", type=Path, default=None,
                        help="exclusive report path; never overwritten")
    args = parser.parse_args(argv)

    if args.output is not None and args.pack.is_dir():
        if args.output.resolve().is_relative_to(args.pack.resolve()):
            parser.error("report output cannot be written inside the read-only pack")

    try:
        report = validate_pack(args.pack)
        serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
            print(json.dumps({"status": report["status"],
                              "output": str(args.output),
                              "block_reasons": report["block_reasons"],
                              "training_authorized": False}))
        else:
            sys.stdout.write(serialized)
        return 2 if report["block_reasons"] else 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "error", "error": str(error),
                          "training_authorized": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
