#!/usr/bin/env python3
"""Run N4 footprint receipt controls against an in-memory synthetic fixture only.

This is a receipt/schema smoke test, not a quantization or benchmark runner. It does
not load a model, artifact, tokenizer or dataset and never makes a network call.
All physical and quality metrics remain ``null`` so the report cannot be mistaken for
size, speed, cost or model-quality evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import validate_nanojev_v3_n4_footprint_contract_v1 as contract_validator


HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = HERE.parent / "research" / "nanojev_v3_n4_footprint_contract_v1.json"
REPORT_STATUS = "synthetic_footprint_controls_passed_not_measurement_evidence"
MEASUREMENT_FIELDS = (
    "artifact_sha256", "tokenizer_sha256", "protocol_sha256", "parameter_count",
    "weight_bytes", "package_bytes", "peak_memory_bytes", "load_time_ms",
    "cold_latency_ms", "warm_p50_ms", "warm_p95_ms", "warm_p99_ms",
    "quality_metrics", "calibration_metrics", "ood_metrics", "protected_error_count",
    "probability_normalization_error", "deterministic",
)
RAW_OR_PATH_KEYS = frozenset({"state", "instructions", "criteria", "raw_text", "artifact_path", "weight_path", "dataset_path"})


class SyntheticFootprintError(RuntimeError):
    """A contract or synthetic receipt invariant failed closed."""


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_value(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _with_hash(row):
    payload = dict(row)
    payload["receipt_sha256"] = None
    row["receipt_sha256"] = sha256_value(payload)
    return row


def _placeholder_receipt(candidate, receipt_fields):
    row = {field: None for field in receipt_fields}
    row.update({
        "candidate_id": candidate["candidate_id"],
        "representation": candidate["representation"],
        "parent_candidate_id": candidate["parent_candidate_id"],
        "paired_with": candidate["paired_with"],
        "network_model_calls": 0,
        "model_loaded": False,
        "quantization_performed": False,
        "training_performed": False,
        "deployment_authorized": False,
    })
    return _with_hash(row)


def build_receipts(protocol):
    receipt_fields = tuple(protocol["receipt_fields"])
    return [_placeholder_receipt(candidate, receipt_fields) for candidate in protocol["candidate_ladder"]]


def validate_receipts(protocol, receipts):
    required = set(protocol["receipt_fields"])
    expected_ids = tuple(item["candidate_id"] for item in protocol["candidate_ladder"])
    if len(receipts) != len(expected_ids):
        raise SyntheticFootprintError("receipt count does not match candidate ladder")
    seen = []
    for row in receipts:
        if set(row) != required:
            raise SyntheticFootprintError("receipt field set does not exactly match the contract")
        candidate_hash = row["receipt_sha256"]
        payload = dict(row)
        payload["receipt_sha256"] = None
        if candidate_hash != sha256_value(payload):
            raise SyntheticFootprintError(f"receipt hash mismatch for {row['candidate_id']}")
        candidate_id = row["candidate_id"]
        if candidate_id in seen:
            raise SyntheticFootprintError(f"duplicate candidate receipt: {candidate_id}")
        seen.append(candidate_id)
        if row["network_model_calls"] != 0 or row["model_loaded"]:
            raise SyntheticFootprintError("synthetic receipt reports model/network activity")
        if row["quantization_performed"] or row["training_performed"] or row["deployment_authorized"]:
            raise SyntheticFootprintError("synthetic receipt contains execution or authorization")
        if any(row[field] is not None for field in MEASUREMENT_FIELDS):
            raise SyntheticFootprintError("synthetic receipt contains a fabricated measurement")
        if any(key in row for key in RAW_OR_PATH_KEYS):
            raise SyntheticFootprintError("synthetic receipt contains raw input or artifact path content")
    if tuple(seen) != expected_ids:
        raise SyntheticFootprintError("candidate receipt order does not match the frozen ladder")
    return {
        "candidate_receipts": {"passed": True, "rows": len(receipts)},
        "paired_candidate_ids": list(seen),
        "placeholder_metric_fields": list(MEASUREMENT_FIELDS),
        "no_fabricated_measurements": True,
        "no_raw_or_path_fields": True,
    }


def run_protocol(protocol, source="<memory>"):
    validation = contract_validator.validate_protocol(protocol, source=source)
    if not validation["valid"]:
        raise SyntheticFootprintError(json.dumps({"contract_blocked": validation["block_reasons"]}, sort_keys=True))
    receipts = build_receipts(protocol)
    controls = validate_receipts(protocol, receipts)
    report = {
        "report_version": "nanojev-v3-n4-synthetic-report-v1",
        "status": REPORT_STATUS,
        "protocol_schema_version": protocol["schema_version"],
        "protocol_sha256": sha256_value(protocol),
        "source": str(source),
        "synthetic_only": True,
        "measurement_evidence": False,
        "candidate_count": len(protocol["candidate_ladder"]),
        "receipt_count": len(receipts),
        "receipt_fields": list(protocol["receipt_fields"]),
        "control_results": controls,
        "receipts": receipts,
        "network_model_calls": 0,
        "model_loaded": False,
        "artifact_generation_performed": False,
        "quantization_performed": False,
        "training_performed": False,
        "deployment_performed": False,
        "training_authorized": False,
        "deployment_authorized": False,
        "production_pruning_authorized": False,
        "promotion_authorized": False,
        "report_sha256": None,
    }
    report["report_sha256"] = sha256_value({key: value for key, value in report.items() if key != "report_sha256"})
    return report


def load_and_run(path):
    try:
        protocol = contract_validator.load_protocol(path)
    except (OSError, ValueError) as error:
        raise SyntheticFootprintError(f"protocol unreadable: {error}") from error
    return run_protocol(protocol, source=path)


def _render(report):
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        report = load_and_run(args.protocol)
        rendered = _render(report)
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
        sys.stdout.write(rendered)
        return 0
    except (OSError, SyntheticFootprintError, TypeError, ValueError) as error:
        blocked = {
            "report_version": "nanojev-v3-n4-synthetic-report-v1",
            "status": "synthetic_footprint_controls_blocked",
            "error": str(error),
            "synthetic_only": True,
            "measurement_evidence": False,
            "network_model_calls": 0,
            "model_loaded": False,
            "artifact_generation_performed": False,
            "quantization_performed": False,
            "training_performed": False,
            "deployment_performed": False,
            "training_authorized": False,
            "deployment_authorized": False,
            "production_pruning_authorized": False,
            "promotion_authorized": False,
        }
        sys.stdout.write(_render(blocked))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
