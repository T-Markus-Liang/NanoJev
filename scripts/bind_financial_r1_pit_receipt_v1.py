#!/usr/bin/env python3
"""Bind a PIT validator report to the full owner-decided R1 protocol.

The underlying ``financial_pit_v1.py`` contract is intentionally unchanged.  This
wrapper only verifies that its report used the projected core contained byte-for-byte
in the full protocol, then emits a content-addressed, fail-closed receipt.  It never
trains, fits, trades, downloads data or edits the input report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "nanojev-financial-r1-pit-bound-receipt-v1"


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def bind(protocol_path: Path, core_path: Path, report_path: Path) -> dict[str, Any]:
    protocol = read(protocol_path)
    core = read(core_path)
    report = read(report_path)
    errors: list[str] = []
    if protocol.get("schema_version") != "nanojev-financial-experiment-protocol-v2":
        errors.append("full protocol is not financial experiment protocol v2")
    if protocol.get("status") != "frozen_r1_rebase_2026-09-20":
        errors.append("full protocol is not the frozen R1 rebase")
    if protocol.get("pit_validator_core") != core:
        errors.append("projected PIT core differs from full protocol.pit_validator_core")
    if not protocol.get("approval_state", {}).get("approved"):
        errors.append("full protocol is not owner-approved")
    expected_core_hash = sha256(core_path)
    expected_protocol_hash = sha256(protocol_path)
    expected_report_hash = sha256(report_path)
    report_protocol = report.get("protocol", {})
    if report_protocol.get("sha256") != expected_core_hash:
        errors.append("validator report is not bound to the supplied core")
    if report.get("input", {}).get("sha256") is None:
        errors.append("validator report has no input hash")
    if report.get("validator", {}).get("sha256") is None:
        errors.append("validator report has no validator hash")
    folds = report.get("folds")
    if not isinstance(folds, list) or not folds or not all(fold.get("all_phases_nonempty") is True for fold in folds):
        errors.append("every PIT fold must retain all four nonempty phases")
    # A projected fold report can be structurally valid while still auditing the
    # older nine-feature pilot.  Bind the actual cohort schema and label identity
    # to R1 before presenting a receipt as anything other than a failed preflight.
    input_path = Path(str(report.get("input", {}).get("path", "")))
    expected_features = {
        item["name"] for item in protocol.get("feature_allowlist", {}).get("features", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    expected_event = protocol.get("event_definition", {}).get("event_name")
    expected_definition = (
        protocol.get("label_predicate", {})
        .get("definition_hash_convention", {})
        .get("computed_value_status")
    )
    frozen_object = protocol.get("label_predicate", {}).get("definition_hash_convention", {}).get("frozen_definition_object")
    if isinstance(frozen_object, dict):
        expected_definition = hashlib.sha256(canonical(frozen_object)).hexdigest()
    if not input_path.is_file():
        errors.append("validator input path is not readable")
    else:
        try:
            with input_path.open(encoding="utf-8") as stream:
                first = json.loads(next(line for line in stream if line.strip()))
            actual_features = set(first.get("features", {}))
            if actual_features != expected_features:
                errors.append("validator input feature schema does not match R1 closed allowlist")
            label = first.get("label", {})
            if label.get("event") != expected_event or label.get("definition_sha256") != expected_definition:
                errors.append("validator input label identity does not match R1 event definition")
        except (OSError, StopIteration, json.JSONDecodeError, TypeError):
            errors.append("validator input cannot be inspected for R1 schema binding")
    receipt = {
        "schema_version": SCHEMA,
        "status": "preflight_passed_not_training_authorized" if not errors else "binding_failed",
        "full_protocol": {"path": str(protocol_path.resolve()), "sha256": expected_protocol_hash},
        "projected_pit_core": {"path": str(core_path.resolve()), "sha256": expected_core_hash},
        "validator_report": {"path": str(report_path.resolve()), "sha256": expected_report_hash},
        "dataset": report.get("input"),
        "validator": report.get("validator"),
        "folds": folds if isinstance(folds, list) else [],
        "errors": errors,
        "training_authorized": False,
        "measurement_authorized": False,
        "model_training_performed": False,
        "order_submission_authorized": False,
        "live_trading_authorized": False,
        "scope": "PIT timestamp/label isolation and fold non-emptiness only; not market-source authenticity, model quality, profitability or deployment readiness",
    }
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = bind(args.protocol, args.core, args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output), "errors": result["errors"]}, ensure_ascii=False))
    return 0 if not result["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
