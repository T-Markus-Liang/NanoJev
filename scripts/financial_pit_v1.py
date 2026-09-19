#!/usr/bin/env python3
"""Point-in-time binary-event records and purged chronological folds, with no trading."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from benchmark_nanojev_v2 import canonical_json, file_identity, sha256_bytes
from predict_toy_decisions import read_json, reject_nonfinite, unique_object


SCHEMA = "nanojev-financial-pit-v1"
PHASES = ("train", "dev", "calibration", "test")


def timestamp(value, field):
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} requires nonnegative UTC Unix nanoseconds, not floats or booleans")
    return value


def nonempty(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} requires a nonempty string")


def validate_record(row):
    required = {"schema_version", "id", "asset_id", "venue", "decision_ns", "universe_available_ns", "features", "label"}
    if not isinstance(row, dict) or set(row) != required or row["schema_version"] != SCHEMA:
        raise ValueError("record must match the financial PIT v1 contract exactly")
    for field in ("id", "asset_id", "venue"):
        nonempty(row[field], field)
    decision = timestamp(row["decision_ns"], "decision_ns")
    if timestamp(row["universe_available_ns"], "universe_available_ns") > decision:
        raise ValueError("universe membership was not available at decision time")
    features = row["features"]
    if not isinstance(features, dict) or not features:
        raise ValueError("nonempty point-in-time features required")
    for name, feature in features.items():
        nonempty(name, "feature name")
        if not isinstance(feature, dict) or set(feature) != {"value", "event_ns", "available_ns", "fit_cutoff_ns", "source_id", "version"}:
            raise ValueError("feature needs value, event/availability/fit timestamps, source ID and version")
        value = feature["value"]
        if type(value) not in {int, float, bool} or (type(value) is not bool and not math.isfinite(value)):
            raise ValueError("feature values must be finite numeric or boolean values; missing values need an explicit upstream policy")
        event = timestamp(feature["event_ns"], "feature event_ns")
        available = timestamp(feature["available_ns"], "feature available_ns")
        fit = timestamp(feature["fit_cutoff_ns"], "feature fit_cutoff_ns")
        if not event <= available <= decision or fit > available:
            raise ValueError("feature contains future or not-yet-available information")
        nonempty(feature["source_id"], "source_id")
        nonempty(feature["version"], "version")
    label = row["label"]
    if not isinstance(label, dict) or set(label) != {"event", "definition_sha256", "end_ns", "available_ns", "outcome"}:
        raise ValueError("label needs explicit event contract, end/availability timestamps and outcome")
    nonempty(label["event"], "label event")
    definition = label["definition_sha256"]
    if not isinstance(definition, str) or len(definition) != 64 or any(c not in "0123456789abcdef" for c in definition):
        raise ValueError("label definition needs a lowercase SHA-256 identity")
    if not decision < timestamp(label["end_ns"], "label end_ns") <= timestamp(label["available_ns"], "label available_ns"):
        raise ValueError("label horizon/availability must be later than the decision")
    if type(label["outcome"]) is not bool:
        raise ValueError("v1 labels are realized binary events, never probability argmax labels")
    return row


def validate_dataset(rows):
    if not rows:
        raise ValueError("financial dataset is empty")
    seen, snapshots, definitions, feature_names = set(), set(), set(), None
    for row in rows:
        validate_record(row)
        if row["id"] in seen:
            raise ValueError("duplicate record ID")
        seen.add(row["id"])
        key = (row["asset_id"], row["venue"], row["decision_ns"], row["label"]["event"])
        if key in snapshots:
            raise ValueError("duplicate decision event under a different record ID")
        snapshots.add(key)
        definitions.add((row["label"]["event"], row["label"]["definition_sha256"]))
        names = set(row["features"])
        if feature_names is not None and names != feature_names:
            raise ValueError("feature schema must be frozen across records")
        feature_names = names
    if len(definitions) != 1:
        raise ValueError("a v1 cohort must have one frozen event definition")


def model_input(row):
    validate_record(row)
    # Future labels, split names, example IDs and audit metadata cannot enter the predictor.
    return {"asset_id": row["asset_id"], "venue": row["venue"], "decision_ns": row["decision_ns"],
            "features": {name: feature["value"] for name, feature in sorted(row["features"].items())}}


def validate_fold(fold, embargo_ns, asof_ns):
    timestamp(embargo_ns, "embargo_ns"); timestamp(asof_ns, "asof_ns")
    if not isinstance(fold, dict) or set(fold) != set(PHASES):
        raise ValueError("fold requires exactly train/dev/calibration/test intervals")
    last_end = None
    for phase in PHASES:
        window = fold[phase]
        if not isinstance(window, list) or len(window) != 2:
            raise ValueError("fold windows are [start_ns,end_ns) pairs")
        start, end = [timestamp(value, "window boundary") for value in window]
        if start >= end or (last_end is not None and start < last_end):
            raise ValueError("fold windows must be nonempty, chronological and disjoint")
        last_end = end
    if asof_ns < fold["test"][1]:
        raise ValueError("evaluation as-of must reach the end of the test window")


def partition(rows, fold, embargo_ns, asof_ns):
    validate_dataset(rows)
    validate_fold(fold, embargo_ns, asof_ns)
    retained, excluded = {phase: [] for phase in PHASES}, []
    for row in sorted(rows, key=lambda r: (r["decision_ns"], r["id"])):
        phase = next((phase for phase in PHASES if fold[phase][0] <= row["decision_ns"] < fold[phase][1]), None)
        if phase is None:
            excluded.append({"id": row["id"], "reason": "outside_windows"})
            continue
        label = row["label"]
        if phase == "test":
            reason = "unmatured_test_label" if label["available_ns"] > asof_ns else None
        else:
            next_phase = PHASES[PHASES.index(phase) + 1]
            if label["end_ns"] + embargo_ns >= fold[next_phase][0]:
                reason = "purged_overlap_or_embargo"
            elif label["available_ns"] > fold[phase][1]:
                reason = "label_unavailable_at_fit_cutoff"
            else:
                reason = None
        if reason:
            excluded.append({"id": row["id"], "phase": phase, "reason": reason})
        else:
            retained[phase].append(row["id"])
    return {"schema_version": "nanojev-financial-fold-audit-v1", "windows": fold,
            "embargo_ns": embargo_ns, "asof_ns": asof_ns, "retained": retained,
            "counts": {phase: len(ids) for phase, ids in retained.items()},
            "excluded": excluded, "exclusion_counts": dict(Counter(item["reason"] for item in excluded)),
            "all_phases_nonempty": all(retained.values()),
            "dataset_sha256": sha256_bytes(canonical_json(rows).encode()),
            "scope": "Declared timestamp and label-isolation checks only; not independent market-source, survivorship, feature-lineage or trading-profit validation."}


def walk_forward(rows, folds, embargo_ns, asof_ns):
    if not isinstance(folds, list) or not folds:
        raise ValueError("at least one frozen fold is required")
    previous_test_end = None
    reports = []
    for fold in folds:
        validate_fold(fold, embargo_ns, asof_ns)
        if previous_test_end is not None and fold["test"][0] < previous_test_end:
            raise ValueError("walk-forward test windows must be chronological and nonoverlapping")
        previous_test_end = fold["test"][1]
        reports.append(partition(rows, fold, embargo_ns, asof_ns))
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_nonfinite)
            for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    protocol = read_json(args.protocol)
    if set(protocol) != {"folds", "embargo_ns", "asof_ns"}:
        raise ValueError("protocol requires folds, embargo_ns and asof_ns")
    report = {"folds": walk_forward(rows, **protocol), "input": file_identity(args.input),
              "protocol": file_identity(args.protocol), "validator": file_identity(__file__)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not all(fold["all_phases_nonempty"] for fold in report["folds"]):
        raise SystemExit("Audit saved, but at least one phase is empty; not training-ready")
    print(json.dumps({"output": str(args.output), "folds": len(report["folds"])}))
