#!/usr/bin/env python3
"""Fail-closed validator for the T11 baseline protocol (no model fitting)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import financial_baselines_v1 as baselines  # noqa: E402


SCHEMA = "nanojev-financial-baselines-protocol-v1"
EXPECTED_PROTOCOL_SHA256 = "55157d6875e2578071f4f06559236bd04ec6ac481ce1af77a42557ecb2539fa4"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity(path):
    data = Path(path).read_bytes()
    return {"path": str(Path(path).resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def verify_ref(path, expected):
    actual = identity(path)
    if actual["sha256"] != expected:
        raise ValueError(f"hash mismatch: {path}")
    return actual


def under_root(path):
    path = Path(path).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError:
        raise ValueError(f"artifact path escapes project root: {path}") from None
    return path


def validate(protocol_path: Path, receipt_path: Path):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if identity(protocol_path)["sha256"] != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("T11 protocol hash is not the pinned review artifact")
    if protocol.get("schema_version") != SCHEMA:
        raise ValueError("wrong T11 protocol schema")
    if protocol.get("status") != "frozen_preflight_not_fit_authorized":
        raise ValueError("protocol is not the frozen preflight version")
    upstream = protocol["upstream_r1"]
    for key, root_key in (("protocol_path", "protocol_sha256"), ("bound_receipt_path", "bound_receipt_sha256")):
        verify_ref(under_root(ROOT / upstream[key]), upstream[root_key])
    receipt = json.loads((ROOT / upstream["bound_receipt_path"]).read_text(encoding="utf-8"))
    if identity(ROOT / upstream["bound_receipt_path"])["sha256"] != upstream["bound_receipt_sha256"]:
        raise ValueError("bound receipt hash mismatch")
    if receipt.get("status") != "preflight_passed_not_training_authorized" or receipt.get("errors") != []:
        raise ValueError("R1 receipt is not a clean preflight")
    if receipt.get("training_authorized") or receipt.get("measurement_authorized"):
        raise ValueError("upstream receipt unexpectedly authorizes work")
    if receipt.get("full_protocol", {}).get("sha256") != upstream["protocol_sha256"]:
        raise ValueError("receipt full protocol hash differs from pinned upstream protocol")
    if receipt.get("dataset", {}).get("sha256") != upstream["dataset_sha256"]:
        raise ValueError("dataset hash is not bound by R1 receipt")
    if receipt.get("validator", {}).get("sha256") != upstream["validator_sha256"]:
        raise ValueError("validator hash is not bound by R1 receipt")
    if receipt.get("projected_pit_core", {}).get("sha256") != upstream["core_sha256"]:
        raise ValueError("core hash is not bound by R1 receipt")
    for ref_name in ("full_protocol", "projected_pit_core", "validator_report", "dataset", "validator"):
        under_root(receipt[ref_name]["path"])
    protocol_r1, rows, folds = baselines.load_bound_cohort(ROOT / upstream["bound_receipt_path"])
    if protocol_r1["schema_version"] != "nanojev-financial-experiment-protocol-v2":
        raise ValueError("wrong upstream protocol")
    if len(rows) != protocol["cohort"]["records"]:
        raise ValueError("cohort count mismatch")
    if sorted(rows[0]["features"]) != protocol["cohort"]["features"]:
        raise ValueError("feature order mismatch")
    if protocol["cohort"]["features"] != sorted(protocol["cohort"]["features"]):
        raise ValueError("declared feature order is not canonical sorted order")
    if any(row["label"]["event"] != protocol["cohort"]["label_event"] for row in rows):
        raise ValueError("label event mismatch")
    groups = baselines.instrument_groups(rows)
    declared = protocol["instrument_holdout"]["groups"]
    if groups != declared:
        raise ValueError("instrument groups differ from declared hash partition")
    if hashlib.sha256(canonical(groups)).hexdigest() != protocol["instrument_holdout"]["groups_sha256"]:
        raise ValueError("instrument group hash mismatch")
    all_keys = set(sum(groups.values(), []))
    if len(all_keys) != 5 or sum(len(v) for v in groups.values()) != 5:
        raise ValueError("instrument groups are not a disjoint five-instrument partition")
    primary = set(groups["primary"])
    fit_counts = []
    for index, fold in enumerate(folds):
        by_id = {row["id"]: row for row in rows}
        counts = {}
        for phase in ("train", "dev", "calibration"):
            ids = [rid for rid in fold["retained"][phase] if baselines.instrument_key(by_id[rid]) in primary]
            counts[phase] = len(ids)
            if not ids:
                raise ValueError(f"fold {index} has empty primary {phase}")
        fit_counts.append(counts)
    planned = baselines.plan_folds(rows, folds, groups, [], apply_calendar_stress=False)
    if [x["fit_counts"] for x in planned] != fit_counts:
        raise ValueError("T11 instrument-only plan differs from validator counts")
    windows = protocol["calendar_stress_windows"]
    if windows["role_in_t11"] != "pre-registered descriptive test strata only; no row is removed from train/dev/calibration and no metric from a stress stratum may drive selection":
        raise ValueError("calendar stress role changed")
    if windows["regime_masks"] != "Deferred to T12 B5 real-data validation. T11 does not claim point-in-time regime labels or regime-holdout generalization.":
        raise ValueError("regime mask boundary changed")
    expected_seeds = [1729, 2718, 3141]
    linear = protocol["candidates"]["linear"]
    if linear["objectives"] != ["ce", "brier"] or linear["steps"] != 200 or linear["learning_rate"] != 0.05 or linear["l2"] != 0.001 or linear["minibatch"] != 16 or linear["seeds"] != expected_seeds:
        raise ValueError("linear baseline budget or seeds changed")
    if protocol["candidates"]["boosted_stumps"] != {
        "name": "gradient_boosted_stumps", "rounds": 30, "learning_rate": 0.1,
        "seed": None, "threshold_rule": "deterministic train-only adjacent-value partitions; no test-derived thresholds"
    }:
        raise ValueError("boosted-stump budget changed")
    if protocol["metrics"]["brier_definition"] != "mean((p-y)^2) for the binary event" or protocol["metrics"].get("ece_bins") != 10 or protocol["metrics"].get("cross_entropy_definition") != "mean(-y*log(clip(p,1e-15,1-1e-15))-(1-y)*log(clip(1-p,1e-15,1-1e-15)))":
        raise ValueError("Brier definition changed")
    if protocol["metrics"]["interval"] != "paired candidate-minus-baseline 95% percentile bootstrap over whole instrument groups, shared resampled indices, seed 1729, 1000 replicates; no row-iid bootstrap":
        raise ValueError("interval definition changed")
    auth = protocol["authorization"]
    if any(auth.get(k) is not False for k in ("real_fit_performed", "training_authorized", "measurement_authorized", "order_submission_authorized", "live_trading_authorized")) or auth.get("network_model_calls") != 0:
        raise ValueError("protocol authorization boundary is not fail-closed")
    return {
        "schema_version": "nanojev-financial-baselines-preflight-receipt-v1",
        "status": "protocol_valid_not_fit_authorized",
        "protocol": identity(protocol_path),
        "upstream_r1_receipt": identity(ROOT / upstream["bound_receipt_path"]),
        "dataset": {"records": len(rows), "sha256": upstream["dataset_sha256"]},
        "features": protocol["cohort"]["features"],
        "instrument_groups": groups,
        "instrument_groups_sha256": protocol["instrument_holdout"]["groups_sha256"],
        "folds": [{"fold": i, "primary_fit_counts": c, "original_counts": f["counts"]} for i, (c, f) in enumerate(zip(fit_counts, folds))],
        "calendar_stress_role": windows["role_in_t11"],
        "regime_masks_deferred_to": "T12",
        "candidates": protocol["candidates"],
        "authorization": {"real_fit_performed": False, "training_authorized": False, "measurement_authorized": False, "network_model_calls": 0},
        "blockers": ["separate reviewer/fit authorization required", "T12 regime masks are intentionally not claimed by T11"],
        "scope": "Protocol and data-bound preflight only; no model fitting, metrics, quality, profitability, or deployment evidence."
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.protocol, args.receipt)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "folds": result["folds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
