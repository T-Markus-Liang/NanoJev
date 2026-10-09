#!/usr/bin/env python3
"""Fail-closed validator for the T14 RLCD-like estimator protocol (no fitting)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import financial_baselines_v1 as baselines  # noqa: E402


SCHEMA = "nanojev-financial-rlcd-protocol-v1"
EXPECTED_PROTOCOL_SHA256 = "b4a0f14b050f73e79d417d449e3f9e62f016dbe6134cdb72d2300cf054ea0ed5"


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
        raise ValueError("T14 protocol hash is not the pinned review artifact")
    if protocol.get("schema_version") != SCHEMA:
        raise ValueError("wrong T14 protocol schema")
    if protocol.get("status") != "frozen_preflight_not_fit_authorized":
        raise ValueError("protocol is not the frozen preflight version")
    upstream = protocol["upstream"]
    for key, sha_key in (("r1_protocol_path", "r1_protocol_sha256"),
                         ("bound_receipt_path", "bound_receipt_sha256"),
                         ("t11_protocol_path", "t11_protocol_sha256"),
                         ("t11_fit_receipt_path", "t11_fit_receipt_sha256")):
        verify_ref(under_root(ROOT / upstream[key]), upstream[sha_key])
    receipt = json.loads((ROOT / upstream["bound_receipt_path"]).read_text(encoding="utf-8"))
    if receipt.get("status") != "preflight_passed_not_training_authorized" or receipt.get("errors") != []:
        raise ValueError("R1 receipt is not a clean preflight")
    if receipt.get("training_authorized") or receipt.get("measurement_authorized"):
        raise ValueError("upstream receipt unexpectedly authorizes work")
    if receipt.get("full_protocol", {}).get("sha256") != upstream["r1_protocol_sha256"]:
        raise ValueError("receipt full protocol hash differs from pinned upstream protocol")
    if receipt.get("dataset", {}).get("sha256") != upstream["dataset_sha256"]:
        raise ValueError("dataset hash is not bound by R1 receipt")
    if receipt.get("validator", {}).get("sha256") != upstream["validator_sha256"]:
        raise ValueError("validator hash is not bound by R1 receipt")
    if receipt.get("projected_pit_core", {}).get("sha256") != upstream["core_sha256"]:
        raise ValueError("core hash is not bound by R1 receipt")
    for ref_name in ("full_protocol", "projected_pit_core", "validator_report", "dataset", "validator"):
        under_root(receipt[ref_name]["path"])
    t11_fit = json.loads((ROOT / upstream["t11_fit_receipt_path"]).read_text(encoding="utf-8"))
    if t11_fit.get("schema_version") != "nanojev-financial-baselines-run-v1":
        raise ValueError("wrong T11 fit receipt schema")
    if t11_fit.get("status") != upstream["t11_fit_receipt_status_required"]:
        raise ValueError("T11 fit receipt is not the completed honest baseline run")
    if t11_fit.get("protocol_sha256") != upstream["t11_protocol_sha256"]:
        raise ValueError("T11 fit receipt is bound to a different T11 protocol")
    protocol_r1, rows, folds = baselines.load_bound_cohort(ROOT / upstream["bound_receipt_path"])
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
    t11_protocol = json.loads((ROOT / upstream["t11_protocol_path"]).read_text(encoding="utf-8"))
    if declared != t11_protocol["instrument_holdout"]["groups"]:
        raise ValueError("T14 instrument partition differs from the frozen T11 partition")
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
    arms = protocol["estimator_arms"]
    if arms["reference"] != ["base_rate_train_only"]:
        raise ValueError("reference arm changed")
    exact = arms["exact_proper_loss_baselines"]
    if exact["objectives"] != ["ce", "brier"] or exact["steps"] != 200 or exact["learning_rate"] != 0.05 or exact["l2"] != 0.001 or exact["minibatch"] != 16 or exact["seeds"] != [1729, 2718, 3141]:
        raise ValueError("exact-loss budget or seeds differ from frozen T11 values")
    sampled = arms["rlcd_like_sampled"]
    if sampled["estimators"] != ["paired_brier_pg", "correctness_pg"]:
        raise ValueError("sampled estimator set changed")
    for name in ("paired_brier_pg", "correctness_pg"):
        if sampled[name]["reward_samples_M"] != 32:
            raise ValueError("reward sample count changed")
    if sampled["steps"] != 200 or sampled["learning_rate"] != 0.05 or sampled["l2"] != 0.001 or sampled["minibatch"] != 16 or sampled["seeds"] != [1729, 2718, 3141]:
        raise ValueError("sampled-arm budget or seeds changed")
    if sampled["sampling_seed_rule"] != "sampling_seed = training_seed * 1000003 + 17; permutation stream is default_rng(training_seed) identical to the exact-loss arms":
        raise ValueError("sampling seed rule changed")
    metrics = protocol["metrics"]
    if metrics["selective_risk"]["coverages"] != [1.0, 0.9, 0.75, 0.5, 0.25]:
        raise ValueError("selective-risk coverages changed")
    if metrics["selective_risk"]["confidence"] != "s_i = 2 * |p_i - 0.5|":
        raise ValueError("selective-risk confidence rule changed")
    if "1000 replicates" not in metrics["interval"] or "instrument groups" not in metrics["interval"]:
        raise ValueError("interval definition changed")
    windows = protocol["calendar_stress_windows"]
    if windows["role_in_t14"] != "pre-registered descriptive test strata only; no row is removed from train/dev/calibration and no metric from a stress stratum may drive selection":
        raise ValueError("calendar stress role changed")
    derived_windows = baselines.calendar_windows(protocol_r1)
    declared_simple = [{"label": w["label"], "start": w["start"], "end_inclusive": w["end_inclusive"]} for w in windows["windows"]]
    derived_simple = [{"label": w["label"], "start_ns": w["start_ns"], "end_ns": w["end_ns"]} for w in derived_windows]
    if [w["label"] for w in declared_simple] != [w["label"] for w in derived_simple]:
        raise ValueError("declared stress windows differ from upstream R1 list")
    auth = protocol["authorization"]
    if any(auth.get(k) is not False for k in ("real_fit_performed", "training_authorized", "measurement_authorized", "order_submission_authorized", "live_trading_authorized")) or auth.get("network_model_calls") != 0:
        raise ValueError("protocol authorization boundary is not fail-closed")
    return {
        "schema_version": "nanojev-financial-rlcd-preflight-receipt-v1",
        "status": "protocol_valid_not_fit_authorized",
        "protocol": identity(protocol_path),
        "upstream_r1_receipt": identity(ROOT / upstream["bound_receipt_path"]),
        "t11_fit_receipt": identity(ROOT / upstream["t11_fit_receipt_path"]),
        "dataset": {"records": len(rows), "sha256": upstream["dataset_sha256"]},
        "features": protocol["cohort"]["features"],
        "instrument_groups": groups,
        "instrument_groups_sha256": protocol["instrument_holdout"]["groups_sha256"],
        "folds": [{"fold": i, "primary_fit_counts": c, "original_counts": f["counts"]} for i, (c, f) in enumerate(zip(fit_counts, folds))],
        "calendar_stress_role": windows["role_in_t14"],
        "estimator_arms": arms,
        "selective_risk": metrics["selective_risk"],
        "authorization": {"real_fit_performed": False, "training_authorized": False, "measurement_authorized": False, "network_model_calls": 0},
        "blockers": ["separate reviewer/fit authorization required", "T12 regime masks are intentionally not claimed by T14"],
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
