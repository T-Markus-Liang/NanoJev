#!/usr/bin/env python3
"""Fail-closed T11 baseline runner.

The runner is deliberately split from the T11 preflight.  Without a separately
reviewed fit-authorization receipt it emits a blocked, content-free receipt and
does not load the dataset into an estimator.  With a valid authorization receipt
it fits only the frozen primary instruments, selects on dev, reports calibration
without fitting a calibration transform, and evaluates the selected candidate on
test exactly once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import financial_baseline_estimators_v1 as estimators
import financial_baselines_v1 as baselines
import validate_financial_baselines_protocol_v1 as protocol_validator


PROTOCOL_PATH = ROOT / "research/financial_baselines_protocol_v1.json"
BOUND_RECEIPT = ROOT / "results/financial_pit_r1_bound_receipt_20260920_v5.json"
AUTH_SCHEMA = "nanojev-financial-baselines-fit-authorization-v1"
RUN_SCHEMA = "nanojev-financial-baselines-run-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _under_root(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        raise ValueError("authorization or output path escapes project root") from None
    return resolved


def _load_protocol(protocol_path: Path = PROTOCOL_PATH, receipt_path: Path = BOUND_RECEIPT):
    # Reuse the authoritative validator.  It checks every upstream hash, split,
    # feature order, budget, and authorization boundary before any fit.
    protocol_path = _under_root(protocol_path)
    receipt_path = _under_root(receipt_path)
    preflight = protocol_validator.validate(protocol_path, receipt_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    return protocol, preflight, _sha256(protocol_path)


def _check_authorization(path: Path | None, protocol_sha256: str):
    if path is None:
        return False, {"status": "missing", "reason": "fit_authorization_receipt_required"}
    path = _under_root(path)
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid fit authorization receipt: {exc}") from None
    if receipt.get("schema_version") != AUTH_SCHEMA:
        raise ValueError("wrong fit authorization receipt schema")
    if receipt.get("protocol_sha256") != protocol_sha256:
        raise ValueError("fit authorization is bound to a different T11 protocol")
    if receipt.get("decision") != "approved_for_real_fit":
        raise ValueError("fit authorization decision is not approved_for_real_fit")
    if receipt.get("fit_authorized") is not True:
        raise ValueError("fit authorization flag is not true")
    if receipt.get("measurement_authorized") is not True:
        raise ValueError("measurement authorization flag is not true")
    reviewer = receipt.get("independent_reviewer")
    if not isinstance(reviewer, dict) or not reviewer.get("id") or not reviewer.get("reviewed_utc"):
        raise ValueError("independent reviewer identity and timestamp are required")
    if receipt.get("network_model_calls") != 0:
        raise ValueError("fit authorization cannot permit network model calls")
    return True, {
        "status": "approved_for_real_fit",
        "receipt_sha256": _sha256(path),
        "reviewer_id": reviewer["id"],
        "reviewed_utc": reviewer["reviewed_utc"],
    }


def _rows_for_ids(by_id, ids, feature_order):
    rows = [by_id[rid] for rid in ids]
    X, y, _ = baselines.feature_matrix(rows, feature_order)
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)


def _rule_predictions(name, X, train_y, seed):
    rate = float(np.mean(train_y))
    if name == "always_negative":
        return np.zeros(X.shape[0], dtype=np.float64)
    if name == "always_positive":
        return np.ones(X.shape[0], dtype=np.float64)
    if name == "base_rate_train_only":
        return np.full(X.shape[0], rate, dtype=np.float64)
    if name == "random_base_rate_seeded":
        rng = np.random.default_rng(seed)
        return (rng.random(X.shape[0]) < rate).astype(np.float64)
    if name == "single_feature_basis_sign":
        # The feature is standardized using train-only statistics by the caller.
        return np.where(X[:, 0] >= 0.0, 0.75, 0.25).astype(np.float64)
    raise ValueError(f"unknown deterministic candidate: {name}")


def _candidate_predictions(X_train, y_train, X_eval, protocol):
    mean, scale = estimators.fit_standardizer(X_train)
    Z_train = estimators.transform(X_train, mean, scale)
    Z_eval = estimators.transform(X_eval, mean, scale)
    candidates = {}
    deterministic = protocol["candidates"]["deterministic"]
    for name in deterministic:
        # Random baseline is evaluated with each declared training seed below;
        # other rules are seed independent.
        candidates[name] = _rule_predictions(name, Z_eval, y_train, 1729)
    linear = protocol["candidates"]["linear"]
    for seed in linear["seeds"]:
        for objective in linear["objectives"]:
            model = estimators.fit_linear(
                Z_train, y_train, objective=objective,
                steps=linear["steps"], lr=linear["learning_rate"],
                l2=linear["l2"], seed=seed,
            )
            candidates[f"linear_{objective}_seed_{seed}"] = estimators.predict_linear(model, Z_eval)
    stump = protocol["candidates"]["boosted_stumps"]
    model = estimators.fit_gbm(Z_train, y_train, rounds=stump["rounds"], lr=stump["learning_rate"])
    candidates[stump["name"]] = estimators.predict_gbm(model, Z_eval)
    return candidates


def _metrics_by_candidate(candidates, y):
    return {name: _json_value(estimators.metrics(y, p)) for name, p in candidates.items()}


def _select_dev(metrics):
    # The only selection signal is dev NLL; ties are resolved by candidate name.
    return min(sorted(metrics), key=lambda name: (metrics[name]["nll"], name))


def _fit_fold(rows, fold, fold_index, groups, protocol):
    by_id = {row["id"]: row for row in rows}
    order = protocol["cohort"]["features"]
    phase = {}
    for name in ("train", "dev", "calibration", "test"):
        ids = fold["retained"][name]
        if name != "test":
            primary = set(groups["primary"])
            ids = [rid for rid in ids if baselines.instrument_key(by_id[rid]) in primary]
        phase[name] = _rows_for_ids(by_id, ids, order)
    X_train, y_train = phase["train"]
    all_predictions = {name: _candidate_predictions(X_train, y_train, X, protocol) for name, (X, _) in phase.items()}
    metrics = {name: _metrics_by_candidate(predictions, phase[name][1]) for name, predictions in all_predictions.items()}
    selected = _select_dev(metrics["dev"])
    test_ids = fold["retained"]["test"]
    test_groups = np.asarray([baselines.instrument_key(by_id[rid]) for rid in test_ids])
    baseline_name = "base_rate_train_only"
    ci = estimators.paired_group_ci(
        phase["test"][1], all_predictions["test"][selected],
        all_predictions["test"][baseline_name], test_groups,
        seed=protocol["metrics"]["interval"].startswith("paired") and 1729 or 1729,
        replicates=1000,
    )
    return {
        "fold": fold_index,
        "counts": {name: int(values[1].shape[0]) for name, values in phase.items()},
        "selection": {"metric": "dev.nll", "candidate": selected, "baseline": baseline_name},
        "metrics": metrics,
        "selected_test_metrics": metrics["test"][selected],
        "calibration_only_metrics": metrics["calibration"][selected],
        "selected_vs_base_rate_group_ci": _json_value(ci),
        "test_group_count": int(len(set(test_groups.tolist()))),
    }


def run(protocol_path=PROTOCOL_PATH, receipt_path=BOUND_RECEIPT, authorization_path=None):
    protocol, preflight, protocol_sha256 = _load_protocol(protocol_path, receipt_path)
    authorized, auth = _check_authorization(authorization_path, protocol_sha256)
    common = {
        "schema_version": RUN_SCHEMA,
        "protocol_sha256": protocol_sha256,
        "upstream_preflight_status": preflight["status"],
        "authorization": auth,
        "network_model_calls": 0,
        "training_authorized": bool(authorized),
        "measurement_authorized": bool(authorized),
        "real_fit_performed": False,
    }
    if not authorized:
        common.update({
            "status": "blocked_fit_authorization_missing",
            "fit_performed": False,
            "scope": "No dataset rows were passed to an estimator; protocol/preflight only.",
            "folds": [],
        })
        return common

    _, rows, folds = baselines.load_bound_cohort(receipt_path)
    groups = baselines.instrument_groups(rows)
    reports = [_fit_fold(rows, fold, i, groups, protocol) for i, fold in enumerate(folds)]
    common.update({
        "status": "fit_completed_not_promoted",
        "fit_performed": True,
        "real_fit_performed": True,
        "scope": "T11 baseline metrics only; no action utility, profitability, trading, or promotion claim.",
        "instrument_groups": groups,
        "folds": reports,
    })
    return common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--receipt", type=Path, default=BOUND_RECEIPT)
    parser.add_argument("--authorization-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.protocol, args.receipt, args.authorization_receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(_json_value(result), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "fit_performed": result["fit_performed"]}, ensure_ascii=False))
    return 0 if result["status"] != "blocked_fit_authorization_missing" else 2


if __name__ == "__main__":
    raise SystemExit(main())
