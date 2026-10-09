#!/usr/bin/env python3
"""Fail-closed T14 RLCD-like estimator-comparison runner.

Without a ``nanojev-financial-rlcd-fit-authorization-v1`` receipt bound to the
pinned T14 protocol hash (decision ``approved_for_real_fit``) the runner emits a
blocked, content-free receipt and exits 2; no dataset row reaches an estimator.
With it, the frozen R1 cohort is refolded by the unmodified PIT validator via
``financial_baselines_v1.load_bound_cohort`` (the same path the T14 protocol
validator re-derives), every frozen estimator arm is fit per fold on primary
instruments only, selection consumes dev NLL alone, and each arm is evaluated on
test exactly once with paired instrument-group bootstrap intervals against the
``base_rate_train_only`` reference.  The data-free paired-gradient bias check is
run before any fit; its failure aborts the run receipt as frozen.

Estimator objectives are compared, not architectures; nothing here is an
action-utility, profitability, trading, or promotion claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import financial_baseline_estimators_v1 as estimators
import financial_baselines_v1 as baselines
import financial_rlcd_estimators_v1 as rlcd
import validate_financial_rlcd_protocol_v1 as protocol_validator


PROTOCOL_PATH = ROOT / "research/financial_rlcd_protocol_v1.json"
BOUND_RECEIPT = ROOT / "results/financial_pit_r1_bound_receipt_20260920_v5.json"
EXPECTED_PROTOCOL_SHA256 = protocol_validator.EXPECTED_PROTOCOL_SHA256
AUTH_SCHEMA = "nanojev-financial-rlcd-fit-authorization-v1"
RUN_SCHEMA = "nanojev-financial-rlcd-run-v1"
REFERENCE_ARM = "base_rate_train_only"
BIAS_CHECK_GRID_P = [0.1, 0.3, 0.5, 0.7, 0.9]
BIAS_CHECK_REPLICATES = 2000
BIAS_CHECK_MAX_SE = 5.0
GRAD_VARIANCE_REPLICATES = 64
BOOTSTRAP_SEED = 1729
BOOTSTRAP_REPLICATES = 1000

CODE_FILES = (
    "scripts/run_financial_rlcd_v1.py",
    "scripts/financial_rlcd_estimators_v1.py",
    "scripts/financial_baseline_estimators_v1.py",
    "scripts/financial_baselines_v1.py",
    "scripts/financial_pit_v1.py",
    "scripts/validate_financial_rlcd_protocol_v1.py",
)

LIMITATIONS = [
    "Estimator-level comparison only: identical linear sigmoid model class, zero "
    "initialization, minibatch schedule, step budget, learning rate, and L2 for every "
    "fitted arm; only the objective differs. No architecture or action-utility claim.",
    "correctness_pg is a documented biased negative control (expected reward linear in "
    "p); it is never an RLCD candidate and a good showing by it would not validate RLCD.",
    "Venue holdout is unsatisfiable in the Binance-only V1 cohort; no venue "
    "generalization is claimed or implied.",
    "Calendar stress windows are pre-registered descriptive test strata; no metric from "
    "a stress stratum drove selection and no row was removed for them.",
    "T12 regime masks are intentionally not claimed by T14.",
    "Gross 25bps/1d mark-price event probability metrics only; not PnL, edge, "
    "profitability, or a tradable result.",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        raise ValueError("path escapes project root") from None
    return resolved


def _load_protocol(protocol_path: Path = PROTOCOL_PATH, receipt_path: Path = BOUND_RECEIPT):
    # Reuse the authoritative T14 validator.  It checks the pinned protocol hash,
    # every upstream artifact hash, the re-derived cohort/folds/groups, the frozen
    # arm budgets and seeds, and the fail-closed authorization boundary.
    protocol_path = _under_root(protocol_path)
    receipt_path = _under_root(receipt_path)
    preflight = protocol_validator.validate(protocol_path, receipt_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    return protocol, preflight, _sha256(protocol_path)


def _check_authorization(path, protocol_sha256: str):
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
        raise ValueError("fit authorization is bound to a different T14 protocol")
    if protocol_sha256 != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("authorization checked against an unpinned T14 protocol")
    if receipt.get("decision") != "approved_for_real_fit":
        raise ValueError("fit authorization decision is not approved_for_real_fit")
    if receipt.get("fit_authorized") is not True:
        raise ValueError("fit authorization flag is not true")
    if receipt.get("measurement_authorized") is not True:
        raise ValueError("measurement authorization flag is not true")
    if receipt.get("replay_authorized") is True:
        raise ValueError("an estimator-fit authorization must not grant replay")
    if receipt.get("training_authorized") is True:
        raise ValueError("an estimator-fit authorization must not grant model training")
    if receipt.get("order_submission_authorized") is not False:
        raise ValueError("authorization may not permit order submission")
    if receipt.get("live_trading_authorized") is not False:
        raise ValueError("authorization may not permit live trading")
    if receipt.get("network_model_calls") != 0:
        raise ValueError("fit authorization cannot permit network model calls")
    reviewer = receipt.get("independent_reviewer")
    if not isinstance(reviewer, dict) or not reviewer.get("id") or not reviewer.get("reviewed_utc"):
        raise ValueError("reviewer identity and timestamp are required")
    return True, {
        "status": "approved_for_real_fit",
        "receipt_sha256": _sha256(path),
        "reviewer_id": reviewer["id"],
        "reviewed_utc": reviewer["reviewed_utc"],
        "independence": reviewer.get("independence"),
    }


def _rows_for_ids(by_id, ids, feature_order):
    rows = [by_id[rid] for rid in ids]
    X, y, _ = baselines.feature_matrix(rows, feature_order)
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)


def _select_dev(metrics):
    # The only selection signal is dev NLL; ties are resolved by arm name.
    return min(sorted(metrics), key=lambda name: (metrics[name]["nll"], name))


def _bias_check(sampled):
    samples = sampled["paired_brier_pg"]["reward_samples_M"]
    return rlcd.paired_gradient_bias_check(
        BIAS_CHECK_GRID_P, BIAS_CHECK_GRID_P, samples,
        BIAS_CHECK_REPLICATES, seed=BOOTSTRAP_SEED,
        max_standard_errors=BIAS_CHECK_MAX_SE)


def _fit_fold(rows, fold, fold_index, groups, windows, protocol):
    by_id = {row["id"]: row for row in rows}
    order = protocol["cohort"]["features"]
    primary = set(groups["primary"])
    phase, phase_ids = {}, {}
    for name in ("train", "dev", "calibration", "test"):
        ids = fold["retained"][name]
        if name != "test":
            ids = [rid for rid in ids if baselines.instrument_key(by_id[rid]) in primary]
        if not ids:
            raise ValueError(f"fold {fold_index} has empty {name} phase")
        phase_ids[name] = ids
        phase[name] = _rows_for_ids(by_id, ids, order)
    X_train, y_train = phase["train"]
    mean, scale = estimators.fit_standardizer(X_train)
    Z = {name: estimators.transform(X, mean, scale) for name, (X, _) in phase.items()}

    arms = protocol["estimator_arms"]
    exact = arms["exact_proper_loss_baselines"]
    sampled = arms["rlcd_like_sampled"]
    rate = float(np.mean(y_train))
    predictions = {name: {REFERENCE_ARM: np.full(phase[name][1].shape[0], rate)}
                   for name in phase}
    models, fit_seconds = {}, {}
    for seed in exact["seeds"]:
        for objective in exact["objectives"]:
            arm = f"linear_{objective}_seed_{seed}"
            start = time.perf_counter()
            model = estimators.fit_linear(
                Z["train"], y_train, objective=objective, steps=exact["steps"],
                lr=exact["learning_rate"], l2=exact["l2"], seed=seed)
            fit_seconds[arm] = time.perf_counter() - start
            models[arm] = model
            for name in phase:
                predictions[name][arm] = estimators.predict_linear(model, Z[name])
    for seed in sampled["seeds"]:
        for estimator in sampled["estimators"]:
            arm = f"{estimator}_seed_{seed}"
            start = time.perf_counter()
            model = rlcd.fit_linear_pg(
                Z["train"], y_train, estimator=estimator, steps=sampled["steps"],
                lr=sampled["learning_rate"], l2=sampled["l2"], seed=seed,
                reward_samples=sampled[estimator]["reward_samples_M"])
            fit_seconds[arm] = time.perf_counter() - start
            models[arm] = model
            for name in phase:
                predictions[name][arm] = estimators.predict_linear(model, Z[name])

    coverages = protocol["metrics"]["selective_risk"]["coverages"]
    metrics = {name: {arm: estimators.metrics(phase[name][1], p)
                      for arm, p in predictions[name].items()}
               for name in phase}
    selective = {name: {arm: rlcd.selective_risk(phase[name][1], p, coverages)
                        for arm, p in predictions[name].items()}
                 for name in phase}
    selected = _select_dev(metrics["dev"])

    reward_decomposition = {arm: rlcd.reward_parts(y_train, predictions["train"][arm])
                            for arm in models}
    gradient_variance = {}
    for seed in sampled["seeds"]:
        for estimator in sampled["estimators"]:
            arm = f"{estimator}_seed_{seed}"
            gradient_variance[arm] = rlcd.gradient_variance_diagnostic(
                models[arm], Z["train"], y_train, estimator,
                sampled[estimator]["reward_samples_M"],
                replicates=GRAD_VARIANCE_REPLICATES, seed=BOOTSTRAP_SEED)

    # Per-seed metric spread per objective (descriptive; never used for selection).
    seed_groups = {f"linear_{o}": [f"linear_{o}_seed_{s}" for s in exact["seeds"]]
                   for o in exact["objectives"]}
    seed_groups.update({e: [f"{e}_seed_{s}" for s in sampled["seeds"]]
                        for e in sampled["estimators"]})
    estimator_variance = {}
    for objective, arm_names in seed_groups.items():
        dev_nll = [metrics["dev"][a]["nll"] for a in arm_names]
        test_nll = [metrics["test"][a]["nll"] for a in arm_names]
        test_brier = [metrics["test"][a]["brier"] for a in arm_names]
        estimator_variance[objective] = {
            "arms": arm_names,
            "dev_nll_per_arm": dev_nll,
            "test_nll_per_arm": test_nll,
            "test_brier_per_arm": test_brier,
            "dev_nll_spread": float(max(dev_nll) - min(dev_nll)),
            "test_nll_spread": float(max(test_nll) - min(test_nll)),
            "test_brier_spread": float(max(test_brier) - min(test_brier)),
        }

    # Stability: per-instrument-group test metrics for every arm.
    test_ids = phase_ids["test"]
    test_keys = np.asarray([baselines.instrument_key(by_id[rid]) for rid in test_ids])
    y_test = phase["test"][1]
    test_by_group = {}
    for group_name, members in groups.items():
        mask = np.asarray([key in set(members) for key in test_keys])
        if not np.any(mask):
            raise ValueError(f"fold {fold_index} test has no rows in {group_name}")
        test_by_group[group_name] = {
            "count": int(mask.sum()),
            "arms": {arm: estimators.metrics(y_test[mask], p[mask])
                     for arm, p in predictions["test"].items()},
        }
    # Pre-registered descriptive calendar-stress strata; no selection use.
    test_rows = [by_id[rid] for rid in test_ids]
    stress_mask = np.asarray([baselines.overlaps_stress(row, windows) for row in test_rows])
    test_by_stress = {}
    for w in windows:
        mask = np.asarray([
            row["decision_ns"] < w["end_ns"] and row["label"]["end_ns"] >= w["start_ns"]
            for row in test_rows])
        entry = {"count": int(mask.sum()), "arms": {}}
        if np.any(mask):
            entry["arms"] = {arm: estimators.metrics(y_test[mask], p[mask])
                             for arm, p in predictions["test"].items()}
        test_by_stress[w["label"]] = entry
    test_by_stress["__any_stress_window__"] = {
        "count": int(stress_mask.sum()),
        "arms": ({arm: estimators.metrics(y_test[stress_mask], p[stress_mask])
                  for arm, p in predictions["test"].items()}
                 if np.any(stress_mask) else {}),
    }

    # Paired instrument-group bootstrap: candidate minus base_rate, per arm.
    intervals = {arm: estimators.paired_group_ci(
        y_test, predictions["test"][arm], predictions["test"][REFERENCE_ARM],
        test_keys, seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES)
        for arm in predictions["test"]}

    sel_ci = intervals[selected]
    beats = bool(sel_ci["nll"]["upper"] < 0.0)
    worse = bool(sel_ci["nll"]["lower"] > 0.0)
    verdict = {
        "selected_candidate": selected,
        "baseline": REFERENCE_ARM,
        "selected_test_metrics": metrics["test"][selected],
        "nll_delta_vs_base_rate": sel_ci["nll"]["delta"],
        "nll_ci95": [sel_ci["nll"]["lower"], sel_ci["nll"]["upper"]],
        "brier_delta_vs_base_rate": sel_ci["brier"]["delta"],
        "brier_ci95": [sel_ci["brier"]["lower"], sel_ci["brier"]["upper"]],
        "selected_beats_base_rate_95ci": beats,
        "selected_significantly_worse_95ci": worse,
    }
    if beats:
        verdict["statement"] = f"fold {fold_index}: dev-selected {selected} beat base_rate on test (95% group bootstrap CI excludes 0)"
    elif worse:
        verdict["statement"] = f"fold {fold_index}: dev-selected {selected} is significantly worse than base_rate on test"
    elif selected == REFERENCE_ARM:
        verdict["statement"] = f"fold {fold_index}: dev selected base_rate itself; no arm improvement claimed"
    else:
        verdict["statement"] = f"fold {fold_index}: dev-selected {selected} did not beat base_rate on test (CI covers 0)"

    return {
        "fold": fold_index,
        "counts": {name: int(values[1].shape[0]) for name, values in phase.items()},
        "selection": {"metric": "dev.nll", "candidate": selected, "baseline": REFERENCE_ARM},
        "metrics": metrics,
        "selective_risk": selective,
        "reward_decomposition_train": reward_decomposition,
        "estimator_variance": estimator_variance,
        "gradient_variance_diagnostic_train": gradient_variance,
        "test_metrics_by_instrument_group": test_by_group,
        "test_metrics_by_stress_window": test_by_stress,
        "selected_test_metrics": metrics["test"][selected],
        "calibration_only_metrics": metrics["calibration"][selected],
        "paired_group_ci_vs_base_rate": intervals,
        "verdict": verdict,
        "test_group_count": int(len(set(test_keys.tolist()))),
        "fit_seconds_per_arm": fit_seconds,
    }


def _code_hashes():
    return {name: _sha256(ROOT / name) for name in CODE_FILES}


def _summarize(reports):
    beats = [r["fold"] for r in reports if r["verdict"]["selected_beats_base_rate_95ci"]]
    worse = [r["fold"] for r in reports if r["verdict"]["selected_significantly_worse_95ci"]]
    better_arms = [
        {"fold": r["fold"], "arm": arm}
        for r in reports for arm, ci in r["paired_group_ci_vs_base_rate"].items()
        if arm != REFERENCE_ARM and ci["nll"]["upper"] < 0.0
    ]
    if beats:
        statement = (f"dev-selected arms beat base_rate on test in folds {beats}; "
                     "see per-fold verdicts before any interpretation")
    elif worse:
        statement = ("no dev-selected arm beat base_rate on test; dev-selected arms were "
                     f"significantly worse in folds {worse}")
    else:
        statement = "no dev-selected arm beat base_rate on test in any fold"
    return {
        "folds_where_selected_beats_base_rate": beats,
        "folds_where_selected_significantly_worse": worse,
        "arms_with_test_nll_ci_below_base_rate_descriptive": better_arms,
        "statement": statement,
    }


def run(protocol_path=PROTOCOL_PATH, receipt_path=BOUND_RECEIPT, authorization_path=None):
    started = time.time()
    protocol, preflight, protocol_sha256 = _load_protocol(protocol_path, receipt_path)
    authorized, auth = _check_authorization(authorization_path, protocol_sha256)
    common = {
        "schema_version": RUN_SCHEMA,
        "protocol_sha256": protocol_sha256,
        "upstream_preflight_status": preflight["status"],
        "authorization": auth,
        "network_model_calls": 0,
        "fit_authorized": bool(authorized),
        "training_authorized": False,
        "measurement_authorized": bool(authorized),
        "order_submission_authorized": False,
        "live_trading_authorized": False,
        "real_fit_performed": False,
    }
    if not authorized:
        common.update({
            "status": "blocked_fit_authorization_missing",
            "fit_performed": False,
            "scope": "No dataset rows were passed to an estimator; protocol/preflight only.",
            "estimator_arms": preflight["estimator_arms"],
            "folds": [],
        })
        return common

    # Frozen data-free bias check; failure aborts the run receipt before any fit.
    sampled = protocol["estimator_arms"]["rlcd_like_sampled"]
    bias = _bias_check(sampled)
    common["pg_bias_check"] = _json_value(bias)
    if not bias["passed"]:
        common.update({
            "status": "aborted_pg_bias_check_failed",
            "fit_performed": False,
            "scope": "Paired-gradient Monte Carlo bias check failed; no fit performed.",
            "folds": [],
        })
        return common

    protocol_r1, rows, folds = baselines.load_bound_cohort(_under_root(receipt_path))
    groups = baselines.instrument_groups(rows)
    windows = baselines.calendar_windows(protocol_r1)
    reports = [_fit_fold(rows, fold, i, groups, windows, protocol)
               for i, fold in enumerate(folds)]
    upstream = protocol["upstream"]
    common.update({
        "status": "fit_completed_not_promoted",
        "fit_performed": True,
        "real_fit_performed": True,
        "scope": ("T14 estimator-level comparison of RLCD-like sampled proper-reward "
                  "policy-gradient objectives against exact proper-loss baselines and the "
                  "base_rate reference on the frozen R1 cohort. Probability-quality metrics "
                  "only; no action-utility, profitability, trading, or promotion claim."),
        "input_hashes": {
            "t14_protocol_sha256": protocol_sha256,
            "r1_protocol_sha256": upstream["r1_protocol_sha256"],
            "r1_bound_receipt_sha256": upstream["bound_receipt_sha256"],
            "dataset_sha256": upstream["dataset_sha256"],
            "pit_validator_sha256": upstream["validator_sha256"],
            "pit_core_sha256": upstream["core_sha256"],
            "t11_protocol_sha256": upstream["t11_protocol_sha256"],
            "t11_fit_receipt_sha256": upstream["t11_fit_receipt_sha256"],
            "t14_preflight_receipt_sha256": _sha256(ROOT / "results/financial_rlcd_protocol_preflight_20260920_v1.json"),
            "authorization_receipt_sha256": auth["receipt_sha256"],
        },
        "code_hashes": _code_hashes(),
        "cohort": {"records": len(rows), "venue": protocol["cohort"]["venue"],
                   "contract": protocol["cohort"]["contract"],
                   "label_event": protocol["cohort"]["label_event"],
                   "features": protocol["cohort"]["features"]},
        "instrument_groups": groups,
        "estimator_arms_compared": {
            "reference": arms_list(protocol),
            "fit_seconds_scope": "per-arm wall-clock fit seconds only; monetary utility, PnL, and execution costs are out of scope at the estimator level",
            "sampling_multiplier_vs_exact": "M=32 Bernoulli reward draws per row-step for sampled arms; exact-loss arms draw none",
        },
        "folds": reports,
        "verdict_summary": _summarize(reports),
        "limitations": LIMITATIONS,
        "runtime": {"wall_seconds": round(time.time() - started, 6),
                    "python": sys.version.split()[0],
                    "numpy": np.__version__},
        "honesty_note": ("Dev NLL is the only selection signal; test was evaluated once per "
                         "arm and never drove selection. Intervals are paired "
                         "instrument-group bootstrap, not row-iid. A negative-control "
                         "(correctness_pg) failure mode is expected and reported, not "
                         "hidden."),
    })
    return common


def arms_list(protocol):
    arms = protocol["estimator_arms"]
    exact = arms["exact_proper_loss_baselines"]
    sampled = arms["rlcd_like_sampled"]
    names = list(arms["reference"])
    names += [f"linear_{o}_seed_{s}" for s in exact["seeds"] for o in exact["objectives"]]
    names += [f"{e}_seed_{s}" for s in sampled["seeds"] for e in sampled["estimators"]]
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--receipt", type=Path, default=BOUND_RECEIPT)
    parser.add_argument("--authorization-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.protocol, args.receipt, args.authorization_receipt)
    args.output = _under_root(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(_json_value(result), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"],
                      "fit_performed": result["fit_performed"],
                      "folds": len(result.get("folds", []))}, ensure_ascii=False))
    return 0 if result["fit_performed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
