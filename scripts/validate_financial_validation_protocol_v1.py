#!/usr/bin/env python3
"""Fail-closed validator for the T12 real-data validation protocol.

Verifies every pinned upstream hash, re-derives the R1 folds through the
unmodified PIT validator, recomputes the point-in-time regime masks and the
calendar-stress windows, checks the declared policy set and execution-policy
values, and confirms the authorization boundary is fail-closed. It emits a
preflight receipt only; it performs no replay and no fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_nanojev_v2 import file_identity  # noqa: E402
import financial_baselines_v1 as baselines  # noqa: E402
import financial_real_data_path_v1 as realpath  # noqa: E402


SCHEMA = "nanojev-financial-validation-protocol-v1"
EXPECTED_PROTOCOL_SHA256 = "507ee3e280d84693b017979a811bf90601cdcc630e1411ebf54a64aa9ecc31cf"

DECLARED_REGIMES = set(realpath.REGIME_IDS)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _under_root(path):
    path = Path(path).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError:
        raise ValueError(f"artifact path escapes project root: {path}") from None
    return path


def _verify_pinned(relative_path, expected_sha256, label):
    path = _under_root(ROOT / relative_path)
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} hash mismatch: {relative_path}")
    return file_identity(path)


def _flatten(policy_dict, prefix=""):
    flat = {}
    for key, value in policy_dict.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _check_authorization_block(protocol):
    auth = protocol["authorization"]
    for flag in ("real_replay_performed", "replay_authorized", "measurement_authorized",
                 "training_authorized", "order_submission_authorized",
                 "live_trading_authorized"):
        if auth.get(flag) is not False:
            raise ValueError(f"protocol authorization flag {flag} is not fail-closed")
    if auth.get("network_model_calls") != 0:
        raise ValueError("protocol may not declare network model calls")


def _check_t11_receipt(protocol):
    upstream = protocol["upstream_t11"]
    path = _under_root(ROOT / upstream["fit_receipt_path"])
    if _sha256(path) != upstream["fit_receipt_sha256"]:
        raise ValueError("T11 fit receipt hash mismatch")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "nanojev-financial-baselines-run-v1":
        raise ValueError("upstream T11 receipt has wrong schema")
    if receipt.get("status") != upstream["required_status"]:
        raise ValueError("T11 fit receipt status is not the bound honest outcome")
    if receipt.get("fit_performed") is not True or receipt.get("real_fit_performed") is not True:
        raise ValueError("bound T11 receipt must be the completed fit")
    folds = receipt.get("folds")
    if not isinstance(folds, list) or len(folds) != 3:
        raise ValueError("bound T11 receipt must carry exactly three fold reports")
    selections = [f["selection"]["candidate"] for f in folds]
    intervals = [f["selected_vs_base_rate_group_ci"]["nll"] for f in folds]
    return {"receipt_sha256": upstream["fit_receipt_sha256"],
            "status": receipt["status"],
            "dev_selected_candidates": selections,
            "selected_vs_base_rate_nll_delta_ci95": intervals}


def _check_cohort_properties(protocol, rows):
    observed = protocol["cohort"]["observed_properties"]
    if len(rows) != protocol["cohort"]["records"]:
        raise ValueError("cohort record count mismatch")
    if {row["asset_id"] for row in rows} != set(protocol["cohort"]["instruments"]):
        raise ValueError("cohort instrument set mismatch")
    if {row["venue"] for row in rows} != {protocol["cohort"]["venue"]}:
        raise ValueError("cohort venue set mismatch")
    if {row["features"]["funding_interval_hours"]["value"] for row in rows} != {
            observed["funding_interval_hours"]}:
        raise ValueError("funding interval feature is not the declared constant")
    if any(row["features"]["mark_price"]["available_ns"] != row["decision_ns"] for row in rows):
        raise ValueError("mark availability must equal decision_ns for every record")
    positive = sum(row["label"]["outcome"] for row in rows) / len(rows)
    if abs(positive - observed["positive_rate"]) > 1e-12:
        raise ValueError("cohort positive rate drifted from the declared value")
    for row in rows:
        if row["label"]["definition_sha256"] != protocol["upstream_r1"]["event_definition_sha256"]:
            raise ValueError("a record carries a different event definition hash")


def _check_regime_masks(protocol, rows):
    spec = protocol["pit_regime_masks"]
    declared_ids = {m["regime_id"] for m in spec["masks"]}
    if declared_ids != DECLARED_REGIMES:
        raise ValueError("declared mask ids differ from the implementation's frozen ids")
    if len(declared_ids) != len(spec["masks"]):
        raise ValueError("duplicate regime mask id")
    masks_a = realpath.compute_regime_masks(rows)
    masks_b = realpath.compute_regime_masks(list(reversed(rows)))
    if masks_a != masks_b:
        raise ValueError("regime masks are not deterministic under input reordering")
    prefix = [row for row in sorted(rows, key=lambda r: (r["decision_ns"], r["id"]))
              [:realpath.MASK_MIN_HISTORY]]
    for entry in realpath.compute_regime_masks(prefix).values():
        if entry["evaluable"]:
            raise ValueError("warmup rule violated: a mask voted before minimum history")
    stats = realpath.mask_stats(masks_a)
    for regime_id, counts in stats.items():
        if counts["evaluable"] == 0:
            raise ValueError(f"regime {regime_id} is never evaluable on the cohort")
        if counts["member"] == 0:
            raise ValueError(f"regime {regime_id} has zero members on the cohort")
    return masks_a, stats


def _check_calendar_windows(protocol, r1_protocol, folds):
    declared = protocol["calendar_stress_windows"]
    windows = baselines.calendar_windows(r1_protocol)
    if len(windows) != len(declared["windows"]):
        raise ValueError("calendar stress window count differs from the frozen R1 list")
    declared_labels = {w["label"] for w in declared["windows"]}
    if {w["label"] for w in windows} != declared_labels:
        raise ValueError("calendar stress labels differ from the frozen R1 list")
    earliest_test = min(fold["windows"]["test"][0] for fold in folds)
    if any(w["end_ns"] > earliest_test for w in windows):
        raise ValueError("a stress window overlaps a test window; declared-empty assumption broken")
    return windows


def _check_execution_policy(protocol):
    declared = protocol["execution_policy"]["values"]
    actual = _flatten(realpath.declared_execution_policy().to_dict())
    for key, expected in declared.items():
        if key not in actual:
            raise ValueError(f"execution policy lacks declared key {key}")
        if actual[key] != expected:
            raise ValueError(f"execution policy {key} differs from the declared frozen value")
    if actual["fills.order_time_to_live_ns"] != realpath.ORDER_TTL_NS:
        raise ValueError("order TTL constant drifted from the declared real-path value")
    if actual["perps.funding_rate_source"] != "quote_funding_rate_field":
        raise ValueError("funding source must read the cohort's settled-rate field")


def _check_policies(protocol):
    names = {p["name"] for p in protocol["policies"]["reference_policies"]}
    if names != set(realpath.REFERENCE_POLICIES):
        raise ValueError("declared reference policy set differs from the implementation")
    interface = protocol["policies"]["probability_policy_interface"]
    if interface.get("status") != "declared_fail_closed_inactive":
        raise ValueError("probability-policy interface must be declared fail-closed inactive")


def validate(protocol_path, receipt_path):
    protocol_path = _under_root(protocol_path)
    receipt_path = _under_root(receipt_path)
    if EXPECTED_PROTOCOL_SHA256.startswith("REPLACE"):
        raise ValueError("validator is unpinned: EXPECTED_PROTOCOL_SHA256 placeholder")
    if _sha256(protocol_path) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("T12 protocol hash is not the pinned review artifact")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != SCHEMA:
        raise ValueError("wrong T12 protocol schema")
    if protocol.get("status") != "frozen_preflight_not_replay_authorized":
        raise ValueError("protocol is not the frozen preflight version")
    upstream = protocol["upstream_r1"]
    identities = {
        "protocol": _verify_pinned(upstream["protocol_path"], upstream["protocol_sha256"],
                                   "R1 protocol"),
        "bound_receipt": _verify_pinned(upstream["bound_receipt_path"],
                                        upstream["bound_receipt_sha256"], "bound receipt"),
        "dataset": _verify_pinned(upstream["dataset_path"], upstream["dataset_sha256"],
                                  "dataset"),
        "pit_validator": _verify_pinned("scripts/financial_pit_v1.py",
                                        upstream["validator_sha256"], "PIT validator"),
        "pit_core": _verify_pinned("research/financial_r1_pit_validator_core_v2.json",
                                   upstream["core_sha256"], "PIT core"),
        "frozen_parameters": _verify_pinned(upstream["frozen_parameters_path"],
                                            upstream["frozen_parameters_sha256"],
                                            "frozen parameters"),
    }
    if _sha256(receipt_path) != upstream["bound_receipt_sha256"]:
        raise ValueError("receipt argument is not the pinned R1 bound receipt")
    r1_protocol, rows, folds = baselines.load_bound_cohort(receipt_path)
    _check_authorization_block(protocol)
    _check_cohort_properties(protocol, rows)
    t11 = _check_t11_receipt(protocol)
    groups = baselines.instrument_groups(rows)
    if groups != protocol["instrument_groups"]["groups"]:
        raise ValueError("instrument groups differ from the declared partition")
    masks, stats = _check_regime_masks(protocol, rows)
    _check_calendar_windows(protocol, r1_protocol, folds)
    _check_execution_policy(protocol)
    _check_policies(protocol)
    cell_plan = []
    by_id = {row["id"]: row for row in rows}
    for index, fold in enumerate(folds):
        if not fold["all_phases_nonempty"]:
            raise ValueError(f"fold {index} has an empty retained phase")
        for group_name in ("primary", "holdout_instrument_A", "holdout_instrument_B"):
            for key in groups[group_name]:
                n = sum(1 for rid in fold["retained"]["test"]
                        if baselines.instrument_key(by_id[rid]) == key)
                cell_plan.append({"fold": index, "instrument_group": group_name,
                                  "instrument_key": key,
                                  "retained_test_records": n,
                                  "policies": list(realpath.REFERENCE_POLICIES)})
    if any(cell["retained_test_records"] == 0 for cell in cell_plan):
        raise ValueError("a declared cell has no retained test records")
    return {
        "schema_version": "nanojev-financial-validation-preflight-v1",
        "status": "protocol_valid_not_replay_authorized",
        "protocol": file_identity(protocol_path),
        "upstream_artifacts": identities,
        "t11_context": t11,
        "dataset": {"records": len(rows), "sha256": upstream["dataset_sha256"]},
        "event_definition_sha256": upstream["event_definition_sha256"],
        "instrument_groups": groups,
        "instrument_groups_sha256": protocol["instrument_groups"]["groups_sha256"],
        "mask_stats": stats,
        "masks_sha256": hashlib.sha256(
            json.dumps(masks, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "calendar_stress_strata": "declared_but_empty_in_all_frozen_test_windows",
        "cell_plan": cell_plan,
        "cell_count": len(cell_plan),
        "folds": [{"fold": i, "counts": f["counts"],
                   "exclusion_counts": f["exclusion_counts"]}
                  for i, f in enumerate(folds)],
        "authorization": {"real_replay_performed": False, "replay_authorized": False,
                          "measurement_authorized": False, "training_authorized": False,
                          "network_model_calls": 0},
        "blockers": ["separate nanojev-financial-validation-authorization-v1 receipt required "
                     "before any replay",
                     "venue holdout unsatisfiable in single-venue V1 cohort"],
        "scope": ("Protocol and data-bound preflight only; no replay, no fit, no model quality, "
                  "profitability, or deployment evidence. The real data flows through the "
                  "unmodified financial_backtest_v1.run_backtest machinery in the runner."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "research/financial_validation_protocol_v1.json")
    parser.add_argument("--receipt", type=Path,
                        default=ROOT / "results/financial_pit_r1_bound_receipt_20260920_v5.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.protocol, args.receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "cell_count": result["cell_count"],
                      "mask_stats": result["mask_stats"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
