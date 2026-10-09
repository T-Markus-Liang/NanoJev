#!/usr/bin/env python3
"""T11 preflight: revalidate R1 artifacts, then apply instrument/calendar exclusions.

No fitting entry point is exposed until the remaining holdout contract is resolved.
The companion estimators operate on synthetic numerical arrays in unit tests only.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

from financial_pit_v1 import PHASES, validate_dataset, walk_forward
from predict_toy_decisions import read_json, reject_nonfinite, unique_object


def identity(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def verified_artifact(reference):
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        raise ValueError("missing artifact reference")
    path = Path(reference["path"])
    actual = identity(path)
    if actual["sha256"] != reference.get("sha256"):
        raise ValueError(f"artifact hash mismatch: {path.name}")
    if "bytes" in reference and actual["bytes"] != reference["bytes"]:
        raise ValueError(f"artifact size mismatch: {path.name}")
    return path


def load_bound_cohort(receipt_path):
    receipt = read_json(receipt_path)
    if receipt.get("schema_version") != "nanojev-financial-r1-pit-bound-receipt-v1":
        raise ValueError("not an R1 bound receipt")
    if receipt.get("status") != "preflight_passed_not_training_authorized" or receipt.get("errors") != []:
        raise ValueError("R1 binding failed")
    paths = {key: verified_artifact(receipt[key]) for key in
             ("full_protocol", "projected_pit_core", "validator_report", "dataset", "validator")}
    if identity(paths["validator"])["sha256"] != identity(Path(__file__).with_name("financial_pit_v1.py"))["sha256"]:
        raise ValueError("live PIT validator differs from receipt")
    protocol, core, report = (read_json(paths[k]) for k in
                              ("full_protocol", "projected_pit_core", "validator_report"))
    if protocol.get("schema_version") != "nanojev-financial-experiment-protocol-v2" or protocol.get("approval_state", {}).get("approved") is not True:
        raise ValueError("not owner-approved R1 protocol v2")
    if core != protocol.get("pit_validator_core"):
        raise ValueError("core projection mismatch")
    for report_key, receipt_key in (("input", "dataset"), ("protocol", "projected_pit_core"), ("validator", "validator")):
        if report.get(report_key, {}).get("sha256") != receipt[receipt_key]["sha256"]:
            raise ValueError("report artifact binding mismatch")
    rows = [json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_nonfinite)
            for line in paths["dataset"].read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_dataset(rows)
    features = {f["name"] for f in protocol["feature_allowlist"]["features"]}
    frozen = protocol["label_predicate"]["definition_hash_convention"]["frozen_definition_object"]
    definition = hashlib.sha256(json.dumps(frozen, sort_keys=True, ensure_ascii=False,
                                          separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    for row in rows:
        if set(row["features"]) != features:
            raise ValueError("R1 feature allowlist mismatch")
        if row["label"]["definition_sha256"] != definition or row["label"]["event"] != protocol["event_definition"]["event_name"]:
            raise ValueError("R1 event definition mismatch")
        if row["venue"] != "binance_um" or not row["asset_id"].endswith("USDT-PERP"):
            raise ValueError("outside Binance-only USDT-linear R1 cohort")
    folds = walk_forward(rows, **core)
    if folds != report.get("folds") or folds != receipt.get("folds"):
        raise ValueError("recomputed PIT folds differ from report or bound receipt")
    if not all(f["all_phases_nonempty"] for f in folds):
        raise ValueError("empty original PIT phase")
    return protocol, rows, folds


def instrument_key(row):
    # Explicit proposed T11 serialization; not an in-place rewrite of R1 asset_id.
    return f"{row['venue']}:linear:{row['asset_id']}"


def feature_matrix(rows, feature_order):
    """Project rows to a feature-only matrix; metadata never reaches an estimator."""
    if not isinstance(feature_order, list) or not feature_order or len(set(feature_order)) != len(feature_order):
        raise ValueError("feature_order must be a nonempty list of unique names")
    values = []
    labels = []
    ids = []
    for row in rows:
        if set(row.get("features", {})) != set(feature_order):
            raise ValueError("row feature schema differs from feature_order")
        try:
            vector = [row["features"][name]["value"] for name in feature_order]
        except (KeyError, TypeError):
            raise ValueError("row has no complete feature-only projection") from None
        values.append(vector)
        labels.append(row["label"]["outcome"])
        ids.append(row["id"])
    if not values:
        raise ValueError("cannot project an empty row set")
    return values, labels, ids


def instrument_groups(rows):
    keys = sorted({instrument_key(row) for row in rows},
                  key=lambda key: (hashlib.sha256(key.encode()).hexdigest(), key))
    if len(keys) < 3:
        raise ValueError("at least three instruments required for three nonempty groups")
    q, remainder = divmod(len(keys), 3)
    result, offset = {}, 0
    for i, name in enumerate(("primary", "holdout_instrument_A", "holdout_instrument_B")):
        n = q + (i < remainder)
        result[name] = keys[offset:offset+n]
        offset += n
    return result


def calendar_windows(protocol):
    result = []
    for item in protocol["regime_holdout_plan"]["candidate_calendar_stress_windows"]:
        start, end = item["window"].split(" .. ")
        lo = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        hi = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
        if lo >= hi:
            raise ValueError("invalid calendar stress interval")
        # Named end date inclusive; emitted interval is half-open UTC nanoseconds.
        result.append({"label": item["label"], "start_ns": int(lo.timestamp())*10**9,
                       "end_ns": int(hi.timestamp())*10**9})
    return result


def overlaps_stress(row, windows):
    # Conservative treatment of a label ending exactly at stress start: exclude.
    return any(row["decision_ns"] < w["end_ns"] and row["label"]["end_ns"] >= w["start_ns"]
               for w in windows)


def plan_folds(rows, folds, groups, windows, apply_calendar_stress=True):
    """Plan legacy stress-purged folds or the T11 protocol's instrument-only folds.

    The default preserves the W38 diagnostic receipt. T11 v1 passes
    ``apply_calendar_stress=False`` because its pre-registered calendar windows
    are descriptive test strata and regime masks belong to T12.
    """
    by_id = {row["id"]: row for row in rows}
    primary = set(groups["primary"])
    plans = []
    for index, fold in enumerate(folds):
        fit, exclusions = {}, {}
        for phase in ("train", "dev", "calibration"):
            selected, excluded = [], {"instrument_holdout": [], "calendar_stress_overlap": []}
            for rid in fold["retained"][phase]:
                row = by_id[rid]
                # Preserve every applicable exclusion, not only the first reason.
                if instrument_key(row) not in primary:
                    excluded["instrument_holdout"].append(rid)
                if apply_calendar_stress and overlaps_stress(row, windows):
                    excluded["calendar_stress_overlap"].append(rid)
                if instrument_key(row) in primary and (not apply_calendar_stress or not overlaps_stress(row, windows)):
                    selected.append(rid)
            fit[phase], exclusions[phase] = selected, excluded
        tests = {name: [rid for rid in fold["retained"]["test"] if instrument_key(by_id[rid]) in members]
                 for name, members in groups.items()}
        plans.append({"fold": index, "original_counts": fold["counts"],
                      "fit_ids": fit, "fit_counts": {k: len(v) for k, v in fit.items()},
                      "excluded_ids": exclusions, "test_ids_by_instrument_group": tests,
                      "test_counts_by_instrument_group": {k: len(v) for k, v in tests.items()}})
    return plans


def preflight(receipt_path):
    protocol, rows, folds = load_bound_cohort(receipt_path)
    groups, windows = instrument_groups(rows), calendar_windows(protocol)
    plans = plan_folds(rows, folds, groups, windows)
    blockers = [f"fold_{p['fold']}_empty_{phase}_after_holdout" for p in plans
                for phase, count in p["fit_counts"].items() if not count]
    # R1 prose has no byte-level identity or date-boundary serialization. Proposals
    # above must be reviewed before any fit; never choose a partition after outcomes.
    blockers += ["instrument_identity_serialization_requires_freeze",
                 "calendar_end_date_convention_requires_freeze",
                 "point_in_time_regime_holdout_masks_missing"]
    return {"schema_version": "nanojev-financial-baselines-preflight-v1",
            "status": "blocked_before_fit", "r1_binding": identity(receipt_path),
            "implementation": identity(__file__), "record_count": len(rows),
            "feature_order": sorted(rows[0]["features"]),
            "training_seeds": protocol["seed_plan"]["training_seeds"],
            "instrument_key_proposal": "venue:linear:asset_id (UTF-8)",
            "instrument_groups_proposed": groups, "calendar_windows_proposed": windows,
            "regime_holdouts_required": [x["regime_id"] for x in protocol["regime_holdout_plan"]["declared_regime_holdouts"]],
            "folds": plans, "block_reasons": blockers, "training_performed": False,
            "measurement_authorized": False, "training_authorized": False,
            "network_model_calls": 0, "venue_holdout": "unsatisfiable_in_single_venue_v1",
            "scope": "Planning exclusions, not model metrics or completed T11. PIT receipt remains valid for its narrower timestamp/fold scope."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = preflight(args.receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "block_reasons": result["block_reasons"],
                      "fit_counts": [p["fit_counts"] for p in result["folds"]]}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
