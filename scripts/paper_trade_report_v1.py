#!/usr/bin/env python3
"""Validate T5 receipts and emit attributed, sensitivity-qualified offline diagnostics."""
import argparse
import copy
import datetime as dt
import itertools
import json
import pathlib
import statistics

from paper_trade_attribution_v1 import attribute, number
from paper_trade_protocol_v1 import load_protocol, sha256_file

ROOT = pathlib.Path(__file__).resolve().parent.parent
STUDY_SCHEMA = "nanojev-paper-sensitivity-v1"


def write_new(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_study(path):
    study = json.loads(pathlib.Path(path).read_text())
    if study["schema_version"] != STUDY_SCHEMA:
        raise ValueError("unsupported sensitivity study")
    for axis in ("seeds", "venues"):
        if not study[axis] or len(set(study[axis])) != len(study[axis]):
            raise ValueError(f"empty or duplicate axis: {axis}")
    if any(isinstance(s, bool) or not isinstance(s, int) for s in study["seeds"]):
        raise ValueError("seeds must be integers")
    if not set(study["venues"]) <= {"binance", "bybit", "aster"}:
        raise ValueError("unsupported source")
    if not set(study["appendix_only_venues"]) <= set(study["venues"]):
        raise ValueError("unknown appendix source")
    if "aster" in study["venues"] and "aster" not in study["appendix_only_venues"]:
        raise ValueError("Aster must remain appendix-only pending owner licence resolution")
    if not study["windows"]:
        raise ValueError("empty window axis")
    for first, last in study["windows"].values():
        if dt.date.fromisoformat(first) > dt.date.fromisoformat(last):
            raise ValueError("reversed window")
    if len(set(map(tuple, study["windows"].values()))) != len(study["windows"]):
        raise ValueError("duplicate window")
    if number(study["headline_spread_threshold_fraction_of_initial"]) < 0:
        raise ValueError("negative spread threshold")
    return study


def derived_protocol(base, window, days, seed):
    result = copy.deepcopy(base)
    result["run_id"] = f"t5-{window}-{seed}"
    result["first_day"], result["last_day"] = days
    result["policy"]["seed"] = seed
    return result


def summary(values):
    values = [number(v) for v in values]
    if not values:
        return None
    return {"n": len(values), "min": min(values), "median": statistics.median(values),
            "max": max(values), "spread": max(values) - min(values)}


def check_runtime_policy(receipt, base):
    """Check supplied run parameters against actual emitted simulator settings."""
    policy, declared = receipt["execution_policy"], base["policy"]
    pairs = [(policy["fees"]["fee_bps"], declared["fee_bps"]),
             (policy["spread"]["half_spread_bps"], declared["half_spread_bps"]),
             (policy["perps"]["default_leverage"], declared["leverage"]),
             (policy["perps"]["margin_mode"], declared["margin_mode"]),
             (policy["fills"]["order_time_to_live_ns"], declared["order_time_to_live_days"] * 86_400_000_000_000),
             (policy["capacity"]["max_participation_fraction"], declared["participation_fraction"]),
             (receipt["execution_mode"]["reference_notional_volume"], declared["reference_notional_volume"]),
             (receipt["execution_mode"]["on_divergence"], declared["on_divergence"])]
    if any(actual != expected for actual, expected in pairs):
        raise ValueError("runtime policy does not match base protocol")
    contracts = {c["asset_id"]: c for c in receipt["contracts"]}
    if len(contracts) != len(receipt["contracts"]) or set(contracts) != {f"{s}-PERP" for s in base["symbols"]}:
        raise ValueError("runtime contracts do not match symbols")
    for spec in base["contracts"]:
        symbol, tick, lot, mult, lev, mmr = spec.split(":")
        if symbol not in base["symbols"]:
            continue
        actual = contracts[f"{symbol}-PERP"]
        for field, expected in (("tick_size", tick), ("lot_size", lot), ("contract_multiplier", mult),
                                ("max_leverage", lev), ("maintenance_margin_rate", mmr)):
            if number(actual[field]) != float(expected):
                raise ValueError("runtime contract differs from protocol")
        if actual["venue"] != "binance_um":
            raise ValueError("runtime contract template mismatch")


def sensitivity(rows, threshold):
    """One-axis perturbations hold the other two axes fixed; never pool all PnLs."""
    result = {}
    for axis in ("venue", "seed", "window"):
        fixed = [x for x in ("venue", "seed", "window") if x != axis]
        groups = {}
        for row in rows:
            groups.setdefault(tuple(row[k] for k in fixed), []).append(row)
        entries = []
        for key, members in sorted(groups.items()):
            stats = summary(row["net_pnl"] for row in members)
            fractions = summary(row["net_pnl_fraction_of_initial"] for row in members)
            entries.append({"fixed": dict(zip(fixed, key)), "varied": [r[axis] for r in members],
                            "net_pnl": stats, "fraction_of_initial": fractions,
                            "spread_exceeds_threshold": fractions["spread"] > threshold})
        result[axis] = entries
    return result


def build_report(receipt_paths, study, base, base_digest):
    expected = set(itertools.product(study["venues"], study["seeds"], study["windows"]))
    seen, rows, manifests = set(), [], {}
    signatures = []
    threshold = number(study["headline_spread_threshold_fraction_of_initial"])
    for path in receipt_paths:
        path = pathlib.Path(path)
        receipt = json.loads(path.read_text())
        if receipt.get("receipt_revision") != "t5-executed-path-v1":
            raise ValueError("old receipts lack independently recomputable attribution")
        repro, data = receipt["reproducibility"], receipt["data"]
        venue, seed = data["source_key"], repro["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("invalid receipt seed")
        matching = [w for w, days in study["windows"].items()
                    if days == [data["first_day"], data["last_day"]]]
        if len(matching) != 1:
            raise ValueError("undeclared receipt window")
        window = matching[0]
        cell = venue, seed, window
        if cell not in expected or cell in seen:
            raise ValueError(f"duplicate or unexpected cell: {cell}")
        seen.add(cell)
        protocol_path = ROOT / repro["protocol_path"]
        protocol, _ = load_protocol(protocol_path, repro["protocol_sha256"])
        if protocol != derived_protocol(base, window, study["windows"][window], seed):
            raise ValueError("receipt protocol is not a declared base-only window/seed perturbation")
        manifest = repro["input_manifest"]
        if manifest["path"] != base["input_manifests"][venue]:
            raise ValueError("input manifest does not match source")
        if sha256_file(ROOT / manifest["path"]) != manifest["sha256"]:
            raise ValueError("input manifest changed")
        if venue in manifests and manifests[venue] != manifest:
            raise ValueError("source manifest differs across cells")
        manifests[venue] = manifest
        if (data["symbols"] != base["symbols"] or receipt["strategy"] != base["strategy"]
                or receipt["execution_mode"]["capacity"] != "fixed"
                or receipt["execution_mode"]["sizing"] != "fixed_notional"
                or data["simulation_template_venue"] != "binance_um"):
            raise ValueError("noncomparable strategy/sizing/capacity/template")
        if receipt["ledger"]["initial_cash"] != base["policy"]["initial_cash"]:
            raise ValueError("initial capital mismatch")
        policy = receipt["execution_policy"]
        check_runtime_policy(receipt, base)
        if policy["fills"]["fill_probability"] != 1 or policy["fills"]["reject_probability"] != 0:
            raise ValueError("seed degeneracy declaration invalid for this execution policy")
        signatures.append({"policy": policy, "contracts": receipt["contracts"],
                           "execution_mode": receipt["execution_mode"]})
        first, last = map(dt.date.fromisoformat, study["windows"][window])
        expected_days = (last - first).days + 1
        coverage_ok = set(data["per_symbol"]) == set(base["symbols"])
        for info in data["per_symbol"].values():
            coverage_ok &= (info["quotes"] == expected_days and
                            info["observed_first_day"] == first.isoformat() and
                            info["observed_last_day"] == last.isoformat())
        if sum(v["quotes"] for v in data["per_symbol"].values()) != data["quote_count"]:
            raise ValueError("quote counts do not reconcile")
        if sum(v["decisions"] for v in data["per_symbol"].values()) != receipt["counts"]["decisions"]:
            raise ValueError("decision counts do not reconcile")
        if not isinstance(receipt.get("replay_sha256"), str) or len(receipt["replay_sha256"]) != 64:
            raise ValueError("missing replay digest")
        attributed = attribute(receipt["ledger"], receipt["accounting_evidence"])
        if attributed != receipt["attribution"]:
            raise ValueError("stored attribution differs from independently recomputed evidence")
        net = number(receipt["ledger"]["net_pnl"])
        initial = number(receipt["ledger"]["initial_cash"])
        if initial <= 0:
            raise ValueError("nonpositive initial capital")
        if receipt["conservation"] is not True:
            raise ValueError("failed conservation")
        orders = receipt["accounting_evidence"]["orders"]
        actual_divergences = sum(order["status"] != "filled" or
                                abs(order["filled_quantity"] - order["requested_quantity"]) > 1e-9
                                for order in orders)
        if (actual_divergences != receipt["divergence_count"] or
                len(orders) != receipt["counts"]["orders"] or
                len(receipt["accounting_evidence"]["fills"]) != receipt["counts"]["fills"]):
            raise ValueError("order/fill/divergence counts disagree with evidence")
        rows.append({"venue": venue, "seed": seed, "window": window,
                     "days": study["windows"][window], "duration_days": expected_days,
                     "receipt_path": str(path), "receipt_sha256": sha256_file(path),
                     "protocol_sha256": repro["protocol_sha256"], "replay_sha256": receipt["replay_sha256"],
                     "net_pnl": net, "net_pnl_fraction_of_initial": net / initial,
                     "attribution": attributed, "coverage_complete": bool(coverage_ok),
                     "per_symbol": data["per_symbol"], "counts": receipt["counts"],
                     "divergence_count": receipt["divergence_count"], "drawdown": receipt["drawdown"]})
    if seen != expected:
        raise ValueError(f"incomplete sensitivity matrix; missing {sorted(expected - seen)}")
    if any(s != signatures[0] for s in signatures):
        raise ValueError("execution policies/contracts changed across sensitivity cells")
    rows.sort(key=lambda row: (row["window"], row["seed"], row["venue"]))
    primary = [r for r in rows if r["venue"] not in study["appendix_only_venues"]]
    appendix = [r for r in rows if r["venue"] in study["appendix_only_venues"]]
    axes = sensitivity(primary, threshold)
    reasons = ["provisional_costs_contracts_and_fill_funding_conventions_pending_R1",
               "current_snapshot_not_asof_vintage", "reference_strategy_not_model_evidence"]
    if len({r["venue"] for r in primary}) < 3:
        reasons.append("fewer_than_three_non_appendix_venues")
    if any(e["spread_exceeds_threshold"] for axis in axes.values() for e in axis):
        reasons.append("declared_spread_threshold_exceeded")
    if not all(r["coverage_complete"] for r in primary):
        reasons.append("incomplete_window_coverage")
    if any(r["divergence_count"] for r in primary):
        reasons.append("order_outcome_divergences")
    if "bybit" in study["venues"]:
        reasons.append("bybit_terms_unverified")
    if any(e["net_pnl"]["spread"] > 1e-6 for e in sensitivity(rows, threshold)["seed"]):
        raise ValueError("unexpected nondegenerate seed outcome")
    return {
        "schema_version": "nanojev-paper-trade-b0-report-v1",
        "base_protocol_sha256": base_digest, "study": study, "matrix_cells": len(rows),
        "input_manifests": manifests,
        "headline_allowed": not reasons, "headline_refusal_reasons": reasons,
        "headline_spread_threshold_fraction_of_initial": threshold,
        "primary": {"matrix": primary, "sensitivity": axes},
        "appendix_licence_conflict": {
            "matrix": appendix,
            "all_three_venue_comparison_for_diagnostics_only": sensitivity(rows, threshold)["venue"],
            "notice": "Aster excluded from all primary tables/headlines: unresolved licence conflict and manifest base_url incorrectly names Hyperliquid; original manifest preserved. No source/legal attestation."},
        "max_abs_attribution_residual": max(r["attribution"]["max_abs_residual"] for r in rows),
        "seed_axis_degenerate": True, "independent_seed_replications": False,
        "conclusion": "Offline plumbing/accounting evidence only. No model edge, deployment or training authorization.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=pathlib.Path, default=ROOT / "research/paper_trade_t5_sensitivity_v1.json")
    parser.add_argument("--receipts", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    study = load_study(args.study)
    base, evidence = load_protocol(ROOT / study["base_protocol"])
    report = build_report(sorted(args.receipts.glob("*.json")), study, base, evidence["protocol_sha256"])
    report["study_sha256"] = sha256_file(args.study)
    write_new(args.output, report)
    print(json.dumps({"output": str(args.output), "matrix_cells": report["matrix_cells"],
                      "headline_allowed": report["headline_allowed"]}))


if __name__ == "__main__":
    main()
