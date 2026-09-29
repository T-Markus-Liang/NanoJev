#!/usr/bin/env python3
"""T13 (B6) risk gate on the T12 real-data replay path.

This module attaches the existing out-of-model guard set
(``financial_risk_v1.RiskEngine``) to the frozen-cohort cells built by
``financial_real_data_path_v1``. It reuses every T12 primitive — record→Quote
adaptation, point-in-time regime masks, reference decision streams, the declared
execution policy — and calls the **unmodified**
``financial_backtest_v1.run_backtest``; the only delta is ``risk_gate`` going
from ``None`` to a ``RiskEngine`` parameterized by the frozen R1 ``limits.*``
set (``research/financial_r1_frozen_parameters_v1.json``).

Gate lifecycle
--------------
``run_backtest(verify_determinism=True)`` replays twice but reuses ONE gate
object across both replays. ``RiskEngine`` keeps mutable trailing order-rate and
turnover windows (and an optional kill-switch latch) that a second replay must
not inherit: the module docstring assigns window bookkeeping to the *caller*
per replay. T13 therefore performs the double replay one level up — two
independent ``run_backtest(verify_determinism=False)`` calls, each with a
freshly constructed ``RiskEngine`` — and compares the full-receipt
fingerprints. Same total work, same pinned machinery, and a strictly stronger
guarantee (independent gate instances rather than a shared mutable one).

Honesty statement
-----------------
Gate outcomes are deterministic guard mechanics under provisional frozen
limits, not market measurements. In particular the frozen set carries
``min_position_quantity = 0``, which makes every net-short intent out of
bounds: ``unit_short_always`` cells are expected to report a 100% block rate.
That is a property of the frozen parameters (written for a spot-style book
before the perp context), reported verbatim — not tuned away, and not evidence
about any instrument. Simulated PnL remains descriptive-only; headline_allowed
stays false.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_nanojev_v2 import canonical_json, file_identity, sha256_bytes
import financial_backtest_v1 as backtest
import financial_risk_v1 as risk
import financial_real_data_path_v1 as realpath

SCHEMA = "nanojev-financial-real-data-risk-v1"
FROZEN_PARAMETERS_PATH = "research/financial_r1_frozen_parameters_v1.json"
GATE_NAME = "financial_risk_v1.RiskEngine"
GATE_POLICY_VERSION = risk.RISK_POLICY_VERSION

ROOT = Path(__file__).resolve().parents[1]


class RiskGatePathError(RuntimeError):
    """Contract violation in the gated real-data path."""


# --------------------------------------------------------------------------
# Frozen limit loading (fail-closed)
# --------------------------------------------------------------------------
def load_frozen_risk_limits(frozen_path=None):
    """Construct RiskLimits from the frozen R1 ``limits.*`` entries, verbatim.

    Fail-closed: the frozen file must carry exactly the RiskLimits field set
    (minus ``policy_version``, which the dataclass pins itself); a missing,
    extra, renamed, or non-finite entry aborts before any replay.
    """
    path = ROOT / FROZEN_PARAMETERS_PATH if frozen_path is None else Path(frozen_path)
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RiskGatePathError(f"cannot read frozen R1 parameters: {exc}") from exc
    parameters = document.get("simulator_and_risk_parameters")
    if not isinstance(parameters, dict):
        raise RiskGatePathError("frozen parameters lack simulator_and_risk_parameters")
    declared = {}
    for key, value in parameters.items():
        if key.startswith("limits."):
            declared[key.split(".", 1)[1]] = value
    expected = set(risk.RiskLimits().to_dict()) - {"policy_version"}
    missing = expected - set(declared)
    extra = set(declared) - expected
    if missing or extra:
        raise RiskGatePathError(
            f"frozen limits.* set differs from RiskLimits fields "
            f"(missing={sorted(missing)}, extra={sorted(extra)})")
    try:
        return risk.RiskLimits(**declared)
    except (TypeError, ValueError) as exc:
        raise RiskGatePathError(f"frozen limits.* values rejected by RiskLimits: {exc}") from exc


def frozen_gate_factory(frozen_path=None):
    """Return a callable producing a fresh RiskEngine with frozen limits."""
    limits = load_frozen_risk_limits(frozen_path)
    return lambda: risk.RiskEngine(limits=limits)


# --------------------------------------------------------------------------
# Gate analytics extracted from the harness decision log
# --------------------------------------------------------------------------
def gate_decision_stats(receipt):
    """Aggregate the per-decision gate outcomes recorded by the simulator.

    ``risk_decision`` on a decision record is the gate verdict for that
    decision's intent (allow / reduce / block); ``None`` means the decision
    never reached the gate (abstain/no-trade, no reference price, target
    already satisfied, or gate absent). ``reduced_quantity_total`` sums
    ``requested - allowed`` over ``reduce`` verdicts: the gate-triggered
    exposure reduction in base units.
    """
    stats = {"evaluated_decisions": 0, "allowed_decisions": 0,
             "reduced_decisions": 0, "blocked_decisions": 0,
             "reason_code_counts": {}, "reduced_quantity_total": 0.0,
             "non_allow_records": []}
    counter = {risk.DECISION_ALLOW: "allowed_decisions",
               risk.DECISION_REDUCE: "reduced_decisions",
               risk.DECISION_BLOCK: "blocked_decisions"}
    for record in receipt["decisions"]:
        verdict = record["risk_decision"]
        if verdict is None:
            continue
        if verdict not in counter:
            raise RiskGatePathError(f"gate returned undeclared verdict {verdict!r}")
        stats["evaluated_decisions"] += 1
        stats[counter[verdict]] += 1
        for code in record["risk_reason_codes"]:
            stats["reason_code_counts"][code] = \
                stats["reason_code_counts"].get(code, 0) + 1
        if verdict == risk.DECISION_REDUCE:
            stats["reduced_quantity_total"] += max(
                0.0, record["requested_quantity"] - record["risk_allowed_quantity"])
        if verdict != risk.DECISION_ALLOW:
            stats["non_allow_records"].append({
                "decision_id": record["decision_id"], "decision_ns": record["decision_ns"],
                "risk_decision": verdict,
                "risk_reason_codes": list(record["risk_reason_codes"]),
                "requested_quantity": record["requested_quantity"],
                "risk_allowed_quantity": record["risk_allowed_quantity"],
                "outcome": record["outcome"]})
    stats["reason_code_counts"] = dict(sorted(stats["reason_code_counts"].items()))
    stats["reduced_quantity_total"] = round(stats["reduced_quantity_total"], 12)
    return stats


def gate_regime_attribution(decision_records, instrument_rows, masks):
    """Attribute every gate-evaluated decision to its record's regime set.

    Same overlapping-strata rule as ``pit_regime_attribution``: a decision
    contributes to every regime label on the record whose ``decision_ns`` it
    was emitted at (reference decisions are emitted exactly at record decision
    timestamps), so bucket totals are not a partition. This is a decision-count
    attribution only; no PnL is pooled here.
    """
    by_ns = {}
    for row in instrument_rows:
        by_ns.setdefault(row["decision_ns"], row)
    buckets = {}
    for record in decision_records:
        if record["risk_decision"] is None:
            continue
        row = by_ns.get(record["decision_ns"])
        regimes = (masks[row["id"]]["regimes"] or [realpath.REGIME_NONE]) if row is not None \
            else [realpath.REGIME_PRE_FIRST]
        for regime in regimes:
            bucket = buckets.setdefault(regime, {"evaluated_decisions": 0,
                                                 "allowed": 0, "reduced": 0, "blocked": 0})
            bucket["evaluated_decisions"] += 1
            bucket[{"allow": "allowed", "reduce": "reduced",
                    "block": "blocked"}[record["risk_decision"]]] += 1
    return buckets


# --------------------------------------------------------------------------
# Gated cell replay (fresh gate per replay; explicit double-run determinism)
# --------------------------------------------------------------------------
def run_gated_cell(path, policy_name, *, gate_factory=None, execution_policy=None,
                   contracts=None, initial_cash=realpath.INITIAL_CASH,
                   seed=realpath.DATA_SEED, replay_id="t13-cell"):
    """Replay one (fold, instrument, policy) cell twice with a fresh gate each.

    Returns the T12 cell field set plus gate analytics; ``byte_identical``
    compares the two independent full-receipt fingerprints.
    """
    factory = gate_factory or frozen_gate_factory()
    policy = execution_policy or realpath.declared_execution_policy()
    specs = tuple(contracts) if contracts is not None else backtest.default_contracts(
        path.instruments, margin_mode=policy.perps.margin_mode)
    decisions = realpath.reference_decisions(policy_name, path)
    receipts = [
        backtest.run_backtest(path, decisions=decisions, execution_policy=policy,
                              contracts=specs, initial_cash=initial_cash, seed=seed,
                              risk_gate=factory(), verify_determinism=False,
                              replay_id=replay_id)
        for _ in range(2)]
    byte_identical = receipts[0]["backtest_sha256"] == receipts[1]["backtest_sha256"]
    if not byte_identical:
        raise RiskGatePathError(
            f"gated replay is not byte-identical across fresh-gate replays: "
            f"{receipts[0]['backtest_sha256']} != {receipts[1]['backtest_sha256']}")
    receipt = receipts[0]
    ledger = receipt["ledger"]
    return {
        "policy": policy_name,
        "decision_count": len(decisions),
        "backtest_sha256": receipt["backtest_sha256"],
        "determinism": {"replays": 2, "fresh_gate_per_replay": True,
                        "byte_identical": byte_identical,
                        "backtest_sha256": [receipts[0]["backtest_sha256"],
                                            receipts[1]["backtest_sha256"]],
                        "note": ("two independent run_backtest calls, each with a freshly "
                                 "constructed RiskEngine; fingerprints are full-receipt "
                                 "sha256 over canonical JSON")},
        "byte_identical": byte_identical,
        "counts": receipt["counts"],
        "risk": receipt["risk"],
        "gate_stats": gate_decision_stats(receipt),
        "equity_curve": realpath._compact_curve(receipt["equity_curve"]),
        "equity_curve_fingerprint": receipt["keys"]["equity_curve"],
        "drawdown": receipt["drawdown"],
        "margin_usage": receipt["margin_usage"],
        "turnover": receipt["turnover"],
        "costs": receipt["costs"],
        "funding": {"payment_count": receipt["funding"]["payment_count"],
                    "cost": receipt["funding"]["cost"],
                    "by_asset": receipt["funding"]["by_asset"]},
        "liquidation_count": receipt["counts"]["liquidations"],
        "net_pnl_simulated_descriptive": ledger["net_pnl"],
        "final_equity": ledger["equity"],
        "per_regime_calendar_stress": receipt["per_regime"],
        "attribution": receipt["attribution"],
        "market_path": receipt["market_path"],
        "_decision_records": receipt["decisions"],
    }


def run_gated_fold(fold, fold_index, rows, groups, masks, regimes, *,
                   policies=realpath.REFERENCE_POLICIES, gate_factory=None,
                   seed=realpath.DATA_SEED, provenance=None):
    """Replay every (instrument, policy) cell of one fold's retained test set,
    gated."""
    by_id = {row["id"]: row for row in rows}
    test_ids = list(fold["retained"]["test"])
    test_window = fold["windows"]["test"]
    cells = []
    for group_name in ("primary", "holdout_instrument_A", "holdout_instrument_B"):
        for key in groups[group_name]:
            venue, asset_id = realpath.parse_instrument_key(key)
            path = realpath.build_cell_path(by_id, test_ids, asset_id, venue,
                                            regimes=regimes, asof_ns=test_window[1],
                                            seed=seed, provenance=provenance)
            instrument_rows = [by_id[rid] for rid in test_ids
                               if by_id[rid]["asset_id"] == asset_id
                               and by_id[rid]["venue"] == venue]
            strata = realpath.event_outcome_strata(instrument_rows, masks)
            for policy_name in policies:
                cell = run_gated_cell(path, policy_name, gate_factory=gate_factory,
                                      seed=seed,
                                      replay_id=f"t13-fold{fold_index}-{asset_id}-{policy_name}")
                decision_records = cell.pop("_decision_records")
                cell["instrument_key"] = key
                cell["instrument_group"] = group_name
                cell["retained_test_records"] = len(instrument_rows)
                cell["pit_regime_attribution"] = realpath.pit_regime_attribution(
                    cell["equity_curve"], instrument_rows, masks)
                cell["pit_gate_attribution"] = gate_regime_attribution(
                    decision_records, instrument_rows, masks)
                cell["event_outcome_strata"] = strata
                cells.append(cell)
    return {"fold": fold_index, "test_window_ns": test_window,
            "fold_audit_counts": fold["counts"],
            "fold_audit_exclusion_counts": fold["exclusion_counts"],
            "retained_test_records": len(test_ids), "cells": cells}


def summarize_gate(folds):
    """Descriptive decision-outcome counts per instrument and per regime.

    Counts only — never PnL — so this aggregation does not create a pooled
    headline. Cells contribute to every regime stratum their blocked/reduced
    decisions carried, matching the overlapping-strata rule.
    """
    per_instrument = {}
    per_regime = {}
    totals = {"evaluated_decisions": 0, "allowed_decisions": 0,
              "reduced_decisions": 0, "blocked_decisions": 0,
              "reason_code_counts": {}, "reduced_quantity_total": 0.0}
    for fold in folds:
        for cell in fold["cells"]:
            instrument = per_instrument.setdefault(
                cell["instrument_key"],
                {"instrument_group": cell["instrument_group"], "policies": {}})
            stats = cell["gate_stats"]
            bucket = instrument["policies"].setdefault(
                cell["policy"],
                {"folds": [], "evaluated_decisions": 0, "allowed_decisions": 0,
                 "reduced_decisions": 0, "blocked_decisions": 0,
                 "reason_code_counts": {}, "reduced_quantity_total": 0.0})
            bucket["folds"].append(fold["fold"])
            for key in ("evaluated_decisions", "allowed_decisions",
                        "reduced_decisions", "blocked_decisions"):
                bucket[key] += stats[key]
            bucket["reduced_quantity_total"] += stats["reduced_quantity_total"]
            for code, count in stats["reason_code_counts"].items():
                bucket["reason_code_counts"][code] = \
                    bucket["reason_code_counts"].get(code, 0) + count
            for key in ("evaluated_decisions", "allowed_decisions",
                        "reduced_decisions", "blocked_decisions"):
                totals[key] += stats[key]
            totals["reduced_quantity_total"] += stats["reduced_quantity_total"]
            for code, count in stats["reason_code_counts"].items():
                totals["reason_code_counts"][code] = \
                    totals["reason_code_counts"].get(code, 0) + count
            for regime, bucket in cell["pit_gate_attribution"].items():
                aggregate = per_regime.setdefault(
                    regime, {"evaluated_decisions": 0, "allowed": 0,
                             "reduced": 0, "blocked": 0})
                for key in aggregate:
                    aggregate[key] += bucket[key]
    for instrument in per_instrument.values():
        for bucket in instrument["policies"].values():
            bucket["folds"] = sorted(set(bucket["folds"]))
            bucket["reason_code_counts"] = dict(sorted(bucket["reason_code_counts"].items()))
            bucket["reduced_quantity_total"] = round(bucket["reduced_quantity_total"], 12)
        instrument["policies"] = dict(sorted(instrument["policies"].items()))
    totals["reason_code_counts"] = dict(sorted(totals["reason_code_counts"].items()))
    totals["reduced_quantity_total"] = round(totals["reduced_quantity_total"], 12)
    return {"note": ("decision-outcome counts under the frozen guard set; overlapping "
                     "regime strata are not a partition; no PnL is pooled"),
            "totals": totals,
            "per_instrument": dict(sorted(per_instrument.items())),
            "per_regime": dict(sorted(per_regime.items()))}


def run_gated_validation(rows, folds, r1_protocol, groups, *, seed=realpath.DATA_SEED,
                         policies=realpath.REFERENCE_POLICIES, gate_factory=None,
                         provenance=None):
    """All folds x instrument groups x reference policies, gated, via run_backtest."""
    for policy in policies:
        if policy not in realpath.REFERENCE_POLICIES:
            raise ValueError(f"undeclared policy {policy!r}; the probability interface "
                             "is fail-closed")
    masks = realpath.compute_regime_masks(rows)
    regimes = realpath.stress_regime_windows(r1_protocol)
    fold_reports = [run_gated_fold(fold, index, rows, groups, masks, regimes,
                                   policies=policies, gate_factory=gate_factory,
                                   seed=seed, provenance=provenance)
                    for index, fold in enumerate(folds)]
    return {"masks_sha256": sha256_bytes(canonical_json(masks).encode()),
            "mask_stats": realpath.mask_stats(masks),
            "gate_summary": summarize_gate(fold_reports),
            "folds": fold_reports}


def _identity(path):
    return file_identity(Path(path))


def main(argv=None):
    """Smoke entry: one gated cell through the harness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True,
                        help="R1 bound receipt used to load the frozen cohort")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--instrument-key", default="binance_um:linear:BTCUSDT-PERP")
    parser.add_argument("--policy", choices=realpath.REFERENCE_POLICIES,
                        default="unit_long_always")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    import financial_baselines_v1 as baselines
    r1_protocol, rows, folds = baselines.load_bound_cohort(args.receipt)
    groups = baselines.instrument_groups(rows)
    masks = realpath.compute_regime_masks(rows)
    regimes = realpath.stress_regime_windows(r1_protocol)
    venue, asset_id = realpath.parse_instrument_key(args.instrument_key)
    by_id = {row["id"]: row for row in rows}
    fold = folds[args.fold]
    path = realpath.build_cell_path(by_id, fold["retained"]["test"], asset_id, venue,
                                    regimes=regimes, asof_ns=fold["windows"]["test"][1],
                                    provenance={"bound_receipt": _identity(args.receipt)})
    cell = run_gated_cell(path, args.policy)
    decision_records = cell.pop("_decision_records")
    cell["instrument_key"] = args.instrument_key
    cell["fold"] = args.fold
    instrument_rows = realpath.instrument_rows_for(rows, asset_id, venue)
    cell["pit_gate_attribution"] = gate_regime_attribution(
        decision_records, instrument_rows, masks)
    out = {"schema_version": SCHEMA,
           "scope": "single-cell gated smoke run; not a validation receipt",
           "risk_gate": {"engine": GATE_NAME, "policy_version": GATE_POLICY_VERSION,
                         "limits": frozen_gate_factory()().limits.to_dict(),
                         "limits_source": FROZEN_PARAMETERS_PATH},
           "cell": cell}
    text = json.dumps(out, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(canonical_json({"policy": args.policy,
                          "evaluated": cell["gate_stats"]["evaluated_decisions"],
                          "blocked": cell["counts"]["risk_blocked_decisions"],
                          "reduced": cell["gate_stats"]["reduced_decisions"],
                          "byte_identical": cell["byte_identical"],
                          "backtest_sha256": cell["backtest_sha256"],
                          "synthetic": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
