#!/usr/bin/env python3
"""T12 (B5) real-data path into the pinned backtest harness.

This module adapts the frozen R1 point-in-time cohort
(``data/perp_pit_v2/records.jsonl``, bound by
``results/financial_pit_r1_bound_receipt_20260920_v5.json``) into a market path
that flows through the **unmodified** ``financial_backtest_v1.run_backtest``
machinery: same replay, same receipt construction, same determinism check and
the same attribution/conservation invariants. No bespoke replay engine is added
and no pinned file is edited (the R1 freeze pins ``financial_backtest_v1.py``
by sha256, so an in-place extension would silently change a recorded
fingerprint; see research/financial_validation_protocol_v1.json ``why_new_module``).

What this is **not**: a profitability, edge, or promotion result. Execution
costs are frozen *assumed* parameters applied to a collapsed bid=ask=mark book;
the gross 25bps/1d mark-price event is not a tradable return. T11's honest
outcome (no dev-selected candidate beat base rate; fold 2's pick was
significantly worse) is part of the bound context.
"""
from __future__ import annotations

import argparse
from bisect import insort, bisect_right
from dataclasses import dataclass, field
import json
from pathlib import Path

from benchmark_nanojev_v2 import canonical_json, file_identity, percentile, sha256_bytes
import financial_backtest_v1 as backtest
import financial_baselines_v1 as baselines
from financial_simulator_v1 import (
    ACTION_LONG, ACTION_SHORT, FUNDING_SOURCE_QUOTE_FIELD, MARGIN_CROSS,
    POSITION_ONE_WAY, PRICE_SOURCE_MARK,
    AccountPolicy, CapacityPolicy, Decision, ExecutionPolicy, FeePolicy, FillPolicy,
    LatencyPolicy, MarkPolicy, PerpPolicy, Quote, SlippagePolicy, SpreadPolicy,
)

SCHEMA = "nanojev-financial-real-data-path-v1"
DAY_NS = 86_400_000_000_000
HOUR_NS = 3_600_000_000_000
DATA_SEED = 20_260_919
INITIAL_CASH = 100_000.0
# Declared real-path value: covers the observed maximum 6-day inter-decision gap
# so an order can fill at the next available bar; expiry is otherwise reported.
ORDER_TTL_NS = 8 * DAY_NS
FUNDING_INTERVAL_NS = 8 * HOUR_NS
FUNDING_ANCHOR_NS = 0
UNIT_QUANTITY = 1.0

REGIME_HIGH_VOL = "regime_high_realized_volatility"
REGIME_FUNDING_EXTREME = "regime_funding_extreme"
REGIME_BASIS_BLOWOUT = "regime_basis_blowout"
REGIME_THIN_LIQUIDITY = "regime_thin_liquidity"
REGIME_IDS = (REGIME_HIGH_VOL, REGIME_FUNDING_EXTREME, REGIME_BASIS_BLOWOUT,
              REGIME_THIN_LIQUIDITY)
REGIME_NONE = "no_declared_regime"
REGIME_PRE_FIRST = "before_first_cohort_record"
TOP_DECILE = 0.9
BOTTOM_QUINTILE = 0.2
MASK_MIN_HISTORY = 30
THIN_LOOKBACK_BARS = 30

REFERENCE_POLICIES = ("no_trade", "unit_long_always", "unit_short_always")


class RealDataPathError(RuntimeError):
    """Contract violation in the real-data path."""


# --------------------------------------------------------------------------
# Records -> simulator quotes
# --------------------------------------------------------------------------
def _feature_value(row, name):
    try:
        return float(row["features"][name]["value"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RealDataPathError(f"record {row.get('id')} lacks numeric feature {name}") from exc


def records_to_quotes(rows):
    """One Quote per retained PIT record; the book is collapsed to the mark.

    bid = ask = mark_price (the cohort carries no bid/ask series; all execution
    cost lives in the declared policy). volume = quote_volume/mark_price, a
    declared base-coin conversion used only by the capacity model.
    funding_rate = last settled funding rate at or before the decision.
    """
    quotes = []
    for row in sorted(rows, key=lambda r: (r["decision_ns"], r["id"])):
        mark = _feature_value(row, "mark_price")
        if mark <= 0:
            raise RealDataPathError(f"record {row['id']} has non-positive mark price")
        volume_base = _feature_value(row, "quote_volume") / mark
        funding = _feature_value(row, "last_funding_rate")
        mark_feature = row["features"]["mark_price"]
        event_ns = int(mark_feature["event_ns"])
        available_ns = int(mark_feature["available_ns"])
        if available_ns > int(row["decision_ns"]):
            raise RealDataPathError("mark feature would be unavailable at decision time")
        quotes.append(Quote(asset_id=row["asset_id"], venue=row["venue"],
                            event_ns=event_ns, available_ns=available_ns,
                            bid=mark, ask=mark, volume=volume_base,
                            source_id="r1_frozen_cohort_record",
                            version="t12-real-cohort-v1",
                            mark_price=mark, funding_rate=funding))
    return tuple(quotes)


# --------------------------------------------------------------------------
# Point-in-time regime masks (expanding quantiles; strictly prior history)
# --------------------------------------------------------------------------
def compute_regime_masks(rows):
    """Per-record overlapping regime strata under the frozen T12 rules.

    Every mask evaluates its statistic against the expanding pooled history of
    records with ``decision_ns`` strictly earlier than the record's own
    timestamp; same-timestamp records never see each other. Returns
    ``{record_id: {"regimes": sorted list, "evaluable": sorted list}}`` where
    ``evaluable`` names the masks that had enough history to vote.
    """
    ordered = sorted(rows, key=lambda r: (r["decision_ns"], r["id"]))
    history = {REGIME_HIGH_VOL: [], REGIME_FUNDING_EXTREME: [], REGIME_BASIS_BLOWOUT: [],
               "thin_median": []}
    own_volumes = {}
    masks = {}
    index = 0
    while index < len(ordered):
        end = index
        while end < len(ordered) and ordered[end]["decision_ns"] == ordered[index]["decision_ns"]:
            end += 1
        group = ordered[index:end]
        staged = []
        for row in group:
            features = row["features"]
            vol = _feature_value(row, "realized_vol_24bar")
            funding = abs(_feature_value(row, "last_funding_rate"))
            basis = abs(_feature_value(row, "mark_index_basis_bps"))
            regimes, evaluable = [], []
            for regime_id, value in ((REGIME_HIGH_VOL, vol), (REGIME_FUNDING_EXTREME, funding),
                                     (REGIME_BASIS_BLOWOUT, basis)):
                past = history[regime_id]
                if len(past) >= MASK_MIN_HISTORY:
                    evaluable.append(regime_id)
                    if value > percentile(past, TOP_DECILE):
                        regimes.append(regime_id)
            key = (row["asset_id"], row["venue"])
            own = own_volumes.get(key, ())
            thin_median = None
            if len(own) >= THIN_LOOKBACK_BARS:
                thin_median = percentile(list(own[-THIN_LOOKBACK_BARS:]), 0.5)
                past_medians = history["thin_median"]
                if len(past_medians) >= MASK_MIN_HISTORY:
                    evaluable.append(REGIME_THIN_LIQUIDITY)
                    if thin_median < percentile(past_medians, BOTTOM_QUINTILE):
                        regimes.append(REGIME_THIN_LIQUIDITY)
            masks[row["id"]] = {"regimes": sorted(regimes), "evaluable": sorted(evaluable)}
            staged.append((row, vol, funding, basis, thin_median))
        for row, vol, funding, basis, thin_median in staged:
            insort(history[REGIME_HIGH_VOL], vol)
            insort(history[REGIME_FUNDING_EXTREME], funding)
            insort(history[REGIME_BASIS_BLOWOUT], basis)
            if thin_median is not None:
                insort(history["thin_median"], thin_median)
            own_volumes.setdefault((row["asset_id"], row["venue"]), []).append(
                _feature_value(row, "quote_volume"))
        index = end
    return masks


def mask_stats(masks):
    stats = {regime_id: {"evaluable": 0, "member": 0} for regime_id in REGIME_IDS}
    for entry in masks.values():
        for regime_id in REGIME_IDS:
            if regime_id in entry["evaluable"]:
                stats[regime_id]["evaluable"] += 1
            if regime_id in entry["regimes"]:
                stats[regime_id]["member"] += 1
    return stats


# --------------------------------------------------------------------------
# Real cohort market path (flows through the unmodified run_backtest)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RealCohortPath(backtest.SyntheticMarketPath):
    """A SyntheticMarketPath populated with real frozen-cohort records.

    ``ticks`` is the quote count of this path and ``tick_ns`` is the declared
    daily-bar interval (86400e9 ns); both are descriptive metadata for the
    receipt, not a claim that the path is evenly spaced. ``seed`` is the frozen
    data seed (it only feeds degenerate fill/reject draws under the declared
    policy). ``to_summary`` marks the path as real, never synthetic.
    """

    provenance: dict = field(default_factory=dict)

    def to_summary(self):
        return {"description": self.description,
                "synthetic": False,
                "data_class": "real_frozen_r1_pit_cohort",
                "quote_construction": ("bid=ask=record mark_price; volume=quote_volume/mark_price "
                                       "(declared base-coin conversion); funding_rate=record "
                                       "last_funding_rate; last_price unset (falls back to mark)"),
                "seed": self.seed, "ticks": self.ticks, "tick_ns": self.tick_ns,
                "asof_ns": self.asof_ns,
                "instrument_count": len(self.instruments),
                "instruments": [{"asset_id": asset_id, "venue": venue}
                                for asset_id, venue in self.instruments],
                "quote_count": len(self.quotes),
                "regimes": [window.to_dict() for window in self.regimes],
                "provenance": dict(self.provenance)}


def stress_regime_windows(r1_protocol):
    """The five frozen R1 calendar stress windows as harness RegimeWindows."""
    return tuple(backtest.RegimeWindow(label=w["label"], start_ns=w["start_ns"],
                                       end_ns=w["end_ns"])
                 for w in baselines.calendar_windows(r1_protocol))


def build_cell_path(rows_by_id, test_ids, asset_id, venue, *, regimes, asof_ns,
                    seed=DATA_SEED, provenance=None):
    """Single-instrument real path over the fold's retained test records."""
    selected = [rows_by_id[rid] for rid in test_ids
                if rows_by_id[rid]["asset_id"] == asset_id
                and rows_by_id[rid]["venue"] == venue]
    if not selected:
        raise RealDataPathError(f"no retained test records for {venue}:{asset_id}")
    quotes = records_to_quotes(selected)
    return RealCohortPath(quotes=quotes, regimes=tuple(regimes), asof_ns=int(asof_ns),
                          instruments=((asset_id, venue),), ticks=len(quotes),
                          tick_ns=DAY_NS, seed=seed,
                          description=("real R1 cohort retained test decision bars "
                                       "(daily mark bars; spacing 2-6 days as recorded)"),
                          provenance=dict(provenance or {}))


def parse_instrument_key(key):
    venue, contract, asset_id = key.split(":", 2)
    if contract != "linear":
        raise RealDataPathError(f"instrument key {key!r} is not a linear contract")
    return venue, asset_id


def reference_decisions(policy, path):
    """Deterministic reference-policy decision stream for one cell path."""
    if policy not in REFERENCE_POLICIES:
        raise ValueError(f"undeclared policy {policy!r}; the probability interface is fail-closed")
    if policy == "no_trade":
        return []
    action = ACTION_LONG if policy == "unit_long_always" else ACTION_SHORT
    return [Decision(decision_id=f"{policy}::{quote.asset_id}::{quote.venue}::{quote.available_ns}",
                     asset_id=quote.asset_id, venue=quote.venue,
                     decision_ns=quote.available_ns, action=action,
                     quantity=UNIT_QUANTITY, reason=f"t12_reference_{policy}")
            for quote in sorted(path.quotes, key=lambda q: (q.available_ns, q.asset_id))]


def declared_execution_policy():
    """Frozen R1 assumption values plus the declared real-path deltas."""
    return ExecutionPolicy(
        fees=FeePolicy(fee_bps=1.0, fixed_fee_per_fill=0.0, minimum_fee=0.0),
        spread=SpreadPolicy(half_spread_bps=1.0, fixed_half_spread_price=0.0),
        slippage=SlippagePolicy(fixed_bps=0.0, impact_bps_at_full_capacity=0.0),
        capacity=CapacityPolicy(max_participation_fraction=0.1,
                                max_order_quantity=1.0e9,
                                max_order_notional=1.0e12,
                                on_order_exceeds_limit="clamp"),
        fills=FillPolicy(reject_probability=0.0, fill_probability=1.0,
                         minimum_fill_quantity=1.0e-9, require_inventory_for_sell=True,
                         order_time_to_live_ns=ORDER_TTL_NS),
        latency=LatencyPolicy(decision_to_order_ns=1_000_000, order_to_ack_ns=500_000,
                              ack_to_execution_ns=1_000_000),
        marks=MarkPolicy(fallback="cost_basis", price_source=PRICE_SOURCE_MARK),
        account=AccountPolicy(require_sufficient_cash=True, on_insufficient_cash="reduce",
                              cash_buffer_fraction=0.0),
        perps=PerpPolicy(allow_short=True, default_leverage=5.0, max_leverage=20.0,
                         margin_mode=MARGIN_CROSS, position_mode=POSITION_ONE_WAY,
                         maintenance_margin_rate=0.005,
                         funding_interval_ns=FUNDING_INTERVAL_NS,
                         funding_anchor_ns=FUNDING_ANCHOR_NS,
                         funding_rate_source=FUNDING_SOURCE_QUOTE_FIELD,
                         default_funding_rate=0.0, funding_rates=(),
                         funding_rate_schedule=(), liquidation_fee_bps=0.0,
                         trace_perp_curve=True))


# --------------------------------------------------------------------------
# Per-cell analytics extracted from the harness receipt
# --------------------------------------------------------------------------
COMPACT_POINT_FIELDS = ("ns", "equity", "cash", "allocated_margin",
                        "maintenance_margin_required", "margin_ratio",
                        "liquidation_distance_fraction", "unrealized_pnl",
                        "gross_position_notional", "net_position_notional",
                        "funding_cost_to_date", "fees_to_date", "turnover_to_date",
                        "fill_count", "liquidation_count")


def _compact_curve(points):
    return [{key: point.get(key) for key in COMPACT_POINT_FIELDS} for point in points]


def pit_regime_attribution(points, instrument_rows, masks):
    """Attribute equity segments to the regime set of the midpoint-active record.

    Overlapping strata: a segment contributes to every regime label carried by
    the active record, so bucket sums are not a partition of total PnL. Records
    with no regime label fall into ``no_declared_regime``; segments before the
    first cohort record fall into ``before_first_cohort_record``.
    """
    instrument_rows = sorted(instrument_rows, key=lambda r: (r["decision_ns"], r["id"]))
    decisions = [row["decision_ns"] for row in instrument_rows]
    buckets = {}
    for previous, current in zip(points, points[1:]):
        midpoint = (previous["ns"] + current["ns"]) // 2
        index = bisect_right(decisions, midpoint) - 1
        if index < 0:
            regimes = [REGIME_PRE_FIRST]
        else:
            regimes = masks[instrument_rows[index]["id"]]["regimes"] or [REGIME_NONE]
        delta = current["equity"] - previous["equity"]
        for regime in regimes:
            bucket = buckets.setdefault(regime, {"segments": 0, "equity_change": 0.0,
                                                 "funding_cost": 0.0, "fees": 0.0,
                                                 "turnover_notional": 0.0, "fills": 0,
                                                 "liquidations": 0})
            bucket["segments"] += 1
            bucket["equity_change"] += delta
            bucket["funding_cost"] += current["funding_cost_to_date"] - previous["funding_cost_to_date"]
            bucket["fees"] += current["fees_to_date"] - previous["fees_to_date"]
            bucket["turnover_notional"] += (current["turnover_to_date"]
                                            - previous["turnover_to_date"])
            bucket["fills"] += current["fill_count"] - previous["fill_count"]
            bucket["liquidations"] += current["liquidation_count"] - previous["liquidation_count"]
    return buckets


def event_outcome_strata(instrument_rows, masks):
    """Descriptive label-outcome strata per regime (not a probability metric)."""
    strata = {}
    for row in instrument_rows:
        entry = masks[row["id"]]
        for regime_id in REGIME_IDS:
            if regime_id not in entry["evaluable"]:
                continue
            bucket = strata.setdefault(regime_id, {"evaluable_records": 0,
                                                   "member_records": 0,
                                                   "member_positive": 0,
                                                   "nonmember_records": 0,
                                                   "nonmember_positive": 0})
            bucket["evaluable_records"] += 1
            if regime_id in entry["regimes"]:
                bucket["member_records"] += 1
                bucket["member_positive"] += int(row["label"]["outcome"])
            else:
                bucket["nonmember_records"] += 1
                bucket["nonmember_positive"] += int(row["label"]["outcome"])
    for bucket in strata.values():
        bucket["member_positive_rate"] = (bucket["member_positive"] / bucket["member_records"]
                                          if bucket["member_records"] else None)
        bucket["nonmember_positive_rate"] = (bucket["nonmember_positive"]
                                             / bucket["nonmember_records"]
                                             if bucket["nonmember_records"] else None)
    return strata


def run_cell(path, policy_name, *, execution_policy=None, contracts=None,
             initial_cash=INITIAL_CASH, seed=DATA_SEED, replay_id="t12-cell"):
    """Replay one (fold, instrument, policy) cell through run_backtest."""
    policy = execution_policy or declared_execution_policy()
    specs = tuple(contracts) if contracts is not None else backtest.default_contracts(
        path.instruments, margin_mode=policy.perps.margin_mode)
    decisions = reference_decisions(policy_name, path)
    receipt = backtest.run_backtest(path, decisions=decisions, execution_policy=policy,
                                    contracts=specs, initial_cash=initial_cash, seed=seed,
                                    risk_gate=None, verify_determinism=True,
                                    replay_id=replay_id)
    ledger = receipt["ledger"]
    return {
        "policy": policy_name,
        "decision_count": len(decisions),
        "backtest_sha256": receipt["backtest_sha256"],
        "byte_identical": receipt["determinism"]["byte_identical"],
        "counts": receipt["counts"],
        "risk": receipt["risk"],
        "equity_curve": _compact_curve(receipt["equity_curve"]),
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
    }


def instrument_rows_for(rows, asset_id, venue):
    return sorted((row for row in rows
                   if row["asset_id"] == asset_id and row["venue"] == venue),
                  key=lambda r: (r["decision_ns"], r["id"]))


def run_fold(fold, fold_index, rows, groups, masks, regimes, *,
             policies=REFERENCE_POLICIES, seed=DATA_SEED, provenance=None):
    """Replay every (instrument, policy) cell of one fold's retained test set."""
    by_id = {row["id"]: row for row in rows}
    test_ids = list(fold["retained"]["test"])
    test_window = fold["windows"]["test"]
    cells = []
    for group_name in ("primary", "holdout_instrument_A", "holdout_instrument_B"):
        for key in groups[group_name]:
            venue, asset_id = parse_instrument_key(key)
            path = build_cell_path(by_id, test_ids, asset_id, venue, regimes=regimes,
                                   asof_ns=test_window[1], seed=seed,
                                   provenance=provenance)
            instrument_rows = [by_id[rid] for rid in test_ids
                               if by_id[rid]["asset_id"] == asset_id
                               and by_id[rid]["venue"] == venue]
            strata = event_outcome_strata(instrument_rows, masks)
            for policy_name in policies:
                cell = run_cell(path, policy_name, seed=seed,
                                replay_id=f"t12-fold{fold_index}-{asset_id}-{policy_name}")
                cell["instrument_key"] = key
                cell["instrument_group"] = group_name
                cell["retained_test_records"] = len(instrument_rows)
                cell["pit_regime_attribution"] = pit_regime_attribution(
                    cell["equity_curve"], instrument_rows, masks)
                cell["event_outcome_strata"] = strata
                cells.append(cell)
    return {"fold": fold_index, "test_window_ns": test_window,
            "fold_audit_counts": fold["counts"],
            "fold_audit_exclusion_counts": fold["exclusion_counts"],
            "retained_test_records": len(test_ids), "cells": cells}


def run_validation(rows, folds, r1_protocol, groups, *, seed=DATA_SEED,
                   policies=REFERENCE_POLICIES, provenance=None):
    """All folds x instrument groups x reference policies through run_backtest."""
    for policy in policies:
        if policy not in REFERENCE_POLICIES:
            raise ValueError(f"undeclared policy {policy!r}; the probability interface is fail-closed")
    masks = compute_regime_masks(rows)
    regimes = stress_regime_windows(r1_protocol)
    return {"masks_sha256": sha256_bytes(canonical_json(masks).encode()),
            "mask_stats": mask_stats(masks),
            "folds": [run_fold(fold, index, rows, groups, masks, regimes,
                               policies=policies, seed=seed, provenance=provenance)
                      for index, fold in enumerate(folds)]}


def _identity(path):
    return file_identity(Path(path))


def main(argv=None):
    """Smoke entry: real cohort through the harness for one fold/instrument/policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True,
                        help="R1 bound receipt used to load the frozen cohort")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--instrument-key", default="binance_um:linear:SOLUSDT-PERP")
    parser.add_argument("--policy", choices=REFERENCE_POLICIES, default="unit_long_always")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    r1_protocol, rows, folds = baselines.load_bound_cohort(args.receipt)
    groups = baselines.instrument_groups(rows)
    masks = compute_regime_masks(rows)
    regimes = stress_regime_windows(r1_protocol)
    venue, asset_id = parse_instrument_key(args.instrument_key)
    by_id = {row["id"]: row for row in rows}
    fold = folds[args.fold]
    path = build_cell_path(by_id, fold["retained"]["test"], asset_id, venue,
                           regimes=regimes, asof_ns=fold["windows"]["test"][1],
                           provenance={"bound_receipt": _identity(args.receipt)})
    cell = run_cell(path, args.policy)
    cell["instrument_key"] = args.instrument_key
    cell["fold"] = args.fold
    out = {"schema_version": SCHEMA, "scope": "single-cell smoke run; not a validation receipt",
           "cell": cell}
    text = json.dumps(out, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(canonical_json({"policy": args.policy, "fills": cell["counts"]["fills"],
                          "byte_identical": cell["byte_identical"],
                          "backtest_sha256": cell["backtest_sha256"],
                          "net_pnl_simulated_descriptive": cell["net_pnl_simulated_descriptive"],
                          "synthetic": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
