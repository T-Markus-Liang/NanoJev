#!/usr/bin/env python3
"""T71: funding-following conditional policy through the REAL harness + risk gate.

Same machinery as T12/T13: real bound cohort -> build_cell_path ->
unmodified financial_backtest_v1.run_backtest with the frozen RiskEngine.
Only the decision stream is new: target-position LONG at bars where the
record's last_funding_rate is in its trailing-180 top quintile (pct>=0.80),
FLAT otherwise. Compared against the reference unit_long_always cell.

Honesty: same caveats as T12 — assumed costs on a collapsed mark book, daily
decision bars, simulated ledger; not a profitability claim. Two fresh-gate
replays per cell must be byte-identical.
"""

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import financial_backtest_v1 as backtest
import financial_baselines_v1 as baselines
import financial_real_data_path_v1 as realpath
import financial_real_data_risk_v1 as riskrun
from financial_simulator_v1 import (ACTION_FLAT, ACTION_LONG, Decision)

RECEIPT = ROOT / "results/financial_pit_r1_bound_receipt_20260920_v5.json"
OUT = ROOT / "results/financial_signal_gated_run_v1.json"
LOOKBACK = 180
PCT = 0.80


def funding_pcts(rows):
    """(asset_id, mark available_ns) -> trailing-180 mid-rank funding pct."""
    by_asset = defaultdict(list)
    for r in rows:
        by_asset[r["asset_id"]].append(r)
    out = {}
    for rs in by_asset.values():
        rs.sort(key=lambda x: x["decision_ns"])
        fund = [x["features"]["last_funding_rate"]["value"] for x in rs]
        for i, r in enumerate(rs):
            akey = (r["asset_id"],
                    int(r["features"]["mark_price"]["available_ns"]))
            if i < LOOKBACK:
                out[akey] = None
                continue
            w = fund[i - LOOKBACK:i]
            out[akey] = (sum(1 for v in w if v < fund[i])
                         + 0.5 * sum(1 for v in w if v == fund[i])) / LOOKBACK
    return out


def signal_decisions(path, pcts):
    decisions = []
    for q in sorted(path.quotes, key=lambda q: (q.available_ns, q.asset_id)):
        pct = pcts.get((q.asset_id, q.available_ns))
        action = ACTION_LONG if (pct is not None and pct >= PCT) else ACTION_FLAT
        decisions.append(Decision(
            decision_id=f"funding80::{q.asset_id}::{q.venue}::{q.available_ns}",
            asset_id=q.asset_id, venue=q.venue, decision_ns=q.available_ns,
            action=action,
            quantity=realpath.UNIT_QUANTITY if action == ACTION_LONG else 0.0,
            reason=(f"t67_funding_pct={pct:.3f}"
                    if pct is not None else "warmup")))
    return decisions


def gated_replay(path, decisions, gate_factory, seed):
    policy = realpath.declared_execution_policy()
    specs = backtest.default_contracts(
        path.instruments, margin_mode=policy.perps.margin_mode)
    receipts = [backtest.run_backtest(
        path, decisions=decisions, execution_policy=policy, contracts=specs,
        initial_cash=realpath.INITIAL_CASH, seed=seed,
        risk_gate=gate_factory(), verify_determinism=False,
        replay_id="t71-cell") for _ in range(2)]
    if receipts[0]["backtest_sha256"] != receipts[1]["backtest_sha256"]:
        raise RuntimeError("non-deterministic gated replay")
    r = receipts[0]
    return {"sha": r["backtest_sha256"], "counts": r["counts"],
            "gate": riskrun.gate_decision_stats(r),
            "net_pnl": r["ledger"]["net_pnl"], "equity": r["ledger"]["equity"],
            "drawdown": r["drawdown"], "funding_cost": r["funding"]["cost"],
            "fees": r["costs"].get("fees"), "liquidations":
            r["counts"]["liquidations"]}


def run():
    r1_protocol, rows, folds = baselines.load_bound_cohort(RECEIPT)
    pcts = funding_pcts(rows)
    by_id = {r["id"]: r for r in rows}
    regimes = realpath.stress_regime_windows(r1_protocol)
    gate_factory = riskrun.frozen_gate_factory()
    cells = {}
    assets = sorted({(r["venue"], r["asset_id"]) for r in rows})
    for fi, fold in enumerate(folds):
        for venue, asset in assets:
            try:
                path = realpath.build_cell_path(
                    by_id, fold["retained"]["test"], asset, venue,
                    regimes=regimes, asof_ns=fold["windows"]["test"][1],
                    provenance={"bound_receipt": str(RECEIPT)})
            except Exception as e:
                cells[f"f{fi}:{asset}"] = {"error": str(e)[:120]}
                continue
            decs = signal_decisions(path, pcts)
            ref = riskrun.run_gated_cell(path, "unit_long_always",
                                         gate_factory=gate_factory)
            sig = gated_replay(path, decs, gate_factory, realpath.DATA_SEED)
            cells[f"f{fi}:{asset}"] = {
                "ticks": path.ticks,
                "signal_decisions": len(decs),
                "signal": sig,
                "always_long_net_pnl": ref["net_pnl_simulated_descriptive"],
                "always_long_sha": ref["backtest_sha256"]}
    return {"schema_version": "nanojev-financial-signal-gated-run-v1",
            "status": "measurement_complete",
            "scope": "conditional policy through real harness+frozen gate; "
                    "simulated ledger on real mark path; not a tradability claim",
            "parameters": {"lookback": LOOKBACK, "funding_pct_threshold": PCT,
                           "quantity": realpath.UNIT_QUANTITY},
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cells": cells}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    receipt = run()
    blob = json.dumps(receipt, indent=2, ensure_ascii=False,
                      default=str) + "\n"
    args.output.write_text(blob)
    print(json.dumps({"output": str(args.output),
                      "sha256": hashlib.sha256(blob.encode()).hexdigest()},
                     indent=2))


if __name__ == "__main__":
    main()
