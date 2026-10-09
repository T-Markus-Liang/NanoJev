#!/usr/bin/env python3
"""T71: regime-gated conditional policy replay on the real cohort.

The global ridge is temporally unstable (fold1 reverses), so this tests a
SPARSE CONDITIONAL policy instead of a global regressor — long only inside
the joint high-funding / high-basis (/ high-vol) regime found in T66/T70:

  tri_gate:     funding_pct>=0.80 AND basis_pct>=0.66 AND rv_pct>=0.66
  double_gate:  funding_pct>=0.80 AND basis_pct>=0.66   (vol gate ablation)

  all pct:     trailing-180 mid-rank of the feature's own history, per asset
               (PIT-safe: window i-180:i excludes the decision bar)
  entry/exit:  mark close of bar i -> mark close of bar i+5 (hold 5 bars)
  overlap:     non-overlapping per asset — gate hits inside a hold are skipped
  costs:       taker 5bps per side; funding accrual per held day
               = direction * last_funding_rate * (24/funding_interval_hours)
               (fallback 8h -> 3 settlements/day; approximation flagged)

Metrics per variant: pooled + per-asset + per frozen test fold
(research/financial_r1_pit_validator_core_v2.json folds[].test, half-open
[start,end) on entry decision_ns): n, mean/median net bps, win rate, per-trade
equity max drawdown, Sharpe-per-trade, total net bps. Comparison vs always-long
matched cadence in every slice. Trade n will be small — that is the point of
the measurement (is the tri-gate too sparse to be useful?).

Measurement only — no fitting, no orders, no profitability claims.
"""

import argparse
import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_v1/records.jsonl"
FOLDS_PATH = ROOT / "research/financial_r1_pit_validator_core_v2.json"
OUT = ROOT / "results/financial_signal_conditional_v1.json"
LOOKBACK, HOLD = 180, 5
FEE_BPS = 5.0
FALLBACK_SETTLEMENTS_PER_DAY = 3  # 8h funding interval fallback

GATES = {
    "tri_gate": {"funding_pct": 0.80, "basis_pct": 0.66, "rv_pct": 0.66},
    "double_gate": {"funding_pct": 0.80, "basis_pct": 0.66},
}


def mid_rank_pct(window, x):
    """Mid-rank percentile of x vs trailing window (ties count half)."""
    return (sum(1 for w in window if w < x)
            + 0.5 * sum(1 for w in window if w == x)) / len(window)


def max_dd(equity):
    peak, dd, cur = 1.0, 0.0, 1.0
    for g in equity:
        cur *= 1 + g
        peak = max(peak, cur)
        dd = min(dd, cur / peak - 1)
    return round(dd * 1e4, 1)  # bps


def summarize(trades, name):
    if not trades:
        return {"name": name, "n": 0}
    gross = [t["gross_bps"] for t in trades]
    net = [t["net_bps"] for t in trades]
    wins = sum(1 for x in net if x > 0)
    mean = statistics.mean(net)
    sd = statistics.pstdev(net) or 1e-9
    return {"name": name, "n": len(trades),
            "mean_gross_bps": round(statistics.mean(gross), 1),
            "mean_net_bps": round(mean, 1),
            "median_net_bps": round(statistics.median(net), 1),
            "win_rate": round(wins / len(trades), 3),
            "sharpe_per_trade": round(mean / sd, 3),
            "max_drawdown_bps": max_dd([x / 1e4 for x in net]),
            "total_net_bps": round(sum(net), 0)}


def fold_index(test_windows, decision_ns):
    """Half-open [start, end) membership in frozen test windows; -1 if none."""
    for k, (lo, hi) in enumerate(test_windows):
        if lo <= decision_ns < hi:
            return k
    return -1


def replay_asset(rs, gate):
    """Non-overlapping gated long replay for one asset. Returns trade list."""
    fund = [x["features"]["last_funding_rate"]["value"] for x in rs]
    interval = [x["features"]["funding_interval_hours"]["value"] for x in rs]
    basis = [x["features"]["mark_index_basis_bps"]["value"] for x in rs]
    rv = [x["features"]["realized_vol_24bar"]["value"] for x in rs]
    mark = [x["features"]["mark_price"]["value"] for x in rs]
    ns = [x["decision_ns"] for x in rs]
    trades = []
    i, last_eligible = LOOKBACK, len(rs) - HOLD - 1
    while i <= last_eligible:
        pcts = {"funding_pct": mid_rank_pct(fund[i - LOOKBACK:i], fund[i]),
                "basis_pct": mid_rank_pct(basis[i - LOOKBACK:i], basis[i]),
                "rv_pct": mid_rank_pct(rv[i - LOOKBACK:i], rv[i])}
        if not all(pcts[k] >= thr for k, thr in gate.items()):
            i += 1
            continue
        entry, exit_ = mark[i], mark[i + HOLD]
        if entry <= 0 or exit_ <= 0:
            i += HOLD
            continue
        gross = (exit_ / entry - 1) * 1e4
        paid = 0.0
        for j in range(i + 1, i + HOLD + 1):
            per_day = (24.0 / interval[j]) if interval[j] > 0 \
                else FALLBACK_SETTLEMENTS_PER_DAY
            paid += fund[j] * per_day * 1e4  # direction = +1 (long only)
        trades.append({"i": i, "decision_ns": ns[i],
                       "gross_bps": round(gross, 1),
                       "funding_bps": round(paid, 1),
                       "net_bps": round(gross - 2 * FEE_BPS - paid, 1),
                       **{k: round(v, 3) for k, v in pcts.items()}})
        i += HOLD  # non-overlapping
    return trades


def run():
    records = [json.loads(l) for l in COHORT.read_text().splitlines()
               if l.strip()]
    by_asset = defaultdict(list)
    for r in records:
        by_asset[r["asset_id"]].append(r)
    for rs in by_asset.values():
        rs.sort(key=lambda x: x["decision_ns"])

    test_windows = [tuple(f["test"]) for f in
                    json.loads(FOLDS_PATH.read_text())["folds"]]

    variant_trades = {name: [] for name in GATES}
    asset_of = {}
    always_long = []
    for asset, rs in sorted(by_asset.items()):
        for name, gate in GATES.items():
            for t in replay_asset(rs, gate):
                t["asset"] = asset
                variant_trades[name].append(t)
                asset_of[id(t)] = asset
        fund = [x["features"]["last_funding_rate"]["value"] for x in rs]
        interval = [x["features"]["funding_interval_hours"]["value"] for x in rs]
        mark = [x["features"]["mark_price"]["value"] for x in rs]
        last_eligible = len(rs) - HOLD - 1
        for j in range(LOOKBACK, last_eligible, HOLD):
            if mark[j] > 0 and mark[j + HOLD] > 0:
                g = (mark[j + HOLD] / mark[j] - 1) * 1e4
                paid = sum(fund[k] * ((24.0 / interval[k])
                                      if interval[k] > 0 else
                                      FALLBACK_SETTLEMENTS_PER_DAY) * 1e4
                           for k in range(j + 1, j + HOLD + 1))
                always_long.append({"asset": asset, "decision_ns": rs[j]["decision_ns"],
                                    "gross_bps": round(g, 1),
                                    "funding_bps": round(paid, 1),
                                    "net_bps": round(g - 2 * FEE_BPS - paid, 1)})

    def slice_report(trades, al_trades):
        rep = summarize(trades, "signal")
        rep["vs_always_long_mean_net_bps"] = None
        al = summarize(al_trades, "always_long")
        if trades and al_trades:
            rep["vs_always_long_mean_net_bps"] = round(
                rep["mean_net_bps"] - al["mean_net_bps"], 1)
        rep["always_long"] = al
        return rep

    results = {}
    for name, trades in variant_trades.items():
        per_asset = {a: summarize([t for t in trades if t["asset"] == a], a)
                     for a in sorted(by_asset)}
        per_fold = {}
        for k, (lo, hi) in enumerate(test_windows):
            ft = [t for t in trades if lo <= t["decision_ns"] < hi]
            fa = [t for t in always_long if lo <= t["decision_ns"] < hi]
            per_fold[f"fold{k+1}_test"] = {
                "window_ns": [lo, hi],
                **slice_report(ft, fa)}
        in_test = sum(1 for t in trades
                      if fold_index(test_windows, t["decision_ns"]) >= 0)
        results[name] = {
            "gate": GATES[name],
            "pooled": slice_report(trades, always_long),
            "per_asset": per_asset,
            "per_test_fold": per_fold,
            "trades_in_test_folds": in_test,
            "trades_outside_test_folds": len(trades) - in_test}

    # honest verdict
    verdict = {}
    tg = results["tri_gate"]["pooled"]
    dg = results["double_gate"]["pooled"]
    al = results["tri_gate"]["pooled"]["always_long"]
    verdict["tri_gate_n"] = tg["n"]
    verdict["double_gate_n"] = dg["n"]
    verdict["beats_always_long_per_trade"] = {
        "tri_gate": (tg.get("vs_always_long_mean_net_bps") or 0) > 0,
        "double_gate": (dg.get("vs_always_long_mean_net_bps") or 0) > 0,
        "always_long_mean_net_bps": al.get("mean_net_bps")}
    fold_means = {name: [results[name]["per_test_fold"][f]["mean_net_bps"]
                         for f in results[name]["per_test_fold"]
                         if results[name]["per_test_fold"][f]["n"] > 0]
                  for name in GATES}
    fold_ns = {name: {f: results[name]["per_test_fold"][f]["n"]
                      for f in results[name]["per_test_fold"]}
               for name in GATES}
    verdict["test_fold_mean_net_bps"] = fold_means
    verdict["test_fold_n"] = fold_ns
    verdict["note"] = (
        "Per-trade mean advantage vs always-long does not imply a usable edge: "
        "gate trades are sparse and clustered, fold counts are tiny, and no "
        "multiple-testing correction can rescue n of this size. Treat as a "
        "regime description, not a deployable signal.")

    return {"schema_version": "nanojev-financial-signal-conditional-v1",
            "status": "measurement_complete",
            "scope": "regime-gated conditional replay on real cohort; "
                     "approximate funding accrual; no market impact model; "
                     "sparse trades; NOT a tradability claim",
            "parameters": {"fee_bps_per_side": FEE_BPS, "hold_bars": HOLD,
                           "lookback": LOOKBACK, "non_overlapping": True,
                           "gates": GATES,
                           "fold_windows": "folds[].test half-open [start,end)"},
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "results": results,
            "verdict": verdict}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    receipt = run()
    blob = json.dumps(receipt, indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(blob)
    print(json.dumps({"output": str(args.output),
                      "sha256": hashlib.sha256(blob.encode()).hexdigest()},
                     indent=2))


if __name__ == "__main__":
    main()
