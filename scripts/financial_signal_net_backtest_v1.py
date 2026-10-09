#!/usr/bin/env python3
"""T67: net-of-cost accounting for the funding-following signal on the real cohort.

NOT a promotion or a tradability claim — an honest net accounting replay:

  signal:  funding_pct >= 0.80 at decision bar i  -> LONG, hold 5 bars
           funding_pct <= 0.20 at decision bar i  -> SHORT, hold 5 bars
  entry:   mark close of decision bar i (PIT-safe: signal uses trailing-180 pct)
  exit:    mark close of bar i+5
  costs:   taker fee 5bps per side (10bps round trip, Binance USD-M VIP0 ≈4-5bps)
  funding: long pays positive funding / receives negative; accrual per held day
           = that day's last_funding_rate x (24 / funding_interval_hours)
           (approximation: settled rate observed per decision bar, marked in receipt)
  overlap: non-overlapping per asset — a signal inside an open hold is skipped

Baselines: buy-and-hold per asset over the same eligible span; every-other-day
always-long overlapping trades for a matched-activity comparison.

Outputs per-asset + pooled: n trades, mean gross/net bps, win rate, equity
max drawdown, net Sharpe (per-trade annualized by trade count), vs baselines.
PIT-safe trailing windows only. Measurement only — no fitting, no orders.
"""

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_v1/records.jsonl"
OUT = ROOT / "results/financial_signal_net_backtest_v1.json"
LOOKBACK = 180
HOLD = 5
FEE_BPS = 5.0
FUNDING_SETTLEMENTS_PER_DAY = 3  # 8h intervals; computed from funding_interval_hours when present


def max_dd(equity):
    peak, dd = 1.0, 0.0
    for g in equity:
        peak = max(peak, peak * (1 + g))
    # recompute properly
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
    eq = [x / 1e4 for x in net]
    mean = statistics.mean(net)
    sd = statistics.pstdev(net) or 1e-9
    sharpe_per_trade = mean / sd
    return {"name": name, "n": len(trades),
            "mean_gross_bps": round(statistics.mean(gross), 1),
            "mean_net_bps": round(mean, 1),
            "median_net_bps": round(statistics.median(net), 1),
            "win_rate": round(wins / len(trades), 3),
            "sharpe_per_trade": round(sharpe_per_trade, 3),
            "max_drawdown_bps": max_dd(eq),
            "total_net_bps": round(sum(net), 0)}


def run():
    records = [json.loads(l) for l in COHORT.read_text().splitlines() if l.strip()]
    by_asset = defaultdict(list)
    for r in records:
        by_asset[r["asset_id"]].append(r)
    for rs in by_asset.values():
        rs.sort(key=lambda x: x["decision_ns"])

    longs, shorts, always_long, bh = [], [], [], {}
    for asset, rs in sorted(by_asset.items()):
        fund = [x["features"]["last_funding_rate"]["value"] for x in rs]
        interval = [x["features"]["funding_interval_hours"]["value"] for x in rs]
        mark = [x["features"]["mark_price"]["value"] for x in rs]
        i, first = LOOKBACK, LOOKBACK
        last_eligible = len(rs) - HOLD - 1
        # buy-hold over eligible span
        if last_eligible > first and mark[first] > 0 and mark[last_eligible] > 0:
            bh[asset] = round((mark[last_eligible] / mark[first] - 1) * 1e4, 0)
        while i <= last_eligible:
            fw = fund[i - LOOKBACK:i]
            pct = (sum(1 for w in fw if w < fund[i])
                   + 0.5 * sum(1 for w in fw if w == fund[i])) / LOOKBACK
            direction = 0
            if pct >= 0.80:
                direction = 1
            elif pct <= 0.20:
                direction = -1
            if direction == 0:
                i += 1
                continue
            entry, exit_ = mark[i], mark[i + HOLD]
            if entry <= 0 or exit_ <= 0:
                i += HOLD
                continue
            gross = direction * (exit_ / entry - 1) * 1e4
            # funding accrual over hold: long pays +rate, short pays -rate
            paid = 0.0
            for j in range(i + 1, i + HOLD + 1):
                per_day = (24.0 / interval[j]) if interval[j] > 0 \
                    else FUNDING_SETTLEMENTS_PER_DAY
                paid += direction * fund[j] * per_day * 1e4
            net = gross - 2 * FEE_BPS - paid
            t = {"asset": asset, "i": i, "gross_bps": round(gross, 1),
                 "funding_bps": round(paid, 1), "net_bps": round(net, 1),
                 "pct": round(pct, 3)}
            (longs if direction > 0 else shorts).append(t)
            i += HOLD  # non-overlapping
        # matched-activity always-long: same cadence, unconditional
        for j in range(LOOKBACK, last_eligible, HOLD):
            if mark[j] > 0 and mark[j + HOLD] > 0:
                g = (mark[j + HOLD] / mark[j] - 1) * 1e4
                always_long.append({"asset": asset, "gross_bps": round(g, 1),
                                    "net_bps": round(g - 2 * FEE_BPS, 1)})

    all_t = longs + shorts
    return {"schema_version": "nanojev-financial-signal-net-backtest-v1",
            "status": "measurement_complete",
            "scope": "net-of-cost accounting on real cohort; approximate funding "
                     "accrual; no market impact model; NOT a tradability claim",
            "parameters": {"fee_bps_per_side": FEE_BPS, "hold_bars": HOLD,
                           "long_pct": 0.80, "short_pct": 0.20,
                           "non_overlapping": True},
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "results": {
                "long_top_decile": summarize(longs, "long_funding>=80pct"),
                "short_bottom_decile": summarize(shorts, "short_funding<=20pct"),
                "all_signal_trades": summarize(all_t, "all"),
                "always_long_matched": summarize(always_long, "always_long"),
                "buy_hold_bps_per_asset": bh},
            "per_asset": {a: summarize(
                [t for t in all_t if t["asset"] == a], a)
                for a in sorted(by_asset)}}


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
