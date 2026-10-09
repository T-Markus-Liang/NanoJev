#!/usr/bin/env python3
"""T83: forward paper-trading ledger for the FROZEN signal spec.

Signal (research/financial_signal_spec_v1.json, crowded_long_carry_follow_v1):
  entry:  funding_pct >= 0.80 AND basis_pct >= 0.66
          AND BTC mark_price 20-bar log return > 0 (market-wide on/off switch)
  exit:   5 decision bars (time stop) OR BTC 20-bar trend turns <= 0
          (regime_off), whichever fires first
  direction: long only
  costs:  taker 5bps per side (10bps round trip) + funding accrual per held
          day = last_funding_rate * (24 / funding_interval_hours), fallback
          3 settlements/day (8h). Same accounting as the net backtest replay.

Ledger contract (results/forward_ledger_v1.jsonl):
  * append-only JSONL; one line per (asset_id, decision_ns) decision bar
  * idempotent: re-running loads recorded (asset_id, decision_ns) keys and
    appends only bars not already in the ledger
  * content-light: each line carries the signal INPUTS (mid-rank pcts, BTC
    trend log return) and the cohort record id — never raw feature payloads
  * entry lines are written at the entry bar; the realized PnL FILLS IN on
    the exit line once the hold window elapses
    (net_bps = gross_bps - 10bps fees - funding accrual bps)
  * convention: an asset that exits on a bar is flat for the rest of that
    bar; entry eligibility resumes the next bar (non-overlapping, one
    position per asset at a time)

--snapshot prints per-asset state (warmup / in / out) as of the latest
cohort bar, plus the open position or most recent closed trade.

Simulated paper accounting only — no orders, no broker, no account access,
not a tradability or profitability claim.
"""

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_v1/records.jsonl"
SUPPLEMENT = ROOT / "data/binance_refresh_v1/records.jsonl"
LEDGER = ROOT / "results/forward_ledger_v1.jsonl"
SPEC = ROOT / "research/financial_signal_spec_v1.json"

SCHEMA = "nanojev-forward-ledger-v1"
BTC_ASSET = "BTCUSDT-PERP"
LOOKBACK = 180          # trailing mid-rank pct window (excludes decision bar)
HOLD = 5                # time stop in decision bars
TREND_BARS = 20         # BTC trend lookback in decision bars
FUNDING_THRESHOLD = 0.80
BASIS_THRESHOLD = 0.66
FEE_BPS_PER_SIDE = 5.0
ROUND_TRIP_FEES_BPS = 2 * FEE_BPS_PER_SIDE
FALLBACK_SETTLEMENTS_PER_DAY = 3  # 8h funding interval fallback


def feat(record, name):
    """Feature value or None (refresh-supplement records may omit features)."""
    f = record.get("features", {}).get(name)
    return f.get("value") if isinstance(f, dict) else None


def mid_rank_pct(window, x):
    """Mid-rank percentile of x vs trailing window (ties count half)."""
    if x is None or not window:
        return None
    return (sum(1 for w in window if w < x)
            + 0.5 * sum(1 for w in window if w == x)) / len(window)


def btc_trend_map(btc_records):
    """decision_ns -> BTC mark_price 20-bar log return (None-safe)."""
    marks = [feat(r, "mark_price") for r in btc_records]
    out = {}
    for i in range(TREND_BARS, len(btc_records)):
        a, b = marks[i - TREND_BARS], marks[i]
        if a and b and a > 0 and b > 0:
            out[btc_records[i]["decision_ns"]] = math.log(b / a)
    return out


def replay_asset(asset, rs, btc_trend):
    """Replay the frozen spec over one asset's sorted records.

    Returns (ledger_lines, open_position). open_position is None or the
    position dict of a trade still inside its hold window at the series end.
    """
    fund = [feat(r, "last_funding_rate") for r in rs]
    interval = [feat(r, "funding_interval_hours") for r in rs]
    basis = [feat(r, "mark_index_basis_bps") for r in rs]
    rv = [feat(r, "realized_vol_24bar") for r in rs]
    mark = [feat(r, "mark_price") for r in rs]

    lines, pos = [], None
    for i, r in enumerate(rs):
        line = {"schema_version": SCHEMA,
                "asset_id": asset,
                "record_id": r["id"],
                "decision_ns": r["decision_ns"],
                "event": None}
        if i < LOOKBACK:
            line["state"] = "warmup"
            line["signal"] = None
            lines.append(line)
            continue

        signal = {
            "funding_pct": mid_rank_pct(fund[i - LOOKBACK:i], fund[i]),
            "basis_pct": mid_rank_pct(basis[i - LOOKBACK:i], basis[i]),
            "rv_pct": mid_rank_pct(rv[i - LOOKBACK:i], rv[i]),
            "btc_trend_20bar_logret": btc_trend.get(r["decision_ns"]),
        }
        line["signal"] = {k: (round(v, 4) if v is not None else None)
                          for k, v in signal.items()}
        f_pct, b_pct, trend = (signal["funding_pct"], signal["basis_pct"],
                               signal["btc_trend_20bar_logret"])

        if pos is None:
            if (f_pct is not None and f_pct >= FUNDING_THRESHOLD
                    and b_pct is not None and b_pct >= BASIS_THRESHOLD
                    and trend is not None and trend > 0
                    and mark[i] is not None and mark[i] > 0):
                pos = {"entry_i": i, "entry_mark": mark[i],
                       "entry_decision_ns": r["decision_ns"],
                       "entry_record_id": r["id"]}
                line["state"] = "long"
                line["event"] = "enter_long"
                line["entry"] = {"mark": mark[i],
                                 "decision_ns": r["decision_ns"],
                                 "record_id": r["id"],
                                 "bars_held": 0}
            else:
                line["state"] = "flat"
        else:
            held = i - pos["entry_i"]
            reason = None
            if held >= HOLD:
                reason = "exit_time_stop"
            elif trend is not None and trend <= 0:
                reason = "exit_regime_off"
            if reason is None:
                line["state"] = "long"
                line["position"] = {"entry_decision_ns":
                                    pos["entry_decision_ns"],
                                    "entry_mark": pos["entry_mark"],
                                    "bars_held": held}
            else:
                exit_mark = mark[i]
                gross = paid = None
                if pos["entry_mark"] > 0 and exit_mark is not None \
                        and exit_mark > 0:
                    gross = (exit_mark / pos["entry_mark"] - 1) * 1e4
                    paid = 0.0
                    for j in range(pos["entry_i"] + 1, i + 1):
                        per_day = (24.0 / interval[j]) \
                            if interval[j] and interval[j] > 0 \
                            else FALLBACK_SETTLEMENTS_PER_DAY
                        paid += (fund[j] or 0.0) * per_day * 1e4
                net = (gross - ROUND_TRIP_FEES_BPS - paid) \
                    if gross is not None else None
                line["state"] = "flat"
                line["event"] = reason
                line["exit"] = {
                    "mark": exit_mark,
                    "decision_ns": r["decision_ns"],
                    "record_id": r["id"],
                    "bars_held": held,
                    "fills_entry_decision_ns": pos["entry_decision_ns"],
                    "fills_entry_record_id": pos["entry_record_id"],
                    "gross_bps": round(gross, 2) if gross is not None else None,
                    "fees_bps": ROUND_TRIP_FEES_BPS,
                    "funding_bps": round(paid, 2) if paid is not None else None,
                    "net_bps": round(net, 2) if net is not None else None,
                }
                pos = None
        lines.append(line)

    open_pos = None
    if pos is not None:
        held = len(rs) - 1 - pos["entry_i"]
        last_mark = mark[-1]
        unrealized = accrual = None
        if pos["entry_mark"] > 0 and last_mark is not None and last_mark > 0:
            unrealized = (last_mark / pos["entry_mark"] - 1) * 1e4
            accrual = 0.0
            for j in range(pos["entry_i"] + 1, len(rs)):
                per_day = (24.0 / interval[j]) \
                    if interval[j] and interval[j] > 0 \
                    else FALLBACK_SETTLEMENTS_PER_DAY
                accrual += (fund[j] or 0.0) * per_day * 1e4
        open_pos = {**pos, "bars_held": held, "last_mark": last_mark,
                    "unrealized_gross_bps":
                        round(unrealized, 2) if unrealized is not None else None,
                    "accrued_funding_bps":
                        round(accrual, 2) if accrual is not None else None}
    return lines, open_pos


def load_inputs(cohort_path, supplement_path):
    """Cohort records + optional refresh supplement, deduped by record id."""
    records = [json.loads(l) for l in cohort_path.read_text().splitlines()
               if l.strip()]
    seen = {r["id"] for r in records}
    extra = 0
    if supplement_path and supplement_path.exists():
        for l in supplement_path.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("id") not in seen:
                records.append(r)
                seen.add(r["id"])
                extra += 1
    by_asset = defaultdict(list)
    for r in records:
        by_asset[r["asset_id"]].append(r)
    for rs in by_asset.values():
        rs.sort(key=lambda x: x["decision_ns"])
    return by_asset, extra


def replay_all(by_asset):
    """Full deterministic replay. Returns {asset: (lines, open_pos, last_line)}."""
    btc_trend = btc_trend_map(by_asset.get(BTC_ASSET, []))
    result = {}
    for asset in sorted(by_asset):
        lines, open_pos = replay_asset(asset, by_asset[asset], btc_trend)
        result[asset] = (lines, open_pos, lines[-1] if lines else None)
    return result


def cmd_run(args):
    by_asset, extra = load_inputs(args.cohort, args.supplement)
    seen = set()
    if args.ledger.exists():
        for l in args.ledger.read_text().splitlines():
            if not l.strip():
                continue
            d = json.loads(l)
            seen.add((d["asset_id"], d["decision_ns"]))

    replayed = replay_all(by_asset)
    appended = skipped = entries = exits = 0
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    with args.ledger.open("a", encoding="utf-8") as stream:
        for asset in sorted(replayed):
            lines, _, _ = replayed[asset]
            for line in lines:
                if (line["asset_id"], line["decision_ns"]) in seen:
                    skipped += 1
                    continue
                stream.write(json.dumps(line, sort_keys=True) + "\n")
                appended += 1
                entries += line["event"] == "enter_long"
                exits += line["event"] in ("exit_time_stop", "exit_regime_off")

    open_positions = {a: p for a, (_, p, _) in replayed.items() if p}
    print(json.dumps({
        "ledger": str(args.ledger),
        "supplement_records_merged": extra,
        "bars_scanned": sum(len(v[0]) for v in replayed.values()),
        "bars_already_recorded": skipped,
        "bars_appended": appended,
        "entries_appended": entries,
        "exits_appended": exits,
        "open_positions": {a: {"entry_record_id": p["entry_record_id"],
                               "bars_held": p["bars_held"]}
                           for a, p in open_positions.items()},
    }, indent=2, sort_keys=True))
    return 0


def cmd_snapshot(args):
    by_asset, extra = load_inputs(args.cohort, args.supplement)
    replayed = replay_all(by_asset)
    snap = {}
    for asset in sorted(replayed):
        lines, open_pos, last = replayed[asset]
        date = last["record_id"].rsplit(":", 1)[-1]
        state = {"warmup": "warmup", "flat": "out", "long": "in"}[last["state"]]
        entry = {"asset": asset, "as_of_bar": date,
                 "decision_ns": last["decision_ns"], "state": state,
                 "signal": last["signal"]}
        if open_pos:
            entry["open_position"] = {
                "entry_bar": open_pos["entry_record_id"].rsplit(":", 1)[-1],
                "entry_mark": open_pos["entry_mark"],
                "bars_held": open_pos["bars_held"],
                "unrealized_gross_bps": open_pos["unrealized_gross_bps"],
                "accrued_funding_bps": open_pos["accrued_funding_bps"]}
        else:
            exits = [l for l in lines if l["event"] in
                     ("exit_time_stop", "exit_regime_off")]
            if exits:
                x = exits[-1]["exit"]
                entry["last_closed_trade"] = {
                    "exit_bar": x["record_id"].rsplit(":", 1)[-1],
                    "reason": exits[-1]["event"],
                    "bars_held": x["bars_held"],
                    "net_bps": x["net_bps"]}
        snap[asset] = entry

    print(json.dumps({"schema_version": SCHEMA + "-snapshot",
                      "signal": "crowded_long_carry_follow_v1",
                      "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                  time.gmtime()),
                      "supplement_records_merged": extra,
                      "assets": snap}, indent=2, sort_keys=True))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", type=Path, default=COHORT)
    ap.add_argument("--supplement", type=Path, default=SUPPLEMENT,
                    help="optional refresh supplement JSONL (merged, deduped by id)")
    ap.add_argument("--ledger", type=Path, default=LEDGER)
    ap.add_argument("--snapshot", action="store_true",
                    help="print per-asset signal state as of the latest cohort bar")
    args = ap.parse_args()
    return cmd_snapshot(args) if args.snapshot else cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
