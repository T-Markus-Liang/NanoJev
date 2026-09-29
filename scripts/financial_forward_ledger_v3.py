#!/usr/bin/env python3
"""T120: forward paper-trading ledger for the rolling-campaign sleeve
(third sleeve).

Sleeve spec (T119 campaign state machine — scripts/financial_campaign_sim_v1.py
and research/financial_campaign_sim_protocol_v1.json — frozen parameters
reused on the refreshed 10-asset XS universe):

  universe: the 10 USDT-M perps in data/perp_pit_xs_v1/records.jsonl,
            merged with both refresh supplements (same dedup-by-id merge
            as financial_forward_ledger_v2.py). dfh20 is recomputed from
            the merged close series exactly as in v2 (strictly-prior 20
            contiguous bars); refreshed legacy bars carry mark_price as
            the close proxy, refreshed XS bars carry close.
  state:    at most ONE open campaign at a time (matching the sim).
            While flat the ledger scans daily; while open it marks to
            market and records pyramid adds / exits.
  entry:    BTC ret20 > 0 AND asset in the top XS dfh20 quintile that day
            (mid-rank pct >= 0.8 over assets with dfh20) AND pullback
            state (dfh20 in [-0.12, -0.03] AND close > prev close) AND
            funding_pct < 0.8 if the asset has a funding percentile that
            day. Highest dfh20 rank wins (tie: asset_id sort). Fill at
            the signal-day close; 100% of equity deployed, 5bps fee.
  pyramid:  while open and no exit fires: unrealized >= +3% vs vwap AND
            gate on AND still top-quintile -> add 50% of current
            position at mark, max 2 adds, margined perp-style (cash may
            go negative; ~1.75x max notional), 5bps per add fill.
  exits:    first to fire, in order: data_end (asset has no bar that
            day -> exit at last known close), btc_regime_off (ret20 <=
            0), funding_euphoria (funding_pct >= 0.95),
            trailing_stop_8pct (close < 0.92 * max close since entry),
            time_stop_30d (>=30 calendar days held), rank_below_median
            (dfh20 pct < 0.5 or unrankable). 5bps exit fee. Same-day
            re-entry after an exit is allowed except into the asset just
            exited. Unlike the bounded sim there is NO end_of_sample
            force-close: a campaign open at the input tail stays open
            and keeps marking to market.
  funding_pct: per-asset mid-rank pct of last_funding_rate vs its
            trailing-180 daily values (min 20 observations), the same
            construction as the sim. All 10 XS assets carry
            last_funding_rate in the cohort and in both supplements.

Documented deviations from the 277-asset T119 sim for the 10-asset
refreshed universe:
  * MIN_RANKED 30 -> 6: a usable XS rank requires >=6 assets with dfh20
    that day (the v2 sleeve's graceful-degradation floor). On n=10
    ranked assets the pct>=0.8 rule admits at most the top-2 (mid-rank
    0.95/0.85; third is 0.75); n=8-9 also admits 2, n=6-7 only top-1.
  * BTC regime gate: ret20 is computed from the merged BTCUSDT-PERP
    close series (close / close 20 contiguous bars back - 1) instead of
    the static data/rc_futures_v1 csv — the csv ends 2025-12 and cannot
    gate refreshed bars, while BTCUSDT-PERP is inside the tracked
    universe and refreshes daily. Unevaluable ret20 -> gate off.
  * No end_of_sample force-close (see above).

Ledger contract (results/forward_ledger_v3.jsonl):
  * append-only JSONL; ONE line per decision date (union of all record
    dates across the 10 assets) — the sleeve is a single campaign book
  * idempotent: re-running loads recorded decision_date keys and appends
    only dates not already in the ledger
  * every line is fully determined at write time: mark-to-market equity,
    position state and fills are all computed from day-t bars
  * events: ``scan`` (flat, gate on, no qualifier — gate status +
    candidate count), ``flat_gate_off`` (flat and BTC gate off or
    unevaluable), ``enter`` (flat -> campaign), ``add`` (pyramid fill),
    ``exit`` (exit fired; ``exit.reason`` carries the rule; an optional
    ``same_day_reentry`` block records a same-day re-entry), ``hold``
    (open, mark-to-market only)
  * content-light: gate inputs, rank pcts, position bookkeeping, equity —
    never raw feature payloads

--snapshot prints the sleeve state as of the latest decision date: gate,
open campaign (entry, adds, unrealized, trailing-stop level, days held)
or the latest scan's candidates, plus closed-campaign stats.

Simulated paper accounting only — no orders, no broker, no account
access, not a tradability or profitability claim.
"""

import argparse
import bisect
import datetime as dt
import json
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_xs_v1/records.jsonl"
SUPPLEMENTS = [ROOT / "data/binance_refresh_v1/records.jsonl",
               ROOT / "data/binance_xs_refresh_v1/records.jsonl"]
LEDGER = ROOT / "results/forward_ledger_v3.jsonl"
# T129 xs_v2 universe (30 assets, research/live_universe_v2.json): opt-in via
# --universe xs_v2; writes to a SEPARATE ledger so the v1 chain is untouched.
COHORT_XS2 = ROOT / "data/perp_pit_xs_v2/records.jsonl"
SUPPLEMENTS_XS2 = SUPPLEMENTS + [ROOT / "data/binance_xs2_refresh_v1/records.jsonl"]
LEDGER_XS2 = ROOT / "results/forward_ledger_v3_xs2.jsonl"

SCHEMA = "nanojev-forward-ledger-v3"
SLEEVE = "xs_campaign_pullback_pyramid_v1"
BTC_ASSET = "BTCUSDT-PERP"
DFH_LOOKBACK = 20            # strictly-prior contiguous bars for the high
BTC_GATE_LOOKBACK = 20       # BTC ret20: close / close[t-20] - 1
FEE = 0.0005                 # 5bps per fill (entry, each add, exit)
TOP_Q = 0.8                  # top XS dfh20 quintile (top-2 on 10 assets)
MEDIAN = 0.5                 # exit (f): rank below median
PULLBACK_LO, PULLBACK_HI = -0.12, -0.03
FUND_ENTRY_MAX = 0.8         # stealth filter at entry
FUND_EXIT = 0.95             # euphoria exit
TRAIL = 0.92                 # 8% trailing stop vs max close since entry
TIME_STOP_DAYS = 30
ADD_TRIGGER = 0.03           # +3% unrealized vs vwap to allow an add
ADD_FRAC = 0.5               # add 50% of current position
MAX_ADDS = 2
MIN_RANKED = 6               # DEVIATION from sim's 30 (10-asset universe)
FUND_LOOKBACK = 180          # trailing-180 mid-rank pct (repo convention)
FUND_MIN_WINDOW = 20
EXIT_REASONS = ("data_end", "btc_regime_off", "funding_euphoria",
                "trailing_stop_8pct", "time_stop_30d",
                "rank_below_median")
EVENTS = ("scan", "flat_gate_off", "enter", "add", "exit", "hold")


def _r(x, nd=4):
    return round(x, nd) if isinstance(x, (int, float)) else x


def feat(record, name):
    f = record.get("features", {}).get(name)
    return f.get("value") if isinstance(f, dict) else None


def record_close(record):
    """XS price basis: klines close as mark proxy; refreshed legacy bars
    carry mark_price instead (refresh_v1 shape). Same as v2."""
    close = feat(record, "close")
    if close is not None:
        return close, "close"
    mark = feat(record, "mark_price")
    if mark is not None:
        return mark, "mark_price"
    return None, None


def load_inputs(cohort_path, supplement_paths):
    """asset -> sorted [{date, close, basis, funding, decision_ns,
    record_id}]. Same merge as v2: cohort + supplement records deduped by
    record id (cohort wins); per asset, a supplement date already present
    in the cohort is skipped."""
    records = [json.loads(l) for l in cohort_path.read_text().splitlines()
               if l.strip()]
    seen = {r["id"] for r in records}
    extra = 0
    for path in supplement_paths:
        if not path.exists():
            continue
        for l in path.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("id") in seen:
                continue
            records.append(r)
            seen.add(r["id"])
            extra += 1
    by_asset = defaultdict(dict)  # asset -> date -> bar
    for r in records:
        date = r["id"].rsplit(":", 1)[-1]
        close, basis = record_close(r)
        asset = r["asset_id"]
        if date in by_asset[asset]:  # cohort already holds this date
            continue
        by_asset[asset][date] = {"date": date, "close": close,
                                 "basis": basis,
                                 "funding": feat(r, "last_funding_rate"),
                                 "decision_ns": r["decision_ns"],
                                 "record_id": r["id"]}
    return {a: [days[d] for d in sorted(days)]
            for a, days in by_asset.items()}, extra


def dt_date(text):
    return dt.date.fromisoformat(text)


def mid_rank_pct(window, x):
    """Mid-rank percentile of x vs trailing window (ties count half)."""
    if x is None or not window:
        return None
    return (sum(1 for w in window if w < x)
            + 0.5 * sum(1 for w in window if w == x)) / len(window)


def add_bar_features(series):
    """Recompute dfh20 (v2 construction), prev_close and funding_pct in
    place on each asset's merged rows."""
    for rows in series.values():
        fund = [r["funding"] for r in rows]
        for i, r in enumerate(rows):
            contiguous = (i >= DFH_LOOKBACK
                          and (dt_date(r["date"]) - dt_date(
                               rows[i - DFH_LOOKBACK]["date"])).days
                          == DFH_LOOKBACK)
            dfh = None
            if contiguous and r["close"] is not None and r["close"] > 0:
                prior = [rows[j]["close"] for j in
                         range(i - DFH_LOOKBACK, i)]
                if all(c is not None and c > 0 for c in prior):
                    dfh = r["close"] / max(prior) - 1.0
            r["dfh20"] = dfh
            r["prev_close"] = rows[i - 1]["close"] if i else None
            fwin = [x for x in fund[max(0, i - FUND_LOOKBACK):i]
                    if x is not None]
            r["funding_pct"] = (mid_rank_pct(fwin, r["funding"])
                                if r["funding"] is not None
                                and len(fwin) >= FUND_MIN_WINDOW
                                else None)


def btc_ret20_map(btc_rows):
    """date -> (btc_close, ret20 or None). ret20 = close / close of the
    bar 20 positions back - 1, only when that bar is exactly 20 calendar
    days prior (contiguous), both closes positive."""
    out = {}
    for i, r in enumerate(btc_rows):
        ret = None
        if (i >= BTC_GATE_LOOKBACK and r["close"] is not None
                and r["close"] > 0):
            back = btc_rows[i - BTC_GATE_LOOKBACK]
            span = (dt_date(r["date"]) - dt_date(back["date"])).days
            if (span == BTC_GATE_LOOKBACK and back["close"] is not None
                    and back["close"] > 0):
                ret = r["close"] / back["close"] - 1.0
        out[r["date"]] = (r["close"], ret)
    return out


def xs_rank_maps(dates, by_date):
    """date -> {asset: dfh20 mid-rank pct}, date -> n assets with dfh20.
    Same mid-rank construction as the sim's build_day_views."""
    dfh_pct, n_ranked = {}, {}
    for date in dates:
        scored = [(a, r["dfh20"]) for a, m in by_date.items()
                  if (r := m.get(date)) is not None
                  and r["dfh20"] is not None]
        n_ranked[date] = len(scored)
        vals = sorted(v for _, v in scored)
        n = len(vals)
        pct = {}
        for a, v in scored:
            lo = bisect.bisect_left(vals, v)
            eq = bisect.bisect_right(vals, v) - lo
            pct[a] = (lo + 0.5 * eq) / n
        dfh_pct[date] = pct
    return dfh_pct, n_ranked


def entry_candidates(day_rows, pcts, excluded):
    """Composite-entry qualifiers, sorted by (-dfh20_pct, asset)."""
    cands = []
    for a, r in day_rows.items():
        if a in excluded:
            continue
        pct = pcts.get(a)
        if pct is None or pct < TOP_Q:
            continue
        d = r["dfh20"]
        if d is None or not (PULLBACK_LO <= d <= PULLBACK_HI):
            continue
        if (r["close"] is None or r["close"] <= 0
                or r["prev_close"] is None
                or r["close"] <= r["prev_close"]):
            continue
        if (r["funding_pct"] is not None
                and r["funding_pct"] >= FUND_ENTRY_MAX):
            continue
        cands.append((pct, a))
    cands.sort(key=lambda x: (-x[0], x[1]))
    return cands


def position_block(pos, qty, date, pcts, row):
    """End-of-day open-position state for the ledger line."""
    held = (dt_date(date) - dt_date(pos["entry_date"])).days
    vwap = pos["cost_qty"] / qty if qty else None
    last = pos["last_close"]
    return {
        "asset": pos["asset"],
        "entry_date": pos["entry_date"],
        "entry_px": _r(pos["entry_px"], 6),
        "qty": _r(qty, 8),
        "vwap": _r(vwap, 6),
        "days_held": held,
        "days_to_time_stop": max(0, TIME_STOP_DAYS - held),
        "adds": pos["adds"],
        "max_close": _r(pos["max_close"], 6),
        "trail_stop_px": _r(pos["max_close"] * TRAIL, 6),
        "last_close": _r(last, 6),
        "unrealized_vs_vwap": (_r(last / vwap - 1.0, 6)
                               if vwap else None),
        "drawdown_from_max": (_r(last / pos["max_close"] - 1.0, 6)
                              if pos["max_close"] else None),
        "dfh20_pct": _r(pcts.get(pos["asset"]), 4),
        "funding_pct": _r(row["funding_pct"], 4) if row else None,
    }


def replay(series):
    """Deterministic full replay of the campaign state machine ->
    (ordered ledger line dicts, closed-campaign list)."""
    dates = sorted({r["date"] for rows in series.values() for r in rows})
    by_date = {asset: {r["date"]: r for r in rows}
               for asset, rows in series.items()}
    ns_by_date = defaultdict(int)
    for rows in series.values():
        for r in rows:
            ns_by_date[r["date"]] = max(ns_by_date[r["date"]],
                                        r["decision_ns"])
    btc_gate = btc_ret20_map(series.get(BTC_ASSET, []))
    dfh_pct, n_ranked = xs_rank_maps(dates, by_date)

    cash, qty = 1.0, 0.0
    pos = None        # {asset, entry_date, entry_px, cost_qty,
                      #  max_close, last_close, adds, entry_equity}
    campaigns = []
    lines = []
    exited_today = set()

    for date in dates:
        day_rows = {a: m[date] for a, m in by_date.items()
                    if date in m}
        btc_close, ret20 = btc_gate.get(date, (None, None))
        gate_on = ret20 is not None and ret20 > 0
        pcts = dfh_pct.get(date, {})
        ranked_ok = n_ranked.get(date, 0) >= MIN_RANKED

        event = None
        fill = None
        exit_info = None
        reentry = None
        scan = None

        # -------- open campaign: exits in order, then pyramid --------
        if pos is not None:
            asset = pos["asset"]
            row = day_rows.get(asset)
            reason = None
            px = None
            if row is None or row["close"] is None:
                reason, px = "data_end", pos["last_close"]
            else:
                pos["last_close"] = row["close"]
                pos["max_close"] = max(pos["max_close"],
                                       pos["last_close"])
                pct = pcts.get(asset)
                fpct = row["funding_pct"]
                held = (dt_date(date)
                        - dt_date(pos["entry_date"])).days
                if ret20 is not None and ret20 <= 0:
                    reason = "btc_regime_off"
                elif fpct is not None and fpct >= FUND_EXIT:
                    reason = "funding_euphoria"
                elif pos["last_close"] < pos["max_close"] * TRAIL:
                    reason = "trailing_stop_8pct"
                elif held >= TIME_STOP_DAYS:
                    reason = "time_stop_30d"
                elif pct is None or pct < MEDIAN:
                    reason = "rank_below_median"
                if reason:
                    px = pos["last_close"]
            if reason:
                exited_today.add(asset)
                proceeds = qty * px
                fee = proceeds * FEE
                cash += proceeds - fee
                ret = cash / pos["entry_equity"] - 1.0
                campaign = {
                    "asset": asset,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "days_held": (dt_date(date)
                                  - dt_date(pos["entry_date"])).days,
                    "adds": pos["adds"],
                    "entry_px": _r(pos["entry_px"], 6),
                    "exit_px": _r(px, 6),
                    "return": _r(ret, 6),
                }
                campaigns.append(campaign)
                exit_info = {"reason": reason, "px": _r(px, 6),
                             "fee_frac": _r(fee, 8),
                             "stale_px": row is None
                             or row["close"] is None,
                             "campaign": campaign}
                qty = 0.0
                pos = None
                event = "exit"
            elif row is not None:
                vwap = pos["cost_qty"] / qty if qty else None
                unreal = (pos["last_close"] / vwap - 1.0
                          if vwap else 0.0)
                pct = pcts.get(asset)
                if (pos["adds"] < MAX_ADDS and unreal >= ADD_TRIGGER
                        and gate_on and pct is not None
                        and pct >= TOP_Q):
                    add_qty = qty * ADD_FRAC
                    notional = add_qty * pos["last_close"]
                    fee = notional * FEE
                    cash -= notional + fee     # margined perp-style
                    qty += add_qty
                    pos["cost_qty"] += notional
                    pos["adds"] += 1
                    fill = {"side": "add", "asset": asset,
                            "px": _r(pos["last_close"], 6),
                            "qty_add": _r(add_qty, 8),
                            "fee_frac": _r(fee, 8),
                            "adds_now": pos["adds"],
                            "unrealized_vs_vwap": _r(unreal, 6)}
                    event = "add"
                else:
                    event = "hold"

        # -------- flat: entry scan (also runs after a same-day exit) --
        if pos is None:
            scan_ran = gate_on and ranked_ok and cash > 0
            cands = (entry_candidates(day_rows, pcts, exited_today)
                     if scan_ran else [])
            scan = {
                "evaluated": bool(scan_ran),
                "gate_on": gate_on,
                "ranked_ok": ranked_ok,
                "n_candidates": len(cands),
                "top_quintile": sorted(a for a, p in pcts.items()
                                       if p >= TOP_Q),
                "candidates": [{"asset": a, "dfh20_pct": _r(p, 4),
                                "dfh20": _r(day_rows[a]["dfh20"], 6),
                                "funding_pct":
                                    _r(day_rows[a]["funding_pct"], 4)}
                               for p, a in cands],
            }
            if cands:
                asset = cands[0][1]
                px = day_rows[asset]["close"]
                entry_equity = cash          # deploy 100% of equity
                qty = cash / (px * (1.0 + FEE))
                fee = qty * px * FEE
                cash = 0.0
                pos = {"asset": asset, "entry_date": date,
                       "entry_px": px, "cost_qty": qty * px,
                       "max_close": px, "last_close": px,
                       "adds": 0, "entry_equity": entry_equity}
                fill = {"side": "entry", "asset": asset,
                        "px": _r(px, 6), "qty": _r(qty, 8),
                        "fee_frac": _r(fee, 8),
                        "dfh20_pct": _r(cands[0][0], 4),
                        "dfh20": _r(day_rows[asset]["dfh20"], 6),
                        "funding_pct":
                            _r(day_rows[asset]["funding_pct"], 4)}
                if event == "exit":
                    reentry = {"asset": asset, "px": _r(px, 6)}
                else:
                    event = "enter"
            elif event is None:
                event = "scan" if gate_on else "flat_gate_off"

        equity = cash + qty * pos["last_close"] if pos is not None \
            else cash
        lines.append({
            "schema_version": SCHEMA,
            "sleeve": SLEEVE,
            "decision_date": date,
            "decision_ns": ns_by_date[date],
            "event": event,
            "gate": {"btc_close": _r(btc_close, 4),
                     "btc_ret20": _r(ret20, 6),
                     "on": gate_on},
            "universe": {"ranked": n_ranked.get(date, 0),
                         "min_required": MIN_RANKED,
                         "ranked_ok": ranked_ok,
                         "assets_with_bars": len(day_rows)},
            "scan": scan,
            "fill": fill,
            "exit": exit_info,
            "same_day_reentry": reentry,
            "position": (position_block(pos, qty, date, pcts,
                                        day_rows.get(pos["asset"]))
                         if pos is not None else None),
            "equity": _r(equity, 8),
            "campaigns_closed": len(campaigns),
        })
        exited_today.clear()
    return lines, campaigns


def campaign_stats(campaigns, lines):
    """Summary over closed campaigns + the daily equity curve."""
    rets = [c["return"] for c in campaigns]
    n = len(campaigns)
    reasons = defaultdict(int)
    for l in lines:  # exit reasons live on the exit lines
        if l["event"] == "exit":
            reasons[l["exit"]["reason"]] += 1
    vals = [l["equity"] for l in lines if l["equity"] is not None]
    peak, mdd = (vals[0], 0.0) if vals else (1.0, 0.0)
    for v in vals:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak if peak else 0.0)
    open_days = sum(1 for l in lines
                    if l["position"] is not None
                    or l["event"] in ("exit", "add", "hold"))
    return {
        "n_campaigns": n,
        "win_rate": (sum(1 for r in rets if r > 0) / n) if n else None,
        "mean_campaign_return": (sum(rets) / n) if n else None,
        "max_campaign_return": max(rets) if rets else None,
        "min_campaign_return": min(rets) if rets else None,
        "avg_adds_per_campaign": (sum(c["adds"] for c in campaigns) / n
                                  if n else None),
        "exit_reason_histogram": dict(sorted(reasons.items())),
        "final_equity": vals[-1] if vals else None,
        "max_drawdown_frac": _r(mdd, 6),
        "time_in_market_frac": (_r(open_days / len(lines), 6)
                                if lines else None),
    }


def replay_full(args):
    series, extra = load_inputs(args.cohort, args.supplement)
    add_bar_features(series)
    lines, campaigns = replay(series)
    return lines, campaigns, extra


def cmd_run(args):
    lines, campaigns, extra = replay_full(args)
    seen = set()
    if args.ledger.exists():
        for l in args.ledger.read_text().splitlines():
            if not l.strip():
                continue
            seen.add(json.loads(l)["decision_date"])

    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    appended = skipped = 0
    events = defaultdict(int)
    with args.ledger.open("a", encoding="utf-8") as stream:
        for line in lines:
            if line["decision_date"] in seen:
                skipped += 1
                continue
            stream.write(json.dumps(line, sort_keys=True) + "\n")
            appended += 1
            events[line["event"]] += 1

    last = lines[-1] if lines else None
    print(json.dumps({
        "ledger": str(args.ledger),
        "sleeve": SLEEVE,
        "supplement_records_merged": extra,
        "decision_dates": len(lines),
        "dates_already_recorded": skipped,
        "dates_appended": appended,
        "events_appended": dict(sorted(events.items())),
        "backfill": campaign_stats(campaigns, lines),
        "open_campaign": (last["position"] if last else None),
        "last_line": ({"decision_date": last["decision_date"],
                       "event": last["event"],
                       "gate_on": last["gate"]["on"],
                       "equity": last["equity"]}
                      if last else None),
    }, indent=2, sort_keys=True))
    return 0


def cmd_snapshot(args):
    lines, campaigns, extra = replay_full(args)
    last = lines[-1] if lines else None
    if last is None:
        print(json.dumps({"schema_version": SCHEMA + "-snapshot",
                          "sleeve": SLEEVE, "status": "empty"}))
        return 0
    stats = campaign_stats(campaigns, lines)
    recent = [l["exit"]["campaign"] | {"reason": l["exit"]["reason"]}
              for l in lines if l["event"] == "exit"][-5:]
    out = {
        "schema_version": SCHEMA + "-snapshot",
        "sleeve": SLEEVE,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                      time.gmtime()),
        "supplement_records_merged": extra,
        "as_of_decision_date": last["decision_date"],
        "event": last["event"],
        "gate": last["gate"],
        "universe": last["universe"],
        "equity": last["equity"],
        "campaigns_closed": last["campaigns_closed"],
        "campaign_stats": stats,
        "recent_exits": recent,
        "open_campaign": last["position"],
        "latest_scan": last["scan"],
    }
    if last["position"] is not None:
        p = last["position"]
        out["watch"] = {
            "asset": p["asset"],
            "exit_if": {
                "btc_ret20_le_0": last["gate"]["btc_ret20"],
                "funding_pct_ge_0_95": p["funding_pct"],
                "close_below_trail_stop": p["trail_stop_px"],
                "dfh20_pct_below_0_5": p["dfh20_pct"],
                "time_stop_in_days": p["days_to_time_stop"],
            },
            "add_if": {
                "unrealized_vs_vwap_ge_0_03":
                    p["unrealized_vs_vwap"],
                "adds_remaining": MAX_ADDS - p["adds"],
                "needs_gate_on": last["gate"]["on"],
                "needs_dfh20_pct_ge_0_8": p["dfh20_pct"],
            },
        }
    else:
        out["watch"] = {
            "flat": True,
            "gate_on": last["gate"]["on"],
            "candidates_today": (last["scan"] or {}).get("candidates"),
            "needs": "BTC ret20>0 AND top-quintile dfh20 AND "
                     "dfh20 in [-0.12,-0.03] AND close>prev AND "
                     "funding_pct<0.8",
        }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", type=Path, default=COHORT)
    ap.add_argument("--supplement", type=Path, action="append",
                    default=None,
                    help="refresh supplement JSONL (repeatable; merged "
                         "and deduped by id). Defaults: "
                         "binance_refresh_v1 (legacy mark-price bars) + "
                         "binance_xs_refresh_v1 (XS close bars)")
    ap.add_argument("--ledger", type=Path, default=LEDGER)
    ap.add_argument("--universe", choices=("xs_v1", "xs_v2"), default="xs_v1",
                    help="xs_v2 = 30-asset T129 cohort (perp_pit_xs_v2 + the "
                         "xs2 refresh supplement) writing to "
                         "results/forward_ledger_v3_xs2.jsonl; xs_v1 is the "
                         "unchanged default used by the daily chain")
    ap.add_argument("--snapshot", action="store_true",
                    help="print sleeve state as of the latest decision "
                         "date")
    args = ap.parse_args()
    if args.universe == "xs_v2":
        if args.cohort == COHORT:
            args.cohort = COHORT_XS2
        if args.supplement is None:
            args.supplement = SUPPLEMENTS_XS2
        if args.ledger == LEDGER:
            args.ledger = LEDGER_XS2
    if args.supplement is None:
        args.supplement = SUPPLEMENTS
    return cmd_snapshot(args) if args.snapshot else cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
