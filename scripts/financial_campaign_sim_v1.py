#!/usr/bin/env python3
"""Rolling-campaign simulator on the 277-symbol MEGA cohort (T119).

Event-driven daily sim of the full entry -> pyramid -> exit state machine,
one campaign at a time, capital = 1.0, on
``data/perp_pit_mega_v1/records.jsonl`` (277 USDT-M perps, 2021-01 ->
2025-12, close-as-mark basis, 52 funding-covered symbols, includes
delisted/renamed early-stoppers). BTC regime gate uses
``data/rc_futures_v1/BTC/BTCUSDT_1d.csv``.

STATE MACHINE (all signals evaluated on day-t PIT features, all fills at
day-t close — the same close basis the cohort labels use):

  ENTRY (composite, all conditions required):
    1. BTC ret20 > 0 (close/close[t-20]-1 on the BTC 1d csv; warmup = off)
    2. asset in the TOP cross-sectional dfh20 quintile that day
       (mid-rank pct >= 0.8 over assets with dfh20; needs >=30 ranked)
    3. pullback-state: dfh20 in [-0.12, -0.03] AND close > prev close
       (recovering off the 20-bar high, not at it, not breaking down)
    4. funding_pct < 0.8 IF the asset has a funding percentile that day
       (stealth positioning, not euphoric; unfunded symbols pass)
    Multiple qualifiers -> highest dfh20 rank wins (tie: asset_id sort).
    No qualifier -> flat that day. Entry deploys 100% of equity.

  PYRAMID (only while a campaign is open and no exit fires):
    unrealized >= +3% vs volume-weighted avg fill AND gate still on AND
    asset still top-quintile -> add 50% of current position (at mark,
    qty_add = 0.5 * qty), max 2 adds. Adds are MARGINED perp-style:
    entry commits the full equity so add notional is borrowed margin
    (cash goes negative); max implicit leverage ~1.75x. Each fill pays
    its own 5bps.

  EXIT (first to fire, checked in this order each day):
    a. data_end: asset has no bar that day (delisted/gap) -> exit at the
       last known close
    b. BTC ret20 <= 0 -> full exit at that day's close (daily bars:
       "next bar open" is approximated by this close)
    c. funding_pct >= 0.95 -> full exit
    d. trailing stop: close < max_close_since_entry * 0.92 (8% trail)
    e. time stop: 30 calendar days in campaign
    f. asset dfh20 rank < median (pct < 0.5, or unrankable) -> exit
    A still-open campaign at the sample end is force-closed
    ("end_of_sample"). Same-day re-entry after an exit is allowed except
    into the asset just exited.

  COSTS: 5bps per fill (entry, each add, exit). Funding carry skipped —
  funding_pct is used only as a positioning signal, not a cashflow.

ARMS:
  composite   full entry filter above.
  no_filter   baseline sibling: entry = gate-on AND top-quintile only
              (drops pullback-state + funding filter); exits identical.
              Isolates the entry filter's marginal value.
  btc_hold    buy-and-hold BTC over the same calendar.

Metrics per arm: n campaigns, win rate, mean/max campaign return, final
compounded equity, max drawdown on the daily equity curve,
time-in-market %, per-year table (by exit year), avg adds/campaign,
exit-reason histogram. Determinism: every arm is simulated twice and the
serialized results must be identical.

Artifacts follow the established convention: a frozen protocol
(``research/financial_campaign_sim_protocol_v1.json``) plus an owner
self-authorization pinning it by sha256
(``results/financial_campaign_sim_authorization_v1.json``) are written
on every run BEFORE measurement. Measurement only: no fitting, no
trading, no network.
"""
import argparse
import bisect
import datetime as dt
import hashlib
import json
import math
import pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_mega_v1/records.jsonl"
BTC_CSV = ROOT / "data/rc_futures_v1/BTC/BTCUSDT_1d.csv"
OUT = ROOT / "results/financial_campaign_sim_v1.json"
PROTOCOL = ROOT / "research/financial_campaign_sim_protocol_v1.json"
AUTH = ROOT / "results/financial_campaign_sim_authorization_v1.json"

# ---- frozen parameters ------------------------------------------------
MAX_CONCURRENT = 1            # ONE campaign at a time (documented choice)
FEE = 0.0005                  # 5bps per fill
TOP_Q = 0.8                   # top XS dfh20 quintile
MEDIAN = 0.5                  # exit (f): rank below median
PULLBACK_LO, PULLBACK_HI = -0.12, -0.03
FUND_ENTRY_MAX = 0.8          # stealth filter at entry
FUND_EXIT = 0.95              # euphoria exit
TRAIL = 0.92                  # 8% trailing stop vs max close since entry
TIME_STOP_DAYS = 30
ADD_TRIGGER = 0.03            # +3% unrealized vs vwap to allow an add
ADD_FRAC = 0.5                # add 50% of current position
MAX_ADDS = 2
MIN_RANKED = 30               # min assets with dfh20 for a usable XS rank
FUND_LOOKBACK = 180           # trailing-180 mid-rank pct (repo convention)
MIN_WINDOW = 20
BTC_GATE_LOOKBACK = 20


def _r(x, nd=4):
    return round(x, nd) if isinstance(x, float) else x


def mid_rank_pct(window, x):
    """Mid-rank percentile of x vs trailing window (ties count half)."""
    if x is None or not window:
        return None
    return (sum(1 for w in window if w < x)
            + 0.5 * sum(1 for w in window if w == x)) / len(window)


def load_cohort(path):
    """asset_id -> sorted rows {day, close, prev_close, dfh20, funding_pct};
    meta: asset -> {listed, last_bar_date}."""
    series, meta = defaultdict(list), {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            asset = record["asset_id"]
            meta[asset] = {"listed": record.get("listed"),
                           "last_bar_date": record.get("last_bar_date")}
            f = record["features"]
            series[asset].append({
                "day": record["id"].rsplit(":", 1)[-1],
                "close": f["close"]["value"],
                "dfh20": f["dfh20"]["value"],
                "funding": f["last_funding_rate"]["value"],
            })
    for rows in series.values():
        rows.sort(key=lambda r: r["day"])
        for i, r in enumerate(rows):
            r["prev_close"] = rows[i - 1]["close"] if i else None
        fund = [r["funding"] for r in rows]
        for i, r in enumerate(rows):
            fwin = [x for x in fund[max(0, i - FUND_LOOKBACK):i]
                    if x is not None]
            r["funding_pct"] = (mid_rank_pct(fwin, r["funding"])
                                if r["funding"] is not None
                                and len(fwin) >= MIN_WINDOW else None)
    return series, meta


def load_btc_ret20(path):
    """day -> {close, ret20} from the BTC daily csv."""
    rows = []
    with path.open(encoding="utf-8") as stream:
        header = stream.readline()
        for line in stream:
            parts = line.strip().split(",")
            if len(parts) >= 5 and parts[0]:
                rows.append((parts[0], float(parts[4])))
    out = {}
    for i, (day, close) in enumerate(rows):
        ret20 = (close / rows[i - BTC_GATE_LOOKBACK][1] - 1.0
                 if i >= BTC_GATE_LOOKBACK
                 and rows[i - BTC_GATE_LOOKBACK][1] > 0 else None)
        out[day] = {"close": close, "ret20": ret20}
    return out


def build_day_views(series):
    """calendar, rows_by_day, dfh20_pct[day][asset], n_ranked[day]."""
    rows_by_day = defaultdict(dict)
    for asset, rows in series.items():
        for r in rows:
            rows_by_day[r["day"]][asset] = r
    calendar = sorted(rows_by_day)
    dfh_pct, n_ranked = {}, {}
    for day in calendar:
        scored = [(a, r["dfh20"]) for a, r in rows_by_day[day].items()
                  if r["dfh20"] is not None]
        n_ranked[day] = len(scored)
        vals = sorted(v for _, v in scored)
        n = len(vals)
        pct = {}
        for a, v in scored:
            lo = bisect.bisect_left(vals, v)
            eq = bisect.bisect_right(vals, v) - lo
            pct[a] = (lo + 0.5 * eq) / n
        dfh_pct[day] = pct
    return calendar, rows_by_day, dfh_pct, n_ranked


def simulate(calendar, rows_by_day, dfh_pct, n_ranked, btc, meta,
             composite_entry=True):
    """Run the state machine. Returns {equity, campaigns, fills}."""
    cash, qty = 1.0, 0.0
    asset = None
    entry_day = entry_date = None
    cost_qty = 0.0            # sum(qty_i * px_i) for vwap
    max_close = 0.0
    last_close = None
    adds = 0
    entry_equity = None
    equity, campaigns, fills = {}, [], []

    def close_position(day, px, reason):
        nonlocal cash, qty, asset, cost_qty, max_close, last_close
        nonlocal adds, entry_equity
        proceeds = qty * px
        fee = proceeds * FEE
        cash += proceeds - fee
        fills.append({"day": day, "asset": asset, "side": "exit",
                      "qty": qty, "px": px, "fee": fee})
        ret = cash / entry_equity - 1.0
        campaigns.append({
            "asset": asset, "entry_day": entry_day, "exit_day": day,
            "days_held": (dt.date.fromisoformat(day)
                          - entry_date).days,
            "adds": adds, "return": ret, "exit_reason": reason,
            "listed": meta[asset]["listed"],
            "last_bar_date": meta[asset]["last_bar_date"],
        })
        qty = 0.0
        asset = None
        cost_qty = max_close = 0.0
        last_close = None
        adds = 0
        entry_equity = None
        return asset

    exited_today = set()
    for day in calendar:
        btc_r = btc.get(day, {}).get("ret20")
        gate_on = btc_r is not None and btc_r > 0
        ranked_ok = n_ranked.get(day, 0) >= MIN_RANKED
        pcts = dfh_pct.get(day, {})

        # ---------------- open campaign: exits, then pyramid ----------
        if asset is not None:
            row = rows_by_day[day].get(asset)
            reason = None
            px = None
            if row is None or row["close"] is None:
                reason, px = "data_end", last_close
            else:
                last_close = row["close"]
                max_close = max(max_close, last_close)
                pct = pcts.get(asset)
                fpct = row["funding_pct"]
                held = (dt.date.fromisoformat(day) - entry_date).days
                if btc_r is not None and btc_r <= 0:
                    reason = "btc_regime_off"
                elif fpct is not None and fpct >= FUND_EXIT:
                    reason = "funding_euphoria"
                elif last_close < max_close * TRAIL:
                    reason = "trailing_stop_8pct"
                elif held >= TIME_STOP_DAYS:
                    reason = "time_stop_30d"
                elif pct is None or pct < MEDIAN:
                    reason = "rank_below_median"
                if reason:
                    px = last_close
            if reason:
                exited_today.add(asset)
                close_position(day, px, reason)
            elif row is not None:
                # pyramid: +3% vs vwap, gate on, still top quintile
                vwap = cost_qty / qty if qty else None
                unreal = (last_close / vwap - 1.0) if vwap else 0.0
                pct = pcts.get(asset)
                if (adds < MAX_ADDS and unreal >= ADD_TRIGGER and gate_on
                        and pct is not None and pct >= TOP_Q):
                    add_qty = qty * ADD_FRAC
                    notional = add_qty * last_close
                    fee = notional * FEE
                    cash -= notional + fee   # margined perp-style
                    qty += add_qty
                    cost_qty += notional
                    adds += 1
                    fills.append({"day": day, "asset": asset, "side": "add",
                                  "qty": add_qty, "px": last_close,
                                  "fee": fee})

        # ---------------- flat: entry scan ------------------------------
        if asset is None and gate_on and ranked_ok and cash > 0:
            cands = []
            for a, r in rows_by_day[day].items():
                if a in exited_today:
                    continue
                pct = pcts.get(a)
                if pct is None or pct < TOP_Q:
                    continue
                if composite_entry:
                    d = r["dfh20"]
                    if not (PULLBACK_LO <= d <= PULLBACK_HI):
                        continue
                    if r["prev_close"] is None or r["close"] <= r["prev_close"]:
                        continue
                    if (r["funding_pct"] is not None
                            and r["funding_pct"] >= FUND_ENTRY_MAX):
                        continue
                cands.append((pct, a))
            if cands:
                cands.sort(key=lambda x: (-x[0], x[1]))  # top rank wins
                asset = cands[0][1]
                px = rows_by_day[day][asset]["close"]
                entry_equity = cash          # deploy 100% of equity
                qty = cash / (px * (1.0 + FEE))
                fee = qty * px * FEE
                cash = 0.0
                fills.append({"day": day, "asset": asset, "side": "entry",
                              "qty": qty, "px": px, "fee": fee})
                cost_qty = qty * px
                entry_day, entry_date = day, dt.date.fromisoformat(day)
                max_close = last_close = px
                adds = 0
        exited_today.clear()

        equity[day] = cash + qty * last_close if asset is not None else cash

    # force-close anything still open at sample end
    if asset is not None:
        close_position(calendar[-1], last_close, "end_of_sample")
        equity[calendar[-1]] = cash
    return {"equity": equity, "campaigns": campaigns, "fills": fills}


def equity_metrics(equity):
    days = sorted(equity)
    vals = [equity[d] for d in days]
    peak, mdd = vals[0], 0.0
    for v in vals:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak if peak else 0.0)
    return {"first_day": days[0], "last_day": days[-1], "n_days": len(days),
            "final_equity": vals[-1], "total_return": vals[-1] - 1.0,
            "max_drawdown_frac": mdd}


def arm_metrics(sim, calendar):
    camps = sim["campaigns"]
    rets = [c["return"] for c in camps]
    n = len(camps)
    wins = sum(1 for r in rets if r > 0)
    reasons = defaultdict(int)
    for c in camps:
        reasons[c["exit_reason"]] += 1
    # time in market: days with an open campaign
    open_days = set()
    for c in camps:
        d0 = dt.date.fromisoformat(c["entry_day"])
        d1 = dt.date.fromisoformat(c["exit_day"])
        cur = d0
        while cur <= d1:
            open_days.add(cur.isoformat())
            cur += dt.timedelta(days=1)
    years = defaultdict(list)
    for c in camps:
        years[c["exit_day"][:4]].append(c["return"])
    per_year = {}
    for y, rs in sorted(years.items()):
        comp = math.prod(1 + r for r in rs) - 1.0
        per_year[y] = {"n_campaigns": len(rs),
                       "win_rate": sum(1 for r in rs if r > 0) / len(rs),
                       "mean_return": sum(rs) / len(rs),
                       "compounded_return": comp}
    m = equity_metrics(sim["equity"])
    m.update({
        "n_campaigns": n,
        "win_rate": wins / n if n else None,
        "mean_campaign_return": (sum(rets) / n) if n else None,
        "max_campaign_return": max(rets) if rets else None,
        "min_campaign_return": min(rets) if rets else None,
        "time_in_market_frac": len(open_days) / len(calendar),
        "avg_adds_per_campaign": (sum(c["adds"] for c in camps) / n
                                 if n else None),
        "exit_reason_histogram": dict(sorted(reasons.items())),
        "per_year": per_year,
    })
    return m


def write_protocol_and_auth():
    protocol = {
        "schema_version": "nanojev-financial-campaign-sim-protocol-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "T119: rolling-campaign simulator — the full entry -> "
                   "pyramid -> exit state machine — measured on the "
                   "277-symbol mega PIT cohort vs a no-entry-filter "
                   "sibling and buy-and-hold BTC.",
        "cohort": {
            "path": "data/perp_pit_mega_v1/records.jsonl",
            "assets": "277 USDT-M perps, 2021-01-01 -> 2025-12-30, close "
                      "basis; includes 11 delisted/renamed early-stoppers "
                      "(listed/last_bar_date on every record)",
            "btc_gate_source": "data/rc_futures_v1/BTC/BTCUSDT_1d.csv "
                               "close/close[t-20]-1",
        },
        "definitions": {
            "concurrency": "MAX_CONCURRENT=1 — one campaign at a time, "
                           "full equity deployed at entry (chosen over "
                           "max-2 for a clean single-book equity curve)",
            "entry": "BTC ret20>0 AND dfh20 XS mid-rank pct>=0.8 "
                     "(top quintile, >=30 ranked) AND dfh20 in "
                     "[-0.12,-0.03] AND close>prev_close AND "
                     "(funding_pct<0.8 if funded). Highest rank wins; "
                     "fill at signal-day close.",
            "pyramid": "unrealized>=+3% vs vwap AND gate-on AND still "
                       "top-quintile -> add 50% of position (qty*0.5 at "
                       "mark), max 2 adds, margined perp-style (cash "
                       "goes negative, ~1.75x max notional)",
            "exit_order": ["data_end (no bar -> last close)",
                           "btc_regime_off (BTC ret20<=0)",
                           "funding_euphoria (funding_pct>=0.95)",
                           "trailing_stop_8pct (close<0.92*max_close)",
                           "time_stop_30d (>=30 calendar days held)",
                           "rank_below_median (dfh20 pct<0.5/unranked)",
                           "end_of_sample (force-close at tail)"],
            "funding_pct": "per-asset mid-rank pct of last_funding_rate "
                           "vs trailing-180 daily values (MIN_WINDOW=20)",
            "costs": "5bps per fill; funding carry skipped (signal only)",
            "fills": "all at day-t close on day-t PIT features — the "
                     "same close basis as the cohort labels; 'next bar "
                     "open' approximated by that close on daily bars",
        },
        "arms": {
            "composite": "full entry filter",
            "no_filter": "entry = gate-on AND top-quintile only; "
                         "identical pyramid/exits — isolates the entry "
                         "filter's value",
            "btc_hold": "buy-and-hold BTC over the sim calendar",
        },
        "metrics": ["n campaigns", "win rate", "mean/max campaign return",
                    "final compounded equity", "max drawdown (daily "
                    "equity)", "time-in-market %", "per-year table by "
                    "exit year", "avg adds/campaign",
                    "exit-reason histogram", "determinism double-run"],
        "caveats": [
            "close basis + daily bars: intra-day stops are coarse — an "
            "8% trail can lose far more than 8% through a gap day",
            "margined adds = implicit leverage up to ~1.75x notional",
            "early-stopped symbols are kept (mega includes delisted — a "
            "PLUS for survivorship honesty); campaigns on them are "
            "reported explicitly",
            "second-hand archive copy, not an as-of vintage",
        ],
        "forbidden": ["fitting", "trading", "profitability claims",
                      "protocol edits post-run", "network"],
    }
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(protocol, indent=2, sort_keys=True)
                        + "\n", encoding="utf-8")
    sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    auth = {
        "schema_version": "nanojev-financial-campaign-sim-authorization-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "protocol_sha256": sha,
        "protocol_path": "research/financial_campaign_sim_protocol_v1.json",
        "decision": "approved_for_measurement",
        "measurement_authorized": True,
        "fit_authorized": False,
        "independent_reviewer": {
            "id": "project-owner",
            "independence": "owner_self_authorization_not_independent_review",
            "note": "Owner directed T119: build the rolling-campaign "
                    "simulator (entry->pyramid->exit state machine) and "
                    "measure it on the mega cohort (delegated task).",
        },
        "scope": {
            "permitted": "PIT-safe event-driven daily simulation on "
                         "data/perp_pit_mega_v1/records.jsonl plus the "
                         "BTC 1d csv: composite-entry arm, no-filter "
                         "sibling, BTC buy-and-hold, campaign metrics "
                         "per the pinned protocol.",
            "not_permitted": "No fitting/trading/profitability claims/"
                             "protocol edits; no network; no other files "
                             "modified.",
        },
        "network_model_calls": 0,
        "order_submission_authorized": False,
        "live_trading_authorized": False,
    }
    AUTH.parent.mkdir(parents=True, exist_ok=True)
    AUTH.write_text(json.dumps(auth, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return sha


def run_all(series, meta, calendar, rows_by_day, dfh_pct, n_ranked, btc):
    """Both sim arms + BTC hold, each run twice for determinism."""
    results = {}
    for arm, composite in (("composite", True), ("no_filter", False)):
        sim1 = simulate(calendar, rows_by_day, dfh_pct, n_ranked, btc,
                        meta, composite_entry=composite)
        sim2 = simulate(calendar, rows_by_day, dfh_pct, n_ranked, btc,
                        meta, composite_entry=composite)
        det = json.dumps(sim1, sort_keys=True) == json.dumps(
            sim2, sort_keys=True)
        results[arm] = {"sim": sim1, "metrics": arm_metrics(sim1, calendar),
                        "deterministic_double_run": det}
    # BTC buy-and-hold over the sim calendar
    first = next((d for d in calendar if d in btc), calendar[0])
    bh = {d: btc[d]["close"] / btc[first]["close"]
          for d in calendar if d in btc}
    results["btc_hold"] = {"metrics": equity_metrics(bh)}
    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", type=pathlib.Path, default=COHORT)
    parser.add_argument("--btc", type=pathlib.Path, default=BTC_CSV)
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    args = parser.parse_args()

    protocol_sha = write_protocol_and_auth()
    report = {
        "schema_version": "nanojev-financial-campaign-sim-v1",
        "task": "T119 rolling-campaign simulator on the mega cohort",
        "protocol_path": "research/financial_campaign_sim_protocol_v1.json",
        "protocol_sha256": protocol_sha,
        "authorization_path":
            "results/financial_campaign_sim_authorization_v1.json",
        "parameters": {
            "max_concurrent_campaigns": MAX_CONCURRENT,
            "fee_per_fill": FEE, "top_quintile": TOP_Q,
            "pullback_dfh20_window": [PULLBACK_LO, PULLBACK_HI],
            "funding_entry_max": FUND_ENTRY_MAX,
            "funding_exit": FUND_EXIT, "trail": TRAIL,
            "time_stop_days": TIME_STOP_DAYS,
            "add_trigger": ADD_TRIGGER, "add_frac": ADD_FRAC,
            "max_adds": MAX_ADDS, "min_ranked_per_day": MIN_RANKED,
            "funding_pct_lookback": FUND_LOOKBACK,
            "concurrency_note": "ONE campaign at a time; adds are "
                                "margined perp-style (implicit leverage "
                                "<= ~1.75x), funding carry skipped",
        },
    }
    if not args.cohort.exists() or not args.btc.exists():
        report["status"] = ("SKIPPED: cohort or BTC series missing; "
                            "nothing was fabricated")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True)
                            + "\n")
        print(json.dumps({"status": report["status"]}))
        return 0

    series, meta = load_cohort(args.cohort)
    btc = load_btc_ret20(args.btc)
    calendar, rows_by_day, dfh_pct, n_ranked = build_day_views(series)
    early_stopped = sorted(a for a, m in meta.items() if m["listed"] is False)
    report["cohort_description"] = {
        "symbols": len(series),
        "symbols_with_funding": sum(
            1 for a, rows in series.items()
            if any(r["funding"] is not None for r in rows)),
        "span": {"first": calendar[0], "last": calendar[-1],
                 "days": len(calendar)},
        "early_stopped_symbols": {"count": len(early_stopped),
                                  "assets": early_stopped},
    }
    report["status"] = "ran"

    res = run_all(series, meta, calendar, rows_by_day, dfh_pct, n_ranked,
                  btc)
    for arm in ("composite", "no_filter"):
        camps = res[arm]["sim"]["campaigns"]
        delisted_camps = [c for c in camps if c["listed"] is False]
        res[arm]["campaigns_on_early_stopped_symbols"] = {
            "count": len(delisted_camps),
            "campaigns": delisted_camps,
        }
        # sensitivity: compounded equity if early-stopped symbols were
        # untradeable (campaigns skipped, capital flat those days)
        res[arm]["metrics"]["equity_excl_early_stopped_campaigns"] = (
            math.prod(1 + c["return"] for c in camps
                      if c["listed"] is not False))
        res[arm]["campaign_log"] = camps
        res[arm]["n_fills"] = len(res[arm]["sim"]["fills"])
        del res[arm]["sim"]

    report["arms"] = res
    comp, nf, bh = (res["composite"]["metrics"],
                    res["no_filter"]["metrics"],
                    res["btc_hold"]["metrics"])
    beats_sibling = comp["final_equity"] > nf["final_equity"]
    beats_btc = comp["final_equity"] > bh["final_equity"]
    report["verdict"] = {
        "composite_beats_no_filter_sibling": beats_sibling,
        "composite_beats_btc_hold": beats_btc,
        "summary": (
            f"composite final equity {_r(comp['final_equity'])} "
            f"(n={comp['n_campaigns']}, win {_r(comp['win_rate'])}, "
            f"mdd {_r(comp['max_drawdown_frac'])}, tim "
            f"{_r(comp['time_in_market_frac'])}) vs no-filter "
            f"{_r(nf['final_equity'])} (n={nf['n_campaigns']}, win "
            f"{_r(nf['win_rate'])}, mdd {_r(nf['max_drawdown_frac'])}) "
            f"vs BTC hold {_r(bh['final_equity'])} (mdd "
            f"{_r(bh['max_drawdown_frac'])}) -> composite "
            f"{'BEATS' if beats_sibling else 'LOSES TO'} its sibling, "
            f"{'BEATS' if beats_btc else 'LOSES TO'} BTC hold"),
    }
    report["honesty"] = {
        "close_basis": ("all fills at day close; 'next bar open' is "
                        "approximated by the same close — daily "
                        "granularity makes intra-day stops coarse (an "
                        "8% trail can realize a much larger loss on a "
                        "gap day)"),
        "survivorship": ("the mega cohort INCLUDES delisted/renamed "
                         "early-stoppers — a plus vs listed-only "
                         "universes; campaigns on early-stopped symbols "
                         "are listed per arm (renames may double-count "
                         "one economic asset)"),
        "funding_carry": "skipped — funding_pct is a positioning signal "
                         "only, no funding cashflows modeled",
        "not_an_asof_vintage": "second-hand archive copy (rc_futures_v1)",
        "not_live": "no orders, no account, no broker, no trading API",
        "no_profitability_claim": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True)
                        + "\n", encoding="utf-8")
    brief = {"status": "ran", "out": str(args.out),
             "verdict": report["verdict"]["summary"],
             "composite_deterministic":
                 res["composite"]["deterministic_double_run"],
             "no_filter_deterministic":
                 res["no_filter"]["deterministic_double_run"]}
    print(json.dumps(brief, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
