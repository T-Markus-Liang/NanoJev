#!/usr/bin/env python3
"""4h rolling-compounding campaign sim — OUR entry vs THEIR entry on THEIR
roll mechanics (T121).

The sibling project's engine (their paper, summarized) is a long-only 4h
trend system on Binance USD-M perps: enter on a shallow pullback off the
30-bar high reclaiming the pullback midpoint under a low-vol gate, then
compound winners — each time price reaches trigger x cost_basis, roll a
fraction alpha of the freed margin back into the position, sweep the rest
to a safe pool, and ratchet the stop up to the new blended cost basis;
leverage cap L, time-stop after 120 bars without a roll. This script runs
THEIR roll mechanics on OUR 4h mega data
(``data/rc_futures_v1/{SYM}/{SYM}USDT_4h.csv``) and swaps the entry rule:
our daily-state port vs their E1, on an identical engine, plus no-roll
flat siblings to isolate what the rolling itself adds.

UNIVERSE (documented choice): the 52 funding-covered symbols — every
``{SYM}/funding.csv`` with a matching 4h csv. Chosen over top-50-by-
quote_volume because (a) our-entry carries the T119 stealth funding
filter, which needs funding.csv, and (b) a full-sample volume ranking is
a lookahead/survivorship construct. EOS is delisted mid-sample (last bar
2025-05-21) and is handled via the data_end exit.

MASTER CLOCK: BTC 4h bar timestamps
(``data/rc_futures_v1/BTC/BTCUSDT_4h.csv``). All signals are evaluated on
the completed bar; close-basis fills use that bar's close, intrabar
events (stop, roll trigger) use the bar's open/low/high — same
close-basis convention as the daily campaign sim.

BTC GATE (hybrid): BTC close > SMA(120 bars) = the 4h analog of the 20d
trend gate (20 days x 6 bars). Gate-off exits at the bar close and blocks
entries.

ENTRY ARM A — ``our_state`` (port of the T119 daily composite):
  dfh30 = close/max(close, prior 30 bars) - 1  (strictly prior, repo conv.)
  qualify iff  XS mid-rank pct of dfh30 >= 0.8 (top quintile, >=30 ranked)
           AND -0.12 <= dfh30 <= -0.03 AND close > prev_close (recovering)
           AND (funding_pct < 0.8 if computable)
  funding_pct = mid-rank pct of the last funding rate vs its trailing
  540 funding observations (~180d x 3/day, MIN_WINDOW=20).
  Rank: highest dfh30 pct, tie by symbol.

ENTRY ARM B — ``sibling_e1`` (their E1, 4h-native):
  over the last 30 bars INCLUDING current: hh = max(high); j = last bar
  attaining hh; plow = min(low[j..i]); depth = (hh-plow)/hh in [0.05,0.12]
  (shallow pullback 5-12% off the 30-bar high); mid = (hh+plow)/2;
  reclaim = prev_close <= mid < close (cross above the pullback
  midpoint); gate atrp = ATR84/close < 0.02 (ATR84 = mean TR of last 84
  bars). Rank: shallowest depth first, tie by symbol.

ROLL MECHANICS (theirs, simplified deterministic margin-account model):
  entry allocates equity A = free_cash/free_slots (even slot split);
  posts ALL of A as campaign margin; buys notional = 1.0 x A at the close
  (qty = A/px) — leverage cap L=3 bounds the position, so initial margin
  requirement is A/3 and free margin is A - q*px/L.
  On any bar with high >= cost_basis*TRIGGER (1.2): roll at the trigger
  price P_t = cost_basis*1.2 —
    M     = margin_cash + q*(P_t - vwap)        (account equity at P_t)
    free  = M - q*P_t/L                          (releasable at 3x cap)
    add   = ALPHA*free*L notional at P_t (alpha=0.6 rolled back as 3x
            margin), fee 5bps on add notional paid from margin
    sweep = (1-ALPHA)*free transferred to the campaign safe pool (a
            collateral withdrawal, not a fill -> no fee)
    vwap' = blended cost of the enlarged position; stop ratchets UP to
    max(stop, vwap'); roll timer resets. Loops while high >= vwap'*1.2
  Stop: initial 0.92*entry_px (8% disaster stop — matches their own
  warning that a ~8% adverse move at 3x nears maintenance margin);
  ratchets to each new blended cost basis. Intrabar fill: open <= stop
  -> exit at open (gap through); else low <= stop -> exit at stop.
  Exits (checked per bar): data_end (no bar AND ts beyond last bar ->
  last close) > stop (intrabar) > btc_gate_off (at close) > time_stop
  (>=120 master bars since last roll, at close). Stop is checked BEFORE
  rolls on the same bar (conservative — intrabar order is unknowable).
  Max 2 concurrent campaigns; same-bar re-entry into the just-exited
  symbol is blocked.

COSTS: 5bps per fill (entry, each roll add, exit). Funding carry NOT
modeled (funding is a positioning signal only). LIQUIDATION NOT MODELED:
at 3x margin cap the account can go margin-negative through a gap — the
sibling paper itself warns a ~8% adverse move approaches maintenance
margin; reported returns are accounting-basis only.

ARMS: our_state / sibling_e1 x {roll, flat(rolls disabled)} + btc_hold
4h baseline. The flat siblings run the identical engine with the roll
step skipped — they isolate whether the rolling adds value over simply
holding the campaign at 4h scale.

Metrics per arm: campaigns, win rate, geo-mean campaign multiple, final
equity, maxDD on the 4h equity curve, per-year table (by exit year),
roll-count distribution, exit-reason histogram, time-in-market.
Determinism: every sim arm is run twice and must serialize identically.

Artifacts: frozen protocol ``research/financial_campaign_4h_protocol_v1.json``
plus owner self-authorization pinning it by sha256
(``results/financial_campaign_4h_authorization_v1.json``), written before
measurement. Measurement only: no fitting, no trading, no network.
"""
import argparse
import bisect
import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data/rc_futures_v1"
OUT = ROOT / "results/financial_campaign_4h_v1.json"
PROTOCOL = ROOT / "research/financial_campaign_4h_protocol_v1.json"
AUTH = ROOT / "results/financial_campaign_4h_authorization_v1.json"

# ---- frozen parameters -------------------------------------------------
MAX_CONCURRENT = 2        # documented choice: up to 2 open campaigns
FEE = 0.0005              # 5bps per fill
LEVERAGE = 3.0            # margin cap L
TRIGGER = 1.2             # roll trigger = cost_basis x 1.2 (low end of 1.2-1.5)
ALPHA = 0.6               # fraction of freed margin rolled back in
INIT_STOP = 0.92          # initial disaster stop = 8% below entry cost
TIME_STOP_BARS = 120      # exit after 120 master bars without a roll
MAX_ROLLS_PER_BAR = 8     # safety bound on same-bar roll loop
BTC_SMA = 120             # 20d x 6 — the 4h analog of the 20d gate
DFH_WIN = 30              # our dfh port: 30 bars ~ 5d
ATR_WIN = 84              # their atrp gate: ATR over 84 bars
ATRP_MAX = 0.02           # atrp = ATR84/close < 0.02
PB_LO, PB_HI = 0.05, 0.12        # their E1 pullback depth band
OUR_DFH_LO, OUR_DFH_HI = -0.12, -0.03   # our recovering-state band
TOP_Q = 0.8               # top XS dfh30 quintile
MIN_RANKED = 30           # min assets for a usable XS rank
FUND_LOOKBACK = 540       # ~180d x 3 funding obs/day
MIN_WINDOW = 20
FUND_ENTRY_MAX = 0.8


def _r(x, nd=4):
    return round(x, nd) if isinstance(x, float) else x


def mid_rank_pct(window, x):
    """Mid-rank percentile of x vs trailing window (ties count half)."""
    if x is None or not window:
        return None
    return (sum(1 for w in window if w < x)
            + 0.5 * sum(1 for w in window if w == x)) / len(window)


def load_bars(path):
    """-> list of {ts,o,h,l,c} sorted by ts."""
    rows = []
    with path.open(encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        for parts in reader:
            if not parts or not parts[0]:
                continue
            rows.append({
                "ts": parts[idx["timestamp"]],
                "o": float(parts[idx["open"]]),
                "h": float(parts[idx["high"]]),
                "l": float(parts[idx["low"]]),
                "c": float(parts[idx["close"]]),
            })
    rows.sort(key=lambda r: r["ts"])
    return rows


def load_funding(path):
    """-> (ts_list, rate_list) sorted by ts."""
    ts, rates = [], []
    with path.open(encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        for parts in reader:
            if not parts or not parts[0]:
                continue
            ts.append(parts[idx["timestamp"]])
            rates.append(float(parts[idx["funding_rate"]]))
    order = sorted(range(len(ts)), key=lambda i: ts[i])
    return [ts[i] for i in order], [rates[i] for i in order]


def funding_pct_series(ts, rates):
    """mid-rank pct of rate[i] vs strictly-prior FUND_LOOKBACK obs."""
    pct = [None] * len(rates)
    for i, x in enumerate(rates):
        win = rates[max(0, i - FUND_LOOKBACK):i]
        if len(win) >= MIN_WINDOW:
            pct[i] = mid_rank_pct(win, x)
    # bar-index -> pct of last funding obs at-or-before bar ts is done by
    # the caller via bisect on ts.
    return pct


def enrich_symbol(bars):
    """Attach per-bar features in place: prev_close, dfh30, atr84, atrp,
    e1 components (hh30/plow/depth/mid/reclaim/qual), e1 rank key."""
    n = len(bars)
    tr = [None] * n
    for i, b in enumerate(bars):
        if i:
            pc = bars[i - 1]["c"]
            b["prev_c"] = pc
            tr[i] = max(b["h"] - b["l"], abs(b["h"] - pc),
                        abs(b["l"] - pc))
        else:
            b["prev_c"] = None
        # our-state: dfh over strictly-prior 30 closes (repo convention)
        if i >= DFH_WIN:
            prior = max(x["c"] for x in bars[i - DFH_WIN:i])
            b["dfh"] = b["c"] / prior - 1.0 if prior > 0 else None
        else:
            b["dfh"] = None
        # their atrp gate: mean TR of last 84 bars (incl current)
        if i >= ATR_WIN:
            win = tr[i - ATR_WIN + 1:i + 1]
            b["atrp"] = (sum(win) / len(win)) / b["c"] if b["c"] else None
        else:
            b["atrp"] = None
        # their E1: 30-bar window inclusive of current bar
        b["e1"] = None
        if i >= DFH_WIN - 1 and i >= 1:
            lo = max(0, i - DFH_WIN + 1)
            window = bars[lo:i + 1]
            hh = max(x["h"] for x in window)
            jstar = max(j for j in range(lo, i + 1)
                        if bars[j]["h"] == hh)
            plow = min(x["l"] for x in bars[jstar:i + 1])
            depth = (hh - plow) / hh if hh > 0 else None
            mid = (hh + plow) / 2.0
            reclaim = (b["prev_c"] is not None
                       and b["prev_c"] <= mid < b["c"])
            qual = (depth is not None and PB_LO <= depth <= PB_HI
                    and reclaim
                    and b["atrp"] is not None and b["atrp"] < ATRP_MAX)
            b["e1"] = {"depth": depth, "qual": qual}


def load_universe(data_dir):
    """52 funding-covered symbols -> {sym: bars+features+idx+fund_pct}."""
    universe = {}
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        fpath = d / "funding.csv"
        bpath = d / f"{d.name}USDT_4h.csv"
        if not fpath.exists() or not bpath.exists():
            continue
        bars = load_bars(bpath)
        if not bars:
            continue
        enrich_symbol(bars)
        fts, frates = load_funding(fpath)
        fpct = funding_pct_series(fts, frates)
        for b in bars:
            fi = bisect.bisect_right(fts, b["ts"]) - 1
            b["fund_pct"] = fpct[fi] if fi >= 0 else None
        universe[d.name] = {
            "bars": bars,
            "idx": {b["ts"]: i for i, b in enumerate(bars)},
            "first_ts": bars[0]["ts"], "last_ts": bars[-1]["ts"],
        }
    return universe


def build_dfh_pct(universe, calendar):
    """XS mid-rank pct of dfh30 per ts -> {ts: {sym: pct}}, {ts: n}."""
    pct_by_ts, n_ranked = {}, {}
    for ts in calendar:
        scored = []
        for sym, u in universe.items():
            i = u["idx"].get(ts)
            if i is None:
                continue
            d = u["bars"][i]["dfh"]
            if d is not None:
                scored.append((sym, d))
        n_ranked[ts] = len(scored)
        vals = sorted(v for _, v in scored)
        n = len(vals)
        pct = {}
        for s, v in scored:
            lo = bisect.bisect_left(vals, v)
            eq = bisect.bisect_right(vals, v) - lo
            pct[s] = (lo + 0.5 * eq) / n
        pct_by_ts[ts] = pct
    return pct_by_ts, n_ranked


def simulate(universe, calendar, btc_close, btc_sma, dfh_pct, n_ranked,
             entry_rule, rolls_enabled):
    """One arm. entry_rule in {'our','e1'}; returns {equity, campaigns,
    fills}."""
    cash = 1.0
    open_camps = []          # list of campaign dicts
    campaigns, fills = [], []
    equity = {}
    exited_this_bar = set()
    last_px = {}             # sym -> last seen close (for marks/data_end)

    def camp_value(c, px):
        return c["m_c"] + c["q"] * (px - c["vwap"]) + c["swept"]

    def close_camp(c, ts, px, reason):
        nonlocal cash
        exit_fee = c["q"] * px * FEE
        returned = camp_value(c, px) - exit_fee
        c["fees"] += exit_fee
        cash += returned
        fills.append({"ts": ts, "sym": c["sym"], "side": "exit",
                      "px": px, "fee": exit_fee})
        campaigns.append({
            "sym": c["sym"], "entry_ts": c["entry_ts"], "exit_ts": ts,
            "bars_held": c["bars_held"], "rolls": c["rolls"],
            "return": returned / c["alloc"] - 1.0,
            "exit_reason": reason, "fees": c["fees"],
            "swept": c["swept"], "alloc": c["alloc"],
        })

    for t, ts in enumerate(calendar):
        sma = btc_sma[t]
        gate_on = sma is not None and btc_close[t] > sma

        # ------------- open campaigns: intrabar exits, rolls, close ----
        still_open = []
        for c in open_camps:
            u = universe[c["sym"]]
            i = u["idx"].get(ts)
            if i is None:
                if ts > u["last_ts"]:
                    c["bars_held"] += 1
                    close_camp(c, ts, last_px[c["sym"]], "data_end")
                    exited_this_bar.add(c["sym"])
                    continue
                still_open.append(c)          # interior gap: carry mark
                continue
            bar = u["bars"][i]
            last_px[c["sym"]] = bar["c"]
            c["bars_held"] += 1
            done = False
            # (a) intrabar stop — checked before rolls (conservative)
            if bar["o"] <= c["stop"]:
                close_camp(c, ts, bar["o"], "stop_gap_open")
                done = True
            elif bar["l"] <= c["stop"]:
                close_camp(c, ts, c["stop"], "stop_cost_basis")
                done = True
            if done:
                exited_this_bar.add(c["sym"])
                continue
            # (b) roll loop on the bar high at trigger prices
            if rolls_enabled:
                guard = 0
                while (guard < MAX_ROLLS_PER_BAR
                       and bar["h"] >= c["vwap"] * TRIGGER):
                    p_t = c["vwap"] * TRIGGER
                    m = c["m_c"] + c["q"] * (p_t - c["vwap"])
                    free = m - c["q"] * p_t / LEVERAGE
                    if free <= 0:
                        break
                    add_notional = ALPHA * free * LEVERAGE
                    dq = add_notional / p_t
                    fee = add_notional * FEE
                    sweep = (1.0 - ALPHA) * free
                    c["m_c"] -= sweep + fee
                    c["swept"] += sweep
                    c["fees"] += fee
                    c["vwap"] = ((c["q"] * c["vwap"] + dq * p_t)
                                 / (c["q"] + dq))
                    c["q"] += dq
                    c["stop"] = max(c["stop"], c["vwap"])
                    c["rolls"] += 1
                    c["last_roll_t"] = t
                    fills.append({"ts": ts, "sym": c["sym"],
                                  "side": "roll_add", "px": p_t,
                                  "fee": fee, "sweep": sweep})
                    guard += 1
            # (c) close-basis exits
            if not gate_on:
                close_camp(c, ts, bar["c"], "btc_gate_off")
                exited_this_bar.add(c["sym"])
                continue
            if t - c["last_roll_t"] >= TIME_STOP_BARS:
                close_camp(c, ts, bar["c"], "time_stop_120bar")
                exited_this_bar.add(c["sym"])
                continue
            still_open.append(c)
        open_camps = still_open

        # ------------------------- entries -----------------------------
        free_slots = MAX_CONCURRENT - len(open_camps)
        if gate_on and free_slots > 0 and cash > 0:
            cands = []
            open_syms = {c["sym"] for c in open_camps}
            for sym in sorted(universe):
                if sym in open_syms or sym in exited_this_bar:
                    continue
                u = universe[sym]
                i = u["idx"].get(ts)
                if i is None:
                    continue
                bar = u["bars"][i]
                if entry_rule == "our":
                    if n_ranked.get(ts, 0) < MIN_RANKED:
                        break
                    pct = dfh_pct[ts].get(sym)
                    d = bar["dfh"]
                    if pct is None or pct < TOP_Q or d is None:
                        continue
                    if not (OUR_DFH_LO <= d <= OUR_DFH_HI):
                        continue
                    if bar["prev_c"] is None or bar["c"] <= bar["prev_c"]:
                        continue
                    if (bar["fund_pct"] is not None
                            and bar["fund_pct"] >= FUND_ENTRY_MAX):
                        continue
                    cands.append((-pct, sym))   # highest pct first
                else:                            # sibling E1
                    e1 = bar["e1"]
                    if e1 is None or not e1["qual"]:
                        continue
                    cands.append((e1["depth"], sym))  # shallowest first
            cands.sort()
            for _, sym in cands[:free_slots]:
                u = universe[sym]
                bar = u["bars"][u["idx"][ts]]
                slots_left = MAX_CONCURRENT - len(open_camps)
                alloc = cash / slots_left
                px = bar["c"]
                fee = alloc * FEE
                open_camps.append({
                    "sym": sym, "alloc": alloc,
                    "m_c": alloc - fee, "q": alloc / px,
                    "vwap": px, "stop": INIT_STOP * px,
                    "swept": 0.0, "fees": fee, "rolls": 0,
                    "entry_ts": ts, "bars_held": 0, "last_roll_t": t,
                })
                cash -= alloc
                fills.append({"ts": ts, "sym": sym, "side": "entry",
                              "px": px, "fee": fee})
                last_px[sym] = px
        exited_this_bar.clear()

        eq = cash
        for c in open_camps:
            eq += camp_value(c, last_px.get(c["sym"], c["vwap"]))
        equity[ts] = eq

    # force-close anything still open at sample end
    for c in open_camps:
        close_camp(c, calendar[-1], last_px[c["sym"]], "end_of_sample")
        equity[calendar[-1]] = cash
    return {"equity": equity, "campaigns": campaigns, "fills": fills}


def equity_metrics(equity):
    tss = sorted(equity)
    vals = [equity[t] for t in tss]
    peak, mdd = vals[0], 0.0
    for v in vals:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak if peak else 0.0)
    return {"first_ts": tss[0], "last_ts": tss[-1], "n_bars": len(tss),
            "final_equity": vals[-1], "total_return": vals[-1] - 1.0,
            "max_drawdown_frac": mdd}


def arm_metrics(sim, calendar):
    camps = sim["campaigns"]
    rets = [c["return"] for c in camps]
    n = len(camps)
    wins = sum(1 for r in rets if r > 0)
    ruined = sum(1 for r in rets if r <= -1.0)
    geo = (math.prod(1.0 + r for r in rets) ** (1.0 / n)
           if n and not ruined else None)
    reasons = defaultdict(int)
    roll_dist = defaultdict(int)
    for c in camps:
        reasons[c["exit_reason"]] += 1
        roll_dist[str(c["rolls"])] += 1
    open_bars = set()
    ts_index = {t: i for i, t in enumerate(calendar)}
    for c in camps:
        i0 = ts_index.get(c["entry_ts"])
        i1 = ts_index.get(c["exit_ts"])
        if i0 is not None and i1 is not None:
            open_bars.update(range(i0, i1 + 1))
    years = defaultdict(list)
    for c in camps:
        years[c["exit_ts"][:4]].append(c["return"])
    per_year = {}
    for y, rs in sorted(years.items()):
        per_year[y] = {
            "n_campaigns": len(rs),
            "win_rate": sum(1 for r in rs if r > 0) / len(rs),
            "mean_return": sum(rs) / len(rs),
            "compounded_return": math.prod(1 + r for r in rs) - 1.0,
        }
    m = equity_metrics(sim["equity"])
    m.update({
        "n_campaigns": n,
        "win_rate": wins / n if n else None,
        "mean_campaign_return": sum(rets) / n if n else None,
        "geo_mean_campaign_mult": geo,
        "max_campaign_return": max(rets) if rets else None,
        "min_campaign_return": min(rets) if rets else None,
        "ruined_campaigns_le_minus_100pct": ruined,
        "time_in_market_frac": len(open_bars) / len(calendar),
        "avg_rolls_per_campaign": (sum(c["rolls"] for c in camps) / n
                                   if n else None),
        "roll_count_distribution": dict(sorted(
            roll_dist.items(), key=lambda kv: int(kv[0]))),
        "exit_reason_histogram": dict(sorted(reasons.items())),
        "per_year": per_year,
    })
    return m


def write_protocol_and_auth():
    protocol = {
        "schema_version": "nanojev-financial-campaign-4h-protocol-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "T121: port the sibling project's rolling-compounding "
                   "engine to our 4h mega data and run OUR entry selection "
                   "vs THEIR E1 entry on identical roll mechanics; flat "
                   "no-roll siblings isolate the roll engine's value.",
        "universe": {
            "choice": "the 52 funding-covered symbols of "
                      "data/rc_futures_v1 (every {SYM}/funding.csv with a "
                      "{SYM}USDT_4h.csv) — chosen over top-50 "
                      "quote_volume because our-entry carries the funding "
                      "stealth filter (needs funding.csv) and a "
                      "full-sample volume ranking is lookahead",
            "span": "2021-01-01 -> 2025-12-31 4h bars; EOS delists "
                    "2025-05-21 (data_end exit)",
            "master_clock": "BTC 4h bar timestamps "
                            "(data/rc_futures_v1/BTC/BTCUSDT_4h.csv)",
        },
        "definitions": {
            "btc_gate": "BTC close > SMA(120 4h bars) — the 4h analog of "
                        "the 20d trend gate; gate-off exits at close and "
                        "blocks entries",
            "entry_our_state": "XS dfh30 mid-rank pct>=0.8 (>=30 ranked) "
                               "AND dfh30 in [-0.12,-0.03] AND "
                               "close>prev_close AND (funding_pct<0.8 if "
                               "computable). dfh30 = close/max(prior 30 "
                               "closes)-1. funding_pct = mid-rank of last "
                               "funding rate vs trailing 540 obs. Rank: "
                               "highest pct, tie by symbol.",
            "entry_sibling_e1": "30-bar window incl current: hh=max(high), "
                                "plow=min(low) since last hh bar, depth="
                                "(hh-plow)/hh in [0.05,0.12], mid=(hh+"
                                "plow)/2, reclaim=prev_close<=mid<close, "
                                "atrp=ATR84/close<0.02 (mean TR, 84 bars). "
                                "Rank: shallowest depth, tie by symbol.",
            "roll_engine": "alloc A=free_cash/free_slots posted as margin; "
                           "notional=1.0xA at entry close (qty=A/px); "
                           "leverage cap L=3. On high>=vwap*1.2: free="
                           "M-qP/L where M=margin_cash+q(P-vwap); add "
                           "0.6*free*3 notional at trigger, sweep "
                           "0.4*free to safe pool, vwap blends, stop "
                           "ratchets to max(stop,vwap), roll timer "
                           "resets. Initial stop=0.92*entry (8% disaster "
                           "stop). Intrabar stop fill: open<=stop->open, "
                           "else low<=stop->stop. Exits: data_end > stop "
                           "> btc_gate_off > 120-bar no-roll time stop "
                           "(stop checked before same-bar rolls).",
            "concurrency": "max 2 campaigns; alloc = cash/free_slots so "
                           "the book deploys evenly across slots",
            "costs": "5bps per fill (entry, roll add, exit); safe-pool "
                     "sweep is a collateral transfer (no fee); funding "
                     "carry NOT modeled",
        },
        "arms": {
            "our_state_roll": "our entry + their roll mechanics",
            "our_state_flat": "our entry, identical engine, rolls "
                              "disabled — isolates roll value",
            "sibling_e1_roll": "their E1 entry + their roll mechanics",
            "sibling_e1_flat": "their E1 entry, rolls disabled",
            "btc_hold": "BTC 4h buy-and-hold over the master calendar",
        },
        "metrics": ["n campaigns", "win rate", "geo-mean campaign mult",
                    "final equity", "maxDD (4h equity)", "per-year by "
                    "exit year", "roll-count distribution", "exit-reason "
                    "histogram", "time-in-market", "determinism "
                    "double-run"],
        "caveats": [
            "4h granularity: stops/triggers resolve at bar open/low/high "
            "— intrabar path unknown; stop-before-roll ordering is the "
            "conservative choice",
            "liquidation NOT modeled: at 3x margin cap a gap can take the "
            "margin account negative; the sibling paper warns ~8% adverse "
            "moves approach maintenance margin — hence the 0.92 initial "
            "stop; returns are accounting-basis only",
            "leverage 3x amplifies both tails; campaign return can be "
            "<-100% in principle (ruined count reported, geo-mult then "
            "null)",
            "funding cashflows skipped (positioning signal only); "
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
        "schema_version": "nanojev-financial-campaign-4h-authorization-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "protocol_sha256": sha,
        "protocol_path": "research/financial_campaign_4h_protocol_v1.json",
        "decision": "approved_for_measurement",
        "measurement_authorized": True,
        "fit_authorized": False,
        "independent_reviewer": {
            "id": "project-owner",
            "independence": "owner_self_authorization_not_independent_review",
            "note": "Owner directed T121: port the sibling rolling-"
                    "compounding engine to the 4h mega data and test our "
                    "entry selection on their roll mechanics (delegated "
                    "task).",
        },
        "scope": {
            "permitted": "PIT-safe event-driven 4h simulation on "
                         "data/rc_futures_v1 perp klines + funding.csv "
                         "for the 52 funded symbols and the BTC 4h gate: "
                         "4 sim arms + BTC-hold baseline per the pinned "
                         "protocol.",
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


def run_all(universe, calendar, btc_close, btc_sma, dfh_pct, n_ranked):
    """4 sim arms each run twice for determinism + BTC hold baseline."""
    results = {}
    for arm, rule, rolls in (
            ("our_state_roll", "our", True),
            ("our_state_flat", "our", False),
            ("sibling_e1_roll", "e1", True),
            ("sibling_e1_flat", "e1", False)):
        sim1 = simulate(universe, calendar, btc_close, btc_sma, dfh_pct,
                        n_ranked, rule, rolls)
        sim2 = simulate(universe, calendar, btc_close, btc_sma, dfh_pct,
                        n_ranked, rule, rolls)
        det = json.dumps(sim1, sort_keys=True) == json.dumps(
            sim2, sort_keys=True)
        results[arm] = {"sim": sim1,
                        "metrics": arm_metrics(sim1, calendar),
                        "deterministic_double_run": det}
    base = btc_close[0]
    bh = {ts: btc_close[i] / base for i, ts in enumerate(calendar)}
    results["btc_hold"] = {"metrics": equity_metrics(bh)}
    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=pathlib.Path, default=DATA_DIR)
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    args = parser.parse_args()

    protocol_sha = write_protocol_and_auth()
    report = {
        "schema_version": "nanojev-financial-campaign-4h-v1",
        "task": "T121 sibling rolling-compounding engine on our 4h mega "
                "data; our entry vs their E1 on identical roll mechanics",
        "protocol_path":
            "research/financial_campaign_4h_protocol_v1.json",
        "protocol_sha256": protocol_sha,
        "authorization_path":
            "results/financial_campaign_4h_authorization_v1.json",
        "parameters": {
            "max_concurrent_campaigns": MAX_CONCURRENT,
            "fee_per_fill": FEE, "leverage_cap": LEVERAGE,
            "roll_trigger": TRIGGER, "roll_alpha": ALPHA,
            "initial_stop": INIT_STOP,
            "time_stop_bars_no_roll": TIME_STOP_BARS,
            "btc_gate_sma_bars": BTC_SMA, "dfh_window_bars": DFH_WIN,
            "atr_window_bars": ATR_WIN, "atrp_max": ATRP_MAX,
            "e1_pullback_depth": [PB_LO, PB_HI],
            "our_dfh_band": [OUR_DFH_LO, OUR_DFH_HI],
            "top_quintile": TOP_Q, "min_ranked": MIN_RANKED,
            "funding_lookback_obs": FUND_LOOKBACK,
            "funding_entry_max": FUND_ENTRY_MAX,
        },
    }
    data_dir = args.data_dir
    btc_path = data_dir / "BTC/BTCUSDT_4h.csv"
    if not data_dir.exists() or not btc_path.exists():
        report["status"] = ("SKIPPED: data dir or BTC 4h series missing; "
                            "nothing was fabricated")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True)
                            + "\n")
        print(json.dumps({"status": report["status"]}))
        return 0

    universe = load_universe(data_dir)
    btc_bars = load_bars(btc_path)
    calendar = [b["ts"] for b in btc_bars]
    btc_close = [b["c"] for b in btc_bars]
    btc_sma = [None] * len(btc_bars)
    for i in range(BTC_SMA - 1, len(btc_bars)):
        btc_sma[i] = (sum(btc_close[i - BTC_SMA + 1:i + 1])
                      / BTC_SMA)
    dfh_pct, n_ranked = build_dfh_pct(universe, calendar)

    report["universe_description"] = {
        "choice": "52 funding-covered symbols (funding.csv + 4h csv "
                  "both present) — see protocol",
        "symbols": len(universe),
        "span": {"first": calendar[0], "last": calendar[-1],
                 "bars": len(calendar)},
        "symbols_listed": sorted(universe),
        "early_ended": {s: u["last_ts"] for s, u in universe.items()
                        if u["last_ts"] < calendar[-1]},
        "gate_occupancy_frac": sum(
            1 for i, ts in enumerate(calendar)
            if btc_sma[i] is not None
            and btc_close[i] > btc_sma[i]) / len(calendar),
    }
    report["status"] = "ran"

    res = run_all(universe, calendar, btc_close, btc_sma, dfh_pct,
                  n_ranked)
    for arm in ("our_state_roll", "our_state_flat", "sibling_e1_roll",
                "sibling_e1_flat"):
        res[arm]["campaign_log"] = res[arm]["sim"]["campaigns"]
        res[arm]["n_fills"] = len(res[arm]["sim"]["fills"])
        del res[arm]["sim"]

    report["arms"] = res
    our_r = res["our_state_roll"]["metrics"]
    our_f = res["our_state_flat"]["metrics"]
    e1_r = res["sibling_e1_roll"]["metrics"]
    e1_f = res["sibling_e1_flat"]["metrics"]
    bh = res["btc_hold"]["metrics"]

    def cmp(a, b):
        return {"a_final": _r(a["final_equity"]),
                "b_final": _r(b["final_equity"]),
                "a_beats_b": a["final_equity"] > b["final_equity"],
                "a_mdd": _r(a["max_drawdown_frac"]),
                "b_mdd": _r(b["max_drawdown_frac"])}

    roll_adds_value = (our_r["final_equity"] > our_f["final_equity"]
                       and e1_r["final_equity"] > e1_f["final_equity"])
    hybrid_beats_baselines = (
        our_r["final_equity"] > bh["final_equity"]
        and our_r["final_equity"] > our_f["final_equity"])
    report["verdict"] = {
        "roll_adds_value_over_flat_both_entries": roll_adds_value,
        "our_hybrid_beats_its_baselines": hybrid_beats_baselines,
        "comparisons": {
            "our_roll_vs_our_flat": cmp(our_r, our_f),
            "e1_roll_vs_e1_flat": cmp(e1_r, e1_f),
            "our_roll_vs_e1_roll": cmp(our_r, e1_r),
            "our_roll_vs_btc_hold": cmp(our_r, bh),
            "e1_roll_vs_btc_hold": cmp(e1_r, bh),
        },
        "summary": (
            f"our_state roll {_r(our_r['final_equity'])}x vs flat "
            f"{_r(our_f['final_equity'])}x | sibling_e1 roll "
            f"{_r(e1_r['final_equity'])}x vs flat "
            f"{_r(e1_f['final_equity'])}x | btc_hold "
            f"{_r(bh['final_equity'])}x -> roll mechanics "
            f"{'ADD' if roll_adds_value else 'do NOT add'} value over "
            f"flat at 4h for both entries; our hybrid "
            f"{'BEATS' if hybrid_beats_baselines else 'does NOT beat'} "
            f"its baselines"),
    }
    report["honesty"] = {
        "granularity": ("4h bars: stops and roll triggers resolve on "
                        "open/low/high of a completed bar — the intrabar "
                        "path is unknown; stop-before-roll ordering and "
                        "gap-through-open fills are the conservative "
                        "choices"),
        "liquidation": ("NOT modeled — at the 3x margin cap a gap can "
                        "take margin negative; the sibling paper itself "
                        "warns a ~8% adverse move approaches maintenance "
                        "margin (hence the 0.92 initial stop); ruined "
                        "campaigns are counted, not hidden"),
        "leverage": "3x cap amplifies both tails of every campaign",
        "funding_carry": "skipped — funding_pct is a positioning signal "
                         "only, no funding cashflows modeled",
        "universe": ("52 funded symbols — not the top-50-by-volume "
                     "alternative, which would embed full-sample volume "
                     "lookahead; funded set also includes BTC itself"),
        "not_an_asof_vintage": "second-hand archive copy (rc_futures_v1)",
        "not_live": "no orders, no account, no broker, no trading API",
        "no_profitability_claim": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True)
                        + "\n", encoding="utf-8")
    brief = {"status": "ran", "out": str(args.out),
             "verdict": report["verdict"]["summary"],
             "deterministic": {a: res[a]["deterministic_double_run"]
                               for a in ("our_state_roll",
                                         "our_state_flat",
                                         "sibling_e1_roll",
                                         "sibling_e1_flat")}}
    print(json.dumps(brief, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
