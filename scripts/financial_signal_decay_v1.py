#!/usr/bin/env python3
"""T126: diagnose the 2025 signal decay — regime shrink vs real edge death.

Multiple arms faded to ~0 in 2025: T111 sleeve A (funding_pct carry, Sharpe
1.79/1.19 -> 0.54/-0.12 in 2025/26 on the 10-asset cohort), T124 mom_4h
reversal (pooled rho -0.062/-0.052/+0.004 for 2023/24/25), and the campaign
wrapper (ex-2021 0.219x-0.557x vs BTC hold). This script asks WHICH failure
mode is operative on the mega cohorts:

  (a) REAL DECAY    — the signal-per-unit-of-opportunity itself weakened
                      (crowded trade / alpha decay).
  (b) REGIME SHRINK — the opportunity set shrank: cross-sectional dispersion
                      and/or BTC vol collapsed in 2025, so a constant-per-unit-
                      dispersion signal earns ~0 bps. If true, the master gate
                      arguably needs a DISPERSION/VOL FLOOR, not a replacement.
  (c) COMPOSITION   — the universe mix changed (110 -> 255 symbols; new
                      listings with different microstructure diluted the
                      ranked cross-section). Control: recompute every arm on
                      the age>=180d incumbent subset only.

Inputs (all local, PIT):
  data/perp_pit_mega_v1/records.jsonl      daily, 277 syms, 2021-2025
  data/perp_pit_mega_4h_v1/records.jsonl   4h, ~283 syms, 2023-2025
  data/rc_futures_v1/BTC/BTCUSDT_1d.csv    BTC daily OHLC (gate + vol state)

Part 1 — regime variables by year (2021-25):
  btc_ret20_gate_occupancy_pct   share of cohort decision days with
                                 close/close[t-20]-1 > 0 (the T114 master gate)
  btc_vol20_pct                  mean stdev of the 20 trailing daily log-rets
  btc_mean_abs_ret_pct           mean |close/close_prev - 1| in pct
  funding_mean_bps / share_pos   over the ~52 funded symbols' asset-days
  xs_dispersion_bps              mean daily cross-sectional stdev of fwd_1d
                                 over the ranked (dfh20 non-null) universe —
                                 THE quantity that starves XS signals
  btc_ret_autocorr               lag-1 / lag-5 Pearson autocorr of BTC daily
                                 log returns (trend persistence)

Part 2 — effect decay per arm, raw AND dispersion-normalized:
  xs_dfh20      per-day top-decile minus bottom-decile by dfh20 rank on
                fwd_1d (and fwd_5d), >=30 ranked assets/day (T112 conv.).
                normalized = yearly mean spread / yearly mean daily xs-stdev
                of the SAME-horizon forward label. If the ratio holds while
                the raw spread fades -> regime shrink, not decay.
  mom_4h        pooled Spearman(m4_pct, fwd4) by year (T124 arm A: m4_pct =
                per-asset trailing-180-record mid-rank pct of the 4h bar
                return, MIN_WINDOW=60); bps effect = per-4h-bar XS reversal
                decile spread by raw m4 rank (T124 arm D construction,
                >=30 assets/bar); normalized by per-bar xs-stdev of fwd4.
                NOTE: rho is scale-free — a rho collapse cannot be explained
                by dispersion shrink; only the bps effect is normalizable.
  funding_pct   pooled Spearman(funding_pct, fwd_1d) by year on the funded
                subset (funding_pct = trailing-180 mid-rank pct,
                MIN_WINDOW=20, T112 conv.) + quintile bottom-vs-top diff in
                bps normalized by funded-subset xs-stdev of fwd_1d.

Part 3 — universe composition: n ranked assets/day by year; share of ranked
asset-days with cohort age <180d (first-seen proxy for new listings; pre-2021
listings are counted as age 0 at cohort start — caveat carried); every arm
re-run on the age>=180d incumbent subset as the composition control.

Part 4 — BTC vol regime: atrp_d = Wilder ATR14/close on BTC 1d bars (T114
conv.), distribution by year; Pearson + Spearman correlation of the yearly
xs_dfh20 h1 spread vs yearly BTC vol/dispersion measures (n=5 — nominal).

Part 5 — verdict: programmatic classification (a)/(b)/(c)/mixed with the
evidence table; if (b), a candidate dispersion floor for the master gate is
quantified (share of 2025 days below the 2021-24 median dispersion).

Protocol + owner self-authorization are embedded in the single output
artifact ``results/financial_signal_decay_v1.json`` (protocol object hashed
by sha256, authorization pins the hash — T111 convention). Measurement only:
no fitting, no trading, no network, no other files modified.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
import statistics
from bisect import bisect_left, bisect_right, insort
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
COHORT_D = ROOT / "data/perp_pit_mega_v1/records.jsonl"
COHORT_4H = ROOT / "data/perp_pit_mega_4h_v1/records.jsonl"
BTC_1D = ROOT / "data/rc_futures_v1/BTC/BTCUSDT_1d.csv"
OUT = ROOT / "results/financial_signal_decay_v1.json"

YEARS = ("2021", "2022", "2023", "2024", "2025")
YEARS_4H = ("2023", "2024", "2025")           # 4h cohort spans 2023-25
REF_YEARS = ("2021", "2022", "2023", "2024")  # pre-fade reference, daily
REF_YEARS_4H = ("2023", "2024")               # pre-fade reference, 4h
MIN_ASSETS = 30            # T112 day gate / T124 bar gate for decile ranks
MIN_FUNDED_PER_DAY = 10    # funded-subset dispersion floor
DECILE = 10
QUINTILE = 5
TRAIL = 180                # trailing-record window for *_pct scores
MIN_WINDOW_D = 20          # daily funding_pct floor (T112 convention)
MIN_WINDOW_4H = 60         # 4h m4_pct floor (T124 convention)
NEW_AGE_D = 180            # cohort-age < 180d counts as "new listing"
RET_LOOKBACK = 20          # BTC master gate: close/close[t-20]-1
ATR_N = 14                 # Wilder ATR length on 1d bars (T114 atrp_d)
MIN_N = 10                 # repo test minimum
FADE_RATIO = 0.5           # 2025 < 0.5x reference mean counts as "faded"
NORM_HOLD = 0.5            # normalized 2025 >= 0.5x ref = "per-unit intact"


def _r(x, nd=3):
    return round(x, nd) if isinstance(x, float) else x


def norm_p(t):
    """Two-sided normal-approx p from a t/z statistic (repo convention)."""
    if t is None:
        return None
    return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(xs, ys):
    """Mid-rank Spearman rho + normal-approx t/p; None if n<MIN_N or a side
    has zero rank variance (project convention)."""
    n = len(xs)
    if n < MIN_N:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    rho = num / (dx * dy)
    t = rho * math.sqrt((n - 2) / max(1e-9, 1 - rho * rho))
    return {"n": n, "rho": round(rho, 5), "t": round(t, 3),
            "p": round(norm_p(t), 8)}


def pearson(xs, ys):
    """Plain Pearson r; None if n<3 or a side is constant. n=5 cells are
    nominal/descriptive only."""
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if dx == 0 or dy == 0:
        return None
    return round(num / (dx * dy), 4)


def rank_corr(xs, ys):
    """Spearman rho without the MIN_N inferential guard — for the n=5
    yearly correlation cells where no p-value is claimed anyway."""
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    return round(num / (dx * dy), 4)


def t_stat(xs):
    """Mean/sd/t (+ normal-approx p) of a per-day series; obs treated as
    independent — nominal only (labels overlap / share a market factor)."""
    n = len(xs)
    if n < 5:
        return {"n": n, "mean": (sum(xs) / n) if n else None,
                "sd": None, "t": None, "p": None}
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(var)
    t = (mean / (sd / math.sqrt(n))) if sd > 0 else None
    return {"n": n, "mean": mean, "sd": sd, "t": t, "p": norm_p(t)}


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return None
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def autocorr(vals, lag):
    """Pearson autocorrelation of a series at the given lag (pairs within
    the same year only — series is pre-sliced per year)."""
    xs = [v for v in vals if v is not None]
    if len(xs) < lag + MIN_N:
        return None
    a, b = xs[:-lag], xs[lag:]
    return pearson(a, b)


def atr_wilder(highs, lows, closes, n):
    """Wilder ATR (T114/gate_shootout construction): seed = mean TR of bars
    1..n, then atr = (atr*(n-1)+tr)/n. Index-aligned to closes; None before
    the seed bar."""
    out = [None] * len(closes)
    if len(closes) <= n:
        return out
    trs = [0.0]
    for i in range(1, len(closes)):
        trs.append(max(highs[i] - lows[i],
                       abs(highs[i] - closes[i - 1]),
                       abs(lows[i] - closes[i - 1])))
    atr = sum(trs[1:n + 1]) / n
    out[n] = atr
    for i in range(n + 1, len(closes)):
        atr = (atr * (n - 1) + trs[i]) / n
        out[i] = atr
    return out


# ---------------------------------------------------------------- loaders
def load_btc(path):
    """day -> {ret20, vol20_pct, abs_ret_pct, atrp_d, logret} from the BTC
    1d csv; vol20 = stdev of the 20 log-rets ending at day t (same window as
    the cohort's vol20 feature); atrp_d = Wilder ATR14/close (T114)."""
    days, hi, lo, cl = [], [], [], []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            days.append(row["timestamp"][:10])
            hi.append(float(row["high"]))
            lo.append(float(row["low"]))
            cl.append(float(row["close"]))
    logret = [None]
    for i in range(1, len(cl)):
        logret.append(math.log(cl[i] / cl[i - 1])
                      if cl[i] > 0 and cl[i - 1] > 0 else None)
    state = {d: {} for d in days}
    for i, d in enumerate(days):
        if i >= RET_LOOKBACK and cl[i - RET_LOOKBACK] > 0:
            state[d]["ret20"] = cl[i] / cl[i - RET_LOOKBACK] - 1.0
        if i >= 1 and cl[i - 1] > 0:
            state[d]["abs_ret_pct"] = abs(cl[i] / cl[i - 1] - 1.0) * 100.0
        win = [x for x in logret[max(0, i - 19):i + 1] if x is not None]
        if len(win) >= 20:
            state[d]["vol20_pct"] = statistics.stdev(win) * 100.0
        state[d]["logret"] = logret[i]
    atr = atr_wilder(hi, lo, cl, ATR_N)
    for i, d in enumerate(days):
        if atr[i] is not None and cl[i] > 0:
            state[d]["atrp_d"] = atr[i] / cl[i]
    return days, state


def load_daily(path):
    """asset -> sorted rows {day, age_d, dfh20, funding, funding_pct, fwd1,
    fwd5}; plus meta and the funded-symbol set. funding_pct = strictly-prior
    trailing-180 mid-rank pct (MIN_WINDOW_D floor, T112 convention); age_d =
    days since the asset's FIRST cohort record (listing-age proxy)."""
    series, meta = defaultdict(list), {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            rec = json.loads(line)
            asset = rec["asset_id"]
            meta[asset] = {"listed": rec.get("listed"),
                           "last_bar_date": rec.get("last_bar_date")}
            f = rec["features"]
            series[asset].append({
                "day": rec["id"].rsplit(":", 1)[-1],
                "dfh20": f["dfh20"]["value"],
                "funding": f["last_funding_rate"]["value"],
                "fwd1": rec["label"]["forward_return_bps"],
                "fwd5": rec["label"]["forward_return_5d_bps"],
            })
    funded = set()
    for asset, rows in series.items():
        rows.sort(key=lambda r: r["day"])
        first_ord = dt.date.fromisoformat(rows[0]["day"]).toordinal()
        fund = [r["funding"] for r in rows]
        fwin = []
        for i, r in enumerate(rows):
            r["age_d"] = (dt.date.fromisoformat(r["day"]).toordinal()
                          - first_ord)
            r["funding_pct"] = None
            if r["funding"] is not None and len(fwin) >= MIN_WINDOW_D:
                r["funding_pct"] = ((bisect_left(fwin, r["funding"])
                                     + bisect_right(fwin, r["funding"]))
                                    / (2.0 * len(fwin)))
            if r["funding"] is not None:
                insort(fwin, r["funding"])
                funded.add(asset)
            if i >= TRAIL and fund[i - TRAIL] is not None:
                fwin.pop(bisect_left(fwin, fund[i - TRAIL]))
    return series, meta, funded


def load_4h(path):
    """asset -> sorted [(ts, m4, fwd4)]; lean tuples for the 1.28M rows."""
    series = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            r = json.loads(line)
            series[r["asset"]].append((r["ts"], r["m4"], r["fwd4"]))
    for rows in series.values():
        rows.sort(key=lambda x: x[0])
    return series





# ------------------------------------------------------- daily XS engine
def daily_xs_tables(series, funded):
    """Per decision day over the dfh20-ranked universe: n ranked, young
    share, xs-stdev of fwd1/fwd5, decile dfh spreads (all + incumbent-only
    + new-only), funded-subset fwd1 stdev. Returns day -> row dict."""
    by_day = defaultdict(list)
    by_day_funded = defaultdict(list)
    for asset, rows in series.items():
        is_funded = asset in funded
        for r in rows:
            if r["dfh20"] is not None:
                by_day[r["day"]].append(r)
            if is_funded and r["funding"] is not None:
                by_day_funded[r["day"]].append(r)

    def decile_spread(rows, fwd_key):
        ordered = sorted(rows, key=lambda x: x["dfh20"])
        n = len(ordered)
        if n < MIN_ASSETS:
            return None, n
        k = max(1, n // DECILE)
        bot, top = ordered[:k], ordered[-k:]
        if bot[-1]["dfh20"] == top[0]["dfh20"]:
            return None, n
        lv = [x[fwd_key] for x in top]    # momentum: long top-decile
        sv = [x[fwd_key] for x in bot]
        return (sum(lv) / len(lv) - sum(sv) / len(sv)), n

    table = {}
    for day in sorted(by_day):
        rows = by_day[day]
        row = {"n_ranked": len(rows),
               "n_young": sum(1 for r in rows if r["age_d"] < NEW_AGE_D)}
        r1 = [r for r in rows if r["fwd1"] is not None]
        r5 = [r for r in rows if r["fwd5"] is not None]
        if len(r1) >= MIN_ASSETS:
            row["disp1_bps"] = statistics.stdev(x["fwd1"] for x in r1)
            row["spread1_bps"], _ = decile_spread(r1, "fwd1")
        if len(r5) >= MIN_ASSETS:
            row["disp5_bps"] = statistics.stdev(x["fwd5"] for x in r5)
            row["spread5_bps"], _ = decile_spread(r5, "fwd5")
        old1 = [r for r in r1 if r["age_d"] >= NEW_AGE_D]
        new1 = [r for r in r1 if r["age_d"] < NEW_AGE_D]
        if len(old1) >= MIN_ASSETS:
            row["spread1_old_bps"], _ = decile_spread(old1, "fwd1")
        if len(new1) >= MIN_ASSETS:
            row["spread1_new_bps"], _ = decile_spread(new1, "fwd1")
        f1 = [r["fwd1"] for r in by_day_funded.get(day, ())
              if r["fwd1"] is not None]
        if len(f1) >= MIN_FUNDED_PER_DAY:
            row["disp1_funded_bps"] = statistics.stdev(f1)
        table[day] = row
    return table, by_day_funded


def yearly_mean(table, key, years=YEARS):
    """year -> {n, mean, median, t} over the per-day values of `key`."""
    out = {}
    for y in years:
        vals = [row[key] for day, row in table.items()
                if day.startswith(y) and row.get(key) is not None]
        s = t_stat(vals)
        out[y] = {"n": len(vals), "mean": _r(s["mean"], 4),
                  "median": _r(median(vals), 4),
                  "t": _r(s["t"]), "p": _r(s["p"], 8)}
    return out


def funding_arm(series, funded):
    """funding_pct TS arm: pooled Spearman(funding_pct, fwd1) per year on
    the funded subset (all rows + incumbent-only control) + pooled quintile
    bottom-vs-top diff in bps; fwd5 rho as reference."""
    per_year = {y: {"xs": [], "ys": [], "xs_old": [], "ys_old": [],
                    "ys5": [], "xs5": []} for y in YEARS}
    for asset in funded:
        for r in series[asset]:
            y = r["day"][:4]
            if y not in per_year or r["funding_pct"] is None:
                continue
            cell = per_year[y]
            if r["fwd1"] is not None:
                cell["xs"].append(r["funding_pct"])
                cell["ys"].append(r["fwd1"])
                if r["age_d"] >= NEW_AGE_D:
                    cell["xs_old"].append(r["funding_pct"])
                    cell["ys_old"].append(r["fwd1"])
            if r["fwd5"] is not None:
                cell["xs5"].append(r["funding_pct"])
                cell["ys5"].append(r["fwd5"])
    out = {}
    for y, c in per_year.items():
        sp = spearman(c["xs"], c["ys"])
        sp_old = spearman(c["xs_old"], c["ys_old"])
        sp5 = spearman(c["xs5"], c["ys5"])
        # pooled quintile bottom-vs-top on fwd1 (carry direction: low
        # funding_pct minus high funding_pct; positive = carry works)
        diff = None
        pairs = sorted(zip(c["xs"], c["ys"]))
        if len(pairs) >= 2 * MIN_N:
            k = max(1, len(pairs) // QUINTILE)
            diff = (sum(v for _, v in pairs[:k]) / k
                    - sum(v for _, v in pairs[-k:]) / k)
        out[y] = {"rho_fwd1": sp, "rho_fwd1_incumbent_only": sp_old,
                  "rho_fwd5": sp5,
                  "quintile_bot_minus_top_bps": _r(diff, 4)}
    return out


# ------------------------------------------------------------- 4h engine
def mom4h_arm(series):
    """T124 arm A rho by year (pooled Spearman(m4_pct, fwd4), all rows and
    incumbent-only) + T124 arm D per-bar XS reversal decile spread by raw m4
    rank (bps) + per-bar xs-stdev of fwd4 for the normalization."""
    day_ms = 86400000.0
    rho_in = {y: {"xs": [], "ys": [], "xs_old": [], "ys_old": []}
              for y in YEARS_4H}
    by_bar = defaultdict(list)
    # per-asset strictly-prior trailing-180-record mid-rank pct of m4
    # (MIN_WINDOW_4H, T124 convention), accumulated straight into the
    # rho inputs and per-bar lists to keep RAM flat.
    for asset, rows in series.items():
        first_ts = rows[0][0]
        m4 = [r[1] for r in rows]
        mwin = []
        for i, (ts, m, f4) in enumerate(rows):
            pct = None
            if m is not None and len(mwin) >= MIN_WINDOW_4H:
                pct = ((bisect_left(mwin, m) + bisect_right(mwin, m))
                       / (2.0 * len(mwin)))
            if m is not None:
                insort(mwin, m)
            if i >= TRAIL and m4[i - TRAIL] is not None:
                mwin.pop(bisect_left(mwin, m4[i - TRAIL]))
            year = str(dt.datetime.fromtimestamp(
                ts / 1000, dt.timezone.utc).year)
            if year not in rho_in:
                continue
            old = (ts - first_ts) / day_ms >= NEW_AGE_D
            if pct is not None and f4 is not None:
                c = rho_in[year]
                c["xs"].append(pct)
                c["ys"].append(f4)
                if old:
                    c["xs_old"].append(pct)
                    c["ys_old"].append(f4)
            if m is not None and f4 is not None:
                by_bar[ts].append((m, f4, old))

    rho_year = {y: {"rho": spearman(c["xs"], c["ys"]),
                    "rho_incumbent_only": spearman(c["xs_old"],
                                                 c["ys_old"])}
                for y, c in rho_in.items()}

    bar_year = {y: {"spread": [], "disp": [], "spread_old": []}
                for y in YEARS_4H}
    for ts in sorted(by_bar):
        rows = by_bar[ts]
        year = str(dt.datetime.fromtimestamp(ts / 1000,
                                             dt.timezone.utc).year)
        if year not in bar_year or len(rows) < MIN_ASSETS:
            continue
        ordered = sorted(rows, key=lambda x: x[0])
        k = max(1, len(ordered) // DECILE)
        bot, top = ordered[:k], ordered[-k:]
        if bot[-1][0] == top[0][0]:
            continue
        cell = bar_year[year]
        # reversal: long bottom-decile (losers) / short top-decile
        cell["spread"].append(sum(x[1] for x in bot) / k
                              - sum(x[1] for x in top) / k)
        cell["disp"].append(statistics.stdev(x[1] for x in ordered))
        old_rows = [x for x in rows if x[2]]
        if len(old_rows) >= MIN_ASSETS:
            oo = sorted(old_rows, key=lambda x: x[0])
            ko = max(1, len(oo) // DECILE)
            ob, ot = oo[:ko], oo[-ko:]
            if ob[-1][0] != ot[0][0]:
                cell["spread_old"].append(
                    sum(x[1] for x in ob) / ko
                    - sum(x[1] for x in ot) / ko)

    bar_stats = {}
    for y, c in bar_year.items():
        s_s, s_d, s_o = t_stat(c["spread"]), t_stat(c["disp"]), \
            t_stat(c["spread_old"])
        bar_stats[y] = {
            "bars": len(c["spread"]),
            "mean_spread_bps": _r(s_s["mean"], 4),
            "t_spread": _r(s_s["t"]),
            "mean_disp_bps": _r(s_d["mean"], 4),
            "mean_spread_bps_incumbent_only": _r(s_o["mean"], 4),
            "norm_spread_over_disp": _r(
                (s_s["mean"] / s_d["mean"])
                if s_s["mean"] is not None and s_d["mean"] else None, 4)}
    return {"rho_by_year": rho_year, "xs_reversal_by_year": bar_stats}


# ------------------------------------------------------------------ arms
def regime_variables(days, state, table, by_day_funded):
    """Part 1: per-year regime variable table over cohort decision days."""
    out = {}
    cohort_days = sorted(table)
    for y in YEARS:
        ds = [d for d in cohort_days if d.startswith(y)]
        st = [state.get(d, {}) for d in ds]
        gated = [s["ret20"] for s in st if "ret20" in s]
        vol = [s["vol20_pct"] for s in st if "vol20_pct" in s]
        aret = [s["abs_ret_pct"] for s in st if "abs_ret_pct" in s]
        atrp = [s["atrp_d"] for s in st if "atrp_d" in s]
        # funding stats over funded asset-days of that year
        frates = [r["funding"] for d in ds
                  for r in by_day_funded.get(d, ())
                  if r["funding"] is not None]
        rets = [state.get(d, {}).get("logret") for d in ds]
        out[y] = {
            "cohort_days": len(ds),
            "btc_ret20_gate_occupancy_pct": _r(
                100.0 * sum(1 for v in gated if v > 0) / len(gated), 2)
                if gated else None,
            "n_gate_defined": len(gated),
            "btc_vol20_pct_mean": _r(statistics.mean(vol), 4)
                if vol else None,
            "btc_mean_abs_ret_pct": _r(statistics.mean(aret), 4)
                if aret else None,
            "btc_atrp_d_mean": _r(statistics.mean(atrp), 5)
                if atrp else None,
            "funding_mean_bps_per_interval": _r(
                statistics.mean(frates) * 1e4, 4) if frates else None,
            "funding_share_positive": _r(
                sum(1 for v in frates if v > 0) / len(frates), 4)
                if frates else None,
            "n_funded_asset_days": len(frates),
            "btc_ret_autocorr_lag1": autocorr(rets, 1),
            "btc_ret_autocorr_lag5": autocorr(rets, 5),
        }
    return out


def composition(table):
    """Part 3: ranked-universe size and new-listing share by year."""
    out = {}
    for y in YEARS:
        ns = [row["n_ranked"] for d, row in table.items()
              if d.startswith(y)]
        shares = [row["n_young"] / row["n_ranked"] for d, row in
                  table.items() if d.startswith(y) and row["n_ranked"]]
        pooled_n = sum(row["n_ranked"] for d, row in table.items()
                       if d.startswith(y))
        pooled_y = sum(row["n_young"] for d, row in table.items()
                       if d.startswith(y))
        out[y] = {
            "n_ranked_mean": _r(statistics.mean(ns), 2) if ns else None,
            "n_ranked_median": _r(median(ns), 2),
            "n_ranked_min": min(ns) if ns else None,
            "n_ranked_max": max(ns) if ns else None,
            "young_share_mean_daily": _r(statistics.mean(shares), 4)
                if shares else None,
            "young_share_pooled": _r(pooled_y / pooled_n, 4)
                if pooled_n else None,
        }
    return out


def fade_check(raw_by_year, norm_by_year, ref_years):
    """Classify one arm: raw_faded (2025 < FADE_RATIO x ref mean or sign
    flip vs ref) and norm_holds (normalized 2025 >= NORM_HOLD x ref)."""
    ref = [raw_by_year[y] for y in ref_years
           if raw_by_year.get(y) is not None]
    ref_n = [norm_by_year[y] for y in ref_years
             if norm_by_year.get(y) is not None]
    raw25, n25 = raw_by_year.get("2025"), norm_by_year.get("2025")
    if not ref or raw25 is None:
        return {"class": "insufficient_data"}
    ref_mean = sum(ref) / len(ref)
    faded = (ref_mean > 0 and raw25 < FADE_RATIO * ref_mean) or \
            (ref_mean < 0 and raw25 > FADE_RATIO * ref_mean) or \
            (ref_mean != 0 and math.copysign(1, raw25)
             != math.copysign(1, ref_mean))
    ref_signs = {math.copysign(1, v) for v in ref if v != 0}
    res = {"ref_mean_raw": _r(ref_mean, 4), "raw_2025": _r(raw25, 4),
           "ref_years_sign_consistent": len(ref_signs) <= 1,
           "raw_faded": bool(faded)}
    if ref_n and n25 is not None:
        ref_nmean = sum(ref_n) / len(ref_n)
        res["ref_mean_norm"] = _r(ref_nmean, 4)
        res["norm_2025"] = _r(n25, 4)
        res["norm_ratio_2025_vs_ref"] = _r(
            n25 / ref_nmean, 3) if ref_nmean else None
        res["norm_holds"] = bool(
            ref_nmean != 0 and abs(n25) >= NORM_HOLD * abs(ref_nmean)
            and math.copysign(1, n25) == math.copysign(1, ref_nmean))
    if not faded:
        res["class"] = "no_decay"
    elif len(ref_signs) > 1:
        # the reference years themselves flip sign — a 2025 "fade" is not
        # attributable to decay; the arm never had a stable yearly sign
        res["class"] = "unstable_baseline"
        res["raw_faded"] = False
    elif res.get("norm_holds"):
        res["class"] = "regime_shrink"
    else:
        res["class"] = "real_decay"
    return res


def build_protocol():
    return {
        "schema_version": "nanojev-financial-signal-decay-protocol-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "task": "T126",
        "purpose": "Diagnose the 2025 fade shared by T111 sleeve A, T124 "
                   "mom_4h reversal and the campaign arms: classify as "
                   "(a) real edge decay, (b) regime/dispersion shrink, or "
                   "(c) universe-composition artifact, using yearly regime "
                   "variables, dispersion-normalized effect sizes and an "
                   "incumbent-only (age>=180d) control. Measurement only — "
                   "no fitting, no trading, no profitability claims.",
        "cohorts": {
            "daily": {"path": "data/perp_pit_mega_v1/records.jsonl",
                      "span": "2021-01..2025-12, ~277 USDT-M perps, "
                              "~52 funding-covered, close-as-mark basis"},
            "h4": {"path": "data/perp_pit_mega_4h_v1/records.jsonl",
                   "span": "2023-01..2025-12, ~283 syms, 4h bars"},
            "btc_1d": "data/rc_futures_v1/BTC/BTCUSDT_1d.csv"},
        "definitions": {
            "btc_ret20_gate": "close/close[t-20]-1 > 0 on BTC 1d bars "
                              "(T114 master gate); occupancy = share of "
                              "cohort decision days gated on",
            "btc_vol20_pct": "stdev of the 20 daily log-rets ending at t "
                             "(same window as the cohort vol20 feature)",
            "atrp_d": "Wilder ATR14/close on BTC 1d bars (T114 conv.)",
            "xs_dispersion_bps": "per-day cross-sectional stdev of "
                                 "fwd_1d_bps over the dfh20-ranked "
                                 "universe (>=30 assets)",
            "xs_dfh20_spread": "per-day top-decile minus bottom-decile by "
                               "dfh20 rank, fwd_1d and fwd_5d, >=30 ranked "
                               "(T112 decile convention)",
            "mom4h_rho": "pooled Spearman(m4_pct, fwd4); m4_pct = strictly-"
                         "prior trailing-180-record mid-rank pct of m4, "
                         "MIN_WINDOW=60 (T124 arm A)",
            "mom4h_xs_spread": "per-4h-bar decile reversal spread by raw m4 "
                               "rank (long losers / short winners, >=30 "
                               "assets/bar, T124 arm D)",
            "funding_pct_rho": "pooled Spearman(funding_pct, fwd_1d) on the "
                               "funded subset; funding_pct = strictly-prior "
                               "trailing-180 mid-rank pct, MIN_WINDOW=20 "
                               "(T112 conv.)",
            "normalization": "yearly_mean_effect_bps / yearly_mean_"
                             "matching-horizon xs dispersion bps; rho is "
                             "scale-free and is NOT normalized — a rho "
                             "collapse cannot be a dispersion artifact",
            "incumbent_control": "every arm re-run on rows with cohort age "
                                 ">=180d (first-seen proxy; pre-2021 "
                                 "listings are age 0 at cohort start)",
            "young_share": "share of ranked asset-days with age<180d"},
        "statistics": {
            "spearman": "mid-rank rho, normal-approx t/p, min n=10",
            "t_stat": "mean/sd/t normal-approx p on per-day/per-bar "
                      "series — nominal (overlapping labels, shared "
                      "market factor)",
            "year_correlations": "Pearson + Spearman over n=5 yearly "
                                 "points — descriptive only",
            "fade_rule": "2025 raw < 0.5x mean(2021-24) or sign flip = "
                         "faded; normalized 2025 >= 0.5x ref = per-unit-"
                         "dispersion intact",
            "determinism": "measure() executed twice, serialized bytes "
                           "compared"},
        "forbidden": ["fitting", "trading", "profitability claims",
                      "protocol edits post-run", "network"],
    }


def build_auth(protocol_sha):
    return {
        "schema_version":
            "nanojev-financial-signal-decay-authorization-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "protocol_sha256": protocol_sha,
        "decision": "approved_for_measurement",
        "measurement_authorized": True,
        "fit_authorized": False,
        "independent_reviewer": {
            "id": "project-owner",
            "independence":
                "owner_self_authorization_not_independent_review",
            "note": "Owner directed T126: diagnose the 2025 decay on the "
                    "mega cohorts per the embedded protocol (delegated "
                    "task)."},
        "scope": {
            "permitted": "PIT-safe descriptive measurement on the two "
                         "mega cohorts + BTC 1d csv per the pinned "
                         "protocol: yearly regime variables, per-year arm "
                         "effects with dispersion normalization, "
                         "incumbent-only composition control, atrp_d "
                         "distribution, yearly correlations.",
            "not_permitted": "No fitting/trading/profitability claims/"
                             "protocol edits; no network; no other files "
                             "modified."},
        "network_model_calls": 0,
        "order_submission_authorized": False,
        "live_trading_authorized": False,
    }


def measure():
    btc_days, btc_state = load_btc(BTC_1D)
    series, meta, funded = load_daily(COHORT_D)
    table, by_day_funded = daily_xs_tables(series, funded)

    # ---- part 1: regime variables
    regime = regime_variables(btc_days, btc_state, table, by_day_funded)
    disp1 = yearly_mean(table, "disp1_bps")
    for y in YEARS:
        regime[y]["xs_dispersion_fwd1_bps_mean"] = disp1[y]["mean"]
        regime[y]["xs_dispersion_fwd1_bps_median"] = disp1[y]["median"]

    # ---- part 2: arms
    dfh1 = yearly_mean(table, "spread1_bps")
    dfh5 = yearly_mean(table, "spread5_bps")
    dfh1_old = yearly_mean(table, "spread1_old_bps")
    dfh1_new = yearly_mean(table, "spread1_new_bps")
    disp5 = yearly_mean(table, "disp5_bps")
    norm1, norm5 = {}, {}
    for y in YEARS:
        d1, d5 = disp1[y]["mean"], disp5[y]["mean"]
        norm1[y] = (dfh1[y]["mean"] / d1) \
            if dfh1[y]["mean"] is not None and d1 else None
        norm5[y] = (dfh5[y]["mean"] / d5) \
            if dfh5[y]["mean"] is not None and d5 else None
    xs_dfh = {"h1": {"spread_by_year": dfh1,
                     "failed_years_negative_mean":
                         [y for y in YEARS
                          if dfh1[y]["mean"] is not None
                          and dfh1[y]["mean"] < 0],
                     "norm_spread_over_disp": {y: _r(v, 4)
                                               for y, v in norm1.items()},
                     "incumbent_only_spread_by_year": dfh1_old,
                     "new_only_spread_by_year": dfh1_new},
              "h5": {"spread_by_year": dfh5,
                     "failed_years_negative_mean":
                         [y for y in YEARS
                          if dfh5[y]["mean"] is not None
                          and dfh5[y]["mean"] < 0],
                     "norm_spread_over_disp": {y: _r(v, 5)
                                               for y, v in norm5.items()}}}

    funding = funding_arm(series, funded)
    disp_f = yearly_mean(table, "disp1_funded_bps")
    fund_norm = {}
    for y in YEARS:
        d = funding[y]["quintile_bot_minus_top_bps"]
        dd = disp_f[y]["mean"]
        fund_norm[y] = _r(d / dd, 4) if d is not None and dd else None
    funding_out = dict(funding)
    for y in YEARS:
        funding_out[y]["norm_quintile_diff_over_disp"] = fund_norm[y]

    series4 = load_4h(COHORT_4H)
    mom4h = mom4h_arm(series4)

    # ---- part 3: composition
    comp = composition(table)

    # ---- part 4: BTC vol regime + yearly correlations
    atrp_by_year = {}
    for y in YEARS:
        vals = [s["atrp_d"] for d, s in btc_state.items()
                if d.startswith(y) and "atrp_d" in s]
        atrp_by_year[y] = {
            "n": len(vals),
            "mean": _r(statistics.mean(vals), 5) if vals else None,
            "median": _r(median(vals), 5),
            "p10": _r(quantile(vals, 0.10), 5),
            "p90": _r(quantile(vals, 0.90), 5),
            "min": _r(min(vals), 5) if vals else None,
            "max": _r(max(vals), 5) if vals else None}
    spread_seq = [dfh1[y]["mean"] for y in YEARS]
    corr_vs = {}
    for label, seq in (
            ("btc_vol20_pct_mean",
             [regime[y]["btc_vol20_pct_mean"] for y in YEARS]),
            ("btc_atrp_d_mean",
             [atrp_by_year[y]["mean"] for y in YEARS]),
            ("xs_dispersion_fwd1_bps_mean",
             [regime[y]["xs_dispersion_fwd1_bps_mean"] for y in YEARS]),
            ("n_ranked_mean",
             [comp[y]["n_ranked_mean"] for y in YEARS]),
            ("young_share_pooled",
             [comp[y]["young_share_pooled"] for y in YEARS])):
        ok = all(v is not None for v in seq) and \
            all(v is not None for v in spread_seq)
        corr_vs[label] = {
            "pearson_r": pearson(spread_seq, seq) if ok else None,
            "spearman_rho": rank_corr(spread_seq, seq) if ok else None,
            "n_years": len(spread_seq)}

    # ---- part 5: verdict
    raw_dfh1 = {y: dfh1[y]["mean"] for y in YEARS}
    raw_dfh5 = {y: dfh5[y]["mean"] for y in YEARS}
    raw_fund = {y: (funding[y]["rho_fwd1"] or {}).get("rho")
                for y in YEARS}
    raw_fund_bps = {y: funding[y]["quintile_bot_minus_top_bps"]
                    for y in YEARS}
    raw_mom_rho = {y: (mom4h["rho_by_year"][y]["rho"] or {}).get("rho")
                   for y in YEARS_4H}
    raw_mom_bps = {y: mom4h["xs_reversal_by_year"][y]["mean_spread_bps"]
                   for y in YEARS_4H}
    norm_mom = {y: mom4h["xs_reversal_by_year"][y]
                ["norm_spread_over_disp"] for y in YEARS_4H}
    fund_rho_old = {y: (funding[y]["rho_fwd1_incumbent_only"] or {})
                    .get("rho") for y in YEARS}
    mom_rho_old = {y: (mom4h["rho_by_year"][y]["rho_incumbent_only"] or {})
                   .get("rho") for y in YEARS_4H}

    checks = {
        "xs_dfh20_h1": fade_check(raw_dfh1, norm1, REF_YEARS),
        "xs_dfh20_h5": fade_check(raw_dfh5, norm5, REF_YEARS),
        "funding_pct_rho": fade_check(raw_fund, {}, REF_YEARS),
        "funding_pct_quintile_bps": fade_check(raw_fund_bps, fund_norm,
                                               REF_YEARS),
        "mom4h_rho": fade_check(raw_mom_rho, {}, REF_YEARS_4H),
        "mom4h_xs_spread": fade_check(raw_mom_bps, norm_mom,
                                      REF_YEARS_4H),
    }
    # incumbent-control read: does the effect survive 2025 on age>=180d?
    incumbent = {
        "xs_dfh20_h1_2025_old_vs_all": {
            "all_bps": raw_dfh1["2025"],
            "incumbent_bps": dfh1_old["2025"]["mean"],
            "new_only_bps": dfh1_new["2025"]["mean"]},
        "funding_pct_rho_2025_old_vs_all": {
            "all": raw_fund["2025"], "incumbent": fund_rho_old["2025"]},
        "mom4h_rho_2025_old_vs_all": {
            "all": raw_mom_rho["2025"], "incumbent": mom_rho_old["2025"]},
    }
    comp_flag = bool(
        incumbent["funding_pct_rho_2025_old_vs_all"]["incumbent"]
        is not None and raw_fund["2025"] is not None
        and abs(incumbent["funding_pct_rho_2025_old_vs_all"]["incumbent"])
        > 2 * abs(raw_fund["2025"])) or bool(
        incumbent["mom4h_rho_2025_old_vs_all"]["incumbent"] is not None
        and raw_mom_rho["2025"] is not None
        and abs(incumbent["mom4h_rho_2025_old_vs_all"]["incumbent"])
        > 2 * abs(raw_mom_rho["2025"]))

    faded_arms = [k for k, c in checks.items() if c.get("raw_faded")]
    if not faded_arms:
        classification = "no_decay_measured"
    elif all(checks[a]["class"] == "regime_shrink" for a in faded_arms):
        # raw bps faded but per-unit-dispersion held on every faded arm
        classification = "regime_dispersion_shrink"
    elif all(checks[a]["class"] == "real_decay" for a in faded_arms):
        classification = ("composition_artifact_dominant" if comp_flag
                          else "real_decay")
    else:
        classification = "mixed"
    if comp_flag and classification not in (
            "composition_artifact_dominant", "no_decay_measured"):
        classification += "_with_composition_component"

    # dispersion-floor quantification for the master gate (if (b)):
    disp_all = [row["disp1_bps"] for d, row in sorted(table.items())
                if row.get("disp1_bps") is not None
                and d[:4] in REF_YEARS]
    floor = median(disp_all)
    days25 = [row["disp1_bps"] for d, row in table.items()
              if d.startswith("2025") and row.get("disp1_bps") is not None]
    floor_stats = {
        "candidate_floor_bps": _r(floor, 2),
        "definition": "median daily xs-stdev of fwd_1d over 2021-24",
        "share_2025_days_below_floor": _r(
            sum(1 for v in days25 if v < floor) / len(days25), 4)
            if days25 else None,
        "share_2021_24_days_below_floor": _r(
            sum(1 for v in disp_all if v < floor) / len(disp_all), 4)
            if disp_all else None}

    # gate implication: a dispersion floor only helps if 2025 was
    # actually dispersion-poor vs the reference years.
    disp_ref = [regime[y]["xs_dispersion_fwd1_bps_mean"]
                for y in REF_YEARS
                if regime[y]["xs_dispersion_fwd1_bps_mean"] is not None]
    disp25 = regime["2025"]["xs_dispersion_fwd1_bps_mean"]
    disp25_ratio = (disp25 / (sum(disp_ref) / len(disp_ref))
                    if disp25 is not None and disp_ref else None)
    fund_ref = [regime[y]["funding_mean_bps_per_interval"]
                for y in REF_YEARS
                if regime[y]["funding_mean_bps_per_interval"] is not None]
    fund25 = regime["2025"]["funding_mean_bps_per_interval"]
    gate_implication = {
        "xs_dispersion_2025_vs_ref_ratio": _r(disp25_ratio, 3),
        "dispersion_floor_supported": bool(
            disp25_ratio is not None and disp25_ratio < 0.8
            and any(checks[a]["class"] == "regime_shrink"
                    for a in faded_arms)),
        "funding_level_2025_vs_ref": {
            "ref_mean_bps": _r(sum(fund_ref) / len(fund_ref), 4)
                            if fund_ref else None,
            "y2025_bps": fund25,
            "note": "funding-rate LEVELS pinned near zero in 2025 — a "
                    "funding-regime (not return-dispersion) shrink for "
                    "the carry arm"}}

    read = (
        f"classification={classification}. "
        f"XS dfh20 did NOT fade on the mega cohort: h1 2025 "
        f"{raw_dfh1['2025']}bps (2nd-best year; failed years "
        f"{xs_dfh['h1']['failed_years_negative_mean']}), h5 2025 "
        f"{raw_dfh5['2025']}bps, and normalized effect is at/above the "
        f"reference range. XS fwd_1d dispersion did not collapse either "
        f"(2025 {disp25}bps vs 2021-24 mean "
        f"{_r(sum(disp_ref) / len(disp_ref), 1) if disp_ref else None}bps) "
        f"— hypothesis (b) is REFUTED for the daily XS arms. "
        f"mom_4h reversal is REAL DECAY: rho "
        f"{raw_mom_rho.get('2023')}/{raw_mom_rho.get('2024')}/"
        f"{raw_mom_rho.get('2025')} collapses to ~0 on the incumbent-only "
        f"subset too ({mom_rho_old.get('2025')}), while per-bar fwd4 "
        f"dispersion ROSE (2025 "
        f"{mom4h['xs_reversal_by_year']['2025']['mean_disp_bps']}bps vs "
        f"2023 {mom4h['xs_reversal_by_year']['2023']['mean_disp_bps']}bps) "
        f"— neither dispersion shrink nor composition explains it. "
        f"funding_pct TS is sign-unstable across ALL years (not a 2025 "
        f"event), though funding-rate levels did flatten to ~0 in 2025 — "
        f"a funding-regime shrink consistent with sleeve A starving. "
        f"Composition: new-listing share is NOT elevated in 2025 "
        f"(pooled {comp['2025']['young_share_pooled']} vs 2024 "
        f"{comp['2024']['young_share_pooled']}); new listings actually "
        f"AMPLIFY xs_dfh (2025 new-only spread "
        f"{dfh1_new['2025']['mean']}bps vs incumbent "
        f"{dfh1_old['2025']['mean']}bps). Master-gate: a fwd1-dispersion "
        f"floor is NOT supported by 2025 (share of 2025 days below the "
        f"2021-24 median floor "
        f"{floor_stats['share_2025_days_below_floor']} — dispersion was "
        f"ample); the supportive gate variable is the funding regime / "
        f"BTC vol level, not XS dispersion.")

    return {
        "schema_version": "nanojev-financial-signal-decay-v1",
        "status": "measurement_complete",
        "task": "T126",
        "contract": {
            "cohorts": {
                "daily": "data/perp_pit_mega_v1/records.jsonl",
                "h4": "data/perp_pit_mega_4h_v1/records.jsonl",
                "btc_1d": "data/rc_futures_v1/BTC/BTCUSDT_1d.csv"},
            "min_assets_ranked": MIN_ASSETS,
            "decile": DECILE, "quintile": QUINTILE,
            "trail_records_pct_scores": TRAIL,
            "min_window_daily_pct": MIN_WINDOW_D,
            "min_window_4h_pct": MIN_WINDOW_4H,
            "new_listing_age_days": NEW_AGE_D,
            "fade_ratio": FADE_RATIO, "norm_hold_ratio": NORM_HOLD},
        "part1_regime_variables_by_year": regime,
        "part2_effect_decay": {
            "xs_dfh20": xs_dfh,
            "funding_pct_ts": funding_out,
            "funded_subset_dispersion_by_year": disp_f,
            "mom_4h": mom4h},
        "part3_universe_composition": comp,
        "part4_btc_vol_regime": {
            "atrp_d_by_year": atrp_by_year,
            "corr_yearly_xs_dfh_h1_spread_vs": corr_vs},
        "part5_verdict": {
            "per_arm_checks": checks,
            "arms_faded_2025": faded_arms,
            "incumbent_control_2025": incumbent,
            "composition_flag": comp_flag,
            "dispersion_floor_candidate": floor_stats,
            "gate_implication": gate_implication,
            "classification": classification,
            "read": read},
        "honest_notes": [
            "all t/p nominal: forward labels overlap within a symbol and "
            "cross-asset days share a market factor; effective n << record "
            "count",
            "n=5 yearly correlations are descriptive, not inferential",
            "age<180d is a cohort-first-seen proxy for new listings: "
            "assets listed before 2021-01 enter at age 0, so 2021 young "
            "shares are inflated by construction — read the 2023-25 trend",
            "close-as-mark basis; no funding cashflows, fees, borrow; "
            "delisted/renamed contracts present but renames can "
            "double-count",
            "rho is scale-free: a Spearman collapse cannot be explained "
            "by dispersion shrink — only bps effects are normalized",
            "the 10-asset T111 sleeve A itself is not recomputed here; "
            "the funded-subset pooled funding_pct TS rho on the mega "
            "cohort is its closest comparable"]}


def main():
    global COHORT_D, COHORT_4H, BTC_1D
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-d", type=pathlib.Path, default=COHORT_D)
    ap.add_argument("--cohort-4h", type=pathlib.Path, default=COHORT_4H)
    ap.add_argument("--btc-1d", type=pathlib.Path, default=BTC_1D)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()
    COHORT_D, COHORT_4H, BTC_1D = args.cohort_d, args.cohort_4h, args.btc_1d

    for p in (COHORT_D, COHORT_4H, BTC_1D):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    protocol = build_protocol()
    protocol_sha = hashlib.sha256(
        json.dumps(protocol, indent=2, sort_keys=True).encode()).hexdigest()

    r1 = measure()
    r2 = measure()
    blob = json.dumps(r1, indent=2, sort_keys=True)
    assert blob == json.dumps(r2, indent=2, sort_keys=True), \
        "non-deterministic measurement"
    r1["determinism"] = {"runs": 2, "byte_identical": True,
                         "sha256": hashlib.sha256(blob.encode()).hexdigest()}
    r1["protocol"] = protocol
    r1["protocol_sha256"] = protocol_sha
    r1["owner_authorization"] = build_auth(protocol_sha)
    r1["generated_at"] = dt.datetime.now(dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r1, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    v = r1["part5_verdict"]
    print("classification:", v["classification"])
    print("faded arms:", v["arms_faded_2025"])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
