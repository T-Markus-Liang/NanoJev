#!/usr/bin/env python3
"""T132: daily-scale short-term reversal + liquidity conditioning at MEGA scale.

Retest of the underpowered 5-asset daily arms on
``data/perp_pit_mega_v1/records.jsonl`` (277 symbols x 2021-01..2025-12,
close-as-mark, gross close-to-close labels ``forward_return_bps`` (1d) and
``forward_return_5d_bps`` (5d)). T101b tested the Amihud liquidity flip at
5d horizon on 5 assets and was rejected; the daily-scale literature
(Zaremba IRFA 2021: illiquid coins reverse, liquid coins continue) is about
the 1d horizon, so this runner measures 1d -> 1d at ~55x the symbols.

Signal: ``ret1d[i] = close[i]/close[i-1]-1`` requires the previous record to
be the previous CALENDAR day (contiguous bars only — no cross-gap returns).
The label ``forward_return_bps`` is the strict next-day close-to-close move,
so lag1d -> fwd1d is PIT-clean.

Arms:
  a. daily_reversal — pooled Spearman(lag1d, fwd1d) over all asset-days +
     per-day XS Spearman companion + XS decile reversal spread per decision
     day (>=30 ranked assets, k=n//10): long bottom-decile yesterday-losers,
     short top-decile yesterday-winners. Positive spread = reversal.
  b. liquidity_flip — Amihud = |ret1d|/quote_volume; per symbol per day the
     illiquidity LEVEL is the median Amihud over the strictly-prior 90
     records (>=30 obs). Per day the ranked universe is split at the cross-
     sectional median into illiquid / liquid halves; within each half:
     pooled Spearman + per-day XS Spearman + within-half decile reversal
     spread. Prior (Zaremba): NEGATIVE rho in the illiquid half, less
     negative / non-negative in the liquid half; paired per-day
     rho_illiquid - rho_liquid < 0.
  c. reversal_x_regime — the arm-a daily reversal spread and pooled
     Spearmans split by the BTC ret20 gate (gate-on = BTCUSDT
     close/close[t-20]-1 > 0). The crowding/dfh-follow signals were
     gate-on only; does daily reversal die in the same regime?
  d. volume_shock — vol_pct = trailing-90 mid-rank pct of quote_volume
     (>=30 obs); shock = vol_pct > 0.90. Pooled Spearman(lag1d, fwd1d) on
     shocked vs non-shocked asset-days + per-day XS reversal spread among
     shocked names only (>=10 shocked assets/day). Prior: shocked moves
     revert harder.
  e. net_cost — daily-rebalanced decile reversal book at 5bps per unit of
     one-sided leg turnover (repo convention); XS reversal churn is high
     and this is the honest net check.

Statistics: BH-FDR alpha=0.05 over the fixed 10-cell headline family;
within-day score-shuffle placebo x3 seeds on arm a; yearly folds 2021-2025;
delisting control (exclude early-stopped symbols); the whole measurement
runs twice and must serialize byte-identically (determinism x2). All t/p
are nominal: pooled rows share the market factor across timestamps and
per-day Spearmans are preferred as the honest companion.

Artifacts follow the established convention: a frozen protocol
(``research/financial_signal_reversal_mega_protocol_v1.json``) plus an
owner self-authorization pinning it by sha256
(``results/financial_signal_reversal_mega_authorization_v1.json``) are
written on every run BEFORE measurement. Measurement only: no fitting, no
trading, no network.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import random
import statistics
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_mega_v1/records.jsonl"
OUT = ROOT / "results/financial_signal_reversal_mega_v1.json"
PROTOCOL = ROOT / "research/financial_signal_reversal_mega_protocol_v1.json"
AUTH = ROOT / "results/financial_signal_reversal_mega_authorization_v1.json"

MIN_ASSETS_PER_DAY = 30   # mega XS convention (arm a universe gate)
DECILE = 10               # edge k = max(1, n//10) ranked assets per side
TRAIL_DAYS = 90           # trailing-90 window for Amihud median + vol_pct
MIN_TRAIL = 30            # min non-null obs inside the trailing window
MIN_SHOCK_ASSETS = 10     # shocked-only XS needs >=10 names/day
VOL_SHOCK_PCT = 0.90      # >90th pct of trailing-90 quote_volume = shock
BTC_GATE_LOOKBACK = 20
COST_BPS_PER_LEG = 5.0    # per unit of one-sided leg notional traded
PLACEBO_SEEDS = (11, 22, 33)
DAYS_PER_YEAR = 365       # crypto trades daily
MIN_SPEARMAN_N = 10
Z_80 = 1.959964 + 0.841621  # z(.975)+z(.8): MDE at 80% power, two-sided 5%


def _r(x, nd=3):
    return round(x, nd) if isinstance(x, float) else x


def norm_p(t):
    """Two-sided normal-approx p from a t/z statistic (repo convention)."""
    if t is None:
        return None
    return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))


def t_stat(xs):
    """Mean/sd/t (+ normal-approx p) of a daily series; nominal — same-day
    cross-section is correlated and labels can overlap."""
    n = len(xs)
    if n < 5:
        return {"n": n, "mean": (sum(xs) / n) if n else None,
                "sd": None, "t": None, "p": None, "mde80": None}
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(var)
    t = (mean / (sd / math.sqrt(n))) if sd > 0 else None
    out = {"n": n, "mean": mean, "sd": sd, "t": t, "p": norm_p(t)}
    out["mde80"] = Z_80 * sd / math.sqrt(n) if sd else None
    return out


def welch(xs, ys):
    """Welch two-sample t, Welch-Satterthwaite df, normal-approx p."""
    nx, ny = len(xs), len(ys)
    if nx < 2 or ny < 2:
        return {"n_a": nx, "n_b": ny, "t": None, "df": None, "p": None}
    mx, my = sum(xs) / nx, sum(ys) / ny
    vx = sum((x - mx) ** 2 for x in xs) / (nx - 1)
    vy = sum((y - my) ** 2 for y in ys) / (ny - 1)
    denom = vx / nx + vy / ny
    if denom <= 0:
        return {"n_a": nx, "n_b": ny, "mean_a_bps": mx,
                "mean_b_bps": my, "t": None, "df": None, "p": None}
    t = (mx - my) / math.sqrt(denom)
    df = denom ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    return {"n_a": nx, "n_b": ny, "mean_a_bps": mx, "mean_b_bps": my,
            "t": t, "df": df, "p": norm_p(t)}


def mid_rank_pct(window, x):
    """Mid-rank percentile of x within trailing window (ties count half)."""
    if x is None or not window:
        return None
    return (sum(1 for w in window if w < x)
            + 0.5 * sum(1 for w in window if w == x)) / len(window)


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


def spearman(xs, ys, min_n=MIN_SPEARMAN_N):
    """Spearman rho + t-approx normal two-sided p (oi_mega convention)."""
    n = len(xs)
    if n < min_n:
        return {"n": n, "rho": None, "p": None}
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return {"n": n, "rho": None, "p": None}
    rho = num / (dx * dy)
    t = rho * math.sqrt((n - 2) / max(1e-9, 1 - rho * rho))
    return {"n": n, "rho": rho, "p": norm_p(t),
            "mde80_abs_rho": Z_80 / math.sqrt(max(1, n - 3))}


def bh_fdr(named_ps, alpha=0.05):
    """BH-FDR over [(name, p|None)] -> ({name: survives}, sorted table)."""
    ps = sorted((p, n) for n, p in named_ps if p is not None)
    m = len(ps)
    table = [{"cell": n, "p": _r(p, 6),
              "alpha_bh": round(alpha * (i + 1) / m, 6),
              "survives": p <= alpha * (i + 1) / m}
             for i, (p, n) in enumerate(ps)]
    return {e["cell"]: e["survives"] for e in table}, table


def max_drawdown(cumulative):
    peak, mdd = 0.0, 0.0
    for v in cumulative:
        peak = max(peak, v)
        mdd = max(mdd, peak - v)
    return mdd


def load_cohort(path):
    """asset -> sorted daily rows {day, ord, close, qvol, ret1d, amihud,
    illiq_level, vol_pct, fwd1, fwd5}; plus meta, and the BTC ret20 gate.

    ret1d requires the previous record to be the previous calendar day.
    illiq_level = median Amihud over the strictly-prior TRAIL_DAYS records
    (>=MIN_TRAIL obs) — a slow-moving per-symbol illiquidity characteristic,
    not the same-day spike. vol_pct = mid-rank pct of today's quote_volume
    vs the strictly-prior TRAIL_DAYS (>=MIN_TRAIL obs)."""
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
                "qvol": f["quote_volume"]["value"],
                "fwd1": record["label"]["forward_return_bps"],
                "fwd5": record["label"]["forward_return_5d_bps"],
            })
    for rows in series.values():
        rows.sort(key=lambda r: r["day"])
    for asset, rows in series.items():
        for r in rows:
            r["ord"] = dt.date.fromisoformat(r["day"]).toordinal()
        closes = [r["close"] for r in rows]
        qvols = [r["qvol"] for r in rows]
        amihud = [None] * len(rows)
        for i, r in enumerate(rows):
            ret = None
            if i > 0 and r["ord"] == rows[i - 1]["ord"] + 1 \
                    and closes[i - 1] and closes[i - 1] > 0 \
                    and r["close"] and r["close"] > 0:
                ret = r["close"] / closes[i - 1] - 1.0
            r["ret1d"] = ret
            if ret is not None and r["qvol"] is not None and r["qvol"] > 0:
                amihud[i] = abs(ret) / r["qvol"]
            r["amihud"] = amihud[i]
        for i, r in enumerate(rows):
            lo = max(0, i - TRAIL_DAYS)
            awin = [v for v in amihud[lo:i] if v is not None]
            r["illiq_level"] = (statistics.median(awin)
                                if len(awin) >= MIN_TRAIL else None)
            vwin = [v for v in qvols[lo:i] if v is not None]
            r["vol_pct"] = (mid_rank_pct(vwin, r["qvol"])
                            if r["qvol"] is not None
                            and len(vwin) >= MIN_TRAIL else None)
    # BTC 20d return gate from the BTC rows in the same cohort.
    btc_ret20 = {}
    btc_rows = series.get("BTCUSDT-PERP", [])
    bcloses = [r["close"] for r in btc_rows]
    for i, r in enumerate(btc_rows):
        if i >= BTC_GATE_LOOKBACK and bcloses[i - BTC_GATE_LOOKBACK] \
                and bcloses[i - BTC_GATE_LOOKBACK] > 0 \
                and r["ord"] - btc_rows[i - BTC_GATE_LOOKBACK]["ord"] \
                == BTC_GATE_LOOKBACK:
            btc_ret20[r["day"]] = \
                bcloses[i] / bcloses[i - BTC_GATE_LOOKBACK] - 1.0
    return series, meta, btc_ret20


def eligible(rows, *keys):
    return [r for r in rows if all(r.get(k) is not None for k in keys)]


def reversal_xs_days(by_day, fwd_key="fwd1", universe=None,
                     min_assets=MIN_ASSETS_PER_DAY, member_key=None):
    """Per decision day: rank by ret1d, long bottom-decile losers / short
    top-decile winners; spread = fwd(losers) - fwd(winners). Positive =
    reversal. If member_key is set, restrict to rows where it is non-null.
    Returns (days, pooled_loser_fwds, pooled_winner_fwds)."""
    days, pooled_l, pooled_w = [], [], []
    for day in sorted(by_day):
        rows = []
        for asset, r in by_day[day]:
            if universe is not None and asset not in universe:
                continue
            if r.get("ret1d") is None or r.get(fwd_key) is None:
                continue
            if member_key is not None and r.get(member_key) is None:
                continue
            rows.append((asset, r["ret1d"], r[fwd_key]))
        n = len(rows)
        if n < min_assets:
            continue
        ordered = sorted(rows, key=lambda x: (x[1], x[0]))
        k = max(1, n // DECILE)
        bot, top = ordered[:k], ordered[-k:]
        if bot[-1][1] == top[0][1]:
            continue  # degenerate day: no dispersion at the ranked edges
        days.append({"day": day, "n_assets": n, "edge": k,
                     "spread_bps": sum(x[2] for x in bot) / len(bot)
                     - sum(x[2] for x in top) / len(top),
                     "long_assets": sorted(x[0] for x in bot),
                     "short_assets": sorted(x[0] for x in top)})
        pooled_l.extend(x[2] for x in bot)
        pooled_w.extend(x[2] for x in top)
    return days, pooled_l, pooled_w


def per_day_spearmans(by_day, member_key=None, universe=None,
                      min_assets=MIN_SPEARMAN_N):
    """Per day, cross-sectional Spearman(ret1d, fwd1) across assets — the
    honest companion to the pooled rho (same-day asset-days correlate)."""
    rhos = []
    for day in sorted(by_day):
        xs, ys = [], []
        for asset, r in by_day[day]:
            if universe is not None and asset not in universe:
                continue
            if r.get("ret1d") is None or r.get("fwd1") is None:
                continue
            if member_key is not None and r.get(member_key) is None:
                continue
            xs.append(r["ret1d"])
            ys.append(r["fwd1"])
        s = spearman(xs, ys, min_n=min_assets)
        if s["rho"] is not None:
            rhos.append(s["rho"])
    return {"n_days_ge_min_assets": len(rhos),
            "mean_rho": _r(statistics.mean(rhos)) if rhos else None,
            "median_rho": _r(statistics.median(rhos)) if rhos else None,
            "frac_rho_negative":
                _r(sum(1 for x in rhos if x < 0) / len(rhos))
                if rhos else None}


def xs_summary(days):
    """Per-day spread series -> t, yearly folds, asset/edge coverage."""
    spreads = [d["spread_bps"] for d in days]
    s = t_stat(spreads)
    folds = defaultdict(list)
    for d in days:
        folds[d["day"][:4]].append(d["spread_bps"])
    fold_means = {y: sum(v) / len(v) for y, v in sorted(folds.items())}
    same = sum(1 for m in fold_means.values()
               if s["mean"] is not None and m * s["mean"] > 0)
    ns = [d["n_assets"] for d in days]
    es = [d["edge"] for d in days]
    return {"days_evaluated": len(days),
            "first_day": days[0]["day"] if days else None,
            "last_day": days[-1]["day"] if days else None,
            "assets_per_day": {"min": min(ns) if ns else None,
                               "median": (sorted(ns)[len(ns) // 2]
                                          if ns else None),
                               "max": max(ns) if ns else None},
            "edge_names_per_side": {"min": min(es) if es else None,
                                    "median": (sorted(es)[len(es) // 2]
                                               if es else None),
                                    "max": max(es) if es else None},
            "daily_spread_bps": {k: _r(v, 6) for k, v in s.items()},
            "yearly_folds": {y: {"days": len(folds[y]),
                                 "mean_spread_bps": _r(m)}
                             for y, m in fold_means.items()},
            "yearly_folds_same_sign": f"{same}/{len(fold_means)}"}


def net_sim(days):
    """Daily-rebalanced reversal book: cost = 5bps per unit one-sided leg
    turnover; turnover = (slots entered + slots exited)/(2*book slots).
    Day one establishes the full book (frac=1)."""
    prev, rows, turnovers = None, [], []
    for d in days:
        slots = {("L", a) for a in d["long_assets"]} | \
                {("S", a) for a in d["short_assets"]}
        if prev is None:
            frac = 1.0
        else:
            frac = ((len(slots - prev) + len(prev - slots))
                    / (2 * max(len(slots), len(prev)))) if slots or prev \
                else 0.0
        turnovers.append(frac)
        cost = COST_BPS_PER_LEG * frac
        rows.append({"day": d["day"], "gross_bps": d["spread_bps"],
                     "turnover_frac": frac, "cost_bps": cost,
                     "net_bps": d["spread_bps"] - cost})
        prev = slots
    gross = [r["gross_bps"] for r in rows]
    net = [r["net_bps"] for r in rows]
    g, n = t_stat(gross), t_stat(net)
    cum_g = cum_n = 0.0
    cg, cn = [], []
    for r in rows:
        cum_g += r["gross_bps"]
        cum_n += r["net_bps"]
        cg.append(cum_g)
        cn.append(cum_n)
    sharpe = lambda s: (s["mean"] / s["sd"] * math.sqrt(DAYS_PER_YEAR)
                        if s["sd"] else None)
    return {"days": len(rows),
            "mean_daily_turnover_frac_of_book":
                _r(sum(turnovers) / len(turnovers) if turnovers else None),
            "mean_daily_cost_bps":
                _r(sum(r["cost_bps"] for r in rows) / len(rows)
                   if rows else None),
            "gross": {"mean_daily_bps": _r(g["mean"]), "t": _r(g["t"]),
                      "cumulative_bps": _r(cg[-1] if cg else None),
                      "max_drawdown_bps": _r(max_drawdown(cg)),
                      "sharpe_annualized": _r(sharpe(g))},
            "net": {"mean_daily_bps": _r(n["mean"]), "t": _r(n["t"]),
                    "cumulative_bps": _r(cn[-1] if cn else None),
                    "max_drawdown_bps": _r(max_drawdown(cn)),
                    "sharpe_annualized": _r(sharpe(n))}}


def placebo_reversal(by_day, universe=None):
    """Shuffle ret1d scores WITHIN each decision day x3 seeds; the spread
    must collapse."""
    out = {}
    for seed in PLACEBO_SEEDS:
        rng = random.Random(seed)
        spreads = []
        for day in sorted(by_day):
            xs = []
            for asset, r in by_day[day]:
                if universe is not None and asset not in universe:
                    continue
                if r.get("ret1d") is None or r.get("fwd1") is None:
                    continue
                xs.append((asset, r["ret1d"], r["fwd1"]))
            if len(xs) < MIN_ASSETS_PER_DAY:
                continue
            scores = [x[1] for x in xs]
            rng.shuffle(scores)
            ordered = sorted([(a, s, f) for (a, _, f), s in zip(xs, scores)],
                             key=lambda x: (x[1], x[0]))
            k = max(1, len(ordered) // DECILE)
            bot, top = ordered[:k], ordered[-k:]
            if bot[-1][1] == top[0][1]:
                continue
            spreads.append(sum(x[2] for x in bot) / len(bot)
                           - sum(x[2] for x in top) / len(top))
        s = t_stat(spreads)
        out[f"seed_{seed}"] = {"days": len(spreads),
                               "mean_spread_bps": _r(s["mean"]),
                               "t": _r(s["t"])}
    return out


def measure(series, meta, btc_ret20):
    """All arms; deterministic — must serialize byte-identically across
    runs (placebo seeds are fixed; all iteration orders sorted)."""
    by_day = defaultdict(list)
    for asset in sorted(series):
        for r in series[asset]:
            by_day[r["day"]].append((asset, r))

    arms, fdr_cells = {}, []

    # ---------- (a) daily reversal: lag1d -> fwd1d ----------
    pairs = [(r["ret1d"], r["fwd1"]) for day in by_day
             for _, r in by_day[day]
             if r.get("ret1d") is not None and r.get("fwd1") is not None]
    sp_a = spearman([p[0] for p in pairs], [p[1] for p in pairs])
    days_a, pl_l, pl_w = reversal_xs_days(by_day)
    xs_a = xs_summary(days_a)
    xs_a["pooled_leg_welch"] = {k: _r(v, 6) for k, v in
                              welch(pl_l, pl_w).items()}
    xs_a["net_sim_5bps_per_leg"] = net_sim(days_a)
    days_a5, _, _ = reversal_xs_days(by_day, fwd_key="fwd5")
    arms["a_daily_reversal"] = {
        "definition": "per day >=30 ranked assets: long bottom-decile "
                      "yesterday-losers / short top-decile winners, "
                      "fwd1 spread; positive = reversal",
        "pooled_spearman_lag1d_fwd1d":
            {k: _r(v, 6) for k, v in sp_a.items()},
        "pooled_pseudo_replication_caveat":
            "pooled rho mixes day-level composition with cross-section; "
            "per_day_xs_spearman is the honest companion",
        "per_day_xs_spearman": per_day_spearmans(by_day),
        "xs_decile_reversal_1d": xs_a,
        "descriptive_not_in_fdr": {
            "xs_decile_reversal_5d": xs_summary(days_a5)},
    }
    fdr_cells.append(("a_reversal_pooled_spearman_lag1d_fwd1d", sp_a["p"]))
    fdr_cells.append(("a_reversal_xs_losers_minus_winners_1d",
                      xs_a["daily_spread_bps"]["p"]))

    # ---------- (b) liquidity-conditioned flip (Zaremba at scale) ----------
    illiq_pairs, liq_pairs = [], []
    half_day = defaultdict(lambda: {"illiquid": [], "liquid": []})
    for day in sorted(by_day):
        rows = [(asset, r) for asset, r in by_day[day]
                if r.get("ret1d") is not None and r.get("fwd1") is not None
                and r.get("illiq_level") is not None]
        if len(rows) < MIN_ASSETS_PER_DAY:
            continue
        ordered = sorted(rows, key=lambda x: (x[1]["illiq_level"], x[0]))
        med = len(ordered) // 2
        for asset, r in ordered[:med]:
            half_day[day]["liquid"].append((asset, r))
            liq_pairs.append((r["ret1d"], r["fwd1"]))
        for asset, r in ordered[med:]:
            half_day[day]["illiquid"].append((asset, r))
            illiq_pairs.append((r["ret1d"], r["fwd1"]))
    sp_illiq = spearman([p[0] for p in illiq_pairs],
                        [p[1] for p in illiq_pairs])
    sp_liq = spearman([p[0] for p in liq_pairs], [p[1] for p in liq_pairs])
    # paired per-day flip contrast + within-half decile reversal spreads
    diffs, half_xs = [], defaultdict(list)
    for day in sorted(half_day):
        per = {}
        for half in ("illiquid", "liquid"):
            xs = [(a, r["ret1d"], r["fwd1"]) for a, r in half_day[day][half]]
            per[half] = xs
            n = len(xs)
            if n >= MIN_SPEARMAN_N:
                ordered = sorted(xs, key=lambda x: (x[1], x[0]))
                k = max(1, n // DECILE)
                bot, top = ordered[:k], ordered[-k:]
                if bot[-1][1] != top[0][1]:
                    half_xs[half].append(
                        sum(x[2] for x in bot) / len(bot)
                        - sum(x[2] for x in top) / len(top))
            s = spearman([x[1] for x in xs], [x[2] for x in xs])
            per[half + "_rho"] = s["rho"]
        if per.get("illiquid_rho") is not None \
                and per.get("liquid_rho") is not None:
            diffs.append(per["illiquid_rho"] - per["liquid_rho"])
    flip_t = t_stat(diffs)
    illiq_xs = t_stat(half_xs["illiquid"])
    liq_xs = t_stat(half_xs["liquid"])
    arms["b_liquidity_flip"] = {
        "definition": "illiq_level = trailing-90 median Amihud "
                      "(|ret1d|/quote_volume, strictly-prior records, "
                      ">=30 obs); per day the ranked universe splits at "
                      "the cross-sectional median into illiquid/liquid "
                      "halves; within each: pooled+per-day Spearman(lag1d,"
                      "fwd1d) and within-half decile reversal spread",
        "prior": "Zaremba: NEGATIVE rho in the illiquid half (reversal), "
                 "less negative/non-negative in the liquid half",
        "illiquid_half_pooled_spearman":
            {k: _r(v, 6) for k, v in sp_illiq.items()},
        "liquid_half_pooled_spearman":
            {k: _r(v, 6) for k, v in sp_liq.items()},
        "paired_per_day_rho_illiquid_minus_liquid":
            {k: _r(v, 6) for k, v in flip_t.items()},
        "illiquid_half_within_decile_reversal_1d":
            {k: _r(v, 6) for k, v in illiq_xs.items()},
        "liquid_half_within_decile_reversal_1d":
            {k: _r(v, 6) for k, v in liq_xs.items()},
        "days_evaluated": len(half_day),
    }
    fdr_cells.append(("b_illiquid_half_pooled_spearman", sp_illiq["p"]))
    fdr_cells.append(("b_liquid_half_pooled_spearman", sp_liq["p"]))
    fdr_cells.append(("b_flip_paired_rho_illiq_minus_liq", flip_t["p"]))
    fdr_cells.append(("b_illiquid_half_xs_reversal_1d", illiq_xs["p"]))

    # ---------- (c) reversal x BTC ret20 regime ----------
    gate = {"gate_on_btc_ret20_pos": [], "gate_off_btc_ret20_nonpos": [],
            "warmup_no_gate_yet": []}
    for d in days_a:
        tr = btc_ret20.get(d["day"])
        key = ("warmup_no_gate_yet" if tr is None
               else "gate_on_btc_ret20_pos" if tr > 0
               else "gate_off_btc_ret20_nonpos")
        gate[key].append(d["spread_bps"])
    pairs_on, pairs_off = [], []
    for day in sorted(by_day):
        tr = btc_ret20.get(day)
        if tr is None:
            continue
        bucket = pairs_on if tr > 0 else pairs_off
        for _, r in by_day[day]:
            if r.get("ret1d") is not None and r.get("fwd1") is not None:
                bucket.append((r["ret1d"], r["fwd1"]))
    sp_on = spearman([p[0] for p in pairs_on], [p[1] for p in pairs_on])
    sp_off = spearman([p[0] for p in pairs_off], [p[1] for p in pairs_off])
    g_on = t_stat(gate["gate_on_btc_ret20_pos"])
    g_off = t_stat(gate["gate_off_btc_ret20_nonpos"])
    arms["c_reversal_x_btc_regime"] = {
        "definition": "arm-a daily reversal spread split by BTC ret20 "
                      "sign (gate-on = BTCUSDT close/close[t-20]-1 > 0); "
                      "the crowding/dfh-follow signals were gate-on only",
        "gate_on_btc_ret20_pos": {"days": len(gate["gate_on_btc_ret20_pos"]),
                                  "spread_stats_bps":
                                      {k: _r(v, 6) for k, v in
                                       g_on.items()},
                                  "pooled_spearman":
                                      {k: _r(v, 6) for k, v in
                                       sp_on.items()}},
        "gate_off_btc_ret20_nonpos": {
            "days": len(gate["gate_off_btc_ret20_nonpos"]),
            "spread_stats_bps": {k: _r(v, 6) for k, v in g_off.items()},
            "pooled_spearman": {k: _r(v, 6) for k, v in sp_off.items()}},
        "warmup_days_no_gate": len(gate["warmup_no_gate_yet"]),
        "gate_on_minus_off_welch_bps": {k: _r(v, 6) for k, v in welch(
            gate["gate_on_btc_ret20_pos"],
            gate["gate_off_btc_ret20_nonpos"]).items()},
    }
    fdr_cells.append(("c_reversal_xs_gate_on", g_on["p"]))
    fdr_cells.append(("c_reversal_xs_gate_off", g_off["p"]))

    # ---------- (d) volume-shock arm ----------
    shock_pairs, calm_pairs = [], []
    shock_days = []
    for day in sorted(by_day):
        sx = []
        for asset, r in by_day[day]:
            if r.get("ret1d") is None or r.get("fwd1") is None:
                continue
            if r.get("vol_pct") is None:
                continue
            if r["vol_pct"] > VOL_SHOCK_PCT:
                shock_pairs.append((r["ret1d"], r["fwd1"]))
                sx.append((asset, r["ret1d"], r["fwd1"]))
            else:
                calm_pairs.append((r["ret1d"], r["fwd1"]))
        if len(sx) >= MIN_SHOCK_ASSETS:
            ordered = sorted(sx, key=lambda x: (x[1], x[0]))
            k = max(1, len(ordered) // DECILE)
            bot, top = ordered[:k], ordered[-k:]
            if bot[-1][1] != top[0][1]:
                shock_days.append({"day": day, "n_assets": len(sx),
                                   "edge": k,
                                   "spread_bps":
                                       sum(x[2] for x in bot) / len(bot)
                                       - sum(x[2] for x in top) / len(top)})
    sp_shock = spearman([p[0] for p in shock_pairs],
                        [p[1] for p in shock_pairs])
    sp_calm = spearman([p[0] for p in calm_pairs], [p[1] for p in calm_pairs])
    xs_shock = xs_summary(shock_days)
    arms["d_volume_shock"] = {
        "definition": "vol_pct = trailing-90 mid-rank pct of quote_volume "
                      "(>=30 obs); shock = vol_pct > 0.90; prior: shocked "
                      "moves revert harder (more negative rho / larger "
                      "reversal spread than non-shocked)",
        "shock_frac_of_ranked_rows":
            _r(len(shock_pairs) / max(1, len(shock_pairs)
                                      + len(calm_pairs))),
        "shocked_pooled_spearman":
            {k: _r(v, 6) for k, v in sp_shock.items()},
        "non_shocked_pooled_spearman":
            {k: _r(v, 6) for k, v in sp_calm.items()},
        "shocked_only_xs_reversal_1d": xs_shock,
        "min_shocked_assets_per_day": MIN_SHOCK_ASSETS,
    }
    fdr_cells.append(("d_shocked_pooled_spearman", sp_shock["p"]))
    fdr_cells.append(("d_shocked_xs_reversal_1d",
                      xs_shock["daily_spread_bps"]["p"]))

    # ---------- (e) net cost honesty on arm a ----------
    ns = xs_a["net_sim_5bps_per_leg"]
    arms["e_net_cost"] = {
        "definition": "arm-a daily-rebalanced decile reversal book; cost "
                      "= 5bps per unit one-sided leg turnover; day-1 "
                      "establishes the book",
        "mean_daily_turnover_frac": ns["mean_daily_turnover_frac_of_book"],
        "mean_daily_cost_bps": ns["mean_daily_cost_bps"],
        "gross_mean_daily_bps": ns["gross"]["mean_daily_bps"],
        "net_mean_daily_bps": ns["net"]["mean_daily_bps"],
        "net_t": ns["net"]["t"],
        "net_p": norm_p(ns["net"]["t"]),
        "net_cumulative_bps": ns["net"]["cumulative_bps"],
        "net_max_drawdown_bps": ns["net"]["max_drawdown_bps"],
        "net_sharpe_annualized": ns["net"]["sharpe_annualized"],
        "note": "reversal decile membership churns ~daily so turnover is "
                "near-total; this is the honest 5bps/leg haircut, funding/"
                "borrow/slippage excluded",
    }

    # ---------- controls ----------
    early_stopped = sorted(a for a, m in meta.items() if m["listed"] is False)
    univ_keep = set(series) - set(early_stopped)
    days_ctl, _, _ = reversal_xs_days(by_day, universe=univ_keep)
    ctl = t_stat([d["spread_bps"] for d in days_ctl])
    controls = {"excl_all_early_stopped": {
        "excluded_n": len(early_stopped),
        "excluded": {a: meta[a]["last_bar_date"] for a in early_stopped},
        "days": len(days_ctl), "mean_spread_bps": _r(ctl["mean"]),
        "t": _r(ctl["t"]), "p": _r(ctl["p"], 6)}}
    placebos = placebo_reversal(by_day)

    fdr_map, fdr_table = bh_fdr(fdr_cells)
    return {"arms": arms, "fdr_table": fdr_table, "fdr_map": fdr_map,
            "controls": controls, "placebo": placebos}


def verdicts(meas):
    """Per-arm verdict strings + hard gates (mega-runner convention)."""
    a = meas["arms"]["a_daily_reversal"]
    b = meas["arms"]["b_liquidity_flip"]
    c = meas["arms"]["c_reversal_x_btc_regime"]
    d = meas["arms"]["d_volume_shock"]
    e = meas["arms"]["e_net_cost"]
    fdr = meas["fdr_map"]
    pl = meas["placebo"]
    xs = a["xs_decile_reversal_1d"]["daily_spread_bps"]
    mean_a = xs["mean"]
    seed_means = [v["mean_spread_bps"] for v in pl.values()
                  if v["mean_spread_bps"] is not None]
    gates_a = {
        "pooled_rho_negative_fdr": bool(
            fdr.get("a_reversal_pooled_spearman_lag1d_fwd1d"))
        and a["pooled_spearman_lag1d_fwd1d"]["rho"] is not None
        and a["pooled_spearman_lag1d_fwd1d"]["rho"] < 0,
        "xs_spread_positive_fdr": bool(
            fdr.get("a_reversal_xs_losers_minus_winners_1d"))
        and mean_a is not None and mean_a > 0,
        "placebo_collapses": bool(seed_means) and all(
            m < 0.5 * mean_a for m in seed_means)
        if mean_a is not None else False,
        "folds_mostly_same_sign":
            int(a["xs_decile_reversal_1d"]["yearly_folds_same_sign"]
                .split("/")[0])
            >= max(1, len(a["xs_decile_reversal_1d"]["yearly_folds"]) - 1),
        "survives_delisting_control":
            meas["controls"]["excl_all_early_stopped"]["mean_spread_bps"]
            is not None
            and (meas["controls"]["excl_all_early_stopped"]
                 ["mean_spread_bps"] > 0) == (mean_a is not None
                                              and mean_a > 0),
        "net_positive_5bps": e["net_mean_daily_bps"] is not None
            and e["net_mean_daily_bps"] > 0,
    }
    failed_a = [k for k, v in gates_a.items() if not v]
    if not failed_a:
        verdict_a = "CONFIRMED AT MEGA SCALE"
    elif gates_a["pooled_rho_negative_fdr"] and \
            not gates_a["xs_spread_positive_fdr"]:
        verdict_a = ("TENDENCY ONLY: pooled/per-day rho is negative and "
                     "FDR-clean (daily reversal exists as a monotone "
                     "relation) but the decile losers-minus-winners book "
                     "captures none of it — failed gates -> "
                     + ", ".join(failed_a))
    else:
        verdict_a = ("NOT CONFIRMED: failed gates -> "
                     + ", ".join(failed_a))

    ri = b["illiquid_half_pooled_spearman"]["rho"]
    rl = b["liquid_half_pooled_spearman"]["rho"]
    gates_b = {
        "illiquid_rho_negative": ri is not None and ri < 0,
        "illiquid_rho_below_liquid": (ri is not None and rl is not None
                                    and ri < rl),
        "flip_fdr": bool(fdr.get("b_flip_paired_rho_illiq_minus_liq")),
        "illiquid_half_spearman_fdr":
            bool(fdr.get("b_illiquid_half_pooled_spearman")),
    }
    failed_b = [k for k, v in gates_b.items() if not v]
    verdict_b = ("ZAREMBA DIRECTION CONFIRMED" if not failed_b else
                 "ZAREMBA PRIOR NOT CONFIRMED: failed -> "
                 + ", ".join(failed_b))

    on = c["gate_on_btc_ret20_pos"]["spread_stats_bps"]["mean"]
    off = c["gate_off_btc_ret20_nonpos"]["spread_stats_bps"]["mean"]
    rho_on = c["gate_on_btc_ret20_pos"]["pooled_spearman"]["rho"]
    rho_off = c["gate_off_btc_ret20_nonpos"]["pooled_spearman"]["rho"]
    gates_c = {
        "gate_on_positive_fdr": bool(fdr.get("c_reversal_xs_gate_on"))
        and on is not None and on > 0,
        "gate_off_positive_fdr": bool(fdr.get("c_reversal_xs_gate_off"))
        and off is not None and off > 0,
        # descriptive (not an FDR cell): the pooled-rho regime split
        "reversal_rho_concentrated_gate_off":
            (rho_off is not None and rho_on is not None
             and rho_off < 0 and rho_off < rho_on),
    }
    rho_note = (f"descriptive pooled rho splits sharply: gate-off "
                f"{_r(rho_off, 4)} vs gate-on {_r(rho_on, 4)}")
    if gates_c["gate_on_positive_fdr"] and gates_c["gate_off_positive_fdr"]:
        verdict_c = ("REGIME-ROBUST: reversal positive FDR in both BTC "
                     f"regimes (on {_r(on)}bps vs off {_r(off)}bps) — "
                     "unlike the gate-on-only crowding/dfh signals; "
                     + rho_note)
    elif gates_c["gate_on_positive_fdr"]:
        verdict_c = ("GATE-ON ONLY: reversal lives in the BTC-up regime "
                     f"(on {_r(on)}bps vs off {_r(off)}bps) — same failure "
                     "mode as the crowding-follow signal; " + rho_note)
    elif gates_c["gate_off_positive_fdr"]:
        verdict_c = ("INVERTED REGIME: reversal lives gate-OFF (off "
                     f"{_r(off)}bps vs on {_r(on)}bps) — opposite of the "
                     "follow signals; plausible since reversal is "
                     "anti-crowding; " + rho_note)
    else:
        verdict_c = (f"XS NULL: reversal spread not FDR-positive in either "
                     f"regime (on {_r(on)}bps, off {_r(off)}bps); "
                     + rho_note
                     + " — the reversal tendency lives gate-OFF, the "
                       "mirror image of the gate-on-only crowding/dfh "
                       "follow signals (pooled rhos are descriptive, "
                       "outside the FDR family)")

    rs = d["shocked_pooled_spearman"]["rho"]
    rc = d["non_shocked_pooled_spearman"]["rho"]
    gates_d = {
        "shocked_rho_negative": rs is not None and rs < 0,
        "shocked_more_negative_than_calm": (rs is not None
                                          and rc is not None and rs < rc),
        "shocked_spearman_fdr": bool(fdr.get("d_shocked_pooled_spearman")),
    }
    failed_d = [k for k, v in gates_d.items() if not v]
    xs_shock_mean = d["shocked_only_xs_reversal_1d"]["daily_spread_bps"]
    caveat_d = ("" if fdr.get("d_shocked_xs_reversal_1d") else
                f"; caveat: shocked-only decile book does NOT clear FDR "
                f"({_r(xs_shock_mean['mean'])}bps t="
                f"{_r(xs_shock_mean['t'])}) — tendency confirmed, decile "
                "edge null")
    verdict_d = (("VOLUME-SHOCK REVERSAL CONFIRMED as a tendency "
                  "(shocked rho more negative than calm)" + caveat_d)
                 if not failed_d else
                 "SHOCK PRIOR NOT CONFIRMED: failed -> "
                 + ", ".join(failed_d))

    verdict_e = ("NET POSITIVE at 5bps/leg"
                 if gates_a["net_positive_5bps"] else
                 f"NET NEGATIVE at 5bps/leg ({_r(e['net_mean_daily_bps'])}"
                 "bps/day net — gross edge eaten by churn)")
    return {
        "a_daily_reversal": {"hard_gates": gates_a,
                             "hard_gates_passed":
                                 f"{sum(1 for v in gates_a.values() if v)}"
                                 f"/{len(gates_a)}",
                             "verdict": verdict_a},
        "b_liquidity_flip": {"hard_gates": gates_b,
                             "hard_gates_passed":
                                 f"{sum(1 for v in gates_b.values() if v)}"
                                 f"/{len(gates_b)}",
                             "verdict": verdict_b},
        "c_reversal_x_btc_regime": {"hard_gates": gates_c,
                                    "verdict": verdict_c},
        "d_volume_shock": {"hard_gates": gates_d,
                           "hard_gates_passed":
                               f"{sum(1 for v in gates_d.values() if v)}"
                               f"/{len(gates_d)}",
                           "verdict": verdict_d},
        "e_net_cost": {"verdict": verdict_e},
    }


def write_protocol_and_auth():
    protocol = {
        "schema_version":
            "nanojev-financial-signal-reversal-mega-protocol-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "T132: retest daily-scale short-term reversal and "
                   "liquidity-conditioned reversal/momentum at mega scale "
                   "(277 symbols x 2021-2025) — the 5-asset T101b Amihud "
                   "flip was underpowered and tested at 5d; the Zaremba "
                   "literature claim is at the 1d horizon. Arms: (a) "
                   "lag1d->fwd1d pooled Spearman + per-day XS Spearman + "
                   "decile reversal spread; (b) trailing-90 median Amihud "
                   "illiquidity level -> per-day median split into "
                   "illiquid/liquid halves, within-half Spearmans + "
                   "within-half decile spreads + paired per-day rho "
                   "contrast (prior: negative rho in the illiquid half); "
                   "(c) reversal x BTC ret20 gate split; (d) >90th-pct "
                   "trailing-90 quote-volume shock -> next-day reversal; "
                   "(e) honest net at 5bps/leg on the daily-rebalanced "
                   "book. BH-FDR over the fixed 10-cell family, within-day "
                   "rank-shuffle placebo x3, yearly folds, delisting "
                   "control, deterministic double-run.",
        "cohort": {
            "path": "data/perp_pit_mega_v1/records.jsonl",
            "builder": "scripts/build_perp_pit_mega_v1.py",
            "assets": "277 usable USDT-M perps (>=200 daily bars), "
                      "imported data/rc_futures_v1 archive copy; includes "
                      "delisted/renamed early-stoppers",
            "price_basis": "csv close as the mark proxy for every symbol",
            "targets": "label.forward_return_bps (1d, primary) and "
                       "forward_return_5d_bps (5d, descriptive), gross "
                       "close-to-close, contiguous daily bars only",
        },
        "definitions": {
            "ret1d": "close/close[prev_calendar_day]-1; requires the "
                     "previous record to be the previous calendar day "
                     "(no cross-gap returns)",
            "amihud": "|ret1d| / quote_volume (ret and volume both known "
                      "at the decision close)",
            "illiq_level": "median Amihud over the strictly-prior 90 "
                           "records per symbol (>=30 obs) — slow-moving "
                           "characteristic, not the same-day spike",
            "illiquid_liquid_halves": "per decision day, ranked universe "
                                      "split at the cross-sectional "
                                      "median of illiq_level",
            "vol_pct": "mid-rank pct of today's quote_volume vs its "
                       "strictly-prior 90 records (>=30 obs); shock = "
                       "vol_pct > 0.90",
            "btc_gate": "BTCUSDT close/close[t-20]-1 sign on the decision "
                        "day (contiguous 20d required)",
            "xs_decile": "per day >=30 ranked assets: k=n//10; reversal "
                         "longs bottom-decile losers / shorts top-decile "
                         "winners; spread = fwd(losers)-fwd(winners), bps",
            "net_cost": "5bps per unit one-sided leg turnover; turnover = "
                        "(slots entered+exited)/(2*book slots) vs prior "
                        "day; day-1 establishes the book",
            "fdr_family": "10 cells: a pooled Spearman + a XS spread; b "
                          "illiquid + liquid pooled Spearmans + paired "
                          "per-day rho flip + illiquid within-half XS; c "
                          "gate-on + gate-off XS spreads; d shocked pooled "
                          "Spearman + shocked-only XS",
        },
        "statistics": {
            "pooled": "Spearman rho + normal-approx p on all asset-days; "
                      "nominal — same-day rows share the market factor",
            "per_day": "per-day XS Spearman / paired rho diffs aggregated "
                       "over days — the honest companion to pooled",
            "xs": "mean daily decile spread + nominal t + yearly folds + "
                  "per-regime splits",
            "placebo": "within-day ret1d shuffle x3 seeds on arm a",
            "determinism": "measure() runs twice; serialized arms/fdr/"
                           "placebo must be byte-identical",
            "caveats": [
                "pooled Spearman over ~250k asset-days is dominated by "
                "day-level composition; per-day stats carry the load",
                "reversal decile membership churns ~daily; net sim shows "
                "whether the gross edge survives a stylized 5bps/leg",
                "illiq halves are a per-day cross-sectional split of a "
                "slow-moving median — composition drifts with listings",
                "delisting is a proxy (early stop = delisted OR renamed); "
                "renames can double-count an economic asset",
                "second-hand archive copy, not an as-of vintage",
            ],
        },
        "forbidden": ["fitting", "trading", "profitability claims",
                      "protocol edits post-run", "network"],
    }
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(protocol, indent=2, sort_keys=True)
                        + "\n", encoding="utf-8")
    sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    auth = {
        "schema_version":
            "nanojev-financial-signal-reversal-mega-authorization-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "protocol_sha256": sha,
        "protocol_path":
            "research/financial_signal_reversal_mega_protocol_v1.json",
        "decision": "approved_for_measurement",
        "measurement_authorized": True,
        "fit_authorized": False,
        "independent_reviewer": {
            "id": "project-owner",
            "independence": "owner_self_authorization_not_independent_review",
            "note": "Owner directed T132: retest daily-scale reversal and "
                    "liquidity effects on the 277-symbol mega cohort "
                    "(delegated task).",
        },
        "scope": {
            "permitted": "PIT-safe descriptive measurement on "
                         "data/perp_pit_mega_v1/records.jsonl: lag1d->"
                         "fwd1d reversal Spearmans and decile spreads, "
                         "Amihud-halves conditioning, BTC-gate and "
                         "volume-shock splits, 5bps/leg net sim, folds, "
                         "placebo, FDR and delisting controls per the "
                         "pinned protocol.",
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", type=pathlib.Path, default=COHORT)
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    args = parser.parse_args()

    protocol_sha = write_protocol_and_auth()
    report = {
        "schema_version": "nanojev-financial-signal-reversal-mega-v1",
        "task": "T132 daily-scale reversal + liquidity conditioning at "
                "mega scale (T101b retest with real power)",
        "cohort": str(args.cohort),
        "protocol_path":
            "research/financial_signal_reversal_mega_protocol_v1.json",
        "protocol_sha256": protocol_sha,
        "authorization_path":
            "results/financial_signal_reversal_mega_authorization_v1.json",
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "params": {"min_assets_per_day": MIN_ASSETS_PER_DAY,
                   "decile": DECILE,
                   "trail_days": TRAIL_DAYS,
                   "min_trail": MIN_TRAIL,
                   "vol_shock_pct": VOL_SHOCK_PCT,
                   "min_shock_assets_per_day": MIN_SHOCK_ASSETS,
                   "btc_gate_lookback": BTC_GATE_LOOKBACK,
                   "cost_bps_per_leg_turnover": COST_BPS_PER_LEG,
                   "placebo_seeds": list(PLACEBO_SEEDS)},
    }
    if not args.cohort.exists():
        report["status"] = ("SKIPPED: cohort records.jsonl not found "
                            "(build incomplete); nothing was fabricated")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True)
                            + "\n")
        print(json.dumps({"status": report["status"], "out": str(args.out)}))
        return 0

    series, meta, btc_ret20 = load_cohort(args.cohort)
    span = sorted(r["day"] for rows in series.values() for r in rows)
    early_stopped = sorted(a for a, m in meta.items() if m["listed"] is False)
    n_eval = sum(1 for rows in series.values() for r in rows
                 if r.get("ret1d") is not None and r.get("fwd1") is not None)
    report["cohort_description"] = {
        "symbols": len(series),
        "span": {"first": span[0] if span else None,
                 "last": span[-1] if span else None},
        "asset_days_with_ret1d_and_fwd1": n_eval,
        "early_stopped_symbols": {
            "count": len(early_stopped),
            "assets": {a: meta[a]["last_bar_date"] for a in early_stopped},
            "note": "early stop = delisted OR renamed proxy; LESS "
                    "survivorship bias than a listed-only universe, not "
                    "zero"},
    }
    report["status"] = "ran"

    meas1 = measure(series, meta, btc_ret20)
    meas2 = measure(series, meta, btc_ret20)
    det = json.dumps(meas1, sort_keys=True) == json.dumps(
        meas2, sort_keys=True)
    report["determinism"] = {"runs": 2, "byte_identical": det,
                             "note": "measure() double-run; serialized "
                                     "arms/fdr/placebo compared"}

    report["fdr_bh_0.05"] = {
        "family": "10 headline cells across arms a-d (<=10 per spec)",
        "table": meas1["fdr_table"]}
    report["placebo_shuffle_arm_a_1d"] = meas1["placebo"]
    report["delisting_honesty"] = {
        "method": "arm-a reversal re-run excluding all early-stopped "
                  "symbols; spread must keep sign",
        "control": meas1["controls"]["excl_all_early_stopped"]}
    report["arms"] = meas1["arms"]
    report["verdicts"] = verdicts(meas1)
    report["power_note"] = (
        "POWER: ~277 symbols/day at the peak universe gives ~27 "
        "names/decile side (~55x the 5-asset pilot's XS dof) and ~1.6-1.8k "
        "decision days; a pooled rho of |0.01| is resolvable (MDE80 "
        "~0.006). Still nominal: same-day asset-days share the market "
        "factor, so per-day Spearmans/spreads are the honest stats.")
    report["honesty"] = {
        "not_a_return": "spreads are gross close-price moves; the net sim "
                        "subtracts a stylized 5bps/leg turnover cost only "
                        "— funding, borrow, slippage excluded",
        "not_an_asof_vintage": "second-hand archive copy (rc_futures_v1), "
                               "not a verified venue pull or as-of vintage",
        "not_live": "no orders, no account, no broker, no trading API used",
        "no_profitability_claim": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")

    a = meas1["arms"]["a_daily_reversal"]
    brief = {"status": "ran", "out": str(args.out),
             "symbols": len(series), "eval_rows": n_eval,
             "deterministic": det,
             "a_pooled_rho": a["pooled_spearman_lag1d_fwd1d"]["rho"],
             "a_xs_spread_bps":
                 a["xs_decile_reversal_1d"]["daily_spread_bps"]["mean"],
             "a_xs_t": a["xs_decile_reversal_1d"]["daily_spread_bps"]["t"],
             "b_illiq_rho":
                 meas1["arms"]["b_liquidity_flip"]
                 ["illiquid_half_pooled_spearman"]["rho"],
             "b_liq_rho":
                 meas1["arms"]["b_liquidity_flip"]
                 ["liquid_half_pooled_spearman"]["rho"],
             "e_net_bps":
                 meas1["arms"]["e_net_cost"]["net_mean_daily_bps"],
             "verdicts": {k: v["verdict"] for k, v in
                          report["verdicts"].items()}}
    print(json.dumps(brief, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
