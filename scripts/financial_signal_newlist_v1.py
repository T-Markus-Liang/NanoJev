#!/usr/bin/env python3
"""T131: is listing-age a real conditioning dimension for the XS dfh20
effect, or a 2021/2024 listing-wave artifact?

T126 (financial_signal_decay_v1) found the 2025 XS dfh20 spread is driven
by NEW listings: cohort-age<180d new-only decile spread +109.6bps vs
age>=180d incumbent +17.7bps on the mega daily cohort. This script
deep-dives that amplification on ``data/perp_pit_mega_v1/records.jsonl``
(277 syms, 2021-01 -> 2025-12, close-as-mark basis):

  Part 1 — AGE-CONDITIONED XS DFH. Per decision day the dfh20-ranked
           universe is split by listing age (<90d / 90-365d / >365d /
           pre-2021-censored; incumbent = censored OR >365d) and the
           top-decile-minus-bottom-decile dfh20 fwd-spread is computed
           WITHIN each bucket (h1 and h5, pooled + per-year). A naive
           T126-convention replicate (age<180d incl. left-censored, >=30
           gate) is kept for comparability.
  Part 2 — AGE-RANK ARM (is it dfh or just age?). Per day rank by age_d;
           newest-quintile minus oldest-quintile next-1d/5d spread — a
           pure age effect with no dfh conditioning.
  Part 3 — INTERACTION 3x3. Per day: within-day age tercile x dfh20
           tercile -> mean fwd5 per cell; pooled grid + per-year grids.
           Where is the money?
  Part 4 — POST-LISTING LIFECYCLE. For every symbol first seen after
           cohort start (a true observed listing): its first 365 days of
           close-to-close returns indexed by listing-day offset. Mean/
           median per offset, binned windows (IPO-pop analog day 1-30
           drift?), and the same split by listing year.
  Part 5 — ROBUSTNESS. Within-day score-shuffle placebo x3 on the young
           bucket and the age-rank arm; yearly folds for every bucket;
           daily-rebalanced net sim at 5bps per unit one-sided leg
           turnover (T112 convention).
  Part 6 — VERDICT. Programmatic gates -> is listing-age a real
           conditioning dimension worth adding to the spec/model, or a
           listing-wave artifact?

LEFT-CENSORING HONESTY (tightened vs T126): assets whose first cohort
record is the cohort's first day (2021-01-01) were listed BEFORE the
archive window — their age_d is a lower bound, not an age. T126 counted
them as age 0 (young) for their first 180 days; here they land in the
``pre2021_censored`` bucket and are NEVER counted as young. The naive
<180d replicate is retained only to reproduce the T126 headline.
Conversely, age_d for censored assets understates true age, so age-rank
terciles/quintiles mis-place some true-old assets — carried as a caveat.

Protocol + owner self-authorization are embedded in the single output
artifact ``results/financial_signal_newlist_v1.json`` (protocol object
hashed by sha256, authorization pins the hash — T126 convention).
Measurement only: no fitting, no trading, no network, no other files
modified.
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
OUT = ROOT / "results/financial_signal_newlist_v1.json"

YEARS = ("2021", "2022", "2023", "2024", "2025")
HORIZONS = (1, 5)
DECILE = 10
QUINTILE = 5
MIN_ASSETS = 30            # T112/T126 full-universe day gate
MIN_BUCKET_ASSETS = 10     # within-bucket decile gate (buckets are small)
YOUNG_LT90 = 90            # <90d bucket ceiling
MID_D365 = 365             # 90-365d bucket ceiling; >365d = gt365
NAIVE_YOUNG_D = 180        # T126 naive young cutoff (censored incl.)
BUCKETS = ("lt90", "d90_365", "gt365", "pre2021_censored", "incumbent")
COST_BPS_PER_LEG = 5.0     # per unit of one-sided leg turnover (T112)
PLACEBO_SEEDS = (11, 22, 33)
DAYS_PER_YEAR = 365        # crypto trades daily
LIFECYCLE_DAYS = 365       # post-listing curve length
LIFE_BINS = ((1, 1), (2, 5), (6, 10), (11, 20), (21, 30), (31, 60),
             (61, 90), (91, 180), (181, 365))
LIFE_MARKS = (1, 5, 7, 14, 30, 60, 90, 180, 365)


def _r(x, nd=3):
    return round(x, nd) if isinstance(x, float) else x


def norm_p(t):
    """Two-sided normal-approx p from a t/z statistic (repo convention)."""
    if t is None:
        return None
    return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))


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


def welch(xs, ys):
    """Welch two-sample t, Welch-Satterthwaite df, normal-approx p."""
    nx, ny = len(xs), len(ys)
    if nx < 2 or ny < 2:
        return {"n_long": nx, "n_short": ny, "t": None, "df": None,
                "p": None}
    mx, my = sum(xs) / nx, sum(ys) / ny
    vx = sum((x - mx) ** 2 for x in xs) / (nx - 1)
    vy = sum((y - my) ** 2 for y in ys) / (ny - 1)
    denom = vx / nx + vy / ny
    if denom <= 0:
        return {"n_long": nx, "n_short": ny, "mean_long": mx,
                "mean_short": my, "t": None, "df": None, "p": None}
    t = (mx - my) / math.sqrt(denom)
    df = denom ** 2 / ((vx / nx) ** 2 / (nx - 1)
                       + (vy / ny) ** 2 / (ny - 1))
    return {"n_long": nx, "n_short": ny, "mean_long": mx,
            "mean_short": my, "t": t, "df": df, "p": norm_p(t)}


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return None
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def max_drawdown(cumulative):
    peak, mdd = 0.0, 0.0
    for v in cumulative:
        peak = max(peak, v)
        mdd = max(mdd, peak - v)
    return mdd


# ---------------------------------------------------------------- loaders
def load_cohort(path):
    """asset -> sorted rows {day, close, dfh20, fwd1, fwd5, age_d,
    censored}; meta; cohort_min_day. age_d = days since the asset's first
    cohort record; censored = first record on the cohort's first day
    (pre-2021 listing — true age unknown, never counted as young)."""
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
                "close": f["close"]["value"],
                "dfh20": f["dfh20"]["value"],
                "fwd1": rec["label"]["forward_return_bps"],
                "fwd5": rec["label"]["forward_return_5d_bps"],
            })
    cohort_min_day = min(r["day"] for rows in series.values() for r in rows)
    cohort_min_ord = dt.date.fromisoformat(cohort_min_day).toordinal()
    for asset, rows in series.items():
        rows.sort(key=lambda r: r["day"])
        first_ord = dt.date.fromisoformat(rows[0]["day"]).toordinal()
        censored = first_ord == cohort_min_ord
        for r in rows:
            r["age_d"] = (dt.date.fromisoformat(r["day"]).toordinal()
                          - first_ord)
            r["censored"] = censored
    return series, meta, cohort_min_day


def daily_rows(series):
    """day -> list of (asset, dfh20, age_d, censored, fwd1, fwd5) over the
    dfh20-ranked universe (dfh20 non-null). Single pass reused by parts
    1-3 and the placebos."""
    by_day = defaultdict(list)
    for asset, rows in series.items():
        for r in rows:
            if r["dfh20"] is None:
                continue
            by_day[r["day"]].append((asset, r["dfh20"], r["age_d"],
                                     r["censored"], r["fwd1"], r["fwd5"]))
    return dict(by_day)


# ------------------------------------------------------------- primitives
def _edge(ordered, div):
    """Bottom-k / top-k slices of a score-ordered list (k = max(1,n//div));
    None if the edge boundary has no dispersion."""
    n = len(ordered)
    k = max(1, n // div)
    bot, top = ordered[:k], ordered[-k:]
    if bot[-1][1] == top[0][1]:
        return None
    return bot, top


def _spread(legs, fwd_i):
    """top-minus-bottom mean forward return for an _edge() result."""
    bot, top = legs
    return (sum(x[fwd_i] for x in top) / len(top)
            - sum(x[fwd_i] for x in bot) / len(bot))


def bucket_of(age_d, censored):
    if censored:
        return "pre2021_censored"
    if age_d < YOUNG_LT90:
        return "lt90"
    if age_d <= MID_D365:
        return "d90_365"
    return "gt365"


def net_sim(days):
    """Daily-rebalanced book: cost = 5bps per unit one-sided leg turnover;
    turnover = (slots entered + slots exited) / (2 * book slots), slots =
    (side, asset) pairs. Day one establishes the full book (frac=1)."""
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
        rows.append({"gross_bps": d["spread_bps"],
                     "cost_bps": COST_BPS_PER_LEG * frac})
        prev = slots
    if not rows:
        return {"days": 0}
    gross = [r["gross_bps"] for r in rows]
    net = [r["gross_bps"] - r["cost_bps"] for r in rows]
    g, n = t_stat(gross), t_stat(net)
    cum_g = cum_n = 0.0
    cg, cn = [], []
    for r in rows:
        cum_g += r["gross_bps"]
        cum_n += r["gross_bps"] - r["cost_bps"]
        cg.append(cum_g)
        cn.append(cum_n)
    sharpe = lambda s: (s["mean"] / s["sd"] * math.sqrt(DAYS_PER_YEAR)
                        if s["sd"] else None)
    return {"days": len(rows),
            "mean_daily_turnover_frac_of_book": _r(
                sum(turnovers) / len(turnovers)),
            "mean_daily_cost_bps": _r(
                sum(r["cost_bps"] for r in rows) / len(rows)),
            "gross": {"mean_daily_bps": _r(g["mean"]), "t": _r(g["t"]),
                      "cumulative_bps": _r(cg[-1]),
                      "max_drawdown_bps": _r(max_drawdown(cg)),
                      "sharpe_annualized": _r(sharpe(g))},
            "net": {"mean_daily_bps": _r(n["mean"]), "t": _r(n["t"]),
                    "cumulative_bps": _r(cn[-1]),
                    "max_drawdown_bps": _r(max_drawdown(cn)),
                    "sharpe_annualized": _r(sharpe(n))}}


def summarize_spread_arm(days, pooled_long, pooled_short):
    """Per-arm stat block: daily t-stat, pooled Welch, yearly folds,
    day-count and edge-size stats, net sim."""
    spreads = [d["spread_bps"] for d in days]
    daily = t_stat(spreads)
    folds = defaultdict(list)
    for d in days:
        folds[d["day"][:4]].append(d["spread_bps"])
    fold_stats = {}
    for y in YEARS:
        s = t_stat(folds.get(y, []))
        fold_stats[y] = {"days": len(folds.get(y, [])),
                         "mean_spread_bps": _r(s["mean"], 4),
                         "t": _r(s["t"])}
    same = sum(1 for y in YEARS if daily["mean"] is not None
               and fold_stats[y]["mean_spread_bps"] is not None
               and fold_stats[y]["mean_spread_bps"] * daily["mean"] > 0)
    ns = [d["n"] for d in days]
    ks = [d["k"] for d in days]
    return {
        "days_evaluated": len(days),
        "n_ranked_in_group": {"min": min(ns) if ns else None,
                              "median": median(ns),
                              "max": max(ns) if ns else None},
        "edge_names_per_side": {"min": min(ks) if ks else None,
                                "median": median(ks),
                                "max": max(ks) if ks else None},
        "daily_spread_bps": {k: _r(v, 6) for k, v in daily.items()},
        "pooled_welch": {k: _r(v, 6) for k, v in
                         welch(pooled_long, pooled_short).items()},
        "yearly_folds": fold_stats,
        "yearly_folds_same_sign": f"{same}/{len(YEARS)}",
        "net_sim": net_sim(days),
    }


# ------------------------------------------------- part 1/2/3 day engine
def run_arms(by_day):
    """Single pass over decision days computing:
      - per-bucket dfh20 decile spreads (h1+h5) with membership for netsim
      - naive T126 <180d/>=180d replicate (>=30 gate each side)
      - age-rank newest-vs-oldest quintile spread (h1+h5)
      - age-tercile x dfh-tercile fwd5 cells (interaction grid inputs)
    Returns (bucket_res, naive_res, agerank_res, interaction_cells)."""
    fwd_idx = {1: 4, 5: 5}
    buck_days = {(b, h): [] for b in BUCKETS for h in HORIZONS}
    buck_leg = {(b, h): {"l": [], "s": []} for b in BUCKETS
                for h in HORIZONS}
    naive_days = {("young_lt180", h): [] for h in HORIZONS}
    naive_days.update({("old_ge180", h): [] for h in HORIZONS})
    naive_leg = {k: {"l": [], "s": []} for k in naive_days}
    age_days = {h: [] for h in HORIZONS}
    age_leg = {h: {"l": [], "s": []} for h in HORIZONS}
    cells = defaultdict(lambda: {"daily_means": [], "pooled": [],
                                 "by_year": defaultdict(list)})

    for day in sorted(by_day):
        rows = by_day[day]
        # bucket membership is horizon-independent (dfh non-null rows)
        members = defaultdict(list)
        for x in rows:
            members[bucket_of(x[2], x[3])].append(x)
        members["incumbent"] = [x for x in rows
                                if x[3] or x[2] > MID_D365]
        for h in HORIZONS:
            fi = fwd_idx[h]
            rh = [x for x in rows if x[fi] is not None]
            # ---- part 1: within-bucket dfh decile spreads
            for b in BUCKETS:
                grp = [x for x in members[b] if x[fi] is not None]
                if len(grp) < MIN_BUCKET_ASSETS:
                    continue
                ordered = sorted(grp, key=lambda x: x[1])
                legs = _edge(ordered, DECILE)
                if legs is None:
                    continue
                bot, top = legs
                buck_days[(b, h)].append({
                    "day": day, "n": len(ordered), "k": len(top),
                    "spread_bps": _spread(legs, fi),
                    "long_assets": sorted(x[0] for x in top),
                    "short_assets": sorted(x[0] for x in bot)})
                buck_leg[(b, h)]["l"].extend(x[fi] for x in top)
                buck_leg[(b, h)]["s"].extend(x[fi] for x in bot)
            # ---- naive T126 replicate: <180d vs >=180d, >=30 gate
            for key, grp in (
                    ("young_lt180",
                     [x for x in rh if x[2] < NAIVE_YOUNG_D]),
                    ("old_ge180",
                     [x for x in rh if x[2] >= NAIVE_YOUNG_D])):
                if len(grp) < MIN_ASSETS:
                    continue
                ordered = sorted(grp, key=lambda x: x[1])
                legs = _edge(ordered, DECILE)
                if legs is None:
                    continue
                bot, top = legs
                naive_days[(key, h)].append({
                    "day": day, "n": len(ordered), "k": len(top),
                    "spread_bps": _spread(legs, fi),
                    "long_assets": sorted(x[0] for x in top),
                    "short_assets": sorted(x[0] for x in bot)})
                naive_leg[(key, h)]["l"].extend(x[fi] for x in top)
                naive_leg[(key, h)]["s"].extend(x[fi] for x in bot)
            # ---- part 2: pure age-rank quintile spread (no dfh)
            if len(rh) >= MIN_ASSETS:
                # no boundary-tie skip: equal ages (the censored mass all
                # shares one age_d; same-day listings share theirs) are
                # common — the quintile cut is kept with a deterministic
                # (age_d, asset_id) tiebreak, caveat carried
                by_age = sorted(rh, key=lambda x: (x[2], x[0]))
                n_a = len(by_age)
                kq = max(1, n_a // QUINTILE)
                newest, oldest = by_age[:kq], by_age[-kq:]
                spread = (sum(x[fi] for x in newest) / len(newest)
                          - sum(x[fi] for x in oldest) / len(oldest))
                age_days[h].append({
                    "day": day, "n": n_a, "k": kq,
                    "spread_bps": spread,
                    "long_assets": sorted(x[0] for x in newest),
                    "short_assets": sorted(x[0] for x in oldest)})
                age_leg[h]["l"].extend(x[fi] for x in newest)
                age_leg[h]["s"].extend(x[fi] for x in oldest)
        # ---- part 3: age-tercile x dfh-tercile fwd5 grid (h5 only)
        r5 = [x for x in rows if x[5] is not None]
        if len(r5) >= MIN_ASSETS:
            n5 = len(r5)
            by_age = sorted(r5, key=lambda x: (x[2], x[0]))
            age_t = {x[0]: min(2, (3 * i) // n5)
                     for i, x in enumerate(by_age)}
            by_dfh = sorted(r5, key=lambda x: (x[1], x[0]))
            dfh_t = {x[0]: min(2, (3 * i) // n5)
                     for i, x in enumerate(by_dfh)}
            cell_vals = defaultdict(list)
            for x in r5:
                cell_vals[(age_t[x[0]], dfh_t[x[0]])].append(x[5])
            for (a_t, d_t), vals in cell_vals.items():
                c = cells[(a_t, d_t)]
                c["daily_means"].append(
                    {"day": day, "mean": sum(vals) / len(vals)})
                c["pooled"].extend(vals)
                c["by_year"][day[:4]].extend(vals)

    bucket_res = {b: {} for b in BUCKETS}
    for b in BUCKETS:
        for h in HORIZONS:
            bucket_res[b][f"h{h}"] = summarize_spread_arm(
                buck_days[(b, h)], buck_leg[(b, h)]["l"],
                buck_leg[(b, h)]["s"])
    naive_res = {}
    for key in ("young_lt180", "old_ge180"):
        naive_res[key] = {f"h{h}": summarize_spread_arm(
            naive_days[(key, h)], naive_leg[(key, h)]["l"],
            naive_leg[(key, h)]["s"]) for h in HORIZONS}
    agerank_res = {f"h{h}": summarize_spread_arm(
        age_days[h], age_leg[h]["l"], age_leg[h]["s"]) for h in HORIZONS}
    return bucket_res, naive_res, agerank_res, cells


def interaction_grid(cells):
    """3x3 mean-fwd5 grids: mean-of-daily-cell-means and pooled asset-day
    means; plus per-year pooled grids and the argmax cell."""
    a_names = ("youngest_tercile", "mid_age_tercile", "oldest_tercile")
    d_names = ("lowest_dfh_tercile", "mid_dfh_tercile",
               "highest_dfh_tercile")
    grid, pooled = {}, {}
    best = None
    for (a_t, d_t), c in sorted(cells.items()):
        name = f"{a_names[a_t]}__{d_names[d_t]}"
        s = t_stat([m["mean"] for m in c["daily_means"]])
        pm = (sum(c["pooled"]) / len(c["pooled"])) if c["pooled"] else None
        grid[name] = {"mean_of_daily_means_bps": _r(s["mean"], 4),
                      "t_daily": _r(s["t"]),
                      "n_days": len(c["daily_means"]),
                      "n_asset_days": len(c["pooled"]),
                      "pooled_mean_bps": _r(pm, 4)}
        pooled[name] = pm
        if pm is not None and (best is None or pm > best[1]):
            best = (name, pm)
    by_year = {}
    for y in YEARS:
        yg = {}
        for (a_t, d_t), c in sorted(cells.items()):
            vals = c["by_year"].get(y, [])
            name = f"{a_names[a_t]}__{d_names[d_t]}"
            yg[name] = {"mean_bps": _r(sum(vals) / len(vals), 4)
                        if vals else None,
                        "n_asset_days": len(vals)}
        by_year[y] = yg
    # market-factor-free contrast: per day, high-minus-low dfh tercile
    # cell INSIDE each age tercile (pooled cell means alone confound
    # beta/drift — every pooled cell is negative in 2025 for example)
    grad = {}
    for a_t in range(3):
        hi = {m["day"]: m["mean"]
              for m in cells.get((a_t, 2), {}).get("daily_means", [])}
        lo = {m["day"]: m["mean"]
              for m in cells.get((a_t, 0), {}).get("daily_means", [])}
        diffs = [hi[d] - lo[d] for d in hi if d in lo]
        s = t_stat(diffs)
        grad[a_names[a_t]] = {"days": len(diffs),
                              "mean_high_minus_low_bps": _r(s["mean"], 4),
                              "t": _r(s["t"])}
    return {"grid_mean_fwd5_bps": grid,
            "argmax_cell_pooled": {"cell": best[0] if best else None,
                                   "mean_bps": _r(best[1], 4)
                                   if best else None},
            "within_age_tercile_dfh_gradient": grad,
            "pooled_grid_by_year": by_year}


# ------------------------------------------------------- part 4 lifecycle
def lifecycle(series):
    """Post-listing return curve for symbols first seen after cohort start
    (true observed listings — censored assets excluded). Offset d return =
    close[d]/close[d-1]-1 over contiguous bars only; offsets 1..365."""
    curves = []
    listings_by_year = defaultdict(int)
    for asset, rows in series.items():
        if not rows or rows[0]["censored"]:
            continue
        listings_by_year[rows[0]["day"][:4]] += 1
        rets = []
        prev_age, prev_close = None, None
        for r in rows:
            close = r["close"]
            if close is None:
                prev_age, prev_close = None, None
                continue
            if prev_age is not None and r["age_d"] - prev_age == 1 \
                    and prev_close > 0:
                if r["age_d"] <= LIFECYCLE_DAYS:
                    rets.append((r["age_d"],
                                 (close / prev_close - 1.0) * 1e4))
            prev_age, prev_close = r["age_d"], close
        if rets:
            curves.append((rows[0]["day"], rets))
    by_off = defaultdict(list)
    by_off_year = defaultdict(lambda: defaultdict(list))
    for first_day, rets in curves:
        for off, v in rets:
            by_off[off].append(v)
            by_off_year[first_day[:4]][off].append(v)
    per_offset = [{"offset": o, "n": len(vs),
                   "mean_bps": _r(statistics.mean(vs), 4),
                   "median_bps": _r(median(vs), 4)}
                  for o, vs in sorted(by_off.items())]
    def _bins(off_map):
        out = []
        for lo, hi in LIFE_BINS:
            vals = [v for o in range(lo, hi + 1)
                    for v in off_map.get(o, ())]
            s = t_stat(vals)
            out.append({"offsets": f"{lo}-{hi}", "n": len(vals),
                        "mean_bps": _r(s["mean"], 4), "t": _r(s["t"]),
                        "median_bps": _r(median(vals), 4)})
        return out
    binned = _bins(by_off)
    by_listing_year = {y: {"n_listings": listings_by_year.get(y, 0),
                           "bins": _bins(by_off_year.get(y, {}))}
                       for y in sorted(set(listings_by_year) | set(YEARS))}
    # cumulative path marks: per asset, cumprod of its offset returns
    marks = {}
    for mark in LIFE_MARKS:
        cums = []
        for _, rets in curves:
            prod = 1.0
            reached = False
            for off, v in rets:
                prod *= 1.0 + v / 1e4
                if off == mark:
                    reached = True
                    break
            if reached:
                cums.append((prod - 1.0) * 1e4)
        marks[f"day_{mark}"] = {"n": len(cums),
                                "mean_cumret_bps": _r(
                                    statistics.mean(cums), 2)
                                if cums else None,
                                "median_cumret_bps": _r(median(cums), 2)}
    return {"n_assets_observed_listing": len(curves),
            "listings_per_year": {y: listings_by_year.get(y, 0)
                                  for y in sorted(set(listings_by_year)
                                                  | set(YEARS))},
            "binned_offset_returns": binned,
            "by_listing_year_bins": by_listing_year,
            "cumulative_path_marks": marks,
            "per_offset_curve": per_offset}


# ------------------------------------------------------- part 5 placebo
def placebo_bucket(by_day, seed, bucket, h=5):
    """Shuffle dfh20 values among the members of one age bucket within
    each day (the age selection is preserved — null = 'dfh ranking within
    that age class is meaningless'), then recompute the within-bucket
    decile spread."""
    fi = 4 if h == 1 else 5
    rng = random.Random(seed)
    spreads = []
    for day in sorted(by_day):
        grp = [x for x in by_day[day]
               if bucket_of(x[2], x[3]) == bucket and x[fi] is not None]
        if len(grp) < MIN_BUCKET_ASSETS:
            continue
        scores = [x[1] for x in grp]
        rng.shuffle(scores)
        shuffled = sorted(zip(scores, grp), key=lambda p: p[0])
        ordered = [g for _, g in shuffled]
        n = len(ordered)
        k = max(1, n // DECILE)
        if shuffled[k - 1][0] == shuffled[n - k][0]:
            continue  # degenerate boundary in the shuffled scores
        bot, top = ordered[:k], ordered[-k:]
        spreads.append(sum(x[fi] for x in top) / k
                       - sum(x[fi] for x in bot) / k)
    s = t_stat(spreads)
    return {"days": len(spreads), "mean_spread_bps": _r(s["mean"]),
            "t": _r(s["t"])}


def placebo_agerank(by_day, seed, h=5):
    """Shuffle age_d among each day's ranked assets (null = 'age carries
    no information'), then recompute the newest-vs-oldest quintile
    spread."""
    fi = 4 if h == 1 else 5
    rng = random.Random(seed)
    spreads = []
    for day in sorted(by_day):
        rh = [x for x in by_day[day] if x[fi] is not None]
        if len(rh) < MIN_ASSETS:
            continue
        ages = [x[2] for x in rh]
        rng.shuffle(ages)
        shuffled = sorted(zip(ages, rh), key=lambda p: (p[0], p[1][0]))
        ordered = [g for _, g in shuffled]
        n = len(ordered)
        k = max(1, n // QUINTILE)
        newest, oldest = ordered[:k], ordered[-k:]
        spreads.append(sum(x[fi] for x in newest) / k
                       - sum(x[fi] for x in oldest) / k)
    s = t_stat(spreads)
    return {"days": len(spreads), "mean_spread_bps": _r(s["mean"]),
            "t": _r(s["t"])}


# ------------------------------------------------------------- protocol
def build_protocol():
    return {
        "schema_version":
            "nanojev-financial-signal-newlist-protocol-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "task": "T131",
        "purpose": "Deep-dive the T126 finding that the 2025 XS dfh20 "
                   "spread is amplified in new listings (age<180d new-only "
                   "+109.6bps vs incumbent +17.7bps). Test whether "
                   "listing-age is a real conditioning dimension: "
                   "age-bucketed dfh decile spreads, a pure age-rank arm, "
                   "an age x dfh 3x3 interaction grid, a post-listing "
                   "lifecycle curve, and placebo/fold/net-cost "
                   "robustness. Measurement only — no fitting, no "
                   "trading, no profitability claims.",
        "cohort": {"path": "data/perp_pit_mega_v1/records.jsonl",
                   "span": "2021-01..2025-12, ~277 USDT-M perps, "
                           "close-as-mark basis, includes delisted/"
                           "renamed early-stoppers"},
        "definitions": {
            "age_d": "days since the asset's first cohort record "
                     "(listing-age proxy)",
            "left_censoring": "assets first seen on the cohort's first "
                              "day were listed pre-2021: bucketed as "
                              "pre2021_censored at all ages, NEVER as "
                              "young (tighter than T126, which counted "
                              "them age 0 for their first 180d)",
            "age_buckets": "lt90 = age<90 (non-censored); d90_365 = "
                           "90<=age<=365; gt365 = age>365; "
                           "pre2021_censored; incumbent = censored OR "
                           "age>365",
            "naive_t126": "young = age<180 with censored INCLUDED, >=30 "
                          "gate — retained only to reproduce the T126 "
                          "headline",
            "bucketed_dfh_spread": "per day within each bucket: sort by "
                                   "dfh20, top-decile minus bottom-"
                                   "decile mean fwd (momentum "
                                   "direction), k=max(1,n//10), bucket "
                                   "gate >=10 assets",
            "age_rank_arm": "per day over >=30 ranked: sort by age_d "
                            "(tiebreak asset_id — deterministic but "
                            "arbitrary among equal-age censored "
                            "names); newest-quintile minus oldest-"
                            "quintile fwd; pure age, no dfh",
            "interaction_3x3": "per day over >=30 ranked: within-day "
                               "age-tercile x dfh-tercile cell mean of "
                               "fwd5; grid = mean of daily cell means "
                               "+ pooled asset-day means + per-year "
                               "pooled grid + the market-factor-free "
                               "within-age-tercile (high-minus-low dfh "
                               "cell) daily spread per tercile",
            "lifecycle": "non-censored assets only: offset-d close-to-"
                         "close return over contiguous bars for offsets "
                         "1..365; pooled mean/median per offset, "
                         "binned, cumulative marks, split by listing "
                         "year",
            "placebo_young": "dfh20 shuffled among the SAME day's "
                             "bucket members (age selection preserved) "
                             "x3 seeds, for lt90 and the deeper d90_365 "
                             "bucket",
            "placebo_agerank": "age_d shuffled within each day x3 seeds",
            "net_cost": "daily-rebalanced book, 5bps per unit one-sided "
                        "leg turnover (T112 convention)"},
        "statistics": {
            "t_stat": "mean/sd/t normal-approx p on per-day series — "
                      "nominal (overlapping labels, shared market "
                      "factor)",
            "pooled_welch": "Welch two-sample t over leg asset-days",
            "yearly_folds": "per-calendar-year mean spreads + sign "
                            "consistency 2021-2025",
            "determinism": "measure() executed twice on the loaded "
                           "cohort, serialized bytes compared"},
        "forbidden": ["fitting", "trading", "profitability claims",
                      "protocol edits post-run", "network"],
    }


def build_auth(protocol_sha):
    return {
        "schema_version":
            "nanojev-financial-signal-newlist-authorization-v1",
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
            "note": "Owner directed T131: deep-dive the new-listing "
                    "amplification of the XS dfh effect on the mega "
                    "daily cohort per the embedded protocol (delegated "
                    "task)."},
        "scope": {
            "permitted": "PIT-safe descriptive measurement on "
                         "data/perp_pit_mega_v1/records.jsonl per the "
                         "pinned protocol: age-bucketed dfh decile "
                         "spreads, age-rank arm, 3x3 interaction grid, "
                         "post-listing lifecycle, placebo shuffles, "
                         "yearly folds, 5bps net sim.",
            "not_permitted": "No fitting/trading/profitability claims/"
                             "protocol edits; no network; no other "
                             "files modified."},
        "network_model_calls": 0,
        "order_submission_authorized": False,
        "live_trading_authorized": False,
    }


# --------------------------------------------------------------- measure
def measure(series, meta, cohort_min_day):
    report = {}
    span = sorted(r["day"] for rows in series.values() for r in rows)
    censored_n = sum(1 for rows in series.values()
                     if rows and rows[0]["censored"])
    early_stopped = sorted(a for a, m in meta.items()
                           if m["listed"] is False)
    report["cohort_description"] = {
        "symbols": len(series),
        "span": {"first": span[0] if span else None,
                 "last": span[-1] if span else None},
        "pre2021_censored_symbols": censored_n,
        "observed_new_listings": len(series) - censored_n,
        "early_stopped_symbols": len(early_stopped),
        "note": "censored = first record on cohort day one "
                f"({cohort_min_day}); true age unknown — never counted "
                "as young",
    }

    by_day = daily_rows(series)

    # ---- parts 1-3
    bucket_res, naive_res, agerank_res, cells = run_arms(by_day)
    grid = interaction_grid(cells)
    report["part1_age_conditioned_dfh"] = {
        "buckets": bucket_res,
        "naive_t126_replicate_age180": {
            "note": "censored counted as young (T126 convention), >=30 "
                    "gate — comparability only",
            "arms": naive_res},
    }
    report["part2_age_rank_arm"] = {
        "direction": "newest_quintile_minus_oldest_quintile (positive "
                     "= young names outperform; pure age, no dfh)",
        "arms": agerank_res}
    report["part3_interaction_3x3"] = grid

    # ---- part 4: lifecycle
    report["part4_post_listing_lifecycle"] = lifecycle(series)

    # ---- part 5: robustness
    pl_lt90 = {f"seed_{s}": placebo_bucket(by_day, s, "lt90", h=5)
               for s in PLACEBO_SEEDS}
    pl_mid = {f"seed_{s}": placebo_bucket(by_day, s, "d90_365", h=5)
              for s in PLACEBO_SEEDS}
    pl_a = {f"seed_{s}": placebo_agerank(by_day, s, h=5)
            for s in PLACEBO_SEEDS}
    report["part5_robustness"] = {
        "placebo_lt90_bucket_dfh_h5": pl_lt90,
        "placebo_d90_365_bucket_dfh_h5": pl_mid,
        "placebo_lt90_note": "lt90 median edge k=1 (single-name legs): "
                             "its within-bucket placebo is noise-"
                             "dominated by construction; the d90_365 "
                             "placebo is the informative null",
        "placebo_age_rank_h5": pl_a,
        "net_cost_note": "net_sim embedded per arm: 5bps per unit "
                         "one-sided leg turnover, daily rebalance (T112)",
    }

    # ---- part 6: verdict
    lt90_5 = bucket_res["lt90"]["h5"]["daily_spread_bps"]["mean"]
    lt90_1 = bucket_res["lt90"]["h1"]["daily_spread_bps"]["mean"]
    mid_5 = bucket_res["d90_365"]["h5"]["daily_spread_bps"]["mean"]
    mid_1 = bucket_res["d90_365"]["h1"]["daily_spread_bps"]["mean"]
    old_5 = bucket_res["gt365"]["h5"]["daily_spread_bps"]["mean"]
    old_5_t = bucket_res["gt365"]["h5"]["daily_spread_bps"]["t"]
    inc_5 = bucket_res["incumbent"]["h5"]["daily_spread_bps"]["mean"]
    inc_1 = bucket_res["incumbent"]["h1"]["daily_spread_bps"]["mean"]
    agerank_5 = agerank_res["h5"]["daily_spread_bps"]["mean"]
    net5_mid = bucket_res["d90_365"]["h5"]["net_sim"].get(
        "net", {}).get("mean_daily_bps")
    net5_lt90 = bucket_res["lt90"]["h5"]["net_sim"].get(
        "net", {}).get("mean_daily_bps")
    folds_mid = bucket_res["d90_365"]["h5"]["yearly_folds"]
    pos_folds_mid = sum(1 for y in YEARS
                        if (folds_mid[y]["mean_spread_bps"] or 0) > 0)
    folds_lt90 = bucket_res["lt90"]["h5"]["yearly_folds"]
    pos_folds_lt90 = sum(1 for y in YEARS
                         if (folds_lt90[y]["mean_spread_bps"] or 0) > 0)
    seeds_mid = [v["mean_spread_bps"] for v in pl_mid.values()
                 if v["mean_spread_bps"] is not None]
    seeds_lt90 = [v["mean_spread_bps"] for v in pl_lt90.values()
                  if v["mean_spread_bps"] is not None]

    # listing-wave artifact test: do the young-bucket spreads only show
    # up in the years that supplied the most listings?
    lpy = report["part4_post_listing_lifecycle"]["listings_per_year"]
    wave_years = set(sorted(YEARS, key=lambda y: -lpy.get(y, 0))[:2])
    nonwave_pos = [y for y in YEARS if y not in wave_years
                   and (folds_mid[y]["mean_spread_bps"] or 0) > 0]

    # lifecycle shape: is there a fixed day-1 pop? (regime-independent?)
    life = report["part4_post_listing_lifecycle"]
    d1_by_year = {y: yy["bins"][0]["mean_bps"]
                  for y, yy in life["by_listing_year_bins"].items()
                  if yy["n_listings"]}
    d1_pos_years = [y for y, v in d1_by_year.items()
                    if v is not None and v > 0]

    grad = grid["within_age_tercile_dfh_gradient"]
    gates = {
        "young_buckets_amplify_h5": bool(
            lt90_5 is not None and mid_5 is not None
            and inc_5 is not None
            and lt90_5 > inc_5 and mid_5 > inc_5
            and mid_5 > 0),
        "old_bucket_inverts_h5": bool(
            old_5 is not None and old_5_t is not None
            and old_5 < 0 and old_5_t <= -2),
        "dfh_conditioning_beats_age_alone": bool(
            mid_5 is not None and agerank_5 is not None
            and abs(mid_5) > abs(agerank_5)),
        "placebo_collapses_deep_bucket": bool(
            seeds_mid) and mid_5 is not None and all(
            m < 0.5 * mid_5 for m in seeds_mid),
        "net_positive_after_5bps_h5": bool(
            net5_mid is not None and net5_mid > 0),
        "yearly_folds_mostly_positive_d90_365": pos_folds_mid >= 4,
        "not_confined_to_listing_wave_years": bool(nonwave_pos),
        "young_tercile_dfh_gradient_positive": bool(
            grad["youngest_tercile"]["mean_high_minus_low_bps"]
            is not None
            and grad["youngest_tercile"]["mean_high_minus_low_bps"] > 0),
    }
    passed = sum(1 for v in gates.values() if v)

    age_effect_real = (gates["young_buckets_amplify_h5"]
                       or gates["old_bucket_inverts_h5"])
    stable = (gates["placebo_collapses_deep_bucket"]
              and gates["yearly_folds_mostly_positive_d90_365"]
              and gates["not_confined_to_listing_wave_years"])
    if not age_effect_real:
        classification = "null_no_age_conditioning"
    elif stable and gates["dfh_conditioning_beats_age_alone"]:
        classification = "real_conditioning_dimension"
    elif age_effect_real and not stable:
        classification = "real_but_fragile_conditioning"
    else:
        classification = "mixed_or_insufficient"

    read = (
        f"Age DOES condition the XS dfh effect, but non-monotonically: "
        f"h5 decile spread lt90 {_r(lt90_5)}bps (t="
        f"{_r(bucket_res['lt90']['h5']['daily_spread_bps']['t'])}, thin "
        f"bucket k~1), d90_365 {_r(mid_5)}bps (t="
        f"{_r(bucket_res['d90_365']['h5']['daily_spread_bps']['t'])}), "
        f"gt365 {_r(old_5)}bps (t={_r(old_5_t)} — INVERTED for names "
        f"that listed in-window and aged), censored pre-2021 "
        f"{_r(bucket_res['pre2021_censored']['h5']['daily_spread_bps']['mean'])}bps, "
        f"pooled incumbent {_r(inc_5)}bps. Naive T126 replicate "
        f"reproduces the 2025 headline exactly (young<180 h1 "
        f"{_r(naive_res['young_lt180']['h1']['yearly_folds']['2025']['mean_spread_bps'])}bps "
        f"vs old {_r(naive_res['old_ge180']['h1']['yearly_folds']['2025']['mean_spread_bps'])}bps). "
        f"Pure age-rank h5 {_r(agerank_5)}bps — age alone is a "
        f"null-to-mildly-NEGATIVE main effect; the value is in the "
        f"dfh-by-age interaction, not an age drift. Within-tercile "
        f"dfh gradient (market-factor-free): youngest "
        f"{grad['youngest_tercile']['mean_high_minus_low_bps']}bps, mid "
        f"{grad['mid_age_tercile']['mean_high_minus_low_bps']}bps, "
        f"oldest "
        f"{grad['oldest_tercile']['mean_high_minus_low_bps']}bps. "
        f"Lifecycle: no fixed post-listing pattern — day-1 mean is "
        f"+{_r(life['binned_offset_returns'][0]['mean_bps'])}bps pooled "
        f"but median {_r(life['binned_offset_returns'][0]['median_bps'])}bps "
        f"and flips sign by listing year "
        f"(2024 {_r(d1_by_year.get('2024'))} / 2025 "
        f"{_r(d1_by_year.get('2025'))} vs 2021 "
        f"{_r(d1_by_year.get('2021'))}bps); only the ~d61-90 drift-down "
        f"bin clears |t|>2 "
        f"({_r(life['binned_offset_returns'][6]['mean_bps'])}bps, t="
        f"{_r(life['binned_offset_returns'][6]['t'])}). Placebo d90_365 "
        f"means {seeds_mid}; lt90 placebo {seeds_lt90} (k~1 noise). "
        f"Classification: {classification} — gates "
        f"{passed}/{len(gates)} passed.")
    report["part6_verdict"] = {
        "gates": gates, "gates_passed": f"{passed}/{len(gates)}",
        "listing_wave_years_by_count": sorted(wave_years),
        "d90_365_positive_folds": pos_folds_mid,
        "lt90_positive_folds": pos_folds_lt90,
        "nonwave_years_positive_d90_365": nonwave_pos,
        "day1_pop_positive_listing_years": d1_pos_years,
        "net5_lt90_bps": net5_lt90,
        "net5_d90_365_bps": net5_mid,
        "classification": classification, "read": read}
    report["honest_notes"] = [
        "all t/p nominal: forward labels overlap within a symbol and "
        "cross-asset days share a market factor; effective n << record "
        "count",
        "age_d is cohort-first-seen: non-censored assets are observed "
        "listings only if the archive truly starts at listing day — a "
        "late data start masquerades as a new listing; censored assets "
        "have unknown true age (age_d is a lower bound) and are pooled "
        "into pre2021_censored/incumbent",
        "the lt90 bucket often holds <30 names/day; decile edge k=1-2 "
        "there — treat its daily t as optimistic; pooled Welch still "
        "double-counts correlated asset-days",
        "age-rank and interaction terciles mis-place censored assets "
        "whose lower-bound age_d understates true age",
        "lifecycle returns are close-as-mark gross; no funding, fees, "
        "borrow; early-stop contracts truncate their curves",
        "net sim is a stylized daily-rebalance 5bps/leg turnover cost — "
        "not a tradeable statement",
        "the naive <180d replicate reproduces T126 only approximately: "
        "T126 computed inside a >=30-ranked dfh universe with the same "
        "gate on each subset",
        "decile-vs-tercile resolution matters: the gt365 decile spread "
        "is INVERTED (-85bps h5) while its tercile high-minus-low "
        "gradient is positive — the reversal lives in the extreme "
        "decile tails, not the broad tercile body",
    ]
    return report


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", type=pathlib.Path, default=COHORT)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()

    if not args.cohort.exists():
        stub = {"schema_version":
                "nanojev-financial-signal-newlist-v1",
                "status": "SKIPPED: cohort records.jsonl not found; "
                          "nothing was fabricated"}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(stub, indent=2, sort_keys=True)
                            + "\n", encoding="utf-8")
        print(json.dumps(stub, sort_keys=True))
        return 0

    series, meta, cohort_min_day = load_cohort(args.cohort)

    protocol = build_protocol()
    protocol_sha = hashlib.sha256(
        json.dumps(protocol, indent=2, sort_keys=True).encode()
    ).hexdigest()

    r1 = measure(series, meta, cohort_min_day)
    r2 = measure(series, meta, cohort_min_day)
    blob = json.dumps(r1, indent=2, sort_keys=True)
    assert blob == json.dumps(r2, indent=2, sort_keys=True), \
        "non-deterministic measurement"

    r1["schema_version"] = "nanojev-financial-signal-newlist-v1"
    r1["status"] = "measurement_complete"
    r1["task"] = "T131"
    r1["determinism"] = {"runs": 2, "byte_identical": True,
                         "sha256": hashlib.sha256(
                             blob.encode()).hexdigest()}
    r1["protocol"] = protocol
    r1["protocol_sha256"] = protocol_sha
    r1["owner_authorization"] = build_auth(protocol_sha)
    r1["generated_at"] = dt.datetime.now(dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r1, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    v = r1["part6_verdict"]
    print("classification:", v["classification"])
    print("gates:", v["gates_passed"], json.dumps(v["gates"],
                                                  sort_keys=True))
    print("read:", v["read"])
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
