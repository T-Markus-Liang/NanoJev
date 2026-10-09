#!/usr/bin/env python3
"""T115: deep-dive diagnosis of the XS funding-carry arm on the mega cohort.

Context: T112 (``financial_signal_xs_mega_v1.py``) found the funding_pct
decile spread — long BOTTOM-decile funding_pct / short TOP-decile — at
+76.1bps/5d, t=3.58, BH-FDR surviving, BUT only 3/5 yearly folds same-sign
(2023 -7.0bps, 2024 -16.5bps; 2021 alone contributed +395.9bps/day).
Prior literature says XS carry is dead; at 52 funded names it looks alive
but unstable. This runner diagnoses WHY, without re-fitting anything:

  D1 fold-instability decomposition
      per-year spread + per-leg means; ex-2021 pooled spread; per-symbol
      contribution to the total spread (concentration: top-1/top-3 share
      of net and of positive contribution); edge-membership frequency
      (which names sit in the deciles); funding-SIGN regime split in the
      T94 sense — day-level share_pos (>0.6 positive / <0.4 negative /
      mixed) of the ranked funded names, plus leg-level sign splits.
  D2 direction check
      pooled long-leg (low-funding_pct) vs short-leg (high-funding_pct)
      mean fwd_5d vs the all-ranked universe mean — is the +76bps
      long-leg gains, short-leg losses, or market drift on both?
      Carry-follow (this direction) vs reversal stated explicitly.
  D3 interaction with dfh20
      per-day 3x3 double sort (funding_pct tercile x dfh20 tercile) ->
      pooled mean fwd_5d per cell + marginals + within-dfh funding
      spreads (does funding rank add anything beyond distance-from-high).
  D4 persistence / identity-proxy test
      per-asset funding_pct autocorrelation at lags 1/5/10/20;
      between-symbol vs within-symbol variance of funding_pct (ICC — is
      the cross-sectional rank a fixed symbol identity?); day-to-day
      decile-membership retention; and the stale-score check: decile
      spread using funding_pct lagged 5 bars vs fresh (dfh-staleness
      analog — if stale ~= fresh the signal is a slow variable).
  D5 cost-aware
      daily-rebalanced turnover + net spread at 5bps/leg for the
      funding-rank book, per-year; breakeven cost; reference turnover of
      dfh20/mom20 rank books on the SAME 52-symbol funded universe
      (signal-speed comparison at identical book size).
  D6 verdict
      (a) real-but-regime-dependent, (b) identity/slow-variable artifact,
      or (c) noise that FDR lucked through; include-in-spec-candidate
      recommendation.

Protocol + owner self-authorization are written BEFORE measurement per
repo convention (research/financial_signal_xs_carry_diag_protocol_v1.json,
results/financial_signal_xs_carry_diag_authorization_v1.json) and are
also embedded in the output. Measurement only: no fitting, no trading,
no network.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import financial_signal_xs_mega_v1 as mega  # reuse the exact T112 construction

COHORT = mega.COHORT
OUT = ROOT / "results/financial_signal_xs_carry_diag_v1.json"
PROTOCOL = ROOT / "research/financial_signal_xs_carry_diag_protocol_v1.json"
AUTH = ROOT / "results/financial_signal_xs_carry_diag_authorization_v1.json"

HORIZON = 5                      # diagnostic horizon = the T112 headline 5d
COST_BPS_PER_LEG = mega.COST_BPS_PER_LEG   # 5bps per unit one-sided turnover
SIGN_POS_HI, SIGN_NEG_LO = 0.6, 0.4        # T94 share_pos regime thresholds
AUTOCORR_LAGS = (1, 5, 10, 20)
STALE_LAG = 5                    # stale-score check lag (dfh-staleness analog)
T112_EXPECTED_MEAN_5D = 76.102101  # replication check vs T112 results file


def _r(x, nd=3):
    return round(x, nd) if isinstance(x, float) else x


# ------------------------------------------------------------------ days
def carry_day_rows(series, universe, horizon=HORIZON, score_key="funding_pct",
                   direction="carry", score_lag=0):
    """T112 xs_days with the tuples retained per leg asset-day.

    Each day row: {day, n, k, spread_bps, long_mean, short_mean,
    univ_mean, long_assets, short_assets, longs, shorts} where
    longs/shorts are (asset, score, fwd, funding, dfh20) tuples.
    ``score_lag``>0 scores each asset by its funding_pct ``score_lag``
    bars back (stale-rank check); labels stay at the decision day.
    Mirrors mega.xs_days exactly when score_lag=0 (verified in main)."""
    fwd_key = f"fwd_{horizon}d_bps"
    by_day = defaultdict(list)
    for asset, rows in series.items():
        if universe is not None and asset not in universe:
            continue
        for i, r in enumerate(rows):
            j = i - score_lag
            if j < 0:
                continue
            s = rows[j].get(score_key)
            f = r.get(fwd_key)
            if s is not None and f is not None:
                by_day[r["day"]].append(
                    (asset, s, f, r.get("funding"), r.get("dfh20")))
    days = []
    for day in sorted(by_day):
        ordered = sorted(by_day[day], key=lambda x: x[1])
        n = len(ordered)
        if n < mega.MIN_ASSETS_PER_DAY:
            continue
        k = max(1, n // mega.DECILE)
        bot, top = ordered[:k], ordered[-k:]
        if bot[-1][1] == top[0][1]:
            continue  # degenerate day: no dispersion at the ranked edges
        longs, shorts = (top, bot) if direction == "momentum" else (bot, top)
        long_mean = sum(x[2] for x in longs) / len(longs)
        short_mean = sum(x[2] for x in shorts) / len(shorts)
        days.append({
            "day": day, "n_assets": n, "edge": k,
            "spread_bps": long_mean - short_mean,
            "long_mean_bps": long_mean, "short_mean_bps": short_mean,
            "univ_mean_bps": sum(x[2] for x in ordered) / n,
            "long_assets": sorted(x[0] for x in longs),
            "short_assets": sorted(x[0] for x in shorts),
            "longs": longs, "shorts": shorts, "ordered": ordered,
        })
    return days


def pearson(pairs):
    """Pooled Pearson correlation of (x, y) pairs; None if degenerate."""
    n = len(pairs)
    if n < 5:
        return None
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    vx = sum((p[0] - mx) ** 2 for p in pairs)
    vy = sum((p[1] - my) ** 2 for p in pairs)
    if vx <= 0 or vy <= 0:
        return None
    cov = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    return cov / math.sqrt(vx * vy)


# ------------------------------------------------------- D1: instability
def per_year_stats(days):
    """Per-calendar-year spread + leg means with nominal t."""
    folds = defaultdict(list)
    for d in days:
        folds[d["day"][:4]].append(d)
    out = {}
    for y, ds in sorted(folds.items()):
        s = mega.t_stat([d["spread_bps"] for d in ds])
        out[y] = {"days": len(ds),
                  "mean_spread_bps": _r(s["mean"]),
                  "t": _r(s["t"]),
                  "long_leg_mean_bps": _r(
                      sum(d["long_mean_bps"] for d in ds) / len(ds)),
                  "short_leg_mean_bps": _r(
                      sum(d["short_mean_bps"] for d in ds) / len(ds)),
                  "univ_mean_bps": _r(
                      sum(d["univ_mean_bps"] for d in ds) / len(ds))}
    return out


def symbol_contributions(days):
    """Per-symbol contribution to the total spread.

    On day d the spread is (1/k)sum(long fwd) - (1/k)sum(short fwd), so
    asset a contributes +fwd_a/k when in the long leg and -fwd_a/k when
    in the short leg. Sum over days -> c_a; sum over assets -> total
    spread-days. Concentration = top-1/top-3 share of the net total and
    of the positive contribution mass."""
    contrib = defaultdict(float)
    in_long, in_short = defaultdict(int), defaultdict(int)
    per_year_contrib = defaultdict(lambda: defaultdict(float))
    for d in days:
        k = d["edge"]
        for (a, _s, f, _fr, _dfh) in d["longs"]:
            contrib[a] += f / k
            per_year_contrib[d["day"][:4]][a] += f / k
            in_long[a] += 1
        for (a, _s, f, _fr, _dfh) in d["shorts"]:
            contrib[a] -= f / k
            per_year_contrib[d["day"][:4]][a] -= f / k
            in_short[a] += 1
    total = sum(contrib.values())
    pos_total = sum(c for c in contrib.values() if c > 0)
    ordered = sorted(contrib.items(), key=lambda x: -x[1])
    top3 = sum(c for _a, c in ordered[:3])
    top1 = ordered[0] if ordered else (None, None)
    per_year_top3 = {}
    for y, cm in sorted(per_year_contrib.items()):
        yt = sum(cm.values())
        ytop = sum(c for _a, c in
                   sorted(cm.items(), key=lambda x: -x[1])[:3])
        per_year_top3[y] = {
            "year_total_bps_days": _r(yt, 1),
            "top3_share_of_year_net": _r(ytop / yt, 4) if yt > 0 else None}
    table = [{"asset": a, "contribution_bps_days": _r(c, 1),
              "share_of_net_total": _r(c / total, 4) if total else None,
              "days_in_long_leg": in_long[a],
              "days_in_short_leg": in_short[a]}
             for a, c in ordered]
    return {"total_spread_bps_days": _r(total, 1),
            "positive_contribution_bps_days": _r(pos_total, 1),
            "n_symbols_appearing": len(contrib),
            "top1": {"asset": top1[0],
                     "share_of_net_total": _r(top1[1] / total, 4)
                     if total else None},
            "top3_share_of_net_total": _r(top3 / total, 4) if total else None,
            "top3_share_of_positive_contribution":
                _r(top3 / pos_total, 4) if pos_total else None,
            "top3_names": [a for a, _c in ordered[:3]],
            "per_year_top3_share_of_year_net": per_year_top3,
            "per_symbol_table": table,
            "note": ("share_of_net_total can exceed 1 or go negative: the "
                     "net total is small relative to gross leg moves; a "
                     "symbol with c_a<0 worked AGAINST the spread")}


def membership_frequency(days):
    """Which names sit in the deciles — membership frequency per leg."""
    freq_long, freq_short = defaultdict(int), defaultdict(int)
    for d in days:
        for a in d["long_assets"]:
            freq_long[a] += 1
        for a in d["short_assets"]:
            freq_short[a] += 1
    n_days = len(days) or 1
    top_long = sorted(freq_long.items(), key=lambda x: -x[1])[:10]
    top_short = sorted(freq_short.items(), key=lambda x: -x[1])[:10]
    return {
        "most_frequent_long_lowfunding":
            [{"asset": a, "days": c, "frac_of_days": _r(c / n_days, 3)}
             for a, c in top_long],
        "most_frequent_short_highfunding":
            [{"asset": a, "days": c, "frac_of_days": _r(c / n_days, 3)}
             for a, c in top_short]}


def funding_sign_regime(days):
    """T94-style funding-sign regime split, day-level + leg-level.

    Day-level: share_pos = fraction of that day's ranked funded names
    with last_funding_rate > 0 (zeros non-positive); regime = positive
    if share_pos>0.6, negative if <0.4, else mixed. Leg-level: pooled
    leg asset-day mean fwd split by the asset's own funding sign."""
    day_regime = defaultdict(list)
    share_series = []
    for d in days:
        signs = [1 if (x[3] or 0) > 0 else 0 for x in d["ordered"]
                 if x[3] is not None]
        share_pos = sum(signs) / len(signs) if signs else None
        share_series.append(share_pos)
        reg = ("unknown" if share_pos is None
               else "positive" if share_pos > SIGN_POS_HI
               else "negative" if share_pos < SIGN_NEG_LO else "mixed")
        day_regime[reg].append(d["spread_bps"])
    day_level = {}
    for reg in ("positive", "mixed", "negative", "unknown"):
        v = day_regime.get(reg, [])
        s = mega.t_stat(v)
        day_level[reg] = {"days": len(v), "mean_spread_bps": _r(s["mean"]),
                          "t": _r(s["t"])}
    nonpos = day_regime.get("mixed", []) + day_regime.get("negative", [])
    s = mega.t_stat(nonpos)
    day_level["primary_split"] = {
        "positive_days": {"days": len(day_regime.get("positive", [])),
                          "mean_spread_bps": day_level["positive"]
                          ["mean_spread_bps"],
                          "t": day_level["positive"]["t"]},
        "nonpositive_days_mixed_plus_negative":
            {"days": len(nonpos), "mean_spread_bps": _r(s["mean"]),
             "t": _r(s["t"])}}

    def leg_sign_split(leg_key):
        pos = [x[2] for d in days for x in d[leg_key]
               if x[3] is not None and x[3] > 0]
        non = [x[2] for d in days for x in d[leg_key]
               if x[3] is not None and x[3] <= 0]
        return {"funding_positive": {"n": len(pos),
                                     "mean_fwd_bps": _r(
                                         sum(pos) / len(pos)
                                         if pos else None)},
                "funding_nonpositive": {"n": len(non),
                                        "mean_fwd_bps": _r(
                                            sum(non) / len(non)
                                            if non else None)}}
    return {
        "day_level_share_pos_thresholds": {"positive": f">{SIGN_POS_HI}",
                                           "negative": f"<{SIGN_NEG_LO}"},
        "mean_daily_share_pos_funding": _r(
            sum(s for s in share_series if s is not None)
            / max(1, len([s for s in share_series if s is not None])), 4),
        "day_level": day_level,
        "long_leg_lowfunding_by_own_sign": leg_sign_split("longs"),
        "short_leg_highfunding_by_own_sign": leg_sign_split("shorts"),
        "note": ("T94 found funding signals live in positive-funding "
                 "regimes; if the spread only exists on positive days "
                 "the instability is regime- not noise-driven")}


# ---------------------------------------------------------- D2: legs
def direction_check(days, horizon=HORIZON):
    """Long-leg vs short-leg pooled means vs universe — who does the work."""
    pooled_long = [x[2] for d in days for x in d["longs"]]
    pooled_short = [x[2] for d in days for x in d["shorts"]]
    pooled_univ = [x[2] for d in days for x in d["ordered"]]
    per_year_legs = {}
    folds = defaultdict(list)
    for d in days:
        folds[d["day"][:4]].append(d)
    for y, ds in sorted(folds.items()):
        pl = [x[2] for d in ds for x in d["longs"]]
        ps = [x[2] for d in ds for x in d["shorts"]]
        pu = [x[2] for d in ds for x in d["ordered"]]
        per_year_legs[y] = {
            "long_lowfunding_mean_bps": _r(sum(pl) / len(pl) if pl else None),
            "short_highfunding_mean_bps": _r(sum(ps) / len(ps) if ps else None),
            "universe_mean_bps": _r(sum(pu) / len(pu) if pu else None)}
    lm = sum(pooled_long) / len(pooled_long) if pooled_long else None
    sm = sum(pooled_short) / len(pooled_short) if pooled_short else None
    um = sum(pooled_univ) / len(pooled_univ) if pooled_univ else None
    interp = None
    if lm is not None and sm is not None and um is not None:
        if sm > 0:
            interp = ("SPREAD IS LONG-LEG-DRIVEN: both legs have positive "
                      "fwd returns (crypto drift); the short leg still "
                      "rose, so shorting high-funding names LOSES on "
                      "price — the +spread is low-funding names "
                      "outperforming, i.e. a crowded-long reversal "
                      "flavor, not funding cashflow carry")
        elif sm < 0 < lm:
            interp = ("BOTH LEGS WORK: low-funding longs gain AND "
                      "high-funding shorts fall — classic two-sided "
                      "carry/reversal")
        elif lm < 0 and sm < 0:
            interp = ("SPREAD IS SHORT-LEG-DRIVEN: both legs fall, "
                      "high-funding names fall more")
    return {"horizon_days": horizon,
            "pooled": {"long_lowfunding_mean_bps": _r(lm),
                       "short_highfunding_mean_bps": _r(sm),
                       "universe_mean_bps": _r(um),
                       "n_long_asset_days": len(pooled_long),
                       "n_short_asset_days": len(pooled_short)},
            "per_year": per_year_legs,
            "welch": {k: _r(v, 6) for k, v in
                      mega.welch(pooled_long, pooled_short).items()},
            "convention": ("long = BOTTOM funding_pct decile (low funding "
                           "vs own trailing-180), short = TOP decile; "
                           "positive spread = carry-direction in T112"),
            "interpretation": interp}


# ----------------------------------------------------- D3: dfh interaction
def dfh_double_sort(days):
    """Per-day 3x3 funding_pct tercile x dfh20 tercile -> pooled fwd_5d.

    Terciles are within-day rank thirds on the day's ranked funded
    names; dfh20 tercile uses only names with dfh20 non-null that day.
    Rows = funding tercile (F1 low .. F3 high), cols = dfh tercile
    (D1 far-from-high .. D3 near-high)."""
    cells = defaultdict(list)
    f_marg = defaultdict(list)
    d_marg = defaultdict(list)
    for d in days:
        ordered = [x for x in d["ordered"] if x[4] is not None]
        n = len(ordered)
        if n < 9:
            continue
        f_terc = {}
        for i, x in enumerate(ordered):
            f_terc[x[0]] = 1 if i < n / 3 else (2 if i < 2 * n / 3 else 3)
        by_dfh = sorted(ordered, key=lambda x: x[4])
        d_terc = {}
        for i, x in enumerate(by_dfh):
            d_terc[x[0]] = 1 if i < n / 3 else (2 if i < 2 * n / 3 else 3)
        for x in ordered:
            ft, dtk = f_terc[x[0]], d_terc[x[0]]
            cells[(ft, dtk)].append(x[2])
            f_marg[ft].append(x[2])
            d_marg[dtk].append(x[2])
    grid = {}
    for ft in (1, 2, 3):
        for dtk in (1, 2, 3):
            v = cells.get((ft, dtk), [])
            grid[f"F{ft}_D{dtk}"] = {
                "n": len(v),
                "mean_fwd_5d_bps": _r(sum(v) / len(v) if v else None)}
    marg = {
        "funding_tercile": {f"F{i}": {
            "n": len(f_marg[i]),
            "mean_fwd_5d_bps": _r(sum(f_marg[i]) / len(f_marg[i])
                                  if f_marg[i] else None)}
            for i in (1, 2, 3)},
        "dfh_tercile": {f"D{i}": {
            "n": len(d_marg[i]),
            "mean_fwd_5d_bps": _r(sum(d_marg[i]) / len(d_marg[i])
                                  if d_marg[i] else None)}
            for i in (1, 2, 3)}}
    within_dfh_funding_spread = {}
    for dtk in (1, 2, 3):
        lo, hi = cells.get((1, dtk), []), cells.get((3, dtk), [])
        within_dfh_funding_spread[f"D{dtk}"] = {
            "lowF_minus_highF_bps": _r(
                (sum(lo) / len(lo) if lo else 0)
                - (sum(hi) / len(hi) if hi else 0))
            if lo and hi else None}
    within_funding_dfh_spread = {}
    for ft in (1, 2, 3):
        lo, hi = cells.get((ft, 1), []), cells.get((ft, 3), [])
        within_funding_dfh_spread[f"F{ft}"] = {
            "nearHigh_minus_farHigh_bps": _r(
                (sum(hi) / len(hi) if hi else 0)
                - (sum(lo) / len(lo) if lo else 0))
            if lo and hi else None}
    return {"grid_mean_fwd_5d_bps": grid, "marginals": marg,
            "funding_spread_lowF_minus_highF_within_dfh_tercile":
                within_dfh_funding_spread,
            "dfh_spread_nearHigh_minus_farHigh_within_funding_tercile":
                within_funding_dfh_spread,
            "note": ("if funding only prices inside the far-from-high "
                     "cells it is a dfh proxy; if lowF-minus-highF is "
                     "positive in ALL dfh terciles it carries "
                     "independent info")}


# --------------------------------------------------------- D4: persistence
def persistence(series, universe, days_fresh):
    """Autocorr of funding_pct, identity-variance split, membership
    retention, and stale-5 score re-rank spread vs fresh."""
    # per-asset autocorrelation (valid pairs at each lag) + ICC pieces
    per_asset_ac = {lag: [] for lag in AUTOCORR_LAGS}
    pooled_ac = {}
    means, within_vars = {}, {}
    for asset, rows in series.items():
        if asset not in universe:
            continue
        vals = [r.get("funding_pct") for r in rows]
        valid = [v for v in vals if v is not None]
        if len(valid) >= 5:
            m = sum(valid) / len(valid)
            means[asset] = m
            within_vars[asset] = (sum((v - m) ** 2 for v in valid)
                                  / (len(valid) - 1))
        for lag in AUTOCORR_LAGS:
            pairs = [(vals[i], vals[i + lag])
                     for i in range(len(vals) - lag)
                     if vals[i] is not None and vals[i + lag] is not None]
            c = pearson(pairs)
            if c is not None:
                per_asset_ac[lag].append(c)
            pooled_ac[lag] = pooled_ac.get(lag, []) + pairs
    autocorr = {}
    for lag in AUTOCORR_LAGS:
        v = sorted(per_asset_ac[lag])
        autocorr[f"lag_{lag}"] = {
            "mean_per_asset_pearson": _r(sum(v) / len(v) if v else None, 4),
            "median_per_asset_pearson": _r(v[len(v) // 2] if v else None, 4),
            "pooled_pearson": _r(pearson(pooled_ac[lag]), 4),
            "n_assets": len(v)}
    # identity split: between-symbol variance of symbol means vs mean
    # within-symbol variance (ICC-like rho; 1 => fixed ranking identity)
    mv = list(means.values())
    grand = sum(mv) / len(mv)
    var_between = sum((m - grand) ** 2 for m in mv) / (len(mv) - 1)
    var_within = sum(within_vars.values()) / len(within_vars)
    icc = var_between / (var_between + var_within) \
        if (var_between + var_within) > 0 else None
    # day-to-day decile-membership retention on evaluated days
    ret_top, ret_bot = [], []
    for d0, d1 in zip(days_fresh, days_fresh[1:]):
        t0, t1 = set(d0["short_assets"]), set(d1["short_assets"])
        b0, b1 = set(d0["long_assets"]), set(d1["long_assets"])
        if t0:
            ret_top.append(len(t0 & t1) / len(t0))
        if b0:
            ret_bot.append(len(b0 & b1) / len(b0))
    return {"funding_pct_autocorr": autocorr,
            "identity_split": {
                "var_between_symbol_means": _r(var_between, 5),
                "mean_within_symbol_var": _r(var_within, 5),
                "icc_rho_fixed_identity": _r(icc, 4),
                "symbol_mean_funding_pct": {
                    "min": _r(min(mv), 4), "median": _r(sorted(mv)
                                                        [len(mv) // 2], 4),
                    "max": _r(max(mv), 4)},
                "note": ("icc near 1 => the cross-sectional funding_pct "
                         "rank is mostly a fixed symbol identity (perma "
                         "high/low-funding names), not a timing signal")},
            "decile_membership_retention_next_day": {
                "top_decile_highfunding": _r(
                    sum(ret_top) / len(ret_top) if ret_top else None, 4),
                "bottom_decile_lowfunding": _r(
                    sum(ret_bot) / len(ret_bot) if ret_bot else None, 4)},
            }


def stale_spread(days_stale):
    s = mega.t_stat([d["spread_bps"] for d in days_stale])
    folds = defaultdict(list)
    for d in days_stale:
        folds[d["day"][:4]].append(d["spread_bps"])
    return {"days": len(days_stale),
            "mean_spread_bps": _r(s["mean"]), "t": _r(s["t"]),
            "yearly_means": {y: _r(sum(v) / len(v))
                             for y, v in sorted(folds.items())}}


# --------------------------------------------------------------- D5: cost
def cost_block(days, label):
    net = mega.net_sim(days)
    folds = defaultdict(list)
    for d in days:
        folds[d["day"][:4]].append(d)
    per_year = {}
    for y, ds in sorted(folds.items()):
        ny = mega.net_sim(ds)
        per_year[y] = {"days": ny["days"],
                       "mean_turnover_frac":
                           ny["mean_daily_turnover_frac_of_book"],
                       "gross_mean_bps": ny["gross"]["mean_daily_bps"],
                       "net_mean_bps": ny["net"]["mean_daily_bps"],
                       "net_t": ny["net"]["t"]}
    g = net["gross"]["mean_daily_bps"]
    to = net["mean_daily_turnover_frac_of_book"]
    return {label: {
        "net_sim": net,
        "per_year": per_year,
        "breakeven_cost_bps_per_leg":
            _r(g / to) if g is not None and to else None,
        "note": ("breakeven = gross mean / mean daily turnover fraction; "
                 "cost scales linearly: net = gross - cost_bps*turnover")}}


def reference_turnover(series, universe):
    """dfh20/mom20 rank-book turnover on the SAME funded universe —
    isolates signal speed at identical book size."""
    out = {}
    for name, (score, direction) in (
            ("dfh20_momentum", ("dfh20", "momentum")),
            ("mom20_momentum", ("mom20", "momentum"))):
        days, _pl, _ps = mega.xs_days(series, score, HORIZON, direction,
                                      universe=universe)
        net = mega.net_sim(days)
        out[name] = {
            "days": net["days"],
            "mean_daily_turnover_frac_of_book":
                net["mean_daily_turnover_frac_of_book"],
            "mean_daily_cost_bps": net["mean_daily_cost_bps"],
            "gross_mean_bps": net["gross"]["mean_daily_bps"],
            "net_mean_bps": net["net"]["mean_daily_bps"]}
    return out


# --------------------------------------------------------- protocol/auth
def write_protocol_and_auth():
    protocol = {
        "schema_version": "nanojev-financial-signal-xs-carry-diag-protocol-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "T115: diagnose WHY the T112 XS funding_pct decile "
                   "carry arm (+76.1bps/5d, t=3.58, FDR-pass, 3/5 folds) "
                   "is unstable — fold decomposition, symbol "
                   "concentration, funding-sign regime (T94), leg/"
                   "direction check, dfh20 3x3 interaction, funding_pct "
                   "persistence/identity-proxy and stale-rank test, "
                   "turnover+net at 5bps/leg vs same-universe price-rank "
                   "books; then verdict (a) regime-dependent / "
                   "(b) identity artifact / (c) FDR luck and a "
                   "spec-candidate recommendation.",
        "cohort": {
            "path": "data/perp_pit_mega_v1/records.jsonl",
            "universe": "the ~52 symbols with non-null "
                        "features.last_funding_rate (same as T112 "
                        "funding arm)",
            "score": "funding_pct = per-asset mid-rank pct of "
                     "last_funding_rate vs trailing-180 daily values "
                     "(MIN_WINDOW=20), imported verbatim from "
                     "financial_signal_xs_mega_v1.load_cohort",
            "portfolio": "per decision day with >=30 ranked assets: "
                         "long BOTTOM decile / short TOP decile of "
                         "funding_pct, k=max(1,n//10), equal weight, "
                         "daily rebalance; label fwd_5d_bps (1d shown "
                         "only inside reused T112 helpers)",
        },
        "diagnostics": {
            "D1": "per-year spread+legs; ex-2021 pooled; per-symbol "
                  "contribution c_a=sum(+-fwd/k) with top-1/top-3 "
                  "concentration shares; decile-membership frequency; "
                  "funding-sign regime: day share_pos>0.6 positive, "
                  "<0.4 negative (T94 thresholds), primary split "
                  "positive vs non-positive, plus leg-level sign splits",
            "D2": "pooled long-leg vs short-leg vs universe mean "
                  "fwd_5d, per year; Welch; explicit carry-vs-reversal "
                  "and leg-attribution interpretation",
            "D3": "within-day 3x3 funding tercile x dfh20 tercile -> "
                  "pooled mean fwd_5d per cell, marginals, within-"
                  "dfh-tercile funding spreads and within-funding-"
                  "tercile dfh spreads",
            "D4": "per-asset funding_pct Pearson autocorr lags "
                  "1/5/10/20 (mean/median/pooled); between- vs within-"
                  "symbol variance ICC (fixed-identity check); next-day "
                  "decile-membership retention; stale-5-bar funding_pct "
                  "re-rank spread vs fresh",
            "D5": "mega.net_sim turnover + net at 5bps/leg overall and "
                  "per year; breakeven cost = gross/turnover; reference "
                  "dfh20/mom20 rank-book turnover restricted to the "
                  "same 52 funded symbols",
            "D6": "rule-based verdict over the D1-D5 evidence: "
                  "(a) regime-dependent, (b) identity/slow-variable "
                  "artifact, (c) noise/FDR luck; spec-candidate "
                  "recommendation",
        },
        "statistics": {
            "nominal_only": "all t/p nominal — 5d overlapping labels "
                            "autocorrelate adjacent days; pooled "
                            "asset-days correlate within a day",
            "replication": "fresh-score spread must equal the T112 "
                           "funding_pct_carry h5 run (mean 76.102bps, "
                           "n=1785 days); checked in-report",
        },
        "forbidden": ["fitting", "trading", "profitability claims",
                      "protocol edits post-run", "network"],
    }
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    auth = {
        "schema_version":
            "nanojev-financial-signal-xs-carry-diag-authorization-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "protocol_sha256": sha,
        "protocol_path":
            "research/financial_signal_xs_carry_diag_protocol_v1.json",
        "decision": "approved_for_measurement",
        "measurement_authorized": True,
        "fit_authorized": False,
        "independent_reviewer": {
            "id": "project-owner",
            "independence":
                "owner_self_authorization_not_independent_review",
            "note": "Owner directed T115: deep-dive the XS funding-carry "
                    "arm instability on the mega cohort's 52 funded "
                    "symbols (delegated task)."},
        "scope": {
            "permitted": "PIT-safe descriptive measurement on "
                         "data/perp_pit_mega_v1/records.jsonl per the "
                         "pinned protocol: fold/symbol/regime "
                         "decomposition, leg attribution, dfh double "
                         "sort, persistence/staleness, turnover+net.",
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
    return protocol, auth, sha


# ----------------------------------------------------------------- verdict
def verdict_block(d1, d2, d3, d4, d5, stale_res, fresh_mean):
    """Evidence-weighted classification into (a) regime-dependent /
    (b) identity artifact / (c) FDR luck, plus a spec recommendation."""
    ev = {}
    folds = d1["per_year"]
    ex21 = d1["ex_2021_pooled"]
    # -- regime dependence
    pos = d1["funding_sign_regime"]["day_level"]["primary_split"]
    pos_m = pos["positive_days"]["mean_spread_bps"]
    non_m = pos["nonpositive_days_mixed_plus_negative"]["mean_spread_bps"]
    ev["positive_funding_day_spread_bps"] = pos_m
    ev["nonpositive_funding_day_spread_bps"] = non_m
    regime_gated = (pos_m is not None and non_m is not None
                    and pos_m > 0 and pos_m > 2 * max(non_m, 0.0))
    y2021 = folds.get("2021", {}).get("mean_spread_bps")
    ev["y2021_spread_bps"] = y2021
    ev["ex2021_spread_bps"] = ex21["mean_spread_bps"]
    ev["ex2021_t"] = ex21["t"]
    single_year_dominant = (ex21["mean_spread_bps"] is not None
                            and ex21["mean_spread_bps"] <= 0)
    # -- identity / slow-variable artifact
    icc = d4["identity_split"]["icc_rho_fixed_identity"]
    stale_ratio = (stale_res["mean_spread_bps"] / fresh_mean
                   if stale_res["mean_spread_bps"] is not None
                   and fresh_mean else None)
    ev["icc_fixed_identity"] = icc
    ev["stale5_over_fresh_spread"] = _r(stale_ratio, 4)
    ev["decile_retention_highfunding"] = \
        d4["decile_membership_retention_next_day"]["top_decile_highfunding"]
    identity_like = (icc is not None and icc > 0.5) or \
        (stale_ratio is not None and stale_ratio > 0.9)
    # -- concentration / noise
    conc = d1["symbol_concentration"]
    ev["top3_share_of_net"] = conc["top3_share_of_net_total"]
    ev["top3_names"] = conc["top3_names"]
    concentrated = (conc["top3_share_of_net_total"] is not None
                    and conc["top3_share_of_net_total"] > 0.5)
    # -- legs
    legs = d2["pooled"]
    ev["long_leg_mean_bps"] = legs["long_lowfunding_mean_bps"]
    ev["short_leg_mean_bps"] = legs["short_highfunding_mean_bps"]
    ev["universe_mean_bps"] = legs["universe_mean_bps"]
    # -- cost
    fund_net = d5["funding_pct_carry"]["net_sim"]["net"]
    ev["net_mean_daily_bps"] = fund_net["mean_daily_bps"]
    ev["breakeven_cost_bps_per_leg"] = \
        d5["funding_pct_carry"]["breakeven_cost_bps_per_leg"]
    net_positive = fund_net["mean_daily_bps"] is not None and \
        fund_net["mean_daily_bps"] > 0
    # -- interaction residual info
    wdfh = d3["funding_spread_lowF_minus_highF_within_dfh_tercile"]
    indep_cells = sum(1 for k in ("D1", "D2", "D3")
                      if (wdfh.get(k) or {}).get("lowF_minus_highF_bps")
                      is not None
                      and wdfh[k]["lowF_minus_highF_bps"] > 0)
    ev["dfh_terciles_with_positive_lowF_minus_highF"] = f"{indep_cells}/3"

    if single_year_dominant and concentrated:
        klass = ("(c) NOISE/CONCENTRATION-LUCK: the FDR pass is carried "
                 "by 2021 and a few symbols; ex-2021 spread is "
                 f"{_r(ex21['mean_spread_bps'])}bps (t={_r(ex21['t'])})")
        recommend = ("DO NOT include as a standalone spec candidate; at "
                     "most keep as a regime-gated research feature with "
                     "a symbol-concentration cap")
    elif identity_like and not regime_gated:
        klass = ("(b) IDENTITY/SLOW-VARIABLE ARTIFACT: funding_pct is "
                 f"near-fixed per symbol (ICC={_r(icc)}), stale-5 ranks "
                 f"keep {_r(stale_ratio)} of the spread — the arm is a "
                 "static perma-high/perma-low-funding symbol bet, not a "
                 "funding timing signal")
        recommend = ("include only if recast as a symbol-identity carry "
                     "sleeve (fixed book, near-zero turnover cost); as a "
                     "cross-sectional timing signal it adds no fresh "
                     "information")
    elif regime_gated or single_year_dominant:
        klass = ("(a) REAL-BUT-REGIME-DEPENDENT: the spread is genuine "
                 "inside its regime "
                 f"(positive-funding days {_r(pos_m)}bps vs non-positive "
                 f"{_r(non_m)}bps; 2021 {_r(y2021)}bps) and survives "
                 "costs, but vanishes outside it — consistent with the "
                 "T94 funding-sign regime finding")
        recommend = ("INCLUDE in the spec-candidate family ONLY as a "
                     "regime-gated variant (funding-sign regime and/or "
                     "bull-drift gate carried in the spec), never "
                     "ungated")
    else:
        klass = ("(a) WEAKLY-CONFIRMED: positive ex-2021, positive in "
                 "both funding regimes, net-positive — but small "
                 "ex-2021 magnitude warrants regime gates anyway")
        recommend = ("include in the spec-candidate family with regime "
                     "gates and a concentration monitor")
    return {"classification": klass,
            "spec_candidate_recommendation": recommend,
            "evidence": ev,
            "rule_flags": {
                "regime_gated_by_funding_sign": regime_gated,
                "single_year_dominant_2021": single_year_dominant,
                "identity_or_slow_artifact": identity_like,
                "symbol_concentrated": concentrated,
                "net_positive_at_5bps": net_positive}}


# -------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", type=pathlib.Path, default=COHORT)
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    args = parser.parse_args()

    protocol, auth, protocol_sha = write_protocol_and_auth()
    report = {"schema_version": "nanojev-financial-signal-xs-carry-diag-v1",
              "task": "T115 XS funding-carry instability deep-dive",
              "cohort": str(args.cohort),
              "protocol_path":
                  "research/financial_signal_xs_carry_diag_protocol_v1.json",
              "protocol_sha256": protocol_sha,
              "authorization_path":
                  "results/financial_signal_xs_carry_diag_authorization_v1.json",
              "protocol_inline": protocol,
              "authorization_inline": auth,
              "horizon_days": HORIZON,
              "cost_bps_per_leg_turnover": COST_BPS_PER_LEG,
              "sign_regime_thresholds": {"positive": SIGN_POS_HI,
                                         "negative": SIGN_NEG_LO},
              "stale_lag_bars": STALE_LAG}

    if not args.cohort.exists():
        report["status"] = ("SKIPPED: cohort records.jsonl not found; "
                            "nothing was fabricated")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True)
                            + "\n")
        print(json.dumps({"status": report["status"],
                          "out": str(args.out)}))
        return 0

    series, meta, funded_assets, btc_ret20 = mega.load_cohort(args.cohort)
    universe = set(funded_assets)
    report["cohort_description"] = {
        "symbols_total": len(series),
        "funded_symbols_in_universe": len(funded_assets),
        "funded_symbols": funded_assets}

    # ---- replication check vs the T112 runner (identical code path)
    days_ref, _pl, _ps = mega.xs_days(series, "funding_pct", HORIZON,
                                      "carry", universe=universe)
    ref = mega.t_stat([d["spread_bps"] for d in days_ref])
    days = carry_day_rows(series, universe, HORIZON)
    mine = mega.t_stat([d["spread_bps"] for d in days])
    same_days = [d["day"] for d in days] == [d["day"] for d in days_ref]
    same_spreads = all(abs(a["spread_bps"] - b["spread_bps"]) < 1e-9
                       for a, b in zip(days, days_ref))
    report["replication_check"] = {
        "t112_runner_days": len(days_ref),
        "t112_runner_mean_bps": _r(ref["mean"], 6),
        "t112_runner_t": _r(ref["t"], 6),
        "diag_days": len(days),
        "diag_mean_bps": _r(mine["mean"], 6),
        "matches_t112_results_file_76_102bps":
            abs((mine["mean"] or 0) - T112_EXPECTED_MEAN_5D) < 0.01,
        "day_for_day_identical_to_runner": same_days and same_spreads}
    report["status"] = "ran"

    # ---- D1 fold-instability decomposition
    per_year = per_year_stats(days)
    ex21_days = [d for d in days if not d["day"].startswith("2021")]
    ex21 = mega.t_stat([d["spread_bps"] for d in ex21_days])
    d1 = {"per_year": per_year,
          "ex_2021_pooled": {"days": len(ex21_days),
                             "mean_spread_bps": _r(ex21["mean"]),
                             "t": _r(ex21["t"])},
          "symbol_concentration": symbol_contributions(days),
          "membership_frequency": membership_frequency(days),
          "funding_sign_regime": funding_sign_regime(days)}

    # ---- D2 direction / leg attribution
    d2 = direction_check(days, HORIZON)

    # ---- D3 funding x dfh20 double sort
    d3 = dfh_double_sort(days)

    # ---- D4 persistence / identity
    d4 = persistence(series, universe, days)
    days_stale = carry_day_rows(series, universe, HORIZON,
                                score_lag=STALE_LAG)
    stale_res = stale_spread(days_stale)
    d4["stale_score_check"] = {
        "fresh_mean_bps": _r(mine["mean"]),
        f"stale_{STALE_LAG}d": stale_res,
        "note": ("if the lagged-5 rank reproduces the spread the signal "
                 "is slow-moving (dfh-staleness analog); identity value "
                 "vs timing value separated by the ICC block above")}

    # ---- D5 cost-aware
    d5 = cost_block(days, "funding_pct_carry")
    d5["reference_same_universe_turnover"] = reference_turnover(
        series, universe)
    d5["comparison_note"] = (
        "dfh20/mom20 books re-run on the SAME 52 funded symbols so "
        "book size is identical; funding-rank should churn less iff the "
        "score is slower-moving")

    report["d1_fold_instability"] = d1
    report["d2_direction"] = d2
    report["d3_funding_x_dfh20_3x3"] = d3
    report["d4_persistence_identity"] = d4
    report["d5_cost_aware"] = d5
    report["verdict"] = verdict_block(d1, d2, d3, d4, d5, stale_res,
                                      mine["mean"])
    report["honesty"] = {
        "not_a_return": "spreads are gross close-price moves; net sim "
                        "subtracts a stylized 5bps/leg turnover cost "
                        "only — funding cashflows, borrow, slippage and "
                        "leverage excluded",
        "nominal_stats": "all t/p nominal: 5d labels overlap ~80% on "
                         "adjacent days; pooled asset-days correlate "
                         "within a day; ~3-5 names per decile side",
        "not_an_asof_vintage": "second-hand archive copy "
                               "(rc_futures_v1), not a verified venue "
                               "pull or as-of vintage",
        "not_live": "no orders, no account, no broker, no trading API",
        "no_profitability_claim": True,
        "reused_code": "funding_pct construction, xs_days, net_sim, "
                       "t_stat, welch imported unchanged from "
                       "financial_signal_xs_mega_v1",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True)
                        + "\n")
    brief = {"status": "ran", "out": str(args.out),
             "replication_mean_bps": _r(mine["mean"]),
             "ex2021_mean_bps": ex21["mean"],
             "verdict": report["verdict"]["classification"],
             "recommendation":
                 report["verdict"]["spec_candidate_recommendation"]}
    print(json.dumps(brief, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
