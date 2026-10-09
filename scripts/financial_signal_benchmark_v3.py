#!/usr/bin/env python3
"""T108: cross-scale unified financial-signal benchmark (v3).

Extends the frozen T82/T91 scorecard with the T101 dfh arm in the daily
block and a new cross-sectional (XS) block on the 10-asset cohort:

  daily      run() of scripts/financial_signal_benchmark_v1.py verbatim —
             the frozen 10 cells/fdr/verdicts must stay BYTE-IDENTICAL to
             v2 (verified at runtime against the v2 receipt).
             NEW daily["dfh_cells"]: dfh20_pct — mid-rank pct of dfh20
             (mark[i]/max(mark[i-20:i])-1, strictly-prior 20 bars) vs its
             trailing-180 window, per-asset (T101 convention). Metrics:
             Spearman rho/t/p + top-vs-bottom DECILE Welch on gross 5d
             bps + top-decile arm net stats + fold signs + label-shuffle
             placebos. Reported OUTSIDE the frozen 10-cell FDR family as
             its own 1-cell family — adding it to the frozen family would
             change the surviving thresholds of the existing cells.

  intraday   run_intraday() of scripts/financial_signal_benchmark_v2.py
             verbatim — cohort data/perp_pit_intraday_v1/records.jsonl is
             pinned by sha and verified to be the expanded 17,960-record
             (Apr-Aug) cohort, not the original 2-month slice.
             NOTE carried forward: mom_1h and mom_4h both PASS with
             NEGATIVE rho — short-horizon reversal, not continuation.

  xs         NEW. cohort=data/perp_pit_xs_v1/records.jsonl (sha pinned;
             10 USDT-M perps, 2023-01-01..2026-08-30, close-as-mark).
             label = label.forward_return_5d_bps (gross close-to-close).
             Per decision day with >=8 scored assets, rank the cross-
             section and take top-2 vs bottom-2 mean fwd 5d spread:
               dfh20       long top-2 (momentum: near-high -> continue)
               mom20       long top-2 (momentum continuation)
               funding_pct long bottom-2 (carry: high funding should
                           underperform -> positive spread = carry works)
             funding_pct = per-asset mid-rank pct of last_funding_rate vs
             trailing-90 daily values, MIN_WINDOW=20 (T104 convention,
             recomputed standalone — T104 results are NOT imported).
             Metrics: day-series mean/sd/t/p, pooled Welch (top-2 vs
             bottom-2 asset-days, Welch-Satterthwaite df), per-calendar-
             year fold sign consistency (2023/24/25/26), in-day rank-
             shuffle placebos (3 deterministic LCG seeds), PASS /
             DIRECTIONAL_ONLY / FAIL — FDR family 3 (3 XS cells).

  cross_scale  v2 funding_pct daily-vs-hourly row (byte-identical) plus a
             NEW dfh20 daily-vs-XS replication row.

  summary    verdict counts per block + verdict changes vs v2 (must be
             empty for all carried-forward cells).

Measurement only — no fitting, no trading, no promotion claims.
"""

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import financial_signal_benchmark_v1 as v1
import financial_signal_benchmark_v2 as v2

ROOT = Path(__file__).resolve().parent.parent
XS_COHORT = ROOT / "data/perp_pit_xs_v1/records.jsonl"
V2_RECEIPT = ROOT / "results/financial_signal_benchmark_v2.json"
OUT = ROOT / "results/financial_signal_benchmark_v3.json"
DFH_WIN = 20                 # strictly-prior bars for the trailing high
XS_LOOKBACK = 90             # trailing-90 records for funding_pct (T104)
XS_MIN_WINDOW = 20           # floor for a usable trailing pct
XS_MIN_ASSETS = 8            # a top-2/bottom-2 contrast needs a real XS
XS_EDGE = 2                  # assets per spread leg
XS_ARMS = ["dfh20", "mom20", "funding_pct"]
# long leg of each spread: momentum arms buy the top-2 ranks, the carry
# arm buys the bottom-2 (high-funding assets should underperform)
XS_LONG_LEG = {"dfh20": "top2", "mom20": "top2", "funding_pct": "bottom2"}
EXPECTED_INTRADAY_RECORDS = 17960
pct, spearman, welch, net_stats = v1.pct, v1.spearman, v1.welch, v1.net_stats
FEE_BPS, PLACEBO_SEEDS = v1.FEE_BPS, v1.PLACEBO_SEEDS


def norm_p(t):
    """Two-sided normal-approx p for a t-stat (repo convention)."""
    if t is None:
        return None
    return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))


def t_stat(xs):
    """Mean/sd/t of a series (obs treated as independent; nominal)."""
    n = len(xs)
    if n < 5:
        return {"n": n, "mean": (sum(xs) / n) if n else None,
                "sd": None, "t": None, "p": None}
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(var)
    t = (mean / (sd / math.sqrt(n))) if sd > 0 else None
    return {"n": n, "mean": mean, "sd": sd, "t": t, "p": norm_p(t)}


def welch_df(xs, ys):
    """Welch two-sample t + Welch-Satterthwaite df (T104) + normal p."""
    nx, ny = len(xs), len(ys)
    if nx < 2 or ny < 2:
        return {"n_top": nx, "n_bottom": ny, "t": None, "df": None,
                "p": None}
    mx, my = sum(xs) / nx, sum(ys) / ny
    vx = sum((x - mx) ** 2 for x in xs) / (nx - 1)
    vy = sum((y - my) ** 2 for y in ys) / (ny - 1)
    denom = vx / nx + vy / ny
    if denom <= 0:
        return {"n_top": nx, "n_bottom": ny, "mean_top": mx,
                "mean_bottom": my, "t": None, "df": None, "p": None}
    t = (mx - my) / math.sqrt(denom)
    df = denom ** 2 / ((vx / nx) ** 2 / (nx - 1)
                       + (vy / ny) ** 2 / (ny - 1))
    return {"n_top": nx, "n_bottom": ny, "mean_top": mx,
            "mean_bottom": my, "t": t, "df": df, "p": norm_p(t)}


def lcg_shuffle(vals, state):
    """In-place Fisher-Yates on a copy with a persistent LCG state —
    same generator as T82/T91 but the state carries across calls so each
    decision day gets a different permutation. Returns (copy, state)."""
    v = vals[:]
    s = state
    for i in range(len(v) - 1, 0, -1):
        s = (s * 6364136223846793005 + 1442695040888963407) & (2**64 - 1)
        j = s % (i + 1)
        v[i], v[j] = v[j], v[i]
    return v, s


# ---------------------------------------------------------------- daily dfh
def daily_dfh_rows():
    """v1 eval rows + dfh20_pct (T101 trailing-180 mid-rank convention).

    dfh20[i] = mark[i]/max(mark[i-20:i]) - 1 over strictly-prior bars;
    dfh20_pct = mid-rank pct of dfh20[i] vs the trailing-180 non-None
    dfh20 window, per asset.
    """
    records = [json.loads(l)
               for l in v1.COHORT.read_text().splitlines() if l.strip()]
    by_asset = defaultdict(list)
    for r in records:
        by_asset[r["asset_id"]].append(r)
    for rs in by_asset.values():
        rs.sort(key=lambda x: x["decision_ns"])
    dfh = {}
    for asset, rs in by_asset.items():
        mark = [x["features"]["mark_price"]["value"] for x in rs]
        d = [None] * len(rs)
        for i in range(DFH_WIN, len(rs)):
            w = [m for m in mark[i - DFH_WIN:i] if m and m > 0]
            if len(w) == DFH_WIN and mark[i] and mark[i] > 0:
                d[i] = mark[i] / max(w) - 1.0
        dfh[asset] = d

    rows = []
    for r in v1.build_dataset():
        a, i = r["asset"], r["i_in_asset"]
        if dfh[a][i] is None:
            continue
        w = [x for x in dfh[a][i - v1.LOOKBACK:i] if x is not None]
        if not w:
            continue
        r["dfh20_pct"] = pct(w, dfh[a][i])
        rows.append(r)
    return rows


def daily_dfh_block(daily):
    """Scorecard-shaped cell for dfh20_pct: Spearman + decile Welch."""
    rows = daily_dfh_rows()
    folds = json.loads(v1.FOLDS.read_text())["folds"]
    windows = [(f["test"][0], f["test"][1]) for f in folds]
    name = "dfh20_pct"
    xs = [r[name] for r in rows]
    ys = [r["ret"] for r in rows]
    sp = spearman(xs, ys)

    k = max(10, len(rows) // 10)
    srt = sorted(rows, key=lambda r: r[name])
    w = welch([r["gross_bps"] for r in srt[-k:]],
              [r["gross_bps"] for r in srt[:k]])
    welch_dec = ({"n_top": w["n_a"], "n_bottom": w["n_b"],
                  "mean_top_bps": round(w["mean_a"], 5),
                  "mean_bottom_bps": round(w["mean_b"], 5),
                  "diff_bps": round(w["mean_a"] - w["mean_b"], 5),
                  "t": round(w["t"], 5), "p": round(w["p"], 5)}
                 if w else None)

    arm = srt[-k:]
    fold_signs = []
    for a, b_ in windows:
        sub = [r["ret"] for r in arm if a <= r["ns"] < b_]
        fold_signs.append((1 if statistics.mean(sub) > 0 else -1)
                          if len(sub) >= 5 else None)
    cell = {"n": len(rows),
            "primary": {"spearman": ({k_: round(vv, 5)
                                      for k_, vv in sp.items()}
                                     | {"n": len(rows)} if sp else None)},
            "welch_top_vs_bottom_decile": welch_dec,
            "arm_n": len(arm),
            "arm_mean_gross_bps":
                round(statistics.mean([r["gross_bps"] for r in arm]), 1)
                if arm else None,
            "arm_net_stats": net_stats([r["net_bps"] for r in arm]),
            "fold_arm_signs": fold_signs,
            "placebo_rhos": v1.placebo_rhos(rows, name),
            "p": sp["p"] if sp else None,
            "rho": sp["rho"] if sp else None,
            "score_key": True}
    cell["placebo_max_abs_rho"] = max(
        (abs(x) for x in cell["placebo_rhos"] if x is not None),
        default=None)

    # 1-cell FDR family — deliberately OUTSIDE the frozen 10-cell family
    fdr_pass = cell["p"] is not None and cell["p"] <= 0.05
    rho, band = cell["rho"], cell["placebo_max_abs_rho"]
    pl = (rho is not None and band is not None
          and abs(rho) > max(0.02, band))
    always = daily["baseline_always_long"]
    econ = (cell["arm_net_stats"] and always and
            cell["arm_net_stats"]["mean_net_bps"]
            > always["mean_net_bps"])
    folds_ok = sum(1 for s in fold_signs if s == 1)
    verdict = {"fdr_pass": fdr_pass,
               "fdr_note": "1-cell family — outside the frozen 10-cell "
                           "daily family so frozen thresholds stay "
                           "byte-identical",
               "placebo_clear": pl,
               "econ_vs_always_long": econ,
               "folds_positive":
                   f"{folds_ok}/"
                   f"{sum(1 for s in fold_signs if s is not None)}",
               "verdict": "PASS" if fdr_pass and pl else
                          ("ECON_ONLY" if econ and not fdr_pass
                           else "FAIL")}

    return {"contract": {
                "score": "dfh20_pct = mid-rank pct of dfh20 "
                         "(mark[i]/max(mark[i-20:i])-1, strictly-prior "
                         "20 bars) vs trailing-180 non-None dfh20 "
                         "window, per-asset (T101 convention)",
                "label": "log(mark[t+5]/mark[t]) — same as frozen daily "
                         "cells",
                "welch": "top vs bottom decile of dfh20_pct on gross "
                         "fwd-5d bps",
                "fdr_family": [name],
                "placebo_seeds": PLACEBO_SEEDS},
            "cells": {name: cell},
            "fdr_pass": {name: fdr_pass},
            "verdicts": {name: verdict}}


# ---------------------------------------------------------------- xs block
def load_xs_series():
    """asset -> sorted rows {day, ns, funding, dfh20, mom20, fwd_bps,
    funding_pct}; funding_pct recomputed standalone per T104 convention
    (mid-rank pct of last_funding_rate vs trailing-90, MIN_WINDOW=20)."""
    series = defaultdict(list)
    for line in XS_COHORT.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        f = r["features"]
        series[r["asset_id"]].append({
            "day": r["id"].rsplit(":", 1)[-1],
            "ns": r["decision_ns"],
            "funding": f["last_funding_rate"]["value"],
            "dfh20": f["dfh20"]["value"],
            "mom20": f["mom20"]["value"],
            "fwd_bps": r["label"]["forward_return_5d_bps"]})
    for rows in series.values():
        rows.sort(key=lambda x: x["day"])
        fund = [r["funding"] for r in rows]
        for i, r in enumerate(rows):
            win = [x for x in fund[max(0, i - XS_LOOKBACK):i]
                   if x is not None]
            r["funding_pct"] = (pct(win, r["funding"])
                                if r["funding"] is not None
                                and len(win) >= XS_MIN_WINDOW else None)
    return series


def xs_days(series, score_key):
    """Per decision day (>=8 scored assets): ranked top-2/bottom-2 legs."""
    by_day = defaultdict(list)
    for asset, rows in series.items():
        for r in rows:
            if r.get(score_key) is not None and r["fwd_bps"] is not None:
                by_day[r["day"]].append(
                    (asset, r[score_key], r["fwd_bps"]))
    days = []
    for day in sorted(by_day):
        xs = by_day[day]
        if len(xs) < XS_MIN_ASSETS:
            continue
        ordered = sorted(xs, key=lambda x: x[1])
        bot, top = ordered[:XS_EDGE], ordered[-XS_EDGE:]
        if bot[-1][1] == top[0][1]:
            continue          # degenerate: no dispersion at the edges
        days.append({"day": day, "n_assets": len(xs),
                     "scores": [x[1] for x in xs],
                     "fwds": [x[2] for x in xs],
                     "top_fwds": [x[2] for x in top],
                     "bot_fwds": [x[2] for x in bot],
                     "top_assets": sorted(x[0] for x in top),
                     "bot_assets": sorted(x[0] for x in bot)})
    return days


def xs_spread(day, long_leg):
    top_m = sum(day["top_fwds"]) / len(day["top_fwds"])
    bot_m = sum(day["bot_fwds"]) / len(day["bot_fwds"])
    return (top_m - bot_m) if long_leg == "top2" else (bot_m - top_m)


def xs_placebo_means(days, long_leg):
    """In-day rank shuffle: permute scores within each decision day
    (persistent LCG state, sorted day order), recompute the spread."""
    out = []
    for seed in PLACEBO_SEEDS:
        s = seed
        spreads = []
        for d in days:                       # sorted day order
            shuf, s = lcg_shuffle(d["scores"], s)
            paired = sorted(zip(shuf, d["fwds"]), key=lambda x: x[0])
            bot, top = paired[:XS_EDGE], paired[-XS_EDGE:]
            if bot[-1][0] == top[0][0]:
                continue
            top_m = sum(x[1] for x in top) / XS_EDGE
            bot_m = sum(x[1] for x in bot) / XS_EDGE
            spreads.append(top_m - bot_m if long_leg == "top2"
                           else bot_m - top_m)
        out.append(round(statistics.mean(spreads), 5)
                   if spreads else None)
    return out


def xs_cell(days, score_key, long_leg):
    spreads = [xs_spread(d, long_leg) for d in days]
    ts = t_stat(spreads)
    pooled_top = [x for d in days for x in d["top_fwds"]]
    pooled_bot = [x for d in days for x in d["bot_fwds"]]
    leg_a, leg_b = ((pooled_top, pooled_bot) if long_leg == "top2"
                    else (pooled_bot, pooled_top))
    w = welch_df(leg_a, leg_b)

    folds = defaultdict(list)
    for d, s in zip(days, spreads):
        folds[d["day"][:4]].append(s)
    fold_means = {y: sum(v) / len(v) for y, v in sorted(folds.items())}
    pooled_mean = ts["mean"]
    consistent = sum(1 for m in fold_means.values()
                     if pooled_mean is not None and m * pooled_mean > 0)

    placebo = xs_placebo_means(days, long_leg)
    n_assets = [d["n_assets"] for d in days]
    return {"score": score_key,
            "spread_direction": (
                "fwd(top2)-fwd(bottom2) [momentum continuation]"
                if long_leg == "top2" else
                "fwd(bottom2)-fwd(top2) [carry: high funding should "
                "underperform]"),
            "days_evaluated": len(days),
            "day_span": {"first": days[0]["day"] if days else None,
                         "last": days[-1]["day"] if days else None},
            "assets_per_day": {
                "min": min(n_assets) if n_assets else None,
                "median": (sorted(n_assets)[len(n_assets) // 2]
                           if n_assets else None),
                "max": max(n_assets) if n_assets else None},
            "mean_daily_spread_bps":
                {k: (round(v, 5) if isinstance(v, float) else v)
                 for k, v in ts.items()},
            "pooled_welch":
                {k: (round(v, 5) if isinstance(v, float) else v)
                 for k, v in w.items()},
            "fold_sign_consistency": {
                "folds": len(fold_means),
                "same_sign_as_pooled": consistent,
                "fold_mean_bps": {y: round(m, 5)
                                  for y, m in fold_means.items()}},
            "placebo_mean_spreads_bps": placebo,
            "placebo_max_abs_spread_bps": max(
                (abs(x) for x in placebo if x is not None),
                default=None),
            "p": ts["p"],
            "mean_spread_bps": ts["mean"]}


def run_xs():
    records = [json.loads(l)
               for l in XS_COHORT.read_text().splitlines() if l.strip()]
    series = load_xs_series()
    cells = {}
    for name in XS_ARMS:
        cells[name] = xs_cell(xs_days(series, name), name,
                              XS_LONG_LEG[name])

    # BH-FDR family 3: the 3 XS cells only (separate from daily/hourly)
    ps = sorted((m["p"], n) for n, m in cells.items()
                if m.get("p") is not None)
    M = len(ps)
    fdr = {}
    for i, (p, n) in enumerate(ps):
        fdr[n] = p <= 0.05 * (i + 1) / M

    verdicts = {}
    for n, m in cells.items():
        stat = fdr.get(n, False)
        band = m["placebo_max_abs_spread_bps"]
        pl = (m["mean_spread_bps"] is not None and band is not None
              and abs(m["mean_spread_bps"]) > band)
        folds_ok = m["fold_sign_consistency"]["same_sign_as_pooled"]
        folds_n = m["fold_sign_consistency"]["folds"]
        verdicts[n] = {
            "fdr_pass": stat,
            "placebo_clear": pl,
            "folds_same_sign": f"{folds_ok}/{folds_n}",
            "verdict": "PASS" if stat and pl else
                       ("DIRECTIONAL_ONLY" if pl and folds_ok > folds_n / 2
                        else "FAIL")}

    return {
        "contract": {
            "cohort": str(XS_COHORT.relative_to(ROOT)),
            "cohort_sha256": hashlib.sha256(
                XS_COHORT.read_bytes()).hexdigest(),
            "n_records": len(records),
            "assets": sorted({r["asset_id"] for r in records}),
            "label": "label.forward_return_5d_bps (gross close-to-close)",
            "scores": {
                "dfh20": "precomputed feature: close/max(close[i-20:i])"
                         "-1 (strictly-prior 20 bars)",
                "mom20": "precomputed feature: close/close[i-20]-1 "
                         "(strictly-prior 20-bar return)",
                "funding_pct": "per-asset mid-rank pct of "
                               "last_funding_rate vs trailing-90 daily "
                               "values, MIN_WINDOW=20 (T104 convention, "
                               "recomputed standalone)"},
            "min_assets_per_day": XS_MIN_ASSETS,
            "edge_size": XS_EDGE,
            "fdr_alpha": 0.05,
            "fdr_family": XS_ARMS,
            "folds": "per-calendar-year fold mean spreads "
                     "(2023/2024/2025/2026)",
            "placebo": "in-day score shuffle across assets, persistent "
                       "LCG state, seeds "
                       f"{list(PLACEBO_SEEDS)}"},
        "cells": cells,
        "fdr_pass": fdr,
        "verdicts": verdicts,
        "caveats": [
            "top-2/bottom-2 on ~10 assets = ~4-5 effective dof per day; "
            "one asset's jump swings a daily spread",
            "5d overlapping labels autocorrelate adjacent days; "
            "day-series t and pooled Welch overstate effective n "
            "(nominal)",
            "gross close moves only — no fees, funding cashflows or "
            "tradability; measurement, not a signal claim",
            "funding_pct carries the honest T104 prior: post-2020 "
            "literature reports XS funding carry dead on majors — a "
            "non-null here is surprising, not confirming"]}


# ---------------------------------------------------------------- cross-scale
def cross_scale_dfh(daily_dfh, xs):
    d = daily_dfh["cells"]["dfh20_pct"]
    dv = daily_dfh["verdicts"]["dfh20_pct"]
    x = xs["cells"]["dfh20"]
    xv = xs["verdicts"]["dfh20"]
    same_sign = (d["rho"] is not None and x["mean_spread_bps"] is not None
                 and d["rho"] * x["mean_spread_bps"] > 0)
    if dv["verdict"] == "PASS" and xv["verdict"] == "PASS" and same_sign:
        rep = "replicated_both_scales"
    elif same_sign and dv["placebo_clear"] and xv["placebo_clear"]:
        rep = "directional_only"
    else:
        rep = "not_replicated"
    return {"cell": "dfh20",
            "daily": {"score": "dfh20_pct (trailing-180 pct)",
                      "rho": d["rho"], "p": d["p"], "n": d["n"],
                      "verdict": dv["verdict"],
                      "placebo_max_abs_rho": d["placebo_max_abs_rho"]},
            "xs": {"score": "dfh20 rank top2-vs-bottom2 5d spread",
                   "mean_spread_bps": x["mean_spread_bps"],
                   "t": x["mean_daily_spread_bps"]["t"],
                   "p": x["p"], "days": x["days_evaluated"],
                   "verdict": xv["verdict"],
                   "placebo_max_abs_spread_bps":
                       x["placebo_max_abs_spread_bps"]},
            "same_sign": same_sign,
            "replication": rep,
            "note": "same underlying dfh20 feature, different contrast "
                    "geometry: daily cell is a within-asset trailing-180 "
                    "percentile vs 5d fwd log ret; XS arm ranks dfh20 "
                    "across 10 assets per day on gross 5d bps"}


# ---------------------------------------------------------------- run
def verdict_counts(*blocks):
    out = {}
    total = {"PASS": 0, "ECON_ONLY": 0, "DIRECTIONAL_ONLY": 0, "FAIL": 0}
    for name, verdicts in blocks:
        c = {"PASS": 0, "ECON_ONLY": 0, "DIRECTIONAL_ONLY": 0, "FAIL": 0}
        for v in verdicts.values():
            c[v["verdict"]] += 1
            total[v["verdict"]] += 1
        out[name] = c
    out["total"] = total
    return out


def run():
    daily = v1.run()                       # frozen T82 block, verbatim
    daily["dfh_cells"] = daily_dfh_block(daily)
    intraday = v2.run_intraday()           # frozen T91 block, verbatim
    xs = run_xs()
    cs = {"funding_pct_daily_vs_hourly":
          v2.cross_scale(daily, intraday),
          "dfh20_daily_vs_xs": cross_scale_dfh(daily["dfh_cells"], xs)}

    # ---- verification vs the frozen v2 receipt ----
    ver = {"v2_receipt": str(V2_RECEIPT.relative_to(ROOT)),
           "daily_cells_byte_identical_to_v2": None,
           "intraday_block_byte_identical_to_v2": None,
           "cross_scale_funding_row_byte_identical_to_v2": None,
           "intraday_cohort_n_records":
               intraday["contract"]["n_records"],
           "intraday_on_expanded_17960_cohort":
               intraday["contract"]["n_records"]
               == EXPECTED_INTRADAY_RECORDS,
           "verdict_changes_vs_v2": []}
    if V2_RECEIPT.exists():
        old = json.loads(V2_RECEIPT.read_text())
        frozen = {k: v for k, v in daily.items() if k != "dfh_cells"}
        ver["daily_cells_byte_identical_to_v2"] = (
            json.dumps(frozen, sort_keys=True)
            == json.dumps(old["daily"], sort_keys=True))
        ver["intraday_block_byte_identical_to_v2"] = (
            json.dumps(intraday, sort_keys=True)
            == json.dumps(old["intraday"], sort_keys=True))
        ver["cross_scale_funding_row_byte_identical_to_v2"] = (
            json.dumps(cs["funding_pct_daily_vs_hourly"], sort_keys=True)
            == json.dumps(old["cross_scale"], sort_keys=True))
        for block_name, new_v, old_v in (
                ("daily", daily["verdicts"], old["daily"]["verdicts"]),
                ("intraday", intraday["verdicts"],
                 old["intraday"]["verdicts"])):
            for cell, vv in new_v.items():
                ov = old_v.get(cell, {}).get("verdict")
                if ov != vv["verdict"]:
                    ver["verdict_changes_vs_v2"].append(
                        {"block": block_name, "cell": cell,
                         "v2": ov, "v3": vv["verdict"]})
    else:
        ver["note"] = "v2 receipt not found — byte-identity checks skipped"

    summary = verdict_counts(
        ("daily", daily["verdicts"]),
        ("daily_dfh", daily["dfh_cells"]["verdicts"]),
        ("intraday", intraday["verdicts"]),
        ("xs", xs["verdicts"]))

    return {"schema_version": "nanojev-financial-signal-benchmark-v3",
            "status": "benchmark_complete",
            "task": "T108",
            "protocol": "extends research/"
                        "financial_signal_benchmark_protocol_v2.json "
                        "in-place: frozen daily+intraday cells unchanged; "
                        "new dfh daily cell (T101 convention) and 3-cell "
                        "XS block added as separate FDR families",
            "fdr_family_note": "four separate BH-FDR families at "
                               "alpha=0.05: 10 frozen daily cells, 1 "
                               "daily dfh cell (outside the frozen "
                               "family), 7 intraday cells, 3 XS cells — "
                               "no cross-scale pooling",
            "findings_note": "mom_1h (rho=-0.0350, p=3.6e-6) and mom_4h "
                             "(rho=-0.0471, p=4.5e-10) PASS with "
                             "NEGATIVE rho on the 17,960-record intraday "
                             "cohort — short-horizon reversal, not "
                             "continuation; carried forward from v2",
            "daily": daily,
            "intraday": intraday,
            "xs": xs,
            "cross_scale": cs,
            "summary_verdict_counts": summary,
            "verification": ver}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    r1, r2 = run(), run()
    same = json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    r1["determinism"] = {"replays": 2, "byte_identical": same}
    r1["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    blob = json.dumps(r1, indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(blob)
    print(json.dumps({"output": str(args.output), "deterministic": same,
                      "sha256": hashlib.sha256(blob.encode()).hexdigest()},
                     indent=2))


if __name__ == "__main__":
    main()
