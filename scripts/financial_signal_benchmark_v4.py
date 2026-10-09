#!/usr/bin/env python3
"""T133: cross-scale unified financial-signal benchmark (v4).

Extends the frozen v3 scorecard (``financial_signal_benchmark_v3.run()``
is imported and embedded verbatim — byte-identity vs the v3 receipt is
verified at runtime) with the mega-scale cohorts:

  carried_v3  v3.run() verbatim — daily 10-cell + dfh 1-cell, intraday
              7-cell, XS 3-cell on 10 assets, plus v3's own v2-receipt
              verification block. NOTHING in it is recomputed here.

  mega_xs    NEW. cohort=data/perp_pit_mega_v1/records.jsonl (sha
             pinned; 277 USDT-M perps incl. delisted early-stoppers,
             2021-01..2025-12, close-as-mark, ~52 funding-covered).
             Independent recompute of the T112 headline arms at decile
             depth (top-decile vs bottom-decile, >=30 ranked
             assets/day, k=max(1,n//10) per side, 5d fwd spread);
             T112's JSON is never read — only conventions are shared:
               dfh20         long top-decile (momentum continuation)
               mom20         long top-decile
               xs_rank_mom20 NEW cell: same-day XS mid-rank pct of
                             mom20 (the ridge feature). Within a day it
                             is a monotone re-label of mom20 so its
                             decile book is construction-identical —
                             kept as a pipeline consistency cell.
               funding_pct   carry on the funded subset: per-asset
                             mid-rank pct of last_funding_rate vs
                             trailing-180 daily values (MIN_WINDOW=20,
                             T112 convention — NOT the 10-asset block's
                             trailing-90); long bottom-decile.
             Metrics: day-series t/p, pooled Welch on leg asset-days,
             per-calendar-year folds (2021-2025), in-day rank-shuffle
             placebos x3 (persistent LCG, benchmark seeds 17/73/991),
             BH-FDR family of 4. Verdicts PASS / DIRECTIONAL_ONLY /
             FAIL; PASS additionally requires >=(n-1)/n yearly folds
             same sign — the gate that catches funding_pct's
             fold-unstable +76bps.

  ridge_model NEW. Refit of the T127 ridge_min3 arm {dfh20, btc_ret20,
             dfh20*btc_ret20} -> forward_return_5d_bps via
             financial_signal_ridge_v2_v1's own machinery (imported:
             load_btc_ret20 / build_dataset / run_arm — the same
             walkforward 21->22..24->25, 5d embargo, in-train-year
             forward-chaining lambda CV). Per-fold OOS Spearman rho +
             net LS decile book bps/day (5bps/leg one-way turnover,
             daily rebalance). Verdict PASS iff >=3/4 folds positive
             rho AND pooled net LS > 0. Requires numpy — run with the
             repo venv (.venv/bin/python); without numpy the block is
             marked SKIPPED, never fabricated.

  mega_4h    NEW (sampled). cohort=data/perp_pit_mega_4h_v1/
             records.jsonl (~1.28M lean rows, ~276 syms, 2023-2025).
             The full stream is too heavy for a scorecard cell, so a
             deterministic subsample is used: EVERY 4TH UTC DATE (all
             4h bars of kept dates; sorted epoch-day index%4==0) ->
             ~320k rows, all symbols kept. NOTE: sampling every 4th
             timestamp instead would keep a single 4h phase and
             destroy the settle-vs-rest split — date sampling keeps
             all six daily phases. mom_4h score = per-asset trailing-180
             SAMPLED-record mid-rank pct of m4 (MIN_WINDOW=60; effective
             span ~4x the T124 window — documented). Cells:
               mom_4h_rho        pooled Spearman(m4_pct, fwd4) +
                                 yearly folds; T96/T124 prior: rho<0
                                 (4h REVERSAL)
               settlement_window Welch(fwd4 | htsg<=2 vs htsg>2) on the
                                 assumed 00/08/16 grid + yearly folds;
                                 T93/T124 prior: diff<0
             2-cell BH-FDR family; PASS needs right direction, FDR and
             >=2/3 yearly folds same sign.

  cross_scale v2 funding row (byte-identical) + dfh20 on THREE scales
             (daily-5 pct rho / XS-10 / XS-277 decile spreads) + NEW
             mom horizon-split row (daily continuation vs 4h reversal).

  verification  byte-identity of the embedded v3 payload vs
             results/financial_signal_benchmark_v3.json (minus its
             generated_at/determinism trailer) + verdict-change diff +
             cohort sha pins. Determinism: run() x2 byte-compared.

Measurement only — ridge refit is PIT-safe walkforward OOS evaluation,
not strategy promotion; no trading, no profitability claims, no
network.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import statistics
import time
from bisect import bisect_left, bisect_right, insort
from collections import defaultdict
from pathlib import Path

import financial_signal_benchmark_v3 as v3

v1 = v3.v1
v2 = v3.v2

try:
    import financial_signal_ridge_v2_v1 as ridge
    RIDGE_IMPORT_ERROR = None
except Exception as exc:                      # e.g. numpy missing
    ridge = None
    RIDGE_IMPORT_ERROR = repr(exc)

ROOT = Path(__file__).resolve().parent.parent
MEGA_COHORT = ROOT / "data/perp_pit_mega_v1/records.jsonl"
MEGA_4H_COHORT = ROOT / "data/perp_pit_mega_4h_v1/records.jsonl"
V3_RECEIPT = ROOT / "results/financial_signal_benchmark_v3.json"
OUT = ROOT / "results/financial_signal_benchmark_v4.json"

MEGA_MIN_ASSETS = 30       # decile contrasts need a real cross-section
MEGA_DECILE = 10
MEGA_FUND_LOOKBACK = 180   # T112 convention (xs-10 block uses 90)
MEGA_MIN_WINDOW = 20
MEGA_ARMS = ["dfh20", "mom20", "xs_rank_mom20", "funding_pct"]
MEGA_LONG_LEG = {"dfh20": "top", "mom20": "top", "xs_rank_mom20": "top",
                 "funding_pct": "bottom"}
# independent-recompute expectations from the T112 headline (scorecard
# sanity anchors, not copied numbers)
MEGA_EXPECT = {"dfh20": 68.9, "mom20": 57.5, "funding_pct": 76.0}

H4_SAMPLE_MOD = 4          # keep every 4th distinct UTC date
H4_TRAIL = 180             # trailing SAMPLED records for m4_pct
H4_MIN_WINDOW = 60
H4_SETTLE_H = 2.0          # +-2h of a settlement (grid: htsg<=2)
H4_YEARS = ("2023", "2024", "2025")

_pct = v1.pct              # mid-rank percentile (repo convention)


# ============================================================== mega XS
def load_mega():
    """Single parse of the mega cohort serving BOTH mega_xs and the
    ridge refit (ridge-format rows keep ridge.load_rows' field names so
    ridge.build_dataset consumes them unchanged)."""
    series = defaultdict(list)
    ridge_rows = []
    n = 0
    assets, funded = set(), set()
    with MEGA_COHORT.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            a = r["asset_id"]
            assets.add(a)
            ft, lb = r["features"], r["label"]
            day = r["id"].rsplit(":", 1)[-1]
            fund = ft["last_funding_rate"]["value"]
            if fund is not None:
                funded.add(a)
            series[a].append({
                "day": day,
                "dfh20": ft["dfh20"]["value"],
                "mom20": ft["mom20"]["value"],
                "funding": fund,
                "fwd_bps": lb["forward_return_5d_bps"],
                "xs_rank_mom20": None,
                "funding_pct": None})
            ridge_rows.append({
                "asset": a,
                "date": dt.date.fromisoformat(day),
                "close": ft["close"]["value"],
                "dfh20": ft["dfh20"]["value"],
                "mom20": ft["mom20"]["value"],
                "vol20": ft["vol20"]["value"],
                "funding": fund,
                "qvol": ft["quote_volume"]["value"],
                "y": lb["forward_return_5d_bps"],
                "y1": lb["forward_return_bps"]})
    for rows in series.values():
        rows.sort(key=lambda x: x["day"])
    meta = {"n_records": n, "assets": sorted(assets),
            "funded_assets": sorted(funded)}
    return series, ridge_rows, meta


def add_mega_scores(series):
    """funding_pct (trailing-180 mid-rank pct of last_funding_rate,
    MIN_WINDOW=20 — T112 convention) and xs_rank_mom20 (same-day XS
    mid-rank pct of mom20). Sorted-window bisect = same mid-rank math
    as v1.pct."""
    for rows in series.values():
        fund = [r["funding"] for r in rows]
        win = []
        for i, r in enumerate(rows):
            v = fund[i]
            if v is not None and len(win) >= MEGA_MIN_WINDOW:
                r["funding_pct"] = (bisect_left(win, v)
                                    + bisect_right(win, v)) / (2.0 * len(win))
            if v is not None:
                insort(win, v)
            if i >= MEGA_FUND_LOOKBACK:
                old = fund[i - MEGA_FUND_LOOKBACK]
                if old is not None:
                    win.pop(bisect_left(win, old))
    by_day = defaultdict(list)
    for rows in series.values():
        for r in rows:
            if r["mom20"] is not None:
                by_day[r["day"]].append(r)
    for rows in by_day.values():
        vals = sorted(r["mom20"] for r in rows)
        n = len(vals)
        for r in rows:
            r["xs_rank_mom20"] = (bisect_left(vals, r["mom20"])
                                  + bisect_right(vals, r["mom20"])) / (2.0 * n)


def mega_xs_days(series, score_key):
    """Per decision day (>=30 scored assets): top/bottom decile legs,
    k = max(1, n//10)."""
    by_day = defaultdict(list)
    for asset, rows in series.items():
        for r in rows:
            s, f = r.get(score_key), r["fwd_bps"]
            if s is not None and f is not None:
                by_day[r["day"]].append((asset, s, f))
    days = []
    for day in sorted(by_day):
        xs = by_day[day]
        if len(xs) < MEGA_MIN_ASSETS:
            continue
        ordered = sorted(xs, key=lambda x: x[1])
        k = max(1, len(ordered) // MEGA_DECILE)
        bot, top = ordered[:k], ordered[-k:]
        if bot[-1][1] == top[0][1]:
            continue          # degenerate: no dispersion at the edges
        days.append({"day": day, "n_assets": len(xs), "edge": k,
                     "scores": [x[1] for x in xs],
                     "fwds": [x[2] for x in xs],
                     "top_fwds": [x[2] for x in top],
                     "bot_fwds": [x[2] for x in bot]})
    return days


def mega_spread(day, long_leg):
    top_m = sum(day["top_fwds"]) / len(day["top_fwds"])
    bot_m = sum(day["bot_fwds"]) / len(day["bot_fwds"])
    return (top_m - bot_m) if long_leg == "top" else (bot_m - top_m)


def mega_placebo_means(days, long_leg):
    """In-day score shuffle (persistent LCG state, sorted day order),
    decile edge recomputed per day."""
    out = []
    for seed in v1.PLACEBO_SEEDS:
        s = seed
        spreads = []
        for d in days:
            shuf, s = v3.lcg_shuffle(d["scores"], s)
            paired = sorted(zip(shuf, d["fwds"]), key=lambda x: x[0])
            k = max(1, len(paired) // MEGA_DECILE)
            bot, top = paired[:k], paired[-k:]
            if bot[-1][0] == top[0][0]:
                continue
            top_m = sum(x[1] for x in top) / len(top)
            bot_m = sum(x[1] for x in bot) / len(bot)
            spreads.append(top_m - bot_m if long_leg == "top"
                           else bot_m - top_m)
        out.append(round(statistics.mean(spreads), 5)
                   if spreads else None)
    return out


def mega_xs_cell(days, score_key, long_leg):
    spreads = [mega_spread(d, long_leg) for d in days]
    ts = v3.t_stat(spreads)
    pooled_top = [x for d in days for x in d["top_fwds"]]
    pooled_bot = [x for d in days for x in d["bot_fwds"]]
    leg_l, leg_s = ((pooled_top, pooled_bot) if long_leg == "top"
                    else (pooled_bot, pooled_top))
    w = v3.welch_df(leg_l, leg_s)
    pooled_welch = None
    if w.get("t") is not None or w.get("n_top", 0) >= 2:
        pooled_welch = {
            "n_long": w.get("n_top"), "n_short": w.get("n_bottom"),
            "mean_long_bps": w.get("mean_top"),
            "mean_short_bps": w.get("mean_bottom"),
            "t": w.get("t"), "df": w.get("df"), "p": w.get("p")}
        pooled_welch = {k: (round(v, 5) if isinstance(v, float) else v)
                        for k, v in pooled_welch.items()}

    folds = defaultdict(list)
    for d, s in zip(days, spreads):
        folds[d["day"][:4]].append(s)
    fold_means = {y: sum(v) / len(v) for y, v in sorted(folds.items())}
    pooled_mean = ts["mean"]
    consistent = sum(1 for m in fold_means.values()
                     if pooled_mean is not None and m * pooled_mean > 0)

    placebo = mega_placebo_means(days, long_leg)
    n_assets = [d["n_assets"] for d in days]
    edges = [d["edge"] for d in days]
    return {"score": score_key,
            "spread_direction": (
                "fwd(top-decile)-fwd(bottom-decile) [momentum "
                "continuation]" if long_leg == "top" else
                "fwd(bottom-decile)-fwd(top-decile) [carry: high "
                "funding should underperform]"),
            "days_evaluated": len(days),
            "day_span": {"first": days[0]["day"] if days else None,
                         "last": days[-1]["day"] if days else None},
            "assets_per_day": {
                "min": min(n_assets) if n_assets else None,
                "median": (sorted(n_assets)[len(n_assets) // 2]
                           if n_assets else None),
                "max": max(n_assets) if n_assets else None},
            "edge_names_per_side": {
                "min": min(edges) if edges else None,
                "median": (sorted(edges)[len(edges) // 2]
                           if edges else None),
                "max": max(edges) if edges else None},
            "mean_daily_spread_bps":
                {k: (round(v, 5) if isinstance(v, float) else v)
                 for k, v in ts.items()},
            "pooled_welch": pooled_welch,
            "fold_sign_consistency": {
                "folds": len(fold_means),
                "same_sign_as_pooled": consistent,
                "fold_mean_bps": {y: round(m, 5)
                                  for y, m in fold_means.items()}},
            "placebo_mean_spreads_bps": placebo,
            "placebo_max_abs_spread_bps": max(
                (abs(x) for x in placebo if x is not None),
                default=None),
            "t112_expected_bps": MEGA_EXPECT.get(score_key),
            "p": ts["p"],
            "mean_spread_bps": ts["mean"]}


def run_mega_xs(series, meta):
    cells = {}
    for name in MEGA_ARMS:
        cells[name] = mega_xs_cell(mega_xs_days(series, name), name,
                                   MEGA_LONG_LEG[name])
    # xs_rank_mom20 is a within-day monotone re-label of mom20: flag the
    # construction-identical result instead of treating it as evidence.
    mom, xrk = cells["mom20"], cells["xs_rank_mom20"]
    rank_consistent = (
        mom["mean_spread_bps"] is not None
        and xrk["mean_spread_bps"] is not None
        and abs(mom["mean_spread_bps"] - xrk["mean_spread_bps"]) < 1e-9)
    cells["xs_rank_mom20"]["construction_note"] = (
        "monotone within-day re-label of mom20 -> decile membership is "
        "identical BY CONSTRUCTION (verified: mean spread matches mom20 "
        f"to <1e-9 bps: {rank_consistent}); this cell validates the "
        "xs_rank pipeline, it is not an independent replication")

    # BH-FDR family 5: the 4 mega XS cells (5d headline ps only)
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
        mostly = folds_n > 0 and folds_ok >= max(1, folds_n - 1)
        majority = folds_n > 0 and folds_ok > folds_n / 2
        verdicts[n] = {
            "fdr_pass": stat,
            "placebo_clear": pl,
            "folds_same_sign": f"{folds_ok}/{folds_n}",
            "fold_gate_pass": mostly,
            "verdict": "PASS" if stat and pl and mostly else
                       ("DIRECTIONAL_ONLY" if pl and majority
                        else "FAIL")}
    return {
        "contract": {
            "cohort": str(MEGA_COHORT.relative_to(ROOT)),
            "cohort_sha256": hashlib.sha256(
                MEGA_COHORT.read_bytes()).hexdigest(),
            "n_records": meta["n_records"],
            "n_symbols": len(meta["assets"]),
            "n_funded_symbols": len(meta["funded_assets"]),
            "label": "label.forward_return_5d_bps (gross close-to-close)",
            "scores": {
                "dfh20": "precomputed feature: close/max(close[i-20:i])"
                         "-1 (strictly-prior 20 bars)",
                "mom20": "precomputed feature: close/close[i-20]-1",
                "xs_rank_mom20": "same-day cross-sectional mid-rank "
                                 "pct of mom20 (ridge feature set)",
                "funding_pct": "per-asset mid-rank pct of "
                               "last_funding_rate vs trailing-180 daily "
                               "values, MIN_WINDOW=20 (T112 convention; "
                               "funded subset only)"},
            "min_assets_per_day": MEGA_MIN_ASSETS,
            "edge": "k = max(1, n//10) names per side (decile)",
            "fdr_alpha": 0.05,
            "fdr_family": MEGA_ARMS,
            "folds": "per-calendar-year fold mean spreads 2021-2025; "
                     "PASS requires >=(n-1)/n same sign",
            "placebo": "in-day score shuffle across assets, persistent "
                       "LCG state, seeds "
                       f"{list(v1.PLACEBO_SEEDS)}",
            "independence": "recomputed in this file from raw records; "
                            "T112's results JSON is never read"},
        "cells": cells,
        "fdr_pass": fdr,
        "verdicts": verdicts,
        "caveats": [
            "5d overlapping labels autocorrelate adjacent days; all "
            "t/p nominal, effective n << days_evaluated",
            "universe composition changes as symbols list/delist; "
            "decile size floats with n",
            "cohort includes delisted/renamed early-stoppers (less "
            "survivorship bias, not zero); see T112's delisting "
            "controls for the restricted-universe check",
            "funding arm ranks only the ~52 funded symbols (edge "
            "~5/side) and is fold-unstable",
            "gross close moves only — no fees, funding cashflows or "
            "tradability; measurement, not a signal claim"]}


# ========================================================= ridge model
def run_ridge_model(ridge_rows):
    """Refit ridge_min3 walkforward via T127 machinery (imported)."""
    if ridge is None:
        return {"status": "SKIPPED — ridge machinery unavailable",
                "import_error": RIDGE_IMPORT_ERROR,
                "verdicts": {"ridge_min3": {"verdict": "FAIL",
                             "note": "machinery unavailable"}}}
    btc_ret20 = ridge.load_btc_ret20()
    rows, dstats = ridge.build_dataset(ridge_rows, btc_ret20)
    arm = ridge.run_arm("ridge_min3", rows, ridge.MIN3_FEATURES)
    nb = arm["net_book"]
    pooled_ls = nb["pooled"]["ls"] if nb and nb["pooled"] else {}
    per_year_net = {y: (v["ls"]["mean_net_bps_day"] if v else None)
                    for y, v in nb["per_year"].items()}
    pos_rho = arm["folds_rho_positive"]
    n_ev = arm["folds_evaluated"]
    net_pos = (pooled_ls.get("mean_net_bps_day") is not None
               and pooled_ls["mean_net_bps_day"] > 0)
    gates = {"folds_rho_positive_at_least_3_of_4":
             n_ev == len(ridge.FOLD_PAIRS) and pos_rho >= 3,
             "net_ls_bps_day_positive_pooled": net_pos}
    verdict = "PASS" if all(gates.values()) else "FAIL"
    folds_out = []
    for f in arm["folds"]:
        folds_out.append({
            "train_year": f.get("train_year"),
            "test_year": f.get("test_year"),
            "skipped": f.get("skipped", False),
            "lambda": f.get("lambda"),
            "n_train": f.get("n_train"),
            "n_test": f.get("n_test"),
            "oos_spearman_rho": f.get("spearman_pred_ret"),
            "top_decile_spread_bps": f.get("top_decile_spread_bps"),
            "mse_ratio_vs_train_mean": f.get("mse_ratio_vs_train_mean"),
            "coef": f.get("coef")})
    return {
        "contract": {
            "model": "ridge_min3: standardized {dfh20, btc_ret20, "
                     "dfh20*btc_ret20} + unpenalized intercept -> "
                     "label.forward_return_5d_bps",
            "machinery": "financial_signal_ridge_v2_v1 imported: "
                         "load_btc_ret20 / build_dataset / run_arm — "
                         "same walkforward folds (2021->22 .. 2024->25), "
                         "5d embargo, in-train-year forward-chaining "
                         "lambda CV (never test); book = daily EW "
                         "top/bottom-decile of OOS preds, 0.5 notional "
                         "per side, 5bps one-way per leg turnover",
            "book_pnl": "label.forward_return_bps (1d gross)",
            "btc_series": str(ridge.BTC_CSV.relative_to(ROOT)),
            "refit": "refit in this run — T127's results JSON is not "
                     "read"},
        "dataset": dstats,
        "folds": folds_out,
        "pooled_oos": arm["pooled_oos"],
        "coef_stability": arm["coef_stability"],
        "folds_rho_positive": f"{pos_rho}/{n_ev}",
        "net_ls_book": {
            "pooled": {k: pooled_ls.get(k) for k in
                       ("mean_gross_bps_day", "mean_net_bps_day",
                        "total_net_bps", "sharpe_gross", "sharpe_net",
                        "hit_rate_net")},
            "per_year_mean_net_bps_day": per_year_net,
            "turnover": nb["pooled"]["turnover"] if nb["pooled"] else None,
            "n_days": nb["n_days_total"],
            "cost_model": nb["cost_model"]},
        "verdicts": {"ridge_min3": {
            "gates": gates,
            "verdict": verdict,
            "note": "task rule: PASS iff >=3/4 folds positive OOS rho "
                    "AND positive pooled net LS bps/day"}}}


# ============================================================== mega 4h
def _line_ts(line):
    """Cheap ts extraction — records are lean flat dicts with ts last."""
    i = line.rfind('"ts":')
    return int(line[i + 5:line.index("}", i)])


def load_mega_4h_sampled():
    """Two-pass deterministic subsample: every 4th distinct UTC date
    (epoch-day = ts_ms // 86400000, sorted, index % 4 == 0; ALL 4h bars
    of a kept date are retained so the settle-vs-rest split survives).
    Pass 1 reads only the ts tail of each line; pass 2 json-parses
    sampled lines only."""
    days = set()
    with MEGA_4H_COHORT.open(encoding="utf-8") as f:
        for line in f:
            if '"ts":' not in line:
                continue
            try:
                days.add(_line_ts(line) // 86400000)
            except Exception:
                days.add(json.loads(line)["ts"] // 86400000)
    ordered = sorted(days)
    keep = set(ordered[::H4_SAMPLE_MOD])
    series = defaultdict(list)
    n_total = n_kept = 0
    with MEGA_4H_COHORT.open(encoding="utf-8") as f:
        for line in f:
            if '"ts":' not in line:
                continue
            n_total += 1
            try:
                ts = _line_ts(line)
            except Exception:
                ts = json.loads(line)["ts"]
            if ts // 86400000 not in keep:
                continue
            r = json.loads(line)
            n_kept += 1
            series[r["asset"]].append(r)
    for rows in series.values():
        rows.sort(key=lambda x: x["ts"])
    sampling = {"method": "every 4th distinct UTC date (epoch-day = "
                          "ts_ms//86400000, sorted, index%4==0); all 4h "
                          "bars of kept dates and all symbols retained",
                "n_dates_total": len(ordered),
                "n_dates_sampled": len(keep),
                "n_records_total": n_total,
                "n_records_sampled": n_kept}
    return series, sampling


def add_h4_scores(series):
    """m4_pct = per-asset mid-rank pct of m4 vs the trailing-180 SAMPLED
    records (MIN_WINDOW=60). Same sliding-window mechanics as T124 on
    the subsampled stream — the window covers ~4x the wall-clock span."""
    for rows in series.values():
        win = []
        for i, r in enumerate(rows):
            v = r["m4"]
            r["m4_pct"] = ((bisect_left(win, v) + bisect_right(win, v))
                           / (2.0 * len(win))
                           if v is not None and len(win) >= H4_MIN_WINDOW
                           else None)
            if v is not None:
                insort(win, v)
            if i >= H4_TRAIL:
                old = rows[i - H4_TRAIL]["m4"]
                if old is not None:
                    win.pop(bisect_left(win, old))


def _yr(ts_ms):
    return str(dt.datetime.fromtimestamp(ts_ms / 1000,
                                         dt.timezone.utc).year)


def run_mega_4h():
    series, sampling = load_mega_4h_sampled()
    add_h4_scores(series)

    # cell: pooled Spearman(m4_pct, fwd4) + yearly folds
    xs, ys, yrs = [], [], []
    for rows in series.values():
        for r in rows:
            if r["m4_pct"] is not None and r["fwd4"] is not None:
                xs.append(r["m4_pct"])
                ys.append(r["fwd4"])
                yrs.append(_yr(r["ts"]))
    sp = v1.spearman(xs, ys)
    fold_rho = {}
    for y in H4_YEARS:
        fx = [a for a, yy in zip(xs, yrs) if yy == y]
        fy = [b for b, yy in zip(ys, yrs) if yy == y]
        s = v1.spearman(fx, fy)
        fold_rho[y] = {"n": len(fx),
                       "rho": round(s["rho"], 5) if s else None,
                       "p": s["p"] if s else None}
    mom_cell = {"n_rows": len(xs),
                "spearman_pooled": ({k: round(v, 6)
                                     if isinstance(v, float) else v
                                     for k, v in sp.items()}
                                    if sp else None),
                "yearly_folds": fold_rho,
                "direction": "rho<0 = 4h reversal (T96/T124 prior)",
                "p": sp["p"] if sp else None}

    # cell: settlement window Welch on the assumed grid (all symbols)
    set_a, set_b = defaultdict(list), defaultdict(list)   # year -> fwd4
    for rows in series.values():
        for r in rows:
            if r["htsg"] is None or r["fwd4"] is None:
                continue
            (set_a if r["htsg"] <= H4_SETTLE_H
             else set_b)[_yr(r["ts"])].append(r["fwd4"])
    all_a = [v for vals in set_a.values() for v in vals]
    all_b = [v for vals in set_b.values() for v in vals]
    w = v1.welch(all_a, all_b)
    fold_w = {y: v1.welch(set_a.get(y, []), set_b.get(y, []))
              for y in H4_YEARS}
    settle_cell = {
        "definition": "htsg<=2 (4h bar closes on the assumed 00/08/16 "
                      "settlement grid; fwd4 covers the post-settlement "
                      "bar) vs htsg>2 — all sampled symbols",
        "welch_settle_vs_rest": (
            {"n_settle": w["n_a"], "n_rest": w["n_b"],
             "mean_settle_bps": round(w["mean_a"], 4),
             "mean_rest_bps": round(w["mean_b"], 4),
             "diff_bps": round(w["mean_a"] - w["mean_b"], 4),
             "t": round(w["t"], 4), "p": w["p"]} if w else None),
        "yearly_folds": {y: ({"diff_bps":
                             round(f["mean_a"] - f["mean_b"], 4),
                             "p": f["p"]} if f else None)
                         for y, f in fold_w.items()},
        "direction": "diff<0 = post-settlement fwd weaker (T93/T124 "
                     "sawtooth prior)",
        "p": w["p"] if w else None}

    cells = {"mom_4h_rho": mom_cell, "settlement_window": settle_cell}
    ps = sorted((m["p"], n) for n, m in cells.items()
                if m.get("p") is not None)
    M = len(ps)
    fdr = {n: p <= 0.05 * (i + 1) / M for i, (p, n) in enumerate(ps)}

    rho = mom_cell["spearman_pooled"]["rho"] \
        if mom_cell["spearman_pooled"] else None
    m_same = sum(1 for y in H4_YEARS
                 if fold_rho[y]["rho"] is not None and rho is not None
                 and fold_rho[y]["rho"] * rho > 0)
    m_gates = {"fdr_pass": fdr.get("mom_4h_rho", False),
               "reversal_direction_rho_neg": rho is not None
               and rho < 0,
               "folds_same_sign_ge_2of3": m_same >= 2}
    diff = (settle_cell["welch_settle_vs_rest"]["diff_bps"]
            if settle_cell["welch_settle_vs_rest"] else None)
    s_same = sum(1 for y in H4_YEARS
                 if settle_cell["yearly_folds"][y] is not None
                 and diff is not None
                 and settle_cell["yearly_folds"][y]["diff_bps"] * diff > 0)
    s_gates = {"fdr_pass": fdr.get("settlement_window", False),
               "post_settlement_negative": diff is not None
               and diff < 0,
               "folds_same_sign_ge_2of3": s_same >= 2}

    def v_of(gates, direction_ok):
        if all(gates.values()):
            return "PASS"
        if not direction_ok:
            return "FAIL"
        return "DIRECTIONAL_ONLY"

    verdicts = {
        "mom_4h_rho": {"gates": m_gates,
                       "folds_same_sign": f"{m_same}/3",
                       "verdict": v_of(m_gates,
                                       m_gates["reversal_direction_rho_neg"])},
        "settlement_window": {"gates": s_gates,
                              "folds_same_sign": f"{s_same}/3",
                              "verdict": v_of(
                                  s_gates,
                                  s_gates["post_settlement_negative"])}}

    return {
        "contract": {
            "cohort": str(MEGA_4H_COHORT.relative_to(ROOT)),
            "cohort_sha256": hashlib.sha256(
                MEGA_4H_COHORT.read_bytes()).hexdigest(),
            "sampling": sampling,
            "n_symbols_sampled": len(series),
            "score": "m4_pct = per-asset mid-rank pct of m4 vs "
                     "trailing-180 SAMPLED records, MIN_WINDOW=60 "
                     "(T124 mechanics on the subsample; each sampled "
                     "date contributes its 6 bars so the window covers "
                     "~120d wall-clock, not ~30d)",
            "settle_threshold_h": H4_SETTLE_H,
            "fdr_alpha": 0.05,
            "fdr_family": list(cells)},
        "cells": cells,
        "fdr_pass": fdr,
        "verdicts": verdicts,
        "caveats": [
            "SAMPLED stream: every 4th UTC date — numbers are a "
            "sample-scorecard cell, not the T124 headline",
            "all t/p nominal: overlapping fwd labels + shared market "
            "factor; effective dof far below n",
            "gross moves only — not a tradability claim"]}


# =========================================================== cross-scale
def cross_scale_v4(v3res, mega_xs, h4):
    daily_dfh = v3res["daily"]["dfh_cells"]
    xs = v3res["xs"]
    out = {"funding_pct_daily_vs_hourly":
           v2.cross_scale(v3res["daily"], v3res["intraday"])}

    # dfh20 on three scales: daily-5 pct rho, XS-10 spread, XS-277 spread
    d = daily_dfh["cells"]["dfh20_pct"]
    x10 = xs["cells"]["dfh20"]
    x277 = mega_xs["cells"]["dfh20"]
    vals = [d["rho"], x10["mean_spread_bps"], x277["mean_spread_bps"]]
    same = all(v is not None for v in vals) and (
        all(v > 0 for v in vals) or all(v < 0 for v in vals))
    all_pass = (daily_dfh["verdicts"]["dfh20_pct"]["verdict"] == "PASS"
                and xs["verdicts"]["dfh20"]["verdict"] == "PASS"
                and mega_xs["verdicts"]["dfh20"]["verdict"] == "PASS")
    out["dfh20_three_scales"] = {
        "cell": "dfh20",
        "daily_5asset": {"score": "dfh20_pct trailing-180 pct",
                         "rho": d["rho"], "p": d["p"], "n": d["n"],
                         "verdict":
                             daily_dfh["verdicts"]["dfh20_pct"]["verdict"]},
        "xs_10asset": {"mean_spread_bps": x10["mean_spread_bps"],
                       "p": x10["p"], "days": x10["days_evaluated"],
                       "verdict": xs["verdicts"]["dfh20"]["verdict"]},
        "xs_277asset": {"mean_spread_bps": x277["mean_spread_bps"],
                        "p": x277["p"], "days": x277["days_evaluated"],
                        "verdict": mega_xs["verdicts"]["dfh20"]["verdict"]},
        "same_sign_all_three": same,
        "replication": ("replicated_three_scales" if same and all_pass
                        else "directional_only" if same
                        else "not_replicated")}

    # mom horizon split: daily XS continuation vs 4h reversal
    d10 = xs["cells"]["mom20"]["mean_spread_bps"]
    d277 = mega_xs["cells"]["mom20"]["mean_spread_bps"]
    h_rho = v3res["intraday"]["cells"]["mom_4h"]["rho"]
    h4_rho = (h4["cells"]["mom_4h_rho"]["spearman_pooled"] or {}
              ).get("rho") if h4 else None
    cont_ok = d10 is not None and d277 is not None \
        and d10 > 0 and d277 > 0
    rev_ok = h_rho is not None and h4_rho is not None \
        and h_rho < 0 and h4_rho < 0
    out["mom_daily_continuation_vs_4h_reversal"] = {
        "cell": "momentum horizon split",
        "daily_continuation": {
            "xs_10asset_mom20_spread_bps": d10,
            "xs_277asset_mom20_spread_bps": d277,
            "sign": "positive = top-decile momentum continues 5d"},
        "h4_reversal": {
            "intraday_mom_4h_rho_5asset": h_rho,
            "mega_4h_sampled_rho": h4_rho,
            "verdict_4h": h4["verdicts"]["mom_4h_rho"]["verdict"]
            if h4 else None,
            "sign": "negative = last-4h winners mean-revert"},
        "horizon_split_confirmed": bool(cont_ok and rev_ok),
        "note": "same momentum family, opposite sign by horizon: "
                "continuation at daily/5d vs reversal at 4h — the "
                "mega-4h sampled cell replicates the frozen intraday "
                "mom_4h sign"}
    return out


# ------------------------------------------------------------------ run
def run():
    v3res = v3.run()                          # frozen blocks, verbatim
    series, ridge_rows, meta = load_mega()
    add_mega_scores(series)
    mega_xs = run_mega_xs(series, meta)
    ridge_model = run_ridge_model(ridge_rows)
    h4 = run_mega_4h() if MEGA_4H_COHORT.exists() else None
    cs = cross_scale_v4(v3res, mega_xs, h4)

    # ---- verification vs the frozen v3 receipt ----
    ver = {"v3_receipt": str(V3_RECEIPT.relative_to(ROOT)),
           "carried_v3_byte_identical_to_v3_receipt": None,
           "verdict_changes_vs_v3": [],
           "mega_cohort_sha256":
               mega_xs["contract"]["cohort_sha256"],
           "mega_cohort_n_records": meta["n_records"],
           "mega_cohort_n_symbols": len(meta["assets"]),
           "mega_4h_cohort_sha256":
               h4["contract"]["cohort_sha256"] if h4 else None,
           "mega_4h_sampling":
               h4["contract"]["sampling"] if h4 else "SKIPPED",
           "ridge_machinery": ("imported financial_signal_ridge_v2_v1"
                               if ridge else
                               f"unavailable: {RIDGE_IMPORT_ERROR}")}
    if V3_RECEIPT.exists():
        old = json.loads(V3_RECEIPT.read_text())
        old_run = {k: v for k, v in old.items()
                   if k not in ("determinism", "generated_at")}
        ver["carried_v3_byte_identical_to_v3_receipt"] = (
            json.dumps(v3res, sort_keys=True)
            == json.dumps(old_run, sort_keys=True))
        for block_name, new_v, old_v in (
                ("daily", v3res["daily"]["verdicts"],
                 old["daily"]["verdicts"]),
                ("daily_dfh", v3res["daily"]["dfh_cells"]["verdicts"],
                 old["daily"]["dfh_cells"]["verdicts"]),
                ("intraday", v3res["intraday"]["verdicts"],
                 old["intraday"]["verdicts"]),
                ("xs", v3res["xs"]["verdicts"], old["xs"]["verdicts"])):
            for cell, vv in new_v.items():
                ov = old_v.get(cell, {}).get("verdict")
                if ov != vv["verdict"]:
                    ver["verdict_changes_vs_v3"].append(
                        {"block": block_name, "cell": cell,
                         "v3": ov, "v4": vv["verdict"]})
    else:
        ver["note"] = "v3 receipt not found — byte-identity check skipped"

    blocks = [("daily", v3res["daily"]["verdicts"]),
              ("daily_dfh", v3res["daily"]["dfh_cells"]["verdicts"]),
              ("intraday", v3res["intraday"]["verdicts"]),
              ("xs", v3res["xs"]["verdicts"]),
              ("mega_xs", mega_xs["verdicts"]),
              ("ridge_model", ridge_model["verdicts"]),
              ("mega_4h", h4["verdicts"] if h4 else {})]
    summary = v3.verdict_counts(*blocks)

    return {"schema_version": "nanojev-financial-signal-benchmark-v4",
            "status": "benchmark_complete",
            "task": "T133",
            "protocol": "extends v3 in-place: carried daily/dfh/"
                        "intraday/xs blocks are v3.run() verbatim; new "
                        "mega_xs (277-asset decile), ridge_model "
                        "(ridge_min3 refit) and mega_4h (sampled) blocks "
                        "are separate FDR families",
            "fdr_family_note": "seven separate BH-FDR families at "
                               "alpha=0.05: 10 frozen daily, 1 daily "
                               "dfh, 7 intraday, 3 XS-10, 4 mega-XS-277, "
                               "1 ridge cell (gate verdict, no p), 2 "
                               "mega-4h sampled — no cross-scale pooling",
            "carried_v3": v3res,
            "mega_xs": mega_xs,
            "ridge_model": ridge_model,
            "mega_4h": h4,
            "cross_scale": cs,
            "summary_verdict_counts": summary,
            "verification": ver}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    t0 = time.time()
    r1, r2 = run(), run()
    same = json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    r1["determinism"] = {"replays": 2, "byte_identical": same}
    r1["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    blob = json.dumps(r1, indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(blob)
    brief = {"output": str(args.output), "deterministic": same,
             "elapsed_s": round(time.time() - t0, 1),
             "sha256": hashlib.sha256(blob.encode()).hexdigest(),
             "v3_byte_identical":
                 r1["verification"]
                 ["carried_v3_byte_identical_to_v3_receipt"],
             "verdict_changes_vs_v3":
                 r1["verification"]["verdict_changes_vs_v3"]}
    for block in ("mega_xs", "ridge_model", "mega_4h"):
        if r1.get(block) and "verdicts" in r1[block]:
            brief[block] = {k: v["verdict"]
                            for k, v in r1[block]["verdicts"].items()}
    for name in MEGA_ARMS:
        c = r1["mega_xs"]["cells"][name]
        brief.setdefault("mega_xs_spreads_bps", {})[name] = {
            "mean": c["mean_daily_spread_bps"]["mean"],
            "t": c["mean_daily_spread_bps"]["t"],
            "expected": c["t112_expected_bps"]}
    print(json.dumps(brief, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
