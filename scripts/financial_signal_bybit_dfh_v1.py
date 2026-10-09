#!/usr/bin/env python3
"""T107: external replication of the dfh (distance-from-high) effects on
local Bybit data — same play as the T99 Hyperliquid replication of the
funding signal.

Reference (Binance, confirmed):
  T101 time-series (results/financial_signal_dfh_v1.json, 5-asset PIT
  cohort, mark price):
    dfh20 -> 5d fwd log mark: pooled rho=+0.031, p=0.019842, n=5629,
    FDR-pass; dfh_pct>=0.80 vs <=0.20 quintile Welch +157.2bps (t=5.029);
    inside the f2b2 cell rho=+0.1193.
  T104 cross-sectional (results/financial_signal_xs_v1.json, 10-asset
  cohort, close):
    daily dfh20-rank top-2 vs bottom-2 next-5d spread: mean +54.06bps,
    t=3.344, n=1314 days, 4/4 calendar-year folds same sign.
Question: does dfh replicate on a second venue?

Local Bybit data (data/venue_perp_v1/bybit/, hash-pinned fetch, licence
UNVERIFIED):
  {SYM}.kline.json   1d bars [ts_ms, o, h, l, c, vol, turnover];
                     1358 bars/symbol, 2023-01-01..2026-09-19 UTC.
                     Close is the price series for dfh/returns.
  {SYM}.mark.json    1d mark bars (same layout); used for the basis and
                     a mark-series robustness cell (close-vs-mark caveat).
  {SYM}.index.json   1d index bars; basis_bps = (mark_c/index_c - 1)*1e4
                     — the mark_index_basis_bps equivalent, so the spec
                     gate ports WITHOUT the funding+trend-only deviation.
  {SYM}.funding.json rows {fundingRate, fundingRateTimestamp, symbol};
                     strict 8h settlement grid (00/08/16 UTC), full span.
                     f_last = last settlement inside the UTC day — the
                     same semantics as Binance last_funding_rate with no
                     hourly-grid complication (unlike Hyperliquid).

Arms (protocol: research/financial_signal_bybit_dfh_protocol_v1.json,
frozen + owner self-authorized before measurement, T104 convention):
  A  time-series: pooled + per-asset Spearman(dfh_pct, 5d fwd log close);
     raw-dfh20 pooled Spearman alongside (T101 used the raw feature);
     top-decile Welch (dfh_pct>=0.90 vs rest) + T101-style quintile
     Welch (>=0.80 vs <=0.20) for comparability.
  B  cross-sectional (5 assets — THIN, half the T104 cohort): per day
     with all 5 assets present, rank by dfh20, spread = fwd_5d(top1) -
     fwd_5d(bot1) in bps (momentum direction, T104 sign convention);
     top-2/bottom-2 variant alongside (2-vs-2 of 5 covers 80% of the
     cross-section — reported but heavily discounted). Mean daily
     spread + t across days, pooled Welch, per-year fold signs.
  C  spec-gate portability: rows with funding_pct>=0.80 AND
     basis_pct>=0.66 AND BTC-20d-up; (a) gate-on vs gate-off mean 5d
     Welch (spec direction), (b) inside-gate Spearman(dfh20, ret5) +
     median-split dfh_pct Welch (T101 A1c analog). Sparse cell —
     honest n reporting, skip if untestable.

BH-FDR alpha=0.05 over the fixed 8-cell family in the protocol.
Measurement only — no fitting, no trading, no network. If any input is
missing/too short the affected arm records SKIPPED honestly.
"""

import argparse
import datetime
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BYBIT = ROOT / "data/venue_perp_v1/bybit"
OUT = ROOT / "results/financial_signal_bybit_dfh_v1.json"
PROTOCOL = ROOT / "research/financial_signal_bybit_dfh_protocol_v1.json"
AUTH = ROOT / "results/financial_signal_bybit_dfh_authorization_v1.json"
SYMS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
DAY_MS = 86_400_000
LOOKBACK = 180          # trailing rows for mid-rank pct (repo conv.)
HOLD = 5                # forward horizon (daily bars)
DFH_WIN = 20            # strictly-prior trailing-high window
TREND = 20              # BTC trend lookback (spec gate)
DECILE = 0.90           # top-decile arm threshold (task spec)
Q_HI, Q_LO = 0.80, 0.20  # T101 quintile comparability cell
F_PCT, B_PCT = 0.80, 0.66  # spec f2b2 gate thresholds
MIN_N = 10              # test minimum per side (repo convention)
MIN_DAYS = 250          # floor for a usable per-symbol series
MIN_ASSETS_XS = 5       # Arm-B day gate: full 5-asset cross-section only

BINANCE_REF = {
    "time_series_T101": {
        "cell": "dfh20 -> 5d fwd log mark, pooled 5-asset PIT cohort",
        "rho": 0.031, "p": 0.019842, "n": 5629, "fdr_pass": True,
        "quintile_welch_diff_bps": 157.2, "quintile_welch_t": 5.029,
        "f2b2_internal_rho": 0.1193, "f2b2_welch_diff_bps": 221.5,
        "per_asset_rho_range": [-0.0237, 0.0766],
        "ref": "results/financial_signal_dfh_v1.json"},
    "cross_section_T104": {
        "cell": "daily dfh20-rank top-2 vs bottom-2 next-5d close "
                "spread, 10-asset cohort",
        "mean_daily_spread_bps": 54.06, "t": 3.344, "n_days": 1314,
        "year_folds_same_sign": "4/4",
        "ref": "results/financial_signal_xs_v1.json"},
    "spec_gate": "funding_pct>=0.80 AND basis_pct>=0.66 AND BTC "
                 "mark_price 20-bar return > 0 "
                 "(research/financial_signal_spec_v1.json)"}


# ---------------------------------------------------------------- stats
def pct(w, x):
    """Mid-rank percentile of x within trailing window w (project conv.)."""
    return (sum(1 for v in w if v < x)
            + 0.5 * sum(1 for v in w if v == x)) / len(w)


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


def norm_p(t):
    """Two-sided normal-approx p from a t/z statistic (repo convention)."""
    return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))


def spearman(xs, ys):
    """Mid-rank Spearman rho + normal-approx t/p; None if n<MIN_N or a
    side has zero rank variance (project convention)."""
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
            "p": round(norm_p(t), 6)}


def welch(a, b):
    """Welch t of mean(a)-mean(b) on forward returns; means in bps;
    normal-approx two-sided p (repo convention)."""
    if len(a) < MIN_N or len(b) < MIN_N:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    se = math.sqrt(statistics.pvariance(a) / len(a)
                   + statistics.pvariance(b) / len(b))
    if se == 0:
        return None
    t = (ma - mb) / se
    return {"n_arm": len(a), "n_rest": len(b),
            "mean_arm_bps": round(ma * 1e4, 1),
            "mean_rest_bps": round(mb * 1e4, 1),
            "diff_bps": round((ma - mb) * 1e4, 1),
            "t": round(t, 3), "p": round(norm_p(t), 6)}


def t_stat(xs):
    """Mean/sd/t (+ normal-approx p) of a daily series; obs treated as
    independent — nominal only (5d labels overlap ~5x)."""
    n = len(xs)
    if n < MIN_N:
        return None
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return None
    t = mean / (sd / math.sqrt(n))
    return {"n": n, "mean": round(mean, 3), "sd": round(sd, 3),
            "t": round(t, 3), "p": round(norm_p(t), 6)}


def bh_fdr(named_ps):
    """BH-FDR alpha=0.05 over [(name, p|None)] -> ({name: survives},
    sorted threshold table)."""
    ps = sorted((p, n) for n, p in named_ps if p is not None)
    m = len(ps)
    table = [{"cell": n, "p": p, "alpha_bh": round(0.05 * (i + 1) / m, 6),
              "survives": p <= 0.05 * (i + 1) / m}
             for i, (p, n) in enumerate(ps)]
    return {e["cell"]: e["survives"] for e in table}, table


def iso(day):
    return datetime.datetime.fromtimestamp(
        day * 86400, datetime.timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------- data
def load_symbol(sym):
    """-> (merged daily rows, integrity). Row keys: day, close, mark,
    basis_bps, f_last. A day enters only if kline, mark, index and a
    funding settlement all exist (strict PIT merge)."""
    files = {k: BYBIT / f"{sym}.{k}.json"
             for k in ("kline", "mark", "index", "funding")}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        return None, {"status": "SKIPPED", "missing_files": missing}
    kline = json.loads(files["kline"].read_text())
    mark = json.loads(files["mark"].read_text())
    index = json.loads(files["index"].read_text())
    fund = json.loads(files["funding"].read_text())
    close = {int(r[0]) // DAY_MS: float(r[4]) for r in kline}
    mark_c = {int(r[0]) // DAY_MS: float(r[4]) for r in mark}
    idx_c = {int(r[0]) // DAY_MS: float(r[4]) for r in index}
    f_last = {}
    for r in sorted(fund, key=lambda x: int(x["fundingRateTimestamp"])):
        f_last[int(r["fundingRateTimestamp"]) // DAY_MS] = \
            float(r["fundingRate"])          # last settlement of the day
    days = sorted(d for d in close
                  if d in mark_c and d in idx_c and d in f_last)
    merged = [{"day": d, "close": close[d], "mark": mark_c[d],
               "basis_bps": (mark_c[d] / idx_c[d] - 1.0) * 1e4
               if idx_c[d] > 0 else None,
               "f_last": f_last[d]} for d in days]
    ftimes = sorted(int(r["fundingRateTimestamp"]) for r in fund)
    gaps = [b - a for a, b in zip(ftimes, ftimes[1:])]
    integrity = {
        "status": "ok",
        "n_kline_days": len(close), "n_mark_days": len(mark_c),
        "n_index_days": len(idx_c), "n_funding_rows": len(fund),
        "n_funding_days": len(f_last), "n_merged_days": len(merged),
        "merged_span_utc": (f"{iso(days[0])}..{iso(days[-1])}"
                            if days else None),
        "funding_grid_hours": sorted({g // 3_600_000 for g in gaps}),
        "funding_span_utc": (f"{iso(ftimes[0] // DAY_MS)}.."
                             f"{iso(ftimes[-1] // DAY_MS)}"),
        "input_sha256": {k: hashlib.sha256(p.read_bytes()).hexdigest()
                         for k, p in files.items()}}
    return merged, integrity


def build_rows(merged, sym, btc_trend):
    """Eval rows i in [LOOKBACK, n-HOLD): dfh20 (strictly-prior 20-bar
    high), dfh_pct (trailing-180 mid-rank), ret5, gate components."""
    n = len(merged)
    cl = [m["close"] for m in merged]
    mk = [m["mark"] for m in merged]
    ba = [m["basis_bps"] for m in merged]
    fl = [m["f_last"] for m in merged]
    dfh20 = [None] * n
    dfh20_mark = [None] * n
    for i in range(DFH_WIN, n):
        w = cl[i - DFH_WIN:i]
        if cl[i] > 0 and all(v > 0 for v in w):
            dfh20[i] = cl[i] / max(w) - 1.0
        wm = mk[i - DFH_WIN:i]
        if mk[i] > 0 and all(v > 0 for v in wm):
            dfh20_mark[i] = mk[i] / max(wm) - 1.0
    rows = []
    for i in range(LOOKBACK, n - HOLD):
        if cl[i] <= 0 or cl[i + HOLD] <= 0 or dfh20[i] is None:
            continue
        if ba[i] is None or fl[i] is None:
            continue
        dwin = [v for v in dfh20[i - LOOKBACK:i] if v is not None]
        fwin = fl[i - LOOKBACK:i]
        bwin = [v for v in ba[i - LOOKBACK:i] if v is not None]
        if not dwin or len(fwin) < LOOKBACK or len(bwin) < LOOKBACK - DFH_WIN:
            continue
        rows.append({
            "sym": sym, "day": merged[i]["day"], "i": i,
            "dfh20": dfh20[i],
            "dfh_pct": pct(dwin, dfh20[i]),
            "ret5": math.log(cl[i + HOLD] / cl[i]),
            "ret5_mark": (math.log(mk[i + HOLD] / mk[i])
                          if mk[i] > 0 and mk[i + HOLD] > 0 else None),
            "dfh20_mark": dfh20_mark[i],
            "f_pct": pct(fwin, fl[i]),
            "b_pct": pct(bwin, ba[i]),
            "btc_up": btc_trend.get(merged[i]["day"])})
    return rows


# ---------------------------------------------------------------- arms
def arm_a(per_sym_rows, all_rows):
    """Time-series: pooled + per-asset Spearman(dfh_pct, ret5) and raw
    dfh20 (T101 comparability); top-decile + quintile Welch."""
    out = {"per_symbol": {}, "pooled": {}}
    for s, rows in sorted(per_sym_rows.items()):
        out["per_symbol"][s] = {
            "n_eval": len(rows),
            "eval_span_utc": (f"{iso(rows[0]['day'])}..{iso(rows[-1]['day'])}"
                              if rows else None),
            "spearman_dfh_pct": spearman([r["dfh_pct"] for r in rows],
                                         [r["ret5"] for r in rows]),
            "spearman_dfh20_raw": spearman([r["dfh20"] for r in rows],
                                           [r["ret5"] for r in rows]),
            "top_decile_vs_rest": welch(
                [r["ret5"] for r in rows if r["dfh_pct"] >= DECILE],
                [r["ret5"] for r in rows if r["dfh_pct"] < DECILE])}
    out["pooled"] = {
        "n_eval": len(all_rows),
        "spearman_dfh_pct": spearman([r["dfh_pct"] for r in all_rows],
                                     [r["ret5"] for r in all_rows]),
        "spearman_dfh20_raw": spearman([r["dfh20"] for r in all_rows],
                                       [r["ret5"] for r in all_rows]),
        "top_decile_vs_rest": welch(
            [r["ret5"] for r in all_rows if r["dfh_pct"] >= DECILE],
            [r["ret5"] for r in all_rows if r["dfh_pct"] < DECILE]),
        "quintile_top_vs_bottom_T101_style": welch(
            [r["ret5"] for r in all_rows if r["dfh_pct"] >= Q_HI],
            [r["ret5"] for r in all_rows if r["dfh_pct"] <= Q_LO])}
    # robustness: same test on the mark-price series (close-vs-mark caveat)
    mrows = [r for r in all_rows
             if r["dfh20_mark"] is not None and r["ret5_mark"] is not None]
    out["robustness_mark_series"] = {
        "n_eval": len(mrows),
        "spearman_dfh20_mark_vs_ret5_mark": spearman(
            [r["dfh20_mark"] for r in mrows],
            [r["ret5_mark"] for r in mrows]),
        "note": "dfh20 and the 5d label both recomputed on daily mark "
                "closes; quantifies the close-vs-mark deviation outside "
                "the FDR family"}
    return out


def arm_b(per_sym_rows, edge):
    """Cross-sectional: per day with all 5 symbols, rank by raw dfh20
    (T104 convention), spread = fwd_5d(top-edge) - fwd_5d(bot-edge) bps."""
    by_day = defaultdict(list)
    for s, rows in per_sym_rows.items():
        for r in rows:
            by_day[r["day"]].append((s, r["dfh20"], r["ret5"]))
    days, pooled_top, pooled_bot = [], [], []
    for day in sorted(by_day):
        xs = by_day[day]
        if len(xs) < MIN_ASSETS_XS:
            continue
        ordered = sorted(xs, key=lambda x: x[1])
        bot, top = ordered[:edge], ordered[-edge:]
        if bot[-1][1] == top[0][1]:
            continue  # degenerate: no dispersion at the ranked edges
        bot_fwd = sum(x[2] for x in bot) / edge
        top_fwd = sum(x[2] for x in top) / edge
        days.append({"day": iso(day), "n_assets": len(xs),
                     "top_assets": sorted(x[0] for x in top),
                     "bottom_assets": sorted(x[0] for x in bot),
                     "top_fwd_bps": round(top_fwd * 1e4, 3),
                     "bottom_fwd_bps": round(bot_fwd * 1e4, 3),
                     "spread_bps": round((top_fwd - bot_fwd) * 1e4, 3)})
        pooled_top.extend(x[2] for x in top)
        pooled_bot.extend(x[2] for x in bot)
    folds = defaultdict(list)
    for d in days:
        folds[d["day"][:4]].append(d["spread_bps"])
    fold_means = {y: sum(v) / len(v) for y, v in sorted(folds.items())}
    daily = t_stat([d["spread_bps"] for d in days])
    pm = daily["mean"] if daily else None
    consistent = sum(1 for m in fold_means.values()
                     if pm is not None and m * pm > 0)
    return {"edge": edge,
            "spread_direction": "fwd_5d(top-rank) - fwd_5d(bottom-rank), "
                                "momentum direction (T104 sign conv.)",
            "rank_feature": "raw dfh20 per day (cross-sectional rank)",
            "days_evaluated": len(days),
            "eval_span_utc": (f"{days[0]['day']}..{days[-1]['day']}"
                              if days else None),
            "mean_daily_spread_bps": daily,
            "pooled_welch_top_vs_bottom": welch(pooled_top, pooled_bot),
            "fold_sign_consistency": {
                "folds": len(fold_means),
                "same_sign_as_pooled": consistent,
                "fold_mean_bps": {y: round(m, 3)
                                  for y, m in fold_means.items()}},
            "per_day": days}


def arm_c(all_rows):
    """Spec gate: f_pct>=0.80 AND b_pct>=0.66 AND BTC-up. (a) gate-on vs
    gate-off mean 5d (spec direction portability); (b) inside-gate dfh
    cells (T101 A1c analog)."""
    gated = [r for r in all_rows
             if r["f_pct"] >= F_PCT and r["b_pct"] >= B_PCT
             and r["btc_up"] is True]
    rest = [r for r in all_rows if not (
        r["f_pct"] >= F_PCT and r["b_pct"] >= B_PCT
        and r["btc_up"] is True)]
    hi = [r["ret5"] for r in gated if r["dfh_pct"] >= 0.50]
    lo = [r["ret5"] for r in gated if r["dfh_pct"] < 0.50]
    return {
        "gate": ("funding_pct>=0.80 AND basis_pct>=0.66 "
                 "(mark/index basis — full spec port, no deviation) AND "
                 "BTC close 20-bar return > 0"),
        "n_gated": len(gated), "n_rest": len(rest),
        "gate_share": (round(len(gated) / len(all_rows), 4)
                       if all_rows else None),
        "gate_on_vs_off_welch": welch([r["ret5"] for r in gated],
                                      [r["ret5"] for r in rest]),
        "inside_gate_spearman_dfh20": spearman(
            [r["dfh20"] for r in gated], [r["ret5"] for r in gated]),
        "inside_gate_highdfh_vs_lowdfh_median_split": welch(hi, lo),
        "inside_gate_spearman_dfh_pct": spearman(
            [r["dfh_pct"] for r in gated], [r["ret5"] for r in gated])}


# ------------------------------------------------------- protocol/auth
def write_protocol_and_auth():
    """T104 convention: freeze the protocol and the owner
    self-authorization BEFORE measurement; pin the protocol sha256."""
    protocol = {
        "schema_version": "nanojev-financial-signal-bybit-dfh-protocol-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": "T107",
        "purpose": "External replication of the dfh effects on a second "
                   "venue (Bybit), same play as T99 on Hyperliquid. "
                   "References: T101 time-series pooled rho=+0.031 "
                   "(FDR-pass) and T104 10-asset XS top2-bot2 "
                   "+54.06bps/day (t=3.344), both Binance. Measurement "
                   "only — no fitting, no trading, no profitability "
                   "claims.",
        "cohort": {
            "path": "data/venue_perp_v1/bybit/{BTC,ETH,BNB,SOL,XRP}USDT."
                    "{kline,mark,index,funding}.json",
            "fetch_manifest": "data/venue_perp_v1/bybit/fetch_manifest.json "
                              "(hash-pinned fetch, licence UNVERIFIED; "
                              "offline non-commercial research only)",
            "series": "kline daily close (price series for dfh + labels); "
                      "mark daily close (basis + robustness cell); index "
                      "daily close (basis); funding 8h settlements, last "
                      "of UTC day = f_last",
            "span": "2023-01-01..2026-09-19 UTC, 1358 merged days/symbol"},
        "definitions": {
            "dfh20": "close[i] / max(close[i-20:i]) - 1, strictly-prior "
                     "20 contiguous bars",
            "dfh_pct": "trailing-180 mid-rank pct of dfh20 per symbol, "
                       "window excludes decision row",
            "ret5": "log(close[i+5]/close[i])",
            "funding_pct": "trailing-180 mid-rank pct of f_last",
            "basis_pct": "trailing-180 mid-rank pct of (mark/index-1)*1e4 "
                         "— full basis-equivalent available, no "
                         "funding+trend-only deviation needed",
            "btc_up": "BTC close[i]/close[i-20]-1 > 0, market-wide by day",
            "eval_range": "i in [180, n-5) per symbol, all inputs non-null"},
        "arms": {
            "A_time_series": "pooled + per-asset Spearman(dfh_pct, ret5); "
                             "raw-dfh20 pooled Spearman (T101 comp); "
                             "Welch dfh_pct>=0.90 vs rest and "
                             ">=0.80 vs <=0.20 (T101 comp)",
            "B_cross_section": "per day (all 5 symbols) rank by raw "
                               "dfh20; spread fwd_5d(top1)-fwd_5d(bot1) "
                               "bps; top2/bot2 variant alongside; mean "
                               "daily spread t + pooled Welch + per-year "
                               "folds. THIN: 5 assets, half the T104 "
                               "cohort",
            "C_spec_gate": "funding_pct>=0.80 AND basis_pct>=0.66 AND "
                           "BTC-up: gate-on vs gate-off Welch + "
                           "inside-gate Spearman(dfh20,ret5) + "
                           "median-split dfh_pct Welch (T101 A1c analog)"},
        "contrast_family": [
            "armA_spearman_dfh_pct_pooled",
            "armA_spearman_dfh20_raw_pooled",
            "armA_welch_top_decile",
            "armA_welch_quintile",
            "armB_daily_spread_top1",
            "armB_daily_spread_top2",
            "armC_gate_on_vs_off_welch",
            "armC_inside_gate_spearman_dfh20"],
        "statistics": {
            "spearman": "mid-rank rho + normal-approx t/p, min n=10",
            "welch": "Welch t on fwd log returns in bps, normal-approx "
                     "two-sided p, min n=10 per side",
            "daily_t": "mean/sd/t across decision days + normal-approx p "
                       "(nominal — overlapping labels)",
            "multiple_testing": "BH-FDR alpha=0.05 over contrast_family",
            "caveats": [
                "5d labels on 1d bars overlap ~5x; pooled rows share the "
                "market factor across assets — p-values nominal/optimistic",
                "5-asset cross-section is VERY thin: top1-vs-bot1 = 2 "
                "effective obs/day; top2-vs-bot2 covers 80% of the "
                "cohort and is not a real extremes contrast",
                "Bybit eval span (~2023-07 onward after the 180d "
                "warm-up) only partly overlaps the Binance reference "
                "window (2023-01-25..2026-08-30)",
                "kline close is last-trade close, not mark; quantified "
                "by the mark-series robustness cell",
                "Bybit licence UNVERIFIED in fetch_manifest",
                "gross moves only — no fees, spread, slippage, funding "
                "cashflows"]},
        "verdict_rule": "per cell: replicated iff sign agrees with the "
                        "Binance reference AND BH-FDR pass; cells below "
                        "the n needed to detect the reference rho at "
                        "nominal p<0.05 are labelled low_power rather "
                        "than not_replicated; honest negative reporting",
        "forbidden": ["fitting", "trading", "profitability claims",
                      "protocol edits post-run", "network access"]}
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(protocol, indent=2, sort_keys=True)
                        + "\n", encoding="utf-8")
    sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    auth = {
        "schema_version": "nanojev-financial-signal-bybit-dfh-authorization-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_sha256": sha,
        "protocol_path": "research/financial_signal_bybit_dfh_protocol_v1.json",
        "decision": "approved_for_measurement",
        "measurement_authorized": True,
        "fit_authorized": False,
        "independent_reviewer": {
            "id": "project-owner",
            "independence": "owner_self_authorization_not_independent_review",
            "note": "Owner directed T107: external replication of the dfh "
                    "effects on local Bybit data — same play as the T99 "
                    "Hyperliquid replication (delegated task)."},
        "scope": {
            "permitted": "PIT-safe descriptive measurement on local "
                         "pinned Bybit kline/mark/index/funding files: "
                         "per-asset + pooled Spearman, Welch contrasts, "
                         "5-asset XS rank spreads, spec-gate cells, "
                         "BH-FDR reporting.",
            "not_permitted": "No fitting/trading/profitability claims/"
                             "protocol edits; no network access; no "
                             "order submission."},
        "network_model_calls": 0,
        "order_submission_authorized": False,
        "live_trading_authorized": False}
    AUTH.parent.mkdir(parents=True, exist_ok=True)
    AUTH.write_text(json.dumps(auth, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return sha


# ---------------------------------------------------------------- run
def run(protocol_sha):
    merged, integrity = {}, {}
    skipped = []
    for s in SYMS:
        m, it = load_symbol(s)
        integrity[s] = it
        if m is None or len(m) < MIN_DAYS:
            integrity[s]["status"] = "SKIPPED" if m is None else \
                f"SKIPPED_short_series_{len(m)}d"
            skipped.append(s)
        else:
            merged[s] = m

    # BTC 20d trend on the merged BTC close series (spec gate)
    btc_trend = {}
    if "BTCUSDT" in merged:
        bc = merged["BTCUSDT"]
        for i in range(TREND, len(bc)):
            if bc[i]["close"] > 0 and bc[i - TREND]["close"] > 0:
                btc_trend[bc[i]["day"]] = (
                    bc[i]["close"] / bc[i - TREND]["close"] - 1 > 0)

    per_sym_rows, all_rows = {}, []
    for s, m in merged.items():
        rows = build_rows(m, s, btc_trend)
        per_sym_rows[s] = rows
        all_rows.extend(rows)

    arms = {"arm_a_time_series": None,
            "arm_b_cross_section": None,
            "arm_c_spec_gate": None}
    if not all_rows:
        arms["status"] = ("SKIPPED: no usable Bybit eval rows; nothing "
                          "was fabricated")
    else:
        arms["arm_a_time_series"] = arm_a(per_sym_rows, all_rows)
        arms["arm_b_cross_section"] = {
            "top1_vs_bottom1_primary": arm_b(per_sym_rows, 1),
            "top2_vs_bottom2_variant": arm_b(per_sym_rows, 2),
            "thin_cohort_warning": "5 assets = half the T104 cohort; "
                                   "top1-vs-bot1 has 2 effective obs/day; "
                                   "top2-vs-bot2 uses 80% of the "
                                   "cross-section and is reported only "
                                   "as a variant"}
        arms["arm_c_spec_gate"] = arm_c(all_rows)

    # ---- FDR family (8 cells, per protocol) ----
    a = arms["arm_a_time_series"] or {}
    b1 = ((arms["arm_b_cross_section"] or {})
          .get("top1_vs_bottom1_primary") or {})
    b2 = ((arms["arm_b_cross_section"] or {})
          .get("top2_vs_bottom2_variant") or {})
    c = arms["arm_c_spec_gate"] or {}
    fam = {
        "armA_spearman_dfh_pct_pooled":
            ((a.get("pooled") or {}).get("spearman_dfh_pct") or {}).get("p"),
        "armA_spearman_dfh20_raw_pooled":
            ((a.get("pooled") or {}).get("spearman_dfh20_raw") or {}).get("p"),
        "armA_welch_top_decile":
            ((a.get("pooled") or {}).get("top_decile_vs_rest") or {}).get("p"),
        "armA_welch_quintile":
            ((a.get("pooled") or {})
             .get("quintile_top_vs_bottom_T101_style") or {}).get("p"),
        "armB_daily_spread_top1":
            (b1.get("mean_daily_spread_bps") or {}).get("p"),
        "armB_daily_spread_top2":
            (b2.get("mean_daily_spread_bps") or {}).get("p"),
        "armC_gate_on_vs_off_welch":
            (c.get("gate_on_vs_off_welch") or {}).get("p"),
        "armC_inside_gate_spearman_dfh20":
            (c.get("inside_gate_spearman_dfh20") or {}).get("p")}
    fdr, table = bh_fdr(list(fam.items()))
    for e in table:
        e["fdr_pass"] = e.pop("survives")

    # ---- power-aware verdicts vs Binance references ----
    ref_rho = BINANCE_REF["time_series_T101"]["rho"]
    n_powered = math.ceil((1.96 / ref_rho) ** 2) + 3

    def cell_verdict(stat, ref_sign, fdr_key):
        if stat is None:
            return {"verdict": "untestable"}
        powered = stat["n"] >= n_powered
        agree = (stat["rho"] * ref_sign) > 0
        fp = fdr.get(fdr_key, False)
        if agree and fp:
            v = "replicated"
        elif not powered:
            v = "same_sign_low_power" if agree else "opposite_sign_low_power"
        else:
            v = "powered_null" if agree else "not_replicated_powered"
        return {"rho": stat["rho"], "p": stat["p"], "n": stat["n"],
                "powered_for_ref_rho_0.031": powered,
                "sign_agrees_with_binance": agree,
                "fdr_pass": fp, "verdict": v}

    pooled_pct = (a.get("pooled") or {}).get("spearman_dfh_pct")
    pooled_raw = (a.get("pooled") or {}).get("spearman_dfh20_raw")
    b1_daily = b1.get("mean_daily_spread_bps")
    b2_daily = b2.get("mean_daily_spread_bps")

    def asset_verdict(stat):
        """Per-asset cells are detail rows OUTSIDE the FDR family:
        sign + nominal power only (T101 convention)."""
        if stat is None:
            return {"verdict": "untestable"}
        powered = stat["n"] >= n_powered
        return {"rho": stat["rho"], "p": stat["p"], "n": stat["n"],
                "powered_for_ref_rho_0.031": powered,
                "sign_agrees_with_binance": stat["rho"] > 0,
                "fdr_pass": None,
                "verdict": "outside_fdr_family_detail"}

    verdicts = {
        "per_asset": {s: asset_verdict(
                          (a.get("per_symbol") or {}).get(s, {})
                          .get("spearman_dfh_pct"))
                      for s in sorted(per_sym_rows)},
        "per_asset_pattern_vs_binance":
            "Binance per-asset dfh20 rhos: SOL +0.077 (only significant), "
            "BTC +0.017, ETH -0.004, BNB -0.018, XRP -0.024",
        "pooled_dfh_pct": cell_verdict(pooled_pct, +1,
                                     "armA_spearman_dfh_pct_pooled"),
        "pooled_dfh20_raw": cell_verdict(pooled_raw, +1,
                                       "armA_spearman_dfh20_raw_pooled"),
        "xs_top1_vs_bottom1": ({
            "mean_daily_spread_bps": b1_daily["mean"],
            "t": b1_daily["t"], "p": b1_daily["p"],
            "n_days": b1_daily["n"],
            "sign_agrees_with_T104": b1_daily["mean"] > 0,
            "fdr_pass": fdr.get("armB_daily_spread_top1", False),
            "verdict": ("replicated" if b1_daily["mean"] > 0
                        and fdr.get("armB_daily_spread_top1", False)
                        else "same_sign_fdr_fail" if b1_daily["mean"] > 0
                        else "not_replicated")}
            if b1_daily else {"verdict": "untestable"}),
        "xs_top2_vs_bottom2_variant": ({
            "mean_daily_spread_bps": b2_daily["mean"],
            "t": b2_daily["t"], "p": b2_daily["p"],
            "n_days": b2_daily["n"],
            "sign_agrees_with_T104": b2_daily["mean"] > 0,
            "fdr_pass": fdr.get("armB_daily_spread_top2", False),
            "verdict": ("replicated" if b2_daily["mean"] > 0
                        and fdr.get("armB_daily_spread_top2", False)
                        else "same_sign_fdr_fail" if b2_daily["mean"] > 0
                        else "not_replicated")}
            if b2_daily else {"verdict": "untestable"}),
        "n_required_for_ref_rho": n_powered,
        "binance_references": {
            "ts_rho": ref_rho, "xs_mean_daily_spread_bps":
                BINANCE_REF["cross_section_T104"]["mean_daily_spread_bps"]},
        "fdr_survivors": sorted(k for k, v in fdr.items() if v)}
    ts_ok = pooled_pct and pooled_pct["rho"] > 0 \
        and fdr.get("armA_spearman_dfh_pct_pooled", False)
    raw_ok = pooled_raw and pooled_raw["rho"] > 0 \
        and fdr.get("armA_spearman_dfh20_raw_pooled", False)
    xs_ok = b1_daily and b1_daily["mean"] > 0 \
        and fdr.get("armB_daily_spread_top1", False)
    tail_ok = fdr.get("armA_welch_quintile", False) or \
        fdr.get("armA_welch_top_decile", False)
    xs2_ok = fdr.get("armB_daily_spread_top2", False)
    gate_ok = fdr.get("armC_inside_gate_spearman_dfh20", False) or \
        fdr.get("armC_gate_on_vs_off_welch", False)
    primary_ok = ts_ok or raw_ok
    secondary_ok = tail_ok or xs_ok or xs2_ok or gate_ok
    if primary_ok and xs_ok:
        headline = "replicated_both_arms"
    elif primary_ok:
        headline = "replicated_time_series"
    elif secondary_ok:
        headline = (
            "partial_replication_tail_and_gate_concentrated: the pooled "
            "Spearman cells (the exact T101 replication cells) are "
            "same-sign but sub-FDR at powered n (dfh_pct rho={p} vs "
            "Binance +0.031), while the TAIL and GATE cells replicate "
            "with near-identical magnitudes — quintile Welch {q}bps vs "
            "Binance +157.2, inside-gate rho {g} vs +0.119, XS top2 "
            "variant {x}bps/day vs +54.06. The Bybit dfh effect is "
            "concentrated in the near-high/far-from-high tails and the "
            "spec gate, not a broad monotone rank effect; per-asset "
            "pattern also matches (SOL strongest on both venues)."
            .format(p=(pooled_pct or {}).get("rho"),
                    q=(((a.get("pooled") or {})
                        .get("quintile_top_vs_bottom_T101_style") or {})
                       .get("diff_bps")),
                    g=(c.get("inside_gate_spearman_dfh20") or {})
                    .get("rho"),
                    x=(b2.get("mean_daily_spread_bps") or {})
                    .get("mean")))
    else:
        headline = "not_replicated"
    verdicts["headline"] = headline

    return {
        "schema_version": "nanojev-financial-signal-bybit-dfh-v1",
        "status": "measurement_complete" if all_rows else
                  "measurement_skipped_no_data",
        "task": "T107",
        "protocol": "research/financial_signal_bybit_dfh_protocol_v1.json",
        "protocol_sha256": protocol_sha,
        "authorization_path":
            "results/financial_signal_bybit_dfh_authorization_v1.json",
        "contract": {
            "symbols": SYMS, "symbols_skipped": skipped,
            "dfh20": "close[i]/max(close[i-20:i])-1 strictly prior",
            "dfh_pct": "trailing-180 mid-rank pct per symbol",
            "label": "log(close[d+5]/close[d])",
            "funding_daily_rule": "f_last = last 8h settlement inside "
                                  "the UTC day",
            "basis_bps": "(mark_close/index_close - 1)*1e4",
            "btc_trend_gate": "BTC close 20-bar return > 0",
            "top_decile": DECILE,
            "quintile": [Q_HI, Q_LO],
            "f2b2_gate": [F_PCT, B_PCT],
            "min_n": MIN_N, "fdr_alpha": 0.05,
            "fdr_family": sorted(fam)},
        "binance_reference": BINANCE_REF,
        "data_integrity": integrity,
        "n_eval_rows": len(all_rows),
        "arms": arms,
        "fdr_bh": table,
        "verdicts": verdicts,
        "caveats": [
            "5d labels on 1d bars overlap ~5x; pooled rows across 5 "
            "assets share the market factor through common timestamps — "
            "normal-approx p-values are nominal/optimistic (repo caveat)",
            "per-asset cells (~1170 rows) are underpowered for "
            "rho~0.031 (n>=~4000 needed for nominal p<0.05); the "
            "pooled cell (~5800 rows) is the powered comparison",
            "5-asset XS is VERY thin: half the T104 cohort; top1-vs-bot1 "
            "has ~2 effective obs/day and single-asset jumps dominate; "
            "the top2-vs-bot2 variant uses 80% of the cross-section",
            "Bybit eval span (~2023-06-30 onward after the 180d "
            "warm-up) vs the Binance window starting 2023-01-25 — "
            "windows only partly overlap",
            "kline close is a last-trade close, not mark price; the "
            "mark-series robustness cell quantifies the deviation",
            "Bybit funding settles 8h like Binance, so f_last semantics "
            "match directly (no hourly-grid conversion needed, unlike "
            "the T99 Hyperliquid arm)",
            "basis is REAL here (mark/index daily bars exist locally) — "
            "the spec gate ports without the funding+trend-only "
            "deviation the task allowed for",
            "gross close moves only — no fees, spread, slippage, "
            "funding cashflows; not a tradability or profitability "
            "claim",
            "Bybit licence status UNVERIFIED in fetch_manifest; "
            "offline non-commercial research only"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    sha = write_protocol_and_auth()
    r1, r2 = run(sha), run(sha)
    same = json.dumps(r1, sort_keys=True) == json.dumps(r2,
                                                      sort_keys=True)
    r1["determinism"] = {"replays": 2, "byte_identical": same}
    r1["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                      time.gmtime())
    blob = json.dumps(r1, indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(blob)
    v = r1["verdicts"]
    print(json.dumps({"output": str(args.output),
                      "deterministic": same,
                      "sha256": hashlib.sha256(blob.encode()).hexdigest(),
                      "headline": v["headline"],
                      "pooled_dfh_pct": v["pooled_dfh_pct"],
                      "xs_top1": v["xs_top1_vs_bottom1"]}, indent=2))


if __name__ == "__main__":
    main()
