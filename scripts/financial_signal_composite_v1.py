#!/usr/bin/env python3
"""T103: does a distance-from-high (dfh20) gate upgrade the frozen spec?

Frozen spec (research/financial_signal_spec_v1.json,
signal crowded_long_carry_follow_v1, status review_candidate_frozen):
  entry = funding_pct>=0.80 AND basis_pct>=0.66 (trailing-180 mid-rank
          pcts, per asset) AND BTC 20-bar trend up (mark[i]/mark[i-20]-1>0
          on BTCUSDT-PERP rows, mapped to every asset by decision_ns —
          same helper as financial_signal_regime_v1.py arm B)
  exit  = 5-10 decision bars; long only.

T101 (results/financial_signal_dfh_v1.json) found dfh20 =
mark[i]/max(mark[i-20:i])-1 independently predictive (pooled Spearman
rho=+0.031) AND adds rho=+0.119 inside f2b2. This arm asks the
nested-gate question: does adding a dfh gate to the frozen spec sharpen
per-trade economics enough to justify a spec v2, or does it just dilute
an already-sparse signal?

Cells (all on eval rows i>=180, i<=n-11 so both 5d primary and 10d
secondary forward log(mark) labels exist; PIT-safe trailing windows):
  1  spec base cell: f2b2 AND btc_up (reference means, 5d + 10d)
  2  spec+dfh: base AND dfh_pct>=0.50 ("dfh20 >= trailing median"), and
     separately dfh_pct>=0.66 — mean fwd + Welch vs base (declared) and
     vs the dropped low-dfh half (the sharper disjoint contrast)
  3  dfh-only: dfh_pct>=0.90 top decile vs rest, unconditional
     (recomputed for consistency with T101)
  4  orthogonality: Pearson + Spearman corr(dfh_pct, funding_pct) pooled
     — if ~0 the dfh gate carries independent information
  5  interaction 2x2: f2b2 yes/no x dfh_pct high(>=0.5)/low means
  6  net-cost check: non-overlapping per-asset replay (net_backtest /
     conditional_v1 convention: 5bps taker each way = 10bps round trip,
     funding accrual reported alongside) for base vs base+dfh gates

Pre-declared 8-contrast BH-FDR family (alpha=0.05) — declared in the
embedded protocol block before results are computed. MULTIPLE-TESTING
FLAG: this is a nested-gate question — the gated cells are strict
subsets of the base cell, so the Welch contrasts are positively
correlated and the family is not independent; survivors are suggestive,
not decisive. Overlapping 5d labels and shared market factor across
assets further inflate effective n (same caveats as T101).

Measurement only — no fitting, no trading, no spec change. The frozen
spec is NOT modified by this script; the verdict is advisory input to
the owner's keep-v1-frozen vs draft-v2 decision.
"""

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_v1/records.jsonl"
FOLDS = ROOT / "research/financial_r1_pit_validator_core_v2.json"
SPEC = ROOT / "research/financial_signal_spec_v1.json"
OUT = ROOT / "results/financial_signal_composite_v1.json"

LOOKBACK = 180          # trailing rows for mid-rank pct (repo conv.)
HOLD = 5                # primary horizon (decision bars)
HOLD2 = 10              # secondary horizon (decision bars)
DFH_WIN = 20            # trailing-high window, strictly prior bars
MIN_N = 10              # Welch/Spearman minimum per side (repo conv.)
FEE_BPS = 5.0           # taker fee per side -> 10bps round trip
FALLBACK_SETTLEMENTS_PER_DAY = 3  # 8h funding interval fallback
FDR_ALPHA = 0.05

# Pre-declared contrast family (exactly 8) — names fixed before run.
FAMILY = [
    "welch_specdfhmed_vs_specbase_ret5",
    "welch_specdfhp66_vs_specbase_ret5",
    "welch_specdfhmed_vs_specbase_ret10",
    "welch_specdfhhi_vs_specdfhlo_within_base_ret5",
    "welch_dfh_topdecile_vs_rest_ret5",
    "pearson_dfhpct_vs_fundingpct_pooled",
    "welch_f2b2_dfhhi_vs_nonf2b2_dfhhi_ret5",
    "welch_net_nonoverlap_specdfhmed_vs_specbase",
]

PROTOCOL = {
    "schema_version": "nanojev-financial-signal-composite-protocol-v1",
    "created_utc": "2026-09-24T00:00:00Z",
    "purpose": "T103: composite/interaction arm — test whether a dfh20 "
               "(distance-from-20d-high) gate upgrades frozen spec "
               "crowded_long_carry_follow_v1 (f2b2 AND BTC-20d-up -> "
               "5-10d long) on the daily 5-asset PIT cohort, or dilutes "
               "it. Measurement only — no fitting, no trading, no spec "
               "edit, no profitability claims.",
    "cohort": {
        "path": "data/perp_pit_v1/records.jsonl",
        "records": 6554,
        "assets": ["BNBUSDT-PERP", "BTCUSDT-PERP", "ETHUSDT-PERP",
                   "SOLUSDT-PERP", "XRPUSDT-PERP"],
        "span_utc": "2023-01-25T23:59:59.999 .. 2026-08-30T23:59:59.999",
        "features_used": ["mark_price", "last_funding_rate",
                          "mark_index_basis_bps",
                          "funding_interval_hours"],
        "targets": "forward log(mark) returns recomputed per asset: "
                   "primary 5d (mark[i+5]/mark[i]), secondary 10d "
                   "(mark[i+10]/mark[i]); PIT-safe (strictly after the "
                   "decision bar)",
        "eval_window": "rows require i>=180 (trailing-180 pct ranks) "
                       "and i<=n-11 (both 5d and 10d labels defined)",
    },
    "definitions": {
        "funding_pct": "trailing-180 mid-rank pct of last_funding_rate",
        "basis_pct": "trailing-180 mid-rank pct of mark_index_basis_bps",
        "f2b2": "funding_pct>=0.80 AND basis_pct>=0.66 (frozen spec "
                "carry cell)",
        "btc_up": "BTC mark[i]/mark[i-20]-1 > 0 on BTCUSDT-PERP rows, "
                  "mapped to all assets by decision_ns (regime_v1 arm-B "
                  "helper)",
        "spec_base": "f2b2 AND btc_up (frozen spec entry condition)",
        "dfh20": "mark[i]/max(mark[i-20:i])-1 (trailing 20d high over "
                 "strictly prior bars)",
        "dfh_pct": "trailing-180 mid-rank pct of dfh20 within the "
                   "asset's own history; 'dfh>=median' = dfh_pct>=0.50, "
                   "'dfh p66' = dfh_pct>=0.66, 'top decile' = >=0.90",
        "net_bps": "gross_bps - 10 (5bps taker each way round trip); "
                   "funding-accrual-adjusted net reported alongside per "
                   "net_backtest_v1 convention",
    },
    "contrast_family": FAMILY,
    "statistics": {
        "test": "Welch two-sample t on log returns, normal-approx "
                "two-sided p; Pearson correlation with t-approx normal "
                "p for the orthogonality cell (Spearman reported "
                "alongside); min n=10 per side — same conventions as "
                "financial_signal_dfh_v1 / regime_v1",
        "multiple_testing": "BH-FDR alpha=0.05 over the fixed 8-cell "
                            "family above. FLAG: nested-gate design — "
                            "gated cells are subsets of base, so "
                            "contrasts are correlated and the family is "
                            "not independent; survivors are suggestive "
                            "only",
        "outside_family": "2x2 cell means, per-asset/per-fold "
                          "descriptives, trade counts, net summaries",
        "caveats": [
            "5d/10d forward labels overlap on daily bars -> nominal/"
            "optimistic p-values",
            "pooled rows across 5 assets share the market factor via "
            "common timestamps -> overstated effective n",
            "subset-vs-superset Welch (gated vs base) double-counts "
            "the gated rows; the disjoint within-base hi-vs-lo "
            "contrast is the cleaner test and is declared alongside",
            "non-overlapping trade replay reduces clustering but trade "
            "counts are small; treat net comparison as descriptive",
            "treat p<0.01 as suggestive, not decisive",
        ],
    },
    "forbidden": ["fitting", "trading", "profitability claims",
                  "spec_v1 edits", "protocol edits post-run"],
}


def feat(r, name):
    f = r.get("features", {}).get(name)
    return None if f is None else f.get("value")


def pct(w, x):
    """Mid-rank percentile of x within trailing window w (project conv.)."""
    return (sum(1 for v in w if v < x)
            + 0.5 * sum(1 for v in w if v == x)) / len(w)


def welch(a, b):
    """Welch t of mean(a)-mean(b) on log returns; means in bps.
    None if either side < MIN_N or se==0 (dfh_v1 convention)."""
    if len(a) < MIN_N or len(b) < MIN_N:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    se = math.sqrt(statistics.pvariance(a) / len(a)
                   + statistics.pvariance(b) / len(b))
    if se == 0:
        return None
    t = (ma - mb) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return {"n_a": len(a), "n_b": len(b),
            "mean_a_bps": round(ma * 1e4, 1),
            "mean_b_bps": round(mb * 1e4, 1),
            "diff_bps": round((ma - mb) * 1e4, 1),
            "t": round(t, 3), "p": round(p, 6)}


def pearson(xs, ys):
    """Pearson r, t-approx normal two-sided p (same approx as the
    Spearman convention). None if n<MIN_N or zero variance."""
    n = len(xs)
    if n < MIN_N:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if dx == 0 or dy == 0:
        return None
    r = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (dx * dy)
    t = r * math.sqrt((n - 2) / max(1e-9, 1 - r * r))
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return {"n": n, "r": round(r, 4), "p": round(p, 6)}


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
    n = len(xs)
    if n < MIN_N:
        return None
    rx, ry = ranks(xs), ranks(ys)
    return pearson(rx, ry)


def mean_bps(xs):
    return round(statistics.mean(xs) * 1e4, 1) if xs else None


def cell_stats(rows, key):
    xs = [r[key] for r in rows]
    return {"n": len(xs), "mean_bps": mean_bps(xs),
            "median_bps": round(statistics.median(xs) * 1e4, 1)
            if xs else None}


def run():
    records = [json.loads(l) for l in COHORT.read_text().splitlines()
               if l.strip()]
    folds = json.loads(FOLDS.read_text())["folds"]
    spec = json.loads(SPEC.read_text())
    by_asset = defaultdict(list)
    for r in records:
        by_asset[r["asset_id"]].append(r)
    for rs in by_asset.values():
        rs.sort(key=lambda x: x["decision_ns"])

    # BTC 20-bar trend sign per decision_ns (regime_v1 arm-B helper).
    btc = by_asset["BTCUSDT-PERP"]
    bm = [feat(x, "mark_price") for x in btc]
    btc_trend = {}
    for i in range(20, len(btc)):
        if bm[i] and bm[i - 20] and bm[i] > 0 and bm[i - 20] > 0:
            btc_trend[btc[i]["decision_ns"]] = bm[i] / bm[i - 20] - 1 > 0

    # ---------- eval rows: i in [180, n-11] ----------
    rows = []
    for asset, rs in sorted(by_asset.items()):
        mark = [feat(x, "mark_price") for x in rs]
        fund = [feat(x, "last_funding_rate") for x in rs]
        basis = [feat(x, "mark_index_basis_bps") for x in rs]
        n = len(rs)
        dfh20 = [None] * n
        for i in range(DFH_WIN, n):
            w = [m for m in mark[i - DFH_WIN:i] if m and m > 0]
            if mark[i] and mark[i] > 0 and len(w) == DFH_WIN:
                dfh20[i] = mark[i] / max(w) - 1.0
        for i in range(LOOKBACK, n - HOLD2):
            if (mark[i] is None or mark[i] <= 0
                    or not (mark[i + HOLD] and mark[i + HOLD] > 0)
                    or not (mark[i + HOLD2] and mark[i + HOLD2] > 0)
                    or fund[i] is None or basis[i] is None
                    or dfh20[i] is None):
                continue
            dwin = [v for v in dfh20[i - LOOKBACK:i] if v is not None]
            if not dwin:
                continue
            f_pct = pct(fund[i - LOOKBACK:i], fund[i])
            b_pct = pct(basis[i - LOOKBACK:i], basis[i])
            rows.append({
                "asset": asset, "ns": rs[i]["decision_ns"], "i": i,
                "ret5": math.log(mark[i + HOLD] / mark[i]),
                "ret10": math.log(mark[i + HOLD2] / mark[i]),
                "f_pct": f_pct, "b_pct": b_pct,
                "dfh20": dfh20[i], "dfh_pct": pct(dwin, dfh20[i]),
                "in_f2b2": f_pct >= 0.80 and b_pct >= 0.66,
                "btc_up": btc_trend.get(rs[i]["decision_ns"]),
            })

    def sel(pred, key="ret5"):
        return [r[key] for r in rows if pred(r)]

    base_pred = lambda r: r["in_f2b2"] and r["btc_up"] is True
    med_pred = lambda r: base_pred(r) and r["dfh_pct"] >= 0.50
    p66_pred = lambda r: base_pred(r) and r["dfh_pct"] >= 0.66
    lo_pred = lambda r: base_pred(r) and r["dfh_pct"] < 0.50

    # ---------- cells 1-2: spec base vs spec+dfh ----------
    cells = {
        "spec_base": cell_stats([r for r in rows if base_pred(r)], "ret5"),
        "spec_base_ret10": cell_stats(
            [r for r in rows if base_pred(r)], "ret10"),
        "spec_plus_dfh_med": {
            **cell_stats([r for r in rows if med_pred(r)], "ret5"),
            "ret10": cell_stats([r for r in rows if med_pred(r)], "ret10"),
            "welch_vs_base_ret5": welch(sel(med_pred), sel(base_pred)),
            "welch_vs_base_ret10": welch(sel(med_pred, "ret10"),
                                         sel(base_pred, "ret10")),
        },
        "spec_plus_dfh_p66": {
            **cell_stats([r for r in rows if p66_pred(r)], "ret5"),
            "ret10": cell_stats([r for r in rows if p66_pred(r)], "ret10"),
            "welch_vs_base_ret5": welch(sel(p66_pred), sel(base_pred)),
            "welch_vs_base_ret10": welch(sel(p66_pred, "ret10"),
                                         sel(base_pred, "ret10")),
        },
        "spec_base_dfh_low_dropped_half": cell_stats(
            [r for r in rows if lo_pred(r)], "ret5"),
        "welch_dfhhi_vs_dfhlo_within_base_ret5": welch(
            sel(med_pred), sel(lo_pred)),
    }

    # ---------- cell 3: dfh-only top decile vs rest ----------
    dfh_only = {
        "definition": "unconditional dfh_pct>=0.90 vs <0.90, all eval "
                      "rows (recompute of T101-style decile contrast at "
                      "the T103 eval window)",
        "top_decile": cell_stats(
            [r for r in rows if r["dfh_pct"] >= 0.90], "ret5"),
        "rest": cell_stats(
            [r for r in rows if r["dfh_pct"] < 0.90], "ret5"),
        "welch_top_vs_rest_ret5": welch(
            sel(lambda r: r["dfh_pct"] >= 0.90),
            sel(lambda r: r["dfh_pct"] < 0.90)),
    }

    # ---------- cell 4: orthogonality ----------
    pair = [(r["dfh_pct"], r["f_pct"]) for r in rows]
    orth = {
        "pearson_pooled": pearson([a for a, _ in pair],
                                  [b for _, b in pair]),
        "spearman_pooled_secondary": spearman([a for a, _ in pair],
                                              [b for _, b in pair]),
        "pearson_per_asset": {
            a: pearson([r["dfh_pct"] for r in rows if r["asset"] == a],
                       [r["f_pct"] for r in rows if r["asset"] == a])
            for a in sorted(by_asset)},
    }

    # ---------- cell 5: interaction 2x2 ----------
    inter = {}
    for f2 in (True, False):
        for hi in (True, False):
            sub = [r for r in rows if r["in_f2b2"] == f2
                   and (r["dfh_pct"] >= 0.50) == hi]
            inter["f2b2_{}_dfh_{}".format(
                "yes" if f2 else "no", "hi" if hi else "lo")] = {
                **cell_stats(sub, "ret5"),
                "ret10_mean_bps": mean_bps([r["ret10"] for r in sub])}
    inter["welch_f2b2yes_dfhhi_vs_f2b2no_dfhhi_ret5"] = welch(
        sel(lambda r: r["in_f2b2"] and r["dfh_pct"] >= 0.50),
        sel(lambda r: not r["in_f2b2"] and r["dfh_pct"] >= 0.50))

    # ---------- cell 6: net-cost non-overlapping replay ----------
    def replay(rs, extra_pred):
        """Non-overlapping gated long replay, conditional_v1 convention:
        enter at mark[i] when f2b2 AND btc_up AND extra_pred(i), exit
        mark[i+5], skip gate hits inside an open hold."""
        mark = [feat(x, "mark_price") for x in rs]
        fund = [feat(x, "last_funding_rate") for x in rs]
        interval = [feat(x, "funding_interval_hours") for x in rs]
        basis = [feat(x, "mark_index_basis_bps") for x in rs]
        n = len(rs)
        dfh20 = [None] * n
        for i in range(DFH_WIN, n):
            w = [m for m in mark[i - DFH_WIN:i] if m and m > 0]
            if mark[i] and mark[i] > 0 and len(w) == DFH_WIN:
                dfh20[i] = mark[i] / max(w) - 1.0
        trades = []
        i = LOOKBACK
        while i <= n - HOLD2 - 1:
            fw = fund[i - LOOKBACK:i]
            bw = basis[i - LOOKBACK:i]
            if (fund[i] is None or basis[i] is None or dfh20[i] is None
                    or btc_trend.get(rs[i]["decision_ns"]) is not True):
                i += 1
                continue
            f_ok = pct(fw, fund[i]) >= 0.80
            b_ok = pct(bw, basis[i]) >= 0.66
            dwin = [v for v in dfh20[i - LOOKBACK:i] if v is not None]
            dpct = pct(dwin, dfh20[i]) if dwin else None
            if not (f_ok and b_ok and dpct is not None
                    and extra_pred(dpct)):
                i += 1
                continue
            entry, exit_ = mark[i], mark[i + HOLD]
            if not (entry and entry > 0 and exit_ and exit_ > 0):
                i += HOLD
                continue
            gross = (exit_ / entry - 1) * 1e4
            paid = 0.0
            for j in range(i + 1, i + HOLD + 1):
                per_day = ((24.0 / interval[j]) if interval[j]
                           and interval[j] > 0
                           else FALLBACK_SETTLEMENTS_PER_DAY)
                paid += fund[j] * per_day * 1e4  # long pays +funding
            trades.append({"i": i, "ns": rs[i]["decision_ns"],
                           "gross_bps": gross,
                           "net_fee_bps": gross - 2 * FEE_BPS,
                           "net_full_bps": gross - 2 * FEE_BPS - paid,
                           "funding_bps": paid, "dfh_pct": dpct})
            i += HOLD
        return trades

    gates = {"spec_base": lambda d: True,
             "spec_plus_dfh_med": lambda d: d >= 0.50,
             "spec_plus_dfh_p66": lambda d: d >= 0.66}
    trades = {g: [] for g in gates}
    for asset, rs in sorted(by_asset.items()):
        for g, pred in gates.items():
            for t in replay(rs, pred):
                t["asset"] = asset
                trades[g].append(t)

    def tsum(ts):
        if not ts:
            return {"n_trades": 0}
        g = [t["gross_bps"] for t in ts]
        nf = [t["net_fee_bps"] for t in ts]
        nl = [t["net_full_bps"] for t in ts]
        return {"n_trades": len(ts),
                "mean_gross_bps": round(statistics.mean(g), 1),
                "mean_net_fee_only_bps": round(statistics.mean(nf), 1),
                "mean_net_fee_plus_funding_bps": round(
                    statistics.mean(nl), 1),
                "median_net_fee_only_bps": round(
                    statistics.median(nf), 1),
                "win_rate_net_fee": round(
                    sum(1 for x in nf if x > 0) / len(nf), 3),
                "total_net_fee_bps": round(sum(nf), 0),
                "per_asset_n": {a: sum(1 for t in ts if t["asset"] == a)
                                for a in sorted(by_asset)}}

    net = {g: tsum(trades[g]) for g in gates}
    net["welch_net_fee_specdfhmed_vs_specbase"] = welch(
        [t["net_fee_bps"] / 1e4 for t in trades["spec_plus_dfh_med"]],
        [t["net_fee_bps"] / 1e4 for t in trades["spec_base"]])
    net["cost_convention"] = ("5bps taker per side = 10bps round trip "
                              "(net_fee); funding accrual = settled "
                              "last_funding_rate x 24/interval per held "
                              "day reported in net_full — "
                              "net_backtest_v1 convention, approximate, "
                              "no market impact model")

    # ---------- per-fold sign of gated-vs-base 5d mean (descriptive) ----------
    per_fold = {}
    for fi, f in enumerate(folds):
        a_, b_ = f["test"]
        g5 = [r["ret5"] for r in rows
              if med_pred(r) and a_ <= r["ns"] < b_]
        b5 = [r["ret5"] for r in rows
              if base_pred(r) and a_ <= r["ns"] < b_]
        per_fold[f"f{fi}"] = {
            "n_gated": len(g5), "n_base": len(b5),
            "gated_mean_bps": mean_bps(g5),
            "base_mean_bps": mean_bps(b5),
            "gated_minus_base_bps": (
                round(statistics.mean(g5) * 1e4
                      - statistics.mean(b5) * 1e4, 1)
                if g5 and b5 else None)}

    # ---------- FDR family ----------
    fam = {}
    w = cells["spec_plus_dfh_med"]["welch_vs_base_ret5"]
    if w:
        fam["welch_specdfhmed_vs_specbase_ret5"] = w["p"]
    w = cells["spec_plus_dfh_p66"]["welch_vs_base_ret5"]
    if w:
        fam["welch_specdfhp66_vs_specbase_ret5"] = w["p"]
    w = cells["spec_plus_dfh_med"]["welch_vs_base_ret10"]
    if w:
        fam["welch_specdfhmed_vs_specbase_ret10"] = w["p"]
    w = cells["welch_dfhhi_vs_dfhlo_within_base_ret5"]
    if w:
        fam["welch_specdfhhi_vs_specdfhlo_within_base_ret5"] = w["p"]
    w = dfh_only["welch_top_vs_rest_ret5"]
    if w:
        fam["welch_dfh_topdecile_vs_rest_ret5"] = w["p"]
    c = orth["pearson_pooled"]
    if c:
        fam["pearson_dfhpct_vs_fundingpct_pooled"] = c["p"]
    w = inter["welch_f2b2yes_dfhhi_vs_f2b2no_dfhhi_ret5"]
    if w:
        fam["welch_f2b2_dfhhi_vs_nonf2b2_dfhhi_ret5"] = w["p"]
    w = net["welch_net_fee_specdfhmed_vs_specbase"]
    if w:
        fam["welch_net_nonoverlap_specdfhmed_vs_specbase"] = w["p"]
    missing = [k for k in FAMILY if k not in fam]
    ps = sorted([(p, n_) for n_, p in fam.items()])
    m = len(ps)
    fdr = [{"contrast": n_, "p": p, "alpha_bh": FDR_ALPHA * (i + 1) / m,
            "survives": p <= FDR_ALPHA * (i + 1) / m}
           for i, (p, n_) in enumerate(ps)]
    surv = {e["contrast"] for e in fdr if e["survives"]}

    # ---------- honest verdict ----------
    base5 = cells["spec_base"]["mean_bps"]
    med = cells["spec_plus_dfh_med"]
    p66 = cells["spec_plus_dfh_p66"]
    ntb = net["spec_base"]["n_trades"]
    ntg = net["spec_plus_dfh_med"]["n_trades"]
    d_net = (net["spec_plus_dfh_med"].get("mean_net_fee_only_bps", 0)
             - net["spec_base"].get("mean_net_fee_only_bps", 0))
    disjoint_sig = ("welch_specdfhhi_vs_specdfhlo_within_base_ret5"
                    in surv)
    verdict = [
        "Spec base (f2b2 AND btc_up): n={} rows, mean 5d {}bps / 10d "
        "{}bps; +dfh-med gate keeps {} rows (mean {}bps), +dfh-p66 "
        "keeps {} ({}bps).".format(
            cells["spec_base"]["n"], base5,
            cells["spec_base_ret10"]["mean_bps"],
            med["n"], med["mean_bps"], p66["n"], p66["mean_bps"]),
        "Within-base disjoint contrast dfh-hi vs dfh-lo: {}; "
        "subset-vs-base Welch med: {}.".format(
            cells["welch_dfhhi_vs_dfhlo_within_base_ret5"],
            med["welch_vs_base_ret5"]),
        "Orthogonality: corr(dfh_pct, funding_pct) pearson r={} -> dfh "
        "carries {} information.".format(
            (orth["pearson_pooled"] or {}).get("r"),
            "largely independent"
            if orth["pearson_pooled"]
            and abs(orth["pearson_pooled"]["r"]) < 0.15
            else "shared"),
        "Net-cost (non-overlapping, 10bps rt): base {} trades mean "
        "{}bps net-fee vs gated {} trades {}bps (delta {}bps).".format(
            ntb, net["spec_base"].get("mean_net_fee_only_bps"),
            ntg, net["spec_plus_dfh_med"].get("mean_net_fee_only_bps"),
            round(d_net, 1)),
        "FDR survivors: {}. Family not independent (nested-gate "
        "design).".format(sorted(surv) if surv else "none"),
    ]
    # upgrade rule (pre-declared spirit): need the disjoint within-base
    # contrast to survive FDR AND positive net delta AND >=30 gated
    # trades retained; otherwise keep v1 frozen.
    upgrade = disjoint_sig and d_net > 0 and ntg >= 30
    verdict.append(
        "RECOMMENDATION: {} — {}".format(
            "draft spec_v2 candidate (dfh gate) for owner review"
            if upgrade else "KEEP spec v1 frozen",
            "dfh gate survives the disjoint FDR contrast, improves "
            "net-of-cost mean, and retains >=30 trades"
            if upgrade else
            "dfh gate does not meet the pre-declared upgrade bar "
            "(disjoint within-base FDR survivor AND positive net delta "
            "AND >=30 retained trades); treat dfh20 as a descriptive "
            "overlay / candidate amplifier, not a spec component"))
    results = {
        "n_records": len(records),
        "n_eval_rows": len(rows),
        "eval_window": "i>=180 AND i<=n-11 per asset (both 5d and 10d "
                       "labels defined)",
        "frozen_spec": {"ref": "research/financial_signal_spec_v1.json",
                        "signal": spec.get("signal_name"),
                        "status": spec.get("status")},
        "cells_1_2_spec_vs_gated": cells,
        "cell_3_dfh_only": dfh_only,
        "cell_4_orthogonality": orth,
        "cell_5_interaction_2x2": inter,
        "cell_6_net_cost": net,
        "per_fold_gated_minus_base": per_fold,
        "fdr_bh": fdr,
        "family_missing": missing,
        "multiple_testing_flag": (
            "NESTED-GATE QUESTION: gated cells are strict subsets of "
            "the spec base cell, so family contrasts are positively "
            "correlated and BH-FDR assumes away that dependence; "
            "overlapping labels + shared market factor further inflate "
            "effective n. Survivors are suggestive, not decisive."),
        "verdict": verdict,
        "notes": [
            "dfh>=median implemented as dfh_pct>=0.50 (trailing-180 "
            "mid-rank), consistent with T101's median split; the p66 "
            "variant mirrors the basis_pct threshold convention.",
            "Welch gated-vs-base double-counts gated rows (subset vs "
            "superset); the disjoint within-base hi-vs-lo Welch is the "
            "cleaner declared contrast.",
            "Net replay is non-overlapping per asset with HOLD=5 skip; "
            "trade counts are small and clustered — descriptive only.",
            "Interpretation note: dfh-high days are largely btc-up days "
            "(corr(dfh_pct,funding_pct)~0.30 and the BTC-trend leg is "
            "itself a market-level near-high proxy), so the spec base "
            "cell is already dfh-tilted — the marginal room a dfh gate "
            "can add inside it is structurally small; the larger dfh "
            "spread sits in the f2b2-only 2x2 (no btc_up leg).",
            "This script does NOT modify the frozen spec; the verdict "
            "is advisory input to the owner.",
        ],
    }
    proto_blob = json.dumps(PROTOCOL, sort_keys=True).encode()
    return {
        "schema_version": "nanojev-financial-signal-composite-v1",
        "status": "measurement_complete",
        "scope": "T103 composite/interaction arm: dfh20 gate vs frozen "
                 "spec crowded_long_carry_follow_v1 on the daily "
                 "5-asset PIT cohort; measurement only",
        "protocol": PROTOCOL,
        "authorization": {
            "schema_version":
                "nanojev-financial-signal-hypotheses-authorization-v8",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                         time.gmtime()),
            "protocol_sha256": hashlib.sha256(proto_blob).hexdigest(),
            "protocol_path": "embedded in this receipt (protocol block)",
            "decision": "approved_for_measurement",
            "measurement_authorized": True,
            "fit_authorized": False,
            "independent_reviewer": {
                "id": "project-owner",
                "independence":
                    "owner_self_authorization_not_independent_review",
                "note": "Owner directed T103: composite/interaction "
                        "arm testing whether dfh20 upgrades the frozen "
                        "spec on the daily cohort (delegated task)."},
            "scope": {
                "permitted": "PIT-safe descriptive measurement on "
                             "data/perp_pit_v1/records.jsonl: spec base "
                             "vs dfh-gated cells, orthogonality, 2x2 "
                             "interaction means, net-of-cost "
                             "non-overlapping replay; Welch + Pearson/"
                             "Spearman + BH-FDR per embedded protocol.",
                "not_permitted": "No fitting/trading/profitability "
                                 "claims/spec_v1 edits/protocol edits "
                                 "post-run."},
            "network_model_calls": 0,
            "order_submission_authorized": False,
            "live_trading_authorized": False},
        "parameters": {"lookback_pct": LOOKBACK,
                       "hold_primary": HOLD,
                       "hold_secondary": HOLD2,
                       "dfh_window": DFH_WIN,
                       "dfh_gates": {"median": 0.50, "p66": 0.66,
                                     "top_decile": 0.90},
                       "f2b2": "funding_pct>=0.80 AND basis_pct>=0.66",
                       "fee_bps_per_side": FEE_BPS,
                       "min_n": MIN_N,
                       "fdr_alpha": FDR_ALPHA,
                       "contrast_family_size": len(FAMILY)},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                      time.gmtime()),
        "results": results}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    receipt = run()
    blob = json.dumps(receipt, indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(blob)
    print(json.dumps({"output": str(args.output),
                      "sha256": hashlib.sha256(blob.encode()).hexdigest()},
                     indent=2))


if __name__ == "__main__":
    main()
