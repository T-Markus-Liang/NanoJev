#!/usr/bin/env python3
"""T96: hourly momentum-reversal economics + settlement-window contamination.

Cohort: data/perp_pit_intraday_v1/records.jsonl — 17,960 hourly decision
bars, 5 USDT-M perp assets (BNB/BTC/ETH/SOL/XRP), 2026-04-02..2026-08-31
UTC. Label = label.forward_return_bps (4h forward mark return, gross, bps).
Scores are per-asset trailing-90-record mid-rank percentiles (benchmark v2
convention); eval rows require i >= ILOOKBACK + MOM4_LAG = 94 per asset
with complete momentum windows -> 17,490 eval rows (470 warmup dropped).

Prior results under test:
  benchmark v2 intraday cells: mom_1h rho=-0.035, mom_4h rho=-0.047
  (both FDR-passing and placebo-clear; negative = reversal: strong
  down-moves are followed by positive 4h forward returns).
  settlement v1 (T93): the post-settlement dip is the post half of a
  settlement-centered weak window — cycle hours 0-1 (post) and 6-7
  (pre) sit at ~0/negative mean 4h fwd vs +3.7..+5.7bps mid-cycle.

Arms:
  A1 reversal economics: mom_4h score bottom quintile (<=0.20, strongest
     relative down-moves) -> next-4h mean gross fwd bps vs the remaining
     rows; symmetric top-quintile (>=0.80) and mom_1h pair. Net-of-cost:
     5bps/side taker (10bps round trip) plus scheduled funding cashflow
     when a settlement lands inside the 4h hold (benchmark v2 convention;
     pro-rata fallback on the 20 null-distance records). Long pays
     funding, short receives it. Honest tradability flag vs fees.
     Per-asset table for the mom_4h bottom-quintile arm.
  A2 reversal conditioning: interaction of the mom_4h reversal with
     rv_pct terciles and with the funding-sign regime (share_pos =
     trailing-90 fraction of last_funding_rate > 0, current record
     excluded; positive >0.6 / negative <0.4 / mixed — T94 convention).
     Per cell: n, Spearman rho of mom_4h score vs fwd, bottom-quintile
     vs rest Welch tilt.
  A3 settlement contamination: recompute the benchmark v2 key cells
     (funding_pct, mom_4h, taker_pct Spearman on 4h fwd) on the full
     eval set and EXCLUDING bars within +-2h of a settlement. Rule:
     pre side = ns_until_next_funding_settlement <= 2h; post side =
     distance >= 6h; for the 20 null-distance records fall back to
     decision_ms mod 28_800_000 >= 21_600_000 (pre) / < 7_200_000
     (post). On this cohort the rule selects exactly cycle hours
     {0,1,6,7} (~49.8%). Rho/t/p/n both ways, within-asset
     label-shuffle placebo rhos (3 seeds), BH-FDR per 3-cell family,
     verdict flips, parity vs stored benchmark v2 rhos.
  A4 verdicts: is the hourly reversal economically meaningful or
     sub-fee noise; does conditioning strengthen it; does the
     settlement-window exclusion move any cell verdict.

Statistics: Welch two-sample t, normal-approx two-sided p (repo
convention), min n=10 per side; mid-rank Spearman with normal-approx
t p; BH-FDR alpha=0.05 computed separately per family (A1: 4 contrasts,
A2: 6 contrasts, A3: 3 cells x 2 sets — no cross-family pooling).

Caveats: 4h forward labels on 1h bars overlap ~4x and cross-asset bars
share timestamps (common market factor), so p-values are
nominal/optimistic; A1 "trades" overlap and are not independent round
trips. Measurement only — no fitting, no trading.
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
COHORT = ROOT / "data/perp_pit_intraday_v1/records.jsonl"
BENCH = ROOT / "results/financial_signal_benchmark_v2.json"
OUT = ROOT / "results/financial_signal_reversal_v1.json"
PROTOCOL = "research/financial_signal_reversal_protocol_v1.json"

TRAIL = 90                  # trailing per-asset records for pct/regime
MOM4_LAG = 4                # mom_4h lookback in bars
HOLD_H = 4                  # label horizon, hours
IQ = 0.20                   # quintile fraction
MIN_N = 10                  # Welch/Spearman minimum per side (repo conv.)
FEE_BPS = 5.0               # taker per side
SETTLE_MS = 28_800_000      # 8h settlement grid in ms
HOUR_MS = 3_600_000
HOUR_NS = 3_600_000_000_000
PRE_NS = 2 * HOUR_NS        # dist <= 2h -> within 2h before settlement
POST_NS = 6 * HOUR_NS       # dist >= 6h -> within 2h after settlement
PLACEBO_SEEDS = (17, 73, 991)
POS_HI, POS_LO = 0.6, 0.4   # share_pos regime thresholds (T94 conv.)
A3_CELLS = ["funding_pct", "mom_4h", "taker_pct"]


# ---------------------------------------------------------------- stats
def pct(w, x):
    """Mid-rank percentile of x within trailing window w."""
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


def spearman(xs, ys):
    """Mid-rank Spearman rho/t/p — benchmark v1/v2 convention."""
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
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return {"rho": rho, "t": t, "p": p}


def welch(a, b):
    """Welch t of mean(a)-mean(b); normal-approx two-sided p (repo
    convention). Returns None if either side < MIN_N or zero variance."""
    if len(a) < MIN_N or len(b) < MIN_N:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.pvariance(a), statistics.pvariance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return None
    t = (ma - mb) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    pooled_sd = math.sqrt((va * len(a) + vb * len(b)) / (len(a) + len(b)))
    mde = (1.96 + 0.84) * pooled_sd * math.sqrt(1 / len(a) + 1 / len(b))
    return {"n_a": len(a), "n_b": len(b),
            "mean_a_bps": round(ma, 2), "mean_b_bps": round(mb, 2),
            "diff_bps": round(ma - mb, 2), "t": round(t, 3),
            "p": round(p, 6), "approx_mde80_bps": round(mde, 1)}


def summ(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0, "mean_bps": None, "median_bps": None}
    return {"n": len(vals), "mean_bps": round(statistics.mean(vals), 2),
            "median_bps": round(statistics.median(vals), 2)}


def net_summ(vals):
    """Per-trade net stats — net_stats convention of benchmark v1."""
    vals = [v for v in vals if v is not None]
    if len(vals) < 5:
        return {"n": len(vals)}
    mean = statistics.mean(vals)
    sd = statistics.pstdev(vals) or 1e-9
    return {"n": len(vals), "mean_net_bps": round(mean, 1),
            "median_net_bps": round(statistics.median(vals), 1),
            "win_rate": round(sum(1 for v in vals if v > 0) / len(vals), 3),
            "sharpe_per_trade": round(mean / sd, 3),
            "total_net_bps": round(sum(vals), 0)}


def shuffle(vals, seed):
    """Deterministic LCG Fisher-Yates (same generator as T82/T88/T91)."""
    v = vals[:]
    s = seed
    for i in range(len(v) - 1, 0, -1):
        s = (s * 6364136223846793005 + 1442695040888963407) & (2**64 - 1)
        j = s % (i + 1)
        v[i], v[j] = v[j], v[i]
    return v


def tercile(p):
    if p is None:
        return None
    return "low" if p < 1 / 3 else ("mid" if p < 2 / 3 else "high")


def regime(share_pos):
    if share_pos > POS_HI:
        return "positive"
    if share_pos < POS_LO:
        return "negative"
    return "mixed"


def bh_fdr(named_ps):
    """BH-FDR alpha=0.05 over (name, p) pairs -> (table, {name: bool})."""
    ps = sorted((p, n) for n, p in named_ps if p is not None)
    m = len(ps)
    table = [{"cell": n, "p": p, "alpha_bh": round(0.05 * (i + 1) / m, 6),
              "survives": p <= 0.05 * (i + 1) / m}
             for i, (p, n) in enumerate(ps)]
    return table, {e["cell"]: e["survives"] for e in table}


# ---------------------------------------------------------------- dataset
def build_rows():
    """Eval rows identical in construction to benchmark v2 intraday:
    i >= TRAIL+MOM4_LAG per asset, label present, all momentum windows
    complete; trailing-90 mid-rank pct scores. Adds funding cashflow,
    net long/short, share_pos regime, cycle hour, contamination flag.

    Row filter matches benchmark v2: basis_pct/abnvol_pct windows are
    not scored here but are complete for every record in this cohort
    (all 10 features present on all 17,960 records; abnvol falls back
    to 1.0 when the trailing median volume is non-positive), so the
    eval set is identical (17,490 rows)."""
    by_asset = defaultdict(list)
    n_rec = 0
    for line in COHORT.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by_asset[r["asset_id"]].append(r)
            n_rec += 1
    for rs in by_asset.values():
        rs.sort(key=lambda x: x["decision_ns"])

    hold_ns = HOLD_H * HOUR_NS
    rows = []
    n_null_dist = 0
    n_dist_mod_agree = 0
    for asset, rs in sorted(by_asset.items()):
        f = lambda name: [x["features"][name]["value"] for x in rs]
        mark, fund = f("mark_price"), f("last_funding_rate")
        rv, tkr = f("realized_vol_24h"), f("taker_buy_ratio")
        stl = f("ns_until_next_funding_settlement")
        ivl = f("funding_interval_hours")
        n = len(rs)
        mom1, mom4 = [None] * n, [None] * n
        for i in range(1, n):
            if mark[i] > 0 and mark[i - 1] > 0:
                mom1[i] = math.log(mark[i] / mark[i - 1])
        for i in range(MOM4_LAG, n):
            if mark[i] > 0 and mark[i - MOM4_LAG] > 0:
                mom4[i] = math.log(mark[i] / mark[i - MOM4_LAG])
        series = {"funding_pct": fund, "rv_pct": rv, "taker_pct": tkr,
                  "mom_1h": mom1, "mom_4h": mom4}

        for i in range(TRAIL + MOM4_LAG, n):
            fwd = rs[i].get("label", {}).get("forward_return_bps")
            if fwd is None:
                continue
            row = {"asset": asset, "ns": rs[i]["decision_ns"],
                   "fwd": fwd}
            ok = True
            for name, s in series.items():
                w = s[i - TRAIL:i]
                if s[i] is None or any(v is None for v in w):
                    ok = False
                    break
                row[name] = pct(w, s[i])
            if not ok:
                continue
            w = fund[i - TRAIL:i]
            row["share_pos"] = sum(1 for v in w if v > 0) / len(w)
            row["fund_regime"] = regime(row["share_pos"])
            s = stl[i]
            if s is not None:
                paid = fund[i] * 1e4 if 0 <= s <= hold_ns else 0.0
            else:
                n_null_dist += 1
                paid = (fund[i] * (HOLD_H / ivl[i]) * 1e4
                        if ivl[i] > 0 else 0.0)
            row["paid_bps"] = paid
            row["net_long"] = fwd - 2 * FEE_BPS - paid
            row["net_short"] = -fwd - 2 * FEE_BPS + paid
            ms = row["ns"] // 1_000_000
            mod = ms % SETTLE_MS
            row["cyc"] = mod // HOUR_MS
            mod_pre = mod >= 6 * HOUR_MS     # cycle hours 6-7
            mod_post = mod < 2 * HOUR_MS     # cycle hours 0-1
            if s is not None:
                contam = s <= PRE_NS or s >= POST_NS
                if contam == (mod_pre or mod_post):
                    n_dist_mod_agree += 1
            else:
                contam = mod_pre or mod_post
            row["contam"] = contam
            rows.append(row)
    return n_rec, rows, n_null_dist, n_dist_mod_agree


# ---------------------------------------------------------------- arm A1
def arm_block(sel, net_key):
    gross = [r["fwd"] for r in sel]
    out = {"n": len(sel), "mean_gross_bps": None, "median_gross_bps": None,
           "mean_paid_bps": None}
    if sel:
        out["mean_gross_bps"] = round(statistics.mean(gross), 2)
        out["median_gross_bps"] = round(statistics.median(gross), 2)
        out["mean_paid_bps"] = round(
            statistics.mean([r["paid_bps"] for r in sel]), 3)
        out["net"] = net_summ([r[net_key] for r in sel])
    return out


def run_a1(rows):
    def sel(pred):
        return [r for r in rows if pred(r)]

    arms = {
        "mom4_bottom_quintile": arm_block(
            sel(lambda r: r["mom_4h"] <= IQ), "net_long"),
        "mom4_rest": arm_block(
            sel(lambda r: r["mom_4h"] > IQ), "net_long"),
        "mom4_top_quintile": arm_block(
            sel(lambda r: r["mom_4h"] >= 1 - IQ), "net_short"),
        "mom4_not_top": arm_block(
            sel(lambda r: r["mom_4h"] < 1 - IQ), "net_long"),
        "mom1_bottom_quintile": arm_block(
            sel(lambda r: r["mom_1h"] <= IQ), "net_long"),
        "mom1_rest": arm_block(
            sel(lambda r: r["mom_1h"] > IQ), "net_long"),
        "mom1_top_quintile": arm_block(
            sel(lambda r: r["mom_1h"] >= 1 - IQ), "net_short"),
        "mom1_not_top": arm_block(
            sel(lambda r: r["mom_1h"] < 1 - IQ), "net_long"),
    }

    per_asset = {}
    for asset in sorted({r["asset"] for r in rows}):
        a = [r for r in rows
             if r["asset"] == asset and r["mom_4h"] <= IQ]
        per_asset[asset] = {
            "n": len(a),
            "mean_gross_bps": (round(statistics.mean(
                [r["fwd"] for r in a]), 2) if a else None),
            "median_gross_bps": (round(statistics.median(
                [r["fwd"] for r in a]), 2) if a else None),
            "mean_net_long_bps": (round(statistics.mean(
                [r["net_long"] for r in a]), 2) if a else None),
            "win_rate_net": (round(sum(
                1 for r in a if r["net_long"] > 0) / len(a), 3)
                if a else None),
            "share_of_arm": round(len(a) / arms[
                "mom4_bottom_quintile"]["n"], 3)
            if arms["mom4_bottom_quintile"]["n"] else None}

    contrast_defs = [
        ("A1_mom4_bottom_quintile_vs_rest",
         lambda r: r["mom_4h"] <= IQ, lambda r: r["mom_4h"] > IQ),
        ("A1_mom4_top_quintile_vs_rest",
         lambda r: r["mom_4h"] >= 1 - IQ, lambda r: r["mom_4h"] < 1 - IQ),
        ("A1_mom1_bottom_quintile_vs_rest",
         lambda r: r["mom_1h"] <= IQ, lambda r: r["mom_1h"] > IQ),
        ("A1_mom1_top_quintile_vs_rest",
         lambda r: r["mom_1h"] >= 1 - IQ, lambda r: r["mom_1h"] < 1 - IQ)]
    contrasts, named_ps = {}, []
    for name, pa, pb in contrast_defs:
        c = welch([r["fwd"] for r in rows if pa(r)],
                  [r["fwd"] for r in rows if pb(r)])
        contrasts[name] = c
        if c:
            named_ps.append((name, c["p"]))
    fdr_table, fdr_map = bh_fdr(named_ps)
    for name, c in contrasts.items():
        if c:
            c["fdr_survives"] = fdr_map.get(name, False)

    rt = 2 * FEE_BPS
    mb, mt = arms["mom4_bottom_quintile"], arms["mom4_top_quintile"]
    economics = {
        "fee_bps_per_side": FEE_BPS,
        "round_trip_cost_bps": rt,
        "hold_hours": HOLD_H,
        "funding_convention": "last_funding_rate*1e4 bps iff a settlement "
                              "lands inside the 4h hold (long pays, short "
                              "receives); pro-rata fallback on "
                              "null-distance records",
        "mom4_bottom_mean_gross_bps": mb["mean_gross_bps"],
        "mom4_bottom_mean_net_long_bps":
            mb["net"].get("mean_net_bps"),
        "mom4_bottom_net_long": mb["net"],
        "mom4_top_mean_gross_bps": mt["mean_gross_bps"],
        "mom4_top_mean_net_short_bps":
            mt["net"].get("mean_net_bps"),
        "gross_edge_exceeds_round_trip":
            (mb["mean_gross_bps"] is not None
             and mb["mean_gross_bps"] > rt),
        "net_long_positive":
            (mb["net"].get("mean_net_bps") is not None
             and mb["net"]["mean_net_bps"] > 0)}

    return {"quintile_edge": IQ, "arms": arms,
            "per_asset_mom4_bottom_quintile": per_asset,
            "contrasts": contrasts, "fdr_bh": fdr_table,
            "economics": economics}


# ---------------------------------------------------------------- arm A2
def run_a2(rows):
    """mom_4h reversal conditioned on rv_pct terciles and funding-sign
    regime. Per cell: n, Spearman(mom_4h, fwd), bottom-vs-rest Welch."""
    def cell_block(sub):
        bot = [r for r in sub if r["mom_4h"] <= IQ]
        rest = [r for r in sub if r["mom_4h"] > IQ]
        sp = spearman([r["mom_4h"] for r in sub],
                      [r["fwd"] for r in sub])
        return {"n": len(sub),
                "spearman_mom4_vs_fwd":
                    ({k: round(v, 5) for k, v in sp.items()}
                     if sp else None),
                "mom4_bottom_quintile": summ([r["fwd"] for r in bot]),
                "mom4_bottom_net_long": net_summ(
                    [r["net_long"] for r in bot]),
                "mom4_rest": summ([r["fwd"] for r in rest]),
                "tilt_bps": (round(
                    statistics.mean([r["fwd"] for r in bot])
                    - statistics.mean([r["fwd"] for r in rest]), 2)
                    if len(bot) >= MIN_N and len(rest) >= MIN_N
                    else None),
                "welch_bottom_vs_rest": welch(
                    [r["fwd"] for r in bot], [r["fwd"] for r in rest])}

    grids = {"rv_pct_tercile": {}, "funding_regime": {}}
    for t in ("low", "mid", "high"):
        grids["rv_pct_tercile"][t] = cell_block(
            [r for r in rows if tercile(r["rv_pct"]) == t])
    for g in ("positive", "mixed", "negative"):
        grids["funding_regime"][g] = cell_block(
            [r for r in rows if r["fund_regime"] == g])

    named_ps, contrasts = [], {}
    for t in ("low", "mid", "high"):
        name = "A2_rv_%s_mom4_bottom_vs_rest" % t
        w = grids["rv_pct_tercile"][t]["welch_bottom_vs_rest"]
        contrasts[name] = w
        if w:
            named_ps.append((name, w["p"]))
    for g in ("positive", "mixed", "negative"):
        name = "A2_fundreg_%s_mom4_bottom_vs_rest" % g
        w = grids["funding_regime"][g]["welch_bottom_vs_rest"]
        contrasts[name] = w
        if w:
            named_ps.append((name, w["p"]))
    fdr_table, fdr_map = bh_fdr(named_ps)
    for name, c in contrasts.items():
        if c:
            c["fdr_survives"] = fdr_map.get(name, False)

    return {"grids": grids, "contrasts": contrasts,
            "fdr_bh": fdr_table}


# ---------------------------------------------------------------- arm A3
def placebo_rhos(sub, key):
    """Shuffle fwd within asset; rho of score vs shuffled label."""
    xs = [r[key] for r in sub]
    by_asset = defaultdict(list)
    for r in sub:                       # sub ordered asset-major
        by_asset[r["asset"]].append(r["fwd"])
    out = []
    for seed in PLACEBO_SEEDS:
        shuffled = []
        for asset, vals in sorted(by_asset.items()):
            shuffled.extend(shuffle(vals, seed))
        sp = spearman(xs, shuffled)
        out.append(round(sp["rho"], 4) if sp else None)
    return out


def run_a3(rows):
    clean = [r for r in rows if not r["contam"]]
    cells = {}
    for name in A3_CELLS:
        entry = {}
        for tag, sub in (("full", rows), ("clean", clean)):
            sp = spearman([r[name] for r in sub],
                          [r["fwd"] for r in sub])
            pl = placebo_rhos(sub, name)
            entry[tag] = {
                "n": len(sub),
                "spearman": ({k: round(v, 5) for k, v in sp.items()}
                             if sp else None),
                "placebo_rhos": pl,
                "placebo_max_abs_rho": max(
                    (abs(x) for x in pl if x is not None), default=None)}
        cells[name] = entry

    # BH-FDR over each 3-cell family (separate per set, benchmark conv.)
    verdicts = {}
    fdr = {}
    for tag in ("full", "clean"):
        table, fmap = bh_fdr(
            [(n, cells[n][tag]["spearman"]["p"]) for n in A3_CELLS
             if cells[n][tag]["spearman"]])
        fdr[tag] = table
        for n in A3_CELLS:
            c = cells[n][tag]
            rho = c["spearman"]["rho"] if c["spearman"] else None
            band = c["placebo_max_abs_rho"]
            pc = (rho is not None and band is not None
                  and abs(rho) > max(0.02, band))
            verdicts.setdefault(n, {})[tag] = {
                "fdr_pass": fmap.get(n, False),
                "placebo_clear": pc,
                "verdict": "PASS" if (fmap.get(n, False) and pc)
                           else "FAIL"}

    parity = {}
    if BENCH.exists():
        bcells = json.loads(BENCH.read_text())["intraday"]["cells"]
        for n in A3_CELLS:
            stored = bcells[n]["primary"]["spearman"]["rho"]
            mine = cells[n]["full"]["spearman"]["rho"]
            parity[n] = {"benchmark_v2_rho": stored,
                         "recomputed_rho": mine,
                         "abs_diff": round(abs(stored - mine), 8)}

    for n in A3_CELLS:
        f_, c_ = cells[n]["full"], cells[n]["clean"]
        cells[n]["delta_rho_clean_minus_full"] = round(
            c_["spearman"]["rho"] - f_["spearman"]["rho"], 5)
        cells[n]["verdict_full"] = verdicts[n]["full"]
        cells[n]["verdict_clean"] = verdicts[n]["clean"]
        cells[n]["verdict_changed"] = (
            verdicts[n]["full"]["verdict"]
            != verdicts[n]["clean"]["verdict"])

    return {"exclusion_rule": "contaminated = ns_until_next_funding_"
                              "settlement <= 2h (pre) or >= 6h (post); "
                              "null distance -> decision_ms mod 8h grid",
            "n_full": len(rows), "n_clean": len(clean),
            "n_excluded": len(rows) - len(clean),
            "excluded_fraction": round(
                (len(rows) - len(clean)) / len(rows), 4),
            "cells": cells, "fdr_bh_full": fdr["full"],
            "fdr_bh_clean": fdr["clean"],
            "benchmark_v2_parity": parity}


# ---------------------------------------------------------------- run
def run():
    n_rec, rows, n_null_dist, n_agree = build_rows()
    a1 = run_a1(rows)
    a2 = run_a2(rows)
    a3 = run_a3(rows)

    # ---------------- A4: honest verdicts ----------------
    econ = a1["economics"]
    c1 = a1["contrasts"]["A1_mom4_bottom_quintile_vs_rest"]
    if econ["net_long_positive"]:
        rev_v = ("reversal_long_edge_positive_net_of_fees: mom_4h "
                 "bottom-quintile mean gross {:.2f}bps > {:.0f}bps round "
                 "trip; mean net long {:.1f}bps/trade (n={}) — see "
                 "overlap caveats").format(
                     econ["mom4_bottom_mean_gross_bps"],
                     econ["round_trip_cost_bps"],
                     econ["mom4_bottom_mean_net_long_bps"],
                     econ["mom4_bottom_net_long"]["n"])
    elif econ["gross_edge_exceeds_round_trip"]:
        rev_v = ("reversal_edge_gross_but_funding_eroded: mom_4h "
                 "bottom-quintile mean gross {:.2f}bps exceeds the "
                 "{:.0f}bps round trip but mean net long is {:.1f}bps — "
                 "scheduled funding drag eats the edge").format(
                     econ["mom4_bottom_mean_gross_bps"],
                     econ["round_trip_cost_bps"],
                     econ["mom4_bottom_mean_net_long_bps"])
    else:
        rev_v = ("reversal_sub_fee_noise: mom_4h bottom-quintile mean "
                 "gross {:.2f}bps is below the {:.0f}bps round-trip "
                 "cost (mean net long {:.1f}bps/trade); bottom-vs-rest "
                 "tilt {:+.2f}bps p={} fdr_survives={} — the "
                 "FDR+placebo-passing rho is not an executable edge "
                 "at 5bps/side").format(
                     econ["mom4_bottom_mean_gross_bps"],
                     econ["round_trip_cost_bps"],
                     econ["mom4_bottom_mean_net_long_bps"],
                     c1["diff_bps"], c1["p"], c1["fdr_survives"])

    rv_g = a2["grids"]["rv_pct_tercile"]
    fr_g = a2["grids"]["funding_regime"]
    rv_tilts = {t: rv_g[t]["tilt_bps"] for t in ("low", "mid", "high")}
    fr_tilts = {g: fr_g[g]["tilt_bps"] for g in
                ("positive", "mixed", "negative")}
    rv_rhos = {t: (rv_g[t]["spearman_mom4_vs_fwd"]["rho"]
                   if rv_g[t]["spearman_mom4_vs_fwd"] else None)
               for t in ("low", "mid", "high")}
    fr_rhos = {g: (fr_g[g]["spearman_mom4_vs_fwd"]["rho"]
                   if fr_g[g]["spearman_mom4_vs_fwd"] else None)
               for g in ("positive", "mixed", "negative")}
    rv_mono = (rv_tilts["low"] is not None
               and rv_tilts["high"] is not None
               and rv_tilts["high"] > rv_tilts["low"])
    fr_stronger_neg = (fr_tilts["negative"] is not None
                       and fr_tilts["positive"] is not None
                       and fr_tilts["negative"] > fr_tilts["positive"])
    neg_net = (fr_g["negative"]["mom4_bottom_net_long"] or {}).get(
        "mean_net_bps")
    cond_v = ("rv-tercile tilts (bottom-vs-rest bps): low={} mid={} "
              "high={} — {} in high vol; funding-regime tilts: "
              "positive={} mixed={} negative={} — {} in "
              "negative-funding regime; per-cell rhos rv={} "
              "fundreg={}; strongest cell economics: negative-regime "
              "bottom quintile mean net long {}bps/trade "
              "(gross {}bps vs {}bps round trip)").format(
                  rv_tilts["low"], rv_tilts["mid"], rv_tilts["high"],
                  "stronger" if rv_mono else "not stronger",
                  fr_tilts["positive"], fr_tilts["mixed"],
                  fr_tilts["negative"],
                  "stronger" if fr_stronger_neg else "not stronger",
                  {k: round(v, 4) if v is not None else None
                   for k, v in rv_rhos.items()},
                  {k: round(v, 4) if v is not None else None
                   for k, v in fr_rhos.items()},
                  neg_net,
                  (fr_g["negative"]["mom4_bottom_quintile"] or {})
                  .get("mean_bps"),
                  2 * FEE_BPS)

    flips = []
    for n in A3_CELLS:
        if not a3["cells"][n]["verdict_changed"]:
            continue
        vf, vc = (a3["cells"][n]["verdict_full"],
                  a3["cells"][n]["verdict_clean"])
        mech = []
        if vf["fdr_pass"] != vc["fdr_pass"]:
            mech.append("fdr")
        if vf["placebo_clear"] != vc["placebo_clear"]:
            mech.append("placebo")
        flips.append("{}:{}->{}({})".format(
            n, vf["verdict"], vc["verdict"], "+".join(mech) or "?"))
    deltas = {n: a3["cells"][n]["delta_rho_clean_minus_full"]
              for n in A3_CELLS}
    max_delta = max(abs(v) for v in deltas.values())
    contam_v = ("excluding the +-2h settlement window ({}/{} rows, "
                "cycle hours 0,1,6,7) moved rho by at most {:.4f} "
                "(deltas {}); verdict flips: {} -> {}").format(
                    a3["n_excluded"], a3["n_full"], max_delta,
                    {k: round(v, 4) for k, v in deltas.items()},
                    flips if flips else "none",
                    "contamination changes a verdict"
                    if flips else
                    "no verdict changes — signals are not a "
                    "settlement-window artifact")

    out = {
        "n_records": n_rec,
        "n_eval_rows": len(rows),
        "warmup_dropped": n_rec - len(rows),
        "n_null_distance": n_null_dist,
        "dist_vs_mod_flag_agreement": {
            "n_checked": len(rows) - n_null_dist,
            "n_agree": n_agree,
            "note": "distance-rule and mod-grid-rule flags agree on "
                    "every non-null record iff n_agree == n_checked"},
        "cycle_hour_counts": {str(c): sum(
            1 for r in rows if r["cyc"] == c) for c in range(8)},
        "label_stats_bps": {
            "mean": round(statistics.mean([r["fwd"] for r in rows]), 3),
            "sd": round(statistics.pstdev([r["fwd"] for r in rows]), 3)},
        "a1_reversal_economics": a1,
        "a2_reversal_conditioning": a2,
        "a3_settlement_contamination": a3,
        "a4_verdicts": {
            "reversal_economics": rev_v,
            "conditioning": cond_v,
            "settlement_contamination": contam_v,
            "c1_contrast": c1},
        "caveats": [
            "4h forward labels on 1h bars overlap ~4x and cross-asset "
            "bars share timestamps; Welch/Spearman treat rows as "
            "independent so p-values are nominal/optimistic",
            "A1 net stats assume each bar is a round-trip trade; "
            "overlapping 4h trades are not independent and "
            "total_net_bps is an accounting sum, not an equity curve",
            "A3's clean set is a mid-cycle subsample (~50% of rows); "
            "rho changes reflect contamination removal AND regime "
            "selection — interpret deltas jointly",
            "gross mark move minus fees/scheduled funding only — no "
            "spread, slippage, or impact; not a tradability claim"]}

    return {"schema_version": "nanojev-financial-signal-reversal-v1",
            "status": "measurement_complete",
            "task": "T96",
            "protocol": PROTOCOL,
            "scope": "T96 hourly momentum-reversal economics + "
                     "settlement-window contamination on the hourly PIT "
                     "cohort; measurement only; overlapping 4h labels "
                     "and cross-asset timestamp correlation make test "
                     "statistics nominal (optimistic)",
            "parameters": {"pct_trailing_records_per_asset": TRAIL,
                           "mom4_lag_bars": MOM4_LAG,
                           "hold_hours": HOLD_H,
                           "quintile": IQ,
                           "welch_min_n": MIN_N,
                           "fee_bps_per_side": FEE_BPS,
                           "settlement_grid_ms": SETTLE_MS,
                           "exclusion_pre_ns": PRE_NS,
                           "exclusion_post_ns": POST_NS,
                           "excluded_cycle_hours": [0, 1, 6, 7],
                           "funding_regime_thresholds":
                               {"positive": POS_HI, "negative": POS_LO},
                           "placebo_seeds": list(PLACEBO_SEEDS),
                           "fdr_alpha": 0.05,
                           "fdr_families": {"a1_contrasts": 4,
                                            "a2_contrasts": 6,
                                            "a3_cells_per_set": 3}},
            "results": out}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    r1, r2 = run(), run()
    same = json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    r1["determinism"] = {"replays": 2, "byte_identical": same}
    r1["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                     time.gmtime())
    blob = json.dumps(r1, indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(blob)
    print(json.dumps({"output": str(args.output), "deterministic": same,
                      "sha256": hashlib.sha256(blob.encode()).hexdigest()},
                     indent=2))


if __name__ == "__main__":
    main()
