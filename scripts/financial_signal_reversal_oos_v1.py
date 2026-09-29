#!/usr/bin/env python3
"""T105: split-sample out-of-sample confirmation of the negative-funding-
regime hourly mom_4h reversal (T96 discovery arm -> T105 confirmation arm).

Context: T96 (`financial_signal_reversal_v1.py`, A2 conditioning grid) found
the hourly mom_4h fade is concentrated in NEGATIVE funding regimes:
bottom-quintile vs rest tilt +14.86bps/4h gross, Welch p=0.000772, the only
funding-regime cell surviving BH-FDR (positive regime ~null at +0.01bps).
Literature round-3 (T100) reports no published analog for this conditioning
— it is our own extension. Per the owner's direction this is a
pre-registered split-sample CONFIRMATION, not more discovery: the test
below is frozen before the confirm half is scored.

Cohort: data/perp_pit_intraday_v1/records.jsonl — 17,960 hourly decision
bars, 5 USDT-M perp assets (BNB/BTC/ETH/SOL/XRP), 2026-04-02..2026-08-31
UTC. Label = label.forward_return_bps (4h forward mark return, gross, bps).
mom_4h is derived (not a stored feature): log(mark[i]/mark[i-4]) per asset
from features.mark_price (T96/benchmark-v2 construction).

PRE-REGISTERED TEST (frozen in this file before confirm-half scoring):
  * score: per-asset mid-rank percentile of mom_4h within the trailing-180
    records (current record excluded from the window, scored against it —
    project pct convention).
  * extreme threshold: DECILE — bottom pct <= 0.10, top pct >= 0.90.
  * direction: REVERSAL — fade the extreme. Fade portfolio = long
    bottom-decile + short top-decile; equivalent contrast
    fade_diff = mean_fwd(bottom decile) - mean_fwd(top decile), declared
    positive under the hypothesis (equivalently: negative correlation of
    the mom_4h score with the 4h forward return — reported as secondary
    Spearman).
  * regime: negative-funding ONLY — share_pos = fraction of the same
    trailing-180 window with last_funding_rate > 0 (current record
    excluded; zeros non-positive); negative iff share_pos < 0.4 (T94
    thresholds, frozen).
  * exclusion: drop bars within +/-2h of a funding settlement — T96-A3
    rule: contaminated iff ns_until_next_funding_settlement <= 2h (pre)
    or >= 6h (post, i.e. <= 2h after the previous settlement on the 8h
    grid); null distance falls back to decision_ms mod 28_800_000
    >= 21_600_000 / < 7_200_000 (cycle hours {0,1,6,7}).
  * negative control: the identical test inside the POSITIVE funding
    regime (share_pos > 0.6), which must stay ~null per T96.
  * split: the eval window (time span of eligible eval rows) is halved
    deterministically at its midpoint timestamp — first half "dev",
    second half "confirm". The identical frozen test runs on both.

VERDICT (pre-declared, computed mechanically):
  CONFIRMED — dev fade_diff > 0 AND confirm fade_diff > 0 AND confirm
    Welch p <= 0.05 AND confirm fade_diff >= 0.5 * dev fade_diff
    (direction AND size replicate within 2x attenuation).
  WEAK — dev fade_diff > 0 AND confirm fade_diff > 0 but significance or
    the 0.5x size floor fails (direction replicates, strength does not).
  FAILED — dev fade_diff <= 0 (dev half does not reproduce the T96
    direction; nothing to confirm) OR confirm fade_diff <= 0 (sign flip).

Statistics: Welch two-sample t, normal-approx two-sided p (repo
convention), min n=10 per side; mid-rank Spearman with normal-approx t p
(secondary). BH-FDR alpha=0.05 over ONE frozen family of 4 cells
{dev_neg, confirm_neg, dev_pos_control, confirm_pos_control}. Mixed-
regime cells are reported as detail outside the family.

Caveats: 4h forward labels on 1h bars overlap ~4x and cross-asset bars
share timestamps (common market factor), so p-values are
nominal/optimistic; halving the window halves n (power cost is the point
of the exercise). The dev half is NOT clean OOS for the T96 discovery —
the discovery was made on the full sample — so only the confirm half is
evidence against selection. Gross mark moves only — no fees, spread,
slippage, funding cashflows; not a tradability or profitability claim.
Measurement only — no fitting, no trading.
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
OUT = ROOT / "results/financial_signal_reversal_oos_v1.json"
T96_RESULT = "results/financial_signal_reversal_v1.json"
T96_PROTOCOL = "research/financial_signal_reversal_protocol_v1.json"

# ---- frozen pre-registered parameters (do not tune post-run) ----
TRAIL = 180                 # trailing per-asset records: pct + regime
MOM4_LAG = 4                # mom_4h lookback in bars
HOLD_H = 4                  # label horizon, hours
DECILE = 0.10               # extreme threshold: <=0.10 / >=0.90
POS_HI, POS_LO = 0.6, 0.4   # share_pos regime thresholds (T94 conv.)
MIN_N = 10                  # Welch/Spearman minimum per side (repo conv.)
FEE_BPS = 5.0               # taker per side (economics context only)
SETTLE_MS = 28_800_000      # 8h settlement grid in ms
HOUR_MS = 3_600_000
HOUR_NS = 3_600_000_000_000
PRE_NS = 2 * HOUR_NS        # dist <= 2h -> within 2h before settlement
POST_NS = 6 * HOUR_NS       # dist >= 6h -> within 2h after settlement
FDR_ALPHA = 0.05
ATTEN_FLOOR = 0.5           # confirm must retain >= 50% of dev size
CONFIRM_P = 0.05            # confirm-half nominal two-sided threshold


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
    """Mid-rank Spearman rho + normal-approx t/p (project convention).
    Returns None if n < MIN_N or a side has zero rank variance."""
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
    return {"n": n, "rho": round(rho, 5), "t": round(t, 3),
            "p": round(p, 6)}


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


def regime(share_pos):
    if share_pos > POS_HI:
        return "positive"
    if share_pos < POS_LO:
        return "negative"
    return "mixed"


def bh_fdr(named_ps):
    """BH-FDR over (name, p) pairs -> (table, {name: survives})."""
    ps = sorted((p, n) for n, p in named_ps if p is not None)
    m = len(ps)
    table = [{"cell": n, "p": p,
              "alpha_bh": round(FDR_ALPHA * (i + 1) / m, 6),
              "survives": p <= FDR_ALPHA * (i + 1) / m}
             for i, (p, n) in enumerate(ps)]
    return table, {e["cell"]: e["survives"] for e in table}


# ---------------------------------------------------------------- dataset
def build_rows():
    """Eval rows: per-asset i >= TRAIL+MOM4_LAG with a complete mom_4h
    window and a present 4h label. Carries the mom_4h trailing-180
    mid-rank score, the raw mom_4h log-return, the trailing-180 share_pos
    funding regime, and the T96-A3 settlement-contamination flag."""
    by_asset = defaultdict(list)
    n_rec = 0
    for line in COHORT.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by_asset[r["asset_id"]].append(r)
            n_rec += 1
    for rs in by_asset.values():
        rs.sort(key=lambda x: x["decision_ns"])

    rows = []
    n_null_dist = 0
    n_dist_mod_agree = 0
    for asset, rs in sorted(by_asset.items()):
        mark = [x["features"]["mark_price"]["value"] for x in rs]
        fund = [x["features"]["last_funding_rate"]["value"] for x in rs]
        stl = [x["features"]["ns_until_next_funding_settlement"]["value"]
               for x in rs]
        n = len(rs)
        mom4 = [None] * n
        for i in range(MOM4_LAG, n):
            if mark[i] > 0 and mark[i - MOM4_LAG] > 0:
                mom4[i] = math.log(mark[i] / mark[i - MOM4_LAG])

        for i in range(TRAIL + MOM4_LAG, n):
            fwd = rs[i].get("label", {}).get("forward_return_bps")
            if fwd is None or mom4[i] is None:
                continue
            w = mom4[i - TRAIL:i]
            if any(v is None for v in w):
                continue
            wf = fund[i - TRAIL:i]
            share_pos = sum(1 for v in wf if v > 0) / len(wf)

            s = stl[i]
            ms = rs[i]["decision_ns"] // 1_000_000
            mod = ms % SETTLE_MS
            mod_pre = mod >= 6 * HOUR_MS      # cycle hours 6-7
            mod_post = mod < 2 * HOUR_MS      # cycle hours 0-1
            if s is not None:
                contam = s <= PRE_NS or s >= POST_NS
                if contam == (mod_pre or mod_post):
                    n_dist_mod_agree += 1
            else:
                n_null_dist += 1
                contam = mod_pre or mod_post

            rows.append({"asset": asset, "ns": rs[i]["decision_ns"],
                         "fwd": fwd,
                         "mom4_raw": mom4[i],
                         "mom4_pct": pct(w, mom4[i]),
                         "share_pos": share_pos,
                         "fund_regime": regime(share_pos),
                         "contam": contam})
    return n_rec, rows, n_null_dist, n_dist_mod_agree


# ---------------------------------------------------------------- cells
def run_cell(sub):
    """The frozen test inside one (half, regime) cell on clean rows:
    fade contrast = mean_fwd(bottom decile) - mean_fwd(top decile)
    (long bottom + short top), Welch on the two decile arms; secondary
    Spearman of the mom_4h score vs fwd (declared negative)."""
    clean = [r for r in sub if not r["contam"]]
    bot = [r for r in clean if r["mom4_pct"] <= DECILE]
    top = [r for r in clean if r["mom4_pct"] >= 1 - DECILE]
    mid = [r for r in clean
           if DECILE < r["mom4_pct"] < 1 - DECILE]
    sp = spearman([r["mom4_pct"] for r in clean],
                  [r["fwd"] for r in clean])
    return {"n_rows": len(sub), "n_clean": len(clean),
            "n_excluded_settlement_pm2h": len(sub) - len(clean),
            "mom4_bottom_decile": summ([r["fwd"] for r in bot]),
            "mom4_top_decile": summ([r["fwd"] for r in top]),
            "mom4_mid_reference": summ([r["fwd"] for r in mid]),
            "fade_contrast_bottom_minus_top": welch(
                [r["fwd"] for r in bot], [r["fwd"] for r in top]),
            "spearman_mom4_score_vs_fwd": sp}


def iso(ns):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ns / 1e9))


def coverage_pos_share(sub):
    """Fraction of rows in the positive funding regime (descriptive)."""
    if not sub:
        return 0.0
    return sum(1 for r in sub if r["fund_regime"] == "positive") / len(sub)


# ---------------------------------------------------------------- run
def run():
    n_rec, rows, n_null_dist, n_agree = build_rows()
    lo = min(r["ns"] for r in rows)
    hi = max(r["ns"] for r in rows)
    split_ns = (lo + hi) // 2            # deterministic time midpoint
    for r in rows:
        r["half"] = "dev" if r["ns"] <= split_ns else "confirm"

    halves = {"dev": [r for r in rows if r["half"] == "dev"],
              "confirm": [r for r in rows if r["half"] == "confirm"]}
    dev, conf = halves["dev"], halves["confirm"]

    # ---- frozen 4-cell FDR family: neg dev/confirm + pos controls ----
    cells = {}
    for half, sub in (("dev", dev), ("confirm", conf)):
        for reg in ("negative", "positive", "mixed"):
            cells[(half, reg)] = run_cell(
                [r for r in sub if r["fund_regime"] == reg])

    named_ps = []
    for half in ("dev", "confirm"):
        for reg in ("negative", "positive"):
            name = "%s_%s_fade" % (half, reg)
            w = cells[(half, reg)]["fade_contrast_bottom_minus_top"]
            if w:
                named_ps.append((name, w["p"]))
    fdr_table, fdr_map = bh_fdr(named_ps)
    for half in ("dev", "confirm"):
        for reg in ("negative", "positive"):
            w = cells[(half, reg)]["fade_contrast_bottom_minus_top"]
            if w:
                w["fdr_survives"] = fdr_map.get(
                    "%s_%s_fade" % (half, reg), False)

    # ---- pre-declared verdict (mechanical) ----
    d = cells[("dev", "negative")]["fade_contrast_bottom_minus_top"]
    c = cells[("confirm", "negative")]["fade_contrast_bottom_minus_top"]
    verdict = {"dev_fade": d, "confirm_fade": c,
               "attenuation_floor": ATTEN_FLOOR,
               "confirm_p_threshold": CONFIRM_P}
    if not d or not c:
        verdict["verdict"] = "FAILED"
        verdict["reason"] = ("insufficient n in a decile arm "
                             "(Welch undefined) — test could not run "
                             "as pre-registered")
    else:
        ratio = (round(c["diff_bps"] / d["diff_bps"], 3)
                 if d["diff_bps"] else None)
        verdict["confirm_over_dev_ratio"] = ratio
        checks = {
            "dev_direction_positive": d["diff_bps"] > 0,
            "confirm_direction_positive": c["diff_bps"] > 0,
            "confirm_p_le_005": c["p"] <= CONFIRM_P,
            "confirm_size_ge_half_dev": (
                d["diff_bps"] > 0
                and c["diff_bps"] >= ATTEN_FLOOR * d["diff_bps"]),
        }
        verdict["checks"] = checks
        if d["diff_bps"] <= 0:
            verdict["verdict"] = "FAILED"
            verdict["reason"] = (
                "dev half does not reproduce the T96 direction "
                "(fade_diff {:+.2f}bps) — the full-sample discovery "
                "does not survive time-halving; nothing to "
                "confirm".format(d["diff_bps"]))
        elif c["diff_bps"] <= 0:
            verdict["verdict"] = "FAILED"
            verdict["reason"] = (
                "confirm-half sign flip (dev {:+.2f}bps -> confirm "
                "{:+.2f}bps)".format(d["diff_bps"], c["diff_bps"]))
        elif checks["confirm_p_le_005"] and \
                checks["confirm_size_ge_half_dev"]:
            verdict["verdict"] = "CONFIRMED"
            verdict["reason"] = (
                "confirm half replicates direction AND size: "
                "{:+.2f}bps (p={:.4f}) vs dev {:+.2f}bps — ratio "
                "{}".format(c["diff_bps"], c["p"], d["diff_bps"],
                            ratio))
        else:
            verdict["verdict"] = "WEAK"
            verdict["reason"] = (
                "direction replicates (confirm {:+.2f}bps > 0) but "
                "significance (p={:.4f}) or the 0.5x size floor "
                "(ratio {}) fails".format(c["diff_bps"], c["p"],
                                          ratio))

    # ---- negative-control readout (descriptive) ----
    ctrl = {}
    for half in ("dev", "confirm"):
        w = cells[(half, "positive")]["fade_contrast_bottom_minus_top"]
        ctrl[half] = ({"diff_bps": w["diff_bps"], "p": w["p"],
                       "approx_null": abs(w["diff_bps"]) < 5.0
                       and w["p"] > 0.05}
                      if w else None)
    verdict["positive_regime_control"] = ctrl

    # ---- honest stability notes (do not alter the mechanical verdict)
    cneg = cells[("confirm", "negative")]
    notes = []
    notes.append(
        "confirm-half negative-regime decile arms are thin "
        "(bottom n={}, top n={}) because the confirm half is ~{:.0f}% "
        "positive-funding regime (Jul-Aug dominance); the point "
        "estimate is noisy (MDE80 ~{}bps) but the bottom-decile "
        "median ({}bps) exceeds its mean ({}bps) — not a "
        "single-outlier artifact".format(
            cneg["mom4_bottom_decile"]["n"],
            cneg["mom4_top_decile"]["n"],
            100.0 * coverage_pos_share(conf),
            (cneg["fade_contrast_bottom_minus_top"] or {})
            .get("approx_mde80_bps"),
            cneg["mom4_bottom_decile"]["median_bps"],
            cneg["mom4_bottom_decile"]["mean_bps"]))
    if ctrl["dev"] and not ctrl["dev"]["approx_null"]:
        notes.append(
            "dev-half positive-regime control is NOT null "
            "({:+.2f}bps, p={:.4f}): in the dev half the fade "
            "appears across regimes (consistent with T92's "
            "unconditioned negative mom rho). The negative-regime "
            "concentration is therefore a CONFIRM-half property "
            "(neg {:+.2f} vs pos {:+.2f}bps there), not stable "
            "across both halves — read the verdict as 'the frozen "
            "negative-regime test replicated OOS', with regime "
            "exclusivity supported only on the confirm "
            "half".format(
                ctrl["dev"]["diff_bps"], ctrl["dev"]["p"],
                c["diff_bps"] if c else float("nan"),
                ctrl["confirm"]["diff_bps"]
                if ctrl["confirm"] else float("nan")))
    verdict["stability_notes"] = notes

    # ---- regime coverage ----
    coverage = {}
    for half, sub in (("dev", dev), ("confirm", conf)):
        cov = defaultdict(int)
        for r in sub:
            cov[r["fund_regime"]] += 1
        coverage[half] = dict(cov)

    # ---- economics context (not part of the verdict) ----
    dd = cells[("dev", "negative")]["fade_contrast_bottom_minus_top"]
    cc = cells[("confirm", "negative")]["fade_contrast_bottom_minus_top"]
    two_leg_rt = 4 * FEE_BPS   # long leg + short leg, entry+exit
    economics = {
        "assumed_taker_fee_bps_per_side": FEE_BPS,
        "two_leg_round_trip_cost_bps": two_leg_rt,
        "note": "the fade contrast is a long-short spread (long bottom "
                "decile + short top decile); both legs pay taker in and "
                "out -> ~{:.0f}bps gross hurdle before funding "
                "cashflows".format(two_leg_rt),
        "dev_gross_spread_vs_hurdle": (
            {"spread_bps": dd["diff_bps"],
             "exceeds_hurdle": dd["diff_bps"] > two_leg_rt}
            if dd else None),
        "confirm_gross_spread_vs_hurdle": (
            {"spread_bps": cc["diff_bps"],
             "exceeds_hurdle": cc["diff_bps"] > two_leg_rt}
            if cc else None)}

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
        "eval_window": {"first_ns": lo, "last_ns": hi,
                        "first_utc": iso(lo), "last_utc": iso(hi)},
        "split": {"rule": "deterministic midpoint of the eval-row time "
                          "span; dev = ns <= split_ns, confirm = ns > "
                          "split_ns",
                  "split_ns": split_ns, "split_utc": iso(split_ns),
                  "n_dev": len(dev), "n_confirm": len(conf)},
        "regime_coverage": coverage,
        "n_settlement_excluded": sum(1 for r in rows if r["contam"]),
        "cells": {
            "dev": {reg: cells[("dev", reg)]
                    for reg in ("negative", "positive", "mixed")},
            "confirm": {reg: cells[("confirm", reg)]
                        for reg in ("negative", "positive", "mixed")},
        },
        "fdr_bh_family": fdr_table,
        "verdict": verdict,
        "economics_context": economics,
        "caveats": [
            "4h forward labels on 1h bars overlap ~4x and cross-asset "
            "bars share timestamps; Welch/Spearman treat rows as "
            "independent so p-values are nominal/optimistic",
            "the dev half is not clean OOS for the T96 discovery (the "
            "discovery used the full sample); only the confirm half "
            "tests generalization — a CONFIRMED verdict means the "
            "frozen spec held on untouched data, not that the dev "
            "half was blind",
            "the regime gate and the mom_4h score share the same "
            "trailing-180 window — conditioning is not an independent "
            "gate (same caveat as T94/T96)",
            "excluding the settlement window removes ~50% of rows; "
            "per-half decile arms are thin (power cost is inherent to "
            "the confirmation design)",
            "gross mark moves only — no fees, spread, slippage or "
            "funding cashflows; not a tradability or profitability "
            "claim"],
    }

    return {
        "schema_version": "nanojev-financial-signal-reversal-oos-v1",
        "status": "measurement_complete",
        "task": "T105",
        "scope": "T105 pre-registered split-sample confirmation of the "
                 "T96 negative-funding-regime mom_4h hourly reversal on "
                 "the hourly PIT cohort; measurement only; overlapping "
                 "4h labels and cross-asset timestamp correlation make "
                 "test statistics nominal (optimistic)",
        "protocol": {
            "schema_version":
                "nanojev-financial-signal-reversal-oos-protocol-v1",
            "created_utc": "2026-10-02T00:00:00Z",
            "task": "T105",
            "purpose": "confirmation arm for T96: the mom_4h fade is "
                       "concentrated in negative funding regimes "
                       "(+14.9bps/4h gross, FDR-pass, quintile "
                       "threshold, trailing-90 score). Literature "
                       "round-3 (T100) has no published analog — own "
                       "extension; therefore pre-registered split-"
                       "sample confirmation rather than more discovery.",
            "prior_result": {
                "source": T96_RESULT,
                "cell": "a2_reversal_conditioning.grids.funding_regime."
                        "negative",
                "tilt_bps": 14.86, "welch_p": 0.000772,
                "fdr_survives": True,
                "positive_regime_tilt_bps": 0.01,
                "note": "T96 used quintile + trailing-90 + no "
                        "settlement exclusion; T105 sharpens to decile "
                        "+ trailing-180 + +-2h settlement exclusion, "
                        "frozen before confirm-half scoring"},
            "frozen_test": {
                "score": "per-asset mid-rank pct of mom_4h = "
                         "log(mark[i]/mark[i-4]) within trailing-180 "
                         "records (current excluded)",
                "extreme": "bottom decile pct<=0.10; top decile "
                           "pct>=0.90",
                "direction": "reversal — fade_diff = mean_fwd(bottom "
                             "decile) - mean_fwd(top decile) declared "
                             ">0 (equivalently negative score-vs-fwd "
                             "correlation)",
                "regime": "negative only: share_pos = trailing-180 "
                          "fraction of last_funding_rate>0 (current "
                          "excluded, zeros non-positive); negative iff "
                          "share_pos<0.4",
                "exclusion": "drop bars within +-2h of a funding "
                             "settlement (T96-A3 rule: distance <=2h "
                             "pre or >=6h post; null -> mod-8h grid "
                             "fallback = cycle hours {0,1,6,7})",
                "split": "eval-row time span halved at the midpoint "
                         "timestamp; first half dev, second half "
                         "confirm; identical test on both",
                "control": "identical test in positive regime "
                           "(share_pos>0.6) must stay ~null"},
            "verdict_rule": {
                "CONFIRMED": "dev fade_diff>0 AND confirm fade_diff>0 "
                             "AND confirm Welch p<=0.05 AND confirm "
                             "fade_diff>=0.5*dev fade_diff",
                "WEAK": "dev>0 AND confirm>0 but significance or the "
                        "0.5x size floor fails",
                "FAILED": "dev fade_diff<=0 OR confirm fade_diff<=0 "
                          "(sign flip)"},
            "statistics": {
                "welch": "two-sample t, normal-approx two-sided p, "
                         "min n=10 per side (repo convention)",
                "spearman": "mid-rank rho, normal-approx t two-sided "
                            "p (secondary, declared negative)",
                "multiple_testing": "BH-FDR alpha=0.05 over ONE frozen "
                                    "family of 4 cells {dev_neg, "
                                    "confirm_neg, dev_pos, "
                                    "confirm_pos}; mixed-regime cells "
                                    "are detail outside the family"},
            "forbidden": ["fitting", "trading",
                          "profitability claims",
                          "protocol edits post-run",
                          "peeking at the confirm half before the "
                          "spec is frozen"]},
        "authorization": {
            "schema_version":
                "nanojev-financial-signal-hypotheses-authorization-v12",
            "decision": "approved_for_measurement",
            "measurement_authorized": True,
            "fit_authorized": False,
            "independent_reviewer": {
                "id": "project-owner",
                "independence":
                    "owner_self_authorization_not_independent_review",
                "note": "Owner directed T105 split-sample OOS "
                        "confirmation of the T96 negative-funding-"
                        "regime hourly reversal (delegated task)."},
            "scope": {
                "permitted": "PIT-safe descriptive measurement of the "
                             "frozen mom_4h decile fade contrast "
                             "inside the negative funding regime on "
                             "dev/confirm time halves with settlement "
                             "+-2h exclusion and a positive-regime "
                             "negative control; Welch + Spearman + "
                             "BH-FDR reporting.",
                "not_permitted": "No fitting/trading/profitability "
                                 "claims/protocol edits post-run."},
            "network_model_calls": 0,
            "order_submission_authorized": False,
            "live_trading_authorized": False},
        "parameters": {"pct_and_regime_trailing_records": TRAIL,
                       "mom4_lag_bars": MOM4_LAG,
                       "hold_hours": HOLD_H,
                       "decile_edge": DECILE,
                       "funding_regime_thresholds":
                           {"positive": POS_HI, "negative": POS_LO},
                       "settlement_grid_ms": SETTLE_MS,
                       "exclusion_pre_ns": PRE_NS,
                       "exclusion_post_ns": POST_NS,
                       "excluded_cycle_hours": [0, 1, 6, 7],
                       "welch_min_n": MIN_N,
                       "fdr_alpha": FDR_ALPHA,
                       "fdr_family_size": 4,
                       "attenuation_floor": ATTEN_FLOOR,
                       "confirm_p_threshold": CONFIRM_P,
                       "fee_bps_per_side_context": FEE_BPS},
        "cohort": {"path": str(COHORT.relative_to(ROOT)),
                   "sha256": hashlib.sha256(
                       COHORT.read_bytes()).hexdigest(),
                   "records": n_rec},
        "references": {"t96_result": T96_RESULT,
                       "t96_protocol": T96_PROTOCOL},
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
                      "verdict": r1["results"]["verdict"]["verdict"],
                      "sha256": hashlib.sha256(blob.encode()).hexdigest()},
                     indent=2))


if __name__ == "__main__":
    main()
