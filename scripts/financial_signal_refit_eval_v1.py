#!/usr/bin/env python3
"""T136: does periodic refitting beat the frozen model for the ridge book?

Context: ledger v4 refits ridge_min3 on the static mega cohort every run
(training ends 2025-12-26 forever — the same fitted model is re-derived
each time, so "refit every run" is currently equivalent to "frozen").
Question: would monthly/quarterly expanding-window refits have produced
better OOS ranking than a frozen 2021 model — and does recency-weighting
help or hurt? T130's ``recent_regime`` arm (train only on 2023+ rows)
BACKFIRED on the live window (pooled rho -0.107 vs +0.025 full-history
top30), which predicts rolling windows should underperform expanding
windows here. This script measures that directly on the mega cohort.

Model: ridge_min3 {dfh20, btc_ret20, dfh20*btc_ret20} — the T127 winner —
fit on label.forward_return_5d_bps with lambda=100 frozen (v4's lambda;
no inner CV so all policies differ ONLY in training window/cadence).
Features are standardized on each fit's own train window (mean/std);
coefficient drift is reported in raw (unstandardized) space so it is
comparable across refits.

Walkforward: month-by-month OOS over 2022-01..2025-12 (48 test months).
Train history starts 2021-01-01 (mega cohort start). A 5-day embargo is
dropped before every refit boundary (label is a 5d forward return).

Refit policies:
  frozen_2021          fit once on 2021 rows (embargoed vs 2022-01-01),
                       never refit — worst-case anchor
  expanding_monthly    refit every month on ALL history to date
  expanding_quarterly  refit only at quarter starts (Jan/Apr/Jul/Oct)
                       on all history to date; held through the quarter
  rolling_2y           refit every month on the trailing 2 years only
  rolling_1y           refit every month on the trailing 1 year only
                       (max recency; T130 predicts this hurts)

Metrics per policy:
  monthly_oos          per test month: n rows/days, pooled spearman
                       (pred vs ret5d), mean per-day cross-sectional
                       spearman, LS book net bps/day for the month
  net_book             top2/bot2 within-day book: long the 2 highest
                       preds / short the 2 lowest, EW within each leg,
                       0.5 notional per side, daily rebalance, PnL =
                       label.forward_return_bps (1d), cost = 5bps one-way
                       on traded notional per leg; pooled + per-year
  coef_drift           raw-space coefficient path per refit: per-feature
                       mean/std/min/max, sign flips, mean L2 step between
                       successive fits, first-vs-last drift

Answers computed deterministically:
  expanding vs rolling — is more history always better? (T130 says yes)
  cadence — does monthly beat quarterly expanding?
  verdict — what the production ledger should do (refit cadence on the
  mega cohort; "frozen until regime break" is the alternative)

Artifacts follow convention: frozen protocol
(``research/financial_signal_refit_eval_protocol_v1.json``) plus owner
self-authorization pinning it by sha256
(``results/financial_signal_refit_eval_authorization_v1.json``) written
on every run BEFORE measurement; both are also embedded in the output.
Measurement only: no trading, no promotion claims, no network.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from financial_signal_ridge_v2_v1 import (  # noqa: E402
    _r, leg_turnover, load_btc_ret20, ridge_fit, ridge_pred, spearman)

COHORT = ROOT / "data/perp_pit_mega_v1/records.jsonl"
BTC_CSV = ROOT / "data/rc_futures_v1/BTC/BTCUSDT_1d.csv"
OUT = ROOT / "results/financial_signal_refit_eval_v1.json"
PROTOCOL = ROOT / "research/financial_signal_refit_eval_protocol_v1.json"
AUTH = ROOT / "results/financial_signal_refit_eval_authorization_v1.json"

LAMBDA = 100.0            # v4's frozen lambda; identical for all policies
EMBARGO_DAYS = 5          # repo convention (5d label overlap)
TOP_N = 2                 # long leg size (matches v4 top2/bot2 book)
BOT_N = 2                 # short leg size
COST_BPS_PER_LEG = 5.0    # one-way cost per unit traded notional, per leg
MIN_NAMES_DAY = 20        # skip book days with fewer assets (repo floor)
MIN_TRAIN_ROWS = 500
TEST_MONTHS = tuple((y, m) for y in range(2022, 2026) for m in range(1, 13))
QUARTER_STARTS = (1, 4, 7, 10)
POLICIES = ("frozen_2021", "expanding_monthly", "expanding_quarterly",
            "rolling_2y", "rolling_1y")
FEATURES = ("dfh20", "btc_ret20", "dfh20_x_btc")
TRAIN_START = dt.date(2021, 1, 1)   # cohort start; "train starts 2021"


def month_after(yy, mm):
    """First day of the month after (yy, mm)."""
    return dt.date(yy + 1, 1, 1) if mm == 12 else dt.date(yy, mm + 1, 1)


def train_lo(policy, m_start):
    """Inclusive lower date bound of a policy's train window at refit."""
    if policy == "rolling_2y":
        return dt.date(m_start.year - 2, m_start.month, 1)
    if policy == "rolling_1y":
        return dt.date(m_start.year - 1, m_start.month, 1)
    return TRAIN_START  # frozen_2021 + both expanding arms: all history


def refit_due(policy, i, mm):
    """Whether `policy` refits at test-month index i (month mm)."""
    if policy == "frozen_2021":
        return i == 0
    if policy == "expanding_quarterly":
        return mm in QUARTER_STARTS
    return True  # expanding_monthly, rolling_2y, rolling_1y


def load_rows_min3(btc_ret20):
    """Minimal loader: only the fields ridge_min3 + the book need.

    Returns rows as tuples (date, asset, dfh20, btc_ret20, dfh20*btc,
    y5_bps, y1_bps) sorted by (date, asset), plus drop stats.
    """
    rows = []
    stats = {"n_rows_total": 0, "n_dropped_missing_core": 0,
             "n_dropped_no_btc": 0}
    with COHORT.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            stats["n_rows_total"] += 1
            ft, lb = r["features"], r["label"]
            dfh = ft["dfh20"]["value"]
            y5 = lb["forward_return_5d_bps"]
            y1 = lb["forward_return_bps"]
            if dfh is None or y5 is None or y1 is None:
                stats["n_dropped_missing_core"] += 1
                continue
            d = dt.date.fromisoformat(r["id"].rsplit(":", 1)[-1])
            b = btc_ret20.get(d)
            if b is None:
                stats["n_dropped_no_btc"] += 1
                continue
            rows.append((d, r["asset_id"], dfh, b, dfh * b, y5, y1))
    rows.sort(key=lambda t: (t[0], t[1]))
    stats["n_modelable"] = len(rows)
    stats["first_date"] = rows[0][0].isoformat() if rows else None
    stats["last_date"] = rows[-1][0].isoformat() if rows else None
    stats["n_assets"] = len({t[1] for t in rows})
    return rows, stats


def daily_rhos(pred, y, ordinals_slice):
    """Per-day cross-sectional spearmans inside a contiguous sorted
    slice; returns (mean, median, n_days_used)."""
    rhos = []
    i, n = 0, len(ordinals_slice)
    while i < n:
        j = i
        while j + 1 < n and ordinals_slice[j + 1] == ordinals_slice[i]:
            j += 1
        r = spearman(pred[i:j + 1], y[i:j + 1])
        if r is not None:
            rhos.append(r)
        i = j + 1
    if not rhos:
        return None, None, 0
    v = np.asarray(rhos)
    return float(v.mean()), float(np.median(v)), len(rhos)


def net_book_top2(oos_rows):
    """Daily EW top2/bot2 book on OOS predictions (v4 book convention).

    oos_rows: list of (date, asset, pred, y1_bps). Long the TOP_N highest
    preds, short the BOT_N lowest, EW within each leg, 0.5 notional per
    side, daily rebalance; cost = COST_BPS_PER_LEG x one-way turnover per
    leg (turnover = 0.5*sum|w_new - w_old| per leg); first day of a
    contiguous run counts as full entry. Returns pooled + per-year stats
    and the daily record list (for per-month aggregation).
    """
    by_date = defaultdict(list)
    for d, a, p, y1 in oos_rows:
        by_date[d].append((a, p, y1))
    recs = []
    n_skipped = 0
    prev_long, prev_short = {}, {}
    for d in sorted(by_date):
        rows = by_date[d]
        if len(rows) < MIN_NAMES_DAY:
            n_skipped += 1
            prev_long, prev_short = {}, {}
            continue
        rows.sort(key=lambda t: t[1], reverse=True)
        longs, shorts = rows[:TOP_N], rows[-BOT_N:]
        w_long = {r[0]: 1.0 / len(longs) for r in longs}
        w_short = {r[0]: 1.0 / len(shorts) for r in shorts}
        to_l = leg_turnover(prev_long, w_long) if prev_long else 1.0
        to_s = leg_turnover(prev_short, w_short) if prev_short else 1.0
        long_ret = float(np.mean([r[2] for r in longs]))
        short_ret = float(np.mean([r[2] for r in shorts]))
        gross = 0.5 * long_ret - 0.5 * short_ret
        cost = COST_BPS_PER_LEG * (0.5 * to_l + 0.5 * to_s)
        recs.append({"date": d, "month": "%04d-%02d" % (d.year, d.month),
                     "year": d.year, "gross": gross,
                     "net": gross - cost, "cost": cost,
                     "to_long": to_l, "to_short": to_s})
        prev_long, prev_short = w_long, w_short

    def summarize(sub):
        if not sub:
            return None
        g = np.array([r["gross"] for r in sub])
        n = np.array([r["net"] for r in sub])
        c = np.array([r["cost"] for r in sub])
        t = np.array([r["to_long"] + r["to_short"] for r in sub])
        sd = n.std()
        return {"n_days": len(sub),
                "mean_gross_bps_day": _r(g.mean(), 3),
                "mean_net_bps_day": _r(n.mean(), 3),
                "total_net_bps": _r(n.sum(), 1),
                "sharpe_net": _r(n.mean() / sd * math.sqrt(252), 3)
                if sd > 0 else None,
                "hit_rate_net": _r(np.mean(n > 0)),
                "avg_daily_cost_bps": _r(c.mean(), 3),
                "avg_daily_turnover_two_leg": _r(t.mean(), 4)}

    by_year = defaultdict(list)
    by_month = defaultdict(list)
    for r in recs:
        by_year[r["year"]].append(r)
        by_month[r["month"]].append(r)
    return {"n_days_total": len(recs), "n_days_skipped": n_skipped,
            "book": ("top2/bot2 EW within-day, 0.5 notional per side, "
                     "daily rebalance, 5bps one-way per leg on traded "
                     "notional; first day of a run = full entry; days "
                     "with <20 assets skipped"),
            "pooled": summarize(recs),
            "per_year": {str(y): summarize(rs)
                         for y, rs in sorted(by_year.items())},
            "per_month_net_bps_day": {
                m: _r(np.mean([r["net"] for r in rs]), 3)
                for m, rs in sorted(by_month.items())}}


def coef_drift(fit_records):
    """Coefficient path summary; w_raw = raw-space coefs (intercept,
    dfh20, btc_ret20, dfh20_x_btc), comparable across refits because the
    per-fit scaler is folded back out."""
    if not fit_records:
        return {}
    names = ("intercept",) + FEATURES
    W = np.array([f["w_raw"] for f in fit_records])
    out = {"n_fits": len(fit_records),
           "refit_months": [f["month"] for f in fit_records],
           "features": list(names), "space": "raw (unstandardized)"}
    for j, n in enumerate(names):
        v = W[:, j]
        signs = np.sign(v)
        out[n] = {"first": _r(v[0], 6), "last": _r(v[-1], 6),
                  "mean": _r(v.mean(), 6), "std": _r(v.std(), 6),
                  "min": _r(v.min(), 6), "max": _r(v.max(), 6),
                  "n_sign_flips": int(np.sum(signs[1:] != signs[:-1]))
                  if len(v) > 1 else 0}
    if len(W) > 1:
        steps = np.linalg.norm(np.diff(W, axis=0), axis=1)
        denom = np.linalg.norm(W[:-1], axis=1)
        denom[denom == 0] = np.nan
        out["successive_refit_drift"] = {
            "mean_l2_step": _r(np.nanmean(steps), 4),
            "max_l2_step": _r(np.nanmax(steps), 4),
            "mean_relative_step": _r(np.nanmean(steps / denom), 4),
            "first_last_l2": _r(np.linalg.norm(W[-1] - W[0]), 4),
            "first_last_relative": _r(
                np.linalg.norm(W[-1] - W[0]) / np.linalg.norm(W[0]), 4)
            if np.linalg.norm(W[0]) > 0 else None}
    return out


def run_policy(policy, ords, X, y5, y1, dates, assets):
    """One refit policy across the 48 test months."""
    w = mu = sd = None
    fit_records, monthly, oos_rows = [], [], []
    all_pred, all_y = [], []
    for i, (yy, mm) in enumerate(TEST_MONTHS):
        m_start = dt.date(yy, mm, 1)
        m_end = month_after(yy, mm)
        if refit_due(policy, i, mm):
            hi_ord = (m_start - dt.timedelta(days=EMBARGO_DAYS)
                      ).toordinal()
            lo_ord = train_lo(policy, m_start).toordinal()
            lo_i = int(np.searchsorted(ords, lo_ord))
            hi_i = int(np.searchsorted(ords, hi_ord))
            Xtr, ytr = X[lo_i:hi_i], y5[lo_i:hi_i]
            if len(Xtr) < MIN_TRAIN_ROWS:
                monthly.append({"month": "%04d-%02d" % (yy, mm),
                                "skipped": True,
                                "reason": "n_train<%d" % MIN_TRAIN_ROWS,
                                "n_train": int(len(Xtr))})
                continue
            mu = Xtr.mean(0)
            sd = Xtr.std(0)
            sd[sd == 0] = 1.0
            w = ridge_fit((Xtr - mu) / sd, ytr, LAMBDA)
            w_raw = np.concatenate(
                [[w[0] - float((mu / sd) @ w[1:])], w[1:] / sd])
            fit_records.append({
                "month": "%04d-%02d" % (yy, mm),
                "train_lo": dt.date.fromordinal(lo_ord).isoformat(),
                "train_hi": dt.date.fromordinal(hi_ord).isoformat(),
                "n_train": int(len(Xtr)),
                "w_std": [_r(c, 6) for c in w],
                "w_raw": [_r(c, 6) for c in w_raw]})
        te_lo = int(np.searchsorted(ords, m_start.toordinal()))
        te_hi = int(np.searchsorted(ords, m_end.toordinal()))
        if te_hi <= te_lo or w is None:
            monthly.append({"month": "%04d-%02d" % (yy, mm),
                            "skipped": True, "reason": "no_test_rows"})
            continue
        pred = ridge_pred((X[te_lo:te_hi] - mu) / sd, w)
        yte = y5[te_lo:te_hi]
        rho_pool = spearman(pred, yte)
        rho_d_mean, rho_d_med, n_d = daily_rhos(
            pred, yte, ords[te_lo:te_hi])
        monthly.append({"month": "%04d-%02d" % (yy, mm),
                        "n_rows": int(te_hi - te_lo), "n_days": n_d,
                        "refit_this_month": bool(
                            fit_records and
                            fit_records[-1]["month"]
                            == "%04d-%02d" % (yy, mm)),
                        "rho_pooled": _r(rho_pool),
                        "rho_daily_mean": _r(rho_d_mean),
                        "rho_daily_median": _r(rho_d_med)})
        all_pred.append(pred)
        all_y.append(yte)
        for k in range(te_lo, te_hi):
            oos_rows.append((dates[k], assets[k], float(pred[k - te_lo]),
                             y1[k]))
    book = net_book_top2(oos_rows)
    for m in monthly:
        m["ls_net_bps_day"] = book["per_month_net_bps_day"].get(
            m["month"])
    ok = [m for m in monthly if not m.get("skipped")]
    rp = np.array([m["rho_pooled"] for m in ok
                   if m["rho_pooled"] is not None])
    rd = np.array([m["rho_daily_mean"] for m in ok
                   if m["rho_daily_mean"] is not None])
    by_year_rho = defaultdict(list)
    for m in ok:
        if m["rho_pooled"] is not None:
            by_year_rho[int(m["month"][:4])].append(m["rho_pooled"])
    return {"policy": policy,
            "fit_records": fit_records,
            "coef_drift": coef_drift(fit_records),
            "monthly_oos": monthly,
            "pooled_oos": {
                "n_months": len(ok),
                "n_rows": int(sum(m["n_rows"] for m in ok)),
                "spearman_all_rows": _r(spearman(
                    np.concatenate(all_pred), np.concatenate(all_y)))
                if all_pred else None,
                "mean_monthly_rho_pooled": _r(rp.mean()) if len(rp)
                else None,
                "median_monthly_rho_pooled": _r(np.median(rp))
                if len(rp) else None,
                "frac_months_rho_positive": _r(np.mean(rp > 0))
                if len(rp) else None,
                "mean_daily_rho": _r(rd.mean()) if len(rd) else None,
                "per_year_mean_monthly_rho": {
                    str(y): _r(np.mean(v))
                    for y, v in sorted(by_year_rho.items())}},
            "net_book": {k: v for k, v in book.items()
                         if k != "per_month_net_bps_day"},
            "monthly_book_net_bps_day": book["per_month_net_bps_day"]}


def write_protocol_and_auth():
    protocol = {
        "schema_version": "nanojev-financial-signal-refit-eval-protocol-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": ("T136: month-by-month walkforward 2022-01..2025-12 "
                    "comparing five ridge refit policies "
                    "(frozen_2021, expanding_monthly, "
                    "expanding_quarterly, rolling_2y, rolling_1y) on the "
                    "static mega cohort; model ridge_min3 "
                    "{dfh20, btc_ret20, dfh20*btc_ret20} lambda=100; "
                    "answers whether periodic refits beat a frozen model "
                    "and whether recency windows hurt (T130 prior: yes "
                    "they hurt). Measurement only."),
        "cohort": {
            "path": "data/perp_pit_mega_v1/records.jsonl",
            "fit_target": "label.forward_return_5d_bps",
            "book_pnl": "label.forward_return_bps (1d close-to-close)",
            "btc_series": "data/rc_futures_v1/BTC/BTCUSDT_1d.csv",
            "train_history_start": "2021-01-01 (cohort start)",
        },
        "definitions": {
            "model": "ridge on 3 features standardized by each fit's own "
                     "train mean/std; intercept unpenalized; lambda=100 "
                     "fixed for all policies (no inner CV — policies "
                     "differ only in window/cadence)",
            "embargo": "train rows dated in the 5 days before each refit "
                       "boundary are dropped (5d label overlap)",
            "policies": {
                "frozen_2021": "fit once on 2021 rows (embargoed vs "
                               "2022-01-01); never refit",
                "expanding_monthly": "refit each test month on all rows "
                                     "< month_start - 5d",
                "expanding_quarterly": "refit only at Jan/Apr/Jul/Oct "
                                       "month starts on all history; "
                                       "model held through the quarter",
                "rolling_2y": "refit each month on rows in "
                              "[month_start-2y, month_start-5d)",
                "rolling_1y": "refit each month on rows in "
                              "[month_start-1y, month_start-5d)"},
            "monthly_oos": "per test month: pooled spearman(pred, ret5d) "
                           "over all rows + mean/median of per-day "
                           "cross-sectional spearmans (days with >=10 "
                           "assets)",
            "book": "per test date: long 2 highest preds / short 2 "
                    "lowest, EW within leg, 0.5 notional per side, daily "
                    "rebalance; cost = 5bps one-way on traded notional "
                    "per leg (turnover = 0.5*sum|w_new-w_old|); days "
                    "with <20 assets skipped; book runs continuously "
                    "across month boundaries",
            "coef_drift": "raw-space coefficient path (per-fit scaler "
                          "folded back out); per-feature stats + L2 "
                          "distance between successive fits",
        },
        "statistics": {
            "caveats": [
                "monthly pooled rho mixes cross-sectional and cross-day "
                "level effects; mean daily rho is the ranked-book "
                "metric; both reported",
                "top2/bot2 book is concentrated (2 names/leg) — high "
                "idiosyncratic variance, turnover-heavy; net results "
                "are sensitive to the flat 5bps assumption",
                "no borrow/funding-carry/slippage beyond flat 5bps; "
                "short leg assumes perp shortability at index",
                "all policies see identical test rows; differences are "
                "purely the training window/cadence — paired comparison",
                "coef drift in raw space is comparable across fits but "
                "raw dfh20*btc scale shifts with btc_ret20 dispersion",
            ],
        },
        "forbidden": ["trading", "profitability claims",
                      "protocol edits post-run", "network",
                      "model selection on test data"],
    }
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(protocol, indent=2, sort_keys=True)
                        + "\n")
    sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    auth = {
        "schema_version":
            "nanojev-financial-signal-refit-eval-authorization-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision": "approved_for_measurement",
        "protocol_path":
            "research/financial_signal_refit_eval_protocol_v1.json",
        "protocol_sha256": sha,
        "measurement_authorized": True,
        "fit_authorized": True,
        "live_trading_authorized": False,
        "order_submission_authorized": False,
        "network_model_calls": 0,
        "independent_reviewer": {
            "id": "project-owner",
            "independence":
                "owner_self_authorization_not_independent_review",
            "note": "Owner directed T136: evaluate ridge refit cadence "
                    "vs frozen model on the mega cohort (delegated "
                    "task).",
        },
        "scope": {
            "permitted": "PIT-safe ridge refits + OOS measurement + "
                         "simulated daily top2/bot2 book net accounting "
                         "on data/perp_pit_mega_v1/records.jsonl with "
                         "the BTC daily series as regime feature, per "
                         "the pinned protocol: 5 policies x 48 test "
                         "months, lambda=100 frozen.",
            "not_permitted": "No trading/profitability claims/protocol "
                             "edits; no network; no other files "
                             "modified.",
        },
    }
    AUTH.parent.mkdir(parents=True, exist_ok=True)
    AUTH.write_text(json.dumps(auth, indent=2, sort_keys=True) + "\n")
    return protocol, sha, auth


def verdict_lines(policies):
    """Deterministic answers from computed metrics."""
    by = {p["policy"]: p for p in policies}
    lines = []
    net = {}
    rho = {}
    for name, p in by.items():
        ls = (p["net_book"] or {}).get("pooled") or {}
        net[name] = ls.get("mean_net_bps_day")
        rho[name] = (p["pooled_oos"] or {}).get("mean_monthly_rho_pooled")
        per_year = (p["net_book"] or {}).get("per_year") or {}
        pos_years = sum(1 for v in per_year.values()
                        if v and v.get("mean_net_bps_day") is not None
                        and v["mean_net_bps_day"] > 0)
        lines.append(
            "%s: monthly-rho %s (daily %s, pos %s of months), LS net "
            "%s bps/day (Sharpe %s, net-positive %d/%d years), %d fits."
            % (name, rho[name],
               (p["pooled_oos"] or {}).get("mean_daily_rho"),
               (p["pooled_oos"] or {}).get("frac_months_rho_positive"),
               net[name], ls.get("sharpe_net"), pos_years,
               len(per_year),
               (p["coef_drift"] or {}).get("n_fits", 0)))
    em, eq = net.get("expanding_monthly"), net.get("expanding_quarterly")
    r2, r1 = net.get("rolling_2y"), net.get("rolling_1y")
    fz = net.get("frozen_2021")
    exp_best = max(v for k, v in net.items()
                   if k.startswith("expanding") and v is not None)
    roll_best = max(v for k, v in net.items()
                    if k.startswith("rolling") and v is not None)
    best = max(net, key=lambda k: net[k] if net[k] is not None else -1e9)
    # --- history length
    if exp_best is not None and roll_best is not None:
        monotonic = (em is not None and r2 is not None and r1 is not None
                     and em >= r2 >= r1)
        lines.append(
            "HISTORY: expanding_monthly net %s vs rolling_2y %s vs "
            "rolling_1y %s bps/day — more training history %s "
            "(%s; consistent with T130's recent_regime backfire)."
            % (em, r2, r1,
               "is better" if exp_best > roll_best else "is NOT better",
               "strictly monotone in window length" if monotonic
               else "not strictly monotone"))
    # --- cadence
    cadence_diff = (em - eq) if (em is not None and eq is not None) \
        else None
    lines.append(
        "CADENCE: expanding_monthly %s vs expanding_quarterly %s bps/day "
        "(delta %s) — cadence %s." % (
            em, eq, _r(cadence_diff, 3) if cadence_diff is not None
            else None,
            "matters" if cadence_diff is not None
            and abs(cadence_diff) > 0.5
            else "is immaterial (<0.5 bps/day)"))
    # --- frozen anchor
    if fz is not None:
        lines.append(
            "FROZEN_ANCHOR: frozen_2021 net %s bps/day vs best refit "
            "policy %s (%s bps/day) — refitting %s." % (
                fz, best, net[best],
                "helps" if net[best] > fz else "does NOT help"))
    # --- production verdict
    if best.startswith("expanding"):
        rec = ("REFIT ON ALL HISTORY — production should keep fitting on "
               "the full mega history each run (the current v4 behavior "
               "is already expanding-refit-equivalent since the cohort "
               "is static); do NOT switch to a rolling/recency window.")
        if cadence_diff is not None and abs(cadence_diff) <= 0.5:
            rec += (" Monthly vs quarterly cadence is within noise — "
                    "monthly is fine and simpler to reason about.")
        else:
            rec += " Prefer %s on measured net." % best
    elif best == "frozen_2021":
        rec = ("KEEP FROZEN — no refit policy beat the 2021-frozen "
               "anchor; freeze coefficients until a regime break is "
               "detected.")
    else:
        rec = ("RECENCY WINS (unexpected vs T130) — %s had the best net; "
               "investigate regime shift before adopting a rolling "
               "window." % best)
    lines.append("VERDICT: " + rec)
    return lines, {"best_policy_by_net": best,
                   "net_bps_day_by_policy": net,
                   "mean_monthly_rho_by_policy": rho,
                   "expanding_beats_rolling": bool(
                       exp_best is not None and roll_best is not None
                       and exp_best > roll_best),
                   "cadence_diff_bps_day": _r(cadence_diff, 3)
                   if cadence_diff is not None else None,
                   "refit_beats_frozen": bool(fz is not None
                                              and net[best] > fz),
                   "production_recommendation": rec}


def run():
    protocol, protocol_sha, auth = write_protocol_and_auth()
    t0 = time.time()
    btc_ret20 = load_btc_ret20()
    rows, dstats = load_rows_min3(btc_ret20)
    ords = np.array([t[0].toordinal() for t in rows])
    dates = [t[0] for t in rows]
    assets = [t[1] for t in rows]
    X = np.array([[t[2], t[3], t[4]] for t in rows])
    y5 = np.array([t[5] for t in rows])
    y1 = np.array([t[6] for t in rows])
    policies = [run_policy(p, ords, X, y5, y1, dates, assets)
                for p in POLICIES]
    lines, answer = verdict_lines(policies)
    return {
        "schema_version": "nanojev-financial-signal-refit-eval-v1",
        "task": "T136",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                      time.gmtime()),
        "runtime_s": _r(time.time() - t0, 1),
        "protocol_path":
            "research/financial_signal_refit_eval_protocol_v1.json",
        "protocol_sha256": protocol_sha,
        "authorization_path":
            "results/financial_signal_refit_eval_authorization_v1.json",
        "protocol": protocol,
        "authorization": auth,
        "data": {"cohort": str(COHORT.relative_to(ROOT)),
                 "btc_series": str(BTC_CSV.relative_to(ROOT)),
                 **dstats},
        "design": {"model": "ridge_min3",
                   "features": list(FEATURES),
                   "lambda": LAMBDA,
                   "fit_target": "forward_return_5d_bps",
                   "book_pnl": "forward_return_bps (1d)",
                   "book": "top2/bot2 EW within-day, 0.5 notional/side, "
                           "daily rebalance, 5bps one-way per leg",
                   "test_months": ["%04d-%02d" % m for m in TEST_MONTHS],
                   "train_start": TRAIN_START.isoformat(),
                   "embargo_days": EMBARGO_DAYS,
                   "policies": list(POLICIES)},
        "policies": policies,
        "answer": answer,
        "verdict": " | ".join(lines),
        "verdict_lines": lines,
        "scope": "measurement only; not a strategy or tradability claim",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=pathlib.Path, default=OUT)
    args = ap.parse_args()
    report = run()
    blob = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(blob)
    print(json.dumps({"output": str(args.output),
                      "sha256": hashlib.sha256(blob.encode()).hexdigest(),
                      "verdict": report["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
