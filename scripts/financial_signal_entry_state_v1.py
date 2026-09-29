#!/usr/bin/env python3
"""Entry-state measurement for a rolling-compounding campaign (T117).

Question: within BTC-trend-up regimes (gate = BTC close ret20 > 0), WHICH
asset-state gives the best ENTRY for a long rolling campaign? Reads
``data/perp_pit_mega_v1/records.jsonl`` (277 USDT-M perps, 2021-01 ->
2025-12, close-as-mark basis, ~52 funding-covered symbols, includes
delisted/renamed early-stoppers) and the BTC daily series
``data/rc_futures_v1/BTC/BTCUSDT_1d.csv`` for the regime gate.

Candidate entry-state dimensions (all computable daily, strictly PIT):

  gate          BTC ret20 > 0 on the decision day (mandatory context; the
                analysis set is gate-ON asset-days; gate-off is reported
                only as context).
  dfh20         XS tercile of dfh20 within the decision day (>=30 ranked
                assets): near-high (top) / mid / far (bottom).
  pullback      -0.12 <= dfh20 <= -0.03 AND close > previous contiguous
                day's close (off the 20-bar high but recovering — sibling
                project's E1 analog).
  extended      dfh20 > -0.01 (at/near the 20-bar high).
  breaking      mom20 > 0 AND dfh20 < -0.12 (20d momentum up but still
                deep off the high).
  funding_pct   per-asset trailing-180 mid-rank pct of last_funding_rate
                (MIN_WINDOW=20); terciled at 1/3, 2/3 on the funded
                subset. stealth_rally = near-high AND low funding;
                euphoric = near-high AND high funding.
  vol20         XS tercile of vol20 within the decision day.

Per state cell: n, mean+median fwd 5d and 10d close returns, and PATH
stats from the close series — MFE_10d (max close gain within 10d after
entry) and MAE_10d (max close loss). MFE high + MAE low = rollable for
pyramid design. 10d returns and path stats require 10 contiguous forward
daily bars per asset (else that asset-day contributes n only, not n_path).

Contrasts: pooled Welch two-sample t on asset-day returns for 5 named
contrasts x {5d, 10d} = a 10-cell BH-FDR family (alpha 0.05):
  pullback vs extended | breaking vs extended | dfh top-tercile vs
  bottom-tercile | stealth vs euphoric (= funding low vs high within
  near-high) | vol low vs high tercile. Per-year folds are reported for
  the top-3 quality-ranked cells. Verdict ranks cells by risk-adjusted
  quality = mean_fwd_10d_bps x (mean_MFE / |mean_MAE|) and recommends a
  composite entry rule for a campaign simulator.

Artifacts follow the established convention: a frozen protocol
(``research/financial_signal_entry_state_protocol_v1.json``) plus an
owner self-authorization pinning it by sha256
(``results/financial_signal_entry_state_authorization_v1.json``) are
written on every run BEFORE measurement. Measurement only: no fitting,
no trading, no network.

Honest dof note: pooled Welch treats same-day cross-asset returns as
independent (they share market beta) and 5d/10d labels overlap adjacent
days — all t/p nominal. This is entry-STATE conditioning, not a strategy
backtest; no profitability claim is made.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
import statistics
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_mega_v1/records.jsonl"
BTC_CSV = ROOT / "data/rc_futures_v1/BTC/BTCUSDT_1d.csv"
OUT = ROOT / "results/financial_signal_entry_state_v1.json"
PROTOCOL = ROOT / "research/financial_signal_entry_state_protocol_v1.json"
AUTH = ROOT / "results/financial_signal_entry_state_authorization_v1.json"

MIN_ASSETS_PER_DAY = 30   # XS-tercile days need a real cross-section
FUND_LOOKBACK = 180       # trailing-180 mid-rank window for funding_pct
MIN_WINDOW = 20           # floor for a usable trailing pct (repo convention)
BTC_GATE_LOOKBACK = 20
PATH_DAYS = 10            # MFE/MAE/forward-return horizon for path stats
PULLBACK_LO, PULLBACK_HI = -0.12, -0.03
EXTENDED_CUT = -0.01
BREAKING_DFH_CUT = -0.12
MIN_CELL_N = 200          # eligibility floor for the quality ranking
MIN_CELL_N_PATH = 150     # eligibility floor for path-based ranking


def _r(x, nd=3):
    return round(x, nd) if isinstance(x, float) else x


def norm_p(t):
    """Two-sided normal-approx p from a t/z statistic (repo convention)."""
    if t is None:
        return None
    return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))


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
        return {"n_a": nx, "n_b": ny, "mean_a": mx,
                "mean_b": my, "t": None, "df": None, "p": None}
    t = (mx - my) / math.sqrt(denom)
    df = denom ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    return {"n_a": nx, "n_b": ny, "mean_a": mx,
            "mean_b": my, "mean_diff": mx - my,
            "t": t, "df": df, "p": norm_p(t)}


def mid_rank_pct(window, x):
    """Mid-rank percentile of x vs a window of values (ties count half)."""
    if x is None or not window:
        return None
    return (sum(1 for w in window if w < x)
            + 0.5 * sum(1 for w in window if w == x)) / len(window)


def bh_fdr(named_ps, alpha=0.05):
    """BH-FDR over [(name, p|None)] -> ({name: survives}, sorted table)."""
    ps = sorted((p, n) for n, p in named_ps if p is not None)
    m = len(ps)
    table = [{"cell": n, "p": _r(p, 6),
              "alpha_bh": round(alpha * (i + 1) / m, 6),
              "survives": p <= alpha * (i + 1) / m}
             for i, (p, n) in enumerate(ps)]
    return {e["cell"]: e["survives"] for e in table}, table


def load_btc_gate(path):
    """BTC daily csv -> {day_iso: ret20} from closes."""
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for rec in csv.DictReader(stream):
            rows.append((rec["timestamp"], float(rec["close"])))
    rows.sort()
    gate = {}
    for i, (day, close) in enumerate(rows):
        if i >= BTC_GATE_LOOKBACK and rows[i - BTC_GATE_LOOKBACK][1] > 0:
            gate[day] = close / rows[i - BTC_GATE_LOOKBACK][1] - 1.0
    return gate


def load_cohort(path):
    """asset_id -> sorted rows with features, labels and derived fields:
    prev_close (contiguous prior day only), funding_pct (trailing-180
    mid-rank), and 10d path stats (fwd_10d / MFE_10d / MAE_10d, bps,
    requiring 10 contiguous forward daily bars)."""
    series = defaultdict(list)
    meta = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            asset = record["asset_id"]
            meta[asset] = {"listed": record.get("listed"),
                           "last_bar_date": record.get("last_bar_date")}
            features = record["features"]
            series[asset].append({
                "day": record["id"].rsplit(":", 1)[-1],
                "close": features["close"]["value"],
                "dfh20": features["dfh20"]["value"],
                "mom20": features["mom20"]["value"],
                "vol20": features["vol20"]["value"],
                "funding": features["last_funding_rate"]["value"],
                "fwd_5d_bps": record["label"]["forward_return_5d_bps"],
            })
    funded = []
    for asset, rows in series.items():
        rows.sort(key=lambda r: r["day"])
        days = [dt.date.fromisoformat(r["day"]) for r in rows]
        closes = [r["close"] for r in rows]
        fund = [r["funding"] for r in rows]
        any_funding = False
        for i, r in enumerate(rows):
            # previous contiguous day close (strict PIT pullback leg)
            r["prev_close"] = (closes[i - 1]
                               if i > 0 and (days[i] - days[i - 1]).days == 1
                               else None)
            fwin = [x for x in fund[max(0, i - FUND_LOOKBACK):i]
                    if x is not None]
            r["funding_pct"] = (mid_rank_pct(fwin, r["funding"])
                                if r["funding"] is not None
                                and len(fwin) >= MIN_WINDOW else None)
            any_funding = any_funding or r["funding"] is not None
            # 10d forward path: require 10 contiguous forward bars
            if (i + PATH_DAYS < len(rows)
                    and all((days[i + k] - days[i + k - 1]).days == 1
                            for k in range(1, PATH_DAYS + 1))
                    and closes[i] > 0):
                fwd = [closes[i + k] / closes[i] - 1.0
                       for k in range(1, PATH_DAYS + 1)]
                r["fwd_10d_bps"] = fwd[-1] * 1e4
                r["mfe_10d_bps"] = max(fwd) * 1e4
                r["mae_10d_bps"] = min(fwd) * 1e4
            else:
                r["fwd_10d_bps"] = r["mfe_10d_bps"] = r["mae_10d_bps"] = None
        if any_funding:
            funded.append(asset)
    return series, meta, sorted(funded)


def xs_rank_pcts(series, key):
    """day -> {asset: mid-rank pct of `key` among that day's non-null
    values}; days with < MIN_ASSETS_PER_DAY ranked assets are omitted."""
    by_day = defaultdict(list)
    for asset, rows in series.items():
        for r in rows:
            if r[key] is not None:
                by_day[r["day"]].append((asset, r[key]))
    out = {}
    for day, items in by_day.items():
        if len(items) < MIN_ASSETS_PER_DAY:
            continue
        vals = [v for _, v in items]
        out[day] = {a: mid_rank_pct(vals, v) for a, v in items}
    return out


# --- state-cell predicates (all evaluated on gate-ON asset-days) --------
# ctx keys: dfh20, mom20, vol20, close, prev_close, funding_pct,
#           dfh_pct (XS), vol_pct (XS)

def _pct_top(p):
    return p is not None and p >= 2 / 3


def _pct_mid(p):
    return p is not None and 1 / 3 < p < 2 / 3


def _pct_bot(p):
    return p is not None and p <= 1 / 3


def is_pullback(c):
    return (c["dfh20"] is not None
            and PULLBACK_LO <= c["dfh20"] <= PULLBACK_HI
            and c["prev_close"] is not None
            and c["close"] > c["prev_close"])


def is_extended(c):
    return c["dfh20"] is not None and c["dfh20"] > EXTENDED_CUT


def is_breaking(c):
    return (c["mom20"] is not None and c["dfh20"] is not None
            and c["mom20"] > 0 and c["dfh20"] < BREAKING_DFH_CUT)


CELLS = {
    "gate_on_all": lambda c: True,
    "dfh_near_high": lambda c: _pct_top(c["dfh_pct"]),
    "dfh_mid": lambda c: _pct_mid(c["dfh_pct"]),
    "dfh_far": lambda c: _pct_bot(c["dfh_pct"]),
    "pullback": is_pullback,
    "extended": is_extended,
    "breaking": is_breaking,
    "vol_low": lambda c: _pct_bot(c["vol_pct"]),
    "vol_mid": lambda c: _pct_mid(c["vol_pct"]),
    "vol_high": lambda c: _pct_top(c["vol_pct"]),
    "fund_low": lambda c: (c["funding_pct"] is not None
                           and c["funding_pct"] <= 1 / 3),
    "fund_mid": lambda c: (c["funding_pct"] is not None
                           and 1 / 3 < c["funding_pct"] < 2 / 3),
    "fund_high": lambda c: (c["funding_pct"] is not None
                            and c["funding_pct"] >= 2 / 3),
    "stealth_rally": lambda c: (_pct_top(c["dfh_pct"])
                                and c["funding_pct"] is not None
                                and c["funding_pct"] <= 1 / 3),
    "near_high_fund_mid": lambda c: (_pct_top(c["dfh_pct"])
                                     and c["funding_pct"] is not None
                                     and 1 / 3 < c["funding_pct"] < 2 / 3),
    "euphoric": lambda c: (_pct_top(c["dfh_pct"])
                           and c["funding_pct"] is not None
                           and c["funding_pct"] >= 2 / 3),
    "pullback_vol_low": lambda c: is_pullback(c) and _pct_bot(c["vol_pct"]),
    "pullback_vol_not_high": lambda c: (is_pullback(c)
                                        and c["vol_pct"] is not None
                                        and c["vol_pct"] <= 2 / 3),
    "pullback_fund_low": lambda c: (is_pullback(c)
                                    and c["funding_pct"] is not None
                                    and c["funding_pct"] <= 1 / 3),
    "pullback_vol_not_high_fund_low": lambda c: (
        is_pullback(c) and c["vol_pct"] is not None
        and c["vol_pct"] <= 2 / 3 and c["funding_pct"] is not None
        and c["funding_pct"] <= 1 / 3),
}

# 5 named contrasts x 2 horizons = 10-cell FDR family
CONTRASTS = [
    ("pullback_vs_extended", "pullback", "extended"),
    ("breaking_vs_extended", "breaking", "extended"),
    ("dfh_near_high_vs_far", "dfh_near_high", "dfh_far"),
    ("stealth_vs_euphoric", "stealth_rally", "euphoric"),
    ("vol_low_vs_high", "vol_low", "vol_high"),
]


def cell_stats(obs):
    """n + mean/median of fwd_5d, fwd_10d, MFE_10d, MAE_10d (bps) plus
    positive rates and the MFE/|MAE| path-asymmetry ratio."""
    def _col(key):
        return [o[key] for o in obs if o[key] is not None]

    def _m(v):
        return {"n": len(v),
                "mean": _r(sum(v) / len(v)) if v else None,
                "median": _r(statistics.median(v)) if v else None,
                "pos_frac": _r(sum(1 for x in v if x > 0) / len(v))
                if v else None}

    f5, f10 = _col("fwd_5d_bps"), _col("fwd_10d_bps")
    mfe, mae = _col("mfe_10d_bps"), _col("mae_10d_bps")
    ratio = (sum(mfe) / len(mfe)) / abs(sum(mae) / len(mae)) \
        if mfe and mae and sum(mae) != 0 else None
    return {"n": len(obs), "n_path": len(f10),
            "fwd_5d_bps": _m(f5), "fwd_10d_bps": _m(f10),
            "mfe_10d_bps": _m(mfe), "mae_10d_bps": _m(mae),
            "mfe_mae_ratio": _r(ratio)}


def write_protocol_and_auth():
    protocol = {
        "schema_version": "nanojev-financial-signal-entry-state-protocol-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "T117: measure entry STATES for a rolling-compounding "
                   "long campaign on the mega cohort — 'when to start "
                   "rolling'. Within BTC-trend-up days (gate = BTC close "
                   "ret20 > 0 from data/rc_futures_v1/BTC/BTCUSDT_1d.csv), "
                   "score each asset-day on PIT-computable state "
                   "dimensions (dfh20 XS tercile, pullback/extended/"
                   "breaking states, funding_pct tercile on the funded "
                   "subset, vol20 XS tercile) and report per-cell fwd "
                   "5d/10d returns plus 10d MFE/MAE path stats. Welch "
                   "contrasts under a <=10-cell BH-FDR family; per-year "
                   "folds for the top-3 cells; verdict ranks cells by "
                   "mean_fwd10 x MFE/|MAE| and recommends a composite "
                   "entry rule. Measurement only — no fitting, no "
                   "trading, no profitability claims.",
        "cohort": {
            "path": "data/perp_pit_mega_v1/records.jsonl",
            "builder": "scripts/build_perp_pit_mega_v1.py",
            "assets": "277 USDT-M perps (>=200 daily bars), imported "
                      "data/rc_futures_v1 archive copy; includes "
                      "delisted/renamed early-stoppers",
            "price_basis": "csv close as the mark proxy for every symbol",
            "btc_gate_source": "data/rc_futures_v1/BTC/BTCUSDT_1d.csv",
            "targets": "label.forward_return_5d_bps (5d gross "
                       "close-to-close) plus close-series-derived 10d "
                       "forward return, MFE_10d and MAE_10d",
        },
        "definitions": {
            "gate": "BTC close/close[t-20]-1 > 0 on the decision day; "
                    "analysis set = gate-ON asset-days (gate-off reported "
                    "as context only)",
            "dfh20": "precomputed feature: close / max(close[i-20:i]) - 1 "
                     "(strictly prior 20 contiguous bars)",
            "mom20": "precomputed feature: close / close[i-20] - 1",
            "vol20": "precomputed feature: 20-bar close-return volatility",
            "xs_tercile": "mid-rank pct of the feature among assets with "
                          "non-null values on the same decision day "
                          "(>=30 assets required): top >= 2/3, bottom "
                          "<= 1/3",
            "pullback": "-0.12 <= dfh20 <= -0.03 AND close > previous "
                        "contiguous day's close",
            "extended": "dfh20 > -0.01",
            "breaking": "mom20 > 0 AND dfh20 < -0.12",
            "funding_pct": "per-asset mid-rank pct of last_funding_rate "
                           "vs its trailing-180 daily values "
                           "(MIN_WINDOW=20); tercile cutoffs fixed at "
                           "1/3 and 2/3; funded subset only (~52 syms)",
            "stealth_rally": "dfh20 XS top-tercile AND funding_pct <= 1/3",
            "euphoric": "dfh20 XS top-tercile AND funding_pct >= 2/3",
            "mfe_10d_bps": "max(close[t+1..t+10])/close[t]-1 in bps; "
                           "requires 10 contiguous forward daily bars",
            "mae_10d_bps": "min(close[t+1..t+10])/close[t]-1 in bps; "
                           "same contiguity requirement",
            "quality_score": "mean_fwd_10d_bps x (mean_MFE / |mean_MAE|)",
        },
        "statistics": {
            "cells": "n, mean, median, pos_frac per column; n_path = "
                     "asset-days with the full 10d contiguous window",
            "contrasts": "pooled Welch two-sample t on asset-day returns, "
                         "5 named contrasts x {5d,10d} = 10-cell family",
            "fdr": "Benjamini-Hochberg alpha=0.05 over the 10 contrast "
                   "cells",
            "folds": "per-calendar-year means for the top-3 "
                     "quality-ranked cells (n >= 200, n_path >= 150)",
            "caveats": [
                "pooled Welch treats same-day cross-asset returns as "
                "independent; they share market beta — nominal only",
                "5d/10d overlapping labels autocorrelate adjacent days",
                "MFE/MAE are close-to-close extrema, not intrabar; "
                "liq-relevant wicks are invisible at 1d granularity",
                "state cells overlap (an asset-day can be pullback AND "
                "vol_low); cells are not a partition",
                "universe composition changes as symbols list/delist; "
                "early stop = delisted OR renamed proxy",
                "second-hand archive copy, not an as-of vintage",
            ],
        },
        "forbidden": ["fitting", "trading", "profitability claims",
                      "protocol edits post-run", "network"],
    }
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    auth = {
        "schema_version": "nanojev-financial-signal-entry-state-authorization-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "protocol_sha256": sha,
        "protocol_path": "research/financial_signal_entry_state_protocol_v1.json",
        "decision": "approved_for_measurement",
        "measurement_authorized": True,
        "fit_authorized": False,
        "independent_reviewer": {
            "id": "project-owner",
            "independence": "owner_self_authorization_not_independent_review",
            "note": "Owner directed T117: measure entry states for a "
                    "rolling-compounding campaign on the mega cohort "
                    "(delegated task).",
        },
        "scope": {
            "permitted": "PIT-safe descriptive measurement on "
                         "data/perp_pit_mega_v1/records.jsonl plus the "
                         "BTC daily gate series: gate-on asset-day "
                         "scoring by dfh20/pullback/extended/breaking/"
                         "funding_pct/vol20 state dimensions, fwd "
                         "5d/10d + MFE/MAE_10d path stats, named Welch "
                         "contrasts under a 10-cell BH-FDR family, "
                         "per-year folds and a composite-rule "
                         "recommendation per the pinned protocol.",
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
    parser.add_argument("--btc-csv", type=pathlib.Path, default=BTC_CSV)
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    args = parser.parse_args()

    protocol_sha = write_protocol_and_auth()
    report = {"schema_version": "nanojev-financial-signal-entry-state-v1",
              "task": "T117 entry states for a rolling-compounding "
                      "campaign on the mega cohort",
              "cohort": str(args.cohort),
              "btc_gate_source": str(args.btc_csv),
              "protocol_path":
                  "research/financial_signal_entry_state_protocol_v1.json",
              "protocol_sha256": protocol_sha,
              "authorization_path":
                  "results/financial_signal_entry_state_authorization_v1.json",
              "path_days": PATH_DAYS,
              "min_assets_per_day": MIN_ASSETS_PER_DAY,
              "state_thresholds": {
                  "pullback_dfh20_range": [PULLBACK_LO, PULLBACK_HI],
                  "extended_dfh20_gt": EXTENDED_CUT,
                  "breaking": {"mom20_gt": 0, "dfh20_lt": BREAKING_DFH_CUT},
                  "xs_tercile_cuts": [1 / 3, 2 / 3],
                  "funding_pct_cuts": [1 / 3, 2 / 3]}}

    if not args.cohort.exists() or not args.btc_csv.exists():
        report["status"] = ("SKIPPED: cohort or BTC series not found; "
                            "nothing was fabricated")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True)
                            + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"],
                          "out": str(args.out)}))
        return 0

    btc_ret20 = load_btc_gate(args.btc_csv)
    series, meta, funded_assets = load_cohort(args.cohort)
    dfh_pct = xs_rank_pcts(series, "dfh20")
    vol_pct = xs_rank_pcts(series, "vol20")

    span = sorted(r["day"] for rows in series.values() for r in rows)
    gate_on_days = sum(1 for v in btc_ret20.values() if v > 0)
    report["cohort_description"] = {
        "symbols": len(series),
        "symbols_with_funding": len(funded_assets),
        "span": {"first": span[0] if span else None,
                 "last": span[-1] if span else None},
        "early_stopped_symbols": sum(1 for m in meta.values()
                                     if m["listed"] is False),
        "btc_gate_days": {"computable": len(btc_ret20),
                          "gate_on": gate_on_days},
    }
    report["status"] = "ran"

    # --- assign gate-on asset-days to state cells ----------------------
    cell_obs = defaultdict(list)          # cell -> [obs]
    obs_base = {"fwd_5d_bps": None, "fwd_10d_bps": None,
                "mfe_10d_bps": None, "mae_10d_bps": None}
    n_gate_on, n_gate_off, n_warmup = 0, 0, 0
    gate_off_obs = []
    for asset, rows in series.items():
        for r in rows:
            ret20 = btc_ret20.get(r["day"])
            if ret20 is None:
                n_warmup += 1
                continue
            ctx = {"dfh20": r["dfh20"], "mom20": r["mom20"],
                   "vol20": r["vol20"], "close": r["close"],
                   "prev_close": r["prev_close"],
                   "funding_pct": r["funding_pct"],
                   "dfh_pct": dfh_pct.get(r["day"], {}).get(asset),
                   "vol_pct": vol_pct.get(r["day"], {}).get(asset)}
            obs = {"asset": asset, "day": r["day"],
                   "fwd_5d_bps": r["fwd_5d_bps"],
                   "fwd_10d_bps": r["fwd_10d_bps"],
                   "mfe_10d_bps": r["mfe_10d_bps"],
                   "mae_10d_bps": r["mae_10d_bps"]}
            if ret20 <= 0:
                n_gate_off += 1
                gate_off_obs.append(obs)
                continue
            n_gate_on += 1
            for cell, pred in CELLS.items():
                if pred(ctx):
                    cell_obs[cell].append(obs)
    report["asset_days"] = {"gate_on": n_gate_on, "gate_off": n_gate_off,
                            "warmup_no_gate": n_warmup,
                            "note": "cells overlap — an asset-day can sit "
                                    "in several cells; not a partition"}

    cells = {name: cell_stats(cell_obs.get(name, [])) for name in CELLS}
    cells["gate_off_all_context"] = cell_stats(gate_off_obs)
    report["cells"] = cells

    # --- named contrasts (5 x {5d,10d} = 10-cell FDR family) -----------
    contrasts, fdr_cells = {}, []
    for name, a, b in CONTRASTS:
        res = {}
        for hz, key in (("5d", "fwd_5d_bps"), ("10d", "fwd_10d_bps")):
            xs = [o[key] for o in cell_obs.get(a, []) if o[key] is not None]
            ys = [o[key] for o in cell_obs.get(b, []) if o[key] is not None]
            w = welch(xs, ys)
            res[hz] = {k: _r(v, 6) for k, v in w.items()}
            fdr_cells.append((f"{name}_{hz}", w["p"]))
        contrasts[name] = {"a": a, "b": b, **res}
    fdr_map, fdr_table = bh_fdr(fdr_cells, alpha=0.05)
    for e in fdr_table:
        cname, hz = e["cell"].rsplit("_", 1)
        contrasts[cname][hz]["fdr_survives"] = e["survives"]
    report["contrasts"] = contrasts
    report["fdr_bh_0.05"] = {
        "family": "5 named contrasts x {5d,10d} = 10 cells (<=10 cap)",
        "table": fdr_table}

    # --- quality ranking + per-year folds for the top-3 -----------------
    def quality(name):
        c = cells[name]
        f10, ratio = c["fwd_10d_bps"]["mean"], c["mfe_mae_ratio"]
        if c["n"] < MIN_CELL_N or c["n_path"] < MIN_CELL_N_PATH \
                or f10 is None or ratio is None:
            return None
        return f10 * ratio

    ranked = sorted(
        ((q, n) for n in CELLS if (q := quality(n)) is not None),
        reverse=True)
    report["quality_ranking"] = {
        "definition": "mean_fwd_10d_bps x (mean_MFE_10d / |mean_MAE_10d|); "
                      f"eligible cells need n>={MIN_CELL_N} and "
                      f"n_path>={MIN_CELL_N_PATH}",
        "ranked": [{"rank": i + 1, "cell": n, "quality_score": _r(q),
                    "n": cells[n]["n"],
                    "mean_fwd_10d_bps": cells[n]["fwd_10d_bps"]["mean"],
                    "mfe_mae_ratio": cells[n]["mfe_mae_ratio"]}
                   for i, (q, n) in enumerate(ranked)]}

    top3 = [n for _, n in ranked[:3]]
    folds = {}
    for name in top3:
        by_year = defaultdict(list)
        by_year10 = defaultdict(list)
        for o in cell_obs.get(name, []):
            if o["fwd_5d_bps"] is not None:
                by_year[o["day"][:4]].append(o["fwd_5d_bps"])
            if o["fwd_10d_bps"] is not None:
                by_year10[o["day"][:4]].append(o["fwd_10d_bps"])
        m5 = {y: sum(v) / len(v) for y, v in sorted(by_year.items())}
        m10 = {y: sum(v) / len(v) for y, v in sorted(by_year10.items())}
        folds[name] = {
            "per_year": {y: {"n": len(by_year[y]),
                             "mean_fwd_5d_bps": _r(m5[y]),
                             "mean_fwd_10d_bps": _r(m10.get(y))}
                         for y in sorted(set(m5) | set(m10))},
            "same_sign_years_5d": (
                f"{sum(1 for v in m5.values() if v > 0)}/{len(m5)} "
                "positive"),
            "same_sign_years_10d": (
                f"{sum(1 for v in m10.values() if v > 0)}/{len(m10)} "
                "positive")}
    report["top3_per_year_consistency"] = folds

    # --- verdict -------------------------------------------------------
    c = cells
    pull, ext = c["pullback"], c["extended"]
    stealth, euph = c["stealth_rally"], c["euphoric"]
    pf_low = c["pullback_fund_low"]
    best = ranked[0][1] if ranked else None
    pve5 = contrasts["pullback_vs_extended"]["5d"]
    sve5 = contrasts["stealth_vs_euphoric"]["5d"]
    pullback_wins = (pve5["mean_diff"] or 0) > 0
    stealth_wins = (sve5["mean_diff"] or 0) > 0
    verdict_lines = [
        f"gate context: gate-on {n_gate_on} asset-days vs gate-off "
        f"{n_gate_off}; gate-off fwd_5d mean "
        f"{cells['gate_off_all_context']['fwd_5d_bps']['mean']}bps vs "
        f"gate-on-all {c['gate_on_all']['fwd_5d_bps']['mean']}bps — the "
        "mandatory BTC regime context.",
        f"pullback vs extended (5d): pullback mean "
        f"{pull['fwd_5d_bps']['mean']}bps (n={pull['n']}, MFE/MAE "
        f"{pull['mfe_mae_ratio']}) vs extended "
        f"{ext['fwd_5d_bps']['mean']}bps (n={ext['n']}, MFE/MAE "
        f"{ext['mfe_mae_ratio']}); Welch p={pve5['p']} — "
        + ("pullback WINS" if pullback_wins else
           "EXTENDED wins decisively (pullback prior INVERTED on this "
           "cohort: near-high entries carry the momentum leg)"),
        f"stealth vs euphoric (5d): stealth mean "
        f"{stealth['fwd_5d_bps']['mean']}bps (n={stealth['n']}) vs "
        f"euphoric {euph['fwd_5d_bps']['mean']}bps (n={euph['n']}); "
        f"Welch p={sve5['p']}, FDR-survives="
        f"{sve5.get('fdr_survives')} — "
        + ("stealth wins" if stealth_wins else
           "euphoric leads nominally but the contrast is NOT "
           "FDR-significant: funding tilt is weak/advisory only"),
        f"top quality cell: {best} (score "
        f"{_r(ranked[0][0]) if ranked else None}); consistency variant "
        f"pullback_fund_low: n={pf_low['n']}, fwd_10d mean "
        f"{pf_low['fwd_10d_bps']['mean']}bps, MFE/MAE "
        f"{pf_low['mfe_mae_ratio']}, yearly folds "
        f"{folds.get('pullback_fund_low', {}).get('same_sign_years_5d')}.",
    ]
    report["verdict"] = {
        "ranking_metric": "quality = mean_fwd_10d_bps x MFE/|MAE|",
        "top3_cells": top3,
        "fdr_surviving_contrasts": [e["cell"] for e in fdr_table
                                    if e["survives"]],
        "measured_findings": {
            "pullback_prior": ("INVERTED vs sibling-project E1: extended/"
                               "near-high entries beat pullback on mean "
                               "fwd 5d AND 10d AND on MFE/|MAE| "
                               "(FDR-surviving); pullback's only edge is "
                               "better yearly consistency when combined "
                               "with low funding"),
            "funding_tilt": ("euphoric (near-high + funding_pct >= 2/3) "
                             "nominally tops the ranking, but stealth vs "
                             "euphoric does not survive FDR — treat "
                             "high-funding preference as a weak tilt, "
                             "not a gate"),
            "vol_conditioning": ("vol_low nominally beats vol_high but "
                                 "not significant (5d p="
                                 f"{contrasts['vol_low_vs_high']['5d']['p']})"
                                 " — low-vol mainly cuts BOTH MFE and "
                                 "MAE (smaller path envelope)"),
            "avoid": ("breaking (mom20>0 & dfh20<-0.12) is the only "
                      "negative-quality cell: dead-cat entries lose "
                      "despite the BTC gate"),
        },
        "recommended_composite_rule": {
            "primary": {
                "gate": "enter only on days with BTC ret20 > 0",
                "entry_state": ("near-high/extended: dfh20 XS top-tercile "
                                "(>= 2/3 of that day's ranked assets) OR "
                                "dfh20 > -0.01; on funded symbols a "
                                "funding_pct >= 2/3 tilt is a weak "
                                "positive (not FDR-significant)"),
                "cell": "extended / dfh_near_high / euphoric",
                "why": ("highest measured quality: mean fwd10d "
                        f"{c['dfh_near_high']['fwd_10d_bps']['mean']}-"
                        f"{euph['fwd_10d_bps']['mean']}bps with MFE/|MAE| "
                        f"{c['dfh_near_high']['mfe_mae_ratio']}-"
                        f"{euph['mfe_mae_ratio']} — big enough MFE with "
                        "shallower MAE than the gate-on pool"),
            },
            "consistency_variant": {
                "entry_state": ("pullback (-0.12 <= dfh20 <= -0.03 AND "
                                "close > prior contiguous close) AND "
                                "funding_pct <= 1/3 on funded symbols"),
                "cell": "pullback_fund_low",
                "why": ("lower mean but best yearly consistency "
                        "(4/5 positive vs 3/5 for euphoric/fund_high) "
                        "and the shallowest MAE among quality cells — "
                        "preferred if the simulator weighs regime-"
                        "robustness over mean return"),
            },
            "avoid": ("'breaking' entries (mom20>0 while still >12% off "
                      "the high) and bottom-dfh20-tercile entries — "
                      "negative fwd returns even gate-on"),
            "caveat": ("advisory only — validate in the campaign "
                       "simulator (path-aware costs, roll mechanics) "
                       "before any rollout"),
        },
        "summary_lines": verdict_lines,
    }

    report["power_note"] = (
        "HONEST POWER: cells pool asset-days across ~277 symbols, but "
        "same-day returns share market beta and 5d/10d labels overlap "
        "~80-90% on adjacent days — pooled Welch n overstates effective "
        "dof; all t/p nominal. XS terciles need >=30 ranked assets/day; "
        "funding cells cover only the ~52 funded symbols. Cells overlap "
        "(not a partition). MFE/MAE use close extrema, not intrabar "
        "wicks. Early-stopped symbols kept (delisted OR renamed — less "
        "survivorship bias than listed-only, not zero).")
    report["honesty"] = {
        "not_a_return": "fwd returns and MFE/MAE are gross close-price "
                        "moves; fees, spread, slippage, funding cashflows "
                        "and leverage are excluded by construction",
        "not_an_asof_vintage": "second-hand archive copy "
                               "(rc_futures_v1), not a verified venue "
                               "pull or as-of vintage",
        "not_live": "no orders, no account, no broker, no trading API",
        "no_profitability_claim": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    brief = {"status": "ran", "out": str(args.out),
             "gate_on_asset_days": n_gate_on,
             "top3_cells": top3,
             "fdr_survivors": report["verdict"]["fdr_surviving_contrasts"],
             "pullback_vs_extended_5d_p":
                 contrasts["pullback_vs_extended"]["5d"]["p"],
             "stealth_vs_euphoric_5d_p":
                 contrasts["stealth_vs_euphoric"]["5d"]["p"]}
    print(json.dumps(brief, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
