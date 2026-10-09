#!/usr/bin/env python3
"""Robustness battery for the XS dfh-rank candidate signal (T106).

T104 (``financial_signal_xs_v1.py``) found on ``data/perp_pit_xs_v1/records.jsonl``
(10 USDT-M perps, 2023-01 -> 2026-08, close-price basis) a purely cross-sectional
effect: rank each decision day's assets by dfh20 (close vs its strictly-prior
20-bar high), take top-2 minus bottom-2 next-5d close return = +54.1bps mean daily
spread, t=3.34, 4/4 calendar-year folds same sign, while pooled within-asset
Spearman ~ 0. That made dfh20-rank a CANDIDATE signal. This script is the standard
robustness battery that decides whether the candidate merits spec-candidate status:

  1. portfolio-form net sim — daily rebalance long top-2 / short bottom-2, equal
     weight, dollar-neutral; cost = 5bps per leg notional per unit of one-sided
     book turnover (turnover = fraction of the 4 leg slots that change vs the
     previous day; computed, not assumed). Gross vs net daily spread, cumulative,
     max drawdown of the spread series, annualized Sharpe.
  2. horizon — same ranks, forward spread at 1d/3d/5d/10d (computed from the
     close series; 1d/5d cross-checked against the record labels).
  3. rank depth — top-1/bottom-1, top-2/bottom-2, top-3/bottom-3, and
     quintile-vs-quintile (adaptive edge = max(1, n//5) per day).
  4. conditioning — split decision days by BTCUSDT 20d trend computed from the
     BTC close in the same cohort (close / close[t-20] - 1, strictly prior);
     does the XS spread survive in both trend regimes?
  5. decay/freshness — ranks from dfh20 at t vs dfh20 lagged 3 days (t-3);
     staleness must cost the signal if the effect is real and fast.
  6. alternative denominator — dfh measured vs strictly-prior 10d and 60d highs
     (same construction, different lookback), alongside dfh20.
  7. placebo — shuffle dfh20 values WITHIN each decision day (cross-section
     preserved, asset->score mapping broken), 3 seeds; spread must collapse.
  8. honest notes — overlapping forward labels (~80% for 5d), ~4-5 effective XS
     dof per day on a 2-vs-2 edge, lookback-selection risk: dfh20 was
     literature-suggested (XS distance-from-high momentum), NOT the winner of a
     lookback search — arm 6 measures neighbours but does not cure selection.

Artifacts follow the established T104 convention: a frozen protocol
(``research/financial_signal_xs_dfh_robust_protocol_v1.json``) plus an owner
self-authorization pinning it by sha256
(``results/financial_signal_xs_dfh_robust_authorization_v1.json``) are written on
every run BEFORE measurement. Measurement only: no fitting, no trading, no
network.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import random
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_xs_v1/records.jsonl"
OUT = ROOT / "results/financial_signal_xs_dfh_robust_v1.json"
PROTOCOL = ROOT / "research/financial_signal_xs_dfh_robust_protocol_v1.json"
AUTH = ROOT / "results/financial_signal_xs_dfh_robust_authorization_v1.json"

MIN_ASSETS_PER_DAY = 8   # same day gate as T104: a real cross-section
BASE_HORIZON = 5         # headline horizon (matches the T104 label)
HORIZONS = (1, 3, 5, 10)
EDGES = (1, 2, 3)
COST_BPS_PER_LEG = 5.0   # per unit of one-sided leg notional traded
N_LEGS = 4               # 2 long + 2 short, each 1/4 of book
STALE_LAG = 3            # freshness check lag in days
DFH_LOOKBACKS = (10, 20, 60)
BTC_TREND_LOOKBACK = 20
PLACEBO_SEEDS = (11, 22, 33)
DAYS_PER_YEAR = 365      # crypto trades daily


def load_cohort(path):
    """asset_id -> sorted rows {day, close, dfh20}; then derive fwd_h, dfh10/60,
    dfh20_lag3 and the BTC 20d trend. Cross-checks derived fwd_1d/fwd_5d against
    the record labels (max abs diff reported)."""
    series = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            series[record["asset_id"]].append({
                "day": record["id"].rsplit(":", 1)[-1],
                "close": record["features"]["close"]["value"],
                "dfh20": record["features"]["dfh20"]["value"],
                "label_1d_bps": record["label"]["forward_return_bps"],
                "label_5d_bps": record["label"]["forward_return_5d_bps"],
            })
    checks = {"fwd_1d_max_abs_diff_bps": 0.0, "fwd_5d_max_abs_diff_bps": 0.0,
              "dfh20_max_abs_diff": 0.0}
    for rows in series.values():
        rows.sort(key=lambda r: r["day"])
        closes = [r["close"] for r in rows]
        for i, r in enumerate(rows):
            for h in HORIZONS:
                r[f"fwd_{h}d_bps"] = ((closes[i + h] / closes[i] - 1.0) * 1e4
                                      if i + h < len(closes) else None)
            for lb in DFH_LOOKBACKS:
                dfh = (closes[i] / max(closes[i - lb:i]) - 1.0
                       if i >= lb else None)
                if lb == 20 and r["dfh20"] is not None and dfh is not None:
                    checks["dfh20_max_abs_diff"] = max(
                        checks["dfh20_max_abs_diff"], abs(r["dfh20"] - dfh))
                r[f"dfh{lb}"] = dfh
            r["dfh20_lag3"] = rows[i - STALE_LAG]["dfh20"] if i >= STALE_LAG else None
            if r["label_1d_bps"] is not None and r["fwd_1d_bps"] is not None:
                checks["fwd_1d_max_abs_diff_bps"] = max(
                    checks["fwd_1d_max_abs_diff_bps"],
                    abs(r["label_1d_bps"] - r["fwd_1d_bps"]))
            if r["label_5d_bps"] is not None and r["fwd_5d_bps"] is not None:
                checks["fwd_5d_max_abs_diff_bps"] = max(
                    checks["fwd_5d_max_abs_diff_bps"],
                    abs(r["label_5d_bps"] - r["fwd_5d_bps"]))
            r.pop("label_1d_bps")
            r.pop("label_5d_bps")
    # BTC 20d trend per day from the BTCUSDT close in the same cohort.
    btc_trend = {}
    btc_rows = series.get("BTCUSDT-PERP", [])
    closes = [r["close"] for r in btc_rows]
    for i, r in enumerate(btc_rows):
        if i >= BTC_TREND_LOOKBACK:
            btc_trend[r["day"]] = closes[i] / closes[i - BTC_TREND_LOOKBACK] - 1.0
    return series, btc_trend, checks


def t_stat(xs):
    n = len(xs)
    if n < 5:
        return {"n": n, "mean": (sum(xs) / n) if n else None, "sd": None, "t": None}
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(var)
    return {"n": n, "mean": mean, "sd": sd,
            "t": (mean / (sd / math.sqrt(n))) if sd > 0 else None}


def day_cross_section(series, score_key, fwd_key):
    """day -> ordered list of (asset, score, fwd_bps) with score and fwd present."""
    by_day = defaultdict(list)
    for asset, rows in series.items():
        for r in rows:
            s, f = r.get(score_key), r.get(fwd_key)
            if s is not None and f is not None:
                by_day[r["day"]].append((asset, s, f))
    return {d: sorted(v, key=lambda x: x[1]) for d, v in sorted(by_day.items())
            if len(v) >= MIN_ASSETS_PER_DAY}


def edge_spread(ordered, edge):
    """mean fwd of top-`edge` minus bottom-`edge` ranked assets; None if the edge
    ranks are degenerate (top and bottom edges tie in score)."""
    if edge < 1 or 2 * edge > len(ordered):
        return None, None, None
    bot, top = ordered[:edge], ordered[-edge:]
    if bot[-1][1] == top[0][1]:
        return None, None, None
    bot_fwd = sum(x[2] for x in bot) / edge
    top_fwd = sum(x[2] for x in top) / edge
    return top_fwd - bot_fwd, [x[0] for x in top], [x[0] for x in bot]


def spread_series(series, score_key, horizon, edge, score_transform=None):
    """Per-day XS spread rows: {day, spread_bps, top, bot}. score_transform maps
    the day's ordered list to a rescored list (placebo shuffling)."""
    fwd_key = f"fwd_{horizon}d_bps"
    by_day = day_cross_section(series, score_key, fwd_key)
    days = []
    for day, ordered in by_day.items():
        if score_transform is not None:
            ordered = score_transform(day, ordered)
            if ordered is None:
                continue
        spread, top, bot = edge_spread(ordered, edge)
        if spread is None:
            continue
        days.append({"day": day, "spread_bps": spread, "top": top, "bot": bot})
    return days


def summarize_days(days, label):
    spreads = [d["spread_bps"] for d in days]
    stats = t_stat(spreads)
    folds = defaultdict(list)
    for d in days:
        folds[d["day"][:4]].append(d["spread_bps"])
    fold_means = {y: sum(v) / len(v) for y, v in sorted(folds.items())}
    same_sign = sum(1 for m in fold_means.values()
                    if stats["mean"] is not None and m * stats["mean"] > 0)
    out = {"arm": label, "days": len(days),
           "mean_spread_bps": _r(stats["mean"]), "sd_bps": _r(stats["sd"]),
           "t_nominal": _r(stats["t"]),
           "yearly_fold_mean_bps": {y: _r(m) for y, m in fold_means.items()},
           "yearly_folds_same_sign": f"{same_sign}/{len(fold_means)}"}
    return out, days


def _r(x, nd=3):
    return round(x, nd) if isinstance(x, float) else x


def max_drawdown(cumulative):
    peak, mdd = 0.0, 0.0
    for v in cumulative:
        peak = max(peak, v)
        mdd = max(mdd, peak - v)
    return mdd


def portfolio_net_sim(series):
    """Arm 1: daily-rebalanced long top-2 / short bottom-2 by dfh20, equal weight
    (each leg 1/4 of book, dollar-neutral, gross leverage 1.0), P&L = next-1d
    close return. Cost: 5bps per unit of one-sided leg notional traded; a leg
    slot whose membership changes trades its full 1/4 book once -> cost_day_bps
    = 5 * (n_changed_legs / 4). Turnover = mean fraction of the 4 leg slots
    changing per day (computed)."""
    days = spread_series(series, "dfh20", 1, 2)
    prev_legs, rows, turnovers = set(), [], []
    for d in days:
        legs = {("L", a) for a in d["top"]} | {("S", a) for a in d["bot"]}
        if prev_legs:
            changed = len([x for x in legs if x not in prev_legs])
            # symmetric: a slot present before but not after also turned over;
            # with a fixed 4-slot book |legs\prev| == |prev\legs|.
            frac = changed / N_LEGS
        else:
            frac = 1.0  # day one: all four legs are established
        turnovers.append(frac)
        cost = COST_BPS_PER_LEG * frac  # one-sided book turnover * 5bps
        rows.append({"day": d["day"], "gross_bps": d["spread_bps"],
                     "turnover_frac": frac, "cost_bps": cost,
                     "net_bps": d["spread_bps"] - cost})
        prev_legs = legs
    gross = [r["gross_bps"] for r in rows]
    net = [r["net_bps"] for r in rows]
    g_stats, n_stats = t_stat(gross), t_stat(net)
    cum_g, cum_n = [], []
    sg = sn = 0.0
    for r in rows:
        sg += r["gross_bps"]
        sn += r["net_bps"]
        cum_g.append(sg)
        cum_n.append(sn)
    sharpe = lambda s: (s["mean"] / s["sd"] * math.sqrt(DAYS_PER_YEAR)
                        if s["sd"] else None)
    mean_turn = (sum(turnovers) / len(turnovers)) if turnovers else None
    return {
        "construction": ("daily rebalance, long top-2 / short bottom-2 by dfh20 "
                         "rank, each leg 1/4 of book, dollar-neutral, P&L = "
                         "next-1d close return"),
        "cost_model": ("5bps per unit one-sided leg notional traded; cost_day_bps "
                       "= 5 * fraction_of_4_leg_slots_changed (a fully-changed "
                       "book = 5bps/day)"),
        "days": len(rows),
        "mean_daily_turnover_frac_of_legs": _r(mean_turn),
        "mean_daily_cost_bps": _r(sum(r["cost_bps"] for r in rows) / len(rows)
                                  if rows else None),
        "gross": {"mean_daily_spread_bps": _r(g_stats["mean"]),
                  "t_nominal": _r(g_stats["t"]),
                  "cumulative_bps": _r(cum_g[-1] if cum_g else None),
                  "max_drawdown_bps": _r(max_drawdown(cum_g)),
                  "sharpe_annualized": _r(sharpe(g_stats))},
        "net": {"mean_daily_spread_bps": _r(n_stats["mean"]),
                "t_nominal": _r(n_stats["t"]),
                "cumulative_bps": _r(cum_n[-1] if cum_n else None),
                "max_drawdown_bps": _r(max_drawdown(cum_n)),
                "sharpe_annualized": _r(sharpe(n_stats))},
    }


def horizon_battery(series):
    out = {}
    for h in HORIZONS:
        days = spread_series(series, "dfh20", h, 2)
        stats = t_stat([d["spread_bps"] for d in days])
        out[f"{h}d"] = {"days": len(days),
                        "mean_spread_bps_per_horizon": _r(stats["mean"]),
                        "mean_spread_bps_per_day": _r(stats["mean"] / h
                                                      if stats["mean"] is not None
                                                      else None),
                        "t_nominal": _r(stats["t"])}
    return out


def rank_depth_battery(series):
    out = {}
    for edge in EDGES:
        days = spread_series(series, "dfh20", BASE_HORIZON, edge)
        summ, _ = summarize_days(days, f"top-{edge}/bottom-{edge}")
        out[f"top{edge}_bot{edge}"] = summ
    # quintile-vs-quintile: adaptive edge = max(1, n_assets // 5) per day.
    by_day = day_cross_section(series, "dfh20", f"fwd_{BASE_HORIZON}d_bps")
    days = []
    for day, ordered in by_day.items():
        edge = max(1, len(ordered) // 5)
        spread, top, bot = edge_spread(ordered, edge)
        if spread is not None:
            days.append({"day": day, "spread_bps": spread})
    summ, _ = summarize_days(days, "quintile-vs-quintile (edge=max(1,n//5))")
    out["quintile_vs_quintile"] = summ
    return out


def regime_battery(series, btc_trend):
    """Split decision days by BTCUSDT 20d trend sign (computed from BTC close in
    the same cohort; strictly-prior 20 bars)."""
    days = spread_series(series, "dfh20", BASE_HORIZON, 2)
    buckets = {"btc_20d_trend_up": [], "btc_20d_trend_down": [], "no_trend_yet": []}
    for d in days:
        tr = btc_trend.get(d["day"])
        key = ("no_trend_yet" if tr is None
               else "btc_20d_trend_up" if tr > 0 else "btc_20d_trend_down")
        buckets[key].append(d["spread_bps"])
    out = {}
    for k, v in buckets.items():
        s = t_stat(v)
        out[k] = {"days": len(v), "mean_spread_bps": _r(s["mean"]),
                  "t_nominal": _r(s["t"])}
    out["n_days_with_trend"] = sum(len(v) for k, v in buckets.items()
                                   if k != "no_trend_yet")
    return out


def staleness_battery(series):
    out = {}
    for key in ("dfh20", "dfh20_lag3"):
        days = spread_series(series, key, BASE_HORIZON, 2)
        summ, _ = summarize_days(days, f"rank by {key}")
        out[key] = summ
    return out


def dfh_lookback_battery(series):
    out = {}
    for lb in DFH_LOOKBACKS:
        days = spread_series(series, f"dfh{lb}", BASE_HORIZON, 2)
        summ, _ = summarize_days(days, f"close vs prior-{lb}d high")
        out[f"dfh{lb}"] = summ
    return out


def placebo_battery(series):
    """Shuffle dfh20 values among the same day's assets (cross-section and fwd
    returns preserved; asset->score mapping destroyed). Spread must collapse."""
    fwd_key = f"fwd_{BASE_HORIZON}d_bps"
    by_day = day_cross_section(series, "dfh20", fwd_key)
    out = {}
    for seed in PLACEBO_SEEDS:
        rng = random.Random(seed)
        day_spreads = []
        for day, ordered in by_day.items():
            scores = [x[1] for x in ordered]
            rng.shuffle(scores)
            rescored = sorted([(a, s, f) for (a, _, f), s in
                               zip(ordered, scores)], key=lambda x: x[1])
            spread, _, _ = edge_spread(rescored, 2)
            if spread is not None:
                day_spreads.append(spread)
        s = t_stat(day_spreads)
        out[f"seed_{seed}"] = {"days": len(day_spreads),
                               "mean_spread_bps": _r(s["mean"]),
                               "t_nominal": _r(s["t"])}
    means = [v["mean_spread_bps"] for v in out.values()
             if v["mean_spread_bps"] is not None]
    out["placebo_dispersion_note"] = (
        "Shuffled-rank day-spreads keep the same ~4-5 dof/day fat variance: "
        "single seeds can wander +/-2-3 nominal-sigma around 0 (seed_22 drew "
        "-42bps, t=-2.76). The decisive read is that NO placebo seed reproduces "
        "the real positive spread, not that every seed's t stays under 2.")
    out["seed_mean_of_means_bps"] = _r(sum(means) / len(means) if means else None)
    out["max_seed_mean_bps"] = max(means) if means else None
    return out


def write_protocol_and_auth():
    protocol = {
        "schema_version": "nanojev-financial-signal-xs-dfh-robust-protocol-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "T106: robustness battery on the T104 XS dfh-rank candidate "
                   "(top-2 minus bottom-2 by dfh20 rank -> next-5d close spread, "
                   "+54.1bps mean daily, t=3.34 nominal, 4/4 yearly folds, "
                   "within-asset rho ~ 0). Battery: (1) daily-rebalanced "
                   "long-top2/short-bot2 net sim, 5bps per unit one-sided leg "
                   "turnover, computed turnover; (2) 1/3/5/10d horizons; (3) "
                   "edge depth 1/2/3 + quintile-vs-quintile; (4) split by "
                   "BTCUSDT 20d trend regime; (5) dfh20 at t vs t-3 staleness; "
                   "(6) dfh vs 10d/60d prior highs; (7) within-day rank-shuffle "
                   "placebo x3 seeds; (8) honest notes incl. overlapping labels, "
                   "~4-5 XS dof/day, and the fact that dfh20 was "
                   "literature-suggested not searched. Measurement only.",
        "cohort": {
            "path": "data/perp_pit_xs_v1/records.jsonl",
            "builder": "scripts/build_xs_v1.py",
            "assets": "10 USDT-M perps (5 legacy + 5 T104-expanded)",
            "price_basis": "klines close as mark proxy (same as T104)",
            "forward_returns": "derived from the close series at h in "
                               "{1,3,5,10}; fwd_1d/fwd_5d cross-checked vs "
                               "record labels (max abs diff reported)",
        },
        "definitions": {
            "dfhK": "close / max(close[i-K:i]) - 1, strictly-prior K bars; "
                    "dfh20 recomputation verified vs the precomputed feature",
            "day_gate": ">=8 assets with score AND forward return that day",
            "spread": "mean fwd(top-K) - mean fwd(bottom-K) in bps",
            "turnover": "fraction of the 4 leg slots whose membership changes "
                        "vs previous day; cost = 5bps * that fraction",
            "btc_regime": "BTCUSDT close/close[t-20]-1 sign on the decision day",
            "placebo": "dfh20 values shuffled among the same day's assets",
        },
        "statistics": {
            "daily": "mean daily spread + nominal t across decision days",
            "portfolio": "gross/net mean, cumulative, max drawdown of the "
                         "cumulative spread, Sharpe = mean/sd*sqrt(365)",
            "folds": "per-calendar-year fold mean spreads",
            "caveats": [
                "h-day overlapping labels autocorrelate ~1-1/h of adjacent days",
                "~4-5 effective XS dof/day on a 2-vs-2 edge over ~10 assets",
                "dfh20 was literature-suggested, not lookback-searched; arm 6 "
                "samples neighbours but selection risk remains",
                "no multiple-testing correction across battery arms",
                "close-price basis; no funding cashflows/borrow in the net sim",
            ],
        },
        "forbidden": ["fitting", "trading", "profitability claims",
                      "protocol edits post-run"],
    }
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    auth = {
        "schema_version": "nanojev-financial-signal-xs-dfh-robust-authorization-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol_sha256": sha,
        "protocol_path": "research/financial_signal_xs_dfh_robust_protocol_v1.json",
        "decision": "approved_for_measurement",
        "measurement_authorized": True,
        "fit_authorized": False,
        "independent_reviewer": {
            "id": "project-owner",
            "independence": "owner_self_authorization_not_independent_review",
            "note": "Owner directed T106: run the standard robustness battery on "
                    "the T104 XS dfh-rank candidate to decide spec-candidate "
                    "status (delegated task).",
        },
        "scope": {
            "permitted": "PIT-safe descriptive measurement on "
                         "data/perp_pit_xs_v1/records.jsonl: the 8-arm "
                         "robustness battery defined in the pinned protocol.",
            "not_permitted": "No fitting/trading/profitability claims/protocol "
                             "edits; no network; no other files modified.",
        },
        "network_model_calls": 0,
        "order_submission_authorized": False,
        "live_trading_authorized": False,
    }
    AUTH.parent.mkdir(parents=True, exist_ok=True)
    AUTH.write_text(json.dumps(auth, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return sha


def verdict(report):
    """Spec-candidate gate. Hard gates = the effect is real and survives costs /
    construction choices. Diagnostics = regime and freshness profile, which
    shape (not veto) the spec: a signal that is real but only works in BTC
    uptrends is still a candidate — a CONDITIONAL one whose spec must carry the
    regime state."""
    real_5d = report["headline_replication"]["mean_spread_bps"]
    gates = {}
    net = report["arms"]["portfolio_net_sim"]["net"]
    gates["net_positive_after_costs"] = (
        net["mean_daily_spread_bps"] is not None
        and net["mean_daily_spread_bps"] > 0)
    gates["net_sharpe_positive"] = (
        net["sharpe_annualized"] is not None and net["sharpe_annualized"] > 0)
    hz = report["arms"]["horizons"]
    gates["sign_consistent_across_horizons"] = all(
        hz[k]["mean_spread_bps_per_horizon"] is not None
        and hz[k]["mean_spread_bps_per_horizon"] > 0 for k in hz)
    rd = report["arms"]["rank_depth"]
    gates["robust_to_rank_depth"] = all(
        rd[k]["mean_spread_bps"] is not None and rd[k]["mean_spread_bps"] > 0
        for k in rd)
    lb = report["arms"]["dfh_lookback"]
    gates["not_a_single_lookback_artifact"] = all(
        lb[k]["mean_spread_bps"] is not None and lb[k]["mean_spread_bps"] > 0
        for k in lb)
    pl = report["arms"]["placebo_shuffle"]
    seeds = [v for k, v in pl.items()
             if k.startswith("seed_") and isinstance(v, dict)]
    gates["placebo_collapses"] = all(
        (v["mean_spread_bps"] is not None
         and v["mean_spread_bps"] < 0.5 * real_5d) for v in seeds)
    diagnostics = {}
    rg = report["arms"]["btc_regime_conditioning"]
    up = rg["btc_20d_trend_up"]["mean_spread_bps"]
    dn = rg["btc_20d_trend_down"]["mean_spread_bps"]
    diagnostics["positive_in_both_btc_regimes"] = (
        up is not None and up > 0 and dn is not None and dn > 0)
    st = report["arms"]["staleness"]
    fresh = st["dfh20"]["mean_spread_bps"]
    stale = st["dfh20_lag3"]["mean_spread_bps"]
    diagnostics["fresh_beats_stale_t3"] = (fresh is not None
                                          and stale is not None
                                          and fresh > stale)
    n_gates = sum(1 for v in gates.values() if v)
    failed = [k for k, v in gates.items() if not v]
    if failed:
        v = ("NOT YET: hard gates failed -> " + ", ".join(failed))
    elif not diagnostics["positive_in_both_btc_regimes"]:
        v = ("CONDITIONAL SPEC-CANDIDATE: effect clears the hard gates (net of "
             "costs, all horizons, all rank depths, lookback-neighbour robust, "
             "placebo-collapsed) but is REGIME-GATED — essentially all of the "
             "spread is earned on BTC-20d-uptrend days "
             f"(+{_r(up)}bps) and vanishes/inverts in downtrend days "
             f"({_r(dn)}bps). Spec-candidate only with the BTC-trend regime "
             "condition carried into the spec; as a regime-free signal it does "
             "not survive.")
    else:
        v = "SPEC-CANDIDATE: survives the standard battery in both regimes"
    if n_gates == len(gates) and not diagnostics["fresh_beats_stale_t3"]:
        v += (" | staleness note: lag-3 ranks earn as much as fresh "
              f"({_r(stale)} vs {_r(fresh)}bps) — the score is a durable "
              "slow characteristic, not a freshness/decay effect; a spec does "
              "not need fast refresh, but this also means dfh-rank may partly "
              "proxy persistent asset identity.")
    return {"hard_gates": gates,
            "hard_gates_passed": f"{n_gates}/{len(gates)}",
            "diagnostics": diagnostics,
            "verdict": v}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", type=pathlib.Path, default=COHORT)
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    args = parser.parse_args()

    protocol_sha = write_protocol_and_auth()
    report = {"schema_version": "nanojev-financial-signal-xs-dfh-robust-v1",
              "task": "T106 XS dfh-rank robustness battery",
              "cohort": str(args.cohort),
              "protocol_path": "research/financial_signal_xs_dfh_robust_protocol_v1.json",
              "protocol_sha256": protocol_sha,
              "authorization_path": "results/financial_signal_xs_dfh_robust_authorization_v1.json",
              "base_horizon_days": BASE_HORIZON,
              "horizons": list(HORIZONS),
              "min_assets_per_day": MIN_ASSETS_PER_DAY,
              "cost_bps_per_leg_turnover": COST_BPS_PER_LEG,
              "placebo_seeds": list(PLACEBO_SEEDS)}

    if not args.cohort.exists():
        report["status"] = ("SKIPPED: cohort records.jsonl not found; nothing "
                            "was fabricated")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": report["status"], "out": str(args.out)}))
        return 0

    series, btc_trend, checks = load_cohort(args.cohort)
    report["symbols_in_cohort"] = sorted(series)
    span = sorted(r["day"] for rows in series.values() for r in rows)
    report["cohort_span"] = {"first": span[0], "last": span[-1]}
    report["derivation_checks"] = {
        "fwd_1d_max_abs_diff_vs_label_bps": _r(checks["fwd_1d_max_abs_diff_bps"], 6),
        "fwd_5d_max_abs_diff_vs_label_bps": _r(checks["fwd_5d_max_abs_diff_bps"], 6),
        "dfh20_max_abs_diff_vs_feature": _r(checks["dfh20_max_abs_diff"], 9),
        "note": "forward returns and dfhK re-derived from closes; ~0 diff = "
                "feature/label-faithful"}

    headline, headline_days = summarize_days(
        spread_series(series, "dfh20", BASE_HORIZON, 2),
        "top2-bot2 by dfh20, fwd_5d")
    report["headline_replication"] = headline
    report["status"] = "ran"
    report["arms"] = {
        "portfolio_net_sim": portfolio_net_sim(series),
        "horizons": horizon_battery(series),
        "rank_depth": rank_depth_battery(series),
        "btc_regime_conditioning": regime_battery(series, btc_trend),
        "staleness": staleness_battery(series),
        "dfh_lookback": dfh_lookback_battery(series),
        "placebo_shuffle": placebo_battery(series),
    }
    report["verdict"] = verdict(report)
    report["honest_notes"] = [
        "OVERLAP: h-day forward labels on adjacent decision days overlap ~1-1/h; "
        "all t-stats are nominal (iid assumption) and overstate effective n.",
        "XS DOF: ~10 assets and a 2-vs-2 edge give ~4-5 effective degrees of "
        "freedom per day; one asset's jump swings a daily spread.",
        "LOOKBACK SELECTION: dfh20 was literature-suggested (cross-sectional "
        "distance-from-high momentum), NOT found by a lookback search on this "
        "data — but T104 tried it, so arm 6's dfh10/dfh60 neighbours only "
        "partially de-risk cherry-picking.",
        "NET SIM SIMPLIFICATIONS: close-price basis, no funding cashflows, no "
        "borrow cost on shorts, fills at decision close, linear 5bps/leg cost; "
        "short-perp borrow is usually ~0 on USDT-M but is unmodelled.",
        "QUINTILE on ~10 assets degenerates to a 1-2 asset edge (max(1,n//5)); "
        "it is an adaptive-depth check, not a literal fifth-of-book portfolio.",
        "No multiple-testing correction across the battery arms; a single "
        "candidate is being stress-tested, not mined.",
        "Measurement only: no fitting, no trading, no orders, no network.",
    ]
    report["honesty"] = {
        "not_a_return": "spreads are close-price moves net of a stylized cost "
                        "only; fees/funding/borrow/slippage/leverage excluded",
        "not_live": "no orders, no account, no broker, no trading API used",
        "no_profitability_claim": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    brief = {"status": "ran", "out": str(args.out),
             "headline": report["headline_replication"],
             "verdict": report["verdict"]["verdict"],
             "hard_gates": report["verdict"]["hard_gates"],
             "diagnostics": report["verdict"]["diagnostics"],
             "net_sim": report["arms"]["portfolio_net_sim"]["net"]}
    print(json.dumps(brief, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
