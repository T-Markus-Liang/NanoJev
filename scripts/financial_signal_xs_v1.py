#!/usr/bin/env python3
"""Cross-sectional signal arms on the FULL 10-asset cohort (T104).

Reads the built cohort ``data/perp_pit_xs_v1/records.jsonl`` (``build_xs_v1.py``:
10 USDT-M perps, full archive span, close-as-mark proxy) and runs the two approved
XS arms plus pooled per-asset controls:

  funding_pct   XS CARRY. Per-asset mid-rank pct of today's last_funding_rate vs its
                trailing-90 daily values (repo convention, PIT-safe, MIN_WINDOW=20).
                Per decision day with >=8 assets present: rank, take top-2 vs
                bottom-2, spread = mean fwd_5d(bottom2) - mean fwd_5d(top2). The
                funding-crowding direction says high-funding assets underperform, so
                positive spread = carry works. HONEST PRIOR: the XS funding-carry
                literature reports the effect dead post-2020 on majors; we expect a
                NULL and say so before running.
  dfh20         XS MOMENTUM. Rank by the precomputed dfh20 feature (close vs its
                strictly-prior 20-bar high); spread = mean fwd_5d(top2) -
                mean fwd_5d(bot2) (near-high -> continuation direction).
  controls      pooled Spearman(score, fwd_5d_bps) across all asset-days plus a
                per-asset Spearman table — the within-asset benchmark each XS rank
                contrast must beat to claim cross-sectional content.

Reported per arm: mean daily XS spread with a t across days, a pooled Welch
two-sample contrast (all top-2 asset-days vs all bottom-2 asset-days), per-calendar-
year fold sign consistency, and an honest power note: ~10 assets gives only ~4-5
effective dof per day (2-vs-2 means), 5d overlapping labels autocorrelate adjacent
days, and no multiple-testing correction is applied — this is measurement, not a
signal claim.

Artifacts written on every run (before measurement): a frozen protocol
(``research/financial_signal_xs_protocol_v1.json``) and an owner self-authorization
(``results/financial_signal_xs_authorization_v1.json``) that pins the protocol by
sha256 — same convention as the T101 dfh arm. Measurement only: no fitting, no
trading, no network.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_xs_v1/records.jsonl"
OUT = ROOT / "results/financial_signal_xs_v1.json"
PROTOCOL = ROOT / "research/financial_signal_xs_protocol_v1.json"
AUTH = ROOT / "results/financial_signal_xs_authorization_v1.json"
LOOKBACK = 90          # trailing-90 records per asset (~90d at daily granularity)
MIN_WINDOW = 20        # floor for a usable trailing pct (repo convention)
MIN_ASSETS_PER_DAY = 8 # task gate: a top-2/bottom-2 contrast needs a real cross-section
FWD_DAYS = 5


def mid_rank_pct(window, x):
    """Mid-rank percentile of x vs trailing window (ties count half)."""
    if x is None or not window:
        return None
    return (sum(1 for w in window if w < x)
            + 0.5 * sum(1 for w in window if w == x)) / len(window)


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(xs, ys):
    n = len(xs)
    if n < 10:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def t_stat(xs):
    """Mean/sd/t of a series (obs treated as independent; see power note)."""
    n = len(xs)
    if n < 5:
        return {"n": n, "mean": (sum(xs) / n) if n else None, "t": None}
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(var)
    return {"n": n, "mean": mean, "sd": sd,
            "t": (mean / (sd / math.sqrt(n))) if sd > 0 else None}


def welch(xs, ys):
    """Welch two-sample t and Welch-Satterthwaite df."""
    nx, ny = len(xs), len(ys)
    if nx < 2 or ny < 2:
        return {"n_top": nx, "n_bottom": ny, "t": None, "df": None}
    mx, my = sum(xs) / nx, sum(ys) / ny
    vx = sum((x - mx) ** 2 for x in xs) / (nx - 1)
    vy = sum((y - my) ** 2 for y in ys) / (ny - 1)
    denom = vx / nx + vy / ny
    if denom <= 0:
        return {"n_top": nx, "n_bottom": ny, "mean_top": mx, "mean_bottom": my,
                "t": None, "df": None}
    t = (mx - my) / math.sqrt(denom)
    df = denom ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    return {"n_top": nx, "n_bottom": ny, "mean_top": mx, "mean_bottom": my,
            "t": t, "df": df}


def load_cohort(path):
    """asset_id -> sorted list of {day, funding, dfh20, fwd_5d_bps} rows."""
    series = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            features = record["features"]
            series[record["asset_id"]].append({
                "day": record["id"].rsplit(":", 1)[-1],
                "decision_ns": record["decision_ns"],
                "funding": features["last_funding_rate"]["value"],
                "dfh20": features["dfh20"]["value"],
                "fwd_5d_bps": record["label"]["forward_return_5d_bps"],
            })
    for rows in series.values():
        rows.sort(key=lambda r: r["day"])
    return series


def score_series(series):
    """Add the funding_pct trailing-window score to each row (dfh20 is built-in)."""
    for rows in series.values():
        fund = [r["funding"] for r in rows]
        for i, r in enumerate(rows):
            fwin = [x for x in fund[max(0, i - LOOKBACK):i] if x is not None]
            r["funding_pct"] = mid_rank_pct(fwin, r["funding"]) \
                if len(fwin) >= MIN_WINDOW else None


def cross_section_arm(series, score_key, spread_sign):
    """Per decision day (>=8 assets): top-2 vs bottom-2 next-5d spread.

    spread_sign=+1 -> fwd(bottom2)-fwd(top2) (funding-crowding/carry direction);
    spread_sign=-1 -> fwd(top2)-fwd(bottom2) (momentum continuation direction).
    Returns per-day rows, pooled Welch on the underlying asset-day returns,
    t across daily spreads, and per-calendar-year fold sign consistency.
    """
    by_day = defaultdict(list)
    for asset, rows in series.items():
        for r in rows:
            if r.get(score_key) is not None and r["fwd_5d_bps"] is not None:
                by_day[r["day"]].append((asset, r[score_key], r["fwd_5d_bps"]))

    days, pooled_top, pooled_bot = [], [], []
    for day in sorted(by_day):
        xs = by_day[day]
        if len(xs) < MIN_ASSETS_PER_DAY:
            continue
        ordered = sorted(xs, key=lambda x: x[1])
        bot2, top2 = ordered[:2], ordered[-2:]
        if bot2[-1][1] == top2[0][1]:
            continue  # degenerate day: no dispersion at the ranked edges
        bot_fwd = sum(x[2] for x in bot2) / 2.0
        top_fwd = sum(x[2] for x in top2) / 2.0
        days.append({"day": day, "n_assets": len(xs),
                     "top2_assets": sorted(x[0] for x in top2),
                     "bottom2_assets": sorted(x[0] for x in bot2),
                     "top2_fwd_bps": top_fwd, "bottom2_fwd_bps": bot_fwd,
                     "spread_bps": spread_sign * (bot_fwd - top_fwd)})
        pooled_top.extend(x[2] for x in top2)
        pooled_bot.extend(x[2] for x in bot2)

    daily = t_stat([d["spread_bps"] for d in days])
    # Pooled Welch on the signed-contrast basis: the two contrast legs.
    leg_a, leg_b = (pooled_bot, pooled_top) if spread_sign > 0 \
        else (pooled_top, pooled_bot)
    pooled_welch = welch(leg_a, leg_b)

    folds = defaultdict(list)
    for d in days:
        folds[d["day"][:4]].append(d["spread_bps"])
    fold_means = {year: sum(v) / len(v) for year, v in sorted(folds.items())}
    pooled_mean = daily["mean"]
    consistent = sum(1 for m in fold_means.values()
                     if pooled_mean is not None and m * pooled_mean > 0)

    n_assets = [d["n_assets"] for d in days]
    return {
        "score": score_key,
        "spread_direction": ("fwd(bottom2)-fwd(top2) [carry: high funding should "
                             "underperform]" if spread_sign > 0 else
                             "fwd(top2)-fwd(bottom2) [momentum: near-high should "
                             "continue]"),
        "days_evaluated": len(days),
        "assets_per_day": {"min": min(n_assets) if n_assets else None,
                           "median": (sorted(n_assets)[len(n_assets) // 2]
                                      if n_assets else None),
                           "max": max(n_assets) if n_assets else None},
        "mean_daily_spread_bps": {k: (round(v, 3) if isinstance(v, float) else v)
                                  for k, v in daily.items()},
        "pooled_welch": {k: (round(v, 3) if isinstance(v, float) else v)
                         for k, v in pooled_welch.items()},
        "fold_sign_consistency": {"folds": len(fold_means),
                                  "same_sign_as_pooled": consistent,
                                  "fold_mean_bps": {y: round(m, 3)
                                                    for y, m in fold_means.items()}},
        "per_day": days,
    }


def controls(series, score_key):
    """Pooled + per-asset Spearman(score, fwd_5d_bps) — the within-asset benchmark."""
    pooled_x, pooled_y, per_asset = [], [], {}
    for asset, rows in sorted(series.items()):
        xs = [r[score_key] for r in rows
              if r.get(score_key) is not None and r["fwd_5d_bps"] is not None]
        ys = [r["fwd_5d_bps"] for r in rows
              if r.get(score_key) is not None and r["fwd_5d_bps"] is not None]
        per_asset[asset] = {"n": len(xs),
                            "spearman": (round(spearman(xs, ys), 4)
                                       if spearman(xs, ys) is not None else None)}
        pooled_x.extend(xs)
        pooled_y.extend(ys)
    return {"pooled_spearman": (round(spearman(pooled_x, pooled_y), 4)
                              if spearman(pooled_x, pooled_y) is not None else None),
            "pooled_n": len(pooled_x),
            "per_asset": per_asset}


def write_protocol_and_auth():
    protocol = {
        "schema_version": "nanojev-financial-signal-xs-protocol-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "T104: first REAL cross-sectional arm on the full 10-asset "
                   "USDT-M perp cohort (5 legacy + 5 T104-expanded symbols, full "
                   "archive span). Arm 1 = XS funding carry (funding_pct rank, "
                   "top-2 vs bottom-2 next-5d close return spread; honest prior "
                   "NULL — post-2020 literature reports the carry dead on majors). "
                   "Arm 2 = XS distance-from-high momentum (dfh20 rank, same "
                   "top-2/bottom-2 spread; direction prior POSITIVE). Pooled "
                   "per-asset Spearman controls benchmark each rank contrast. "
                   "Measurement only — no fitting, no trading, no profitability "
                   "claims.",
        "cohort": {
            "path": "data/perp_pit_xs_v1/records.jsonl",
            "builder": "scripts/build_xs_v1.py",
            "assets": "10 USDT-M perps: BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT "
                      "(data/binance_vision_v1) + ADAUSDT DOGEUSDT LINKUSDT "
                      "LTCUSDT DOTUSDT (data/binance_xs_v1)",
            "price_basis": "klines close as the mark proxy for ALL symbols (bounded "
                           "T104 slice has no markPriceKlines); deviation quantified "
                           "on legacy symbols in perp_pit_xs_v1/build_summary.json",
            "targets": "label.forward_return_5d_bps (gross close-to-close, "
                       "contiguous daily bars only)",
        },
        "definitions": {
            "funding_pct": "per-asset mid-rank pct of last_funding_rate vs its "
                           "trailing-90 daily record values (MIN_WINDOW=20)",
            "dfh20": "precomputed feature: close / max(close[i-20:i]) - 1 "
                     "(strictly prior 20 contiguous bars)",
            "decision_day_gate": ">=8 assets with score AND fwd_5d on that day",
            "spread": "mean fwd_5d of the 2 extreme-ranked assets on the positive "
                      "side minus the 2 on the negative side, signed per arm",
        },
        "statistics": {
            "daily": "mean daily XS spread + t across decision days",
            "pooled": "Welch two-sample t (all top-2 vs all bottom-2 asset-days) "
                      "with Welch-Satterthwaite df",
            "folds": "per-calendar-year fold mean spreads; sign consistency count",
            "controls": "pooled + per-asset Spearman(score, fwd_5d)",
            "caveats": [
                "top-2/bottom-2 on ~10 assets = ~4-5 effective dof per day",
                "5d overlapping labels autocorrelate adjacent days; t is nominal",
                "no multiple-testing correction across the two arms",
                "cross-section composition is stable only while all symbols trade",
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
        "schema_version": "nanojev-financial-signal-xs-authorization-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol_sha256": sha,
        "protocol_path": "research/financial_signal_xs_protocol_v1.json",
        "decision": "approved_for_measurement",
        "measurement_authorized": True,
        "fit_authorized": False,
        "independent_reviewer": {
            "id": "project-owner",
            "independence": "owner_self_authorization_not_independent_review",
            "note": "Owner directed T104: expand the XS pilot to full history "
                    "(bounded 2-kind fetch) and run the funding-rank/dfh-rank XS "
                    "arms on the 10-asset cohort (delegated task).",
        },
        "scope": {
            "permitted": "PIT-safe descriptive measurement on "
                         "data/perp_pit_xs_v1/records.jsonl: per-day top-2 vs "
                         "bottom-2 XS spreads by funding_pct and dfh20 ranks, "
                         "pooled Welch, per-year fold sign counts, pooled and "
                         "per-asset Spearman controls.",
            "not_permitted": "No fitting/trading/profitability claims/protocol "
                             "edits; no additional file kinds or symbols beyond "
                             "the bounded T104 fetch.",
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
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", type=pathlib.Path, default=COHORT)
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    args = parser.parse_args()

    protocol_sha = write_protocol_and_auth()

    report = {"schema_version": "nanojev-financial-signal-xs-v1",
              "task": "T104 full-history cross-sectional arms",
              "cohort": str(args.cohort),
              "protocol_path": "research/financial_signal_xs_protocol_v1.json",
              "protocol_sha256": protocol_sha,
              "authorization_path": "results/financial_signal_xs_authorization_v1.json",
              "horizon_days": FWD_DAYS,
              "lookback": LOOKBACK, "min_window": MIN_WINDOW,
              "min_assets_per_day": MIN_ASSETS_PER_DAY,
              "edge_size": 2}

    if not args.cohort.exists():
        report["status"] = ("SKIPPED: cohort records.jsonl not found (fetch/build "
                            "incomplete); nothing was fabricated")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": report["status"], "out": str(args.out)}))
        return 0

    series = load_cohort(args.cohort)
    score_series(series)
    n_symbols = sum(1 for rows in series.values() if rows)
    report["symbols_in_cohort"] = sorted(series)
    report["symbols_with_data"] = n_symbols
    span = sorted(r["day"] for rows in series.values() for r in rows)
    report["cohort_span"] = {"first": span[0] if span else None,
                             "last": span[-1] if span else None}
    if n_symbols < MIN_ASSETS_PER_DAY:
        report["status"] = (f"SKIPPED: only {n_symbols} symbols have usable records "
                            f"(<{MIN_ASSETS_PER_DAY}); cross-sectional arm not run")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": report["status"], "out": str(args.out)}))
        return 0

    report["status"] = "ran"
    report["arms"] = {
        "funding_pct_carry": cross_section_arm(series, "funding_pct", +1),
        "dfh20_momentum": cross_section_arm(series, "dfh20", -1),
    }
    report["controls"] = {
        "funding_pct": controls(series, "funding_pct"),
        "dfh20": controls(series, "dfh20"),
    }
    report["honest_prior"] = (
        "XS funding carry is expected NULL: the post-2020 carry literature finds "
        "the funding-prediction effect dead on majors; a nonzero result here "
        "should be treated as surprising, not as confirmation.")
    report["power_note"] = (
        "HONEST POWER: ~10 assets per day and a 2-vs-2 edge contrast gives only "
        "~4-5 effective degrees of freedom per day — each daily spread is dominated "
        "by which 4 assets land at the rank edges, and one asset's jump swings the "
        "estimate. 5d forward labels on adjacent decision days overlap ~80%, so the "
        "day-series t-stat and pooled Welch overstate effective n (reported as "
        "nominal). Early-2023 funding_pct rows warm up over ~20 records. No "
        "multiple-testing correction across the two arms. Gross close moves only — "
        "no costs, funding cashflows or tradability. This is measurement, not a "
        "signal claim.")
    report["honesty"] = {
        "not_a_return": "forward returns are gross close-price moves; fees, spread, "
                        "slippage, funding cashflows and leverage are excluded",
        "not_live": "no orders, no account, no broker, no trading API was used",
        "no_profitability_claim": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    brief = {"status": "ran", "symbols": n_symbols, "out": str(args.out),
             "protocol": str(PROTOCOL), "auth": str(AUTH)}
    for name, arm in report["arms"].items():
        brief[name] = {"days": arm["days_evaluated"],
                       "median_assets": arm["assets_per_day"]["median"],
                       "mean_daily_spread_bps": arm["mean_daily_spread_bps"],
                       "pooled_welch_t": arm["pooled_welch"].get("t"),
                       "folds_same_sign": arm["fold_sign_consistency"]}
    print(json.dumps(brief, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
