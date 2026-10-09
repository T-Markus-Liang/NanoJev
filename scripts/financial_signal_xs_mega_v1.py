#!/usr/bin/env python3
"""Cross-sectional signal arms on the 301-symbol MEGA cohort (T112).

Reads ``data/perp_pit_mega_v1/records.jsonl`` (``build_perp_pit_mega_v1.py``:
up to 301 USDT-M perps from the imported ``data/rc_futures_v1`` archive,
2021-01 -> 2025-12, close-as-mark basis, ~52 funding-covered symbols, includes
delisted/renamed early-stoppers) and re-runs the T104/T106-style XS arms at
proper depth — decile-vs-decile portfolios instead of the 2-vs-2 edges the
10-asset pilot forced:

  dfh20         XS DISTANCE-FROM-HIGH MOMENTUM. Per decision day with >=30
                ranked assets: rank by dfh20, long top-decile / short
                bottom-decile (equal weight, daily rebalance); spread = mean
                fwd(top-decile) - mean fwd(bottom-decile) at 1d and 5d.
  mom20         XS 20d MOMENTUM. Same construction, rank by mom20.
  funding_pct   XS CARRY (funded subset only, ~52 symbols). Per-asset
                mid-rank pct of today's last_funding_rate vs its trailing-180
                daily values (MIN_WINDOW=20); long BOTTOM-decile / short
                TOP-decile (high funding should underperform).

Per arm / horizon: mean daily spread with t + normal-approx p, pooled Welch on
the leg asset-days, yearly folds 2021-2025, a daily-rebalanced turnover + net
sim (5bps per unit one-sided leg turnover, computed turnover), BTC-20d-return
gate split (gate-on = BTC ret20 > 0), a within-day rank-shuffle placebo x3
seeds, and BH-FDR over the small 3-arm x 2-horizon headline family.

DELISTING HONESTY: the archive includes contracts whose series stops before
the 2025-12 tail (delisted or renamed). The headline arm is re-run excluding
(a) ALL early-stopped symbols and (b) only post-2024-stopped symbols — the
spread must survive both to claim it is not a delisting artifact. This
cohort is LESS survivorship-biased than a listed-only universe, not unbiased:
renames can double-count an economic asset.

Honest dof note: decile edges give ~10-29 names per side at full universe
(~20-60 effective XS dof/day vs the pilot's ~4-5) — a real cross-section.
5d overlapping labels still autocorrelate adjacent days; all t/p nominal.

Artifacts follow the established convention: a frozen protocol
(``research/financial_signal_xs_mega_protocol_v1.json``) plus an owner
self-authorization pinning it by sha256
(``results/financial_signal_xs_mega_authorization_v1.json``) are written on
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
COHORT = ROOT / "data/perp_pit_mega_v1/records.jsonl"
OUT = ROOT / "results/financial_signal_xs_mega_v1.json"
PROTOCOL = ROOT / "research/financial_signal_xs_mega_protocol_v1.json"
AUTH = ROOT / "results/financial_signal_xs_mega_authorization_v1.json"

MIN_ASSETS_PER_DAY = 30  # task gate: decile contrasts need a real cross-section
DECILE = 10              # edge k = max(1, n//10) ranked assets per side
HORIZONS = (1, 5)
FUND_LOOKBACK = 180      # trailing-180 mid-rank window for funding_pct
MIN_WINDOW = 20          # floor for a usable trailing pct (repo convention)
BTC_GATE_LOOKBACK = 20
COST_BPS_PER_LEG = 5.0   # per unit of one-sided leg notional traded
PLACEBO_SEEDS = (11, 22, 33)
DAYS_PER_YEAR = 365      # crypto trades daily
POST2024_STOP_CUTOFF = "2024-01-01"  # last_bar >= this but < tail = post-2024-stopped


def _r(x, nd=3):
    return round(x, nd) if isinstance(x, float) else x


def norm_p(t):
    """Two-sided normal-approx p from a t/z statistic (repo convention)."""
    if t is None:
        return None
    return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))


def t_stat(xs):
    """Mean/sd/t (+ normal-approx p) of a daily series; obs treated as
    independent — nominal only (5d labels overlap ~5x)."""
    n = len(xs)
    if n < 5:
        return {"n": n, "mean": (sum(xs) / n) if n else None,
                "sd": None, "t": None, "p": None}
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(var)
    t = (mean / (sd / math.sqrt(n))) if sd > 0 else None
    return {"n": n, "mean": mean, "sd": sd, "t": t, "p": norm_p(t)}


def welch(xs, ys):
    """Welch two-sample t, Welch-Satterthwaite df, normal-approx p."""
    nx, ny = len(xs), len(ys)
    if nx < 2 or ny < 2:
        return {"n_long": nx, "n_short": ny, "t": None, "df": None, "p": None}
    mx, my = sum(xs) / nx, sum(ys) / ny
    vx = sum((x - mx) ** 2 for x in xs) / (nx - 1)
    vy = sum((y - my) ** 2 for y in ys) / (ny - 1)
    denom = vx / nx + vy / ny
    if denom <= 0:
        return {"n_long": nx, "n_short": ny, "mean_long": mx,
                "mean_short": my, "t": None, "df": None, "p": None}
    t = (mx - my) / math.sqrt(denom)
    df = denom ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    return {"n_long": nx, "n_short": ny, "mean_long": mx,
            "mean_short": my, "t": t, "df": df, "p": norm_p(t)}


def mid_rank_pct(window, x):
    """Mid-rank percentile of x vs trailing window (ties count half)."""
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


def max_drawdown(cumulative):
    peak, mdd = 0.0, 0.0
    for v in cumulative:
        peak = max(peak, v)
        mdd = max(mdd, peak - v)
    return mdd


def load_cohort(path):
    """asset_id -> sorted rows {day, close, dfh20, mom20, funding, fwd_*};
    asset_id -> {listed, last_bar_date}; then add the funding_pct score."""
    series, meta = defaultdict(list), {}
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
                "fwd_1d_bps": record["label"]["forward_return_bps"],
                "fwd_5d_bps": record["label"]["forward_return_5d_bps"],
            })
    for rows in series.values():
        rows.sort(key=lambda r: r["day"])
    # funding_pct: trailing-180 mid-rank pct of last_funding_rate (funded only).
    funded = []
    for asset, rows in series.items():
        fund = [r["funding"] for r in rows]
        any_funding = False
        for i, r in enumerate(rows):
            fwin = [x for x in fund[max(0, i - FUND_LOOKBACK):i]
                    if x is not None]
            r["funding_pct"] = (mid_rank_pct(fwin, r["funding"])
                                if r["funding"] is not None
                                and len(fwin) >= MIN_WINDOW else None)
            any_funding = any_funding or r["funding"] is not None
        if any_funding:
            funded.append(asset)
    # BTC 20d return gate from the BTC rows in the same cohort.
    btc_ret20 = {}
    btc_rows = series.get("BTCUSDT-PERP", [])
    closes = [r["close"] for r in btc_rows]
    for i, r in enumerate(btc_rows):
        if i >= BTC_GATE_LOOKBACK and closes[i - BTC_GATE_LOOKBACK] > 0:
            btc_ret20[r["day"]] = closes[i] / closes[i - BTC_GATE_LOOKBACK] - 1.0
    return series, meta, sorted(funded), btc_ret20


def xs_days(series, score_key, horizon, direction, universe=None):
    """Per decision day (>=30 ranked assets): decile-vs-decile next-h spread.

    direction "momentum" -> long top-decile / short bottom-decile;
    "carry" -> long bottom-decile / short top-decile.
    Returns day rows plus pooled leg-return lists."""
    fwd_key = f"fwd_{horizon}d_bps"
    by_day = defaultdict(list)
    for asset, rows in series.items():
        if universe is not None and asset not in universe:
            continue
        for r in rows:
            s, f = r.get(score_key), r.get(fwd_key)
            if s is not None and f is not None:
                by_day[r["day"]].append((asset, s, f))
    days, pooled_long, pooled_short = [], [], []
    for day in sorted(by_day):
        ordered = sorted(by_day[day], key=lambda x: x[1])
        n = len(ordered)
        if n < MIN_ASSETS_PER_DAY:
            continue
        k = max(1, n // DECILE)
        bot, top = ordered[:k], ordered[-k:]
        if bot[-1][1] == top[0][1]:
            continue  # degenerate day: no dispersion at the ranked edges
        longs, shorts = (top, bot) if direction == "momentum" else (bot, top)
        long_fwd = sum(x[2] for x in longs) / len(longs)
        short_fwd = sum(x[2] for x in shorts) / len(shorts)
        days.append({"day": day, "n_assets": n, "edge": k,
                     "spread_bps": long_fwd - short_fwd,
                     "long_assets": sorted(x[0] for x in longs),
                     "short_assets": sorted(x[0] for x in shorts)})
        pooled_long.extend(x[2] for x in longs)
        pooled_short.extend(x[2] for x in shorts)
    return days, pooled_long, pooled_short


def net_sim(days):
    """Daily-rebalanced decile book: cost = 5bps per unit one-sided leg
    turnover; turnover = (slots entered + slots exited) / (2 * book slots),
    slots = (side, asset) pairs. Day one establishes the full book (frac=1)."""
    prev, rows, turnovers = None, [], []
    for d in days:
        slots = {("L", a) for a in d["long_assets"]} | \
                {("S", a) for a in d["short_assets"]}
        if prev is None:
            frac = 1.0
        else:
            frac = ((len(slots - prev) + len(prev - slots))
                    / (2 * max(len(slots), len(prev)))) if slots or prev else 0.0
        turnovers.append(frac)
        cost = COST_BPS_PER_LEG * frac
        rows.append({"day": d["day"], "gross_bps": d["spread_bps"],
                     "turnover_frac": frac, "cost_bps": cost,
                     "net_bps": d["spread_bps"] - cost})
        prev = slots
    gross = [r["gross_bps"] for r in rows]
    net = [r["net_bps"] for r in rows]
    g, n = t_stat(gross), t_stat(net)
    cum_g = cum_n = 0.0
    cg, cn = [], []
    for r in rows:
        cum_g += r["gross_bps"]
        cum_n += r["net_bps"]
        cg.append(cum_g)
        cn.append(cum_n)
    sharpe = lambda s: (s["mean"] / s["sd"] * math.sqrt(DAYS_PER_YEAR)
                        if s["sd"] else None)
    return {"days": len(rows),
            "mean_daily_turnover_frac_of_book":
                _r(sum(turnovers) / len(turnovers) if turnovers else None),
            "mean_daily_cost_bps":
                _r(sum(r["cost_bps"] for r in rows) / len(rows)
                   if rows else None),
            "gross": {"mean_daily_bps": _r(g["mean"]), "t": _r(g["t"]),
                      "cumulative_bps": _r(cg[-1] if cg else None),
                      "max_drawdown_bps": _r(max_drawdown(cg)),
                      "sharpe_annualized": _r(sharpe(g))},
            "net": {"mean_daily_bps": _r(n["mean"]), "t": _r(n["t"]),
                    "cumulative_bps": _r(cn[-1] if cn else None),
                    "max_drawdown_bps": _r(max_drawdown(cn)),
                    "sharpe_annualized": _r(sharpe(n))}}


def summarize_arm(days, pooled_long, pooled_short, btc_ret20):
    """Full per-arm stat block for one (score, direction, horizon) run."""
    spreads = [d["spread_bps"] for d in days]
    daily = t_stat(spreads)
    folds = defaultdict(list)
    for d in days:
        folds[d["day"][:4]].append(d["spread_bps"])
    fold_means = {y: sum(v) / len(v) for y, v in sorted(folds.items())}
    same_sign = sum(1 for m in fold_means.values()
                    if daily["mean"] is not None and m * daily["mean"] > 0)
    n_assets = [d["n_assets"] for d in days]
    edges = [d["edge"] for d in days]
    gate = {"gate_on_btc_ret20_pos": [], "gate_off_btc_ret20_nonpos": [],
            "warmup_no_gate_yet": []}
    for d in days:
        tr = btc_ret20.get(d["day"])
        key = ("warmup_no_gate_yet" if tr is None
               else "gate_on_btc_ret20_pos" if tr > 0
               else "gate_off_btc_ret20_nonpos")
        gate[key].append(d["spread_bps"])
    gate_out = {}
    for k, v in gate.items():
        s = t_stat(v)
        gate_out[k] = {"days": len(v), "mean_spread_bps": _r(s["mean"]),
                       "t": _r(s["t"])}
    return {
        "days_evaluated": len(days),
        "assets_per_day": {"min": min(n_assets) if n_assets else None,
                           "median": (sorted(n_assets)[len(n_assets) // 2]
                                      if n_assets else None),
                           "max": max(n_assets) if n_assets else None},
        "edge_names_per_side": {"min": min(edges) if edges else None,
                                "median": (sorted(edges)[len(edges) // 2]
                                           if edges else None),
                                "max": max(edges) if edges else None},
        "daily_spread_bps": {k: _r(v, 6) for k, v in daily.items()},
        "pooled_welch": {k: _r(v, 6) for k, v in
                         welch(pooled_long, pooled_short).items()},
        "yearly_folds": {y: {"days": len(folds[y]),
                             "mean_spread_bps": _r(m)}
                         for y, m in fold_means.items()},
        "yearly_folds_same_sign": f"{same_sign}/{len(fold_means)}",
        "btc_gate_split": gate_out,
        "net_sim_1d_basis_note": ("net sim below uses THIS horizon's spread "
                                  "minus per-day turnover cost; for 5d the "
                                  "daily-rebalance churn is still charged "
                                  "daily (conservative: positions refresh "
                                  "every decision day)"),
        "net_sim": net_sim(days),
    }


def placebo(series, score_key, horizon, direction, universe=None):
    """Shuffle scores WITHIN each decision day x3 seeds; spread must collapse."""
    fwd_key = f"fwd_{horizon}d_bps"
    by_day = defaultdict(list)
    for asset, rows in series.items():
        if universe is not None and asset not in universe:
            continue
        for r in rows:
            s, f = r.get(score_key), r.get(fwd_key)
            if s is not None and f is not None:
                by_day[r["day"]].append((asset, s, f))
    out = {}
    for seed in PLACEBO_SEEDS:
        rng = random.Random(seed)
        day_spreads = []
        for day in sorted(by_day):
            xs = by_day[day]
            if len(xs) < MIN_ASSETS_PER_DAY:
                continue
            scores = [x[1] for x in xs]
            rng.shuffle(scores)
            ordered = sorted([(a, s, f) for (a, _, f), s in zip(xs, scores)],
                             key=lambda x: x[1])
            k = max(1, len(ordered) // DECILE)
            bot, top = ordered[:k], ordered[-k:]
            if bot[-1][1] == top[0][1]:
                continue
            longs, shorts = (top, bot) if direction == "momentum" else (bot, top)
            day_spreads.append(sum(x[2] for x in longs) / len(longs)
                               - sum(x[2] for x in shorts) / len(shorts))
        s = t_stat(day_spreads)
        out[f"seed_{seed}"] = {"days": len(day_spreads),
                               "mean_spread_bps": _r(s["mean"]),
                               "t": _r(s["t"])}
    return out


def arm_verdict(name, res, fdr_map, placebo_res, controls=None):
    """Numbers-plus-verdict per arm. Hard gates: FDR-surviving 5d spread in the
    hypothesized direction, net-of-cost positive 1d book, >=4/5 yearly folds
    same sign, all placebo seeds below half the real spread, and (where
    computed) the survivorship controls keep the sign. Regime asymmetry
    demotes to CONDITIONAL rather than failing."""
    h5 = res["h5"]
    mean5 = h5["daily_spread_bps"]["mean"]
    gates = {}
    gates["positive_5d_spread"] = mean5 is not None and mean5 > 0
    gates["fdr_survives_5d"] = bool(fdr_map.get(f"{name}_5d"))
    net1 = res["h1"]["net_sim"]["net"]
    gates["net_positive_after_costs_1d"] = (
        net1["mean_daily_bps"] is not None and net1["mean_daily_bps"] > 0)
    n_folds = len(h5["yearly_folds"])
    same = int(h5["yearly_folds_same_sign"].split("/")[0])
    gates["yearly_folds_mostly_same_sign"] = n_folds == 0 or same >= max(1, n_folds - 1)
    seed_means = [v["mean_spread_bps"] for v in placebo_res.values()
                  if v["mean_spread_bps"] is not None]
    gates["placebo_collapses"] = bool(seed_means) and all(
        m < 0.5 * mean5 for m in seed_means) if mean5 is not None else False
    if controls:
        controls_same = all(
            c["h5"]["mean_spread_bps"] is not None
            and (c["h5"]["mean_spread_bps"] > 0) == (mean5 is not None
                                                   and mean5 > 0)
            for c in controls.values())
        gates["survives_survivorship_controls"] = controls_same
    up = h5["btc_gate_split"]["gate_on_btc_ret20_pos"]["mean_spread_bps"]
    dn = h5["btc_gate_split"]["gate_off_btc_ret20_nonpos"]["mean_spread_bps"]
    both_regimes = up is not None and up > 0 and dn is not None and dn > 0
    failed = [k for k, v in gates.items() if not v]
    if mean5 is not None and mean5 <= 0:
        verdict = ("NULL/INVERTED at mega scale: 5d decile spread "
                   f"{_r(mean5)}bps is not positive in the hypothesized "
                   "direction")
    elif failed:
        verdict = "NOT YET: hard gates failed -> " + ", ".join(failed)
    elif not both_regimes:
        verdict = (f"CONDITIONAL: clears hard gates but is BTC-regime gated "
                   f"(gate-on {_r(up)}bps vs gate-off {_r(dn)}bps); "
                   "candidate only with the regime state carried")
    else:
        verdict = "CONFIRMED AT MEGA SCALE: survives costs, folds, placebo, gate split"
    return {"hard_gates": gates,
            "hard_gates_passed": f"{sum(1 for v in gates.values() if v)}/{len(gates)}",
            "positive_in_both_btc_regimes": both_regimes,
            "verdict": verdict}


def write_protocol_and_auth():
    protocol = {
        "schema_version": "nanojev-financial-signal-xs-mega-protocol-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "T112: re-run the cross-sectional arms at proper depth on "
                   "the 301-symbol mega cohort. Arms: (1) dfh20 decile rank, "
                   "long top-decile / short bottom-decile, next-1d/5d close "
                   "spreads; (2) mom20 decile rank, same; (3) funding_pct "
                   "trailing-180 mid-rank carry on the ~52 funding-covered "
                   "subset, long bottom-decile / short top-decile. >=30 "
                   "ranked assets/day, equal weight, daily rebalance; BTC "
                   "20d-return gate split; turnover + 5bps/leg net sim; "
                   "yearly folds 2021-2025; within-day rank-shuffle placebo "
                   "x3; BH-FDR over the 6 headline cells; delisting controls "
                   "(exclude all early-stopped and post-2024-stopped "
                   "symbols). Measurement only — no fitting, no trading, no "
                   "profitability claims.",
        "cohort": {
            "path": "data/perp_pit_mega_v1/records.jsonl",
            "builder": "scripts/build_perp_pit_mega_v1.py",
            "assets": "up to 301 USDT-M perps (>=200 daily bars), imported "
                      "data/rc_futures_v1 archive copy; includes "
                      "delisted/renamed early-stoppers (listed/last_bar_date "
                      "on every record)",
            "price_basis": "csv close as the mark proxy for every symbol",
            "targets": "label.forward_return_bps (1d) and "
                       "forward_return_5d_bps (5d), gross close-to-close, "
                       "contiguous daily bars only",
        },
        "definitions": {
            "dfh20": "precomputed feature: close / max(close[i-20:i]) - 1 "
                     "(strictly prior 20 contiguous bars)",
            "mom20": "precomputed feature: close / close[i-20] - 1",
            "funding_pct": "per-asset mid-rank pct of last_funding_rate vs "
                           "its trailing-180 daily values (MIN_WINDOW=20); "
                           "only the ~52 funding-covered symbols can rank",
            "day_gate": ">=30 assets with score AND forward return that day",
            "decile_edge": "k = max(1, n_ranked // 10) names per side",
            "spread": "mean fwd(long decile) - mean fwd(short decile), bps; "
                      "momentum arms long top-decile, carry longs "
                      "bottom-decile",
            "turnover": "(slots entered + slots exited)/(2*book slots) vs "
                        "previous day; cost = 5bps * fraction",
            "btc_gate": "BTCUSDT close/close[t-20]-1 sign on the decision "
                        "day (from the BTC series in the same cohort)",
            "placebo": "score values shuffled among the same day's assets, "
                       "3 seeds",
            "survivorship_controls": "headline arm re-run excluding (a) all "
                                     "early-stopped symbols, (b) symbols "
                                     "stopped from 2024 onward",
        },
        "statistics": {
            "daily": "mean daily spread + nominal t + normal-approx p",
            "pooled": "Welch two-sample t (all long-decile vs all "
                      "short-decile asset-days) + Welch-Satterthwaite df",
            "folds": "per-calendar-year fold mean spreads, sign consistency",
            "fdr": "Benjamini-Hochberg alpha=0.05 over the 6 headline "
                   "arm-x-horizon cells (small family)",
            "caveats": [
                "h-day overlapping labels autocorrelate ~1-1/h of adjacent "
                "days; all t/p nominal",
                "~10-29 names/side at full universe: ~20-60 effective XS "
                "dof/day (real cross-section vs pilot's ~4-5), but pooled "
                "Welch still double-counts correlated asset-days",
                "universe composition changes as symbols list/delist; "
                "decile size floats with n",
                "delisting is a proxy: early stop = delisted OR renamed; "
                "renames can double-count an economic asset",
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
        "schema_version": "nanojev-financial-signal-xs-mega-authorization-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "protocol_sha256": sha,
        "protocol_path": "research/financial_signal_xs_mega_protocol_v1.json",
        "decision": "approved_for_measurement",
        "measurement_authorized": True,
        "fit_authorized": False,
        "independent_reviewer": {
            "id": "project-owner",
            "independence": "owner_self_authorization_not_independent_review",
            "note": "Owner directed T112: build the 301-symbol mega daily "
                    "PIT cohort from the imported rc_futures_v1 archive and "
                    "re-run the XS arms at decile depth (delegated task).",
        },
        "scope": {
            "permitted": "PIT-safe descriptive measurement on "
                         "data/perp_pit_mega_v1/records.jsonl: decile-vs-"
                         "decile XS spreads by dfh20 / mom20 / funding_pct "
                         "ranks, 1d+5d horizons, BTC-gate splits, turnover+"
                         "net sims, yearly folds, placebo shuffles, BH-FDR "
                         "and delisting controls per the pinned protocol.",
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
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    args = parser.parse_args()

    protocol_sha = write_protocol_and_auth()
    report = {"schema_version": "nanojev-financial-signal-xs-mega-v1",
              "task": "T112 mega-cohort cross-sectional arms at decile depth",
              "cohort": str(args.cohort),
              "protocol_path": "research/financial_signal_xs_mega_protocol_v1.json",
              "protocol_sha256": protocol_sha,
              "authorization_path":
                  "results/financial_signal_xs_mega_authorization_v1.json",
              "horizons_days": list(HORIZONS),
              "decile": DECILE,
              "min_assets_per_day": MIN_ASSETS_PER_DAY,
              "funding_lookback": FUND_LOOKBACK,
              "funding_min_window": MIN_WINDOW,
              "cost_bps_per_leg_turnover": COST_BPS_PER_LEG,
              "placebo_seeds": list(PLACEBO_SEEDS)}

    if not args.cohort.exists():
        report["status"] = ("SKIPPED: cohort records.jsonl not found (build "
                            "incomplete); nothing was fabricated")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": report["status"], "out": str(args.out)}))
        return 0

    series, meta, funded_assets, btc_ret20 = load_cohort(args.cohort)
    span = sorted(r["day"] for rows in series.values() for r in rows)
    early_stopped = sorted(a for a, m in meta.items() if m["listed"] is False)
    post2024_stopped = sorted(a for a in early_stopped
                              if (meta[a]["last_bar_date"] or "")
                              >= POST2024_STOP_CUTOFF)
    report["cohort_description"] = {
        "symbols": len(series),
        "symbols_with_funding": len(funded_assets),
        "span": {"first": span[0] if span else None,
                 "last": span[-1] if span else None},
        "early_stopped_symbols": {
            "count": len(early_stopped),
            "post_2024_stopped_count": len(post2024_stopped),
            "assets": {a: meta[a]["last_bar_date"] for a in early_stopped},
            "note": "early stop = delisted OR renamed proxy (last_bar before "
                    "the archive tail); LESS survivorship bias than a "
                    "listed-only universe, not zero — rename pairs can "
                    "double-count one asset"},
    }
    report["status"] = "ran"

    arm_specs = {
        "dfh20_momentum": ("dfh20", "momentum"),
        "mom20_momentum": ("mom20", "momentum"),
        "funding_pct_carry": ("funding_pct", "carry"),
    }
    arms, fdr_cells = {}, []
    for name, (score_key, direction) in arm_specs.items():
        res = {}
        for h in HORIZONS:
            days, pl_long, pl_short = xs_days(series, score_key, h, direction)
            res[f"h{h}"] = summarize_arm(days, pl_long, pl_short, btc_ret20)
            fdr_cells.append((f"{name}_{h}d",
                              res[f"h{h}"]["daily_spread_bps"]["p"]))
        arms[name] = res
    fdr_map, fdr_table = bh_fdr(fdr_cells)
    report["fdr_bh_0.05"] = {"family": "3 arms x {1d,5d} headline daily-spread "
                                     "p-values (small family)",
                             "table": fdr_table}

    # Delisting honesty: re-run the headline arm on restricted universes.
    survivorship = {}
    headline_spec = arm_specs["dfh20_momentum"]
    for label, drop in (
            ("excl_all_early_stopped", set(early_stopped)),
            ("excl_post2024_stopped", set(post2024_stopped))):
        universe = set(series) - drop
        sub = {}
        for h in HORIZONS:
            days, pl, ps = xs_days(series, headline_spec[0], h,
                                   headline_spec[1], universe=universe)
            s = t_stat([d["spread_bps"] for d in days])
            sub[f"h{h}"] = {"days": len(days),
                            "mean_spread_bps": _r(s["mean"]),
                            "t": _r(s["t"]), "p": _r(s["p"], 6),
                            "excluded_n": len(drop)}
        survivorship[label] = sub
    report["delisting_honesty"] = {
        "method": "dfh20_momentum headline re-run on restricted universes; "
                  "spread must keep sign and rough magnitude",
        "controls": survivorship,
    }

    placebo_res = {name: placebo(series, sk, 5, dr)
                   for name, (sk, dr) in arm_specs.items()}

    report["arms"] = arms
    report["placebo_shuffle_5d"] = placebo_res
    report["verdicts"] = {}
    for name in arm_specs:
        controls = survivorship if name == "dfh20_momentum" else None
        report["verdicts"][name] = arm_verdict(
            name, arms[name], fdr_map, placebo_res[name], controls)

    report["power_note"] = (
        "HONEST POWER: decile edges give ~10-29 names/side at full universe "
        "(~20-60 effective XS dof/day) — a real cross-section vs the "
        "10-asset pilot's ~4-5. Still: 5d labels overlap ~80% on adjacent "
        "days so day-series t/p overstate effective n (nominal); pooled "
        "Welch treats correlated asset-days as independent; universe "
        "composition and decile size float as symbols list/delist; the "
        "funding arm ranks only the ~52 funded symbols (edge ~5/side).")
    report["honesty"] = {
        "not_a_return": "spreads are gross close-price moves; the net sim "
                        "subtracts a stylized 5bps/leg turnover cost only — "
                        "funding cashflows, borrow, slippage and leverage "
                        "are excluded",
        "not_an_asof_vintage": "second-hand archive copy (rc_futures_v1), "
                               "not a verified venue pull or as-of vintage",
        "not_live": "no orders, no account, no broker, no trading API used",
        "no_profitability_claim": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    brief = {"status": "ran", "out": str(args.out),
             "symbols": report["cohort_description"]["symbols"],
             "early_stopped":
                 report["cohort_description"]["early_stopped_symbols"]["count"]}
    for name in arm_specs:
        brief[name] = {
            "5d_mean_bps": arms[name]["h5"]["daily_spread_bps"]["mean"],
            "5d_t": arms[name]["h5"]["daily_spread_bps"]["t"],
            "1d_net_bps": arms[name]["h1"]["net_sim"]["net"]["mean_daily_bps"],
            "folds": arms[name]["h5"]["yearly_folds_same_sign"],
            "verdict": report["verdicts"][name]["verdict"]}
    print(json.dumps(brief, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
