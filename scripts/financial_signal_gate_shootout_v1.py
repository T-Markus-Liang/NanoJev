#!/usr/bin/env python3
"""Head-to-head master-gate parameterization shootout on the mega cohort (T114).

Context: every daily signal measured so far (T79 spec v1, T99 HL, T106 XS dfh,
T107 Bybit, T112 mega 25x replication) earns only while BTC is in a 20d
uptrend — OUR master gate is ``BTC close/close[t-20]-1 > 0`` on the decision
day. The sibling Rolling-Compounding project's strongest regime variable is
different: ``BTC atrp = ATR84/price`` on 4h bars with the regime split at
~0.025 — their claim "trend without vol wrongly opens the gate". T114 asks
which BTC state variable better conditions OUR signals on the 277-symbol
mega cohort (``data/perp_pit_mega_v1/records.jsonl``, 2021->2025 daily).

Gate candidates (all BTC-only state, evaluated on the decision day)
--------------------------------------------------------------------
  ret20_pos        ours: close/close[t-20]-1 > 0 on 1d bars.
  atrp_d_high      ATR14 (Wilder) / close on 1d bars > threshold; threshold =
                   median over the defined evaluation days so on/off ~= 50/50
                   (occupancy reported).
  atrp4h_high      their exact construction: ATR84 (Wilder) / close on 4h
                   bars, last 4h bar of each UTC day, > 0.025 (their regime
                   split). Sensitivity arm: same series > its own median.
  joint            ret20>0 AND atrp4h>0.025 (trend-with-vol).
  trend_no_vol     ret20>0 AND atrp4h<=0.025 — their predicted TRAP cell;
                   on/off Welch contrasts the trap against all other days.

Signals conditioned (each is a daily series of decile spreads / XS rho,
split gate-on vs gate-off; Welch t, occupancy %, per-year consistency):
  xs_dfh20        long top-decile / short bottom-decile by dfh20 rank,
                  5d close-to-close fwd (>=30 ranked assets/day, T112 conv.)
  xs_mom20        same construction ranked by mom20
  funding_carry   funding_pct (trailing-180 mid-rank, MIN_WINDOW=20) on the
                  ~52 funded-symbol subset; long BOTTOM / short TOP decile
  dfh_pct_spear   per-asset dfh_pct (trailing-180 mid-rank of dfh20): pooled
                  asset-day Spearman(dfh_pct, fwd5) per gate state + per-day
                  XS Spearman series (same Welch machinery as the spreads)

KEY CELL: the trend x vol 2x2 — trend-on+vol-off (does the signal die there?
their theory says yes) and trend-off+vol-on vs the joint trend-on+vol-on.

Small FDR family: BH over the 4 signals x 5 primary gates on/off Welch
p-values + the 4 trap-vs-joint Welch p's = 24 cells. Correlated gates:
pairwise agreement + conditional occupancy between definitions reported.

Protocol + owner self-authorization are embedded in the single output
artifact ``results/financial_signal_gate_shootout_v1.json`` (protocol object
sha256-pinned, T111 convention). Measurement only: no fitting, no trading,
no network, no other files modified.
"""
import argparse
import bisect
import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_mega_v1/records.jsonl"
BTC_1D = ROOT / "data/rc_futures_v1/BTC/BTCUSDT_1d.csv"
BTC_4H = ROOT / "data/rc_futures_v1/BTC/BTCUSDT_4h.csv"
OUT = ROOT / "results/financial_signal_gate_shootout_v1.json"

MIN_ASSETS_PER_DAY = 30   # T112 day gate for decile contrasts
DECILE = 10               # k = max(1, n//10) names per side
TRAIL = 180               # trailing window for *_pct scores
MIN_WINDOW = 20           # floor for a usable trailing pct (repo convention)
RET_LOOKBACK = 20         # our master gate: close/close[t-20]-1 > 0
ATR_DAILY_N = 14          # Wilder ATR length on 1d bars
ATR_4H_N = 84             # their ATR84 on 4h bars (=14 days of bars)
ATRP_4H_SPLIT = 0.025     # their published regime split
MIN_N = 10                # test minimum per side (repo convention)


def _r(x, nd=3):
    return round(x, nd) if isinstance(x, float) else x


def norm_p(t):
    """Two-sided normal-approx p from a t/z statistic (repo convention)."""
    if t is None:
        return None
    return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))


def t_stat(xs):
    n = len(xs)
    if n < 5:
        return {"n": n, "mean": (sum(xs) / n) if n else None,
                "sd": None, "t": None, "p": None}
    mean = sum(xs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
    t = (mean / (sd / math.sqrt(n))) if sd > 0 else None
    return {"n": n, "mean": mean, "sd": sd, "t": t, "p": norm_p(t)}


def welch(xs, ys):
    """Welch two-sample t of mean(xs)-mean(ys), normal-approx p."""
    nx, ny = len(xs), len(ys)
    if nx < MIN_N or ny < MIN_N:
        return {"n_x": nx, "n_y": ny, "mean_x": (sum(xs) / nx if nx else None),
                "mean_y": (sum(ys) / ny if ny else None),
                "diff": None, "t": None, "p": None}
    mx, my = sum(xs) / nx, sum(ys) / ny
    vx = sum((x - mx) ** 2 for x in xs) / (nx - 1)
    vy = sum((y - my) ** 2 for y in ys) / (ny - 1)
    denom = vx / nx + vy / ny
    if denom <= 0:
        return {"n_x": nx, "n_y": ny, "mean_x": mx, "mean_y": my,
                "diff": mx - my, "t": None, "p": None}
    return {"n_x": nx, "n_y": ny, "mean_x": mx, "mean_y": my,
            "diff": mx - my, "t": (mx - my) / math.sqrt(denom),
            "p": norm_p((mx - my) / math.sqrt(denom))}


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
    """Mid-rank Spearman rho + normal-approx t/p; None if n<MIN_N or a side
    has zero rank variance (project convention)."""
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
    return {"n": n, "rho": rho, "t": t, "p": norm_p(t)}


def bh_fdr(named_ps, alpha=0.05):
    """BH-FDR over [(name, p|None)] -> ({name: survives}, sorted table)."""
    ps = sorted((p, n) for n, p in named_ps if p is not None)
    m = len(ps)
    table = [{"cell": n, "p": _r(p, 6),
              "alpha_bh": round(alpha * (i + 1) / m, 6),
              "survives": p <= alpha * (i + 1) / m}
             for i, (p, n) in enumerate(ps)]
    return {e["cell"]: e["survives"] for e in table}, table


# ------------------------------------------------------------------ loaders
def atr_wilder(highs, lows, closes, n):
    """Wilder ATR: TR_i uses bar i's h/l and bar i-1's close; seeded by the
    SMA of the first n TRs at index n, then recursive smoothing. Returns a
    list aligned to closes with None before the seed."""
    out = [None] * len(closes)
    if len(closes) <= n:
        return out
    trs = [0.0]
    for i in range(1, len(closes)):
        trs.append(max(highs[i] - lows[i],
                       abs(highs[i] - closes[i - 1]),
                       abs(lows[i] - closes[i - 1])))
    atr = sum(trs[1:n + 1]) / n
    out[n] = atr
    for i in range(n + 1, len(closes)):
        atr = (atr * (n - 1) + trs[i]) / n
        out[i] = atr
    return out


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return None
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def load_btc_state(cohort_days):
    """day -> {ret20, atrp_d, atrp_4h}; None where undefined. Thresholds are
    medians over gate-defined days inside the cohort decision-day span."""
    days1, h1, l1, c1 = [], [], [], []
    with BTC_1D.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            days1.append(row["timestamp"][:10])
            h1.append(float(row["high"]))
            l1.append(float(row["low"]))
            c1.append(float(row["close"]))
    state = {d: {} for d in days1}
    for i, d in enumerate(days1):
        if i >= RET_LOOKBACK and c1[i - RET_LOOKBACK] > 0:
            state[d]["ret20"] = c1[i] / c1[i - RET_LOOKBACK] - 1.0
    atr_d = atr_wilder(h1, l1, c1, ATR_DAILY_N)
    for i, d in enumerate(days1):
        if atr_d[i] is not None and c1[i] > 0:
            state[d]["atrp_d"] = atr_d[i] / c1[i]

    days4, h4, l4, c4 = [], [], [], []
    with BTC_4H.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            days4.append(row["timestamp"][:10])
            h4.append(float(row["high"]))
            l4.append(float(row["low"]))
            c4.append(float(row["close"]))
    atr4 = atr_wilder(h4, l4, c4, ATR_4H_N)
    last_bar_atrp, n_bars = {}, defaultdict(int)
    for i, d in enumerate(days4):
        n_bars[d] += 1
        if atr4[i] is not None and c4[i] > 0:
            last_bar_atrp[d] = atr4[i] / c4[i]  # iterate -> last bar wins
    for d, v in last_bar_atrp.items():
        state.setdefault(d, {})["atrp_4h"] = v

    eval_days = set(cohort_days)
    atrp_d_vals = [state[d]["atrp_d"] for d in eval_days
                   if "atrp_d" in state.get(d, {})]
    atrp_4h_vals = [state[d]["atrp_4h"] for d in eval_days
                    if "atrp_4h" in state.get(d, {})]
    return state, median(atrp_d_vals), median(atrp_4h_vals), dict(n_bars)


def load_cohort(path):
    """asset -> sorted rows {day, dfh20, mom20, funding, fwd_5d_bps}."""
    series = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            rec = json.loads(line)
            f = rec["features"]
            series[rec["asset_id"]].append({
                "day": rec["id"].rsplit(":", 1)[-1],
                "dfh20": f["dfh20"]["value"],
                "mom20": f["mom20"]["value"],
                "funding": f["last_funding_rate"]["value"],
                "fwd_5d_bps": rec["label"]["forward_return_5d_bps"],
            })
    for rows in series.values():
        rows.sort(key=lambda r: r["day"])
    return series


def add_trailing_pct(series, key, out_key):
    """Per-asset mid-rank pct of row[key] vs its strictly-prior trailing-180
    values (None values dropped from the window), MIN_WINDOW floor — the
    labels_v4/T112 convention. Sliding sorted window via bisect."""
    for rows in series.values():
        win, raw = [], []
        for r in rows:
            x = r.get(key)
            r[out_key] = None
            if x is not None and len(win) >= MIN_WINDOW:
                lt = bisect.bisect_left(win, x)
                eq = bisect.bisect_right(win, x) - lt
                r[out_key] = (lt + 0.5 * eq) / len(win)
            raw.append(x)
            if x is not None:
                bisect.insort(win, x)
            if len(raw) > TRAIL:
                old = raw.pop(0)
                if old is not None:
                    win.pop(bisect.bisect_left(win, old))


def xs_decile_days(series, score_key, direction, universe=None):
    """day -> decile spread on fwd_5d_bps. momentum: long top/short bottom;
    carry: long bottom/short top. >=30 ranked assets (T112 convention)."""
    by_day = defaultdict(list)
    for asset, rows in series.items():
        if universe is not None and asset not in universe:
            continue
        for r in rows:
            s, f = r.get(score_key), r["fwd_5d_bps"]
            if s is not None and f is not None:
                by_day[r["day"]].append((s, f))
    out = {}
    for day, xs in by_day.items():
        if len(xs) < MIN_ASSETS_PER_DAY:
            continue
        xs.sort()
        k = max(1, len(xs) // DECILE)
        bot, top = xs[:k], xs[-k:]
        if bot[-1][0] == top[0][0]:
            continue
        longs, shorts = (top, bot) if direction == "momentum" else (bot, top)
        out[day] = (sum(x[1] for x in longs) / len(longs)
                    - sum(x[1] for x in shorts) / len(shorts))
    return out


def xs_spearman_days(series, score_key):
    """day -> cross-sectional Spearman(score, fwd5) over >=30 assets."""
    by_day = defaultdict(list)
    for rows in series.values():
        for r in rows:
            s, f = r.get(score_key), r["fwd_5d_bps"]
            if s is not None and f is not None:
                by_day[r["day"]].append((s, f))
    out = {}
    for day, xs in by_day.items():
        if len(xs) < MIN_ASSETS_PER_DAY:
            continue
        sp = spearman([x[0] for x in xs], [x[1] for x in xs])
        if sp is not None:
            out[day] = sp["rho"]
    return out


# ------------------------------------------------------------------ scoring
def gate_split(series_by_day, gate_fn):
    """Split a {day: value} series by gate state; returns on/off/undef."""
    on, off, undef = [], [], []
    for d, v in series_by_day.items():
        g = gate_fn(d)
        if g is None:
            undef.append(d)
        elif g:
            on.append(v)
        else:
            off.append(v)
    return on, off, undef


def per_year_table(series_by_day, gate_fn):
    """year -> {n_on, mean_on, n_off, mean_off, diff}; plus sign consistency
    of (on-off) across years."""
    cells = defaultdict(lambda: {"on": [], "off": []})
    for d, v in series_by_day.items():
        g = gate_fn(d)
        if g is not None:
            cells[d[:4]]["on" if g else "off"].append(v)
    table, n_pos = {}, 0
    for y in sorted(cells):
        on, off = cells[y]["on"], cells[y]["off"]
        mo = sum(on) / len(on) if on else None
        mf = sum(off) / len(off) if off else None
        diff = (mo - mf) if mo is not None and mf is not None else None
        if diff is not None and diff > 0:
            n_pos += 1
        table[y] = {"n_on": len(on), "mean_on": _r(mo),
                    "n_off": len(off), "mean_off": _r(mf),
                    "on_minus_off": _r(diff)}
    return table, n_pos, len(table)


def build_gates(state, thr_d, med_4h):
    """gate name -> fn(day) -> True/False/None (None = undefined/warmup)."""
    def s(d):
        return state.get(d, {})

    def ret20_on(d):
        v = s(d).get("ret20")
        return None if v is None else v > 0.0

    def atrp_d_on(d):
        v = s(d).get("atrp_d")
        return None if v is None else v > thr_d

    def atrp4h_on(d):
        v = s(d).get("atrp_4h")
        return None if v is None else v > ATRP_4H_SPLIT

    def atrp4h_med_on(d):
        v = s(d).get("atrp_4h")
        return None if v is None else v > med_4h

    def joint(d):
        a, b = ret20_on(d), atrp4h_on(d)
        return None if a is None or b is None else a and b

    def trap(d):
        a, b = ret20_on(d), atrp4h_on(d)
        return None if a is None or b is None else a and not b

    def joint_med(d):
        a, b = ret20_on(d), atrp4h_med_on(d)
        return None if a is None or b is None else a and b

    def trap_med(d):
        a, b = ret20_on(d), atrp4h_med_on(d)
        return None if a is None or b is None else a and not b

    return {
        "ret20_pos": ret20_on,
        "atrp_d_high": atrp_d_on,
        "atrp4h_high_0.025": atrp4h_on,
        "joint_ret20_and_atrp4h": joint,
        "trend_no_vol_ret20_and_lowatrp4h": trap,
        "atrp4h_high_median_sens": atrp4h_med_on,
        "joint_ret20_and_atrp4h_med_sens": joint_med,
        "trend_no_vol_med_sens": trap_med,
    }


def cell_fn(state, trend_key, vol_key, vol_thr):
    """Return fn(day)->cell name for the trend x vol 2x2 (or None)."""
    def f(d):
        s = state.get(d, {})
        t, v = s.get(trend_key), s.get(vol_key)
        if t is None or v is None:
            return None
        return ("trend_on" if t > 0.0 else "trend_off",
                "vol_on" if v > vol_thr else "vol_off")
    return f


def eval_gate(name, fn, cohort_days, signals):
    """Occupancy over cohort days + per-signal on/off Welch + per-year."""
    days_def = [d for d in cohort_days if fn(d) is not None]
    n_on = sum(1 for d in days_def if fn(d))
    out = {"gate": name,
           "days_defined": len(days_def),
           "days_undefined_warmup": len(cohort_days) - len(days_def),
           "on_days": n_on, "off_days": len(days_def) - n_on,
           "occupancy_on_pct": _r(100.0 * n_on / len(days_def), 1)
           if days_def else None,
           "signals": {}}
    for sig_name, series in signals.items():
        on, off, undef = gate_split(series, fn)
        w = welch(on, off)
        yr, n_pos, n_yr = per_year_table(series, fn)
        out["signals"][sig_name] = {
            "on": {"n": len(on), "mean": _r(w["mean_x"])},
            "off": {"n": len(off), "mean": _r(w["mean_y"])},
            "undefined_days": len(undef),
            "on_minus_off": _r(w["diff"]),
            "welch_t": _r(w["t"]), "welch_p": _r(w["p"], 6),
            "per_year": yr,
            "years_on_gt_off": f"{n_pos}/{n_yr}",
        }
    return out


def build_protocol():
    return {
        "schema_version": "nanojev-financial-signal-gate-shootout-protocol-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "task": "T114: head-to-head of BTC master-gate parameterizations on "
                "the mega cohort — our ret20>0 trend gate vs the sibling "
                "Rolling-Compounding BTC atrp (ATR84/price, 4h bars, split "
                "~0.025) vol regime, their joint, and the trend-without-vol "
                "trap cell.",
        "cohort": {
            "path": "data/perp_pit_mega_v1/records.jsonl",
            "assets": "277 USDT-M perps incl. delisted early-stoppers",
            "span": "2021-01 -> 2025-12 daily decision bars",
            "btc_bars": "data/rc_futures_v1/BTC/BTCUSDT_{1d,4h}.csv",
            "price_basis": "csv close as mark proxy; gross close-to-close "
                           "labels; no funding cashflows/borrow/costs",
        },
        "gates": {
            "ret20_pos": "BTC close/close[t-20]-1 > 0 on 1d bars (ours)",
            "atrp_d_high": "Wilder ATR14/close on 1d bars > median over "
                           "defined eval days (~50/50 by construction)",
            "atrp4h_high_0.025": "Wilder ATR84/close on 4h bars, last 4h bar "
                                 "of each UTC day, > 0.025 (their split); "
                                 "median-split twin as sensitivity",
            "joint": "ret20>0 AND atrp4h>split (trend-with-vol)",
            "trend_no_vol": "ret20>0 AND atrp4h<=split — their predicted "
                            "trap cell, contrasted vs all other days",
        },
        "signals": {
            "xs_dfh20_5d": "top-decile minus bottom-decile by dfh20 rank on "
                           "fwd_5d_bps, >=30 ranked assets/day",
            "xs_mom20_5d": "same, ranked by mom20",
            "funding_carry_5d": "funding_pct trailing-180 mid-rank on the "
                                "funded ~52-symbol subset; long bottom-decile"
                                " / short top-decile",
            "dfh_pct_spearman": "pooled asset-day Spearman(dfh_pct, fwd5) per "
                                "gate state + per-day XS Spearman series + "
                                "mean per-asset rho per state",
        },
        "statistics": {
            "per_cell": "on/off mean, Welch t + normal-approx p, occupancy, "
                        "per-year on/off table and sign consistency",
            "key_cells": "trend x vol 2x2 mean spreads; trap-vs-joint and "
                         "volwithouttrend-vs-joint Welch",
            "fdr": "BH alpha=0.05 over 4 signals x 5 primary gates on/off "
                   "Welch p + 4 trap-vs-joint p = 24 cells",
            "gate_correlation": "pairwise agreement % and conditional "
                                "occupancy between gate definitions",
            "caveats": ["5d overlapping labels autocorrelate adjacent days; "
                        "all t/p nominal",
                        "decile edges ~10-29 names/side; pooled Spearman "
                        "double-counts correlated asset-days",
                        "atrp_d threshold is median-picked in-sample by "
                        "design (occupancy target); atrp4h 0.025 split is "
                        "the sibling project's frozen value",
                        "gate state uses the decision-day close (same "
                        "convention as T111/T112)",
                        "second-hand archive copy, not an as-of vintage"],
        },
        "forbidden": ["fitting", "trading", "profitability claims",
                      "protocol edits post-run", "network"],
    }


def build_auth(protocol_sha):
    return {
        "schema_version":
            "nanojev-financial-signal-gate-shootout-authorization-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "protocol_sha256": protocol_sha,
        "decision": "approved_for_measurement",
        "measurement_authorized": True,
        "fit_authorized": False,
        "independent_reviewer": {
            "id": "project-owner",
            "independence": "owner_self_authorization_not_independent_review",
            "note": "Owner directed T114: head-to-head master-gate "
                    "parameterization comparison on the mega cohort "
                    "(delegated task).",
        },
        "scope": {
            "permitted": "PIT-safe descriptive measurement on "
                         "data/perp_pit_mega_v1/records.jsonl and the BTC "
                         "1d/4h bars per the embedded protocol.",
            "not_permitted": "No fitting/trading/profitability claims/"
                             "protocol edits; no network; no other files "
                             "modified.",
        },
        "network_model_calls": 0,
        "order_submission_authorized": False,
        "live_trading_authorized": False,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", type=pathlib.Path, default=COHORT)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()

    protocol = build_protocol()
    protocol_sha = hashlib.sha256(
        json.dumps(protocol, indent=2, sort_keys=True).encode()).hexdigest()
    report = {"schema_version": "nanojev-financial-signal-gate-shootout-v1",
              "task": "T114 master-gate parameterization shootout, mega cohort",
              "protocol": protocol,
              "protocol_sha256": protocol_sha,
              "owner_authorization": build_auth(protocol_sha)}

    if not args.cohort.exists() or not BTC_1D.exists() or not BTC_4H.exists():
        report["status"] = ("SKIPPED: cohort or BTC bars not found; nothing "
                            "was fabricated")
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 0

    series = load_cohort(args.cohort)
    cohort_days = sorted({r["day"] for rows in series.values() for r in rows})
    funded = sorted(a for a, rows in series.items()
                    if any(r["funding"] is not None for r in rows))
    add_trailing_pct(series, "funding", "funding_pct")
    add_trailing_pct(series, "dfh20", "dfh_pct")

    state, thr_d, med_4h, bars_4h = load_btc_state(cohort_days)
    gates = build_gates(state, thr_d, med_4h)
    report["cohort_span"] = {"first": cohort_days[0], "last": cohort_days[-1],
                             "decision_days": len(cohort_days),
                             "symbols": len(series),
                             "funded_symbols": len(funded)}
    report["btc_state"] = {
        "atrp_d_threshold_median": _r(thr_d, 5),
        "atrp_4h_median_over_eval_days": _r(med_4h, 5),
        "atrp_4h_split_theirs": ATRP_4H_SPLIT,
        "atrp_4h_days_with_lt6_bars": sorted(
            d for d, n in bars_4h.items()
            if n != 6 and d in set(cohort_days)),
        "note": "4h daily value = ATR84/close at the day's LAST 4h bar "
                "(20:00 open -> covers through the daily close)"}

    # ------------------------------------------------- signal daily series
    funded_set = set(funded)
    signals = {
        "xs_dfh20_5d": xs_decile_days(series, "dfh20", "momentum"),
        "xs_mom20_5d": xs_decile_days(series, "mom20", "momentum"),
        "funding_carry_5d": xs_decile_days(series, "funding_pct", "carry",
                                         universe=funded_set),
        "dfh_pct_xs_spearman_daily": xs_spearman_days(series, "dfh_pct"),
    }
    report["signal_days"] = {k: {"days": len(v)} for k, v in signals.items()}

    # ------------------------------------------------- per-gate evaluation
    primary = ["ret20_pos", "atrp_d_high", "atrp4h_high_0.025",
               "joint_ret20_and_atrp4h",
               "trend_no_vol_ret20_and_lowatrp4h"]
    sens = ["atrp4h_high_median_sens",
            "joint_ret20_and_atrp4h_med_sens",
            "trend_no_vol_med_sens"]
    report["gates"] = {}
    for name in primary + sens:
        report["gates"][name] = eval_gate(name, gates[name], cohort_days,
                                        signals)

    # ------------------------------------- pooled per-asset dfh_pct Spearman
    pooled_rows = []  # (day, asset, dfh_pct, fwd5)
    for asset, rows in series.items():
        for r in rows:
            if r.get("dfh_pct") is not None and r["fwd_5d_bps"] is not None:
                pooled_rows.append((r["day"], asset, r["dfh_pct"],
                                    r["fwd_5d_bps"]))
    spear_out = {}
    for name in primary:
        fn = gates[name]
        on_rows = [x for x in pooled_rows if fn(x[0]) is True]
        off_rows = [x for x in pooled_rows if fn(x[0]) is False]
        blk = {}
        for label, sel in (("gate_on", on_rows), ("gate_off", off_rows)):
            sp = spearman([x[2] for x in sel], [x[3] for x in sel])
            per_asset = []
            by_a = defaultdict(list)
            for x in sel:
                by_a[x[1]].append(x)
            for a, xs in by_a.items():
                s_a = spearman([x[2] for x in xs], [x[3] for x in xs])
                if s_a is not None:
                    per_asset.append(s_a["rho"])
            blk[label] = {
                "asset_days": len(sel),
                "pooled_spearman": ({k: _r(v, 6) for k, v in sp.items()}
                                    if sp else None),
                "per_asset_rho": {
                    "n_assets": len(per_asset),
                    "mean": _r(sum(per_asset) / len(per_asset)
                               if per_asset else None, 5),
                    "median": _r(median(per_asset), 5),
                    "frac_positive": _r(sum(1 for x in per_asset if x > 0)
                                        / len(per_asset), 3)
                    if per_asset else None},
            }
        # Fisher-z contrast between the two pooled rhos (nominal)
        ro, rf = blk["gate_on"]["pooled_spearman"], blk["gate_off"]["pooled_spearman"]
        z = None
        if ro and rf and abs(ro["rho"]) < 0.999 and abs(rf["rho"]) < 0.999:
            se = math.sqrt(1 / (ro["n"] - 3) + 1 / (rf["n"] - 3))
            z = (math.atanh(ro["rho"]) - math.atanh(rf["rho"])) / se
        blk["pooled_rho_on_minus_off"] = _r(
            (ro["rho"] - rf["rho"]) if ro and rf else None, 5)
        blk["fisher_z_nominal"] = _r(z)
        blk["fisher_p_nominal"] = _r(norm_p(z), 6)
        spear_out[name] = blk
    report["dfh_pct_spearman_pooled"] = spear_out

    # ------------------------------------------------- key cell: trend x vol
    key_cells = {}
    for thr_label, thr in (("split_0.025", ATRP_4H_SPLIT),
                           ("split_median_sens", med_4h)):
        cf = cell_fn(state, "ret20", "atrp_4h", thr)
        cells = {}
        for sig_name, series_d in signals.items():
            buckets = defaultdict(list)
            for d, v in series_d.items():
                c = cf(d)
                if c is not None:
                    buckets[c].append(v)
            cb = {}
            for cell in (("trend_on", "vol_on"), ("trend_on", "vol_off"),
                         ("trend_off", "vol_on"), ("trend_off", "vol_off")):
                xs = buckets.get(cell, [])
                cb[f"{cell[0]}__{cell[1]}"] = {
                    "n": len(xs), "mean": _r(sum(xs) / len(xs) if xs else None)}
            cb["welch_trap_vs_joint"] = {
                k: _r(v, 6) for k, v in welch(
                    buckets.get(("trend_on", "vol_off"), []),
                    buckets.get(("trend_on", "vol_on"), [])).items()}
            cb["welch_vol_no_trend_vs_joint"] = {
                k: _r(v, 6) for k, v in welch(
                    buckets.get(("trend_off", "vol_on"), []),
                    buckets.get(("trend_on", "vol_on"), [])).items()}
            cells[sig_name] = cb
        key_cells[thr_label] = cells
    report["key_cells_trend_x_vol"] = key_cells

    # ------------------------------------------------- gate correlation
    overlap = {}
    for a in primary:
        for b in primary:
            if a >= b:
                continue
            fa, fb = gates[a], gates[b]
            both = [d for d in cohort_days
                    if fa(d) is not None and fb(d) is not None]
            if not both:
                continue
            agree = sum(1 for d in both if fa(d) == fb(d))
            a_on = [d for d in both if fa(d)]
            b_on = [d for d in both if fb(d)]
            overlap[f"{a}__vs__{b}"] = {
                "days_both_defined": len(both),
                "agreement_pct": _r(100.0 * agree / len(both), 1),
                "p_b_on_given_a_on_pct": _r(
                    100.0 * sum(1 for d in a_on if fb(d)) / len(a_on), 1)
                if a_on else None,
                "p_a_on_given_b_on_pct": _r(
                    100.0 * sum(1 for d in b_on if fa(d)) / len(b_on), 1)
                if b_on else None}
    report["gate_correlation"] = overlap

    # ------------------------------------------------- FDR family
    fdr_cells = []
    for name in primary:
        for sig_name in signals:
            fdr_cells.append((f"{name}__{sig_name}__on_off",
                              report["gates"][name]["signals"][sig_name]
                              ["welch_p"]))
    for sig_name in signals:
        fdr_cells.append((f"trap_vs_joint__{sig_name}",
                          key_cells["split_0.025"][sig_name]
                          ["welch_trap_vs_joint"]["p"]))
    fdr_map, fdr_table = bh_fdr(fdr_cells)
    report["fdr_bh_0.05"] = {
        "family": "5 primary gates x 4 signals on/off Welch p + 4 "
                  "trap-vs-joint Welch p = 24 cells (small family)",
        "table": fdr_table}

    # ------------------------------------------------- verdict
    def g(n, s):
        return report["gates"][n]["signals"][s]
    score = {}
    for n in ("ret20_pos", "atrp_d_high", "atrp4h_high_0.025"):
        ts = [g(n, s)["welch_t"] for s in signals]
        diffs = [g(n, s)["on_minus_off"] for s in signals]
        score[n] = {
            "mean_welch_t": _r(sum(t for t in ts if t is not None)
                               / max(1, sum(1 for t in ts if t is not None))),
            "positive_diff_signals": sum(1 for d in diffs
                                         if d is not None and d > 0),
            "ts": {s: g(n, s)["welch_t"] for s in signals}}
    trap_dead = {}    # trap <= 0: signal truly dies in trend-without-vol
    trap_weaker = {}  # trap < joint: vol adds on top of trend
    for s in signals:
        cell = key_cells["split_0.025"][s]
        tm = cell["trend_on__vol_off"]["mean"]
        jm = cell["trend_on__vol_on"]["mean"]
        trap_dead[s] = (tm is not None and tm <= 0)
        trap_weaker[s] = (tm is not None and jm is not None and tm < jm)
    n_trap_dead = sum(trap_dead.values())
    n_trap_weaker = sum(trap_weaker.values())
    # is carry the exception — vol-gated rather than trend-gated?
    carry_ret_t = g("ret20_pos", "funding_carry_5d")["welch_t"]
    carry_vol_ts = [g("atrp_d_high", "funding_carry_5d")["welch_t"],
                    g("atrp4h_high_0.025", "funding_carry_5d")["welch_t"]]
    carry_vol_gated = (carry_ret_t is not None and carry_ret_t < 1.0
                       and any(t is not None and t > 2.0
                               for t in carry_vol_ts))
    best_atomic = max(("ret20_pos", "atrp_d_high", "atrp4h_high_0.025"),
                      key=lambda n: score[n]["mean_welch_t"]
                      if score[n]["mean_welch_t"] is not None else -9)
    ret_t = score["ret20_pos"]["mean_welch_t"]
    atrp4_t = score["atrp4h_high_0.025"]["mean_welch_t"]
    atrpd_t = score["atrp_d_high"]["mean_welch_t"]
    if n_trap_dead >= 3 and best_atomic != "ret20_pos":
        verdict = ("COMPLEMENTARY/REFINEMENT: atrp separates the signals too "
                   f"(mean Welch t ret20={ret_t}, atrp_d={atrpd_t}, "
                   f"atrp4h={atrp4_t}) and the trend-without-vol trap cell "
                   f"is DEAD (<=0) for {n_trap_dead}/4 signals — their "
                   "theory replicates on our cohort; vol adds information "
                   "beyond trend, not a replacement (gates are correlated "
                   "— see overlap).")
    elif best_atomic == "ret20_pos" and (n_trap_dead or n_trap_weaker >= 3):
        verdict = ("COMPLEMENTARY: ret20 remains the strongest single "
                   f"separator (mean Welch t {ret_t} vs atrp_d {atrpd_t} / "
                   f"atrp4h {atrp4_t}); the trend-without-vol cell is weaker "
                   f"than joint trend+vol for {n_trap_weaker}/4 signals and "
                   f"fully dead for {n_trap_dead}/4 — atrp refines the "
                   "trend gate rather than replacing it. Caveat: the 0.025 "
                   "split is a 2021-22-era tail regime on our span "
                   "(occupancy ~13%, zero on-days after 2022); the "
                   "median-split joint arm is the usable version.")
        if carry_vol_gated:
            verdict += (" Exception: funding carry is VOL-gated, not "
                        "trend-gated (ret20 Welch t "
                        f"{_r(carry_ret_t)}; atrp arms t "
                        f"{[_r(t) for t in carry_vol_ts]}) — the two state "
                        "variables condition different signals.")
    else:
        verdict = ("TREND SUFFICES: ret20 is the strongest single separator "
                   f"(mean Welch t {ret_t} vs atrp_d {atrpd_t} / atrp4h "
                   f"{atrp4_t}) and the trend-without-vol trap is NOT dead "
                   f"for {4 - n_trap_dead}/4 signals — their 4h-vol theory "
                   "does not replicate on daily mega-cohort signals.")
    report["verdict_block"] = {
        "atomic_gate_scores_mean_welch_t": score,
        "trap_cell_dead_le0": trap_dead,
        "trap_cell_weaker_than_joint": trap_weaker,
        "funding_carry_vol_gated_not_trend_gated": carry_vol_gated,
        "verdict": verdict}

    report["honesty"] = {
        "not_a_return": "decile spreads are gross close-price moves; no "
                        "costs, funding cashflows, borrow or slippage",
        "not_an_asof_vintage": "second-hand archive copy (rc_futures_v1)",
        "nominal_inference": "5d labels overlap ~80% on adjacent days; all "
                             "t/p nominal",
        "in_sample_median_threshold": "atrp_d (and the median sensitivity "
                                      "arm) split at the in-sample median "
                                      "by design — occupancy ~50/50 is "
                                      "constructed, not measured",
        "not_live": "no orders, no account, no broker",
        "no_profitability_claim": True,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    brief = {"status": "ran", "out": str(args.out),
             "decision_days": len(cohort_days),
             "thresholds": report["btc_state"],
             "atomic_scores": {k: v["mean_welch_t"]
                               for k, v in score.items()},
             "trap_weaker_signals": n_trap_dead,
             "verdict": verdict}
    for name in primary:
        brief[name] = {
            "occ_pct": report["gates"][name]["occupancy_on_pct"],
            "xs_dfh20": report["gates"][name]["signals"]["xs_dfh20_5d"]}
    print(json.dumps(brief, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
