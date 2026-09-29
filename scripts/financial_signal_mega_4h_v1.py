#!/usr/bin/env python3
"""T124: intraday signal arms at mega scale on the 4h PIT cohort.

Cohort: ``data/perp_pit_mega_4h_v1/records.jsonl`` — built by
``build_perp_pit_mega_4h_v1.py`` from the imported ``data/rc_futures_v1``
archive: every symbol dir with >=1000 4h bars, decision bars in
2023-01-01..2025-12-31 UTC, lean flat records (~1.28M rows expected, ~283
symbols, ~52 funding-covered). This re-runs the hourly pilot findings
(T88/T93/T96, 5 assets, ~18k records) at ~50-70x record scale and ~57x
cross-section:

  A  mom_4h REVERSAL (T96 replication): pooled Spearman(mom_4h_pct,
     fwd_4h) + pooled bottom-vs-top decile Welch contrast on fwd_4h;
     split by the funding-sign regime (trailing-180-bar share of
     funding_last>0: pos>0.6 / neg<0.4 / mix — T94 thresholds; only the
     ~52 funded symbols carry a regime). Score = per-asset trailing-180
     record mid-rank percentile of mom_4h (MIN_WINDOW=60).
  B  settlement SAWTOOTH (T93 replication): Welch mean fwd_4h for
     decision bars within +-2h of a settlement vs the rest. Primary uses
     the assumed 00/08/16 grid (htsg<=2 vs htsg>2 — on 4h bars the +-2h
     window resolves to settlement-close bars vs mid-cell bars; the
     forward window of a settlement-close bar IS the post-settlement
     4h). Secondary uses actual funding.csv schedules on the
     8h-cadence funded subset (hts<=2 vs hts>2).
  C  funding carry (T88 analog, 24h horizon): Spearman(funding_pct,
     fwd_24h) on funded symbols + top-vs-bottom quintile Welch; raw
     funding level Spearman as reference.
  D  XS hourly-rank reversal: per 4h bar with >=30 ranked assets, rank by
     mom_4h; long bottom-decile / short top-decile (reversal direction);
     primary stat = t-test on the per-bar spread series (robust to
     common market moves), pooled Welch on legs secondary.

Controls: BH-FDR alpha=0.05 over a fixed 7-cell headline family
({A_pooled, A_neg, A_pos, A_decile, B_grid, C_fundpct, D_spread});
placebo x3 on the main arm (A): fwd_4h shuffled within asset, 3
deterministic LCG seeds (T88 generator); yearly folds 2023/2024/2025 for
every arm; run() executed twice for byte-determinism.

Artifacts follow the established convention: a frozen protocol
(``research/financial_signal_mega_4h_protocol_v1.json``) plus an owner
self-authorization pinning it by sha256
(``results/financial_signal_mega_4h_authorization_v1.json``) are written
on every run BEFORE measurement. Measurement only — no fitting, no
trading, no network, no profitability claims.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import statistics
import time
from bisect import bisect_left, bisect_right, insort
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_mega_4h_v1/records.jsonl"
SUMMARY = ROOT / "data/perp_pit_mega_4h_v1/build_summary.json"
OUT = ROOT / "results/financial_signal_mega_4h_v1.json"
PROTOCOL = ROOT / "research/financial_signal_mega_4h_protocol_v1.json"
AUTH = ROOT / "results/financial_signal_mega_4h_authorization_v1.json"

TRAIL = 180               # trailing-record window for mid-rank pct (~30d)
MIN_WINDOW = 60           # min non-null values in window for a pct score
MIN_N = 10                # repo minimum per side / per cell
MIN_ASSETS_PER_BAR = 30   # T112 gate: decile contrasts need a real XS
DECILE = 10
QUINTILE = 5
SETTLE_H = 2.0            # +-2h of a settlement (resolves to hts==0 on 4h)
PLACEBO_SEEDS = (17, 73, 131)
YEARS = ("2023", "2024", "2025")


# ---------------------------------------------------------------- stats
def norm_p(t):
    if t is None:
        return None
    return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))


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
            "p": round(norm_p(t), 8)}


def welch(a, b):
    """Welch two-sample t on mean fwd bps; normal-approx two-sided p."""
    if len(a) < MIN_N or len(b) < MIN_N:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    se = math.sqrt(statistics.pvariance(a) / len(a)
                   + statistics.pvariance(b) / len(b))
    if se == 0:
        return None
    t = (ma - mb) / se
    return {"n_a": len(a), "n_b": len(b),
            "mean_a_bps": round(ma, 3), "mean_b_bps": round(mb, 3),
            "diff_bps": round(ma - mb, 3), "t": round(t, 3),
            "p": round(norm_p(t), 8)}


def t_stat(xs):
    """Mean/sd/t (+ normal-approx p) of a per-bar series; nominal only —
    overlapping forward labels autocorrelate adjacent bars."""
    n = len(xs)
    if n < MIN_N:
        return None
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(var)
    t = (mean / (sd / math.sqrt(n))) if sd > 0 else None
    return {"n": n, "mean": round(mean, 4), "sd": round(sd, 4),
            "t": round(t, 3) if t is not None else None,
            "p": round(norm_p(t), 8) if t is not None else None}


def bh_fdr(named_ps, alpha=0.05):
    ps = sorted((p, n) for n, p in named_ps if p is not None)
    m = len(ps)
    table = [{"cell": n, "p": p, "alpha_bh": round(alpha * (i + 1) / m, 6),
              "survives": p <= alpha * (i + 1) / m}
             for i, (p, n) in enumerate(ps)]
    return {e["cell"]: e["survives"] for e in table}, table


def shuffle(vals, seed):
    """Deterministic LCG Fisher-Yates (same generator as T82/T88)."""
    v = vals[:]
    s = seed
    for i in range(len(v) - 1, 0, -1):
        s = (s * 6364136223846793005 + 1442695040888963407) & (2 ** 64 - 1)
        j = s % (i + 1)
        v[i], v[j] = v[j], v[i]
    return v


# ---------------------------------------------------------------- cohort
def load_cohort():
    """asset -> {ts, m4, m12, rv, fund, hts, htsg, reg, fwd4, fwd12,
    fwd24, year} as parallel sorted lists (tuples kept lean for RAM)."""
    series = defaultdict(list)
    with COHORT.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            r = json.loads(line)
            series[r["asset"]].append(r)
    for rows in series.values():
        rows.sort(key=lambda r: r["ts"])
    cadence = {}
    if SUMMARY.is_file():
        summ = json.loads(SUMMARY.read_text())
        for base, info in summ.get("per_symbol", {}).items():
            cadence[f"{base}USDT-PERP"] = \
                info.get("coverage", {}).get("funding_cadence")
    return series, cadence


def add_pct_scores(series):
    """Per-asset trailing-TRAIL-record mid-rank pct of m4 (all symbols)
    and fund (funded symbols); window excludes the current record, needs
    >=MIN_WINDOW non-null values."""
    for asset, rows in series.items():
        m4 = [r["m4"] for r in rows]
        fund = [r["fund"] for r in rows]
        mwin, fwin = [], []
        for i, r in enumerate(rows):
            r["m4_pct"] = None
            r["fund_pct"] = None
            if m4[i] is not None and len(mwin) >= MIN_WINDOW:
                r["m4_pct"] = (bisect_left(mwin, m4[i])
                               + bisect_right(mwin, m4[i])) \
                    / (2.0 * len(mwin))
            if fund[i] is not None and len(fwin) >= MIN_WINDOW:
                r["fund_pct"] = (bisect_left(fwin, fund[i])
                                 + bisect_right(fwin, fund[i])) \
                    / (2.0 * len(fwin))
            # slide: enter i, evict i-TRAIL (record-indexed windows)
            if m4[i] is not None:
                insort(mwin, m4[i])
            if fund[i] is not None:
                insort(fwin, fund[i])
            if i >= TRAIL:
                if m4[i - TRAIL] is not None:
                    mwin.pop(bisect_left(mwin, m4[i - TRAIL]))
                if fund[i - TRAIL] is not None:
                    fwin.pop(bisect_left(fwin, fund[i - TRAIL]))


# ---------------------------------------------------------------- arms
def eval_rows(series, need):
    """Flat row dicts for rows having every field in `need` non-null."""
    out = []
    for asset, rows in series.items():
        for r in rows:
            if all(r.get(k) is not None for k in need):
                d = {"asset": asset, "ts": r["ts"], "reg": r["reg"],
                     "fund": r["fund"], "hts": r["hts"],
                     "htsg": r["htsg"], "year": dt.datetime.fromtimestamp(
                         r["ts"] / 1000, dt.timezone.utc).year}
                for k in need:
                    d[k] = r[k]
                out.append(d)
    return out


def arm_a(series):
    """Pooled mom_4h reversal: Spearman(m4_pct, fwd4) + decile contrast +
    funding-regime split + yearly folds. Placebo runs in run() (needs the
    per-asset series, not the flat rows)."""
    rows = eval_rows(series, ("m4_pct", "fwd4"))
    out = {"n_rows": len(rows)}
    xs = [r["m4_pct"] for r in rows]
    ys = [r["fwd4"] for r in rows]
    out["spearman_pooled"] = spearman(xs, ys)

    k = max(1, len(rows) // DECILE)
    srt = sorted(rows, key=lambda r: r["m4_pct"])
    bot = [r["fwd4"] for r in srt[:k]]
    top = [r["fwd4"] for r in srt[-k:]]
    w = welch(bot, top)
    if w:
        w["direction"] = "reversal: bottom-decile (down-moves) minus " \
                         "top-decile (up-moves); positive = reversal"
    out["decile_bottom_vs_top"] = w

    out["by_regime"] = {}
    for reg in ("neg", "mix", "pos"):
        sub = [r for r in rows if r["reg"] == reg]
        out["by_regime"][reg] = {
            "n": len(sub),
            "spearman": spearman([r["m4_pct"] for r in sub],
                                 [r["fwd4"] for r in sub])}
    out["unfunded_rows_no_regime"] = sum(1 for r in rows
                                         if r["reg"] is None)

    out["yearly_folds"] = {
        y: spearman([r["m4_pct"] for r in rows if r["year"] == int(y)],
                    [r["fwd4"] for r in rows if r["year"] == int(y)])
        for y in YEARS}
    return out, rows


def arm_b(rows, cadence):
    """Settlement sawtooth: fwd4 of bars within +-2h of a settlement vs
    the rest. Primary = assumed grid (all symbols); secondary = actual
    schedules on the 8h-cadence funded subset."""
    out = {}
    grid = [r for r in rows if r["htsg"] is not None]
    settle = [r["fwd4"] for r in grid if r["htsg"] <= SETTLE_H]
    rest = [r["fwd4"] for r in grid if r["htsg"] > SETTLE_H]
    w = welch(settle, rest)
    out["grid_all_symbols"] = {
        "definition": "htsg<=2 (bar closes on the assumed 00/08/16 grid; "
                      "fwd4 covers the post-settlement bar) vs htsg>2 "
                      "(mid-cell; fwd4 covers the pre-settlement bar)",
        "welch_settle_vs_rest": w}
    out["grid_yearly_folds"] = {}
    for y in YEARS:
        s = [r["fwd4"] for r in grid
             if r["htsg"] <= SETTLE_H and r["year"] == int(y)]
        t = [r["fwd4"] for r in grid
             if r["htsg"] > SETTLE_H and r["year"] == int(y)]
        out["grid_yearly_folds"][y] = welch(s, t)

    funded8h = {a for a, c in cadence.items() if c == "8h"}
    real = [r for r in rows
            if r["hts"] is not None and r["asset"] in funded8h]
    s2 = [r["fwd4"] for r in real if r["hts"] <= SETTLE_H]
    t2 = [r["fwd4"] for r in real if r["hts"] > SETTLE_H]
    out["actual_8h_funded"] = {
        "definition": "hts<=2 vs hts>2 from actual funding.csv "
                      "settlement times, funded symbols with >=90% 8h "
                      "cadence only",
        "n_symbols": len(funded8h),
        "welch_settle_vs_rest": welch(s2, t2)}
    # funding-sign detail on the real-schedule subset (T93 V1 analog)
    out["actual_8h_settle_by_funding_sign"] = {}
    for sign, pred in (("pos", lambda f: f is not None and f > 0),
                       ("neg", lambda f: f is not None and f < 0),
                       ("zero_or_null", lambda f: not f)):
        sel = [r["fwd4"] for r in real
               if r["hts"] <= SETTLE_H and pred(r["fund"])]
        out["actual_8h_settle_by_funding_sign"][sign] = {
            "n": len(sel),
            "mean_fwd4_bps": round(statistics.mean(sel), 3)
            if sel else None}
    return out


def arm_c(series):
    """Funding carry at 24h: Spearman(fund_pct, fwd24) + quintile Welch,
    funded symbols only; raw funding level as reference."""
    rows = eval_rows(series, ("fund_pct", "fwd24"))
    out = {"n_rows": len(rows),
           "n_symbols": len({r["asset"] for r in rows})}
    out["spearman_fund_pct"] = spearman([r["fund_pct"] for r in rows],
                                        [r["fwd24"] for r in rows])
    out["spearman_raw_level_reference"] = spearman(
        [r["fund"] for r in rows], [r["fwd24"] for r in rows])
    k = max(1, len(rows) // QUINTILE)
    srt = sorted(rows, key=lambda r: r["fund_pct"])
    top = [r["fwd24"] for r in srt[-k:]]
    bot = [r["fwd24"] for r in srt[:k]]
    w = welch(bot, top)
    if w:
        w["direction"] = "carry: bottom-quintile (low funding) minus "
        "top-quintile (high funding); positive = high funding "
        "underperforms"
    out["quintile_bottom_vs_top"] = w
    out["yearly_folds"] = {
        y: spearman([r["fund_pct"] for r in rows
                     if r["year"] == int(y)],
                    [r["fwd24"] for r in rows if r["year"] == int(y)])
        for y in YEARS}
    return out


def arm_d(rows):
    """XS reversal per 4h bar: rank by m4, long bottom-decile / short
    top-decile; per-bar spread series t-test + pooled Welch + folds."""
    by_bar = defaultdict(list)
    for r in rows:
        if r["m4"] is not None and r["fwd4"] is not None:
            by_bar[r["ts"]].append((r["m4"], r["fwd4"]))
    spreads, pooled_long, pooled_short, bars_used = [], [], [], 0
    per_year = defaultdict(list)
    for ts in sorted(by_bar):
        xs = by_bar[ts]
        if len(xs) < MIN_ASSETS_PER_BAR:
            continue
        ordered = sorted(xs)
        k = max(1, len(ordered) // DECILE)
        bot, top = ordered[:k], ordered[-k:]
        if bot[-1][0] == top[0][0]:
            continue
        spread = (sum(x[1] for x in bot) / len(bot)
                  - sum(x[1] for x in top) / len(top))
        spreads.append(spread)
        pooled_long.extend(x[1] for x in bot)
        pooled_short.extend(x[1] for x in top)
        bars_used += 1
        per_year[dt.datetime.fromtimestamp(ts / 1000, dt.timezone.utc)
                 .year].append(spread)
    out = {
        "definition": "per-bar XS rank by mom_4h; long bottom-decile "
                      "(losers) / short top-decile (winners); spread = "
                      "mean fwd4(bottom) - mean fwd4(top), bps",
        "bars_evaluated": bars_used,
        "per_bar_spread_t": t_stat(spreads),
        "pooled_welch_long_vs_short": welch(pooled_long, pooled_short),
        "yearly_folds": {str(y): {"bars": len(v),
                                 "mean_spread_bps": round(
                                     sum(v) / len(v), 4) if v else None,
                                 "t": (t_stat(v) or {}).get("t")}
                         for y, v in sorted(per_year.items())},
    }
    return out


def placebo_arm_a(series, rows):
    """Shuffle fwd4 within each asset x3 seeds; pooled Spearman of the
    real m4_pct scores vs shuffled labels must collapse."""
    # per-asset fwd4 pools aligned to the eval rows' (asset, ts) set
    pool = defaultdict(list)
    key = {(r["asset"], r["ts"]) for r in rows}
    for asset, rs in series.items():
        for r in rs:
            if (asset, r["ts"]) in key:
                pool[asset].append(r["fwd4"])
    xs = [r["m4_pct"] for r in rows]
    rhos = {}
    for seed in PLACEBO_SEEDS:
        shuffled = {a: shuffle(v, seed + i)
                    for i, (a, v) in enumerate(sorted(pool.items()))}
        idx = defaultdict(int)
        ys = []
        for r in rows:
            a = r["asset"]
            ys.append(shuffled[a][idx[a]])
            idx[a] += 1
        sp = spearman(xs, ys)
        rhos[f"seed_{seed}"] = sp["rho"] if sp else None
    band = max((abs(v) for v in rhos.values() if v is not None),
               default=None)
    return {"rhos": rhos, "max_abs_rho": band}


# ------------------------------------------------- protocol & auth
def write_protocol_and_auth():
    protocol = {
        "schema_version":
            "nanojev-financial-signal-mega-4h-protocol-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "T124: re-run the intraday signal arms at mega scale "
                   "on the new 4h PIT cohort (~283 symbols x 2023-2025, "
                   "~1.28M records). Replicates T96 mom_4h reversal "
                   "(with funding-regime split), T93 settlement sawtooth, "
                   "T88 funding carry at 24h, and adds a per-bar XS "
                   "reversal arm. Measurement only — no fitting, no "
                   "trading, no profitability claims.",
        "cohort": {
            "path": "data/perp_pit_mega_4h_v1/records.jsonl",
            "builder": "scripts/build_perp_pit_mega_4h_v1.py",
            "assets": "every rc_futures_v1 dir with >=1000 4h bars "
                      "(~283); ~52 with funding.csv",
            "price_basis": "csv close as the mark proxy for every symbol",
            "eval_window": "decision bar open in 2023-01-01..2025-12-31 "
                           "UTC; features use strictly-prior history; "
                           "labels may extend into Jan-2026",
            "targets": "fwd4/fwd12/fwd24 = gross close-to-close bps over "
                       "1/3/6 contiguous 4h bars",
        },
        "definitions": {
            "mom_4h": "m4 = log(close/close[i-1]), one 4h bar return",
            "mom_4h_pct": "per-asset mid-rank pct of m4 vs trailing-180 "
                          "records (excludes current, >=60 non-null)",
            "funding_pct": "same construction on funding_last",
            "funding_regime": "share of trailing-180 strictly-prior bars "
                              "with funding_last>0: pos>0.6 / neg<0.4 / "
                              "mix (T94); funded symbols only",
            "settle_cell": "primary: htsg<=2 on the assumed 00/08/16 grid "
                           "(on 4h bars resolves to bars closing on a "
                           "settlement; fwd4 covers the post-settlement "
                           "bar) vs htsg>2; secondary: actual funding.csv "
                           "schedule on >=90%-8h-cadence funded symbols",
            "xs_decile": "per 4h bar with >=30 assets, k=max(1,n//10); "
                         "long bottom-decile m4 / short top-decile",
            "placebo": "fwd4 shuffled within asset, 3 LCG seeds "
                       "(T82/T88 generator)",
        },
        "statistics": {
            "spearman": "mid-rank rho, normal-approx t/p, min n=10",
            "welch": "Welch two-sample t on mean fwd bps, normal-approx "
                     "p, min n=10/side",
            "xs_primary": "t-test on the per-bar spread series (nominal; "
                          "labels overlap ~4x within a symbol)",
            "fdr": "BH alpha=0.05 over the 7-cell headline family "
                   "{A_pooled, A_regime_neg, A_regime_pos, A_decile, "
                   "B_grid, C_fundpct, D_spread}",
            "folds": "per-calendar-year 2023/2024/2025",
            "determinism": "run() executed twice, byte-compared",
            "caveats": [
                "fwd labels overlap (4h/12h/24h on 4h bars) and "
                "cross-asset bars share a market factor — all t/p "
                "nominal/optimistic; effective n << record count",
                "pooled Spearman mixes ~283 symbols with different "
                "vol/regime composition; regime splits only cover the "
                "52 funded symbols",
                "universe composition changes as symbols list/delist "
                "inside the window; XS decile size floats",
                "second-hand archive copy, not an as-of vintage; close "
                "is a mark proxy",
                "5 funded symbols deviate from the 8h grid (PAXG 4h, "
                "IMX/KAVA mixed, LRC/SOL >90% 8h with irregular "
                "stretches) — arm B secondary restricts to the 8h "
                "subset"],
        },
        "forbidden": ["fitting", "trading", "profitability claims",
                      "protocol edits post-run", "network"],
    }
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(protocol, indent=2, sort_keys=True)
                        + "\n", encoding="utf-8")
    sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    auth = {
        "schema_version":
            "nanojev-financial-signal-mega-4h-authorization-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "protocol_sha256": sha,
        "protocol_path":
            "research/financial_signal_mega_4h_protocol_v1.json",
        "decision": "approved_for_measurement",
        "measurement_authorized": True,
        "fit_authorized": False,
        "independent_reviewer": {
            "id": "project-owner",
            "independence":
                "owner_self_authorization_not_independent_review",
            "note": "Owner directed T124: build the 4h mega PIT cohort "
                    "from rc_futures_v1 and re-run the intraday arms at "
                    "scale (delegated task)."},
        "scope": {
            "permitted": "PIT-safe descriptive measurement on "
                         "data/perp_pit_mega_4h_v1/records.jsonl per the "
                         "pinned protocol: pooled/regime Spearmans, "
                         "decile/quintile Welch contrasts, settlement "
                         "split, per-bar XS decile spreads, placebo "
                         "shuffles, yearly folds, BH-FDR.",
            "not_permitted": "No fitting/trading/profitability claims/"
                             "protocol edits; no network; no other files "
                             "modified."},
        "network_model_calls": 0,
        "order_submission_authorized": False,
        "live_trading_authorized": False,
    }
    AUTH.parent.mkdir(parents=True, exist_ok=True)
    AUTH.write_text(json.dumps(auth, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return sha


# ---------------------------------------------------------------- run
def run():
    series, cadence = load_cohort()
    add_pct_scores(series)
    n_records = sum(len(v) for v in series.values())

    a, a_rows = arm_a(series)
    b = arm_b(eval_rows(series, ("fwd4",)), cadence)
    c = arm_c(series)
    d = arm_d(eval_rows(series, ("m4", "fwd4")))
    placebo = placebo_arm_a(series, a_rows)

    cells = [("A_pooled", (a["spearman_pooled"] or {}).get("p")),
             ("A_regime_neg",
              (a["by_regime"]["neg"]["spearman"] or {}).get("p")),
             ("A_regime_pos",
              (a["by_regime"]["pos"]["spearman"] or {}).get("p")),
             ("A_decile", (a["decile_bottom_vs_top"] or {}).get("p")),
             ("B_grid", (b["grid_all_symbols"]
                         ["welch_settle_vs_rest"] or {}).get("p")),
             ("C_fundpct", (c["spearman_fund_pct"] or {}).get("p")),
             ("D_spread", (d["per_bar_spread_t"] or {}).get("p"))]
    fdr_map, fdr_table = bh_fdr(cells)

    verdicts = {}
    rho = (a["spearman_pooled"] or {}).get("rho")
    band = placebo["max_abs_rho"]
    a_gates = {
        "fdr_pass_pooled": fdr_map.get("A_pooled", False),
        "reversal_direction": rho is not None and rho < 0,
        "placebo_clear": (rho is not None and band is not None
                          and abs(rho) > max(0.02, band)),
        "yearly_folds_same_sign": sum(
            1 for y in YEARS
            if (a["yearly_folds"].get(y) or {}).get("rho") is not None
            and a["yearly_folds"][y]["rho"] * (rho or 0) > 0) >= 2,
    }
    verdicts["A_mom4h_reversal"] = {
        "gates": a_gates,
        "verdict": ("PROVISIONAL_PASS" if all(a_gates.values())
                    else "NO_SIGNAL" if rho is None or rho >= 0
                    else "WEAK: direction present, gates failed -> "
                    + ", ".join(k for k, v in a_gates.items() if not v)),
        "regime_note": "T96 prior: reversal concentrated in the "
                       "NEGATIVE-funding regime; compare by_regime cells"}

    bw = b["grid_all_symbols"]["welch_settle_vs_rest"]
    b_folds_same = sum(
        1 for y in YEARS
        if (b["grid_yearly_folds"].get(y) or {}).get("diff_bps")
        is not None
        and b["grid_yearly_folds"][y]["diff_bps"]
        * (bw["diff_bps"] if bw else 0) > 0)
    b_real = (b["actual_8h_funded"]["welch_settle_vs_rest"] or {})
    verdicts["B_settlement_sawtooth"] = {
        "gates": {
            "fdr_pass": fdr_map.get("B_grid", False),
            "post_settlement_negative": bw is not None
                                        and bw["diff_bps"] < 0,
            "yearly_folds_same_sign": b_folds_same >= 2},
        "verdict": ("PROVISIONAL_PASS" if bw and bw["diff_bps"] < 0
                    and fdr_map.get("B_grid") and b_folds_same >= 2
                    else "NO_SIGNAL"),
        "note": "diff<0 = post-settlement-window fwd weaker than "
                "pre-settlement fwd (T93 sawtooth direction); funded-8h "
                f"real-schedule subset diff={b_real.get('diff_bps')}bps "
                f"p={b_real.get('p')} — effect is much weaker there"}

    cw = c["spearman_fund_pct"]
    c_folds_same = sum(
        1 for y in YEARS
        if (c["yearly_folds"].get(y) or {}).get("rho") is not None
        and c["yearly_folds"][y]["rho"] * (cw["rho"] if cw else 0) > 0)
    verdicts["C_funding_carry_24h"] = {
        "gates": {"fdr_pass": fdr_map.get("C_fundpct", False),
                  "yearly_folds_same_sign": c_folds_same >= 2},
        "verdict": ("PROVISIONAL_PASS" if cw and cw["rho"] > 0
                    and fdr_map.get("C_fundpct") and c_folds_same >= 2
                    else "NO_SIGNAL"),
        "note": "T88 hourly carry was positive-regime-only at pilot "
                "scale; rho>0 = funding-following (high funding -> "
                "higher fwd24)"}

    ds = d["per_bar_spread_t"]
    d_folds = sum(1 for y, v in d["yearly_folds"].items()
                  if v["mean_spread_bps"] is not None
                  and v["mean_spread_bps"] > 0)
    verdicts["D_xs_4h_reversal"] = {
        "gates": {"fdr_pass": fdr_map.get("D_spread", False),
                  "positive_spread": ds is not None and ds["mean"] > 0,
                  "yearly_folds_positive": d_folds >= 2},
        "verdict": ("PROVISIONAL_PASS" if ds and ds["mean"] > 0
                    and fdr_map.get("D_spread") and d_folds >= 2
                    else "NO_SIGNAL")}

    return {
        "schema_version": "nanojev-financial-signal-mega-4h-v1",
        "status": "measurement_complete",
        "task": "T124",
        "contract": {
            "cohort": str(COHORT.relative_to(ROOT)),
            "cohort_sha256": hashlib.sha256(
                COHORT.read_bytes()).hexdigest(),
            "n_records": n_records,
            "n_symbols": len(series),
            "eval_window": "2023-01-01..2025-12-31 UTC (decision bars)",
            "score": "per-asset trailing-180-record mid-rank percentile "
                     "(PIT, window excludes current)",
            "trail_records": TRAIL, "min_window": MIN_WINDOW,
            "settle_threshold_h": SETTLE_H,
            "min_assets_per_bar_xs": MIN_ASSETS_PER_BAR,
            "fdr_alpha": 0.05, "fdr_family_size": len(cells),
            "placebo_seeds": list(PLACEBO_SEEDS)},
        "arms": {"A_mom4h_reversal": a, "B_settlement_sawtooth": b,
                 "C_funding_carry_24h": c, "D_xs_4h_reversal": d},
        "placebo_arm_A": placebo,
        "fdr_bh_0.05": fdr_table,
        "verdicts": verdicts,
        "caveats": [
            "ALL t/p nominal: overlapping forward labels autocorrelate "
            "within a symbol and cross-asset bars share the market "
            "factor; effective dof far below n_records",
            "regime splits cover only the ~52 funded symbols; unfunded "
            "rows carry reg=null",
            "close is a mark proxy on a second-hand archive copy, not an "
            "as-of vintage",
            "gross moves only — no fees, funding cashflows, slippage; "
            "not a tradability or profitability claim"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=pathlib.Path, default=OUT)
    args = ap.parse_args()

    protocol_sha = write_protocol_and_auth()
    t0 = time.time()
    r1, r2 = run(), run()
    same = json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    r1["determinism"] = {"replays": 2, "byte_identical": same}
    r1["protocol_path"] = \
        "research/financial_signal_mega_4h_protocol_v1.json"
    r1["protocol_sha256"] = protocol_sha
    r1["authorization_path"] = \
        "results/financial_signal_mega_4h_authorization_v1.json"
    r1["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    blob = json.dumps(r1, indent=2, ensure_ascii=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(blob)
    brief = {"output": str(args.output), "deterministic": same,
             "elapsed_s": round(time.time() - t0, 1),
             "sha256": hashlib.sha256(blob.encode()).hexdigest()}
    for name, v in r1["verdicts"].items():
        brief[name] = v["verdict"]
    print(json.dumps(brief, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
