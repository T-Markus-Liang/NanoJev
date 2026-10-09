#!/usr/bin/env python3
"""T88: hourly-scale baseline signal arms — 1h analog of the T82 daily benchmark.

Cohort: data/perp_pit_intraday_v1/records.jsonl — hourly perp PIT records
(asset set, span, and n pinned by sha256 in the receipt; the dataset has been
expanded in place historically, so always read these from the receipt).
Target: label.forward_return_bps — 4h forward mark move, bps (gross).

Arms (all scores = trailing-90-record (~90h) mid-rank percentiles, window
excludes the current record — same PIT convention as the daily benchmark):

  mom_1h      percentile of log(mark[i]/mark[i-1])   momentum vs reversal
  mom_4h      percentile of log(mark[i]/mark[i-4])   momentum vs reversal
  rv_pct      realized_vol_24h percentile            vol clustering
  taker_pct   taker_buy_ratio percentile             flow pressure
  basis_pct   mark_index_basis_bps percentile
  funding_pct last_funding_rate percentile           hourly analog of the
                                                     daily confirmed signal

Metrics per arm: Spearman rho/t/p on (score, fwd_bps) over the common eval
row set; Welch top-vs-bottom quintile mean diff on fwd_bps; n.

Controls: placebo = labels shuffled within the dataset, 2 deterministic seeds
(LCG Fisher-Yates, same as benchmark); run() twice for byte-determinism.

Verdict: BH-FDR alpha=0.05 over the 6 arm primary p-values — the momentum
hypothesis contributes two lookback arms, a strictly more conservative family
than the "5 cells" phrasing in the brief. PROVISIONAL_PASS = FDR-survive AND
|rho| > placebo band. This remains a pilot-scale hourly panel: EVERY verdict
is provisional and underpowered regardless of the current cohort size.

Measurement only — no fitting, no trading, no promotion claims.
"""

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_intraday_v1/records.jsonl"
OUT = ROOT / "results/financial_signal_intraday_v1.json"
LOOKBACK = 90            # trailing-90-record (~90h) percentile window
MOM4_LAG = 4             # mom_4h lookback in bars
Q = 0.20                 # quintile fraction for Welch arm
PLACEBO_SEEDS = (17, 73)
ARMS = ["mom_1h", "mom_4h", "rv_pct", "taker_pct", "basis_pct", "funding_pct"]


# ---------------------------------------------------------------- stats
def pct(w, x):
    """Mid-rank percentile of x within window w (project convention)."""
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
    rho = num / (dx * dy)
    t = rho * math.sqrt((n - 2) / max(1e-9, 1 - rho * rho))
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return {"rho": rho, "t": t, "p": p, "n": n}


def welch(a, b):
    if len(a) < 10 or len(b) < 10:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    se = math.sqrt(statistics.pvariance(a) / len(a)
                   + statistics.pvariance(b) / len(b))
    if se == 0:
        return None
    t = (ma - mb) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return {"n_top": len(a), "n_bottom": len(b),
            "mean_top_bps": ma, "mean_bottom_bps": mb,
            "diff_bps": ma - mb, "t": t, "p": p}


# ---------------------------------------------------------------- dataset
def build_rows():
    """Common eval row set: every arm score defined -> i >= LOOKBACK+MOM4_LAG."""
    records = [json.loads(l) for l in COHORT.read_text().splitlines()
               if l.strip()]
    records.sort(key=lambda r: r["decision_ns"])
    n = len(records)

    def feat(name):
        return [r["features"][name]["value"] for r in records]

    mark = feat("mark_price")
    fund = feat("last_funding_rate")
    basis = feat("mark_index_basis_bps")
    rv = feat("realized_vol_24h")
    taker = feat("taker_buy_ratio")

    mom1 = [None] * n
    mom4 = [None] * n
    for i in range(1, n):
        if mark[i] > 0 and mark[i - 1] > 0:
            mom1[i] = math.log(mark[i] / mark[i - 1])
    for i in range(MOM4_LAG, n):
        if mark[i] > 0 and mark[i - MOM4_LAG] > 0:
            mom4[i] = math.log(mark[i] / mark[i - MOM4_LAG])

    raw = {"mom_1h": mom1, "mom_4h": mom4, "rv_pct": rv,
           "taker_pct": taker, "basis_pct": basis, "funding_pct": fund}

    i_min = LOOKBACK + MOM4_LAG   # all six windows complete
    rows = []
    for i in range(i_min, n):
        fwd = records[i]["label"].get("forward_return_bps")
        if fwd is None:
            continue
        row = {"ns": records[i]["decision_ns"], "i": i, "fwd_bps": fwd}
        ok = True
        for arm in ARMS:
            series = raw[arm]
            window = series[i - LOOKBACK:i]
            if series[i] is None or any(v is None for v in window):
                ok = False
                break
            row[arm] = pct(window, series[i])
        if ok:
            rows.append(row)
    return records, rows


# ---------------------------------------------------------------- cells
def cell_metrics(rows, arm):
    xs = [r[arm] for r in rows]
    ys = [r["fwd_bps"] for r in rows]
    sp = spearman(xs, ys)
    k = max(1, int(len(rows) * Q))
    srt = sorted(rows, key=lambda r: r[arm])
    w = welch([r["fwd_bps"] for r in srt[-k:]],
              [r["fwd_bps"] for r in srt[:k]])
    return {"n": len(rows),
            "spearman": ({k_: round(v, 5) for k_, v in sp.items()}
                         if sp else None),
            "welch_top_vs_bottom_quintile":
                ({k_: round(v, 5) for k_, v in w.items()} if w else None),
            "p": sp["p"] if sp else None,
            "rho": sp["rho"] if sp else None}


def shuffle(vals, seed):
    """Deterministic LCG Fisher-Yates (same generator as T82 benchmark)."""
    v = vals[:]
    s = seed
    for i in range(len(v) - 1, 0, -1):
        s = (s * 6364136223846793005 + 1442695040888963407) & (2**64 - 1)
        j = s % (i + 1)
        v[i], v[j] = v[j], v[i]
    return v


def placebo_bands(rows):
    """Shuffle fwd_bps within the dataset; rho per arm per seed."""
    ys = [r["fwd_bps"] for r in rows]
    out = {}
    for arm in ARMS:
        xs = [r[arm] for r in rows]
        rhos = []
        for seed in PLACEBO_SEEDS:
            sp = spearman(xs, shuffle(ys, seed))
            rhos.append(round(sp["rho"], 5) if sp else None)
        band = max((abs(x) for x in rhos if x is not None), default=None)
        out[arm] = {"rhos": rhos, "max_abs_rho": band}
    return out


def run():
    records, rows = build_rows()
    ys = [r["fwd_bps"] for r in rows]

    cells = {arm: cell_metrics(rows, arm) for arm in ARMS}
    placebo = placebo_bands(rows)

    # BH-FDR over the arm primary p-values
    ps = sorted((m["p"], n) for n, m in cells.items()
                if m.get("p") is not None)
    M = len(ps)
    fdr = {}
    for i, (p, n) in enumerate(ps):
        fdr[n] = p <= 0.05 * (i + 1) / M

    verdicts = {}
    for n, m in cells.items():
        rho = m["rho"]
        band = placebo[n]["max_abs_rho"]
        fdr_pass = fdr.get(n, False)
        placebo_clear = (rho is not None and band is not None
                         and abs(rho) > max(0.02, band))
        verdicts[n] = {
            "fdr_pass": fdr_pass,
            "placebo_clear": placebo_clear,
            "verdict": ("PROVISIONAL_PASS" if fdr_pass and placebo_clear
                        else "NO_SIGNAL")}

    return {
        "schema_version": "nanojev-financial-signal-intraday-v1",
        "status": "measurement_complete",
        "task": "T88",
        "contract": {
            "cohort": str(COHORT.relative_to(ROOT)),
            "cohort_sha256": hashlib.sha256(
                COHORT.read_bytes()).hexdigest(),
            "n_records": len(records),
            "asset": sorted({r["asset_id"] for r in records}),
            "span": [records[0]["id"], records[-1]["id"]],
            "label": "label.forward_return_bps (4h forward mark bps, gross)",
            "score": "trailing-90-record mid-rank percentile (PIT)",
            "lookback_records": LOOKBACK,
            "welch_quantile": Q,
            "fdr_alpha": 0.05,
            "fdr_family": ARMS,
            "fdr_family_note": ("6 arm cells — the momentum hypothesis is "
                                "tested at two lookbacks; more conservative "
                                "than the 5-cell phrasing in the brief"),
            "placebo_seeds": PLACEBO_SEEDS},
        "n_eval_rows": len(rows),
        "warmup_dropped": len(records) - len(rows),
        "label_stats_bps": {
            "mean": round(statistics.mean(ys), 3),
            "median": round(statistics.median(ys), 3),
            "sd": round(statistics.pstdev(ys), 3),
            "min": round(min(ys), 2), "max": round(max(ys), 2)},
        "cells": cells,
        "placebo": placebo,
        "fdr_pass": fdr,
        "verdicts": verdicts,
        "pilot_caveats": [
            "hourly serial correlation makes the effective n far smaller "
            "than n_eval_rows in this receipt; normal-approx p is optimistic",
            "the cohort is a single time-sorted stream — when it contains "
            "multiple assets, trailing-90-record windows mix assets; "
            "cross-asset/cross-regime replication is still limited to the "
            "span pinned in the receipt",
            "4h forward windows overlap ~4x — quintile/Welch t-stats are "
            "not independent",
            "gross mark move only — no fees, spread, slippage, or funding "
            "cashflows; not a tradability claim",
            "any PROVISIONAL_PASS is a hypothesis for a bigger cohort, "
            "not a confirmed signal"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    r1, r2 = run(), run()
    same = json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    r1["determinism"] = {"replays": 2, "byte_identical": same}
    r1["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    blob = json.dumps(r1, indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(blob)
    print(json.dumps({"output": str(args.output), "deterministic": same,
                      "sha256": hashlib.sha256(blob.encode()).hexdigest()},
                     indent=2))


if __name__ == "__main__":
    main()
