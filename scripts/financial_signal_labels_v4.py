#!/usr/bin/env python3
"""T66 label-redesign experiment: does a continuous/vol-normalized/longer-horizon
target recover signal that the 25bps/1d binary label destroys?

Same cohort, same trailing-180 PIT windows, same top feature hypotheses —
only the TARGET changes:

  L1  continuous: next-day mark-price log return (from cohort mark series)
  L2  vol-normalized: next-day return / realized_vol_24bar at decision time
  L3  5-day forward mark log return

Feature arms (per-record trailing percentiles):
  funding_pct      last_funding_rate mid-rank pct in trailing 180
  abnvol_pct       quote_volume/trailing-30-median mid-rank pct
  dbasis_pct       5-bar Δ mark_index_basis_bps mid-rank pct
  rv_pct           realized_vol_24bar pct (control / conditioner)

Statistics: Spearman rank correlation feature_pct -> forward return,
per-asset + pooled; top/bottom decile arm mean-return t-tests; BH-FDR
over all feature×label cells; per frozen test-fold readout of the pooled
Spearman. PIT-safe trailing windows only. Measurement only — no fitting.
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
COHORT = ROOT / "data/perp_pit_v1/records.jsonl"
FOLDS = ROOT / "research/financial_r1_pit_validator_core_v2.json"
OUT = ROOT / "results/financial_signal_labels_v4.json"
LOOKBACK = 180


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
    rho = num / (dx * dy)
    # t approximation
    if abs(rho) >= 1:
        return {"rho": round(rho, 4), "t": None, "p": 0.0}
    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    # normal approx for p (n large)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return {"rho": round(rho, 4), "t": round(t, 3), "p": round(p, 6)}


def mean_t(a, b):
    """Welch t for mean(a)-mean(b)."""
    if len(a) < 10 or len(b) < 10:
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = statistics.pvariance(a) / len(a)
    vb = statistics.pvariance(b) / len(b)
    se = math.sqrt(va + vb)
    if se == 0:
        return None
    t = (ma - mb) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return {"mean_a_bps": round(ma * 1e4, 1), "mean_b_bps": round(mb * 1e4, 1),
            "t": round(t, 3), "p": round(p, 6)}


def run():
    records = [json.loads(l) for l in COHORT.read_text().splitlines() if l.strip()]
    folds = json.loads(FOLDS.read_text())["folds"]
    windows = [(f["test"][0], f["test"][1]) for f in folds]
    in_test = lambda ns: any(a <= ns < b for a, b in windows)
    by_asset = defaultdict(list)
    for r in records:
        by_asset[r["asset_id"]].append(r)
    for rs in by_asset.values():
        rs.sort(key=lambda x: x["decision_ns"])

    cells = defaultdict(list)   # (feature,label) -> [(x,y,ns)]
    for asset, rs in sorted(by_asset.items()):
        fund = [x["features"]["last_funding_rate"]["value"] for x in rs]
        vol = [x["features"]["quote_volume"]["value"] for x in rs]
        basis = [x["features"]["mark_index_basis_bps"]["value"] for x in rs]
        rv = [x["features"]["realized_vol_24bar"]["value"] for x in rs]
        mark = [x["features"]["mark_price"]["value"] for x in rs]
        for i in range(LOOKBACK, len(rs) - 5):
            fw = fund[i - LOOKBACK:i]
            f_pct = (sum(1 for w in fw if w < fund[i])
                     + 0.5 * sum(1 for w in fw if w == fund[i])) / LOOKBACK
            med = statistics.median(vol[i - 30:i])
            abn = vol[i] / med if med > 0 else 1.0
            aw = []
            for j in range(max(30, i - LOOKBACK), i):
                m = statistics.median(vol[max(0, j - 30):j])
                if m > 0:
                    aw.append(vol[j] / m)
            a_pct = (sum(1 for w in aw if w < abn)
                     + 0.5 * sum(1 for w in aw if w == abn)) / len(aw) if aw else 0.5
            db = basis[i] - basis[i - 5]
            bw = [basis[j] - basis[j - 5] for j in range(max(5, i - LOOKBACK), i)]
            b_pct = (sum(1 for w in bw if w < db)
                     + 0.5 * sum(1 for w in bw if w == db)) / len(bw) if bw else 0.5
            vw = rv[i - LOOKBACK:i]
            r_pct = (sum(1 for w in vw if w < rv[i])
                     + 0.5 * sum(1 for w in vw if w == rv[i])) / LOOKBACK
            if mark[i] <= 0 or mark[i + 1] <= 0 or mark[i + 5] <= 0:
                continue
            l1 = math.log(mark[i + 1] / mark[i])
            l2 = l1 / rv[i] if rv[i] > 0 else None
            l3 = math.log(mark[i + 5] / mark[i])
            ns = rs[i]["decision_ns"]
            for feat, xv in (("funding", f_pct), ("abnvol", a_pct),
                             ("dbasis", b_pct), ("rv", r_pct)):
                cells[("l1", feat)].append((xv, l1, ns))
                if l2 is not None:
                    cells[("l2", feat)].append((xv, l2, ns))
                cells[("l3", feat)].append((xv, l3, ns))

    results = {"cells": {}, "p_values": []}
    for (lab, feat), rows in sorted(cells.items()):
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        sp = spearman(xs, ys)
        test_rows = [r for r in rows if in_test(r[2])]
        sp_t = spearman([r[0] for r in test_rows], [r[1] for r in test_rows])
        n = len(rows)
        hi = [r[1] for r in rows if r[0] >= 0.9]
        lo = [r[1] for r in rows if r[0] <= 0.1]
        rest = [r[1] for r in rows if 0.1 < r[0] < 0.9]
        entry = {"n": n, "spearman": sp,
                 "test_fold_spearman": sp_t,
                 "top_decile_vs_rest": mean_t(hi, rest),
                 "bottom_decile_vs_rest": mean_t(lo, rest)}
        results["cells"][f"{lab}:{feat}"] = entry
        if sp:
            results["p_values"].append((f"{lab}:{feat}", sp["p"]))

    ps = sorted([(p, n) for n, p in results["p_values"] if p is not None])
    m = len(ps)
    results["fdr_bh"] = [{"cell": n, "p": p,
                          "bh_threshold": round(0.05 * (i + 1) / m, 6),
                          "survives": p <= 0.05 * (i + 1) / m}
                         for i, (p, n) in enumerate(ps)]
    results["cohort"] = {"records": len(records), "lookback": LOOKBACK}
    return {"schema_version": "nanojev-financial-signal-labels-v4",
            "status": "measurement_complete",
            "scope": "label-variant replay; no fitting/trading/claims",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "results": results}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    receipt = run()
    blob = json.dumps(receipt, indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(blob)
    print(json.dumps({"output": str(args.output),
                      "sha256": hashlib.sha256(blob.encode()).hexdigest()},
                     indent=2))


if __name__ == "__main__":
    main()
