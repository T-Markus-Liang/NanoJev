#!/usr/bin/env python3
"""T69: robustness scan for the funding-following signal.

Grids over (hold_horizon, funding threshold) to check the T66/T67 finding is
not a single-parameter accident. For each cell: top-threshold arm mean 5d..10d
forward return vs rest, Welch t. Also a signed Spearman (funding pct -> return)
per horizon as a smoother check.

Cells: hold in {3,5,10} x threshold in {0.70,0.80,0.90} = 9 arm contrasts
       + 3 Spearman = 12 family members, BH-FDR applied.
PIT-safe trailing-180 windows; measurement only.
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
OUT = ROOT / "results/financial_signal_robustness_v1.json"
LOOKBACK = 180
HOLDS = [3, 5, 10]
THRESHOLDS = [0.70, 0.80, 0.90]


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
    return {"rho": round(rho, 4), "p": round(p, 6)}


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
    return {"n_a": len(a), "mean_a_bps": round(ma * 1e4, 1),
            "mean_b_bps": round(mb * 1e4, 1), "t": round(t, 3),
            "p": round(p, 6)}


def run():
    records = [json.loads(l) for l in COHORT.read_text().splitlines() if l.strip()]
    by_asset = defaultdict(list)
    for r in records:
        by_asset[r["asset_id"]].append(r)
    for rs in by_asset.values():
        rs.sort(key=lambda x: x["decision_ns"])

    max_hold = max(HOLDS)
    rows = []
    for asset, rs in sorted(by_asset.items()):
        fund = [x["features"]["last_funding_rate"]["value"] for x in rs]
        mark = [x["features"]["mark_price"]["value"] for x in rs]
        for i in range(LOOKBACK, len(rs) - max_hold):
            if mark[i] <= 0:
                continue
            fw = fund[i - LOOKBACK:i]
            pct = (sum(1 for w in fw if w < fund[i])
                   + 0.5 * sum(1 for w in fw if w == fund[i])) / LOOKBACK
            rets = {h: (math.log(mark[i + h] / mark[i])
                        if mark[i + h] > 0 else None) for h in HOLDS}
            rows.append({"pct": pct, "rets": rets})

    out = {"cells": {}, "spearmans": {}, "p_values": []}
    for h in HOLDS:
        xs = [r["pct"] for r in rows if r["rets"][h] is not None]
        ys = [r["rets"][h] for r in rows if r["rets"][h] is not None]
        sp = spearman(xs, ys)
        out["spearmans"][f"hold_{h}d"] = sp
        if sp:
            out["p_values"].append((f"spearman_{h}d", sp["p"]))
        for th in THRESHOLDS:
            a = [r["rets"][h] for r in rows
                 if r["rets"][h] is not None and r["pct"] >= th]
            b = [r["rets"][h] for r in rows
                 if r["rets"][h] is not None and r["pct"] < th]
            c = welch(a, b)
            out["cells"][f"h{h}_t{int(th*100)}"] = c
            if c:
                out["p_values"].append((f"h{h}_t{int(th*100)}", c["p"]))

    ps = sorted([(p, n) for n, p in out["p_values"]])
    m = len(ps)
    out["fdr_bh"] = [{"cell": n, "p": p,
                      "survives": p <= 0.05 * (i + 1) / m}
                     for i, (p, n) in enumerate(ps)]
    return {"schema_version": "nanojev-financial-signal-robustness-v1",
            "status": "measurement_complete",
            "scope": "horizon x threshold robustness scan; measurement only",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "results": out}


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
