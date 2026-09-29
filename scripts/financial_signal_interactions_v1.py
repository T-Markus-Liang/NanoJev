#!/usr/bin/env python3
"""T70: funding momentum + funding x basis interactions on the 5d target.

New arms beyond the level signal:
  fmom_pct   5-bar change in last_funding_rate, trailing-180 mid-rank pct
  fb_cross   funding tercile x basis tercile grid -> which cell drives the edge
  fmom_cond  funding-momentum arm conditioned on high funding level
             (crowded AND accelerating vs crowded AND cooling)

Target: 5-day forward mark log return. Welch t contrasts, Spearman for fmom,
BH-FDR over the family. PIT-safe trailing windows; measurement only.
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
OUT = ROOT / "results/financial_signal_interactions_v1.json"
LOOKBACK, HOLD = 180, 5


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


def tercile(w, x):
    p = (sum(1 for v in w if v < x) + 0.5 * sum(1 for v in w if v == x)) / len(w)
    return (0 if p < 1 / 3 else (1 if p < 2 / 3 else 2)), p


def run():
    records = [json.loads(l) for l in COHORT.read_text().splitlines() if l.strip()]
    by_asset = defaultdict(list)
    for r in records:
        by_asset[r["asset_id"]].append(r)
    for rs in by_asset.values():
        rs.sort(key=lambda x: x["decision_ns"])

    fmom = []           # (fmom_pct, ret)
    fb = defaultdict(list)   # (f_ter, b_ter) -> [ret]
    cond = defaultdict(list) # 'accel'/'cool' within high funding -> [ret]
    for asset, rs in sorted(by_asset.items()):
        fund = [x["features"]["last_funding_rate"]["value"] for x in rs]
        basis = [x["features"]["mark_index_basis_bps"]["value"] for x in rs]
        mark = [x["features"]["mark_price"]["value"] for x in rs]
        for i in range(LOOKBACK, len(rs) - HOLD):
            if mark[i] <= 0 or mark[i + HOLD] <= 0:
                continue
            ret = math.log(mark[i + HOLD] / mark[i])
            fw = fund[i - LOOKBACK:i]
            ft, f_pct = tercile(fw, fund[i])
            bw = basis[i - LOOKBACK:i]
            bt, _ = tercile(bw, basis[i])
            fb[(ft, bt)].append(ret)
            dm = fund[i] - fund[i - 5]
            mw = [fund[j] - fund[j - 5] for j in range(max(5, i - LOOKBACK), i)]
            if mw:
                m_pct = (sum(1 for v in mw if v < dm)
                         + 0.5 * sum(1 for v in mw if v == dm)) / len(mw)
                fmom.append((m_pct, ret))
                if f_pct >= 0.80:
                    cond["accel" if m_pct >= 0.5 else "cool"].append(ret)

    out = {"fmom_spearman": spearman([x[0] for x in fmom],
                                     [x[1] for x in fmom]),
           "fb_grid_mean_bps": {}, "fb_grid_n": {},
           "high_funding_momentum": {}, "p_values": []}
    for ft in range(3):
        for bt in range(3):
            c = fb.get((ft, bt), [])
            out["fb_grid_mean_bps"][f"f{ft}_b{bt}"] = \
                round(statistics.mean(c) * 1e4, 1) if c else None
            out["fb_grid_n"][f"f{ft}_b{bt}"] = len(c)
    c = welch(cond["accel"], cond["cool"])
    out["high_funding_momentum"] = c
    if out["fmom_spearman"]:
        out["p_values"].append(("fmom_spearman", out["fmom_spearman"]["p"]))
    if c:
        out["p_values"].append(("accel_vs_cool", c["p"]))
    # within-high-funding basis contrast
    c2 = welch(fb.get((2, 2), []), fb.get((2, 0), []))
    out["high_funding_basis_contrast"] = c2
    if c2:
        out["p_values"].append(("highF_basis_hi_vs_lo", c2["p"]))

    ps = sorted([(p, n) for n, p in out["p_values"]])
    m = len(ps)
    out["fdr_bh"] = [{"cell": n, "p": p,
                      "survives": p <= 0.05 * (i + 1) / m}
                     for i, (p, n) in enumerate(ps)]
    return {"schema_version": "nanojev-financial-signal-interactions-v1",
            "status": "measurement_complete",
            "scope": "funding momentum + funding x basis; measurement only",
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
