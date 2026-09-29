#!/usr/bin/env python3
"""T72: open-interest arms on the v2 cohort (3,266 records with OI features).

Arms on 5-day forward mark log return, trailing-180 mid-rank percentiles:
  oi_pct      open_interest_log_change_1d pct -> does OI expansion predict
  crowding    OI-change tercile x funding tercile grid; the "OI up + funding
              high" cell is the crowded-long squeeze candidate
  oi_cond     within high funding: OI-expanding vs OI-contracting contrast

Welch t + Spearman, BH-FDR. Measurement only.
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
COHORT = ROOT / "data/perp_pit_v2/records.jsonl"
OUT = ROOT / "results/financial_signal_oi_v1.json"
LOOKBACK, HOLD = 180, 5


def pct(w, x):
    return (sum(1 for v in w if v < x) + 0.5 * sum(1 for v in w if v == x)) / len(w)


def ter(w, x):
    p = pct(w, x)
    return (0 if p < 1 / 3 else (1 if p < 2 / 3 else 2)), p


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


def run():
    records = [json.loads(l) for l in COHORT.read_text().splitlines() if l.strip()]
    by_asset = defaultdict(list)
    for r in records:
        by_asset[r["asset_id"]].append(r)
    for rs in by_asset.values():
        rs.sort(key=lambda x: x["decision_ns"])

    oi_rows, crowd, cond = [], defaultdict(list), defaultdict(list)
    for asset, rs in sorted(by_asset.items()):
        fund = [x["features"]["last_funding_rate"]["value"] for x in rs]
        oi = [x["features"]["open_interest_log_change_1d"]["value"] for x in rs]
        mark = [x["features"]["mark_price"]["value"] for x in rs]
        for i in range(LOOKBACK, len(rs) - HOLD):
            if mark[i] <= 0 or mark[i + HOLD] <= 0:
                continue
            ret = math.log(mark[i + HOLD] / mark[i])
            ot, o_pct = ter(oi[i - LOOKBACK:i], oi[i])
            ft, f_pct = ter(fund[i - LOOKBACK:i], fund[i])
            oi_rows.append((o_pct, ret))
            crowd[(ft, ot)].append(ret)
            if f_pct >= 0.80:
                cond["oi_hi" if o_pct >= 0.5 else "oi_lo"].append(ret)

    out = {"oi_spearman": spearman([x[0] for x in oi_rows],
                                   [x[1] for x in oi_rows]),
           "crowd_grid_mean_bps": {}, "crowd_grid_n": {}, "p_values": []}
    for ft in range(3):
        for ot in range(3):
            c = crowd.get((ft, ot), [])
            out["crowd_grid_mean_bps"][f"f{ft}_o{ot}"] = \
                round(statistics.mean(c) * 1e4, 1) if c else None
            out["crowd_grid_n"][f"f{ft}_o{ot}"] = len(c)
    c = welch(cond["oi_hi"], cond["oi_lo"])
    out["high_funding_oi_contrast"] = c
    if out["oi_spearman"]:
        out["p_values"].append(("oi_spearman", out["oi_spearman"]["p"]))
    if c:
        out["p_values"].append(("highF_oi_hi_vs_lo", c["p"]))
    c2 = welch(crowd.get((2, 2), []), crowd.get((2, 0), []))
    out["crowd_f2_oi_hi_vs_lo"] = c2
    if c2:
        out["p_values"].append(("crowd_f2_oi_hi_vs_lo", c2["p"]))
    ps = sorted([(p, n) for n, p in out["p_values"]])
    m = len(ps)
    out["fdr_bh"] = [{"cell": n, "p": p,
                      "survives": p <= 0.05 * (i + 1) / m}
                     for i, (p, n) in enumerate(ps)]
    return {"schema_version": "nanojev-financial-signal-oi-v1",
            "status": "measurement_complete",
            "scope": "OI arms on v2 cohort; measurement only",
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
