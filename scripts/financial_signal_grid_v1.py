#!/usr/bin/env python3
"""T68: funding_pct x realized_vol tercile conditional grid + per-asset split.

Question: is the funding-following edge conditional on the vol regime, and is
it an asset-specific artifact or a pooled effect? Long-side only (short arm
was cut in T67). Target: 5-day forward mark log return, same PIT windows.

Grid: funding tercile (3) x vol tercile (3) -> mean 5d return + n per cell;
contrast = top-vs-bottom funding tercile within each vol tercile (Welch t).
Per-asset: same contrast pooled within each asset.
BH-FDR over the 3 vol-strata contrasts + 5 per-asset contrasts = 8 cells.
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
OUT = ROOT / "results/financial_signal_grid_v1.json"
LOOKBACK, HOLD = 180, 5


def tercile(vals, x):
    below = sum(1 for v in vals if v < x)
    equal = sum(1 for v in vals if v == x)
    p = (below + 0.5 * equal) / len(vals)
    return 0 if p < 1 / 3 else (1 if p < 2 / 3 else 2)


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
    return {"mean_a_bps": round(ma * 1e4, 1), "mean_b_bps": round(mb * 1e4, 1),
            "t": round(t, 3), "p": round(p, 5)}


def run():
    records = [json.loads(l) for l in COHORT.read_text().splitlines() if l.strip()]
    by_asset = defaultdict(list)
    for r in records:
        by_asset[r["asset_id"]].append(r)
    for rs in by_asset.values():
        rs.sort(key=lambda x: x["decision_ns"])

    grid = defaultdict(list)          # (f_ter, v_ter) -> [ret]
    per_asset = defaultdict(lambda: defaultdict(list))  # asset -> f_ter -> [ret]
    for asset, rs in sorted(by_asset.items()):
        fund = [x["features"]["last_funding_rate"]["value"] for x in rs]
        rv = [x["features"]["realized_vol_24bar"]["value"] for x in rs]
        mark = [x["features"]["mark_price"]["value"] for x in rs]
        for i in range(LOOKBACK, len(rs) - HOLD):
            if mark[i] <= 0 or mark[i + HOLD] <= 0:
                continue
            ft = tercile(fund[i - LOOKBACK:i], fund[i])
            vt = tercile(rv[i - LOOKBACK:i], rv[i])
            ret = math.log(mark[i + HOLD] / mark[i])
            grid[(ft, vt)].append(ret)
            per_asset[asset][ft].append(ret)

    out = {"grid_mean_bps": {}, "grid_n": {}, "vol_contrasts": {},
           "per_asset_contrast": {}, "p_values": []}
    for ft in range(3):
        for vt in range(3):
            cell = grid.get((ft, vt), [])
            key = f"f{ft}_v{vt}"
            out["grid_mean_bps"][key] = round(statistics.mean(cell) * 1e4, 1) \
                if cell else None
            out["grid_n"][key] = len(cell)
    for vt in range(3):
        c = welch(grid.get((2, vt), []), grid.get((0, vt), []))
        out["vol_contrasts"][f"vol_t{vt}"] = c
        if c:
            out["p_values"].append((f"vol_t{vt}", c["p"]))
    for asset, d in sorted(per_asset.items()):
        c = welch(d.get(2, []), d.get(0, []))
        out["per_asset_contrast"][asset] = c
        if c:
            out["p_values"].append((f"asset:{asset}", c["p"]))

    ps = sorted([(p, n) for n, p in out["p_values"]])
    m = len(ps)
    out["fdr_bh"] = [{"cell": n, "p": p,
                      "survives": p <= 0.05 * (i + 1) / m}
                     for i, (p, n) in enumerate(ps)]
    return {"schema_version": "nanojev-financial-signal-grid-v1",
            "status": "measurement_complete",
            "scope": "funding x vol conditional grid; measurement only",
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
