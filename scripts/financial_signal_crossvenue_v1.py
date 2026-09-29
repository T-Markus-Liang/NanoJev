#!/usr/bin/env python3
"""T80: Binance<->Bybit funding spread as a signal (local data only).

For each cohort record: binance_rate = last_funding_rate (settled, PIT);
bybit_rate = last Bybit funding settlement at or before the record's
decision_ns (8h UTC grid, loaded from data/venue_perp_v1/bybit/*.funding.json,
hash-pinned fetch). spread_bps = (binance - bybit) * 1e4.

Arms on 5d forward mark log return:
  spread_pct   trailing-180 mid-rank pct of spread -> Spearman + decile arms
  f2b2_spread  inside the confirmed cell (funding_pct>=0.80 AND basis>=0.66):
               spread_pct >= 0.5 vs < 0.5 — does relative crowding add signal?

Welch t + Spearman + BH-FDR; per frozen test-fold readout. Measurement only —
local pinned files, no network, no fitting.
"""

import argparse
import hashlib
import json
import math
import statistics
import time
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_v1/records.jsonl"
BYBIT = ROOT / "data/venue_perp_v1/bybit"
FOLDS = ROOT / "research/financial_r1_pit_validator_core_v2.json"
OUT = ROOT / "results/financial_signal_crossvenue_v1.json"
LOOKBACK, HOLD = 180, 5


def pct(w, x):
    return (sum(1 for v in w if v < x)
            + 0.5 * sum(1 for v in w if v == x)) / len(w)


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
    return {"n_a": len(a), "n_b": len(b), "mean_a_bps": round(ma * 1e4, 1),
            "mean_b_bps": round(mb * 1e4, 1), "t": round(t, 3),
            "p": round(p, 6)}


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


def load_bybit_funding():
    """symbol -> sorted [(settlement_ms, rate)]."""
    out = {}
    for f in BYBIT.glob("*.funding.json"):
        rows = json.loads(f.read_text())
        sym = f.name.split(".")[0]
        pts = sorted((int(r["fundingRateTimestamp"]), float(r["fundingRate"]))
                     for r in rows)
        out[sym] = pts
    return out


def bybit_rate_at(pts, ns):
    """Last settlement at or before ns (ns is epoch ns; pts in ms)."""
    ms = ns // 1_000_000
    i = bisect_right([p[0] for p in pts], ms) - 1
    return pts[i][1] if i >= 0 else None


def run():
    records = [json.loads(l) for l in COHORT.read_text().splitlines() if l.strip()]
    folds = json.loads(FOLDS.read_text())["folds"]
    windows = [(f["test"][0], f["test"][1]) for f in folds]
    in_test = lambda ns: any(a <= ns < b for a, b in windows)
    byb = load_bybit_funding()
    ts_index = {s: [p[0] for p in v] for s, v in byb.items()}

    def rate_at(sym, ns):
        pts = ts_index.get(sym)
        if not pts:
            return None
        i = bisect_right(pts, ns // 1_000_000) - 1
        return byb[sym][i][1] if i >= 0 else None

    by_asset = defaultdict(list)
    for r in records:
        by_asset[r["asset_id"]].append(r)
    for rs in by_asset.values():
        rs.sort(key=lambda x: x["decision_ns"])

    spread_rows = []          # (spread_pct, ret, ns, in_f2b2)
    matched = 0
    for asset, rs in sorted(by_asset.items()):
        sym = asset.replace("-PERP", "")
        fund = [x["features"]["last_funding_rate"]["value"] for x in rs]
        basis = [x["features"]["mark_index_basis_bps"]["value"] for x in rs]
        mark = [x["features"]["mark_price"]["value"] for x in rs]
        spread = []
        for i, r in enumerate(rs):
            br = rate_at(sym, r["decision_ns"])
            spread.append(None if br is None else (fund[i] - br) * 1e4)
        for i in range(LOOKBACK, len(rs) - HOLD):
            if spread[i] is None or mark[i] <= 0 or mark[i + HOLD] <= 0:
                continue
            sw = [s for s in spread[i - LOOKBACK:i] if s is not None]
            if len(sw) < 100:
                continue
            s_pct = pct(sw, spread[i])
            f_pct = pct(fund[i - LOOKBACK:i], fund[i])
            b_pct = pct(basis[i - LOOKBACK:i], basis[i])
            ret = math.log(mark[i + HOLD] / mark[i])
            matched += 1
            spread_rows.append({"s_pct": s_pct, "ret": ret,
                                "ns": rs[i]["decision_ns"],
                                "f2b2": f_pct >= 0.80 and b_pct >= 0.66})

    out = {"matched_records": matched, "p_values": []}
    xs = [r["s_pct"] for r in spread_rows]
    ys = [r["ret"] for r in spread_rows]
    out["spread_spearman"] = spearman(xs, ys)
    if out["spread_spearman"]:
        out["p_values"].append(("spread_spearman", out["spread_spearman"]["p"]))
    hi = [r["ret"] for r in spread_rows if r["s_pct"] >= 0.9]
    lo = [r["ret"] for r in spread_rows if r["s_pct"] <= 0.1]
    mid = [r["ret"] for r in spread_rows if 0.1 < r["s_pct"] < 0.9]
    out["deciles"] = {"hi": welch(hi, mid), "lo": welch(lo, mid)}
    for k in ("hi", "lo"):
        if out["deciles"][k]:
            out["p_values"].append((f"decile_{k}", out["deciles"][k]["p"]))
    inside = [r for r in spread_rows if r["f2b2"]]
    out["f2b2_spread_contrast"] = welch(
        [r["ret"] for r in inside if r["s_pct"] >= 0.5],
        [r["ret"] for r in inside if r["s_pct"] < 0.5])
    if out["f2b2_spread_contrast"]:
        out["p_values"].append(("f2b2_spread", out["f2b2_spread_contrast"]["p"]))
    out["test_fold_spearman"] = spearman(
        [r["s_pct"] for r in spread_rows if in_test(r["ns"])],
        [r["ret"] for r in spread_rows if in_test(r["ns"])])
    ps = sorted([(p, n) for n, p in out["p_values"]])
    m = len(ps)
    out["fdr_bh"] = [{"cell": n, "p": p,
                      "survives": p <= 0.05 * (i + 1) / m}
                     for i, (p, n) in enumerate(ps)]
    return {"schema_version": "nanojev-financial-signal-crossvenue-v1",
            "status": "measurement_complete",
            "scope": "Binance-Bybit funding spread, local pinned data; "
                    "measurement only",
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
