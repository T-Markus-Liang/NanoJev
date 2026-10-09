#!/usr/bin/env python3
"""T79: regime-conditioning arms for the funding×basis signal.

Literature-ranked gates (see docs/TRACK_B_SIGNAL_HYPOTHESES_V1.md round-2):
  A  own-asset 20d trend sign (FRL-2025: crypto momentum UP-UP only)
  B  BTC 20d trend sign applied to all assets (market-wide risk-on)
  C  vol x trend 2x2 within f2b2 (Daniel-Moskowitz panic-state test)
  D  funding-streak >=3d within f2b2 (regime vs spike)
  E  diagnostic: f2b2 split funding_pct>=0.95 vs 0.80-0.95 (non-monotone check)

Base cell f2b2 = funding_pct>=0.80 AND basis_pct>=0.66 (the confirmed cell).
Target: 5d forward mark log return. Welch t per contrast; BH-FDR over the
family; per-fold sign-consistency reported (the instability being fixed).
PIT-safe trailing windows only. Measurement only.
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
OUT = ROOT / "results/financial_signal_regime_v1.json"
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

    # BTC 20d trend sign at each decision_ns (BTC rows only)
    btc = by_asset["BTCUSDT-PERP"]
    btc_trend = {}
    bm = [x["features"]["mark_price"]["value"] for x in btc]
    for i in range(20, len(btc)):
        if bm[i] > 0 and bm[i - 20] > 0:
            btc_trend[btc[i]["decision_ns"]] = bm[i] / bm[i - 20] - 1 > 0

    arms = defaultdict(list)
    for asset, rs in sorted(by_asset.items()):
        fund = [x["features"]["last_funding_rate"]["value"] for x in rs]
        basis = [x["features"]["mark_index_basis_bps"]["value"] for x in rs]
        rv = [x["features"]["realized_vol_24bar"]["value"] for x in rs]
        mark = [x["features"]["mark_price"]["value"] for x in rs]
        f_streak = 0
        for i in range(len(rs)):
            if i >= LOOKBACK:
                p = pct(fund[i - LOOKBACK:i], fund[i])
                f_streak = f_streak + 1 if p >= 0.80 else 0
            else:
                f_streak = 0
            if i < LOOKBACK or i >= len(rs) - HOLD:
                continue
            if mark[i] <= 0 or mark[i + HOLD] <= 0:
                continue
            ret = math.log(mark[i + HOLD] / mark[i])
            f_pct = pct(fund[i - LOOKBACK:i], fund[i])
            b_pct = pct(basis[i - LOOKBACK:i], basis[i])
            v_pct = pct(rv[i - LOOKBACK:i], rv[i])
            if not (f_pct >= 0.80 and b_pct >= 0.66):
                continue
            ns = rs[i]["decision_ns"]
            trend20 = (mark[i] / mark[i - 20] - 1 > 0
                       if i >= 20 and mark[i - 20] > 0 else None)
            btc_up = btc_trend.get(ns)
            row = {"asset": asset, "ns": ns, "ret": ret}
            # A: own-asset trend
            if trend20 is not None:
                arms["A_up" if trend20 else "A_down"].append(row)
            # B: BTC trend gate
            if btc_up is not None:
                arms["B_btc_up" if btc_up else "B_btc_down"].append(row)
            # C: vol x trend 2x2
            if trend20 is not None and v_pct >= 0.66:
                arms["C_highvol_up" if trend20 else "C_highvol_down"].append(row)
            # D: funding streak
            arms["D_streak" if f_streak >= 3 else "D_spike"].append(row)
            # E: extreme vs moderate funding
            arms["E_extreme" if f_pct >= 0.95 else "E_moderate"].append(row)

    pairs = [("A_up", "A_down"), ("B_btc_up", "B_btc_down"),
             ("C_highvol_up", "C_highvol_down"), ("D_streak", "D_spike"),
             ("E_extreme", "E_moderate")]
    out = {"arms": {}, "contrasts": {}, "per_fold_sign": {}, "p_values": []}
    for a, b in pairs:
        ra, rb = arms[a], arms[b]
        for name, rows in ((a, ra), (b, rb)):
            n = len(rows)
            k5 = [x["ret"] for x in rows]
            out["arms"][name] = {
                "n": n,
                "mean_5d_bps": round(statistics.mean(k5) * 1e4, 1) if n else None,
                "per_fold_mean_bps": {
                    f"f{fi}": round(statistics.mean(
                        [x["ret"] for x in rows
                         if f["test"][0] <= x["ns"] < f["test"][1]]
                        or [0]) * 1e4, 1)
                    for fi, f in enumerate(folds)}}
        c = welch([x["ret"] for x in ra], [x["ret"] for x in rb])
        out["contrasts"][f"{a}_vs_{b}"] = c
        if c:
            out["p_values"].append((f"{a}_vs_{b}", c["p"]))
    # fold sign consistency for arm A (the top-prior gate)
    for fi, f in enumerate(folds):
        up = [x["ret"] for x in arms["A_up"]
              if f["test"][0] <= x["ns"] < f["test"][1]]
        dn = [x["ret"] for x in arms["A_down"]
              if f["test"][0] <= x["ns"] < f["test"][1]]
        if len(up) >= 5 and len(dn) >= 5:
            out["per_fold_sign"][f"f{fi}"] = {
                "A_up": round(statistics.mean(up) * 1e4, 1),
                "A_down": round(statistics.mean(dn) * 1e4, 1),
                "n_up": len(up), "n_down": len(dn)}
    ps = sorted([(p, n) for n, p in out["p_values"]])
    m = len(ps)
    out["fdr_bh"] = [{"contrast": n, "p": p,
                      "survives": p <= 0.05 * (i + 1) / m}
                     for i, (p, n) in enumerate(ps)]
    return {"schema_version": "nanojev-financial-signal-regime-v1",
            "status": "measurement_complete",
            "scope": "regime-gate contrasts on confirmed f2b2 cell; "
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
