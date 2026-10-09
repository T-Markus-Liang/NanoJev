#!/usr/bin/env python3
"""T65 round-3 signal replay: H6-H9 on the real PIT cohort.

Pre-registered per research/financial_signal_hypotheses_protocol_v3.json.
Direction priors corrected per the crypto-edge-search falsification meta-prior
("extreme funding persists" — fade was backwards):

H6  funding-FOLLOWING: last_funding_rate trailing-180 pct >= .80 -> P(up) ABOVE base
    (complement arm pct <= .20 -> P(up) BELOW base, reported for symmetry)
H7  abnormal-volume discount: quote_volume / trailing-30 median in top decile
    -> P(up) BELOW base (Garfinkel & Sokobin, Binance daily -0.5%/day)
H8  basis momentum: Δbasis over 5 bars, top/bottom tercile -> P(up) aligned
    with momentum direction (Chi et al., t=7.10 on dated futures)
H9  vol-regime strata: H6+H7 re-reported within realized_vol_24bar terciles
    (conditioning layer; not counted in the FDR family)

Stats: Wilson CI, two-prop z, BH-FDR over the 6 directional arms, per frozen
test-fold readout. Trailing-window PIT-safe only. Measurement only.
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
OUT = ROOT / "results/financial_signal_hypotheses_v3.json"
LOOKBACK = 180


def wilson_ci(k, n, z=1.96):
    if not n:
        return None
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def two_prop_z(k1, n1, k2, n2):
    if not n1 or not n2:
        return None
    p1, p2, pool = k1 / n1, k2 / n2, (k1 + k2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return None
    z = (p1 - p2) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {"z": round(z, 3), "p": round(p, 5)}


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

    arms = defaultdict(list)
    for asset, rs in sorted(by_asset.items()):
        fund = [x["features"]["last_funding_rate"]["value"] for x in rs]
        vol = [x["features"]["quote_volume"]["value"] for x in rs]
        basis = [x["features"]["mark_index_basis_bps"]["value"] for x in rs]
        rv = [x["features"]["realized_vol_24bar"]["value"] for x in rs]
        for i in range(LOOKBACK, len(rs)):
            r = rs[i]
            row = {"asset": asset, "decision_ns": r["decision_ns"],
                   "outcome": r["label"]["outcome"],
                   "vol_tercile": None}
            fw = fund[i - LOOKBACK:i]
            f_pct = (sum(1 for w in fw if w < fund[i])
                     + 0.5 * sum(1 for w in fw if w == fund[i])) / LOOKBACK
            # H9 strata: trailing vol tercile
            vw = sorted(rv[i - LOOKBACK:i])
            v_pct = (sum(1 for w in vw if w < rv[i])
                     + 0.5 * sum(1 for w in vw if w == rv[i])) / LOOKBACK
            row["vol_tercile"] = (0 if v_pct < 1 / 3 else
                                  1 if v_pct < 2 / 3 else 2)
            # H6 funding following
            if f_pct >= 0.80:
                arms["h6_follow_high"].append(row)
            elif f_pct <= 0.20:
                arms["h6_follow_low"].append(row)
            # H7 abnormal volume: current / trailing-30 median, top decile
            med = statistics.median(vol[i - 30:i])
            abn = vol[i] / med if med > 0 else 1.0
            aw = [vol[j] / statistics.median(vol[max(0, j - 30):j] or [1])
                  if statistics.median(vol[max(0, j - 30):j] or [1]) > 0 else 1.0
                  for j in range(max(30, i - LOOKBACK), i)]
            a_pct = (sum(1 for w in aw if w < abn)
                     + 0.5 * sum(1 for w in aw if w == abn)) / len(aw) if aw else 0.5
            if a_pct >= 0.90:
                arms["h7_abnvol_high"].append(row)
            elif a_pct <= 0.10:
                arms["h7_abnvol_low"].append(row)
            # H8 basis momentum over 5 bars, terciles vs trailing distribution
            if i >= 5:
                db = basis[i] - basis[i - 5]
                bw = [basis[j] - basis[j - 5]
                      for j in range(max(5, i - LOOKBACK), i)]
                if bw:
                    b_pct = (sum(1 for w in bw if w < db)
                             + 0.5 * sum(1 for w in bw if w == db)) / len(bw)
                    if b_pct >= 2 / 3:
                        arms["h8_basis_up"].append(row)
                    elif b_pct <= 1 / 3:
                        arms["h8_basis_down"].append(row)

    all_rows = [r for v in arms.values() for r in v]
    base_rate = sum(x["outcome"] for x in all_rows) / len(all_rows) if all_rows else 0

    results = {"arms": {}, "vol_strata": {}, "p_values": []}
    expect_down = {"h6_follow_low", "h7_abnvol_high", "h8_basis_down"}
    for name, rows in sorted(arms.items()):
        n = len(rows)
        k = sum(1 for x in rows if x["outcome"])
        rest = [r for r in all_rows if r not in rows]
        k2 = sum(1 for x in rest if x["outcome"])
        tr = [x for x in rows if in_test(x["decision_ns"])]
        kt = sum(1 for x in tr if x["outcome"])
        stat = two_prop_z(k, n, k2, len(rest))
        results["arms"][name] = {
            "n": n, "up_rate": round(k / n, 4) if n else None,
            "rest_rate": round(k2 / len(rest), 4) if rest else None,
            "expected_direction": "below_base" if name in expect_down
            else "above_base",
            "wilson95": wilson_ci(k, n), "two_prop": stat,
            "test_fold_n": len(tr),
            "test_fold_up_rate": round(kt / len(tr), 4) if tr else None}
        results["p_values"].append((name, stat["p"] if stat else None))
        # H9 strata
        for t in range(3):
            sub = [x for x in rows if x["vol_tercile"] == t]
            if sub:
                ks = sum(1 for x in sub if x["outcome"])
                results["vol_strata"][f"{name}@vol_t{t}"] = {
                    "n": len(sub), "up_rate": round(ks / len(sub), 4),
                    "wilson95": wilson_ci(ks, len(sub))}

    ps = sorted([(p, n) for n, p in results["p_values"] if p is not None])
    m = len(ps)
    results["fdr_bh"] = [{"arm": n, "p": p,
                          "bh_threshold": round(0.05 * (i + 1) / m, 5),
                          "survives": p <= 0.05 * (i + 1) / m}
                         for i, (p, n) in enumerate(ps)]
    results["cohort"] = {"records": len(records),
                         "base_up_rate": round(base_rate, 4)}
    return {"schema_version": "nanojev-financial-signal-hypotheses-v3",
            "status": "measurement_complete",
            "scope": "descriptive+inferential replay; no fitting/trading/claims",
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
