#!/usr/bin/env python3
"""T63 first milestone: replay H1-H3 signal hypotheses on the real PIT cohort.

Measurement only — no fitting, no trading, no promotion claims. Hypotheses and
arms are fixed in research/financial_signal_hypotheses_protocol_v1.json; this
runner reads data/perp_pit_v1/records.jsonl (6,554 records, Binance, 5 assets,
daily decision bars, label = forward mark return > +25bps/1d).

H1  extreme negative funding (bottom trailing-180d decile) -> P(up) above base
H2  funding z-score contrarian fade (|z|>=2), carry stripped (directional only)
H3  extreme |mark-index basis| -> next-day |basis| convergence

Statistics: outcome rates, Wilson 95% CIs, two-proportion z-tests, sign/binomial
test for H3, Benjamini-Hochberg FDR across arms. Reported pooled and per frozen
test fold. All computation is point-in-time safe: trailing windows only.
"""

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_v1/records.jsonl"
FOLDS = ROOT / "research/financial_r1_pit_validator_core_v2.json"
OUT = ROOT / "results/financial_signal_hypotheses_v1.json"
LOOKBACK = 180
NS_PER_DAY = 86_400_000_000_000


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(centre - half, 4), round(centre + half, 4)]


def two_prop_z(k1, n1, k2, n2):
    if not n1 or not n2:
        return None
    p1, p2, pooled = k1 / n1, k2 / n2, (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return None
    z = (p1 - p2) / se
    # two-sided normal approximation p-value
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {"z": round(z, 3), "p": round(p, 5)}


def binom_sign(k, n):
    """One-sided binomial P(X >= k) under p=0.5."""
    if n == 0 or n > 10000:
        return None
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * 0.5 ** n
    return round(total, 5)


def funding_series(records):
    """PIT-safe trailing stats per record index (already time-sorted)."""
    out = []
    for i, r in enumerate(records):
        if i < LOOKBACK:
            out.append((None, None))
            continue
        window = [x["features"]["last_funding_rate"]["value"]
                  for x in records[i - LOOKBACK:i]]
        cur = r["features"]["last_funding_rate"]["value"]
        below = sum(1 for w in window if w < cur)
        equal = sum(1 for w in window if w == cur)
        percentile = (below + 0.5 * equal) / len(window)  # mid-rank
        mean = sum(window) / len(window)
        var = sum((w - mean) ** 2 for w in window) / len(window)
        z = (cur - mean) / math.sqrt(var) if var > 0 else 0.0
        out.append((percentile, z))
    return out


def run():
    records = [json.loads(l) for l in COHORT.read_text().splitlines() if l.strip()]
    folds = json.loads(FOLDS.read_text())["folds"]
    by_asset = defaultdict(list)
    for r in records:
        by_asset[r["asset_id"]].append(r)
    for rs in by_asset.values():
        rs.sort(key=lambda x: x["decision_ns"])

    arms = {"h1_tail": [], "h2_fade_long": [], "h2_fade_short": [],
            "h3_basis_extreme": []}
    meta = {}
    for asset, rs in sorted(by_asset.items()):
        stats = funding_series(rs)
        for i, (pct, z) in enumerate(stats):
            if pct is None:
                continue
            r = rs[i]
            row = {"asset": asset, "decision_ns": r["decision_ns"],
                   "outcome": r["label"]["outcome"], "pct": pct, "z": z,
                   "basis_bps": r["features"]["mark_index_basis_bps"]["value"]}
            if pct <= 0.10:
                arms["h1_tail"].append(row)
            if z <= -2:
                arms["h2_fade_long"].append(row)
            if z >= 2:
                arms["h2_fade_short"].append(row)
        # H3 uses consecutive-record basis convergence
        basis = [abs(x["features"]["mark_index_basis_bps"]["value"]) for x in rs]
        if basis:
            hi = sorted(basis)[int(0.9 * len(basis))]
            for i in range(len(rs) - 1):
                if basis[i] >= hi and basis[i] > 0:
                    arms["h3_basis_extreme"].append(
                        {"asset": asset, "decision_ns": rs[i]["decision_ns"],
                         "converged": basis[i + 1] < basis[i],
                         "basis0": basis[i], "basis1": basis[i + 1]})

    test_windows = [(f["test"][0], f["test"][1]) for f in folds]

    def in_test(ns):
        return any(a <= ns < b for a, b in test_windows)

    base_n = sum(len(v) for v in arms.values())  # for context only
    all_rows = [r for v in (arms["h1_tail"], arms["h2_fade_long"],
                            arms["h2_fade_short"]) for r in v]
    base_rate = sum(x["outcome"] for x in all_rows) / len(all_rows) \
        if all_rows else 0
    cohort_base = sum(1 for r in records if r["label"]["outcome"]) / len(records)

    results = {"arms": {}, "p_values": []}
    for name, rows in arms.items():
        if name == "h3_basis_extreme":
            n = len(rows)
            k = sum(1 for x in rows if x["converged"])
            test_rows = [x for x in rows if in_test(x["decision_ns"])]
            kt = sum(1 for x in test_rows if x["converged"])
            entry = {"n": n, "converged": k, "rate": round(k / n, 4) if n else None,
                     "wilson95": wilson_ci(k, n),
                     "binom_p_vs_0.5": binom_sign(k, n),
                     "test_fold_n": len(test_rows),
                     "test_fold_rate": round(kt / len(test_rows), 4)
                     if test_rows else None}
            results["arms"][name] = entry
            results["p_values"].append(("h3", entry["binom_p_vs_0.5"]))
            continue
        n = len(rows)
        k = sum(1 for x in rows if x["outcome"])
        rest = [r for r in all_rows if r not in rows]
        k2 = sum(1 for x in rest if x["outcome"])
        test_rows = [x for x in rows if in_test(x["decision_ns"])]
        kt = sum(1 for x in test_rows if x["outcome"])
        stat = two_prop_z(k, n, k2, len(rest))
        results["arms"][name] = {
            "n": n, "up_count": k, "up_rate": round(k / n, 4) if n else None,
            "rest_rate": round(k2 / len(rest), 4) if rest else None,
            "wilson95": wilson_ci(k, n), "two_prop": stat,
            "test_fold_n": len(test_rows),
            "test_fold_up_rate": round(kt / len(test_rows), 4)
            if test_rows else None}
        results["p_values"].append((name, stat["p"] if stat else None))

    # Benjamini-Hochberg FDR across arms
    ps = sorted([(p, name) for name, p in results["p_values"] if p is not None])
    m = len(ps)
    results["fdr_bh"] = [
        {"arm": name, "p": p, "bh_threshold": round(0.05 * (i + 1) / m, 5),
         "survives": p <= 0.05 * (i + 1) / m}
        for i, (p, name) in enumerate(ps)]
    results["cohort"] = {"records": len(records), "base_up_rate":
                         round(cohort_base, 4),
                         "arm_rows_total": len(all_rows)}
    results["meta"] = {"lookback": LOOKBACK, "label":
                       "perp_forward_mark_return_up_25bps_1d_gross",
                       "assets": sorted(by_asset)}
    return {"schema_version": "nanojev-financial-signal-hypotheses-v1",
            "status": "measurement_complete",
            "scope": "descriptive+inferential replay on PIT cohort; no fitting, "
                     "no trading, no promotion claims",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    receipt = run()
    encoded = json.dumps(receipt, indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(encoded)
    print(json.dumps({"output": str(args.output),
                      "sha256": hashlib.sha256(encoded.encode()).hexdigest()},
                     indent=2))


if __name__ == "__main__":
    main()
