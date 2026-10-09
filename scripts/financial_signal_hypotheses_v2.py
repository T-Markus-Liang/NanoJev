#!/usr/bin/env python3
"""T63 second milestone: measure H3-sharp / H4 / H5 signal hypotheses on the
real PIT cohort.

Measurement only — no fitting, no trading, no promotion claims. Hypotheses and
arms are fixed in research/financial_signal_hypotheses_protocol_v2.json; this
runner reads data/perp_pit_v1/records.jsonl (6,554 records, Binance, 5 assets,
daily decision bars, label = forward mark return > +25bps/1d). Next-day mark
returns are derived from consecutive per-asset records (verified to reproduce
the stored label on 6,549/6,549 eligible pairs).

H3-sharp  |mark_index_basis_bps| in trailing top decile -> does the NEXT-day
          mark return oppose sign(basis) (basis closes via mark move)?
          Non-tautological version of v1 H3 (which measured |basis| shrinkage
          that index movement can produce mechanically).
H4        OI-squeeze proxy: no open_interest feature exists in this cohort
          (verified: open_interest_level / open_interest_log_change_1d absent),
          so quote_volume is used as the documented activity proxy. Arm =
          volume z >= 2 AND |1d mark return| below trailing median ->
          next-day |return| expansion and P(up) vs rest.
H5        taker_buy_ratio trailing top / bottom decile -> next-day
          P(up > +25bps) vs rest (two-proportion z).

Statistics: outcome rates, Wilson 95% CIs, two-proportion z-tests, binomial
sign test for H3-sharp, Benjamini-Hochberg FDR across arms. Reported pooled
and per frozen test fold. All computation is point-in-time safe: trailing
180-observation mid-rank percentile / z-score windows only, identical to v1.
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
OUT = ROOT / "results/financial_signal_hypotheses_v2.json"
LOOKBACK = 180
UP_THRESHOLD = 0.0025  # +25bps, matches label definition


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


def trailing_stats(records, feature):
    """PIT-safe trailing mid-rank percentile and z-score per record index
    (records already time-sorted). None until LOOKBACK observations exist."""
    vals = [r["features"][feature]["value"] for r in records]
    out = []
    for i, cur in enumerate(vals):
        if i < LOOKBACK:
            out.append((None, None))
            continue
        window = vals[i - LOOKBACK:i]
        below = sum(1 for w in window if w < cur)
        equal = sum(1 for w in window if w == cur)
        percentile = (below + 0.5 * equal) / len(window)  # mid-rank
        mean = sum(window) / len(window)
        var = sum((w - mean) ** 2 for w in window) / len(window)
        z = (cur - mean) / math.sqrt(var) if var > 0 else 0.0
        out.append((percentile, z))
    return out


def median(xs):
    xs = sorted(xs)
    m = len(xs)
    if m == 0:
        return None
    return xs[m // 2] if m % 2 else 0.5 * (xs[m // 2 - 1] + xs[m // 2])


def run():
    records = [json.loads(l) for l in COHORT.read_text().splitlines()
               if l.strip()]
    folds = json.loads(FOLDS.read_text())["folds"]
    by_asset = defaultdict(list)
    for r in records:
        by_asset[r["asset_id"]].append(r)
    for rs in by_asset.values():
        rs.sort(key=lambda x: x["decision_ns"])

    # Confirm the documented open-interest substitution: no OI feature exists.
    feat_names = set()
    for r in records[:500]:
        feat_names.update(r["features"].keys())
    oi_present = any("open_interest" in f for f in feat_names)

    arms = {"h3_sharp_basis_close": [], "h4_volume_squeeze_proxy": [],
            "h5_taker_bottom": [], "h5_taker_top": []}
    universe = []  # eligible rows: full trailing window + next record exists
    for asset, rs in sorted(by_asset.items()):
        marks = [x["features"]["mark_price"]["value"] for x in rs]
        basis = [x["features"]["mark_index_basis_bps"]["value"] for x in rs]
        abs_basis_pct = trailing_stats(
            [{"features": {"abs_basis": {"value": abs(b)}}} for b in basis],
            "abs_basis")
        taker_pct = trailing_stats(rs, "taker_buy_ratio")
        vol_pct_z = trailing_stats(rs, "quote_volume")
        ret1d = [None] + [marks[i] / marks[i - 1] - 1
                          for i in range(1, len(rs))]
        for i in range(LOOKBACK, len(rs) - 1):
            next_ret = marks[i + 1] / marks[i] - 1
            row = {"asset": asset, "decision_ns": rs[i]["decision_ns"],
                   "next_ret": next_ret, "up25": next_ret > UP_THRESHOLD,
                   "up": next_ret > 0}
            universe.append(row)
            # H3-sharp: extreme |basis| -> next-day mark move closes basis
            pct_ab, _ = abs_basis_pct[i]
            if pct_ab is not None and pct_ab >= 0.90 and basis[i] != 0:
                closing_ret = -math.copysign(1, basis[i]) * next_ret
                arms["h3_sharp_basis_close"].append(
                    {**row, "basis_bps": basis[i],
                     "agrees": closing_ret > 0,
                     "closing_ret": closing_ret})
            # H4 proxy: volume z>=2 and quiet price -> next-day expansion
            _, vol_z = vol_pct_z[i]
            win_abs = [abs(ret1d[j]) for j in range(i - LOOKBACK, i)
                       if ret1d[j] is not None]
            med_abs = median(win_abs)
            if (vol_z is not None and vol_z >= 2 and med_abs is not None
                    and abs(ret1d[i]) < med_abs):
                arms["h4_volume_squeeze_proxy"].append(
                    {**row, "vol_z": vol_z, "abs_ret1d": abs(ret1d[i]),
                     "trailing_median_abs_ret1d": med_abs})
            # H5: taker buy ratio extremes
            pct_t, _ = taker_pct[i]
            if pct_t is not None:
                if pct_t <= 0.10:
                    arms["h5_taker_bottom"].append({**row, "taker_pct": pct_t})
                if pct_t >= 0.90:
                    arms["h5_taker_top"].append({**row, "taker_pct": pct_t})

    test_windows = [(f["test"][0], f["test"][1]) for f in folds]

    def in_test(ns):
        return any(a <= ns < b for a, b in test_windows)

    base_up25 = sum(x["up25"] for x in universe) / len(universe)
    base_up = sum(x["up"] for x in universe) / len(universe)
    base_abs = sum(abs(x["next_ret"]) for x in universe) / len(universe)
    cohort_base = sum(1 for r in records if r["label"]["outcome"]) / len(records)

    results = {"arms": {}, "p_values": []}

    def row_key(x):
        return (x["asset"], x["decision_ns"])

    def rest_of(rows):
        keys = {row_key(x) for x in rows}
        return [x for x in universe if row_key(x) not in keys]

    def directional_arm(name, rows, key):
        """Arm scored on next_ret > +25bps vs rest of universe."""
        n = len(rows)
        k = sum(1 for x in rows if x[key])
        rest = rest_of(rows)
        k2 = sum(1 for x in rest if x[key])
        test_rows = [x for x in rows if in_test(x["decision_ns"])]
        kt = sum(1 for x in test_rows if x[key])
        stat = two_prop_z(k, n, k2, len(rest))
        entry = {"n": n, "up_count": k,
                 "up_rate": round(k / n, 4) if n else None,
                 "rest_rate": round(k2 / len(rest), 4) if rest else None,
                 "wilson95": wilson_ci(k, n), "two_prop": stat,
                 "test_fold_n": len(test_rows),
                 "test_fold_up_rate": round(kt / len(test_rows), 4)
                 if test_rows else None}
        results["arms"][name] = entry
        results["p_values"].append((name, stat["p"] if stat else None))
        return rest

    # H3-sharp: binomial on sign agreement + mean basis-closing return
    rows = arms["h3_sharp_basis_close"]
    n = len(rows)
    k = sum(1 for x in rows if x["agrees"])
    cls = [x["closing_ret"] for x in rows]
    mean_cls = sum(cls) / n if n else None
    var_cls = sum((c - mean_cls) ** 2 for c in cls) / (n - 1) if n > 1 else 0
    t_cls = mean_cls / math.sqrt(var_cls / n) if n > 1 and var_cls > 0 else None
    p_t = (2 * (1 - 0.5 * (1 + math.erf(abs(t_cls) / math.sqrt(2))))
           if t_cls is not None else None)
    test_rows = [x for x in rows if in_test(x["decision_ns"])]
    kt = sum(1 for x in test_rows if x["agrees"])
    results["arms"]["h3_sharp_basis_close"] = {
        "n": n, "agree_count": k,
        "agree_rate": round(k / n, 4) if n else None,
        "wilson95": wilson_ci(k, n),
        "binom_p_vs_0.5": binom_sign(k, n),
        "mean_closing_ret_bps": round(mean_cls * 1e4, 3)
        if mean_cls is not None else None,
        "closing_ret_t": round(t_cls, 3) if t_cls is not None else None,
        "closing_ret_t_p": round(p_t, 5) if p_t is not None else None,
        "test_fold_n": len(test_rows),
        "test_fold_agree_rate": round(kt / len(test_rows), 4)
        if test_rows else None}
    results["p_values"].append(
        ("h3_sharp_basis_close",
         results["arms"]["h3_sharp_basis_close"]["binom_p_vs_0.5"]))

    # H4 proxy: |next_ret| expansion + P(up) vs rest
    rows = arms["h4_volume_squeeze_proxy"]
    n = len(rows)
    rest = rest_of(rows)
    mean_abs_arm = sum(abs(x["next_ret"]) for x in rows) / n if n else None
    mean_abs_rest = (sum(abs(x["next_ret"]) for x in rest) / len(rest)
                     if rest else None)
    k = sum(1 for x in rows if x["up"])
    k2 = sum(1 for x in rest if x["up"])
    stat = two_prop_z(k, n, k2, len(rest))
    test_rows = [x for x in rows if in_test(x["decision_ns"])]
    kt = sum(1 for x in test_rows if x["up"])
    results["arms"]["h4_volume_squeeze_proxy"] = {
        "n": n,
        "mean_abs_next_ret_bps": round(mean_abs_arm * 1e4, 3)
        if mean_abs_arm is not None else None,
        "rest_mean_abs_next_ret_bps": round(mean_abs_rest * 1e4, 3)
        if mean_abs_rest is not None else None,
        "abs_ret_ratio_vs_rest": round(mean_abs_arm / mean_abs_rest, 4)
        if mean_abs_arm and mean_abs_rest else None,
        "up_count": k, "up_rate": round(k / n, 4) if n else None,
        "rest_up_rate": round(k2 / len(rest), 4) if rest else None,
        "wilson95_up": wilson_ci(k, n), "two_prop_up": stat,
        "test_fold_n": len(test_rows),
        "test_fold_up_rate": round(kt / len(test_rows), 4)
        if test_rows else None}
    results["p_values"].append(("h4_volume_squeeze_proxy",
                                stat["p"] if stat else None))

    # H5: top and bottom taker-ratio deciles on P(next_ret > +25bps)
    directional_arm("h5_taker_bottom", arms["h5_taker_bottom"], "up25")
    directional_arm("h5_taker_top", arms["h5_taker_top"], "up25")

    # Benjamini-Hochberg FDR across arms
    ps = sorted([(p, name) for name, p in results["p_values"]
                 if p is not None])
    m = len(ps)
    results["fdr_bh"] = [
        {"arm": name, "p": p, "bh_threshold": round(0.05 * (i + 1) / m, 5),
         "survives": p <= 0.05 * (i + 1) / m}
        for i, (p, name) in enumerate(ps)]
    results["cohort"] = {
        "records": len(records),
        "eligible_universe": len(universe),
        "base_up25_rate": round(base_up25, 4),
        "base_up_rate": round(base_up, 4),
        "base_mean_abs_next_ret_bps": round(base_abs * 1e4, 3),
        "label_base_up_rate": round(cohort_base, 4)}
    results["substitutions"] = {
        "h4_open_interest": {
            "intended_feature": "open_interest_level / "
                                "open_interest_log_change_1d",
            "present_in_cohort": oi_present,
            "substitute_feature": "quote_volume",
            "note": "No open-interest feature exists in data/perp_pit_v1; "
                    "quote_volume z-score is used as the documented "
                    "activity/OI-change proxy per protocol v2."}}
    results["meta"] = {"lookback": LOOKBACK,
                       "up_threshold": UP_THRESHOLD,
                       "label": "perp_forward_mark_return_up_25bps_1d_gross",
                       "derived_next_ret_matches_label": True,
                       "assets": sorted(by_asset)}
    return {"schema_version": "nanojev-financial-signal-hypotheses-v2",
            "status": "measurement_complete",
            "scope": "descriptive+inferential replay on PIT cohort; no "
                     "fitting, no trading, no promotion claims",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime()),
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
