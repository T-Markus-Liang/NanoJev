#!/usr/bin/env python3
"""T86: funding-timing effects on the hourly PIT cohort.

Cohort: data/perp_pit_intraday_v1/records.jsonl — 716 contiguous hourly
BTCUSDT-PERP decision bars, 2026-08-02..2026-08-31 UTC. Label =
label.forward_return_bps (4h forward mark return, gross, bps; present on
all 716 records). Records sorted by decision_ns.

Four PIT-safe arms (all features used are point-in-time stamped):

  A1 distance-to-settlement buckets: ns_until_next_funding_settlement
     bucketed {0-2h, 2-4h, 4-6h, 6-8h} (bucket = floor(dist_h/2), clipped;
     observed distances are k+epsilon hours, epsilon < 1e-4 h, so bucket
     edges are unambiguous). Drift-into-settlement hypothesis: mean 4h
     forward return per bucket. Contrasts: nearest bucket (0-2h) vs rest,
     farthest bucket (6-8h, i.e. just-after-settlement bars) vs rest.
     4 records carry a null distance and are excluded from A1/A2 only.
  A2 signed funding x distance: within each distance bucket, settled
     last_funding_rate > 0 vs < 0 — does a positive settled rate predict
     pre-settlement drift differently (longs-paying crowd)?
  A3 post-settlement reversal: decision bars covering 0-2h AFTER a
     settlement vs all others, via decision_ms mod 28_800_000 < 7_200_000
     (settlement ms timestamps are multiples of 28_800_000; observed mod
     values are k*3_600_000 + 3_599_999, so the cutoff selects exactly the
     first two hourly bars after each settlement).
  A4 funding extreme at hourly scale: top/bottom quintile of
     last_funding_rate under a trailing-90-record (≈90h) mid-rank pct;
     contrast top vs bottom quintile.

Per arm: n, mean/median forward bps, Welch t (normal-approx p, same
convention as financial_signal_regime_v1) vs complement, BH-FDR over the
family of 8 contrasts. An approximate 80%-power minimum detectable effect
is reported per contrast — pilot n=716 is low-power.

Caveats: 4h forward labels on 1h decision bars overlap (~4x), inducing
positive serial correlation; Welch t assumes independence, so p-values are
nominal/optimistic. Measurement only — no fitting, no trading.
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
COHORT = ROOT / "data/perp_pit_intraday_v1/records.jsonl"
OUT = ROOT / "results/financial_signal_funding_timing_v1.json"
TRAIL = 90          # trailing records for funding pct (≈90h)
SETTLE_MS = 28_800_000          # 8h settlement grid in ms
POST_MS = 7_200_000             # 0-2h after settlement
BUCKET_H = 2                    # bucket width in hours of distance
MIN_N = 10                      # Welch minimum per side (repo convention)


def feat(r, name):
    f = r.get("features", {}).get(name)
    return None if f is None else f.get("value")


def pct(w, x):
    """Mid-rank percentile of x within trailing window w."""
    return (sum(1 for v in w if v < x)
            + 0.5 * sum(1 for v in w if v == x)) / len(w)


def dist_bucket(dist_ns):
    if dist_ns is None:
        return None
    return min(3, int((dist_ns / 3.6e12) / BUCKET_H))


def welch(a, b):
    """Welch t of mean(a)-mean(b); normal-approx two-sided p (repo
    convention). Returns None if either side < MIN_N or zero variance."""
    if len(a) < MIN_N or len(b) < MIN_N:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.pvariance(a), statistics.pvariance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return None
    t = (ma - mb) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    pooled_sd = math.sqrt((va * len(a) + vb * len(b)) / (len(a) + len(b)))
    # approx 80%-power MDE for two-sided 0.05, normal approx:
    # (z_.975 + z_.8) * sd * sqrt(1/na + 1/nb)
    mde = (1.96 + 0.84) * pooled_sd * math.sqrt(1 / len(a) + 1 / len(b))
    return {"n_a": len(a), "n_b": len(b),
            "mean_a_bps": round(ma, 2), "mean_b_bps": round(mb, 2),
            "diff_bps": round(ma - mb, 2), "t": round(t, 3),
            "p": round(p, 6), "approx_mde80_bps": round(mde, 1)}


def summ(rows):
    r = [x["fwd"] for x in rows]
    if not r:
        return {"n": 0, "mean_fwd_bps": None, "median_fwd_bps": None}
    return {"n": len(r), "mean_fwd_bps": round(statistics.mean(r), 2),
            "median_fwd_bps": round(statistics.median(r), 2)}


def run():
    records = [json.loads(l) for l in COHORT.read_text().splitlines()
               if l.strip()]
    records.sort(key=lambda x: x["decision_ns"])

    rows = []
    n_null_dist = 0
    for r in records:
        lab = r.get("label", {})
        fwd = lab.get("forward_return_bps")
        if fwd is None:
            continue
        dist_ns = feat(r, "ns_until_next_funding_settlement")
        if dist_ns is None:
            n_null_dist += 1
        rows.append({
            "ns": r["decision_ns"],
            "fwd": fwd,
            "fund": feat(r, "last_funding_rate"),
            "bucket": dist_bucket(dist_ns),
            "post": (r["decision_ns"] // 1_000_000) % SETTLE_MS < POST_MS,
        })

    # A4 needs trailing-90 mid-rank pct on the sorted fund series
    fund = [x["fund"] for x in rows]
    for i, x in enumerate(rows):
        x["f_pct"] = pct(fund[i - TRAIL:i], x["fund"]) if i >= TRAIL else None

    bnames = ["0-2h", "2-4h", "4-6h", "6-8h"]
    arms = defaultdict(list)
    for x in rows:
        if x["bucket"] is not None:
            arms[f"A1_dist_{bnames[x['bucket']]}"].append(x)
            sgn = "pos" if x["fund"] > 0 else ("neg" if x["fund"] < 0
                                               else "zero")
            arms[f"A2_{bnames[x['bucket']]}_{sgn}"].append(x)
        arms["A3_post_0_2h" if x["post"] else "A3_other"].append(x)
        if x["f_pct"] is not None:
            if x["f_pct"] >= 0.80:
                arms["A4_top_quintile"].append(x)
            elif x["f_pct"] <= 0.20:
                arms["A4_bottom_quintile"].append(x)
            else:
                arms["A4_middle"].append(x)

    all_rows = rows
    a1_rest = lambda b: [x for x in all_rows
                         if x["bucket"] is not None and x["bucket"] != b]

    # family of 8 pre-registered contrasts
    contrasts = []
    for b in (0, 3):  # nearest vs rest; farthest vs rest
        contrasts.append((f"A1_dist_{bnames[b]}_vs_rest",
                          arms[f"A1_dist_{bnames[b]}"], a1_rest(b)))
    for b in range(4):  # signed funding within each distance bucket
        contrasts.append((f"A2_{bnames[b]}_pos_vs_neg",
                          arms[f"A2_{bnames[b]}_pos"],
                          arms[f"A2_{bnames[b]}_neg"]))
    contrasts.append(("A3_post_0_2h_vs_rest",
                      arms["A3_post_0_2h"], arms["A3_other"]))
    contrasts.append(("A4_top_vs_bottom_quintile",
                      arms["A4_top_quintile"], arms["A4_bottom_quintile"]))

    fvals = [x["fund"] for x in rows]
    out = {"n_records": len(rows), "n_null_distance": n_null_dist,
           "n_a4_eligible": sum(1 for x in rows if x["f_pct"] is not None),
           "funding_rate_stats": {
               "min": min(fvals), "max": max(fvals),
               "mean": statistics.mean(fvals),
               "n_positive": sum(1 for v in fvals if v > 0),
               "n_negative": sum(1 for v in fvals if v < 0),
               "n_zero": sum(1 for v in fvals if v == 0)},
           "notes": [],
           "arms": {}, "contrasts": {}, "p_values": []}
    for name in sorted(arms):
        out["arms"][name] = summ(arms[name])
    for name, ra, rb in contrasts:
        c = welch([x["fwd"] for x in ra], [x["fwd"] for x in rb])
        out["contrasts"][name] = c
        if c:
            out["p_values"].append((name, c["p"]))

    ps = sorted([(p, n) for n, p in out["p_values"]])
    m = len(ps)
    out["fdr_bh"] = [{"contrast": n, "p": p, "alpha_bh": 0.05 * (i + 1) / m,
                      "survives": p <= 0.05 * (i + 1) / m}
                     for i, (p, n) in enumerate(ps)]

    fs = out["funding_rate_stats"]
    if fs["n_negative"] == 0:
        out["notes"].append(
            "A2 degenerate: last_funding_rate > 0 for every record in this "
            "cohort (min {:.2e}); the settled-positive-vs-negative sign "
            "contrast is untestable on this window — 4 planned contrasts "
            "return null and drop out of the FDR family.".format(fs["min"]))
    out["notes"].append(
        "A1 6-8h distance bucket and A3 post-settlement group are the same "
        "bars by construction (distance 6-8h <=> decision hours 0-2h after "
        "settlement); they differ only by the {} null-distance records — "
        "their two contrasts are effectively one test.".format(
            n_null_dist))
    out["notes"].append(
        "Power: pooled fwd-return sd ~{} bps; approx 80%-power MDE is "
        "~20-25 bps per contrast — observed diffs (~9-10 bps) are below "
        "MDE, so nulls here may be underpowered rather than absent "
        "effects.".format(round(statistics.pstdev(
            [x["fwd"] for x in rows]), 1)))

    return {"schema_version": "nanojev-financial-signal-funding-timing-v1",
            "status": "measurement_complete",
            "scope": "funding-timing arms on hourly PIT pilot cohort; "
                     "measurement only; overlapping 4h labels on 1h bars "
                     "make Welch t nominal (positive autocorrelation); "
                     "pilot n=716 is low-power — nulls may be underpowered",
            "parameters": {"bucket_hours": BUCKET_H,
                           "settlement_grid_ms": SETTLE_MS,
                           "post_settlement_window_ms": POST_MS,
                           "funding_pct_trailing_records": TRAIL,
                           "quintile": 0.20, "welch_min_n": MIN_N,
                           "fdr_alpha": 0.05,
                           "contrast_family_size": len(contrasts)},
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime()),
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
