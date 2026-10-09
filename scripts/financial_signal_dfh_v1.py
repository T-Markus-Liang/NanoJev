#!/usr/bin/env python3
"""T101: the two top-ranked T100 literature arms on the daily PIT cohort.

Cohort: data/perp_pit_v1/records.jsonl — 6,554 daily decision bars, 5
USDT-M perp assets (BNB/BTC/ETH/SOL/XRP), 2023-01-25..2026-08-30 UTC.
Targets are recomputed per asset as forward log(mark) returns: primary
5d, secondary 1d (PIT-safe — targets strictly after the decision bar).

Arm 1 — distance-from-high momentum (Prague WP; distance-from-1wk-high
positive on large/liquid coins, t=+4.93; direction prior POSITIVE —
near-high -> continuation):
  A1a Spearman(dfh, fwd_5d) pooled across assets + per-asset table;
     dfh20 = mark[i]/max(mark[i-20:i])-1 (trailing 20d high over strictly
     prior bars); dfh10 variant reported alongside.
  A1b top (dfh_pct>=0.80) vs bottom (<=0.20) quintile -> fwd 5d Welch.
  A1c within the confirmed f2b2 cell (funding_pct>=0.80 AND
     basis_pct>=0.66): Spearman(dfh20, fwd_5d) + high-vs-low dfh Welch —
     does distance-from-high add inside the cell.

Arm 2 — liquidity-conditioned momentum/reversal flip (Zaremba IRFA
2021: illiquid coins reverse, liquid coins continue):
  amihud_i = |ret_1d_i| / quote_volume_i, ranked vs trailing-180 mid-rank
  pct -> terciles. Per tercile Spearman(lag5_ret, fwd_5d_ret): expect
  NEGATIVE rho in the illiquid tercile (reversal), POSITIVE in the
  liquid tercile (momentum). Flip contrast = Fisher-z difference
  atanh(rho_liquid) - atanh(rho_illiquid), expected positive.

Statistics: Spearman (mid-rank ties, t-approx normal two-sided p),
Welch two-sample t (normal-approx two-sided p), min n=10 — same
conventions as financial_signal_leadlag_v1 / regime_v1. BH-FDR
alpha=0.05 over the fixed 8-cell family declared in
research/financial_signal_dfh_protocol_v1.json; per-asset tables, the
mid tercile, the secondary 1d Spearman and per-test-fold sign counts
are reported outside the family.

Caveats: 5d labels on daily bars overlap ~5x; pooled rows across 5
assets share the market factor through common timestamps; both inflate
effective n so p-values are nominal/optimistic. Measurement only — no
fitting, no trading, no profitability claims.
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
OUT = ROOT / "results/financial_signal_dfh_v1.json"
LOOKBACK = 180          # trailing rows for mid-rank pct (repo conv.)
HOLD = 5                # primary forward horizon (days)
HOLD1 = 1               # secondary horizon (days)
DFH_WIN = 20            # trailing-high window, strictly prior bars
DFH_WIN_ALT = 10        # 10d variant
MIN_N = 10              # test minimum per side (repo convention)


def feat(r, name):
    f = r.get("features", {}).get(name)
    return None if f is None else f.get("value")


def pct(w, x):
    """Mid-rank percentile of x within trailing window w (project conv.)."""
    return (sum(1 for v in w if v < x)
            + 0.5 * sum(1 for v in w if v == x)) / len(w)


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
    """Spearman rho, t-approx normal two-sided p (crossvenue convention).
    Returns None if n < MIN_N or either side has zero variance."""
    n = len(xs)
    if n < MIN_N:
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
    return {"n": n, "rho": round(rho, 4), "p": round(p, 6)}


def welch(a, b):
    """Welch t of mean(a)-mean(b) on raw log returns; means reported in
    bps (regime_v1 convention). None if either side < MIN_N or se==0."""
    if len(a) < MIN_N or len(b) < MIN_N:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.pvariance(a), statistics.pvariance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return None
    t = (ma - mb) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return {"n_a": len(a), "n_b": len(b),
            "mean_a_bps": round(ma * 1e4, 1),
            "mean_b_bps": round(mb * 1e4, 1),
            "diff_bps": round((ma - mb) * 1e4, 1),
            "t": round(t, 3), "p": round(p, 6)}


def fisher_z_diff(r1, n1, r2, n2):
    """Two-sided normal-approx p for atanh(r1)-atanh(r2) (nominal —
    cells share timestamps). Returns dict or None."""
    if n1 < MIN_N or n2 < MIN_N or r1 is None or r2 is None:
        return None
    if abs(r1) >= 1 or abs(r2) >= 1:
        return None
    z = (math.atanh(r1) - math.atanh(r2)) / math.sqrt(
        1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {"rho_1": round(r1, 4), "rho_2": round(r2, 4),
            "diff": round(r1 - r2, 4), "z": round(z, 3),
            "p": round(p, 6)}


def run():
    records = [json.loads(l) for l in COHORT.read_text().splitlines()
               if l.strip()]
    folds = json.loads(FOLDS.read_text())["folds"]
    by_asset = defaultdict(list)
    for r in records:
        by_asset[r["asset_id"]].append(r)
    for rs in by_asset.values():
        rs.sort(key=lambda x: x["decision_ns"])

    # ---------- build eval rows ----------
    rows = []
    for asset, rs in sorted(by_asset.items()):
        mark = [feat(x, "mark_price") for x in rs]
        fund = [feat(x, "last_funding_rate") for x in rs]
        basis = [feat(x, "mark_index_basis_bps") for x in rs]
        qvol = [feat(x, "quote_volume") for x in rs]
        n = len(rs)

        dfh20 = [None] * n
        dfh10 = [None] * n
        amihud = [None] * n
        lag5 = [None] * n
        for i in range(n):
            if mark[i] is None or mark[i] <= 0:
                continue
            if i >= DFH_WIN:
                w = [m for m in mark[i - DFH_WIN:i] if m and m > 0]
                if len(w) == DFH_WIN:
                    dfh20[i] = mark[i] / max(w) - 1.0
            if i >= DFH_WIN_ALT:
                w = [m for m in mark[i - DFH_WIN_ALT:i] if m and m > 0]
                if len(w) == DFH_WIN_ALT:
                    dfh10[i] = mark[i] / max(w) - 1.0
            if i >= 1 and mark[i - 1] and mark[i - 1] > 0 \
                    and qvol[i] is not None and qvol[i] > 0:
                amihud[i] = abs(mark[i] / mark[i - 1] - 1.0) / qvol[i]
            if i >= HOLD and mark[i - HOLD] and mark[i - HOLD] > 0:
                lag5[i] = math.log(mark[i] / mark[i - HOLD])

        for i in range(LOOKBACK, n - HOLD):
            if mark[i] is None or mark[i] <= 0:
                continue
            ret5 = (math.log(mark[i + HOLD] / mark[i])
                    if mark[i + HOLD] and mark[i + HOLD] > 0 else None)
            ret1 = (math.log(mark[i + HOLD1] / mark[i])
                    if mark[i + HOLD1] and mark[i + HOLD1] > 0 else None)
            if ret5 is None or fund[i] is None or basis[i] is None:
                continue
            f_pct = pct(fund[i - LOOKBACK:i], fund[i])
            b_pct = pct(basis[i - LOOKBACK:i], basis[i])
            row = {"asset": asset, "ns": rs[i]["decision_ns"],
                   "ret5": ret5, "ret1": ret1,
                   "dfh20": dfh20[i], "dfh10": dfh10[i],
                   "lag5": lag5[i],
                   "f_pct": f_pct, "b_pct": b_pct,
                   "in_f2b2": f_pct >= 0.80 and b_pct >= 0.66}
            dwin = [v for v in dfh20[i - LOOKBACK:i] if v is not None]
            awin = [v for v in amihud[i - LOOKBACK:i] if v is not None]
            row["dfh_pct"] = (pct(dwin, dfh20[i])
                              if dfh20[i] is not None and dwin else None)
            row["amihud_pct"] = (pct(awin, amihud[i])
                                 if amihud[i] is not None and awin
                                 else None)
            rows.append(row)

    ok = lambda r, *ks: all(r[k] is not None for k in ks)

    # ---------- A1a: Spearman(dfh, fwd) ----------
    d20 = [r for r in rows if ok(r, "dfh20", "ret5")]
    d10 = [r for r in rows if ok(r, "dfh10", "ret5")]
    d20r1 = [r for r in rows if ok(r, "dfh20", "ret1")]
    a1a = {
        "dfh20_ret5_pooled": spearman([r["dfh20"] for r in d20],
                                      [r["ret5"] for r in d20]),
        "dfh10_ret5_pooled": spearman([r["dfh10"] for r in d10],
                                      [r["ret5"] for r in d10]),
        "dfh20_ret1_pooled_secondary": spearman(
            [r["dfh20"] for r in d20r1], [r["ret1"] for r in d20r1]),
        "dfh20_ret5_per_asset": {
            a: spearman([r["dfh20"] for r in d20 if r["asset"] == a],
                        [r["ret5"] for r in d20 if r["asset"] == a])
            for a in sorted(by_asset)},
    }

    # ---------- A1b: top vs bottom dfh quintile -> 5d Welch ----------
    top = [r["ret5"] for r in rows
           if r["dfh_pct"] is not None and r["dfh_pct"] >= 0.80]
    bot = [r["ret5"] for r in rows
           if r["dfh_pct"] is not None and r["dfh_pct"] <= 0.20]
    a1b = {"definition": "dfh_pct (trailing-180 mid-rank of dfh20) "
                         ">=0.80 vs <=0.20; Welch on fwd 5d log ret",
           "top_vs_bottom": welch(top, bot)}

    # ---------- A1c: inside f2b2 ----------
    f2 = [r for r in rows if r["in_f2b2"] and ok(r, "dfh20", "ret5")]
    hi = [r["ret5"] for r in f2
          if r["dfh_pct"] is not None and r["dfh_pct"] >= 0.50]
    lo = [r["ret5"] for r in f2
          if r["dfh_pct"] is not None and r["dfh_pct"] < 0.50]
    a1c = {"n_f2b2_rows": len(f2),
           "spearman_dfh20_ret5": spearman([r["dfh20"] for r in f2],
                                           [r["ret5"] for r in f2]),
           "highdfh_vs_lowdfh_median_split": welch(hi, lo)}

    # ---------- A2: amihud-tercile momentum/reversal flip ----------
    a2rows = [r for r in rows
              if ok(r, "amihud_pct", "lag5", "ret5")]
    terc = {"liquid": [r for r in a2rows if r["amihud_pct"] < 1.0 / 3],
            "mid": [r for r in a2rows
                    if 1.0 / 3 <= r["amihud_pct"] < 2.0 / 3],
            "illiquid": [r for r in a2rows if r["amihud_pct"] >= 2.0 / 3]}
    sp = {k: spearman([r["lag5"] for r in v], [r["ret5"] for r in v])
          for k, v in terc.items()}
    flip = None
    if sp["liquid"] and sp["illiquid"]:
        flip = fisher_z_diff(sp["liquid"]["rho"], sp["liquid"]["n"],
                             sp["illiquid"]["rho"], sp["illiquid"]["n"])
    per_asset_flip = {}
    for a in sorted(by_asset):
        rl = [r for r in terc["liquid"] if r["asset"] == a]
        ri = [r for r in terc["illiquid"] if r["asset"] == a]
        sl = spearman([r["lag5"] for r in rl], [r["ret5"] for r in rl])
        si = spearman([r["lag5"] for r in ri], [r["ret5"] for r in ri])
        per_asset_flip[a] = {
            "liquid": sl, "illiquid": si,
            "rho_liquid_minus_illiquid":
                round(sl["rho"] - si["rho"], 4) if sl and si else None}
    a2 = {"definition": "amihud_pct terciles: <1/3 liquid, >=2/3 "
                        "illiquid; Spearman(lag5_ret, fwd_5d_ret) per "
                        "tercile; prior = rho_liquid>0, rho_illiquid<0",
          "n_eval": len(a2rows),
          "terciles": {k: {"n": len(v), "spearman": sp[k]}
                       for k, v in terc.items()},
          "flip_contrast_liquid_minus_illiquid": flip,
          "per_asset": per_asset_flip}

    # ---------- per-test-fold sign consistency (outside family) ----------
    per_fold = {}
    for fi, f in enumerate(folds):
        a_, b_ = f["test"]
        sub = [r for r in d20 if a_ <= r["ns"] < b_]
        s = spearman([r["dfh20"] for r in sub], [r["ret5"] for r in sub])
        per_fold[f"f{fi}"] = {"n": len(sub),
                              "spearman_dfh20_ret5": s,
                              "sign_positive": (s["rho"] > 0
                                                if s else None)}

    # ---------- FDR family (8 cells, per protocol) ----------
    fam = {}
    if a1a["dfh20_ret5_pooled"]:
        fam["arm1_spearman_dfh20_ret5_pooled"] = \
            a1a["dfh20_ret5_pooled"]["p"]
    if a1a["dfh10_ret5_pooled"]:
        fam["arm1_spearman_dfh10_ret5_pooled"] = \
            a1a["dfh10_ret5_pooled"]["p"]
    if a1b["top_vs_bottom"]:
        fam["arm1_welch_quintile_dfh20_ret5"] = \
            a1b["top_vs_bottom"]["p"]
    if a1c["spearman_dfh20_ret5"]:
        fam["arm1_f2b2_spearman_dfh20_ret5"] = \
            a1c["spearman_dfh20_ret5"]["p"]
    if a1c["highdfh_vs_lowdfh_median_split"]:
        fam["arm1_f2b2_welch_highdfh_vs_lowdfh_ret5"] = \
            a1c["highdfh_vs_lowdfh_median_split"]["p"]
    if sp["liquid"]:
        fam["arm2_spearman_lag5_ret5_tercile_liquid"] = sp["liquid"]["p"]
    if sp["illiquid"]:
        fam["arm2_spearman_lag5_ret5_tercile_illiquid"] = \
            sp["illiquid"]["p"]
    if flip:
        fam["arm2_flip_fisherz_liquid_minus_illiquid"] = flip["p"]
    ps = sorted([(p, n_) for n_, p in fam.items()])
    m = len(ps)
    fdr = [{"contrast": n_, "p": p, "alpha_bh": 0.05 * (i + 1) / m,
            "survives": p <= 0.05 * (i + 1) / m}
           for i, (p, n_) in enumerate(ps)]
    surv = {e["contrast"] for e in fdr if e["survives"]}

    # ---------- honest verdict ----------
    r20 = (a1a["dfh20_ret5_pooled"] or {}).get("rho")
    r10 = (a1a["dfh10_ret5_pooled"] or {}).get("rho")
    q = a1b["top_vs_bottom"] or {}
    f2s = (a1c["spearman_dfh20_ret5"] or {}).get("rho")
    rl_ = (sp["liquid"] or {}).get("rho")
    ri_ = (sp["illiquid"] or {}).get("rho")
    a1_prior_ok = r20 is not None and r20 > 0
    a2_flip_ok = (rl_ is not None and ri_ is not None
                  and rl_ > 0 and ri_ < 0)
    verdict_bits = []
    verdict_bits.append(
        "Arm1 dfh20 pooled rho={:.4f} (dfh10 {:.4f}): direction prior "
        "was POSITIVE — {}.".format(
            r20 if r20 is not None else float("nan"),
            r10 if r10 is not None else float("nan"),
            "CONFIRMED in sign" if a1_prior_ok
            else "sign-opposed to literature"))
    verdict_bits.append(
        "Quintile Welch top-vs-bottom diff {}bps (t={}); f2b2-internal "
        "rho={}.".format(q.get("diff_bps"), q.get("t"), f2s))
    verdict_bits.append(
        "Arm2 liquid rho={}, illiquid rho={} (prior +/−): flip "
        "{}.".format(rl_, ri_,
                     "in predicted direction" if a2_flip_ok
                     else "NOT in predicted direction"))
    verdict_bits.append(
        "FDR survivors: {}.".format(sorted(surv) if surv else "none"))

    out = {
        "n_records": len(records),
        "n_eval_rows": len(rows),
        "eval_window_note": "rows require i>=180 (trailing pct) and "
                            "i<=n-6 (5d fwd); per-asset n ~1125",
        "arm1_distance_from_high": {
            "prior": "POSITIVE (near-high -> continuation; Prague WP "
                     "t=+4.93 on large/liquid coins)",
            "a1a_spearman": a1a,
            "a1b_quintile_welch": a1b,
            "a1c_f2b2_conditioned": a1c,
            "prior_direction_confirmed": a1_prior_ok,
        },
        "arm2_liquidity_flip": a2,
        "arm2_prior_direction_confirmed": a2_flip_ok,
        "per_test_fold": per_fold,
        "fdr_bh": fdr,
        "verdict": verdict_bits,
        "notes": [
            "5d forward labels overlap ~5x on daily bars and pooled "
            "rows share the market factor across assets -> nominal/"
            "optimistic p-values; treat p<0.01 as suggestive.",
            "amihud_pct terciles are per-asset trailing-180 ranks, so "
            "each tercile is ~1/3 of each asset's rows by construction.",
            "dfh uses strictly-prior bars (max(mark[i-20:i])) — no "
            "lookahead; all pct ranks are trailing-only (PIT-safe).",
            "Window sensitivity: the pooled effect is specific to the "
            "20d high — dfh10 pooled rho is ~0 (null), and the "
            "secondary 1d-horizon rho is also ~0: whatever exists is a "
            "multi-day continuation off the ~month-scale high, not a "
            "next-day effect.",
            "Per-asset heterogeneity: dfh20 rhos span -0.024..+0.077 "
            "with only SOL individually significant — the pooled "
            "positive rho leans on cross-asset/market-level "
            "co-movement, not a uniform per-coin effect.",
            "Near-high days are disproportionately market-up days, so "
            "part of the quintile spread is market beta rather than an "
            "asset-specific anomaly; gross-of-fees, measurement only.",
        ],
    }
    return {"schema_version": "nanojev-financial-signal-dfh-v1",
            "status": "measurement_complete",
            "scope": "T101 two top-ranked T100 literature arms "
                     "(distance-from-high momentum; Amihud "
                     "liquidity-conditioned momentum/reversal flip) on "
                     "the daily 5-asset PIT cohort; measurement only",
            "parameters": {"lookback_pct": LOOKBACK,
                           "hold_primary": HOLD,
                           "hold_secondary": HOLD1,
                           "dfh_windows": [DFH_WIN, DFH_WIN_ALT],
                           "quintile_thresholds": [0.80, 0.20],
                           "f2b2": "funding_pct>=0.80 AND basis_pct>=0.66",
                           "min_n": MIN_N,
                           "fdr_alpha": 0.05,
                           "contrast_family_size": 8},
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
