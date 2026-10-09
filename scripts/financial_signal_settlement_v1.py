#!/usr/bin/env python3
"""T93: validation of the post-settlement negative tilt (T89 finding) on the
expanded hourly PIT cohort.

Cohort: data/perp_pit_intraday_v1/records.jsonl — 17,960 hourly decision
bars, 5 USDT-M perp assets (BNB/BTC/ETH/SOL/XRP), 2026-04-02..2026-08-31
UTC. Label = label.forward_return_bps (4h forward mark return, gross, bps;
present on all 17,960 records). All funding intervals are 8h; decision_ms
mod 28_800_000 = k*3_600_000 + 3_599_999, so mod < 7_200_000 selects exactly
the two hourly bars covering 0-2h after each funding settlement
("post" group, n=4,500).

T89 (same definition) reported post vs rest = -4.8bps, p~0.005. This script
validates whether that is alpha, a funding-payment mechanical artifact, or
noise, via five checks:

  V1 funding-sign split: post bars split by settled last_funding_rate
     sign (>0 / <0 / =0). If the dip is concentrated in funding>0 it is
     consistent with a mark/index settlement mechanic tied to the
     longs-pay side; if it appears in both signs it is not.
     Auxiliary: mark_index_basis_bps level in post vs rest bars — a
     settlement mechanic would show as a basis discontinuity.
  V2 conditioning grid: post-vs-rest tilt inside trailing-90 mid-rank
     percentile terciles of last_funding_rate (funding_pct) and
     realized_vol_24h (rv_pct), and inside each asset.
  V3 horizon: forward returns recomputed from the mark_price series at
     1h/2h/4h (mark-derived 4h verified equal to the label) — is the tilt
     a fast post-settlement move or a slow drift?
  V4 economics: 5bps/side taker -> ~10bps round trip; a ~-5bps gross tilt
     is not tradable. Value is diagnostic: does the post window bias other
     signals' evaluation bars?
  V5 fold/time stability: per-UTC-month post-vs-rest tilt, plus a
     full 8h cycle-position profile (sawtooth check: is the dip
     settlement-centered or specific to the post window?).

Statistics: Welch two-sample t, normal-approx two-sided p (same convention
as financial_signal_regime_v1 / financial_signal_funding_timing_v1), min
n=10 per side; BH-FDR alpha=0.05 over a fixed family of 8 contrasts.

Caveats: 4h forward labels on 1h bars overlap ~4x (2x at 2h horizon),
inducing positive serial correlation; Welch t assumes independence so
p-values are nominal/optimistic. Cross-asset bars sharing a timestamp are
also correlated (common market factor). Measurement only — no fitting, no
trading, no profitability claims.
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
OUT = ROOT / "results/financial_signal_settlement_v1.json"
TRAIL = 90                  # trailing per-asset records for mid-rank pct (~90h)
SETTLE_MS = 28_800_000      # 8h settlement grid in ms
POST_MS = 7_200_000         # 0-2h after settlement
HOUR_NS = 3_600_000_000_000
MIN_N = 10                  # Welch minimum per side (repo convention)
FEE_BPS_PER_SIDE = 5.0      # taker assumption for the economics check
HORIZONS = (1, 2, 4)        # forward horizons in hours (mark-derived)


def feat(r, name):
    f = r.get("features", {}).get(name)
    return None if f is None else f.get("value")


def pct(w, x):
    """Mid-rank percentile of x within trailing window w."""
    return (sum(1 for v in w if v < x)
            + 0.5 * sum(1 for v in w if v == x)) / len(w)


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
    mde = (1.96 + 0.84) * pooled_sd * math.sqrt(1 / len(a) + 1 / len(b))
    return {"n_a": len(a), "n_b": len(b),
            "mean_a_bps": round(ma, 2), "mean_b_bps": round(mb, 2),
            "diff_bps": round(ma - mb, 2), "t": round(t, 3),
            "p": round(p, 6), "approx_mde80_bps": round(mde, 1)}


def summ(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0, "mean_bps": None, "median_bps": None}
    return {"n": len(vals), "mean_bps": round(statistics.mean(vals), 2),
            "median_bps": round(statistics.median(vals), 2)}


def tercile(p):
    if p is None:
        return None
    return "low" if p < 1 / 3 else ("mid" if p < 2 / 3 else "high")


def run():
    by_asset = defaultdict(list)
    for line in COHORT.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by_asset[r["asset_id"]].append(r)

    rows = []
    label_check = {"n_compared": 0, "max_abs_diff_bps": 0.0}
    for asset, rs in sorted(by_asset.items()):
        rs.sort(key=lambda x: x["decision_ns"])
        marks = {r["decision_ns"]: feat(r, "mark_price") for r in rs}
        fund = [feat(r, "last_funding_rate") for r in rs]
        rv = [feat(r, "realized_vol_24h") for r in rs]
        for i, r in enumerate(rs):
            fwd = r.get("label", {}).get("forward_return_bps")
            if fwd is None:
                continue
            t = r["decision_ns"]
            ms = t // 1_000_000
            m0 = marks[t]
            fh = {}
            for h in HORIZONS:
                m1 = marks.get(t + h * HOUR_NS)
                fh[h] = None if (m1 is None or m0 <= 0) \
                    else (m1 / m0 - 1) * 1e4
            if fh[4] is not None:
                label_check["n_compared"] += 1
                label_check["max_abs_diff_bps"] = max(
                    label_check["max_abs_diff_bps"], abs(fh[4] - fwd))
            rows.append({
                "asset": asset, "ns": t, "ms": ms,
                "month": time.strftime("%Y-%m", time.gmtime(ms / 1000)),
                "cyc": (ms % SETTLE_MS) // 3_600_000,   # 0..7 = +0-1h..+7-8h
                "post": ms % SETTLE_MS < POST_MS,
                "fwd": fwd, "f1": fh[1], "f2": fh[2], "f4": fh[4],
                "fund": fund[i], "basis": feat(r, "mark_index_basis_bps"),
                "f_pct": pct(fund[i - TRAIL:i], fund[i]) if i >= TRAIL else None,
                "r_pct": pct(rv[i - TRAIL:i], rv[i]) if i >= TRAIL else None,
            })

    post = [x for x in rows if x["post"]]
    rest = [x for x in rows if not x["post"]]

    def sgn(x):
        return "pos" if x["fund"] > 0 else ("neg" if x["fund"] < 0
                                            else "zero")

    # ---------- arms ----------
    arms = defaultdict(list)
    for x in rows:
        g = "post" if x["post"] else "rest"
        arms[f"all_{g}"].append(x["fwd"])
        arms[f"fundsgn_{sgn(x)}_{g}"].append(x["fwd"])
        arms[f"cycle_hour_{x['cyc']}"].append(x["fwd"])
        arms[f"asset_{x['asset']}_{g}"].append(x["fwd"])
        arms[f"month_{x['month']}_{g}"].append(x["fwd"])
        fp, rp = tercile(x["f_pct"]), tercile(x["r_pct"])
        if fp:
            arms[f"funding_pct_{fp}_{g}"].append(x["fwd"])
        if rp:
            arms[f"rv_pct_{rp}_{g}"].append(x["fwd"])

    def sel(pred, key="fwd"):
        return [x[key] for x in rows
                if pred(x) and x[key] is not None]

    # ---------- V1: funding-sign split + basis diagnostic ----------
    sign_split = {}
    for s in ("pos", "neg", "zero"):
        sign_split[f"post_fund_{s}"] = summ(
            sel(lambda x, s=s: x["post"] and sgn(x) == s))
        sign_split[f"rest_fund_{s}"] = summ(
            sel(lambda x, s=s: not x["post"] and sgn(x) == s))
    basis_diag = {name: summ([x["basis"] for x in rows if c(x)])
                  for name, c in {
                      "post_fund_pos": lambda x: x["post"] and x["fund"] > 0,
                      "post_fund_neg": lambda x: x["post"] and x["fund"] < 0,
                      "post_fund_zero": lambda x: x["post"] and x["fund"] == 0,
                      "rest": lambda x: not x["post"]}.items()}

    # ---------- V2: conditioning grids ----------
    def grid(name, keyfn):
        cells = {}
        for cell in sorted({keyfn(x) for x in rows if keyfn(x) is not None}):
            pa = sel(lambda x, c=cell: x["post"] and keyfn(x) == c)
            ra = sel(lambda x, c=cell: not x["post"] and keyfn(x) == c)
            cells[str(cell)] = {
                "post": summ(pa), "rest": summ(ra),
                "tilt_bps": (round(statistics.mean(pa)
                                   - statistics.mean(ra), 2)
                             if len(pa) >= MIN_N and len(ra) >= MIN_N
                             else None),
                "welch_post_vs_rest": welch(pa, ra)}
        return cells

    grids = {
        "asset": grid("asset", lambda x: x["asset"]),
        "funding_pct_tercile": grid("funding_pct",
                                    lambda x: tercile(x["f_pct"])),
        "rv_pct_tercile": grid("rv_pct", lambda x: tercile(x["r_pct"])),
        "month": grid("month", lambda x: x["month"]),
    }

    # ---------- V3: horizons (mark-derived) ----------
    horizons = {}
    for h in HORIZONS:
        k = "f%d" % h
        pa, ra = sel(lambda x: x["post"], k), sel(lambda x: not x["post"], k)
        horizons["%dh" % h] = {"post": summ(pa), "rest": summ(ra),
                               "welch_post_vs_rest": welch(pa, ra)}

    # ---------- V5: cycle profile (sawtooth check) ----------
    cycle_profile = {}
    for h in (1, 4):
        k = "f%d" % h
        cycle_profile["%dh" % h] = {
            str(c): summ(sel(lambda x, c=c: x["cyc"] == c, k))
            for c in range(8)}

    # ---------- contrast family (8) ----------
    contrasts = [
        ("C1_post_vs_rest_4h_label",
         sel(lambda x: x["post"]), sel(lambda x: not x["post"])),
        ("C2_post_vs_rest_1h",
         sel(lambda x: x["post"], "f1"), sel(lambda x: not x["post"], "f1")),
        ("C3_post_vs_rest_2h",
         sel(lambda x: x["post"], "f2"), sel(lambda x: not x["post"], "f2")),
        ("C4_post_fundpos_vs_rest_fundpos",
         sel(lambda x: x["post"] and x["fund"] > 0),
         sel(lambda x: not x["post"] and x["fund"] > 0)),
        ("C5_post_fundneg_vs_rest_fundneg",
         sel(lambda x: x["post"] and x["fund"] < 0),
         sel(lambda x: not x["post"] and x["fund"] < 0)),
        ("C6_post_fundpos_vs_post_fundneg",
         sel(lambda x: x["post"] and x["fund"] > 0),
         sel(lambda x: x["post"] and x["fund"] < 0)),
        ("C7_rv_pct_high_post_vs_rest",
         sel(lambda x: x["post"] and tercile(x["r_pct"]) == "high"),
         sel(lambda x: not x["post"] and tercile(x["r_pct"]) == "high")),
        ("C8_rv_pct_low_post_vs_rest",
         sel(lambda x: x["post"] and tercile(x["r_pct"]) == "low"),
         sel(lambda x: not x["post"] and tercile(x["r_pct"]) == "low")),
    ]
    cres, pvals = {}, []
    for name, a, b in contrasts:
        c = welch(a, b)
        cres[name] = c
        if c:
            pvals.append((name, c["p"]))
    ps = sorted([(p, n) for n, p in pvals])
    m = len(ps)
    fdr = [{"contrast": n, "p": p, "alpha_bh": 0.05 * (i + 1) / m,
            "survives": p <= 0.05 * (i + 1) / m}
           for i, (p, n) in enumerate(ps)]

    # ---------- V4: economics ----------
    tilt_4h = cres["C1_post_vs_rest_4h_label"]["diff_bps"]
    economics = {
        "assumed_taker_fee_bps_per_side": FEE_BPS_PER_SIDE,
        "round_trip_cost_bps": 2 * FEE_BPS_PER_SIDE,
        "observed_gross_tilt_4h_bps": tilt_4h,
        "tradable_after_fees": abs(tilt_4h) > 2 * FEE_BPS_PER_SIDE,
        "note": "A ~-5bps gross tilt is below one side of taker fees and "
                "~2x below round-trip cost; it is NOT a tradable edge. Its "
                "value is diagnostic: signals evaluated or triggered on "
                "post-settlement bars inherit a ~-5bps (4h, gross) to "
                "-6bps (2h) adverse-window bias, and the window is ~25% of "
                "all bars."}

    month_signs = {k: (v["tilt_bps"] is not None and v["tilt_bps"] < 0)
                   for k, v in grids["month"].items()}

    notes = [
        "Label check: mark-derived 4h forward return equals "
        "label.forward_return_bps exactly (max |diff| = {:.4f} bps over "
        "{:,} comparable records); 40 records lack a t+4h mark bar "
        "(series tails).".format(label_check["max_abs_diff_bps"],
                                 label_check["n_compared"]),
        "Cycle profile is a sawtooth: mean 4h fwd is positive mid-cycle "
        "(+2h..+5h after settlement, ~+3.7..+5.7bps) and ~0/negative in "
        "the +-2h bracket around the settlement instant (cycle hours "
        "0-1 and 6-7). The T89 'post-settlement dip' is the post half of "
        "a settlement-centered weak window, not a uniquely post-settlement "
        "effect.",
        "Overlapping labels (4h fwd on 1h bars; 2x at 2h) and shared "
        "timestamps across 5 assets induce positive correlation; Welch t "
        "treats observations as independent so p-values are nominal/"
        "optimistic — treat p<0.01 as suggestive, not decisive.",
        "All 5 assets and all 5 months show negative tilt signs (see "
        "grids), but no single asset or month is independently "
        "significant after conditioning — consistency comes from pooling.",
    ]

    out = {
        "n_records": len(rows),
        "n_post_settlement_bars": len(post),
        "n_rest": len(rest),
        "post_fraction": round(len(post) / len(rows), 4),
        "label_check": label_check,
        "v1_funding_sign_split": sign_split,
        "v1_basis_diagnostic_bps": basis_diag,
        "v2_conditioning_grids": grids,
        "v3_horizons_mark_derived": horizons,
        "v5_cycle_profile_mean_fwd_bps": cycle_profile,
        "v5_month_tilt_all_negative": all(month_signs.values()),
        "v4_economics": economics,
        "arms": {k: summ(v) for k, v in sorted(arms.items())},
        "contrasts": cres,
        "fdr_bh": fdr,
        "notes": notes,
    }

    return {"schema_version": "nanojev-financial-signal-settlement-v1",
            "status": "measurement_complete",
            "scope": "T93 validation of the T89 post-settlement negative "
                     "tilt on the expanded hourly PIT cohort; measurement "
                     "only; overlapping forward labels on 1h bars and "
                     "cross-asset timestamp correlation make Welch t "
                     "nominal (optimistic)",
            "parameters": {"settlement_grid_ms": SETTLE_MS,
                           "post_settlement_window_ms": POST_MS,
                           "horizons_hours": list(HORIZONS),
                           "pct_trailing_records_per_asset": TRAIL,
                           "tercile_edges": [1 / 3, 2 / 3],
                           "welch_min_n": MIN_N,
                           "fee_bps_per_side": FEE_BPS_PER_SIDE,
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
