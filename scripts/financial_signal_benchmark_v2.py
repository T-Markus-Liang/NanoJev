#!/usr/bin/env python3
"""T91: cross-scale unified financial-signal benchmark (v2).

Replays the frozen T82 daily contract UNCHANGED (imported from
financial_signal_benchmark_v1 — same code path, same payload) and adds an
intraday layer over the hourly cohort, plus a cross_scale replication block.

  daily      run() of scripts/financial_signal_benchmark_v1.py verbatim:
             cohort data/perp_pit_v1/records.jsonl, 5d continuous label,
             frozen validator folds, trailing-180 PIT, 5bps+funding cost,
             10 cells — FDR family 1.

  intraday   cohort=data/perp_pit_intraday_v1/records.jsonl (sha pinned)
             label = label.forward_return_bps (4h fwd mark bps, gross)
             PIT   = trailing-90-record mid-rank percentiles, per-asset
             cost  = 5bps taker/side + funding accrual when
                     ns_until_next_funding_settlement <= 4h hold
             folds = 3 contiguous equal-width ns windows (advisory signs)
             cells = funding_pct | basis_pct | rv_pct | taker_pct |
                     abnvol_pct | mom_1h | mom_4h
             metrics: Spearman rho/t/p + Welch top-vs-bottom quintile +
                      top-decile arm net stats — same shape as daily cells
             placebo: labels shuffled within asset, 3 deterministic seeds
             verdicts: PASS/ECON_ONLY/FAIL — FDR family 2 (separate)

  cross_scale daily vs hourly funding_pct rho/p/sign + replication flag

FDR families are computed separately per scale (10 daily cells, 7 hourly
cells) — no cross-scale pooling; pooled alpha would not be comparable.

Measurement only — no fitting, no trading, no promotion claims.
"""

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import financial_signal_benchmark_v1 as v1

ROOT = Path(__file__).resolve().parent.parent
INTRADAY_COHORT = ROOT / "data/perp_pit_intraday_v1/records.jsonl"
OUT = ROOT / "results/financial_signal_benchmark_v2.json"
ILOOKBACK = 90              # trailing-90-record (~90h) percentile window
MOM4_LAG = 4                # mom_4h lookback in bars
HOLD_H = 4                  # label horizon, hours
ABN_MEDIAN = 30             # trailing-30-record volume median (abnvol)
IQ = 0.20                   # quintile fraction for Welch arm
INTRADAY_ARMS = ["funding_pct", "basis_pct", "rv_pct", "taker_pct",
                 "abnvol_pct", "mom_1h", "mom_4h"]
pct, spearman, net_stats = v1.pct, v1.spearman, v1.net_stats
FEE_BPS, PLACEBO_SEEDS = v1.FEE_BPS, v1.PLACEBO_SEEDS


# ---------------------------------------------------------------- stats
def welch_quintile(rows, name):
    """Welch mean-diff of fwd_bps, top vs bottom quintile of `name`."""
    k = max(1, int(len(rows) * IQ))
    srt = sorted(rows, key=lambda r: r[name])
    w = v1.welch([r["fwd_bps"] for r in srt[-k:]],
                 [r["fwd_bps"] for r in srt[:k]])
    if w is None:
        return None
    return {"n_top": w["n_a"], "n_bottom": w["n_b"],
            "mean_top_bps": round(w["mean_a"], 5),
            "mean_bottom_bps": round(w["mean_b"], 5),
            "diff_bps": round(w["mean_a"] - w["mean_b"], 5),
            "t": round(w["t"], 5), "p": round(w["p"], 5)}


def shuffle(vals, seed):
    """Deterministic LCG Fisher-Yates (same generator as T82/T88)."""
    v = vals[:]
    s = seed
    for i in range(len(v) - 1, 0, -1):
        s = (s * 6364136223846793005 + 1442695040888963407) & (2**64 - 1)
        j = s % (i + 1)
        v[i], v[j] = v[j], v[i]
    return v


# ---------------------------------------------------------------- dataset
def build_intraday_rows():
    """Per-asset hourly rows: trailing-90 PIT scores + 4h fwd bps label."""
    records = [json.loads(l)
               for l in INTRADAY_COHORT.read_text().splitlines() if l.strip()]
    by_asset = defaultdict(list)
    for r in records:
        by_asset[r["asset_id"]].append(r)
    for rs in by_asset.values():
        rs.sort(key=lambda x: x["decision_ns"])

    hold_ns = HOLD_H * 3600 * 10**9
    rows = []
    for asset, rs in sorted(by_asset.items()):
        def feat(name):
            return [x["features"][name]["value"] for x in rs]

        mark = feat("mark_price")
        fund = feat("last_funding_rate")
        basis = feat("mark_index_basis_bps")
        rv = feat("realized_vol_24h")
        taker = feat("taker_buy_ratio")
        vol = feat("quote_volume")
        settle = feat("ns_until_next_funding_settlement")
        ivl = feat("funding_interval_hours")
        n = len(rs)

        abn = []
        for i in range(n):
            med = statistics.median(vol[max(0, i - ABN_MEDIAN):i]) if i \
                else vol[0]
            abn.append(vol[i] / med if med > 0 else 1.0)
        mom1, mom4 = [None] * n, [None] * n
        for i in range(1, n):
            if mark[i] > 0 and mark[i - 1] > 0:
                mom1[i] = math.log(mark[i] / mark[i - 1])
        for i in range(MOM4_LAG, n):
            if mark[i] > 0 and mark[i - MOM4_LAG] > 0:
                mom4[i] = math.log(mark[i] / mark[i - MOM4_LAG])
        raw = {"funding_pct": fund, "basis_pct": basis, "rv_pct": rv,
               "taker_pct": taker, "abnvol_pct": abn,
               "mom_1h": mom1, "mom_4h": mom4}

        i_min = ILOOKBACK + MOM4_LAG   # all seven windows complete
        for i in range(i_min, n):
            fwd = rs[i]["label"].get("forward_return_bps")
            if fwd is None:
                continue
            row = {"asset": asset, "ns": rs[i]["decision_ns"],
                   "fwd_bps": fwd, "gross_bps": fwd}
            ok = True
            for arm, series in raw.items():
                window = series[i - ILOOKBACK:i]
                if series[i] is None or any(v is None for v in window):
                    ok = False
                    break
                row[arm] = pct(window, series[i])
            if not ok:
                continue
            # funding cashflow iff a settlement lands inside the 4h hold;
            # fallback: pro-rata accrual when the schedule feature is null
            s = settle[i]
            if s is not None:
                paid = fund[i] * 1e4 if 0 <= s <= hold_ns else 0.0
            else:
                paid = (fund[i] * (HOLD_H / ivl[i]) * 1e4
                        if ivl[i] > 0 else 0.0)
            row["net_bps"] = fwd - 2 * FEE_BPS - paid
            rows.append(row)
    return records, rows


# ---------------------------------------------------------------- cells
def intraday_cell_metrics(rows, name, windows):
    xs = [r[name] for r in rows]
    ys = [r["fwd_bps"] for r in rows]
    sp = spearman(xs, ys)
    arm = sorted(rows, key=lambda r: -r[name])[:max(10, len(rows) // 10)]
    fold_signs = []
    for a, b_ in windows:
        sub = [r["fwd_bps"] for r in arm if a <= r["ns"] < b_]
        fold_signs.append((1 if statistics.mean(sub) > 0 else -1)
                          if len(sub) >= 5 else None)
    return {"n": len(rows),
            "primary": {"spearman": ({k: round(v, 5) for k, v in sp.items()}
                                     | {"n": len(rows)} if sp else None)},
            "welch_top_vs_bottom_quintile": welch_quintile(rows, name),
            "arm_n": len(arm),
            "arm_mean_gross_bps":
                round(statistics.mean([r["gross_bps"] for r in arm]), 1)
                if arm else None,
            "arm_net_stats": net_stats([r["net_bps"] for r in arm]),
            "fold_arm_signs": fold_signs,
            "p": sp["p"] if sp else None,
            "rho": sp["rho"] if sp else None}


def intraday_placebo_rhos(rows, name):
    """Shuffle fwd_bps within asset; rho of score vs shuffled label."""
    xs = [r[name] for r in rows]
    by_asset = defaultdict(list)
    for r in rows:                      # rows grouped by asset in build order
        by_asset[r["asset"]].append(r["fwd_bps"])
    out = []
    for seed in PLACEBO_SEEDS:
        shuffled = []
        for asset, vals in sorted(by_asset.items()):
            shuffled.extend(shuffle(vals, seed))
        sp = spearman(xs, shuffled)
        out.append(round(sp["rho"], 4) if sp else None)
    return out


def run_intraday():
    records, rows = build_intraday_rows()
    ys = [r["fwd_bps"] for r in rows]
    ns0, ns1 = min(r["ns"] for r in rows), max(r["ns"] for r in rows)
    third = (ns1 - ns0 + 1) / 3.0       # 3 contiguous advisory windows
    windows = [(ns0 + int(k * third),
                ns0 + int((k + 1) * third) if k < 2 else ns1 + 1)
               for k in range(3)]

    always = net_stats([r["net_bps"] for r in rows])
    cells = {}
    for name in INTRADAY_ARMS:
        m = intraday_cell_metrics(rows, name, windows)
        m["placebo_rhos"] = intraday_placebo_rhos(rows, name)
        m["placebo_max_abs_rho"] = max(
            (abs(x) for x in m["placebo_rhos"] if x is not None), default=None)
        cells[name] = m

    # BH-FDR family 2: the 7 intraday cells only (separate from daily)
    ps = sorted((m["p"], n) for n, m in cells.items()
                if m.get("p") is not None)
    M = len(ps)
    fdr = {}
    for i, (p, n) in enumerate(ps):
        fdr[n] = p <= 0.05 * (i + 1) / M

    verdicts = {}
    for n, m in cells.items():
        rho, band = m["rho"], m["placebo_max_abs_rho"]
        stat = fdr.get(n, False)
        pl = (rho is not None and band is not None
              and abs(rho) > max(0.02, band))
        econ = (m["arm_net_stats"] and always and
                m["arm_net_stats"]["mean_net_bps"]
                > always["mean_net_bps"])
        folds_ok = sum(1 for s in m["fold_arm_signs"] if s == 1)
        verdicts[n] = {
            "fdr_pass": stat,
            "placebo_clear": pl,
            "econ_vs_always_long": econ,
            "folds_positive": f"{folds_ok}/{sum(1 for s in m['fold_arm_signs'] if s is not None)}",
            "verdict": "PASS" if stat and pl else
                       ("ECON_ONLY" if econ and not stat else "FAIL")}

    return {
        "contract": {
            "cohort": str(INTRADAY_COHORT.relative_to(ROOT)),
            "cohort_sha256": hashlib.sha256(
                INTRADAY_COHORT.read_bytes()).hexdigest(),
            "n_records": len(records),
            "assets": sorted({r["asset_id"] for r in records}),
            "label": "label.forward_return_bps (4h forward mark bps, gross)",
            "score": "trailing-90-record mid-rank percentile, per-asset (PIT)",
            "lookback_records": ILOOKBACK,
            "abnvol_median_records": ABN_MEDIAN,
            "hold_hours": HOLD_H,
            "fee_bps_per_side": FEE_BPS,
            "funding_cost": "last_funding_rate*1e4 bps iff "
                            "ns_until_next_funding_settlement <= 4h",
            "welch_quantile": IQ,
            "fdr_alpha": 0.05,
            "fdr_family": INTRADAY_ARMS,
            "folds": "3 contiguous equal-width ns windows (advisory, "
                     "not a frozen validator fold set)",
            "placebo_seeds": PLACEBO_SEEDS},
        "n_eval_rows": len(rows),
        "warmup_dropped": len(records) - len(rows),
        "label_stats_bps": {
            "mean": round(statistics.mean(ys), 3),
            "median": round(statistics.median(ys), 3),
            "sd": round(statistics.pstdev(ys), 3),
            "min": round(min(ys), 2), "max": round(max(ys), 2)},
        "baseline_always_long": always,
        "cells": cells,
        "fdr_pass": fdr,
        "verdicts": verdicts,
        "caveats": [
            "4h forward windows overlap ~4x and hourly serial correlation "
            "shrinks effective n; normal-approx p is optimistic",
            "funding cashflow modeled at settled rate only when a settlement "
            "lands inside the hold — accounting convention, not a fill model",
            "gross mark move minus fees/scheduled funding only — no spread, "
            "slippage, or impact; not a tradability claim",
            "two months, 5 assets — no cross-regime replication"]}


# ---------------------------------------------------------------- cross-scale
def cross_scale(daily, intraday):
    d_sp = daily["cells"]["funding_pct"]["primary"]["spearman"]
    h_sp = intraday["cells"]["funding_pct"]["primary"]["spearman"]
    d_v = daily["verdicts"]["funding_pct"]
    h_v = intraday["verdicts"]["funding_pct"]
    same_sign = (d_sp and h_sp and d_sp["rho"] * h_sp["rho"] > 0)
    if d_v["verdict"] == "PASS" and h_v["verdict"] == "PASS" and same_sign:
        rep = "replicated_both_scales"
    elif same_sign and h_v["placebo_clear"] and d_v["placebo_clear"]:
        rep = "directional_only"
    else:
        rep = "not_replicated"
    return {"cell": "funding_pct",
            "daily": {"rho": d_sp["rho"] if d_sp else None,
                      "p": d_sp["p"] if d_sp else None,
                      "verdict": d_v["verdict"],
                      "placebo_max_abs_rho": max(
                          (abs(x) for x in
                           daily["cells"]["funding_pct"]["placebo_rhos"]
                           if x is not None), default=None)},
            "hourly": {"rho": h_sp["rho"] if h_sp else None,
                       "p": h_sp["p"] if h_sp else None,
                       "n": h_sp["n"] if h_sp else None,
                       "verdict": h_v["verdict"],
                       "placebo_max_abs_rho":
                           intraday["cells"]["funding_pct"]
                           ["placebo_max_abs_rho"]},
            "same_sign": same_sign,
            "replication": rep,
            "note": "funding_pct is the only cell defined at both scales "
                    "with an identical construction (trailing PIT "
                    "percentile of last_funding_rate); horizons differ "
                    "(5d vs 4h) so magnitudes are not expected to match"}


# ---------------------------------------------------------------- run
def run():
    daily = v1.run()
    intraday = run_intraday()
    return {"schema_version": "nanojev-financial-signal-benchmark-v2",
            "status": "benchmark_complete",
            "task": "T91",
            "protocol": "research/financial_signal_benchmark_protocol_v2.json",
            "fdr_family_note": "two separate BH-FDR families at alpha=0.05: "
                               "10 daily cells, 7 intraday cells — no "
                               "cross-scale pooling",
            "daily": daily,
            "intraday": intraday,
            "cross_scale": cross_scale(daily, intraday)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    r1, r2 = run(), run()
    same = json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    r1["determinism"] = {"replays": 2, "byte_identical": same}
    r1["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    blob = json.dumps(r1, indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(blob)
    print(json.dumps({"output": str(args.output), "deterministic": same,
                      "sha256": hashlib.sha256(blob.encode()).hexdigest()},
                     indent=2))


if __name__ == "__main__":
    main()
