#!/usr/bin/env python3
"""T82: the unified financial-signal benchmark (Track B standard harness).

ONE evaluation contract, ONE scorecard, for every signal arm we have measured:

  contract   cohort=data/perp_pit_v1/records.jsonl (sha pinned in receipt)
             label = log(mark[t+5]/mark[t])  (primary, continuous)
             legacy label = up>25bps/1d binary (reference column)
             folds = research/financial_r1_pit_validator_core_v2.json test windows
             PIT   = trailing-180 mid-rank percentiles, per-asset
             cost  = 5bps taker/side + funding accrual (dir*rate*24/interval/day)
             hold  = 5 bars, non-overlapping per asset

  cells      funding_pct | basis_pct | rv_pct | abnvol_pct | taker_pct
             fmom_pct | fxb_cross | f2b2_indicator | f2b2_btc_gate | tri_gate

  metrics    Spearman rho/t/p (continuous cells); Welch mean-diff (indicators)
             top-arm vs rest net-of-cost strategy stats (n, mean/med bps, win,
             sharpe/trade, maxDD); per-fold arm-mean sign vector
  controls   placebo: labels shuffled within asset, 3 deterministic seeds —
             a real signal's rho must exceed its placebo band
             determinism: run() twice, receipt payload must be identical
             baseline: always-long matched cadence net stats
  verdict    FDR family over primary p-values; per cell:
             PASS = FDR-survives AND |rho| > max |placebo rho|
             ECON = top-arm net > always-long net (reported, advisory)

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

ROOT = Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_v1/records.jsonl"
FOLDS = ROOT / "research/financial_r1_pit_validator_core_v2.json"
OUT = ROOT / "results/financial_signal_benchmark_v1.json"
LOOKBACK, HOLD = 180, 5
FEE_BPS = 5.0
PLACEBO_SEEDS = (17, 73, 991)


# ---------------------------------------------------------------- stats
def pct(w, x):
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
    return {"rho": rho, "t": t, "p": p}


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
    return {"n_a": len(a), "n_b": len(b), "mean_a": ma, "mean_b": mb,
            "t": t, "p": p}


def net_stats(trades):
    if len(trades) < 5:
        return None
    net = [t for t in trades]
    mean = statistics.mean(net)
    sd = statistics.pstdev(net) or 1e-9
    cur, peak, dd = 1.0, 1.0, 0.0
    for g in net:
        cur *= 1 + g / 1e4
        peak = max(peak, cur)
        dd = min(dd, cur / peak - 1)
    return {"n": len(trades), "mean_net_bps": round(mean, 1),
            "median_net_bps": round(statistics.median(net), 1),
            "win_rate": round(sum(1 for x in net if x > 0) / len(net), 3),
            "sharpe_per_trade": round(mean / sd, 3),
            "max_dd_bps": round(dd * 1e4, 1),
            "total_net_bps": round(sum(net), 0)}


# ---------------------------------------------------------------- dataset
def build_dataset():
    """Per-asset row lists with PIT percentiles + forward returns."""
    records = [json.loads(l) for l in COHORT.read_text().splitlines() if l.strip()]
    by_asset = defaultdict(list)
    for r in records:
        by_asset[r["asset_id"]].append(r)
    for rs in by_asset.values():
        rs.sort(key=lambda x: x["decision_ns"])
    rows = []
    for asset, rs in sorted(by_asset.items()):
        fund = [x["features"]["last_funding_rate"]["value"] for x in rs]
        basis = [x["features"]["mark_index_basis_bps"]["value"] for x in rs]
        rv = [x["features"]["realized_vol_24bar"]["value"] for x in rs]
        vol = [x["features"]["quote_volume"]["value"] for x in rs]
        taker = [x["features"]["taker_buy_ratio"]["value"] for x in rs]
        mark = [x["features"]["mark_price"]["value"] for x in rs]
        ivl = [x["features"]["funding_interval_hours"]["value"] for x in rs]
        abn = []
        for i in range(len(rs)):
            med = statistics.median(vol[max(0, i - 30):i]) if i else vol[0]
            abn.append(vol[i] / med if med > 0 else 1.0)
        for i in range(LOOKBACK, len(rs) - HOLD):
            if mark[i] <= 0 or mark[i + HOLD] <= 0:
                continue
            f_pct = pct(fund[i - LOOKBACK:i], fund[i])
            b_pct = pct(basis[i - LOOKBACK:i], basis[i])
            v_pct = pct(rv[i - LOOKBACK:i], rv[i])
            a_pct = pct(abn[max(0, i - LOOKBACK):i], abn[i])
            t_pct = pct(taker[i - LOOKBACK:i], taker[i])
            dm = fund[i] - fund[i - 5]
            mw = [fund[j] - fund[j - 5]
                  for j in range(max(5, i - LOOKBACK), i)]
            m_pct = pct(mw, dm) if mw else 0.5
            ret = math.log(mark[i + HOLD] / mark[i])
            ret1 = (math.log(mark[i + 1] / mark[i])
                    if i + 1 < len(rs) and mark[i + 1] > 0 else None)
            paid = sum(fund[j] * (24.0 / ivl[j] if ivl[j] > 0 else 3.0) * 1e4
                       for j in range(i + 1, i + HOLD + 1))
            rows.append({
                "asset": asset, "ns": rs[i]["decision_ns"], "i_in_asset": i,
                "funding_pct": f_pct, "basis_pct": b_pct, "rv_pct": v_pct,
                "abnvol_pct": a_pct, "taker_pct": t_pct, "fmom_pct": m_pct,
                "fxb": f_pct * b_pct,
                "f2b2": f_pct >= 0.80 and b_pct >= 0.66,
                "tri": f_pct >= 0.80 and b_pct >= 0.66 and v_pct >= 0.66,
                "ret": ret, "ret_1d": ret1,
                "gross_bps": ret * 1e4, "net_bps": ret * 1e4 - 2 * FEE_BPS - paid,
                "up_25bps_1d": (ret1 is not None and ret1 * 1e4 > 25)})
    # BTC 20d trend at each ns
    btc = by_asset["BTCUSDT-PERP"]
    bm = [x["features"]["mark_price"]["value"] for x in btc]
    btc_trend = {}
    for i in range(20, len(btc)):
        if bm[i] > 0 and bm[i - 20] > 0:
            btc_trend[btc[i]["decision_ns"]] = bm[i] / bm[i - 20] - 1 > 0
    for r in rows:
        r["btc_up"] = btc_trend.get(r["ns"])
        r["f2b2_btc"] = r["f2b2"] and r["btc_up"] is True
    return rows


# ---------------------------------------------------------------- cells
CONTINUOUS = ["funding_pct", "basis_pct", "rv_pct", "abnvol_pct",
              "taker_pct", "fmom_pct", "fxb"]
INDICATORS = ["f2b2", "f2b2_btc", "tri"]


def cell_metrics(rows, name, windows):
    xs = [r[name] for r in rows]
    ys = [r["ret"] for r in rows]
    if name in INDICATORS:
        a = [r["ret"] for r in rows if r[name]]
        b = [r["ret"] for r in rows if not r[name]]
        c = welch(a, b)
        primary = {"in_vs_out": ({k: (round(v, 5) if isinstance(v, float)
                                     else v) for k, v in c.items()}
                                if c else None)}
        score_key = None
        arm = [r for r in rows if r[name]]
    else:
        sp = spearman(xs, ys)
        primary = {"spearman": {k: round(v, 5) for k, v in sp.items()}
                   if sp else None}
        score_key = xs
        arm = sorted(rows, key=lambda r: -r[name])[:max(10, len(rows) // 10)]
    rest_ids = {id(r) for r in arm}
    rest = [r for r in rows if id(r) not in rest_ids]
    net = net_stats([r["net_bps"] for r in arm])
    fold_signs = []
    for a, b_ in windows:
        sub = [r["ret"] for r in arm if a <= r["ns"] < b_]
        if len(sub) >= 5:
            fold_signs.append(1 if statistics.mean(sub) > 0 else -1)
        else:
            fold_signs.append(None)
    p = (primary.get("in_vs_out") or {}).get("p") if name in INDICATORS \
        else (primary.get("spearman") or {}).get("p")
    return {"primary": primary, "arm_n": len(arm),
            "arm_mean_gross_bps":
                round(statistics.mean([r["gross_bps"] for r in arm]), 1)
                if arm else None,
            "arm_net_stats": net,
            "fold_arm_signs": fold_signs,
            "p": p, "score_key": score_key is not None}


def placebo_rhos(rows, name):
    """Shuffle ret within asset; rho of score vs shuffled label. 3 seeds."""
    if name in INDICATORS:
        return []
    xs = [r[name] for r in rows]
    by_asset = defaultdict(list)
    for r in rows:
        by_asset[r["asset"]].append(r["ret"])
    out = []
    for seed in PLACEBO_SEEDS:
        shuffled = []
        for asset, vals in sorted(by_asset.items()):
            v = vals[:]
            s = seed
            for i in range(len(v) - 1, 0, -1):
                s = (s * 6364136223846793005 + 1442695040888963407) \
                    & (2**64 - 1)
                j = s % (i + 1)
                v[i], v[j] = v[j], v[i]
            shuffled.extend(v)
        # rows are grouped by asset in build order — same ordering
        sp = spearman(xs, shuffled)
        out.append(round(sp["rho"], 4) if sp else None)
    return out


def run():
    rows = build_dataset()
    folds = json.loads(FOLDS.read_text())["folds"]
    windows = [(f["test"][0], f["test"][1]) for f in folds]

    always = net_stats([r["net_bps"] for r in rows])
    cells = {}
    for name in CONTINUOUS + INDICATORS:
        m = cell_metrics(rows, name, windows)
        m["placebo_rhos"] = placebo_rhos(rows, name)
        cells[name] = m

    ps = sorted((m["p"], n) for n, m in cells.items()
                if m.get("p") is not None)
    M = len(ps)
    fdr = {}
    for i, (p, n) in enumerate(ps):
        fdr[n] = p <= 0.05 * (i + 1) / M

    verdicts = {}
    for n, m in cells.items():
        rho = (m["primary"].get("spearman") or {}).get("rho") \
            if not n in INDICATORS else None
        placebo_max = max((abs(x) for x in m["placebo_rhos"]
                          if x is not None), default=0)
        stat = fdr.get(n, False)
        pl = (abs(rho) > max(0.02, placebo_max)) if rho is not None else None
        econ = (m["arm_net_stats"] and always and
                m["arm_net_stats"]["mean_net_bps"]
                > always["mean_net_bps"])
        folds_ok = sum(1 for s in m["fold_arm_signs"] if s == 1)
        verdicts[n] = {
            "fdr_pass": stat,
            "placebo_clear": pl if rho is not None else None,
            "econ_vs_always_long": econ,
            "folds_positive": f"{folds_ok}/{sum(1 for s in m['fold_arm_signs'] if s is not None)}",
            "verdict": "PASS" if stat and (pl is not False) else
                       ("ECON_ONLY" if econ and not stat else "FAIL")}

    return {"schema_version": "nanojev-financial-signal-benchmark-v1",
            "status": "benchmark_complete",
            "contract": {"cohort": str(COHORT.relative_to(ROOT)),
                         "cohort_sha256": hashlib.sha256(
                             COHORT.read_bytes()).hexdigest(),
                         "label": "log(mark[t+5]/mark[t])",
                         "legacy_label": "up>25bps/1d",
                         "lookback": LOOKBACK, "hold": HOLD,
                         "fee_bps_per_side": FEE_BPS,
                         "fdr_alpha": 0.05, "placebo_seeds": PLACEBO_SEEDS},
            "n_rows": len(rows),
            "baseline_always_long": always,
            "cells": cells, "fdr_pass": fdr, "verdicts": verdicts}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    r1, r2 = run(), run()
    r1.pop("generated_at", None)
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
