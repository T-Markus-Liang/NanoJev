#!/usr/bin/env python3
"""T78: supervised regression on the CONTINUOUS 5d-return label.

First fitted-model step past the falsified binary label. Per-record features
(all trailing-180 mid-rank percentiles, PIT-safe):

  funding_pct, basis_pct, rv_pct, abnvol_pct, taker_pct, funding*basis (cross)

Target: 5-day forward mark log return. Closed-form ridge; lambda selected on
dev; test readout once per frozen fold. Baselines: zero-mean and
train-mean predictors. Metrics: MSE ratio, Spearman(pred,actual),
top-quantile predicted vs rest mean return, sign hit rate.

Protocol: research/financial_signal_fit_protocol_v1.json
Authorization: results/financial_signal_fit_authorization_v1.json
No trading, no promotion claims; fold discipline from the frozen validator.
"""

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_v1/records.jsonl"
FOLDS = ROOT / "research/financial_r1_pit_validator_core_v2.json"
OUT = ROOT / "results/financial_signal_fit_v1.json"
LOOKBACK, HOLD = 180, 5
LAMBDAS = [0.0, 0.1, 1.0, 10.0, 100.0]


def pct(w, x):
    return (sum(1 for v in w if v < x) + 0.5 * sum(1 for v in w if v == x)) / len(w)


def build_rows(records):
    by_asset = defaultdict(list)
    for r in records:
        by_asset[r["asset_id"]].append(r)
    for rs in by_asset.values():
        rs.sort(key=lambda x: x["decision_ns"])
    out = []
    for asset, rs in sorted(by_asset.items()):
        fund = [x["features"]["last_funding_rate"]["value"] for x in rs]
        basis = [x["features"]["mark_index_basis_bps"]["value"] for x in rs]
        rv = [x["features"]["realized_vol_24bar"]["value"] for x in rs]
        vol = [x["features"]["quote_volume"]["value"] for x in rs]
        taker = [x["features"]["taker_buy_ratio"]["value"] for x in rs]
        mark = [x["features"]["mark_price"]["value"] for x in rs]
        for i in range(LOOKBACK, len(rs) - HOLD):
            if mark[i] <= 0 or mark[i + HOLD] <= 0:
                continue
            fp = pct(fund[i - LOOKBACK:i], fund[i])
            bp = pct(basis[i - LOOKBACK:i], basis[i])
            rp = pct(rv[i - LOOKBACK:i], rv[i])
            med = statistics.median(vol[i - 30:i])
            abn = vol[i] / med if med > 0 else 1.0
            aw = []
            for j in range(max(30, i - LOOKBACK), i):
                m = statistics.median(vol[max(0, j - 30):j])
                if m > 0:
                    aw.append(vol[j] / m)
            ap = pct(aw, abn) if aw else 0.5
            tp = pct(taker[i - LOOKBACK:i], taker[i])
            feats = [fp, bp, rp, ap, tp, fp * bp]
            ret = math.log(mark[i + HOLD] / mark[i])
            out.append({"ns": rs[i]["decision_ns"], "asset": asset,
                        "x": feats, "y": ret})
    return out


def ridge_fit(X, y, lam):
    n, d = X.shape
    Xm = np.column_stack([np.ones(n), X])
    A = Xm.T @ Xm + lam * np.eye(d + 1)
    A[0, 0] -= lam  # no penalty on intercept
    return np.linalg.solve(A, Xm.T @ y)


def spearman(xs, ys):
    def rk(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r
    n = len(xs)
    if n < 10:
        return None
    rx, ry = rk(xs), rk(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else None


def run():
    records = [json.loads(l) for l in COHORT.read_text().splitlines() if l.strip()]
    folds = json.loads(FOLDS.read_text())["folds"]
    rows = build_rows(records)
    fold_out = []
    for fi, f in enumerate(folds):
        tr = [r for r in rows if f["train"][0] <= r["ns"] < f["train"][1]]
        dv = [r for r in rows if f["dev"][0] <= r["ns"] < f["dev"][1]]
        te = [r for r in rows if f["test"][0] <= r["ns"] < f["test"][1]]
        if len(tr) < 100 or not dv or not te:
            fold_out.append({"fold": fi, "skipped": True,
                             "n_train": len(tr), "n_dev": len(dv),
                             "n_test": len(te)})
            continue
        Xtr = np.array([r["x"] for r in tr])
        ytr = np.array([r["y"] for r in tr])
        Xdv = np.array([r["x"] for r in dv])
        ydv = np.array([r["y"] for r in dv])
        Xte = np.array([r["x"] for r in te])
        yte = np.array([r["y"] for r in te])
        # standardize on train
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        Xtr, Xdv, Xte = (Xtr - mu) / sd, (Xdv - mu) / sd, (Xte - mu) / sd
        best = None
        for lam in LAMBDAS:
            w = ridge_fit(Xtr, ytr, lam)
            mse_dv = float(np.mean(
                (np.column_stack([np.ones(len(dv)), Xdv]) @ w - ydv) ** 2))
            if best is None or mse_dv < best[0]:
                best = (mse_dv, lam, w)
        mse_dv, lam, w = best
        pred = np.column_stack([np.ones(len(te)), Xte]) @ w
        mse_te = float(np.mean((pred - yte) ** 2))
        base_te = float(np.mean((yte - ytr.mean()) ** 2))
        rho = spearman(pred.tolist(), yte.tolist())
        q = np.quantile(pred, 0.9)
        top = yte[pred >= q]
        rest = yte[pred < q]
        sign_hit = float(np.mean((pred > 0) == (yte > 0)))
        fold_out.append({
            "fold": fi, "lambda": lam, "dev_mse": round(mse_dv, 6),
            "n_train": len(tr), "n_dev": len(dv), "n_test": len(te),
            "test_mse": round(mse_te, 6), "base_mse": round(base_te, 6),
            "mse_ratio_vs_base": round(mse_te / base_te, 4)
            if base_te else None,
            "spearman_pred_actual": round(rho, 4) if rho else None,
            "top10pct_pred_mean_bps": round(float(top.mean()) * 1e4, 1)
            if len(top) else None,
            "rest_mean_bps": round(float(rest.mean()) * 1e4, 1)
            if len(rest) else None,
            "sign_hit_rate": round(sign_hit, 4)})
    return {"schema_version": "nanojev-financial-signal-fit-v1",
            "status": "fit_complete",
            "scope": "ridge on continuous 5d label, frozen folds; "
                    "not a strategy or tradability claim",
            "features": ["funding_pct", "basis_pct", "rv_pct", "abnvol_pct",
                         "taker_pct", "funding_pct*basis_pct"],
            "target": "log(mark[t+5]/mark[t])",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "folds": fold_out}


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
