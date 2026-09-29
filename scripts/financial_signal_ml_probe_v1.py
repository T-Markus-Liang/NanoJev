#!/usr/bin/env python3
"""T125: ridge-regression ML probe on the 287k-row MEGA cohort.

Retry of the T78 supervised attempt (``financial_signal_fit_v1.py``, which
failed on the 5-asset cohort: fold1 reversed, temporally unstable) at proper
cross-sectional depth, now that the T112 mega cohort exists and the signal
is known to be CONDITIONAL (BTC-regime gated, per the T104-T112 XS arms).

Design (frozen):

  target      label.forward_return_5d_bps (continuous, gross close-to-close)
  features    per record, all PIT-safe:
    dfh20            raw precomputed feature (dist from 20-bar high)
    mom20            raw precomputed feature (20-bar momentum)
    vol20            raw precomputed feature (20-bar realized vol)
    funding_pct      per-asset mid-rank pct of last_funding_rate vs its
                     trailing-180 non-null daily values (MIN_WINDOW=20);
                     assets/days without a usable window are imputed to the
                     neutral mid-rank 0.5 (rate reported — ~52 of 277
                     symbols carry funding at all)
    btc_ret20        same-day BTCUSDT 20d close/close-1 from
                     data/rc_futures_v1/BTC/BTCUSDT_1d.csv, broadcast to
                     every asset that day (THE regime variable)
    dfh20_x_btc      dfh20 * btc_ret20 — the known gated interaction
    xs_rank_dfh      same-day cross-sectional mid-rank pct of dfh20

  walkforward strictly temporal, 4 folds:
    train = calendar year t, test = year t+1
    (2021->22, 22->23, 23->24, 24->25)
  embargo   last 5 train days before each fold boundary are dropped
            (5d labels would otherwise overlap the test year); the same
            5-day embargo is applied at every inner CV boundary
  lambda    {0.1, 1, 10, 100} selected INSIDE each train year by
            forward-chaining CV: the year's dates are split into 5
            contiguous chunks, inner folds k=1..4 train on chunks[0:k]
            (minus embargo) and validate on chunk k; lambda = argmin mean
            inner-val MSE; refit on the full embargoed train year.
            Lambda never touches test data.

  arms
    ridge_all      all 7 features, all rows
    gated          same feature set, fit ONLY on btc_ret20>0 train rows,
                   evaluated only on btc_ret20>0 test rows (the known
                   regime-on state)
    xs_rank_only   xs_rank_dfh alone — is a single XS state feature enough?

  metrics per fold per arm: n_train/n_test, chosen lambda, test MSE,
  baseline MSE (train-mean predictor), mse_ratio, Spearman(pred, ret),
  top-decile-pred mean fwd ret vs rest (the economic spread, bps),
  sign accuracy. Plus pooled-OOS readout and a verdict.

  verdict    ML-viable iff some arm shows rho>0 AND top-decile spread>0
             in ALL 4 folds with positive pooled values; otherwise this
             confirms "conditional state feature, not a global predictor"
             (T78) at mega-cohort scale.

Standardization: feature mean/std from the (embargoed) train fold only.

Artifacts follow the established convention: a frozen protocol
(``research/financial_signal_ml_probe_protocol_v1.json``) plus an owner
self-authorization pinning it by sha256
(``results/financial_signal_ml_probe_authorization_v1.json``) are written
on every run BEFORE measurement. Measurement only: no trading, no
promotion claims, no network.
"""

import argparse
import bisect
import datetime as dt
import hashlib
import json
import pathlib
import time
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_mega_v1/records.jsonl"
BTC_CSV = ROOT / "data/rc_futures_v1/BTC/BTCUSDT_1d.csv"
OUT = ROOT / "results/financial_signal_ml_probe_v1.json"
PROTOCOL = ROOT / "research/financial_signal_ml_probe_protocol_v1.json"
AUTH = ROOT / "results/financial_signal_ml_probe_authorization_v1.json"

FUND_LOOKBACK = 180   # trailing window for funding_pct mid-rank
MIN_WINDOW = 20       # repo convention: floor for a usable trailing pct
BTC_LOOKBACK = 20     # btc_ret20 lookback in days
EMBARGO_DAYS = 5      # dropped train days before each boundary (label = 5d)
LAMBDAS = (0.1, 1.0, 10.0, 100.0)
INNER_CHUNKS = 5      # contiguous date chunks inside each train year
TOP_Q = 0.9           # top-decile economic test
FOLD_PAIRS = ((2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025))
FEATURE_NAMES = ("dfh20", "mom20", "vol20", "funding_pct", "btc_ret20",
                 "dfh20_x_btc", "xs_rank_dfh")


def _r(x, nd=4):
    return round(float(x), nd) if x is not None else None


def mid_rank_pct(window, x):
    """Mid-rank percentile of x vs a trailing window (ties count half)."""
    n = len(window)
    if n == 0:
        return None
    less = sum(1 for w in window if w < x)
    eq = sum(1 for w in window if w == x)
    return (less + 0.5 * eq) / n


def load_btc_ret20():
    """date -> BTC close/close[t-20]-1 on the contiguous daily series."""
    dates, closes = [], []
    with BTC_CSV.open() as f:
        header = f.readline()
        for line in f:
            if not line.strip():
                continue
            p = line.split(",")
            dates.append(dt.date.fromisoformat(p[0]))
            closes.append(float(p[4]))
    out = {}
    for i in range(BTC_LOOKBACK, len(closes)):
        out[dates[i]] = closes[i] / closes[i - BTC_LOOKBACK] - 1.0
    return out


def load_rows():
    """Extract the fields needed; returns list of raw row dicts."""
    rows = []
    n_total = 0
    with COHORT.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n_total += 1
            ft = r["features"]
            rows.append({
                "asset": r["asset_id"],
                "date": dt.date.fromisoformat(r["id"].rsplit(":", 1)[-1]),
                "dfh20": ft["dfh20"]["value"],
                "mom20": ft["mom20"]["value"],
                "vol20": ft["vol20"]["value"],
                "funding": ft["last_funding_rate"]["value"],
                "y": r["label"]["forward_return_5d_bps"],
            })
    return rows, n_total


def build_dataset(rows, btc_ret20):
    """Attach funding_pct and xs_rank_dfh; return modelable rows + stats."""
    # --- funding_pct: per-asset trailing-180 mid-rank of non-null funding
    by_asset = defaultdict(list)
    for i, r in enumerate(rows):
        by_asset[r["asset"]].append(i)
    funding_pct = [None] * len(rows)
    for idxs in by_asset.values():
        idxs.sort(key=lambda i: rows[i]["date"])
        window = []
        for i in idxs:
            v = rows[i]["funding"]
            funding_pct[i] = (mid_rank_pct(window, v)
                              if v is not None and len(window) >= MIN_WINDOW
                              else None)
            if v is not None:
                window.append(v)
                if len(window) > FUND_LOOKBACK:
                    window.pop(0)
    # --- xs_rank_dfh: same-day cross-sectional mid-rank of dfh20
    by_date_idx = defaultdict(list)
    for i, r in enumerate(rows):
        if r["dfh20"] is not None:
            by_date_idx[r["date"]].append(i)
    xs_rank = [None] * len(rows)
    for idxs in by_date_idx.values():
        vals_sorted = sorted(rows[i]["dfh20"] for i in idxs)
        n = len(vals_sorted)
        for i in idxs:
            lo = bisect.bisect_left(vals_sorted, rows[i]["dfh20"])
            hi = bisect.bisect_right(vals_sorted, rows[i]["dfh20"])
            xs_rank[i] = (lo + 0.5 * (hi - lo)) / n
    # --- assemble
    out = []
    stats = {"n_rows_total": len(rows),
             "n_funding_pct_imputed": 0,
             "n_dropped_missing_core": 0,
             "n_dropped_no_btc": 0}
    for i, r in enumerate(rows):
        if (r["dfh20"] is None or r["mom20"] is None or r["vol20"] is None
                or r["y"] is None):
            stats["n_dropped_missing_core"] += 1
            continue
        b = btc_ret20.get(r["date"])
        if b is None:
            stats["n_dropped_no_btc"] += 1
            continue
        fp = funding_pct[i]
        if fp is None:
            fp = 0.5
            stats["n_funding_pct_imputed"] += 1
        out.append({
            "date": r["date"], "asset": r["asset"],
            "x": (r["dfh20"], r["mom20"], r["vol20"], fp, b,
                  r["dfh20"] * b, xs_rank[i]),
            "y": r["y"], "btc_ret20": b,
        })
    stats["n_modelable"] = len(out)
    return out, stats


def ridge_fit(X, y, lam):
    n, d = X.shape
    Xm = np.column_stack([np.ones(n), X])
    A = Xm.T @ Xm + lam * np.eye(d + 1)
    A[0, 0] -= lam  # intercept unpenalized
    return np.linalg.solve(A, Xm.T @ y)


def ridge_pred(X, w):
    return np.column_stack([np.ones(X.shape[0]), X]) @ w


def rankdata(v):
    """Average-tie ranks, numpy."""
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=float)
    sv = v[order]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x, y):
    n = len(x)
    if n < 10:
        return None
    rx, ry = rankdata(np.asarray(x, float)), rankdata(np.asarray(y, float))
    rx -= rx.mean()
    ry -= ry.mean()
    dx = float(np.sqrt((rx ** 2).sum()))
    dy = float(np.sqrt((ry ** 2).sum()))
    if dx == 0 or dy == 0:
        return None
    return float((rx * ry).sum() / (dx * dy))


def select_lambda(Xtr, ytr, dates_tr, gate_mask=None):
    """Forward-chaining CV inside the train year; returns best lambda."""
    uniq = np.array(sorted(set(dates_tr.tolist())))
    chunks = np.array_split(uniq, INNER_CHUNKS)
    mse_by_lam = {lam: [] for lam in LAMBDAS}
    for k in range(1, INNER_CHUNKS):
        boundary = chunks[k][0]
        embargo_start = boundary - dt.timedelta(days=EMBARGO_DAYS)
        tr_m = dates_tr < embargo_start
        va_m = (dates_tr >= boundary) & (dates_tr < chunks[k][-1]
                                       + dt.timedelta(days=1))
        if gate_mask is not None:
            tr_m &= gate_mask
            va_m &= gate_mask
        Xi, yi = Xtr[tr_m], ytr[tr_m]
        Xv, yv = Xtr[va_m], ytr[va_m]
        if len(Xi) < 200 or len(Xv) < 50:
            continue
        for lam in LAMBDAS:
            w = ridge_fit(Xi, yi, lam)
            mse_by_lam[lam].append(float(np.mean((ridge_pred(Xv, w) - yv)
                                                 ** 2)))
    scored = [(float(np.mean(v)), lam) for lam, v in mse_by_lam.items() if v]
    if not scored:
        return LAMBDAS[1], {}
    scored.sort()
    return scored[0][1], {str(lam): [round(m, 2) for m in v]
                          for lam, v in mse_by_lam.items()}


def eval_fold(pred, yte, train_mean):
    mse = float(np.mean((pred - yte) ** 2))
    base = float(np.mean((yte - train_mean) ** 2))
    rho = spearman(pred, yte)
    q = np.quantile(pred, TOP_Q)
    top = yte[pred >= q]
    rest = yte[pred < q]
    return {
        "n_test": int(len(yte)),
        "test_mse_bps2": _r(mse, 2),
        "baseline_mse_bps2": _r(base, 2),
        "mse_ratio_vs_train_mean": _r(mse / base) if base else None,
        "spearman_pred_ret": _r(rho),
        "top_decile_mean_bps": _r(top.mean(), 2) if len(top) else None,
        "rest_mean_bps": _r(rest.mean(), 2) if len(rest) else None,
        "top_decile_spread_bps": _r(top.mean() - rest.mean(), 2)
        if len(top) and len(rest) else None,
        "sign_accuracy": _r(np.mean((pred > 0) == (yte > 0))),
    }


def run_arm(name, rows, feat_idx, gated):
    """One arm across all walkforward folds."""
    feats = np.array([[r["x"][j] for j in feat_idx] for r in rows])
    y = np.array([r["y"] for r in rows])
    dates = np.array([r["date"] for r in rows])
    btc = np.array([r["btc_ret20"] for r in rows])
    folds, all_pred, all_y = [], [], []
    sse_model = sse_base = 0.0
    for ty, ny in FOLD_PAIRS:
        boundary = dt.date(ny, 1, 1)
        embargo_start = boundary - dt.timedelta(days=EMBARGO_DAYS)
        tr_m = (dates >= dt.date(ty, 1, 1)) & (dates < embargo_start)
        te_m = (dates >= boundary) & (dates < dt.date(ny + 1, 1, 1))
        gate = btc > 0
        if gated:
            te_eval = te_m & gate
        else:
            te_eval = te_m
        Xtr_raw, ytr = feats[tr_m], y[tr_m]
        dtr = dates[tr_m]
        gtr = gate[tr_m] if gated else None
        if gated:
            fit_m = gtr
        else:
            fit_m = np.ones(len(Xtr_raw), bool)
        Xte, yte = feats[te_eval], y[te_eval]
        if fit_m.sum() < 500 or len(Xte) < 50:
            folds.append({"train_year": ty, "test_year": ny,
                          "skipped": True,
                          "n_train": int(fit_m.sum()),
                          "n_test": int(len(Xte))})
            continue
        # standardize on the rows the arm actually fits
        mu = Xtr_raw[fit_m].mean(0)
        sd = Xtr_raw[fit_m].std(0)
        sd[sd == 0] = 1.0
        Xtr = (Xtr_raw - mu) / sd
        Xte_s = (Xte - mu) / sd
        lam, cv_detail = select_lambda(Xtr, ytr, dtr, gtr)
        w = ridge_fit(Xtr[fit_m], ytr[fit_m], lam)
        pred = ridge_pred(Xte_s, w)
        train_mean = float(ytr[fit_m].mean())
        ev = eval_fold(pred, yte, train_mean)
        sse_model += float(((pred - yte) ** 2).sum())
        sse_base += float(((yte - train_mean) ** 2).sum())
        ev.update({"train_year": ty, "test_year": ny, "lambda": lam,
                   "n_train": int(fit_m.sum()),
                   "inner_cv_mse_bps2_by_lambda": cv_detail,
                   "coef": {FEATURE_NAMES[j] if name != "xs_rank_only"
                            else "xs_rank_dfh":
                            _r(w[k + 1], 6)
                            for k, j in enumerate(feat_idx)}})
        folds.append(ev)
        all_pred.append(pred)
        all_y.append(yte)
    pooled = None
    if all_pred:
        p = np.concatenate(all_pred)
        t = np.concatenate(all_y)
        pooled = {"n_test": int(len(t)),
                  "spearman_pred_ret": _r(spearman(p, t)),
                  "mse_ratio_vs_fold_train_means":
                      _r(sse_model / sse_base) if sse_base else None}
        q = np.quantile(p, TOP_Q)
        pooled["top_decile_spread_bps"] = _r(
            t[p >= q].mean() - t[p < q].mean(), 2)
        pooled["sign_accuracy"] = _r(np.mean((p > 0) == (t > 0)))
    pos_rho = sum(1 for f in folds
                  if not f.get("skipped")
                  and (f["spearman_pred_ret"] or 0) > 0)
    pos_spread = sum(1 for f in folds
                     if not f.get("skipped")
                     and (f["top_decile_spread_bps"] or 0) > 0)
    n_ok = sum(1 for f in folds if not f.get("skipped"))
    return {"arm": name, "features": [FEATURE_NAMES[j] for j in feat_idx],
            "gated_on_btc_ret20_pos": gated, "folds": folds,
            "pooled_oos": pooled,
            "folds_rho_positive": pos_rho, "folds_spread_positive": pos_spread,
            "folds_evaluated": n_ok,
            "consistent": bool(n_ok == len(FOLD_PAIRS)
                               and pos_rho == n_ok and pos_spread == n_ok
                               and pooled
                               and (pooled["spearman_pred_ret"] or 0) > 0
                               and (pooled["top_decile_spread_bps"] or 0) > 0)}


def write_protocol_and_auth():
    protocol = {
        "schema_version": "nanojev-financial-signal-ml-probe-protocol-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": ("T125: retry the T78 ridge-regression ML attempt on the "
                    "287k-row mega cohort now that the signal is known to be "
                    "BTC-regime conditional. Ridge on continuous 5d fwd bps; "
                    "features dfh20, mom20, vol20, funding_pct(trailing-180 "
                    "mid-rank), btc_ret20 (broadcast), dfh20*btc_ret20 "
                    "interaction, xs_rank_dfh. Strictly temporal walkforward "
                    "2021->22/22->23/23->24/24->25 with 5-day embargo; lambda "
                    "{0.1,1,10,100} via in-train-year forward-chaining CV "
                    "only. Arms: ridge_all, btc_ret20>0-gated, xs_rank_dfh-"
                    "only. Verdict: consistent positive OOS rho + top-decile "
                    "spread across all folds = first ML-viable formulation; "
                    "else confirms T78 at scale. Measurement only."),
        "cohort": {
            "path": "data/perp_pit_mega_v1/records.jsonl",
            "assets": "277 USDT-M perps (incl. delisted/renamed early-stops)",
            "target": "label.forward_return_5d_bps (gross close-to-close)",
            "btc_series": "data/rc_futures_v1/BTC/BTCUSDT_1d.csv",
        },
        "definitions": {
            "funding_pct": "per-asset mid-rank pct of last_funding_rate vs "
                           "trailing-180 non-null values (MIN_WINDOW=20); "
                           "unusable -> neutral 0.5 imputation, rate reported",
            "btc_ret20": "BTCUSDT close/close[t-20]-1 on the decision date, "
                         "broadcast to all assets",
            "xs_rank_dfh": "same-day cross-sectional mid-rank pct of dfh20",
            "embargo": "train rows in the last 5 days before each fold "
                       "boundary dropped (5d label overlap); same embargo at "
                       "inner CV boundaries",
            "inner_cv": "train-year dates -> 5 contiguous chunks; inner folds "
                        "k=1..4 fit chunks[0:k]-embargo, validate chunk k; "
                        "lambda = argmin mean val MSE; never touches test",
            "top_decile": "pred >= quantile(pred,0.9) vs rest, mean fwd bps",
            "baseline": "train-mean predictor MSE for mse_ratio",
        },
        "statistics": {
            "caveats": [
                "5d overlapping labels autocorrelate ~4/5 of adjacent days; "
                "all metrics nominal",
                "pooled OOS spearman concatenates folds with different "
                "train-fitted scales — descriptive, not a test",
                "funding_pct imputed for ~70% of rows (funding-covered "
                "subset is ~52/277 symbols)",
                "gated arm evaluates a state-conditional subset by design — "
                "its spread is not comparable to a full-universe strategy",
            ],
        },
        "forbidden": ["trading", "profitability claims",
                      "protocol edits post-run", "network",
                      "lambda/model selection on test data"],
    }
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    auth = {
        "schema_version":
            "nanojev-financial-signal-ml-probe-authorization-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision": "approved_for_measurement",
        "protocol_path": "research/financial_signal_ml_probe_protocol_v1.json",
        "protocol_sha256": sha,
        "measurement_authorized": True,
        "fit_authorized": True,
        "live_trading_authorized": False,
        "order_submission_authorized": False,
        "network_model_calls": 0,
        "independent_reviewer": {
            "id": "project-owner",
            "independence": "owner_self_authorization_not_independent_review",
            "note": "Owner directed T125: retry the ridge-regression ML probe "
                    "on the mega cohort with the regime-interaction feature "
                    "set (delegated task).",
        },
        "scope": {
            "permitted": "PIT-safe ridge fitting + OOS measurement on "
                         "data/perp_pit_mega_v1/records.jsonl with the BTC "
                         "series as regime feature, per the pinned protocol: "
                         "3 arms x 4 walkforward folds, lambda via "
                         "in-train-year CV only.",
            "not_permitted": "No trading/profitability claims/protocol edits; "
                             "no network; no other files modified.",
        },
    }
    AUTH.parent.mkdir(parents=True, exist_ok=True)
    AUTH.write_text(json.dumps(auth, indent=2, sort_keys=True) + "\n")
    return sha


def run():
    protocol_sha = write_protocol_and_auth()
    t0 = time.time()
    btc_ret20 = load_btc_ret20()
    raw, n_total = load_rows()
    rows, dstats = build_dataset(raw, btc_ret20)
    arms = [
        run_arm("ridge_all", rows, tuple(range(7)), gated=False),
        run_arm("gated_btc_ret20_pos", rows, tuple(range(7)), gated=True),
        run_arm("xs_rank_only", rows, (6,), gated=False),
    ]
    viable = [a["arm"] for a in arms if a["consistent"]]
    verdict = (
        "ML_VIABLE: %s produced consistent positive OOS rho and positive "
        "top-decile spread in all 4 walkforward folds — first ML-viable "
        "formulation at mega-cohort scale." % ", ".join(viable)
        if viable else
        "NO ML-VIABLE CONFIGURATION: no arm produced consistent positive OOS "
        "rho + positive top-decile spread across all folds — confirms the "
        "T78 read (conditional state feature, not a global predictor) at "
        "287k-row scale.")
    return {
        "schema_version": "nanojev-financial-signal-ml-probe-v1",
        "task": "T125",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_s": _r(time.time() - t0, 1),
        "protocol_path": "research/financial_signal_ml_probe_protocol_v1.json",
        "protocol_sha256": protocol_sha,
        "authorization_path":
            "results/financial_signal_ml_probe_authorization_v1.json",
        "data": {"cohort": str(COHORT.relative_to(ROOT)),
                 "btc_series": str(BTC_CSV.relative_to(ROOT)),
                 "n_records": n_total, **dstats},
        "design": {"target": "forward_return_5d_bps",
                   "features": list(FEATURE_NAMES),
                   "fold_pairs": ["%d->%d" % p for p in FOLD_PAIRS],
                   "embargo_days": EMBARGO_DAYS,
                   "lambdas": list(LAMBDAS),
                   "lambda_selection": "in-train-year forward-chaining CV "
                                       "(5 chunks, 4 inner folds, 5d inner "
                                       "embargo); never test"},
        "arms": arms,
        "verdict": verdict,
        "scope": "measurement only; not a strategy or tradability claim",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=pathlib.Path, default=OUT)
    args = ap.parse_args()
    report = run()
    blob = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(blob)
    print(json.dumps({"output": str(args.output),
                      "sha256": hashlib.sha256(blob.encode()).hexdigest(),
                      "verdict": report["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
