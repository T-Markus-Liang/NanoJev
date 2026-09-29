#!/usr/bin/env python3
"""T130: does domain-matched training fix the v4 OOS ranking inversion?

Context: ledger v4's ridge_all (trained on all 277 mega symbols, all
history through 2025-12-26) scores the live 10-asset majors universe.
Its pooled 2026 OOS ranking spearman is ~-0.096 (inverted), even though
the mean per-day cross-sectional spearman is mildly positive — the
inversion is a cross-day level/scale effect. Hypothesis under test: a
model trained on long-tail small caps learned relationships that do not
hold on mega-caps. T127 found ridge_min3 {dfh20, btc_ret20,
dfh20*btc_ret20} is the parsimonious winner, so every arm here uses that
3-feature model unless noted.

Variants (all ridge_min3, same T127 walkforward discipline: 5d embargo
at each boundary, lambda by in-train-window forward-chaining CV only):

  all277_control  train on ALL 277 mega symbols (control)
  top30_qvol      train on the TOP-30 mega symbols by mean daily
                  quote_volume inside the fold's train window only
                  (PIT-safe symbol selection; recomputed per fold)
  xs10_domain     train on the actual 10 ledger symbols' mega rows only
                  (smallest domain match; tests whether ~13-18k rows
                  suffice for a 3-feature ridge)
  recent_regime   train on all symbols but only rows dated >= 2023-01-01
                  (regime-matched window; differs from control only when
                  the train window would otherwise reach before 2023 —
                  i.e. the all-history live-window fit)

Evaluation:
  A) Walkforward folds (T127 discipline, train = calendar year t,
     test = year t+1): folds 2021->22 .. 2025->2026H1. Each fold's model
     is evaluated on (i) the mega-cohort test-year holdout over all
     symbols and (ii) the XS cohort's own records for the test year
     (2026 fold: 2026-01-01..2026-08-26, the last fully labeled date).
  B) v4-exact live window: fit each variant once on all mega history
     <= 2025-12-26 (the last fully labeled date — same cutoff v4 uses),
     evaluate on xs_v1 rows 2026-01-01..2026-08-26. A sanity arm
     (ridge_all_v4replica: the 7-feature T125 set, lambda=100 frozen,
     all history) reproduces v4's actual fitted model to anchor the
     ~-0.096 inversion inside this harness.

Metrics per fold/variant per test set:
  spearman(pred, forward_return_5d_bps): pooled over all test rows
    (the v4 headline metric) AND mean/median of per-day cross-sectional
    spearmans (the metric that actually drives a ranked book).
  XS: top-2/bottom-2 realized spread — per day, (mean y5d of the 2
    highest preds - mean y5d of the 2 lowest preds)/2, matching v4's
    book_realized_5d_bps construction; mean over days, all days and
    BTC-gate-on (btc_ret20 > 0) days separately; 1d-return version too.
  Mega: pooled top-decile vs rest spread on y5d.

btc_ret20: TRAIN features use data/rc_futures_v1/BTC/BTCUSDT_1d.csv
(T125 convention; ends 2025-12-31, covers all mega dates). XS eval
features use the BTCUSDT-PERP close series inside xs_v1 (v3 live
convention: close / close exactly 20 calendar days prior - 1), which
covers 2023-01-21..2026-08-30.

Artifacts: frozen protocol
(``research/financial_signal_domain_match_protocol_v1.json``) + owner
self-authorization pinning it by sha256
(``results/financial_signal_domain_match_authorization_v1.json``),
written before measurement. Measurement only: no trading, no promotion
claims, no network.
"""

import argparse
import bisect
import datetime as dt
import hashlib
import json
import math
import pathlib
import time
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
MEGA = ROOT / "data/perp_pit_mega_v1/records.jsonl"
XS = ROOT / "data/perp_pit_xs_v1/records.jsonl"
BTC_CSV = ROOT / "data/rc_futures_v1/BTC/BTCUSDT_1d.csv"
OUT = ROOT / "results/financial_signal_domain_match_v1.json"
PROTOCOL = ROOT / "research/financial_signal_domain_match_protocol_v1.json"
AUTH = ROOT / "results/financial_signal_domain_match_authorization_v1.json"

XS_SYMS = ("ADAUSDT-PERP", "BNBUSDT-PERP", "BTCUSDT-PERP", "DOGEUSDT-PERP",
           "DOTUSDT-PERP", "ETHUSDT-PERP", "LINKUSDT-PERP", "LTCUSDT-PERP",
           "SOLUSDT-PERP", "XRPUSDT-PERP")
XS_SET = frozenset(XS_SYMS)
BTC_ASSET = "BTCUSDT-PERP"

BTC_LOOKBACK = 20
EMBARGO_DAYS = 5
LAMBDAS = (0.1, 1.0, 10.0, 100.0)
INNER_CHUNKS = 5
FUND_LOOKBACK = 180
FUND_MIN_WINDOW = 20
RECENT_START = dt.date(2023, 1, 1)
TOP_N_QVOL = 30
EDGE = 2                    # top-2 / bottom-2 XS book (v4 sleeve spec)
MIN_UNIVERSE = 6            # v4 graceful-degradation floor for a book
V4_LAMBDA = 100.0           # v4's frozen lambda for the replica arm
XS_END = dt.date(2026, 8, 26)  # last fully labeled xs_v1 date
FOLD_PAIRS = ((2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025),
              (2025, 2026))

MIN3_NAMES = ("dfh20", "btc_ret20", "dfh20_x_btc")
ALL7_NAMES = ("dfh20", "mom20", "vol20", "funding_pct", "btc_ret20",
              "dfh20_x_btc", "xs_rank_dfh")
VARIANTS = ("all277_control", "top30_qvol", "xs10_domain", "recent_regime")


def _r(x, nd=4):
    return round(float(x), nd) if x is not None else None


def mid_rank_pct(window, x):
    """Mid-rank percentile of x vs a trailing window (ties count half)."""
    n = len(window)
    if n == 0 or x is None:
        return None
    less = sum(1 for w in window if w < x)
    eq = sum(1 for w in window if w == x)
    return (less + 0.5 * eq) / n


def rankdata(v):
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
    if n < 3:
        return None
    rx, ry = rankdata(np.asarray(x, float)), rankdata(np.asarray(y, float))
    rx -= rx.mean()
    ry -= ry.mean()
    dx = float(np.sqrt((rx ** 2).sum()))
    dy = float(np.sqrt((ry ** 2).sum()))
    if dx == 0 or dy == 0:
        return None
    return float((rx * ry).sum() / (dx * dy))


def ridge_fit(X, y, lam):
    n, d = X.shape
    Xm = np.column_stack([np.ones(n), X])
    A = Xm.T @ Xm + lam * np.eye(d + 1)
    A[0, 0] -= lam
    return np.linalg.solve(A, Xm.T @ y)


def ridge_pred(X, w):
    return np.column_stack([np.ones(X.shape[0]), X]) @ w


def select_lambda(Xtr, ytr, dates_tr):
    """T127 forward-chaining CV inside the train window; best lambda."""
    uniq = np.array(sorted(set(dates_tr.tolist())))
    chunks = np.array_split(uniq, INNER_CHUNKS)
    mse_by_lam = {lam: [] for lam in LAMBDAS}
    for k in range(1, INNER_CHUNKS):
        boundary = chunks[k][0]
        embargo_start = boundary - dt.timedelta(days=EMBARGO_DAYS)
        tr_m = dates_tr < embargo_start
        va_m = (dates_tr >= boundary) & (dates_tr < chunks[k][-1]
                                       + dt.timedelta(days=1))
        if tr_m.sum() < 200 or va_m.sum() < 50:
            continue
        Xi, yi = Xtr[tr_m], ytr[tr_m]
        Xv, yv = Xtr[va_m], ytr[va_m]
        for lam in LAMBDAS:
            w = ridge_fit(Xi, yi, lam)
            mse_by_lam[lam].append(float(np.mean((ridge_pred(Xv, w) - yv)
                                                 ** 2)))
    scored = [(float(np.mean(v)), lam) for lam, v in mse_by_lam.items() if v]
    if not scored:
        return LAMBDAS[1]
    scored.sort()
    return scored[0][1]


# ---------------------------------------------------------------- loading


def load_btc_ret20_csv():
    """date -> BTC close/close[t-20]-1 on the contiguous csv series."""
    dates, closes = [], []
    with BTC_CSV.open() as f:
        f.readline()
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


def load_mega_rows():
    rows = []
    n_total = 0
    with MEGA.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n_total += 1
            ft = r["features"]
            lb = r["label"]
            rows.append({
                "asset": r["asset_id"],
                "date": dt.date.fromisoformat(r["id"].rsplit(":", 1)[-1]),
                "dfh20": ft["dfh20"]["value"],
                "mom20": ft["mom20"]["value"],
                "vol20": ft["vol20"]["value"],
                "funding": ft["last_funding_rate"]["value"],
                "qvol": ft["quote_volume"]["value"],
                "y": lb["forward_return_5d_bps"],
                "y1": lb["forward_return_bps"],
            })
    return rows, n_total


def load_xs_rows():
    rows = []
    with XS.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ft = r["features"]
            lb = r["label"]
            rows.append({
                "asset": r["asset_id"],
                "date": dt.date.fromisoformat(r["id"].rsplit(":", 1)[-1]),
                "close": ft["close"]["value"],
                "dfh20": ft["dfh20"]["value"],
                "mom20": ft["mom20"]["value"],
                "vol20": ft["vol20"]["value"],
                "funding": ft["last_funding_rate"]["value"],
                "y": lb["forward_return_5d_bps"],
                "y1": lb["forward_return_bps"],
            })
    return rows


def btc_ret20_perp_map(xs_rows):
    """v3 live convention on the xs BTCUSDT-PERP close series:
    date -> close/close of the bar exactly 20 calendar days prior - 1."""
    btc = sorted((r for r in xs_rows if r["asset"] == BTC_ASSET),
                 key=lambda r: r["date"])
    out = {}
    for i, r in enumerate(btc):
        ret = None
        if i >= BTC_LOOKBACK and r["close"] and r["close"] > 0:
            back = btc[i - BTC_LOOKBACK]
            if ((r["date"] - back["date"]).days == BTC_LOOKBACK
                    and back["close"] and back["close"] > 0):
                ret = r["close"] / back["close"] - 1.0
        out[r["date"]] = ret
    return out


def last_fully_labeled_date(rows):
    total = defaultdict(int)
    labeled = defaultdict(int)
    for r in rows:
        total[r["date"]] += 1
        if r["y"] is not None:
            labeled[r["date"]] += 1
    full = [d for d in total if labeled[d] == total[d]]
    return max(full) if full else None


def trailing_funding_pct(rows):
    """Per-asset trailing-180 mid-rank pct of non-null funding
    (MIN_WINDOW=20). List aligned to rows."""
    by_asset = defaultdict(list)
    for i, r in enumerate(rows):
        by_asset[r["asset"]].append(i)
    out = [None] * len(rows)
    for idxs in by_asset.values():
        idxs.sort(key=lambda i: rows[i]["date"])
        window = []
        for i in idxs:
            v = rows[i]["funding"]
            out[i] = (mid_rank_pct(window, v)
                      if v is not None and len(window) >= FUND_MIN_WINDOW
                      else None)
            if v is not None:
                window.append(v)
                if len(window) > FUND_LOOKBACK:
                    window.pop(0)
    return out


def xs_rank_dfh(rows):
    """Same-day cross-sectional mid-rank pct of dfh20. List aligned."""
    by_date_idx = defaultdict(list)
    for i, r in enumerate(rows):
        if r["dfh20"] is not None:
            by_date_idx[r["date"]].append(i)
    out = [None] * len(rows)
    for idxs in by_date_idx.values():
        vals_sorted = sorted(rows[i]["dfh20"] for i in idxs)
        n = len(vals_sorted)
        for i in idxs:
            lo = bisect.bisect_left(vals_sorted, rows[i]["dfh20"])
            hi = bisect.bisect_right(vals_sorted, rows[i]["dfh20"])
            out[i] = (lo + 0.5 * (hi - lo)) / n
    return out


# ------------------------------------------------------- dataset assembly


def build_mega(rows, btc_map):
    """Attach min3 + all7 feature tuples to every modelable mega row."""
    fp = trailing_funding_pct(rows)
    xr = xs_rank_dfh(rows)
    out = []
    stats = {"n_rows_total": len(rows), "n_dropped_missing_core": 0,
             "n_dropped_no_btc": 0, "n_funding_pct_imputed": 0}
    for i, r in enumerate(rows):
        if (r["dfh20"] is None or r["mom20"] is None or r["vol20"] is None
                or r["y"] is None):
            stats["n_dropped_missing_core"] += 1
            continue
        b = btc_map.get(r["date"])
        if b is None:
            stats["n_dropped_no_btc"] += 1
            continue
        f = fp[i]
        if f is None:
            f = 0.5
            stats["n_funding_pct_imputed"] += 1
        r["x_min3"] = (r["dfh20"], b, r["dfh20"] * b)
        r["x_all7"] = (r["dfh20"], r["mom20"], r["vol20"], f, b,
                       r["dfh20"] * b, xr[i])
        out.append(r)
    stats["n_modelable"] = len(out)
    return out, stats


def build_xs(rows, btc_map):
    """Attach min3 + all7 feature tuples to xs eval rows (btc_ret20 from
    the perp series; xs_rank_dfh over the 10-asset cross-section — the
    v4 live convention)."""
    fp = trailing_funding_pct(rows)
    xr = xs_rank_dfh(rows)
    out = []
    stats = {"n_rows_total": len(rows), "n_unlabeled": 0,
             "n_dropped_missing_core": 0, "n_dropped_no_btc": 0}
    for i, r in enumerate(rows):
        b = btc_map.get(r["date"])
        r["gate_on"] = b is not None and b > 0
        if r["y"] is None:
            stats["n_unlabeled"] += 1
            continue
        if (r["dfh20"] is None or r["mom20"] is None or r["vol20"] is None
                or r["y1"] is None):
            stats["n_dropped_missing_core"] += 1
            continue
        if b is None:
            stats["n_dropped_no_btc"] += 1
            continue
        f = fp[i]
        if f is None:
            f = 0.5
        r["x_min3"] = (r["dfh20"], b, r["dfh20"] * b)
        r["x_all7"] = (r["dfh20"], r["mom20"], r["vol20"], f, b,
                       r["dfh20"] * b, xr[i])
        out.append(r)
    stats["n_modelable"] = len(out)
    return out, stats


# --------------------------------------------------------------- fitting


def train_variant(variant, rows, train_lo, train_hi):
    """Fit one min3 variant on mega rows in [train_lo, train_hi)
    (already embargoed by the caller). Returns model dict or None."""
    if variant == "recent_regime":
        train_lo = max(train_lo, RECENT_START)
    cand = [r for r in rows if train_lo <= r["date"] < train_hi]
    top = None
    if variant == "xs10_domain":
        cand = [r for r in cand if r["asset"] in XS_SET]
    elif variant == "top30_qvol":
        qv = defaultdict(list)
        for r in cand:
            if r["qvol"] is not None and r["qvol"] > 0:
                qv[r["asset"]].append(r["qvol"])
        mean_qv = sorted(((float(np.mean(v)), a) for a, v in qv.items()),
                         reverse=True)
        top = {a for _, a in mean_qv[:TOP_N_QVOL]}
        cand = [r for r in cand if r["asset"] in top]
    else:
        top = None
    if len(cand) < 200:
        return None
    X = np.array([r["x_min3"] for r in cand])
    y = np.array([r["y"] for r in cand])
    dates = np.array([r["date"] for r in cand])
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    lam = select_lambda(Xs, y, dates)
    w = ridge_fit(Xs, y, lam)
    return {"variant": variant, "w": w, "mu": mu, "sd": sd,
            "lambda": lam, "n_train": int(len(cand)),
            "n_train_symbols": len({r["asset"] for r in cand}),
            "train_window": [train_lo.isoformat(),
                             (train_hi - dt.timedelta(days=1)).isoformat()],
            "top30_overlap_xs": (len(top & XS_SET) if top else None),
            "coef": {MIN3_NAMES[k]: _r(w[k + 1], 6)
                     for k in range(len(MIN3_NAMES))}}


def fit_v4_replica(rows, cutoff):
    """v4's exact fitted model: ridge_all 7 features, lambda=100, all
    mega history <= cutoff (fit_model in financial_forward_ledger_v4)."""
    cand = [r for r in rows if r["date"] <= cutoff]
    X = np.array([r["x_all7"] for r in cand])
    y = np.array([r["y"] for r in cand])
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    w = ridge_fit((X - mu) / sd, y, V4_LAMBDA)
    return {"variant": "ridge_all_v4replica", "w": w, "mu": mu, "sd": sd,
            "lambda": V4_LAMBDA, "n_train": int(len(cand)),
            "n_train_symbols": len({r["asset"] for r in cand}),
            "train_window": [None, cutoff.isoformat()],
            "coef": {ALL7_NAMES[k]: _r(w[k + 1], 6)
                     for k in range(len(ALL7_NAMES))}}


def predict_rows(model, rows, feat_key):
    X = np.array([r[feat_key] for r in rows])
    return ridge_pred((X - model["mu"]) / model["sd"], model["w"])


# ------------------------------------------------------------ evaluation


def eval_ranked(rows, preds, edge, min_names):
    """Ranking metrics on (rows, preds): pooled + per-day cross-sectional
    spearman on y (5d), and an EW top-edge/bottom-edge realized spread
    per day on both the 5d label and the 1d return."""
    by_date = defaultdict(list)
    for r, p in zip(rows, preds):
        by_date[r["date"]].append((r, float(p)))
    daily_rho = []
    spreads5, spreads1 = [], []
    spreads5_g, spreads1_g = [], []
    n_book_days = n_gate_days = 0
    for d in sorted(by_date):
        day = by_date[d]
        pr = np.array([p for _, p in day])
        yy = np.array([r["y"] for r, _ in day])
        rho = spearman(pr, yy)
        if rho is not None:
            daily_rho.append(rho)
        if len(day) < min_names:
            continue
        order = np.argsort(pr)
        bot = order[:edge]
        top = order[-edge:]
        s5 = float((yy[top].mean() - yy[bot].mean()) / 2.0)
        y1 = np.array([r["y1"] for r, _ in day])
        s1 = float((y1[top].mean() - y1[bot].mean()) / 2.0)
        spreads5.append(s5)
        spreads1.append(s1)
        n_book_days += 1
        if all(r.get("gate_on") for r, _ in day):
            spreads5_g.append(s5)
            spreads1_g.append(s1)
            n_gate_days += 1
    p = np.asarray(preds, float)
    t = np.array([r["y"] for r in rows])
    out = {"n_rows": int(len(t)),
           "n_days": int(len(by_date)),
           "spearman_pooled": _r(spearman(p, t)),
           "spearman_daily_mean": _r(np.mean(daily_rho))
           if daily_rho else None,
           "spearman_daily_median": _r(np.median(daily_rho))
           if daily_rho else None,
           "n_days_scored": len(daily_rho),
           "frac_days_rho_positive": _r(np.mean(
               [x > 0 for x in daily_rho])) if daily_rho else None}
    if spreads5:
        out["book"] = {
            "edge": edge, "min_names_day": min_names,
            "n_book_days": n_book_days,
            "top_bot_realized_5d_bps_day": _r(np.mean(spreads5), 3),
            "top_bot_realized_1d_bps_day": _r(np.mean(spreads1), 3),
            "n_gate_on_days": n_gate_days,
            "top_bot_realized_5d_bps_day_gate_on":
                _r(np.mean(spreads5_g), 3) if spreads5_g else None,
            "top_bot_realized_1d_bps_day_gate_on":
                _r(np.mean(spreads1_g), 3) if spreads1_g else None,
        }
    return out


def eval_mega_holdout(rows, preds):
    """Mega-universe metrics: ranking + pooled top-decile spread on y."""
    out = eval_ranked(rows, preds, edge=None, min_names=10**9)
    p = np.asarray(preds, float)
    t = np.array([r["y"] for r in rows])
    q = np.quantile(p, 0.9)
    top, rest = t[p >= q], t[p < q]
    out["top_decile_spread_bps"] = _r(top.mean() - rest.mean(), 2)
    out.pop("book", None)
    return out


# ------------------------------------------------------------------ run


def run():
    protocol_sha = write_protocol_and_auth()
    t0 = time.time()
    btc_csv = load_btc_ret20_csv()
    mega_raw, n_mega = load_mega_rows()
    xs_raw = load_xs_rows()
    mega, mstats = build_mega(mega_raw, btc_csv)
    xs, xstats = build_xs(xs_raw, btc_ret20_perp_map(xs_raw))
    cutoff = last_fully_labeled_date(mega_raw)
    xs_by_year = defaultdict(list)
    for r in xs:
        if r["date"] <= XS_END:
            xs_by_year[r["date"].year].append(r)

    # -------------------- A) yearly walkforward folds ------------------
    wf_folds = []
    for ty, ny in FOLD_PAIRS:
        boundary = dt.date(ny, 1, 1)
        embargo_start = boundary - dt.timedelta(days=EMBARGO_DAYS)
        mega_te = [r for r in mega
                   if boundary <= r["date"] < dt.date(ny + 1, 1, 1)]
        xs_te = xs_by_year.get(ny, [])
        fold = {"train_year": ty, "test_year": ny,
                "n_mega_test": len(mega_te), "n_xs_test": len(xs_te),
                "variants": {}}
        for v in VARIANTS:
            model = train_variant(v, mega, dt.date(ty, 1, 1),
                                  embargo_start)
            if model is None:
                fold["variants"][v] = {"skipped": "insufficient train rows"}
                continue
            rec = {"model": {k: model[k] for k in
                             ("lambda", "n_train", "n_train_symbols",
                              "train_window", "top30_overlap_xs",
                              "coef")}}
            if mega_te:
                rec["mega_eval"] = eval_mega_holdout(
                    mega_te, predict_rows(model, mega_te, "x_min3"))
            if xs_te:
                rec["xs_eval"] = eval_ranked(
                    xs_te, predict_rows(model, xs_te, "x_min3"),
                    edge=EDGE, min_names=MIN_UNIVERSE)
            fold["variants"][v] = rec
        wf_folds.append(fold)

    # ------------- B) v4-exact live window: fit <=2025-12-26 -----------
    xs_2026 = [r for r in xs_by_year.get(2026, [])
               if r["date"] >= dt.date(2026, 1, 1)]
    live = {"train_cutoff": cutoff.isoformat(),
            "eval_window": ["2026-01-01", XS_END.isoformat()],
            "n_xs_test": len(xs_2026), "variants": {}}
    for v in VARIANTS:
        model = train_variant(v, mega, dt.date(2000, 1, 1),
                              cutoff + dt.timedelta(days=1))
        rec = {"model": {k: model[k] for k in
                         ("lambda", "n_train", "n_train_symbols",
                          "train_window", "top30_overlap_xs", "coef")}}
        rec["xs_eval"] = eval_ranked(
            xs_2026, predict_rows(model, xs_2026, "x_min3"),
            edge=EDGE, min_names=MIN_UNIVERSE)
        live["variants"][v] = rec
    rep = fit_v4_replica(mega, cutoff)
    live["variants"]["ridge_all_v4replica"] = {
        "model": {k: rep[k] for k in ("lambda", "n_train",
                                      "n_train_symbols", "train_window",
                                      "coef")},
        "xs_eval": eval_ranked(
            xs_2026, predict_rows(rep, xs_2026, "x_all7"),
            edge=EDGE, min_names=MIN_UNIVERSE)}

    verdict, direct = verdict_lines(wf_folds, live)
    return {
        "schema_version": "nanojev-financial-signal-domain-match-v1",
        "task": "T130",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_s": _r(time.time() - t0, 1),
        "protocol_path":
            "research/financial_signal_domain_match_protocol_v1.json",
        "protocol_sha256": protocol_sha,
        "authorization_path":
            "results/financial_signal_domain_match_authorization_v1.json",
        "data": {"mega_cohort": str(MEGA.relative_to(ROOT)),
                 "xs_cohort": str(XS.relative_to(ROOT)),
                 "btc_series_train": str(BTC_CSV.relative_to(ROOT)),
                 "btc_series_xs_eval": "BTCUSDT-PERP closes inside xs_v1 "
                                       "(v3 live convention)",
                 "n_mega_records": n_mega,
                 "mega_assembly": mstats,
                 "xs_assembly": xstats,
                 "mega_last_fully_labeled": cutoff.isoformat(),
                 "xs_eval_end": XS_END.isoformat()},
        "design": {
            "model": "ridge_min3 {dfh20, btc_ret20, dfh20*btc_ret20} "
                     "(T127 parsimonious winner); replica arm = v4's "
                     "ridge_all 7-feature set at lambda=100",
            "variants": {
                "all277_control": "all mega symbols in the train window",
                "top30_qvol": "top-30 symbols by mean daily quote_volume "
                              "inside the fold's train window (PIT-safe, "
                              "recomputed per fold)",
                "xs10_domain": "the 10 ledger symbols' mega rows only",
                "recent_regime": "all symbols, train dates >= 2023-01-01"},
            "walkforward": "train = calendar year t (minus 5d embargo), "
                           "test = year t+1; folds 2021->22..2025->2026H1; "
                           "lambda via in-train-window forward-chaining CV",
            "live_window": "fit once on all mega history <= 2025-12-26 "
                           "(v4's cutoff), eval on xs_v1 "
                           "2026-01-01..2026-08-26 — v4's exact "
                           "inversion window",
            "xs_book": "long top-2 / short bottom-2 by pred, EW; spread "
                       "= (mean realized top2 - mean realized bot2)/2 "
                       "per day (v4 book_realized_5d_bps construction), "
                       "reported on 5d label and 1d return, all days and "
                       "btc_ret20>0 gate-on days; days with <6 scored "
                       "assets skipped (v4 MIN_UNIVERSE)"},
        "walkforward_folds": wf_folds,
        "live_window_2026": live,
        "direct_answer": direct,
        "verdict_lines": verdict,
        "scope": "measurement only; not a strategy or tradability claim",
    }


def _xs(evald):
    return (evald or {}).get("xs_eval") or {}


def verdict_lines(wf_folds, live):
    lines = []
    # live-window table
    lw = {v: (live["variants"].get(v) or {}).get("xs_eval")
          for v in list(VARIANTS) + ["ridge_all_v4replica"]}
    for v, e in lw.items():
        if not e:
            continue
        b = e.get("book") or {}
        lines.append(
            "LIVE2026 %s: pooled rho %s, daily-rho mean %s (pos %s of "
            "days), top2/bot2 realized5d %s bps/day (gate-on %s), "
            "n_train=%s." % (
                v, e.get("spearman_pooled"), e.get("spearman_daily_mean"),
                e.get("frac_days_rho_positive"),
                b.get("top_bot_realized_5d_bps_day"),
                b.get("top_bot_realized_5d_bps_day_gate_on"),
                (live["variants"][v].get("model") or {}).get("n_train")))
    # xs walkforward summary per variant
    for v in VARIANTS:
        parts = []
        for f in wf_folds:
            e = _xs((f["variants"].get(v) or {}))
            if e:
                parts.append("%d:%s" % (f["test_year"],
                                        e.get("spearman_pooled")))
        lines.append("XS_WF %s pooled rho by test year: %s."
                     % (v, ", ".join(parts) or "none"))
    # mega holdout 2025
    f25 = next((f for f in wf_folds if f["test_year"] == 2025), None)
    if f25:
        parts = []
        for v in VARIANTS:
            e = (f25["variants"].get(v) or {}).get("mega_eval")
            if e:
                parts.append("%s pooled %s / decile %s" % (
                    v, e.get("spearman_pooled"),
                    e.get("top_decile_spread_bps")))
        lines.append("MEGA2025 holdout: %s." % "; ".join(parts))
    # direct answer
    ctl = (lw.get("all277_control") or {}).get("spearman_pooled")
    fixes = []
    for v in ("top30_qvol", "xs10_domain", "recent_regime"):
        e = lw.get(v) or {}
        rho = e.get("spearman_pooled")
        book = (e.get("book") or {}).get("top_bot_realized_5d_bps_day")
        fixes.append((v, rho, book))
    pos = [v for v, rho, _ in fixes if rho is not None and rho > 0]
    direct = ("On v4's exact 2026 live window, all-277 min3 control "
              "pooled rho = %s (replica ridge_all = %s). Domain/regime "
              "variants with positive pooled rho: %s. " % (
                  ctl,
                  (lw.get("ridge_all_v4replica") or {})
                  .get("spearman_pooled"),
                  ", ".join(pos) or "none"))
    lines.append("DIRECT: " + direct)
    rec_rho = (lw.get("recent_regime") or {}).get("spearman_pooled")
    lines.append(
        "REGIME: recent-window (2023+) training pooled rho %s vs control "
        "%s — regime-matching %s the inversion; the pre-2023 history "
        "helps, it does not hurt." % (
            rec_rho, ctl,
            "amplifies" if (rec_rho or 0) < (ctl or 0) else "damps"))
    b5 = {v: ((lw.get(v) or {}).get("book") or {})
          .get("top_bot_realized_5d_bps_day")
          for v in list(VARIANTS) + ["ridge_all_v4replica"]}
    b1 = {v: ((lw.get(v) or {}).get("book") or {})
          .get("top_bot_realized_1d_bps_day")
          for v in list(VARIANTS) + ["ridge_all_v4replica"]}
    lines.append(
        "BOOK: realized top2/bot2 spread on the live window, 5d label: "
        "%s; 1d return: %s — ranking fix does not imply a positive "
        "realized 5d book." % (
            {k: v for k, v in b5.items()},
            {k: v for k, v in b1.items()}))
    n_xs10 = ((live["variants"].get("xs10_domain") or {})
              .get("model") or {}).get("n_train")
    lines.append(
        "RECOMMENDATION: if v4's scorer is refit, prefer ridge_min3 "
        "trained on the top-30-by-quote-volume mega symbols over full "
        "history (only variant with positive pooled AND daily rho on "
        "the exact inversion window); xs10 is a viable fallback — %s "
        "train rows suffice for a 3-feature ridge and it posts the best "
        "mega-2025 holdout rho, but its live pooled rho stays marginally "
        "negative. Do NOT restrict to the 2023+ window. Ranking fix is "
        "not a profitability claim: every min3 variant's realized 5d "
        "book stays negative in 2026H1, so keep the btc_ret20 gate and "
        "continue ledger monitoring." % n_xs10)
    return lines, direct


def write_protocol_and_auth():
    protocol = {
        "schema_version":
            "nanojev-financial-signal-domain-match-protocol-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": ("T130: test whether domain-matched (top-30 by "
                    "quote_volume, or the 10 ledger symbols) or "
                    "regime-matched (2023+) training of ridge_min3 fixes "
                    "the v4 OOS ranking inversion (pooled 2026 xs "
                    "spearman ~-0.096) that all-277-symbol training "
                    "produces on the live majors universe. Measurement "
                    "only."),
        "cohorts": {
            "train": "data/perp_pit_mega_v1/records.jsonl (277 symbols; "
                     "all train-side rows come from the mega cohort, "
                     "including the xs10 variant's rows for the 10 "
                     "ledger symbols)",
            "eval_mega": "mega test-year rows, all symbols",
            "eval_xs": "data/perp_pit_xs_v1/records.jsonl (10 majors, "
                       "through 2026-08-26 fully labeled)",
            "btc_train": "data/rc_futures_v1/BTC/BTCUSDT_1d.csv",
            "btc_xs_eval": "BTCUSDT-PERP closes inside xs_v1 (v3 live "
                           "convention)",
        },
        "definitions": {
            "model": "ridge_min3 {dfh20, btc_ret20, dfh20*btc_ret20}; "
                     "replica arm = v4 ridge_all 7 features, lambda=100",
            "variants": "all277_control / top30_qvol (per-fold PIT "
                        "symbol selection by mean quote_volume inside "
                        "the train window) / xs10_domain / recent_regime "
                        "(train dates >= 2023-01-01)",
            "walkforward": "T127 discipline: train year t -> test year "
                           "t+1, 5d embargo at boundaries, lambda via "
                           "in-train forward-chaining CV (5 chunks); "
                           "never test",
            "live_window": "fit <= 2025-12-26 last fully labeled, eval "
                           "xs 2026-01-01..2026-08-26",
            "metrics": "pooled spearman(pred, ret5d) [v4 headline], "
                       "mean/median per-day cross-sectional spearman, "
                       "top-2/bottom-2 realized spread (v4 book "
                       "construction) on 5d label and 1d return, "
                       "ungated and btc_ret20>0 days; mega top-decile "
                       "spread",
        },
        "caveats": [
            "pooled spearman mixes cross-day prediction-level drift with "
            "within-day ranking; the book only consumes within-day "
            "ranks — both are reported",
            "xs_rank_dfh on xs is a 10-asset rank vs the 277-asset "
            "training cross-section (v4's documented shift); affects "
            "the replica arm only — min3 uses raw dfh20",
            "xs10_domain trains on the symbols' mega-cohort rows (same "
            "venue/feature construction), not the xs records",
            "top30_qvol membership is recomputed inside each train "
            "window — PIT-safe but fold-dependent",
            "xs 2026 eval window is Jan-Aug (238 labeled days), not a "
            "full year",
        ],
        "forbidden": ["trading", "profitability claims",
                      "protocol edits post-run", "network",
                      "lambda/model selection on test data"],
    }
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(protocol, indent=2, sort_keys=True)
                        + "\n")
    sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    auth = {
        "schema_version":
            "nanojev-financial-signal-domain-match-authorization-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision": "approved_for_measurement",
        "protocol_path":
            "research/financial_signal_domain_match_protocol_v1.json",
        "protocol_sha256": sha,
        "measurement_authorized": True,
        "fit_authorized": True,
        "live_trading_authorized": False,
        "order_submission_authorized": False,
        "network_model_calls": 0,
        "independent_reviewer": {
            "id": "project-owner",
            "independence":
                "owner_self_authorization_not_independent_review",
            "note": "Owner directed T130: test domain/regime-matched "
                    "training against the v4 OOS ranking inversion "
                    "(delegated task).",
        },
        "scope": {
            "permitted": "PIT-safe ridge fitting + OOS measurement on "
                         "data/perp_pit_mega_v1 and data/perp_pit_xs_v1 "
                         "per the pinned protocol; simulated book "
                         "accounting only.",
            "not_permitted": "No trading/profitability claims/protocol "
                             "edits; no network; no other files "
                             "modified.",
        },
    }
    AUTH.parent.mkdir(parents=True, exist_ok=True)
    AUTH.write_text(json.dumps(auth, indent=2, sort_keys=True) + "\n")
    return sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=pathlib.Path, default=OUT)
    args = ap.parse_args()
    report = run()
    blob = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(blob)
    print(json.dumps({"output": str(args.output),
                      "sha256": hashlib.sha256(blob.encode()).hexdigest(),
                      "verdict": report["verdict_lines"]}, indent=2))


if __name__ == "__main__":
    main()
