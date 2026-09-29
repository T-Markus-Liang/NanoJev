#!/usr/bin/env python3
"""T128: forward paper-trading ledger for the ridge_all sleeve (fourth
sleeve) — the T125 ML-viable ridge model scoring the live 10-asset XS
universe daily.

MODEL (fit once per run; deterministic — the mega cohort is static):
  train rows: data/perp_pit_mega_v1 records with date <= train cutoff,
    where train cutoff = the last date whose records ALL carry a
    complete 5d label (label.forward_return_5d_bps non-null for every
    record that date; currently 2025-12-26 — the archive tail
    2025-12-27..30 cannot carry a contiguous +5d label).
  target:     label.forward_return_5d_bps (gross close-to-close, bps)
  features:   the T125 ridge_all set, identical construction
    (scripts/financial_signal_ml_probe_v1.py):
      dfh20         close / max(strictly-prior 20 contiguous closes) - 1
      mom20         close / close[i-20] - 1 (same contiguous window)
      vol20         stdev of the 20 trailing daily log-returns
      funding_pct   per-asset mid-rank pct of last_funding_rate vs its
                    trailing-180 NON-NULL values (min window 20; neutral
                    0.5 imputed when unusable)
      btc_ret20     BTCUSDT 20d close/close-1 broadcast to every asset
                    that day. TRAIN: data/rc_futures_v1/BTC/BTCUSDT_1d.csv
                    (T125 convention). LIVE: the merged BTCUSDT-PERP
                    series (v3 convention) — the csv ends 2025-12-31 and
                    cannot gate refreshed bars.
      dfh20_x_btc   dfh20 * btc_ret20 — the known gated interaction
      xs_rank_dfh   same-day cross-sectional mid-rank pct of dfh20
  fit:        standardize on train rows (mean/std), ridge lambda=100 —
    T125's inner-CV pick in all 4 walkforward folds — intercept
    unpenalized. lambda is frozen, not re-selected.

Sleeve spec:
  universe:   the 10 USDT-M perps in data/perp_pit_xs_v1/records.jsonl,
              merged with both refresh supplements (identical
              dedup-by-id merge as financial_forward_ledger_v2.py;
              refreshed legacy bars carry mark_price as the close proxy,
              refreshed XS bars carry close). dfh20/mom20/vol20 and
              funding_pct are recomputed from the merged series so
              refreshed bars score identically to cohort bars;
              recomputation is cross-checked against cohort features.
  scoring:    each decision date, every asset with a complete feature
              set (dfh20, mom20, vol20, xs_rank_dfh, btc_ret20 all
              evaluable; funding_pct imputed 0.5) gets pred_bps =
              predicted 5d forward return. Predictions rank descending.
  book:       long top-2 / short bottom-2 by pred_bps, equal 1/4 leg
              weight, daily rebalance — but ONLY when the BTC regime
              gate is on (btc_ret20 > 0, the T119/v3 gate variable: the
              signal is regime-conditional) AND >=6 assets scored (the
              graceful-degradation floor shared with v2/v3). Gate off or
              thin universe -> flat book; predictions are still recorded.
  costs/pnl:  identical to the v2 sleeve: 5bps per one-sided leg
              notional traded, cost_day_bps = 5 * max(added, removed)/4;
              the book decided at t-1 is held t-1 -> t close-to-close,
              stale carry-forward fills flagged, incomplete books are
              not booked. cumulative.net_equity compounds daily net
              ((gross or 0) - cost)/1e4.

  resolution: the ledger is append-only and never rewritten. Each line
              carries BOTH the day's predictions AND the matured outcome
              of the predictions made exactly 5 calendar days earlier:
              per-asset realized 5d close-to-close bps (stale-flagged
              carry-forward fill when the asset has no bar that day), a
              spearman(pred, realized) over the resolved cross-section,
              and the book-level predicted-vs-realized 5d return.

Ledger contract (results/forward_ledger_v4.jsonl): same shape as v2 —
  append-only, one line per merged-input decision date, idempotent by
  decision_date, content-light (ranks/preds/book/costs — never raw
  feature payloads), fully determined at write time.

Documented deviations / caveats:
  * xs_rank_dfh is computed over the 10-asset XS cross-section while the
    model was trained on the ~277-asset mega cross-section — a rank
    definition shift inherent to the live sleeve.
  * predictions on decision dates <= train_cutoff are IN-SAMPLE
    (model.in_sample flag); out-of-sample scoring starts the day after
    the cutoff. Backfill and resolution stats are split accordingly.
  * refresh bars carry no labels; realized outcomes are computed from
    merged closes (the same contiguous close-to-close +5d construction
    as the cohort label).
  * mark_price vs close basis on refreshed legacy bars (~1bp median
    deviation, data/perp_pit_xs_v1/build_summary.json).

--snapshot prints the sleeve state as of the latest decision date:
ranked predictions, book, gate, cumulative net equity, and pooled
predictions-vs-realized stats (all / in-sample / out-of-sample).

Simulated paper accounting only — no orders, no broker, no account
access, not a tradability or profitability claim.
"""

import argparse
import bisect
import datetime as dt
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_xs_v1/records.jsonl"
SUPPLEMENTS = [ROOT / "data/binance_refresh_v1/records.jsonl",
               ROOT / "data/binance_xs_refresh_v1/records.jsonl"]
MEGA = ROOT / "data/perp_pit_mega_v1/records.jsonl"
BTC_CSV = ROOT / "data/rc_futures_v1/BTC/BTCUSDT_1d.csv"
LEDGER = ROOT / "results/forward_ledger_v4.jsonl"
# T129 xs_v2 universe (30 assets, research/live_universe_v2.json): opt-in via
# --universe xs_v2; writes to a SEPARATE ledger so the v1 chain is untouched.
COHORT_XS2 = ROOT / "data/perp_pit_xs_v2/records.jsonl"
SUPPLEMENTS_XS2 = SUPPLEMENTS + [ROOT / "data/binance_xs2_refresh_v1/records.jsonl"]
LEDGER_XS2 = ROOT / "results/forward_ledger_v4_xs2.jsonl"

SCHEMA = "nanojev-forward-ledger-v4"
SLEEVE = "ridge_all_xs_top2_bot2_btc_ret20_gate_v1"
BTC_ASSET = "BTCUSDT-PERP"
WINDOW = 20                # dfh20/mom20/vol20 trailing contiguous bars
BTC_LOOKBACK = 20          # btc_ret20: close / close[t-20] - 1
FUND_LOOKBACK = 180        # trailing non-null funding values for the pct
FUND_MIN_WINDOW = 20       # repo convention floor for a usable pct
LABEL_HORIZON_DAYS = 5     # the 5d forward label being predicted/resolved
EDGE = 2                   # long top-2 / short bottom-2
N_LEG_SLOTS = 4            # 2 long + 2 short, each 1/4 of book
MIN_UNIVERSE = 6           # graceful-degradation floor for a valid book
COST_BPS_PER_LEG = 5.0     # per unit of one-sided leg notional traded
LAMBDA = 100.0             # T125 inner-CV pick in all 4 folds (frozen)
FEATURE_NAMES = ("dfh20", "mom20", "vol20", "funding_pct", "btc_ret20",
                 "dfh20_x_btc", "xs_rank_dfh")
# T127/T130: parsimonious domain-matched variant — 3 features, trained on the
# top-30-by-quote-volume mega symbols only (fixes the thin-universe OOS
# inversion ridge_all shows on majors).
MODEL_VARIANTS = ("ridge_all", "ridge_min3_top30")
MIN3_IDX = (0, 4, 5)       # dfh20, btc_ret20, dfh20_x_btc
UNIVERSE_JSON = ROOT / "research/live_universe_v2.json"
EVENTS = ("score", "score_flat_gate_off", "skipped_insufficient_universe",
          "skipped_no_btc_ret20")


def _r(x, nd=4):
    return round(x, nd) if isinstance(x, (int, float)) else x


def feat(record, name):
    f = record.get("features", {}).get(name)
    return f.get("value") if isinstance(f, dict) else None


def dt_date(text):
    return dt.date.fromisoformat(text)


def mid_rank_pct(window, x):
    """Mid-rank percentile of x vs a trailing window (ties count half)."""
    if x is None or not window:
        return None
    return (sum(1 for w in window if w < x)
            + 0.5 * sum(1 for w in window if w == x)) / len(window)


# ---------------------------------------------------------------- model


def load_btc_ret20_csv(path=BTC_CSV):
    """date -> BTC close/close[t-20]-1 on the contiguous csv series
    (T125 construction; used for TRAIN rows only)."""
    dates, closes = [], []
    with path.open() as f:
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


def load_mega_rows(path=MEGA):
    """T125 load_rows: per record the fields the model needs."""
    rows = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
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
    return rows


def last_fully_labeled_date(rows):
    """Max date on which EVERY record carries a non-null 5d label."""
    total = defaultdict(int)
    labeled = defaultdict(int)
    for r in rows:
        total[r["date"]] += 1
        if r["y"] is not None:
            labeled[r["date"]] += 1
    full = [d for d in total if labeled[d] == total[d]]
    return max(full) if full else None


def build_dataset(rows, btc_ret20, cutoff):
    """T125 build_dataset restricted to date <= cutoff. Returns modelable
    rows ({date, asset, x, y}) + assembly stats."""
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
                              if v is not None and len(window)
                              >= FUND_MIN_WINDOW else None)
            if v is not None:
                window.append(v)
                if len(window) > FUND_LOOKBACK:
                    window.pop(0)
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
    out = []
    stats = {"n_rows_total": len(rows), "n_dropped_past_cutoff": 0,
             "n_funding_pct_imputed": 0, "n_dropped_missing_core": 0,
             "n_dropped_no_btc": 0}
    for i, r in enumerate(rows):
        if cutoff is not None and r["date"] > cutoff:
            stats["n_dropped_past_cutoff"] += 1
            continue
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
        out.append({"date": r["date"], "asset": r["asset"],
                    "x": (r["dfh20"], r["mom20"], r["vol20"], fp, b,
                          r["dfh20"] * b, xs_rank[i]),
                    "y": r["y"]})
    stats["n_modelable"] = len(out)
    return out, stats


def ridge_fit(X, y, lam):
    n, d = X.shape
    Xm = np.column_stack([np.ones(n), X])
    A = Xm.T @ Xm + lam * np.eye(d + 1)
    A[0, 0] -= lam  # intercept unpenalized
    return np.linalg.solve(A, Xm.T @ y)


def _top30_assets():
    u = json.loads(UNIVERSE_JSON.read_text())
    syms = u.get("universe") or u.get("symbols") or []
    return {s if s.endswith("-PERP") else s + "-PERP" for s in syms}


def fit_model(mega_path=MEGA, btc_csv=BTC_CSV, variant="ridge_all"):
    """Fit ridge once on the mega cohort through the last fully
    labeled date. Deterministic: the mega cohort is static."""
    rows = load_mega_rows(mega_path)
    cutoff = last_fully_labeled_date(rows)
    btc_ret20 = load_btc_ret20_csv(btc_csv)
    data, stats = build_dataset(rows, btc_ret20, cutoff)
    idx = list(range(len(FEATURE_NAMES)))
    if variant == "ridge_min3_top30":
        keep = _top30_assets()
        data = [r for r in data if r["asset"] in keep]
        idx = list(MIN3_IDX)
    X = np.array([[r["x"][i] for i in idx] for r in data])
    y = np.array([r["y"] for r in data])
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    w = ridge_fit((X - mu) / sd, y, LAMBDA)
    return {"w": w, "mu": mu, "sd": sd, "idx": idx, "variant": variant,
            "train_cutoff": cutoff.isoformat() if cutoff else None,
            "n_train": int(len(data)), "lambda": LAMBDA,
            "train_mean_bps": float(y.mean()) if len(y) else None,
            "intercept_bps": float(w[0]),
            "coef": {FEATURE_NAMES[idx[i]]: float(w[i + 1])
                     for i in range(len(idx))},
            "assembly": stats}


def predict(model, x):
    """Feature tuple (full FEATURE_NAMES order) -> predicted 5d bps."""
    xi = np.asarray([x[i] for i in model.get("idx",
                                           range(len(FEATURE_NAMES)))],
                    dtype=float)
    z = (xi - model["mu"]) / model["sd"]
    return float(model["w"][0] + z @ model["w"][1:])


def rankdata(v):
    """Average-tie ranks (T125 construction)."""
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


# ------------------------------------------------------- live universe


def record_close(record):
    """XS price basis: klines close as mark proxy; refreshed legacy bars
    carry mark_price instead (refresh_v1 shape). Same as v2/v3."""
    close = feat(record, "close")
    if close is not None:
        return close, "close"
    mark = feat(record, "mark_price")
    if mark is not None:
        return mark, "mark_price"
    return None, None


def load_inputs(cohort_path, supplement_paths):
    """asset -> sorted [{date, close, basis, funding, decision_ns,
    record_id}]. Same merge as v2/v3: cohort + supplements deduped by
    record id (cohort wins); per asset, a supplement date already present
    in the cohort is skipped."""
    records = [json.loads(l) for l in cohort_path.read_text().splitlines()
               if l.strip()]
    seen = {r["id"] for r in records}
    extra = 0
    for path in supplement_paths:
        if not path.exists():
            continue
        for l in path.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("id") in seen:
                continue
            records.append(r)
            seen.add(r["id"])
            extra += 1
    by_asset = defaultdict(dict)  # asset -> date -> bar
    for r in records:
        date = r["id"].rsplit(":", 1)[-1]
        close, basis = record_close(r)
        asset = r["asset_id"]
        if date in by_asset[asset]:  # cohort already holds this date
            continue
        by_asset[asset][date] = {"date": date, "close": close,
                                 "basis": basis,
                                 "funding": feat(r, "last_funding_rate"),
                                 "decision_ns": r["decision_ns"],
                                 "record_id": r["id"]}
    return {a: [days[d] for d in sorted(days)]
            for a, days in by_asset.items()}, extra


def add_bar_features(series):
    """Recompute dfh20/mom20/vol20 (mega+XS construction: strictly-prior
    20 contiguous bars) and funding_pct (T125: trailing-180 NON-NULL
    mid-rank, min window 20) in place on each asset's merged rows."""
    for rows in series.values():
        fwin = []
        for i, r in enumerate(rows):
            contiguous = (i >= WINDOW
                          and (dt_date(r["date"]) - dt_date(
                               rows[i - WINDOW]["date"])).days == WINDOW)
            dfh = mom = vol = None
            if contiguous:
                window = [rows[j]["close"] for j in
                          range(i - WINDOW, i + 1)]
                if (r["close"] is not None and r["close"] > 0
                        and all(c is not None and c > 0 for c in window)):
                    prior = window[:-1]
                    dfh = r["close"] / max(prior) - 1.0
                    mom = r["close"] / window[0] - 1.0
                    rets = [math.log(window[k + 1] / window[k])
                            for k in range(len(window) - 1)]
                    vol = (statistics.stdev(rets)
                           if len(rets) >= WINDOW else None)
            r["dfh20"], r["mom20"], r["vol20"] = dfh, mom, vol
            f = r["funding"]
            r["funding_pct"] = (mid_rank_pct(fwin, f)
                                if f is not None
                                and len(fwin) >= FUND_MIN_WINDOW
                                else None)
            if f is not None:
                fwin.append(f)
                if len(fwin) > FUND_LOOKBACK:
                    fwin.pop(0)


def crosscheck_features(series, cohort_path):
    """max abs diff between recomputed dfh20/mom20/vol20 and the cohort
    features (cohort bars only; refreshed bars have no such features)."""
    cohort_feature = {}
    for l in cohort_path.read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        cohort_feature[r["id"]] = {n: feat(r, n)
                                   for n in ("dfh20", "mom20", "vol20")}
    diff = {n: 0.0 for n in ("dfh20", "mom20", "vol20")}
    n_checked = 0
    for rows in series.values():
        for r in rows:
            f = cohort_feature.get(r["record_id"])
            if f is None:
                continue
            for n in diff:
                if f[n] is not None and r[n] is not None:
                    diff[n] = max(diff[n], abs(f[n] - r[n]))
                    if n == "dfh20":
                        n_checked += 1
    return diff, n_checked


def btc_ret20_map(btc_rows):
    """date -> (btc_close, ret20 or None) on the merged BTCUSDT-PERP
    series. ret20 = close / close of the bar exactly 20 calendar days
    prior - 1, both closes positive (v3 construction)."""
    out = {}
    for i, r in enumerate(btc_rows):
        ret = None
        if (i >= BTC_LOOKBACK and r["close"] is not None
                and r["close"] > 0):
            back = btc_rows[i - BTC_LOOKBACK]
            span = (dt_date(r["date"]) - dt_date(back["date"])).days
            if (span == BTC_LOOKBACK and back["close"] is not None
                    and back["close"] > 0):
                ret = r["close"] / back["close"] - 1.0
        out[r["date"]] = (r["close"], ret)
    return out


def xs_rank_maps(dates, by_date):
    """date -> {asset: dfh20 same-day cross-sectional mid-rank pct} over
    the assets with a usable dfh20 (v3 construction)."""
    dfh_pct = {}
    for date in dates:
        scored = [(a, r["dfh20"]) for a, m in by_date.items()
                  if (r := m.get(date)) is not None
                  and r["dfh20"] is not None]
        vals = sorted(v for _, v in scored)
        n = len(vals)
        pct = {}
        for a, v in scored:
            lo = bisect.bisect_left(vals, v)
            eq = bisect.bisect_right(vals, v) - lo
            pct[a] = (lo + 0.5 * eq) / n
        dfh_pct[date] = pct
    return dfh_pct


def last_close_at_or_before(rows, date):
    prior = [r for r in rows if r["date"] <= date]
    return prior[-1] if prior else None


def replay(series, model):
    """Deterministic full replay -> ordered ledger line dicts. Each line
    carries the day's ranked predictions + book + gate AND the matured
    resolution of the predictions made exactly 5 days earlier."""
    dates = sorted({r["date"] for rows in series.values() for r in rows})
    by_date = {asset: {r["date"]: r for r in rows}
               for asset, rows in series.items()}
    ns_by_date = defaultdict(int)
    for rows in series.values():
        for r in rows:
            ns_by_date[r["date"]] = max(ns_by_date[r["date"]],
                                        r["decision_ns"])
    btc_gate = btc_ret20_map(series.get(BTC_ASSET, []))
    dfh_pct = xs_rank_maps(dates, by_date)
    cutoff = model["train_cutoff"]

    pending = {}          # pred_date -> {preds: {asset:bps}, book: {...}}
    prev_book = []        # (side, asset) legs decided on the prior date
    prev_date = None
    cum = {"gross": 0.0, "cost": 0.0, "net": 0.0}
    equity = 1.0
    lines = []

    for date in dates:
        day_rows = {a: m[date] for a, m in by_date.items() if date in m}
        btc_close, ret20 = btc_gate.get(date, (None, None))
        gate_on = ret20 is not None and ret20 > 0
        pcts = dfh_pct.get(date, {})

        # ---------------- score the cross-section -------------------
        scored = []       # (asset, pred_bps), sorted pred desc
        if ret20 is not None:
            for a, r in day_rows.items():
                if (r["dfh20"] is None or r["mom20"] is None
                        or r["vol20"] is None or r["close"] is None
                        or r["close"] <= 0):
                    continue
                xs = pcts.get(a)
                if xs is None:
                    continue
                fp = (r["funding_pct"]
                      if r["funding_pct"] is not None else 0.5)
                x = (r["dfh20"], r["mom20"], r["vol20"], fp, ret20,
                     r["dfh20"] * ret20, xs)
                scored.append((a, predict(model, x)))
        scored.sort(key=lambda t: (-t[1], t[0]))
        n_scored = len(scored)

        if ret20 is None:
            event = "skipped_no_btc_ret20"
            book = []
        elif n_scored < MIN_UNIVERSE:
            event = "skipped_insufficient_universe"
            book = []
        elif not gate_on:
            event = "score_flat_gate_off"
            book = []
        else:
            event = "score"
            book = ([("long", a) for a, _ in scored[:EDGE]]
                    + [("short", a) for a, _ in scored[-EDGE:]])

        predictions = [{"asset": a, "pred_bps": _r(p, 2)}
                       for a, p in scored]
        if scored:
            pending[date] = {"preds": dict(scored),
                             "book": {"long": [a for s, a in book
                                               if s == "long"],
                                      "short": [a for s, a in book
                                                if s == "short"]}}

        # ------------- resolve the predictions from 5d ago ----------
        res_date = (dt_date(date)
                    - dt.timedelta(days=LABEL_HORIZON_DAYS)).isoformat()
        pend = pending.pop(res_date, None)
        resolution = None
        if pend is not None:
            legs_res = []
            preds_res, reals_res = [], []
            side_of = {a: s for s in ("long", "short")
                       for a in pend["book"][s]}
            for a, p in sorted(pend["preds"].items()):
                ebar = by_date.get(a, {}).get(res_date)
                xbar = by_date.get(a, {}).get(date)
                stale = False
                if xbar is None:
                    xbar = last_close_at_or_before(series[a], date)
                    stale = True
                real = None
                if (ebar and xbar and ebar["close"] and xbar["close"]
                        and ebar["close"] > 0 and xbar["close"] > 0):
                    real = (xbar["close"] / ebar["close"] - 1.0) * 1e4
                    preds_res.append(p)
                    reals_res.append(real)
                legs_res.append({
                    "asset": a, "side": side_of.get(a),
                    "pred_bps": _r(p, 2),
                    "realized_5d_bps": (_r(real, 2)
                                        if real is not None else None),
                    "stale_fill": stale})
            book_pred = book_real = None
            if pend["book"]["long"] and pend["book"]["short"]:
                book_pred = (sum(pend["preds"][a]
                                 for a in pend["book"]["long"])
                             - sum(pend["preds"][a]
                                   for a in pend["book"]["short"])) / 4.0
                leg_reals = {l["asset"]: l["realized_5d_bps"]
                             for l in legs_res}
                signed = []
                for s in ("long", "short"):
                    for a in pend["book"][s]:
                        rv = leg_reals.get(a)
                        if rv is None:
                            signed = None
                            break
                        signed.append(rv if s == "long" else -rv)
                    if signed is None:
                        break
                if signed is not None:
                    book_real = sum(signed) / 4.0
            resolution = {
                "predictions_date": res_date,
                "n_predicted": len(pend["preds"]),
                "n_resolved": len(reals_res),
                "spearman_pred_realized": _r(
                    spearman(preds_res, reals_res)),
                "book_pred_5d_bps": _r(book_pred, 2),
                "book_realized_5d_bps": _r(book_real, 2),
                "legs": legs_res,
            }

        # ------------- daily paper-book accounting (v2 mechanics) ----
        prev_set, new_set = set(prev_book), set(book)
        changed = max(len(new_set - prev_set), len(prev_set - new_set))
        frac = changed / N_LEG_SLOTS
        cost = COST_BPS_PER_LEG * frac

        legs_out = []
        gross = 0.0
        if prev_book:
            for side, asset in prev_book:
                rows = by_date.get(asset, {})
                entry_bar = rows.get(prev_date)
                exit_bar = rows.get(date)
                stale = False
                if exit_bar is None:
                    exit_bar = last_close_at_or_before(
                        series[asset], date)
                    stale = True
                ret_bps = None
                if (entry_bar and exit_bar and entry_bar["close"]
                        and exit_bar["close"]
                        and entry_bar["close"] > 0
                        and exit_bar["close"] > 0):
                    ret = exit_bar["close"] / entry_bar["close"] - 1.0
                    if side == "short":
                        ret = -ret
                    ret_bps = ret * 1e4
                    gross += ret_bps / N_LEG_SLOTS
                legs_out.append({
                    "asset": asset, "side": side,
                    "ret_bps": (round(ret_bps, 2)
                                if ret_bps is not None else None),
                    "entry_close": (entry_bar["close"]
                                    if entry_bar else None),
                    "exit_close": (exit_bar["close"]
                                   if exit_bar else None),
                    "exit_close_date": (exit_bar["date"]
                                        if exit_bar else None),
                    "exit_basis": (exit_bar["basis"]
                                   if exit_bar else None),
                    "stale_fill": stale})
            if sum(1 for l in legs_out
                   if l["ret_bps"] is not None) < len(prev_book):
                gross = None  # incomplete fill: do not book a partial
        else:
            gross = 0.0

        day_net = (gross - cost) if gross is not None else None
        if gross is not None:
            cum["gross"] += gross
        cum["cost"] += cost
        if day_net is not None:
            cum["net"] += day_net
        equity *= 1.0 + ((gross if gross is not None else 0.0)
                         - cost) / 1e4

        lines.append({
            "schema_version": SCHEMA,
            "sleeve": SLEEVE + "_" + model.get("variant", "ridge_all")
                      if model.get("variant") != "ridge_all" else SLEEVE,
            "decision_date": date,
            "decision_ns": ns_by_date[date],
            "event": event,
            "model": {"name": model.get("variant", "ridge_all"),
                      "lambda": LAMBDA,
                      "train_cutoff": cutoff,
                      "n_train": model["n_train"],
                      "in_sample": bool(cutoff and date <= cutoff)},
            "gate": {"btc_close": _r(btc_close, 4),
                     "btc_ret20": _r(ret20, 6),
                     "on": gate_on},
            "universe": {"scored": n_scored,
                         "min_required": MIN_UNIVERSE,
                         "assets": [a for a, _ in scored]},
            "predictions": predictions,
            "book": {"long": sorted(a for s, a in book if s == "long"),
                     "short": sorted(a for s, a in book
                                     if s == "short")},
            "book_detail": [{"asset": a, "side": s,
                             "pred_bps": _r(dict(scored).get(a), 2)}
                            for s, a in book],
            "rebalance": {"prev_book_date": prev_date,
                          "legs_changed": changed,
                          "turnover_frac": round(frac, 4),
                          "cost_bps": round(cost, 4)},
            "realized": {"fills_decision_date": prev_date,
                         "legs": legs_out,
                         "gross_bps": _r(gross, 4),
                         "incomplete_fill": bool(
                             prev_book and gross is None)},
            "resolution": resolution,
            "day_net_bps": _r(day_net, 4),
            "cumulative": {"gross_bps": _r(cum["gross"], 4),
                           "cost_bps": _r(cum["cost"], 4),
                           "net_bps": _r(cum["net"], 4),
                           "net_equity": _r(equity, 8)},
        })
        prev_book = book
        prev_date = date
    return lines


# ---------------------------------------------------------------- stats


def _subset_stats(subset):
    n = len(subset)
    net_bps = sum(l["day_net_bps"] for l in subset
                  if l["day_net_bps"] is not None)
    eq = 1.0
    for l in subset:
        gross = l["realized"]["gross_bps"]
        day_ret = ((gross if gross is not None else 0.0)
                   - l["rebalance"]["cost_bps"]) / 1e4
        eq *= 1.0 + day_ret
    booked = [l["day_net_bps"] for l in subset
              if l["day_net_bps"] is not None]
    return {"days": n,
            "days_booked": len(booked),
            "net_bps": _r(net_bps, 4),
            "mean_day_net_bps": (_r(net_bps / len(booked), 4)
                                 if booked else None),
            "net_equity": _r(eq, 6),
            "score_days": sum(1 for l in subset
                              if l["event"] == "score")}


def backfill_stats(lines):
    events = defaultdict(int)
    peak, mdd = 1.0, 0.0
    for l in lines:
        events[l["event"]] += 1
        eq = l["cumulative"]["net_equity"]
        if eq is not None:
            peak = max(peak, eq)
            if peak:
                mdd = max(mdd, (peak - eq) / peak)
    in_s = [l for l in lines if l["model"]["in_sample"]]
    oos = [l for l in lines if not l["model"]["in_sample"]]
    last = lines[-1] if lines else None
    return {"events": dict(sorted(events.items())),
            "all": _subset_stats(lines),
            "in_sample": _subset_stats(in_s),
            "out_of_sample": _subset_stats(oos),
            "final_cumulative": (last["cumulative"] if last else None),
            "max_drawdown_frac": _r(mdd, 6)}


def _resolution_stats(subset):
    preds, reals, book_p, book_r = [], [], [], []
    n_res_dates = 0
    for l in subset:
        res = l.get("resolution")
        if not res:
            continue
        n_res_dates += 1
        for leg in res["legs"]:
            if leg["realized_5d_bps"] is not None:
                preds.append(leg["pred_bps"])
                reals.append(leg["realized_5d_bps"])
        if (res["book_pred_5d_bps"] is not None
                and res["book_realized_5d_bps"] is not None):
            book_p.append(res["book_pred_5d_bps"])
            book_r.append(res["book_realized_5d_bps"])
    sign_acc = (float(np.mean([(p > 0) == (r > 0)
                               for p, r in zip(preds, reals)]))
                if preds else None)
    return {"n_resolution_dates": n_res_dates,
            "n_asset_pairs": len(preds),
            "spearman_pred_realized": _r(spearman(preds, reals)),
            "sign_accuracy": _r(sign_acc),
            "mean_realized_5d_bps": (_r(float(np.mean(reals)), 2)
                                     if reals else None),
            "n_books": len(book_p),
            "spearman_book": _r(spearman(book_p, book_r)),
            "mean_book_pred_5d_bps": (_r(float(np.mean(book_p)), 2)
                                      if book_p else None),
            "mean_book_realized_5d_bps": (_r(float(np.mean(book_r)), 2)
                                          if book_r else None),
            "book_hit_rate": (_r(float(np.mean([r > 0
                                                for r in book_r])), 4)
                              if book_r else None)}


def pred_vs_realized_stats(lines):
    in_s = [l for l in lines if l["model"]["in_sample"]]
    oos = [l for l in lines if not l["model"]["in_sample"]]
    return {"all": _resolution_stats(lines),
            "in_sample": _resolution_stats(in_s),
            "out_of_sample": _resolution_stats(oos)}


def replay_full(args):
    model = fit_model(args.mega, args.btc_csv, variant=args.model)
    series, extra = load_inputs(args.cohort, args.supplement)
    add_bar_features(series)
    diff, n_checked = crosscheck_features(series, args.cohort)
    lines = replay(series, model)
    return lines, model, extra, diff, n_checked


def model_block(model):
    return {"name": model.get("variant", "ridge_all"),
            "lambda": model["lambda"],
            "train_cutoff": model["train_cutoff"],
            "n_train": model["n_train"],
            "train_mean_bps": _r(model["train_mean_bps"], 4),
            "assembly": model["assembly"],
            "coef": {k: _r(v, 6) for k, v in model["coef"].items()},
            "intercept_bps": _r(model["intercept_bps"], 6)}


def cmd_run(args):
    lines, model, extra, diff, n_checked = replay_full(args)
    seen = set()
    if args.ledger.exists():
        for l in args.ledger.read_text().splitlines():
            if not l.strip():
                continue
            seen.add(json.loads(l)["decision_date"])

    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    appended = skipped = 0
    events = defaultdict(int)
    with args.ledger.open("a", encoding="utf-8") as stream:
        for line in lines:
            if line["decision_date"] in seen:
                skipped += 1
                continue
            stream.write(json.dumps(line, sort_keys=True) + "\n")
            appended += 1
            events[line["event"]] += 1

    last = lines[-1] if lines else None
    print(json.dumps({
        "ledger": str(args.ledger),
        "sleeve": SLEEVE,
        "model": model_block(model),
        "supplement_records_merged": extra,
        "feature_recompute_max_abs_diff": diff,
        "feature_crosschecked_bars": n_checked,
        "decision_dates": len(lines),
        "dates_already_recorded": skipped,
        "dates_appended": appended,
        "events_appended": dict(sorted(events.items())),
        "backfill": backfill_stats(lines),
        "predictions_vs_realized": pred_vs_realized_stats(lines),
        "last_line": ({"decision_date": last["decision_date"],
                       "event": last["event"],
                       "gate": last["gate"],
                       "book": last["book"],
                       "cumulative": last["cumulative"]}
                      if last else None),
    }, indent=2, sort_keys=True))
    return 0


def cmd_snapshot(args):
    lines, model, extra, diff, n_checked = replay_full(args)
    last = lines[-1] if lines else None
    if last is None:
        print(json.dumps({"schema_version": SCHEMA + "-snapshot",
                          "sleeve": SLEEVE, "status": "empty"}))
        return 0
    print(json.dumps({
        "schema_version": SCHEMA + "-snapshot",
        "sleeve": SLEEVE,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                      time.gmtime()),
        "model": model_block(model),
        "supplement_records_merged": extra,
        "feature_recompute_max_abs_diff": diff,
        "as_of_decision_date": last["decision_date"],
        "event": last["event"],
        "gate": last["gate"],
        "universe": last["universe"],
        "ranked_predictions": last["predictions"],
        "current_book": last["book"],
        "book_detail": last["book_detail"],
        "latest_resolution": last["resolution"],
        "cumulative": last["cumulative"],
        "backfill": backfill_stats(lines),
        "predictions_vs_realized": pred_vs_realized_stats(lines),
    }, indent=2, sort_keys=True))
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", type=Path, default=COHORT)
    ap.add_argument("--supplement", type=Path, action="append",
                    default=None,
                    help="refresh supplement JSONL (repeatable; merged "
                         "and deduped by id). Defaults: "
                         "binance_refresh_v1 (legacy mark-price bars) + "
                         "binance_xs_refresh_v1 (XS close bars)")
    ap.add_argument("--mega", type=Path, default=MEGA,
                    help="mega cohort for the ridge_all fit")
    ap.add_argument("--btc-csv", type=Path, default=BTC_CSV,
                    help="BTC daily csv for train-time btc_ret20")
    ap.add_argument("--ledger", type=Path, default=LEDGER)
    ap.add_argument("--model", choices=MODEL_VARIANTS, default="ridge_all",
                    help="ridge_min3_top30 = T127/T130 parsimonious variant: "
                         "{dfh20, btc_ret20, dfh20*btc_ret20} trained on the "
                         "top-30-by-quote-volume mega symbols only")
    ap.add_argument("--universe", choices=("xs_v1", "xs_v2"), default="xs_v1",
                    help="xs_v2 = 30-asset T129 cohort (perp_pit_xs_v2 + the "
                         "xs2 refresh supplement) writing to "
                         "results/forward_ledger_v4_xs2.jsonl; xs_v1 is the "
                         "unchanged default used by the daily chain")
    ap.add_argument("--snapshot", action="store_true",
                    help="print sleeve state as of the latest decision "
                         "date")
    args = ap.parse_args()
    if args.universe == "xs_v2":
        if args.cohort == COHORT:
            args.cohort = COHORT_XS2
        if args.supplement is None:
            args.supplement = SUPPLEMENTS_XS2
        if args.ledger == LEDGER:
            args.ledger = LEDGER_XS2
    if args.model != "ridge_all" and args.ledger.name in (
            "forward_ledger_v4.jsonl", "forward_ledger_v4_xs2.jsonl"):
        args.ledger = args.ledger.with_name(
            args.ledger.stem + "_" + args.model + ".jsonl")
    if args.supplement is None:
        args.supplement = SUPPLEMENTS
    return cmd_snapshot(args) if args.snapshot else cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
