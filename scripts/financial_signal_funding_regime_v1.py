#!/usr/bin/env python3
"""T94: funding-sign regime conditioning of the funding-follow signal.

Context: T89 showed the hourly funding_pct signal only exists in
positive-funding regimes (Jul-Aug ~87% positive funding) and halves/loses
FDR when negative-funding months are added (Apr-Jun 44-50% positive). The
daily-scale spec research/financial_signal_spec_v1.json gates on BTC trend;
this task formalizes the funding-sign regime gate at hourly scale and
cross-checks it at daily scale.

Cohorts:
  hourly  data/perp_pit_intraday_v1/records.jsonl — 17,960 records,
          5 assets, 1h bars, 2026-04-02..2026-08-31 UTC.
          Label = label.forward_return_bps (4h forward mark bps, gross).
  daily   data/perp_pit_v1/records.jsonl — 6,554 records, 5 assets, 1d
          bars, 2023-01-25..2026-08-30 UTC. Target = 5d forward mark log
          return computed from features.mark_price (same construction as
          financial_signal_regime_v1 / labels_v4).

Regime rule (stated in research/financial_signal_funding_regime_protocol_v1.json):
  share_pos = fraction of the trailing window with last_funding_rate > 0
  (window EXCLUDES the current record; zeros count as non-positive).
  regime = positive if share_pos > 0.6, negative if share_pos < 0.4,
  mixed otherwise. Window = trailing-90 records per asset (~90h) on the
  hourly cohort, trailing-180 records per asset (~180d) on the daily
  cohort. Daily primary split: positive vs non-positive (mixed+negative
  pooled); 3-way split reported as detail.

Arms:
  R1  regime coverage: cell counts per asset + monthly positive share.
  R2  funding_pct (trailing-90 mid-rank pct of last_funding_rate,
      per asset) -> 4h fwd bps Spearman WITHIN each regime (3 cells).
  R3  hourly f2b2 analog: funding_pct>=0.80 AND basis_pct>=0.66 ->
      Welch arm-vs-rest mean fwd bps within each regime (3 contrasts).
  R4  daily cross-check: funding_pct (trailing-180) -> 5d log(mark)
      Spearman within positive vs non-positive regime (2 cells +
      3-way detail).

BH-FDR alpha=0.05 per family ({R2:3}, {R3:3}, {R4:2}); n/rho/p per cell.
Measurement only — no fitting, no trading, no promotion claims.
"""

import argparse
import datetime
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOURLY = ROOT / "data/perp_pit_intraday_v1/records.jsonl"
DAILY = ROOT / "data/perp_pit_v1/records.jsonl"
OUT = ROOT / "results/financial_signal_funding_regime_v1.json"
PROTOCOL = "research/financial_signal_funding_regime_protocol_v1.json"

ITRAIL = 90           # hourly: trailing-90 records per asset (~90h)
DTRAIL = 180          # daily: trailing-180 records per asset (~180d)
HOLD = 5              # daily forward horizon in bars (5d)
POS_HI, POS_LO = 0.6, 0.4   # share_pos regime thresholds
F_PCT, B_PCT = 0.80, 0.66   # f2b2 arm thresholds (frozen spec values)
MIN_N = 10            # per-side minimum (repo convention)


# ---------------------------------------------------------------- stats
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
    """Mid-rank Spearman rho + normal-approx t/p; None if n<MIN_N or a
    side has zero rank variance (project convention)."""
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
    return {"n": n, "rho": round(rho, 5), "t": round(t, 3),
            "p": round(p, 6)}


def welch(a, b):
    """Welch mean-diff of a vs b in bps; normal-approx two-sided p."""
    if len(a) < MIN_N or len(b) < MIN_N:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    se = math.sqrt(statistics.pvariance(a) / len(a)
                   + statistics.pvariance(b) / len(b))
    if se == 0:
        return None
    t = (ma - mb) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return {"n_arm": len(a), "n_rest": len(b),
            "mean_arm_bps": round(ma, 2), "mean_rest_bps": round(mb, 2),
            "diff_bps": round(ma - mb, 2), "t": round(t, 3),
            "p": round(p, 6)}


def regime(share_pos):
    if share_pos > POS_HI:
        return "positive"
    if share_pos < POS_LO:
        return "negative"
    return "mixed"


def bh_fdr(named_ps):
    """BH-FDR alpha=0.05 over a list of (name, p) — returns
    {name: survives} plus the sorted threshold table."""
    ps = sorted((p, n) for n, p in named_ps if p is not None)
    m = len(ps)
    table = [{"cell": n, "p": p, "alpha_bh": round(0.05 * (i + 1) / m, 6),
              "survives": p <= 0.05 * (i + 1) / m}
             for i, (p, n) in enumerate(ps)]
    return {e["cell"]: e["survives"] for e in table}, table


def load(path):
    recs = [json.loads(l) for l in path.read_text().splitlines()
            if l.strip()]
    by_asset = defaultdict(list)
    for r in recs:
        by_asset[r["asset_id"]].append(r)
    for rs in by_asset.values():
        rs.sort(key=lambda x: x["decision_ns"])
    return recs, by_asset


# ---------------------------------------------------------------- hourly
def hourly_rows(by_asset):
    """Eval rows: i >= ITRAIL per asset, fwd label present; carries
    share_pos/regime + funding_pct + basis_pct + fwd_bps."""
    rows = []
    for asset, rs in sorted(by_asset.items()):
        fund = [x["features"]["last_funding_rate"]["value"] for x in rs]
        basis = [x["features"]["mark_index_basis_bps"]["value"]
                 for x in rs]
        for i in range(ITRAIL, len(rs)):
            fwd = rs[i].get("label", {}).get("forward_return_bps")
            if fwd is None:
                continue
            w = fund[i - ITRAIL:i]
            sp = sum(1 for v in w if v > 0) / len(w)
            rows.append({"asset": asset, "ns": rs[i]["decision_ns"],
                         "share_pos": sp, "regime": regime(sp),
                         "f_pct": pct(w, fund[i]),
                         "b_pct": pct(basis[i - ITRAIL:i], basis[i]),
                         "fwd": fwd})
    return rows


def run_hourly(by_asset):
    rows = hourly_rows(by_asset)
    out = {"n_eval_rows": len(rows), "regimes": {}, "spearman": {},
           "f2b2": {}}

    # R1: coverage — regime cell counts per asset + per-month share of
    # records whose own last_funding_rate is positive (T89-parity context)
    cov = defaultdict(lambda: defaultdict(int))
    for r in rows:
        cov[r["asset"]][r["regime"]] += 1
    monthly = defaultdict(lambda: [0, 0])
    for asset, rs in sorted(by_asset.items()):
        fund = [x["features"]["last_funding_rate"]["value"] for x in rs]
        for i in range(ITRAIL, len(rs)):
            if rs[i].get("label", {}).get("forward_return_bps") is None:
                continue
            m = datetime.datetime.fromtimestamp(
                rs[i]["decision_ns"] / 1e9,
                datetime.timezone.utc).strftime("%Y-%m")
            monthly[m][0] += 1 if fund[i] > 0 else 0
            monthly[m][1] += 1
    out["regimes"]["per_asset"] = {a: dict(c) for a, c in
                                   sorted(cov.items())}
    out["regimes"]["total"] = {
        k: sum(c[k] for c in cov.values())
        for k in ("positive", "mixed", "negative")}
    out["regimes"]["monthly_positive_funding_share"] = {
        m: {"share_pos": round(p / n, 4), "n": n}
        for m, (p, n) in sorted(monthly.items())}

    # R2: funding_pct -> fwd Spearman within each regime (+ pooled ref)
    sp_ps = []
    for reg in ("positive", "mixed", "negative"):
        sub = [r for r in rows if r["regime"] == reg]
        s = spearman([r["f_pct"] for r in sub], [r["fwd"] for r in sub])
        out["spearman"][reg] = s
        sp_ps.append((f"spearman_{reg}", s["p"] if s else None))
    out["spearman"]["all_rows_reference"] = spearman(
        [r["f_pct"] for r in rows], [r["fwd"] for r in rows])
    fdr, table = bh_fdr(sp_ps)
    for reg in ("positive", "mixed", "negative"):
        if out["spearman"][reg]:
            out["spearman"][reg]["fdr_pass"] = fdr.get(
                f"spearman_{reg}", False)
    out["fdr_spearman_family"] = table

    # R3: f2b2 arm vs rest within each regime (+ pooled ref)
    w_ps = []
    for reg in ("positive", "mixed", "negative", "all_rows_reference"):
        sub = rows if reg == "all_rows_reference" else \
            [r for r in rows if r["regime"] == reg]
        arm = [r["fwd"] for r in sub
               if r["f_pct"] >= F_PCT and r["b_pct"] >= B_PCT]
        rest = [r["fwd"] for r in sub
                if not (r["f_pct"] >= F_PCT and r["b_pct"] >= B_PCT)]
        w = welch(arm, rest)
        out["f2b2"][reg] = w
        if reg != "all_rows_reference":
            w_ps.append((f"f2b2_{reg}", w["p"] if w else None))
    fdr, table = bh_fdr(w_ps)
    for reg in ("positive", "mixed", "negative"):
        if out["f2b2"][reg]:
            out["f2b2"][reg]["fdr_pass"] = fdr.get(f"f2b2_{reg}", False)
    out["fdr_f2b2_family"] = table
    return rows, out


# ---------------------------------------------------------------- daily
def run_daily(by_asset):
    rows = []
    zero_windows = 0
    for asset, rs in sorted(by_asset.items()):
        fund = [x["features"]["last_funding_rate"]["value"] for x in rs]
        mark = [x["features"]["mark_price"]["value"] for x in rs]
        for i in range(DTRAIL, len(rs) - HOLD):
            if mark[i] <= 0 or mark[i + HOLD] <= 0:
                continue
            w = fund[i - DTRAIL:i]
            sp = sum(1 for v in w if v > 0) / len(w)
            zero_windows += sum(1 for v in w if v == 0)
            rows.append({"asset": asset, "ns": rs[i]["decision_ns"],
                         "share_pos": sp, "regime": regime(sp),
                         "regime2": ("positive" if sp > POS_HI
                                     else "non_positive"),
                         "f_pct": pct(w, fund[i]),
                         "ret": math.log(mark[i + HOLD] / mark[i])})

    out = {"n_eval_rows": len(rows),
           "n_zero_funding_inside_windows": zero_windows,
           "regimes": {}, "spearman": {}, "spearman_3way_detail": {}}
    cov = defaultdict(lambda: defaultdict(int))
    for r in rows:
        cov[r["asset"]][r["regime"]] += 1
    out["regimes"]["per_asset"] = {a: dict(c) for a, c in
                                   sorted(cov.items())}
    out["regimes"]["total"] = {
        k: sum(c[k] for c in cov.values())
        for k in ("positive", "mixed", "negative")}

    # R4: funding_pct -> 5d log(mark) Spearman, positive vs non-positive
    sp_ps = []
    for reg in ("positive", "non_positive"):
        sub = [r for r in rows if r["regime2"] == reg]
        s = spearman([r["f_pct"] for r in sub],
                     [r["ret"] for r in sub])
        out["spearman"][reg] = s
        sp_ps.append((f"daily_{reg}", s["p"] if s else None))
    out["spearman"]["all_rows_reference"] = spearman(
        [r["f_pct"] for r in rows], [r["ret"] for r in rows])
    for reg in ("mixed", "negative"):   # 3-way detail (not in FDR family)
        sub = [r for r in rows if r["regime"] == reg]
        out["spearman_3way_detail"][reg] = spearman(
            [r["f_pct"] for r in sub], [r["ret"] for r in sub])
    fdr, table = bh_fdr(sp_ps)
    for reg in ("positive", "non_positive"):
        if out["spearman"][reg]:
            out["spearman"][reg]["fdr_pass"] = fdr.get(
                f"daily_{reg}", False)
    out["fdr_spearman_family"] = table
    return rows, out


# ---------------------------------------------------------------- run
def run():
    h_recs, h_by = load(HOURLY)
    d_recs, d_by = load(DAILY)
    _, hourly = run_hourly(h_by)
    _, daily = run_daily(d_by)

    return {
        "schema_version": "nanojev-financial-signal-funding-regime-v1",
        "status": "measurement_complete",
        "task": "T94",
        "protocol": PROTOCOL,
        "contract": {
            "regime_rule": ("share_pos = fraction of trailing window with "
                            "last_funding_rate>0 (window excludes current "
                            "record; zeros non-positive); positive if "
                            ">0.6, negative if <0.4, mixed otherwise"),
            "hourly": {"cohort": str(HOURLY.relative_to(ROOT)),
                       "cohort_sha256": hashlib.sha256(
                           HOURLY.read_bytes()).hexdigest(),
                       "n_records": len(h_recs),
                       "window_records": ITRAIL,
                       "label": "label.forward_return_bps (4h fwd mark "
                                "bps, gross)"},
            "daily": {"cohort": str(DAILY.relative_to(ROOT)),
                      "cohort_sha256": hashlib.sha256(
                          DAILY.read_bytes()).hexdigest(),
                      "n_records": len(d_recs),
                      "window_records": DTRAIL,
                      "hold_bars": HOLD,
                      "label": "5d forward mark log return from "
                               "features.mark_price"},
            "f2b2_thresholds": {"funding_pct": F_PCT, "basis_pct": B_PCT},
            "min_n": MIN_N, "fdr_alpha": 0.05,
            "fdr_families": {"hourly_spearman": 3, "hourly_f2b2": 3,
                             "daily_spearman": 2}},
        "hourly": hourly,
        "daily": daily,
        "caveats": [
            "4h labels on 1h bars overlap ~4x and 5d labels on 1d bars "
            "overlap ~5x; serial correlation makes normal-approx p "
            "nominal/optimistic",
            "regime and funding_pct share the same trailing funding "
            "window — conditioning is not an independent gate",
            "daily BNBUSDT carries last_funding_rate=0 on 730/1309 "
            "records (~56%, spread across the full span — funding "
            "frequently settles at exactly 0 for BNB); zeros count as "
            "non-positive, so BNB never enters the positive regime",
            "daily non-positive cell pools mixed+negative (heterogeneous, "
            "smaller n); 3-way detail reported",
            "gross mark moves only — no fees, spread, slippage, funding "
            "cashflows; not a tradability or profitability claim"]}


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
