#!/usr/bin/env python3
"""Cross-sectional sanity arm for the T102 multi-asset pilot.

Question: per decision day, rank the pilot assets by a trailing-90 mid-rank
percentile score and ask whether the top-ranked asset's next-5d gross mark return
differs from the bottom-ranked asset's.

Arms:
  funding_pct    per-asset mid-rank pct of today's last_funding_rate vs its
                 trailing 90 daily values (repo convention, PIT-safe); the
                 funding-crowding hypothesis predicts high-funding assets
                 underperform, so the contrast is fwd(bottom) - fwd(top)
  dist_from_high per-asset mark close vs its trailing-90 high (close/max - 1);
                 ranked the same way, contrast reported sign-agnostically

Outputs per-day cross-sectional spreads, a pooled contrast (mean spread with a
t-stat across days), per-day and pooled cross-sectional Spearman, and an honest
power note: 10 assets is a THIN cross-section, ~89 decision days with overlapping
5-day labels are heavily autocorrelated, and no multiple-testing correction is
applied — this is a viability check, not a signal claim.

Gate: the arm only runs if >=8 symbols have usable records; otherwise it writes
an honest skipped report and exits 0. Measurement only — no fitting, no trading,
no network.
"""
import argparse
import datetime as dt
import json
import math
import pathlib
import zipfile
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_xs_pilot_v1/records.jsonl"
ARCHIVE_ROOTS = (ROOT / "data/binance_vision_v1", ROOT / "data/binance_xs_pilot_v1")
OUT = ROOT / "results/financial_signal_xs_pilot_v1.json"
LOOKBACK = 90          # trailing-90 records per asset (~90d at daily granularity)
MIN_WINDOW = 20        # new symbols only have pilot-month history; floor for a usable pct
MIN_ASSETS_PER_DAY = 4 # below this a top/bottom contrast is meaningless
FWD_DAYS = 5


def mid_rank_pct(window, x):
    """Mid-rank percentile of x vs trailing window (ties count half)."""
    if x is None or not window:
        return None
    return (sum(1 for w in window if w < x)
            + 0.5 * sum(1 for w in window if w == x)) / len(window)


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            r[order[k]] = avg
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
    return num / (dx * dy)


def t_stat(xs):
    """Mean and t of a day-series (days treated as independent; see power note)."""
    n = len(xs)
    if n < 5:
        return {"n": n, "mean": (sum(xs) / n) if n else None, "t": None}
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(var)
    return {"n": n, "mean": mean, "sd": sd,
            "t": (mean / (sd / math.sqrt(n))) if sd > 0 else None}


def read_rows(path):
    with zipfile.ZipFile(path) as archive:
        name = archive.namelist()[0]
        return [line for line in archive.read(name).decode().strip().splitlines() if line.strip()]


def resolve_root(roots, symbol):
    for root in roots:
        kind_dir = root / "markPriceKlines" / symbol
        if kind_dir.is_dir() and any(kind_dir.glob("*.zip")):
            return root
    return None


def load_full_history(roots, symbol):
    """Full daily (day, mark_close, last_funding_asof_close) series for one symbol.

    Trailing-window scores must use ALL history legitimately available at each
    decision day, not just the pilot-month slice: a legacy symbol's trailing-90
    funding_pct on 2026-06-01 was knowable from its multi-year archive. New
    symbols are naturally bounded by their pilot-month history (MIN_WINDOW).
    """
    root = resolve_root(roots, symbol)
    if root is None:
        return []
    marks = {}
    for path in sorted((root / "markPriceKlines" / symbol).glob("*.zip")):
        for line in read_rows(path):
            parts = line.split(",")
            if parts[0] == "open_time" or len(parts) < 11:
                continue
            marks[int(parts[0])] = {"close": float(parts[4]), "close_time": int(parts[6])}
    funding = []
    funding_dir = root / "fundingRate" / symbol
    if funding_dir.is_dir():
        for path in sorted(funding_dir.glob("*.zip")):
            for line in read_rows(path):
                parts = line.split(",")
                if parts[0] == "calc_time" or len(parts) < 3:
                    continue
                funding.append({"calc_time": int(parts[0]), "rate": float(parts[2])})
        funding.sort(key=lambda row: row["calc_time"])
    rows, cursor, last_rate = [], 0, None
    for open_time in sorted(marks):
        close_time = marks[open_time]["close_time"]
        while cursor < len(funding) and funding[cursor]["calc_time"] <= close_time:
            last_rate = funding[cursor]["rate"]
            cursor += 1
        rows.append({"day": dt.datetime.fromtimestamp(open_time / 1000, dt.timezone.utc)
                     .strftime("%Y-%m-%d"),
                     "funding": last_rate,
                     "mark": marks[open_time]["close"]})
    return rows


def score_assets(roots, symbols):
    """Per-asset full-history rows scored with PIT trailing windows + 5d fwd."""
    series = {}
    for symbol in symbols:
        rows = load_full_history(roots, symbol)
        fund = [r["funding"] for r in rows]
        mark = [r["mark"] for r in rows]
        for i, r in enumerate(rows):
            fwin = [x for x in fund[max(0, i - LOOKBACK):i] if x is not None]
            r["funding_pct"] = mid_rank_pct(fwin, r["funding"]) \
                if len(fwin) >= MIN_WINDOW else None
            hwin = [x for x in mark[max(0, i - LOOKBACK):i + 1] if x is not None]
            r["dist_from_high"] = (r["mark"] / max(hwin) - 1.0) \
                if r["mark"] is not None and len(hwin) >= MIN_WINDOW else None
            if i + FWD_DAYS < len(rows) and mark[i + FWD_DAYS] and mark[i] and mark[i] > 0:
                # require contiguous daily bars so a gap cannot fake a "5d" label
                gap = (dt.date.fromisoformat(rows[i + FWD_DAYS]["day"])
                       - dt.date.fromisoformat(r["day"])).days
                r["fwd_5d_bps"] = (10000.0 * ((mark[i + FWD_DAYS] / mark[i]) - 1.0)
                                   if gap == FWD_DAYS else None)
            else:
                r["fwd_5d_bps"] = None
        series[f"{symbol}-PERP"] = rows
    return series


def pilot_decision_days(path):
    """asset_id -> set of decision days present in the built pilot cohort."""
    days = defaultdict(set)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            days[record["asset_id"]].add(record["id"].rsplit(":", 1)[-1])
    return days


def cross_section_arm(series, decision_days, score_key, contrast_sign):
    """Per decision day: rank assets by score_key; contrast top vs bottom next-5d.

    Only rows whose day is a built pilot decision day for that asset are used.
    contrast_sign=+1 reports fwd(bottom)-fwd(top) (funding-crowding direction);
    contrast_sign=-1 reports fwd(top)-fwd(bottom).
    """
    by_day = defaultdict(list)
    for asset, rows in series.items():
        for r in rows:
            if r["day"] in decision_days.get(asset, ()) \
                    and r.get(score_key) is not None and r["fwd_5d_bps"] is not None:
                by_day[r["day"]].append((asset, r[score_key], r["fwd_5d_bps"]))

    days, pooled_x, pooled_y = [], [], []
    for day in sorted(by_day):
        xs = by_day[day]
        if len(xs) < MIN_ASSETS_PER_DAY:
            continue
        scores = [x[1] for x in xs]
        fwds = [x[2] for x in xs]
        s_min, s_max = min(scores), max(scores)
        if s_min == s_max:
            continue  # degenerate day: no cross-sectional dispersion
        top_fwd = [f for s, f in zip(scores, fwds) if s == s_max]
        bot_fwd = [f for s, f in zip(scores, fwds) if s == s_min]
        top = sum(top_fwd) / len(top_fwd)
        bot = sum(bot_fwd) / len(bot_fwd)
        rho = spearman(scores, fwds)
        days.append({"day": day, "n_assets": len(xs),
                     "top_score": s_max, "bottom_score": s_min,
                     "top_fwd_bps": top, "bottom_fwd_bps": bot,
                     "contrast_bps": contrast_sign * (bot - top),
                     "xs_spearman": rho})
        pooled_x.extend(scores)
        pooled_y.extend(fwds)

    contrast = t_stat([d["contrast_bps"] for d in days])
    rhos = [d["xs_spearman"] for d in days if d["xs_spearman"] is not None]
    return {
        "score": score_key,
        "contrast_direction": ("fwd(bottom_rank)-fwd(top_rank)"
                               if contrast_sign > 0 else
                               "fwd(top_rank)-fwd(bottom_rank)"),
        "days_evaluated": len(days),
        "assets_per_day": {"min": min(d["n_assets"] for d in days) if days else None,
                           "median": sorted(d["n_assets"] for d in days)[len(days) // 2]
                                     if days else None,
                           "max": max(d["n_assets"] for d in days) if days else None},
        "pooled_contrast_bps": {k: (round(v, 3) if isinstance(v, float) else v)
                                for k, v in contrast.items()},
        "mean_daily_xs_spearman": (round(sum(rhos) / len(rhos), 4)) if rhos else None,
        "pooled_spearman": (round(spearman(pooled_x, pooled_y), 4)
                            if spearman(pooled_x, pooled_y) is not None else None),
        "per_day": days,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", type=pathlib.Path, default=COHORT)
    parser.add_argument("--archive-roots",
                        default=",".join(str(r) for r in ARCHIVE_ROOTS),
                        help="comma-separated archive roots for full-history "
                             "trailing windows")
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    args = parser.parse_args()

    report = {"schema_version": "nanojev-financial-signal-xs-pilot-v1",
              "task": "T102 cross-sectional viability pilot",
              "cohort": str(args.cohort),
              "horizon_days": FWD_DAYS,
              "lookback": LOOKBACK, "min_window": MIN_WINDOW,
              "min_assets_per_day": MIN_ASSETS_PER_DAY}

    if not args.cohort.exists():
        report["status"] = "SKIPPED: cohort records.jsonl not found"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": report["status"], "out": str(args.out)}))
        return 0

    roots = [pathlib.Path(r.strip()) for r in args.archive_roots.split(",") if r.strip()]
    decision_days = pilot_decision_days(args.cohort)
    symbols = sorted(a[:-5] for a in decision_days if a.endswith("-PERP"))
    series = score_assets(roots, symbols)
    n_symbols = sum(1 for s in series.values() if s)
    report["symbols_in_cohort"] = sorted(series)
    if n_symbols < 8:
        report["status"] = (f"SKIPPED: only {n_symbols} symbols have usable archive "
                            "data (<8); cross-sectional arm not run")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": report["status"], "out": str(args.out)}))
        return 0

    report["status"] = "ran"
    report["arms"] = {
        "funding_pct": cross_section_arm(series, decision_days, "funding_pct", +1),
        "dist_from_high": cross_section_arm(series, decision_days, "dist_from_high", -1),
    }
    report["power_note"] = (
        "HONEST POWER: 10 assets is a thin cross-section — each day's contrast is "
        "a single top-vs-bottom difference, so one asset dominates the daily estimate. "
        "The 5 pilot symbols contribute valid scores only after ~20 days of their "
        "June-1-start history, so early-June cross-sections are the 5 legacy assets "
        "and grow to 10 later; composition changes mid-sample. The 5d forward labels "
        "of adjacent decision days overlap ~80%, so the ~85-day series has far fewer "
        "effective independent observations; the t-stat treats days as independent "
        "and overstates significance. No multiple-testing correction across the two "
        "arms. Gross mark moves only — no costs, funding cashflows or tradability. "
        "This is a viability check, not a signal claim.")
    report["honesty"] = {
        "not_a_return": "forward returns are gross mark-price moves; fees, spread, "
                        "slippage, funding cashflows and leverage are excluded",
        "not_live": "no orders, no account, no broker, no trading API was used",
        "no_profitability_claim": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    brief = {"status": "ran", "symbols": n_symbols, "out": str(args.out)}
    for name, arm in report["arms"].items():
        brief[name] = {"days": arm["days_evaluated"],
                       "median_assets": arm["assets_per_day"]["median"],
                       "pooled_contrast_bps": arm["pooled_contrast_bps"],
                       "pooled_spearman": arm["pooled_spearman"],
                       "mean_daily_xs_spearman": arm["mean_daily_xs_spearman"]}
    print(json.dumps(brief, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
