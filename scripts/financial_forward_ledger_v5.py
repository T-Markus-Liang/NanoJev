#!/usr/bin/env python3
"""T148: forward paper-trading ledger for the age-conditioned XS book
(fifth sleeve) — the T135 inversion-aware conditioning operationalized
on the 30-asset xs_v2 universe.

Signal source (T135, results/financial_signal_age_model_v1.json):
  on the mega daily cohort the conditioned book
  ``cond_inversion_flip`` — long the top-dfh20 tail of NON-CENSORED
  assets younger than 365d, short the TOP-dfh20 tail of non-censored
  assets older than 365d (shorting old over-extended names: the T131
  gt365 inversion where top-dfh old names earn -85.2bps fwd5) — earned
  +8.20bps/day hold-5 net (matched-days Sharpe 0.778 vs the plain
  book's 0.245). The model-side age features were rejected; the book
  overlay is the measured effect ("age_helps_book_not_model",
  BOOK_OVERLAY_YES_MODEL_FEATURES_NO). T135's headline number is the
  hold-5 TRANCHE book (EW over the last 5 days' selections, ~1/5
  refresh per day); this ledger runs the sleeve's daily-rebalance form
  for consistency with the v2/v4 ledger mechanics — T135 measured the
  daily-rebalance conditioned book at -0.79bps/d delta vs plain on the
  mega cohort, so the tranche effect is the optimistic bound and this
  ledger is the harder daily accounting.

Sleeve spec:
  universe:   the 30 USDT-M perps in data/perp_pit_xs_v2/records.jsonl
              (research/live_universe_v2.json), merged with the three
              refresh supplements — binance_xs_refresh_v1 (5 XS closes),
              binance_xs2_refresh_v1 (20 xs2 closes) AND
              binance_refresh_v1 (the 5 majors BTC/ETH/BNB/SOL/XRP,
              mark_price basis — required for the BTC gate to stay live
              post-cohort; the same SUPPLEMENTS_XS2 merge v2/v4 use for
              --universe xs_v2). Deduped by record id (cohort wins);
              per asset, a supplement date already in the cohort is
              skipped. dfh20 is recomputed from the merged close series
              (strictly-prior 20 contiguous bars) so refreshed bars rank
              identically to cohort bars; recomputation is
              cross-checked against the cohort feature.
  age:        age_d = decision_date - asset's first record date in the
              MERGED series, in days (T135 convention). An asset whose
              first bar is the cohort minimum date is CENSORED (first
              seen at archive start = listing date unknown): it is never
              young and is kept OUT of the old pool too — T131/T135
              measured the censored bucket separately.
                young pool: not censored AND age_d < 365
                old pool:   not censored AND age_d >= 365
              (task spec uses >=365; T135 used strict >365 so the
              single boundary day sat in neither pool — immaterial)
  gate:       a book is only opened when btc_ret20 > 0 — BTCUSDT-PERP
              close / close of the bar exactly 20 calendar days prior -
              1 on the merged series (v4 construction; the same master
              gate variable T135 conditioned on).
  book:       young candidates = young pool with a usable dfh20 and
              positive close that day; likewise old. If a side has >=3
              candidates it contributes its top-2 by dfh20 (long for
              young, SHORT for old — the inversion flip); a side with
              <3 candidates contributes nothing (its slots stay empty,
              earning 0 and costing nothing). Legs are fixed 1/4 book
              weight each, so a one-sided book runs at 0.5 gross.
              Gate on but no side qualifying -> ``flat_thin_buckets``.
              >=6 scored assets total are required for a valid rank
              (graceful-degradation floor shared with v2/v4).
  costs:      5bps per unit of one-sided leg notional traded;
              cost_day_bps = 5 * max(legs_added, legs_removed) / 4 vs
              the previous book (identical to v2/v4).
  pnl:        the book decided at t-1 is held t-1 -> t close-to-close,
              equal 1/4 leg weight (short legs negated); missing exit
              bars carry forward the asset's last prior close flagged
              stale; a book with any unpriced leg is not booked.
              day_net_bps = gross - cost; cumulative.net_equity
              compounds ((gross or 0) - cost)/1e4 daily (v4 convention).

Ledger contract (results/forward_ledger_v5.jsonl): same shape as v2 —
append-only JSONL, one line per merged-input decision date, idempotent
by decision_date, content-light (buckets/book/costs — never raw
feature payloads), fully determined at write time.

--snapshot prints the sleeve state as of the latest decision date:
gate, age buckets, current book, cumulative, per-year stats.

Simulated paper accounting only — no orders, no broker, no account
access, not a tradability or profitability claim.
"""

import argparse
import datetime as dt
import json
import math
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_xs_v2/records.jsonl"
# xs_v2 refresh convention (v2/v4 SUPPLEMENTS_XS2): the 5 majors ride on
# the legacy mark-price supplement; without it BTCUSDT-PERP stalls at
# the cohort tail and the gate cannot evaluate refreshed dates.
SUPPLEMENTS = [ROOT / "data/binance_refresh_v1/records.jsonl",
               ROOT / "data/binance_xs_refresh_v1/records.jsonl",
               ROOT / "data/binance_xs2_refresh_v1/records.jsonl"]
LEDGER = ROOT / "results/forward_ledger_v5.jsonl"

SCHEMA = "nanojev-forward-ledger-v5"
SLEEVE = "age_cond_xs2_long_top2_young_short_top2_old_btc_ret20_gate_v1"
BTC_ASSET = "BTCUSDT-PERP"
DFH_LOOKBACK = 20        # strictly-prior contiguous bars for the high
BTC_LOOKBACK = 20        # btc_ret20: close / close[t-20] - 1
EDGE = 2                 # top-2 per qualifying side
MIN_SIDE = 3             # a side needs >=3 candidates to field legs
N_LEG_SLOTS = 4          # 2 long + 2 short slots, each 1/4 of book
MIN_UNIVERSE = 6         # graceful-degradation floor (v2/v4 shared)
AGE_YOUNG_LT = 365       # young pool: age_d < 365 (non-censored)
COST_BPS_PER_LEG = 5.0   # per unit of one-sided leg notional traded
EVENTS = ("rebalance", "flat_gate_off", "flat_thin_buckets",
          "skipped_insufficient_universe", "skipped_no_btc_ret20")


def _r(x, nd=4):
    return round(x, nd) if isinstance(x, (int, float)) else x


def feat(record, name):
    f = record.get("features", {}).get(name)
    return f.get("value") if isinstance(f, dict) else None


def dt_date(text):
    return dt.date.fromisoformat(text)


def record_close(record):
    """XS price basis: klines close as mark proxy; refreshed legacy bars
    carry mark_price instead (refresh_v1 shape). Same as v2/v4."""
    close = feat(record, "close")
    if close is not None:
        return close, "close"
    mark = feat(record, "mark_price")
    if mark is not None:
        return mark, "mark_price"
    return None, None


def load_inputs(cohort_path, supplement_paths):
    """asset -> sorted [{date, close, basis, decision_ns, record_id}].
    Same merge as v2/v4: cohort + supplements deduped by record id
    (cohort wins); per asset, a supplement date already present in the
    cohort is skipped."""
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
                                 "decision_ns": r["decision_ns"],
                                 "record_id": r["id"]}
    return {a: [days[d] for d in sorted(days)]
            for a, days in by_asset.items()}, extra


def add_dfh20(series):
    """Recompute dfh20 per bar in place: close / max(strictly-prior 20
    contiguous closes) - 1; None when the contiguous window is
    unavailable (same construction as v2 / build_xs_v2)."""
    for rows in series.values():
        for i, r in enumerate(rows):
            contiguous = (i >= DFH_LOOKBACK
                          and (dt_date(r["date"]) - dt_date(
                               rows[i - DFH_LOOKBACK]["date"])).days
                          == DFH_LOOKBACK)
            dfh = None
            if contiguous and r["close"] is not None and r["close"] > 0:
                prior = [rows[j]["close"] for j in
                         range(i - DFH_LOOKBACK, i)]
                if all(c is not None and c > 0 for c in prior):
                    dfh = r["close"] / max(prior) - 1.0
            r["dfh20"] = dfh


def crosscheck_dfh20(series, cohort_path):
    """max abs diff between recomputed dfh20 and the cohort feature."""
    cohort_feature = {}
    for l in cohort_path.read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        f = feat(r, "dfh20")
        if f is not None:
            cohort_feature[r["id"]] = f
    diff, n = 0.0, 0
    for rows in series.values():
        for r in rows:
            f = cohort_feature.get(r["record_id"])
            if f is not None and r["dfh20"] is not None:
                diff = max(diff, abs(f - r["dfh20"]))
                n += 1
    return diff, n


def add_age(series):
    """T135 censor-aware age on the merged series: age_d = bar date -
    asset's first record date (days); censored = the asset's first bar
    is the cohort minimum date (listing date unknowable). Censored
    assets are in NEITHER conditioning pool."""
    first_ord = {a: dt_date(rows[0]["date"]).toordinal()
                 for a, rows in series.items() if rows}
    cohort_min = min(first_ord.values()) if first_ord else None
    censored = {a for a, fo in first_ord.items() if fo == cohort_min}
    for a, rows in series.items():
        fo = first_ord[a]
        for r in rows:
            r["age_d"] = dt_date(r["date"]).toordinal() - fo
            r["censored"] = a in censored
    return {"cohort_min_date": (dt.date.fromordinal(cohort_min)
                                .isoformat() if cohort_min else None),
            "n_censored_assets": len(censored),
            "censored_assets": sorted(censored)}


def age_bucket(bar):
    """young (<365d) / old (>=365d) / censored — T135 cohorts."""
    if bar["censored"]:
        return "censored"
    return "young" if bar["age_d"] < AGE_YOUNG_LT else "old"


def btc_ret20_map(btc_rows):
    """date -> (btc_close, ret20 or None) on the merged BTCUSDT-PERP
    series. ret20 = close / close of the bar exactly 20 calendar days
    prior - 1, both closes positive (v4 construction — the T135/T119
    master gate variable)."""
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


def last_close_at_or_before(rows, date):
    prior = [r for r in rows if r["date"] <= date]
    return prior[-1] if prior else None


def replay(series):
    """Deterministic full replay -> ordered ledger line dicts."""
    cohort_min_date = min(rows[0]["date"] for rows in series.values())
    dates = sorted({r["date"] for rows in series.values() for r in rows})
    by_date = {asset: {r["date"]: r for r in rows}
               for asset, rows in series.items()}
    ns_by_date = defaultdict(int)
    for rows in series.values():
        for r in rows:
            ns_by_date[r["date"]] = max(ns_by_date[r["date"]],
                                        r["decision_ns"])
    btc_gate = btc_ret20_map(series.get(BTC_ASSET, []))

    lines = []
    prev_book = []          # legs decided on the previous date
    prev_date = None
    cum = {"gross": 0.0, "cost": 0.0, "net": 0.0}
    equity = 1.0
    for date in dates:
        # rankable cross-section + age pools for this decision date
        ranked = []         # (dfh20, asset) over all scored assets
        pools = {"young": [], "old": [], "censored": []}
        for asset, m in by_date.items():
            r = m.get(date)
            if (r is None or r["dfh20"] is None
                    or r["close"] is None or r["close"] <= 0):
                continue
            ranked.append((r["dfh20"], asset))
            pools[age_bucket(r)].append((r["dfh20"], asset))
        for pool in pools.values():
            pool.sort(key=lambda x: (x[0], x[1]))  # dfh asc, tiebreak

        btc_close, ret20 = btc_gate.get(date, (None, None))
        gate_on = ret20 is not None and ret20 > 0

        if ret20 is None:
            event = "skipped_no_btc_ret20"
            book = []
        elif len(ranked) < MIN_UNIVERSE:
            event = "skipped_insufficient_universe"
            book = []
        elif not gate_on:
            event = "flat_gate_off"
            book = []
        else:
            # inversion flip: long TOP dfh of young, short TOP dfh of
            # old — each side fields top-2 only if >=3 candidates
            book = []
            if len(pools["young"]) >= MIN_SIDE:
                book += [("long", a) for _, a in pools["young"][-EDGE:]]
            if len(pools["old"]) >= MIN_SIDE:
                book += [("short", a) for _, a in pools["old"][-EDGE:]]
            event = "rebalance" if book else "flat_thin_buckets"

        # turnover vs the previous book (v2/v4 mechanics)
        prev_set, new_set = set(prev_book), set(book)
        changed = max(len(new_set - prev_set), len(prev_set - new_set))
        frac = changed / N_LEG_SLOTS
        cost = COST_BPS_PER_LEG * frac

        # realized: the PREVIOUS book was held prev_date -> date
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

        score_of = {a: s for s, a in ranked}
        age_of = {}
        for bucket, pool in pools.items():
            for _, a in pool:
                age_of[a] = bucket
        lines.append({
            "schema_version": SCHEMA,
            "sleeve": SLEEVE,
            "decision_date": date,
            "decision_ns": ns_by_date[date],
            "event": event,
            "gate": {"btc_close": _r(btc_close, 4),
                     "btc_ret20": _r(ret20, 6),
                     "on": gate_on},
            "universe": {"scored": len(ranked),
                         "min_required": MIN_UNIVERSE,
                         "assets": [a for _, a in
                                    sorted(ranked, key=lambda x:
                                           (-x[0], x[1]))]},
            "age_buckets": {
                "cohort_min_date": cohort_min_date,
                "young_candidates": len(pools["young"]),
                "old_candidates": len(pools["old"]),
                "censored_scored": len(pools["censored"]),
                "young_assets": sorted(a for _, a in pools["young"]),
                "old_assets": sorted(a for _, a in pools["old"])},
            "book": {"long": sorted(a for s, a in book if s == "long"),
                     "short": sorted(a for s, a in book
                                     if s == "short")},
            "book_detail": [{"asset": a, "side": s,
                             "dfh20": _r(score_of.get(a), 6),
                             "age_d": (by_date[a][date]["age_d"]
                                       if a in by_date
                                       and date in by_date[a]
                                       else None),
                             "age_bucket": age_of.get(a)}
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
    if len(booked) > 1:
        mean = sum(booked) / len(booked)
        var = sum((x - mean) ** 2 for x in booked) / (len(booked) - 1)
        sharpe = (mean / math.sqrt(var) * math.sqrt(365)
                  if var > 0 else None)
    else:
        mean = sharpe = None
    return {"days": n,
            "days_booked": len(booked),
            "rebalance_days": sum(1 for l in subset
                                  if l["event"] == "rebalance"),
            "net_bps": _r(net_bps, 4),
            "mean_day_net_bps": _r(mean, 4),
            "sharpe_annualized": _r(sharpe, 4),
            "net_equity": _r(eq, 6)}


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
    by_year = defaultdict(list)
    for l in lines:
        by_year[l["decision_date"][:4]].append(l)
    last = lines[-1] if lines else None
    return {"events": dict(sorted(events.items())),
            "all": _subset_stats(lines),
            "per_year": {y: _subset_stats(by_year[y])
                         for y in sorted(by_year)},
            "final_cumulative": (last["cumulative"] if last else None),
            "max_drawdown_frac": _r(mdd, 6)}


def replay_with_lines(args):
    series, extra = load_inputs(args.cohort, args.supplement)
    add_dfh20(series)
    diff, n_checked = crosscheck_dfh20(series, args.cohort)
    age_meta = add_age(series)
    lines = replay(series)
    return lines, extra, diff, n_checked, age_meta


def cmd_run(args):
    lines, extra, diff, n_checked, age_meta = replay_with_lines(args)
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
        "cohort": str(args.cohort),
        "supplement_records_merged": extra,
        "dfh20_recompute_max_abs_diff": diff,
        "dfh20_crosschecked_bars": n_checked,
        "age_meta": age_meta,
        "decision_dates": len(lines),
        "dates_already_recorded": skipped,
        "dates_appended": appended,
        "events_appended": dict(sorted(events.items())),
        "backfill": backfill_stats(lines),
        "last_line": ({"decision_date": last["decision_date"],
                       "event": last["event"],
                       "gate": last["gate"],
                       "book": last["book"],
                       "age_buckets": last["age_buckets"],
                       "cumulative": last["cumulative"]}
                      if last else None),
    }, indent=2, sort_keys=True))
    return 0


def cmd_snapshot(args):
    lines, extra, diff, n_checked, age_meta = replay_with_lines(args)
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
        "supplement_records_merged": extra,
        "dfh20_recompute_max_abs_diff": diff,
        "age_meta": age_meta,
        "as_of_decision_date": last["decision_date"],
        "event": last["event"],
        "gate": last["gate"],
        "universe": last["universe"],
        "age_buckets": last["age_buckets"],
        "current_book": last["book"],
        "book_detail": last["book_detail"],
        "cumulative": last["cumulative"],
        "last_realized": last["realized"],
        "backfill": backfill_stats(lines),
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
                         "binance_refresh_v1 (5 majors, mark-price "
                         "basis — keeps the BTC gate live) + "
                         "binance_xs_refresh_v1 + binance_xs2_refresh_v1")
    ap.add_argument("--ledger", type=Path, default=LEDGER)
    ap.add_argument("--snapshot", action="store_true",
                    help="print sleeve state as of the latest decision "
                         "date")
    args = ap.parse_args()
    if args.supplement is None:
        args.supplement = SUPPLEMENTS
    return cmd_snapshot(args) if args.snapshot else cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
