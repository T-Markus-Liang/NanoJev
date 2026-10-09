#!/usr/bin/env python3
"""T110: forward paper-trading ledger for the XS dfh sleeve (second sleeve).

Sleeve spec (T106 conditional spec-candidate, frozen here for forward paper):
  universe: the 10 USDT-M perps in data/perp_pit_xs_v1/records.jsonl
            (5 legacy + 5 T104-expansion), ranked by dfh20 =
            close / max(close of the strictly-prior 20 CONTIGUOUS daily
            bars) - 1. dfh20 is recomputed from the merged close series
            (cohort + refresh supplements), not read from the feature, so
            refreshed bars rank identically to cohort bars; recomputation is
            cross-checked against the record feature where present.
  book:     long top-2 / short bottom-2 by dfh20, equal weight (each leg 1/4
            of book, dollar-neutral, gross leverage 1.0), DAILY rebalance.
  gate:     a book is only opened when BTCUSDT close > SMA20 (mean of the 20
            consecutive BTC daily closes ending on the decision day,
            inclusive). Gate off or unevaluable -> flat book.
  universe  >=6 assets with a usable dfh20 on the decision day are required
            for a valid rank; below that the day is recorded as
            ``skipped_insufficient_universe`` and the book goes flat. This is
            the graceful-degradation path: while the 5 T104-expansion symbols
            lacked a refresh path (tail 2026-08-30), the legacy-only universe
            of 5 correctly degrades to skipped days instead of ranking a
            broken cross-section.
  costs:    5bps per unit of one-sided leg notional traded; cost_day_bps =
            5 * max(legs_added, legs_removed) / 4 vs the previous book (a
            full 4-leg unwind or rebuild costs 5bps; one leg swap 1.25bps).
  pnl:      the book decided at date t-1 is held t-1 -> t; its realized
            return is booked on the date-t line (long legs +, short legs -,
            close-to-close, equal leg weight). Leg closes missing on date t
            are filled with the asset's last prior close and flagged stale.
            day_net_bps = realized_gross(prev book) - cost(this rebalance);
            cumulative sums carried per line.

Ledger contract (results/forward_ledger_v2.jsonl):
  * append-only JSONL; ONE line per decision date (the union of all record
    dates across the 10 assets) — the sleeve is a single book, not
    per-asset state like v1
  * idempotent: re-running loads recorded decision_date keys and appends
    only dates not already in the ledger
  * content-light: dates, gate inputs (btc close/sma20), leg membership +
    dfh20, turnover/cost, realized leg returns, cumulative bps — never raw
    feature payloads
  * every line is fully determined at write time: the prior book's return
    realizes into the NEXT line, so no pending fills are ever rewritten

Supplements: --supplement may be repeated; records are deduped by id with the
cohort winning. Refreshed legacy bars carry mark_price (refresh_v1 record
shape) which is used as the close proxy; refreshed XS bars carry close.
Mixed basis post-refresh is quantified in
data/perp_pit_xs_v1/build_summary.json close_vs_mark_deviation
(~1bp median, <=26bps p95 on legacy symbols).

--snapshot prints the sleeve state as of the latest decision date: gate,
current book, cumulative bps.

Simulated paper accounting only — no orders, no broker, no account access,
not a tradability or profitability claim.
"""

import argparse
import datetime as dt
import json
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_xs_v1/records.jsonl"
SUPPLEMENTS = [ROOT / "data/binance_refresh_v1/records.jsonl",
               ROOT / "data/binance_xs_refresh_v1/records.jsonl"]
LEDGER = ROOT / "results/forward_ledger_v2.jsonl"
# T129 xs_v2 universe (30 assets, research/live_universe_v2.json): opt-in via
# --universe xs_v2; writes to a SEPARATE ledger so the v1 chain is untouched.
COHORT_XS2 = ROOT / "data/perp_pit_xs_v2/records.jsonl"
SUPPLEMENTS_XS2 = SUPPLEMENTS + [ROOT / "data/binance_xs2_refresh_v1/records.jsonl"]
LEDGER_XS2 = ROOT / "results/forward_ledger_v2_xs2.jsonl"

SCHEMA = "nanojev-forward-ledger-v2"
def sleeve_name(gate_mode):
    return f"xs_dfh20_top2_bot2_btc_{gate_mode}_gate_v1"

BTC_ASSET = "BTCUSDT-PERP"
DFH_LOOKBACK = 20        # strictly-prior contiguous bars for the high
SMA_LOOKBACK = 20        # BTC SMA window, inclusive of the decision bar
EDGE = 2                 # long top-2 / short bottom-2
N_LEG_SLOTS = 4          # 2 long + 2 short, each 1/4 of book
MIN_UNIVERSE = 6         # graceful-degradation floor for a valid rank
COST_BPS_PER_LEG = 5.0   # per unit of one-sided leg notional traded
DAY_MS = 86_400_000


def feat(record, name):
    f = record.get("features", {}).get(name)
    return f.get("value") if isinstance(f, dict) else None


def record_close(record):
    """XS price basis: klines close as mark proxy; refreshed legacy bars
    carry mark_price instead (refresh_v1 shape)."""
    close = feat(record, "close")
    if close is not None:
        return close, "close"
    mark = feat(record, "mark_price")
    if mark is not None:
        return mark, "mark_price"
    return None, None


def load_inputs(cohort_path, supplement_paths):
    """asset -> sorted [{date, close, basis, decision_ns, record_id}].
    Cohort + supplement records deduped by record id (cohort wins); per
    asset, a supplement date already present in the cohort is skipped."""
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
    series = {}
    for asset, days in by_asset.items():
        rows = [days[d] for d in sorted(days)]
        series[asset] = rows
    return series, extra


def dt_date(text):
    return dt.date.fromisoformat(text)


def add_dfh20(series):
    """Recompute dfh20 per bar in place: close / max(strictly-prior 20
    contiguous closes) - 1; None when the contiguous window is unavailable
    (same construction as build_xs_v1.py / financial_signal_xs_v1.py)."""
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


def crosscheck_dfh20(series, cohort_feature):
    """max abs diff between recomputed dfh20 and the cohort feature."""
    diff = 0.0
    n = 0
    for rows in series.values():
        for r in rows:
            f = cohort_feature.get(r["record_id"])
            if f is not None and r["dfh20"] is not None:
                diff = max(diff, abs(f - r["dfh20"]))
                n += 1
    return diff, n


def btc_gate_map(btc_rows):
    """date -> (btc_close, sma20 or None, ret20 or None). SMA20 = mean of the
    20 consecutive daily closes ending on that date (inclusive); ret20 =
    close/close[-20]-1 over the same contiguous window. None when <20
    contiguous bars or a nonpositive close is inside the window."""
    out = {}
    for i, r in enumerate(btc_rows):
        sma = ret = None
        if i >= SMA_LOOKBACK - 1 and r["close"] is not None:
            window = btc_rows[i - SMA_LOOKBACK + 1:i + 1]
            span = (dt_date(r["date"])
                    - dt_date(window[0]["date"])).days
            closes = [w["close"] for w in window]
            if (span == SMA_LOOKBACK - 1
                    and all(c is not None and c > 0 for c in closes)):
                sma = sum(closes) / len(closes)
        # ret20 follows the project convention (mom20): strictly-prior 20-bar
        # return close[i]/close[i-20]-1, requiring 21 contiguous bars.
        if i >= SMA_LOOKBACK and r["close"] is not None:
            back = btc_rows[i - SMA_LOOKBACK]
            span = (dt_date(r["date"]) - dt_date(back["date"])).days
            if (span == SMA_LOOKBACK and back["close"] is not None
                    and back["close"] > 0):
                ret = r["close"] / back["close"] - 1.0
        out[r["date"]] = (r["close"], sma, ret)
    return out


def replay(series, gate_mode="sma20"):
    """Deterministic full replay -> ordered ledger line dicts."""
    dates = sorted({r["date"] for rows in series.values() for r in rows})
    by_date = {asset: {r["date"]: r for r in rows}
               for asset, rows in series.items()}
    ns_by_date = defaultdict(int)
    for rows in series.values():
        for r in rows:
            ns_by_date[r["date"]] = max(ns_by_date[r["date"]],
                                        r["decision_ns"])
    btc_gate = btc_gate_map(series.get(BTC_ASSET, []))

    lines = []
    prev_book = []          # legs decided on the previous date: (side, asset)
    prev_date = None
    cum = {"gross": 0.0, "cost": 0.0, "net": 0.0}
    for date in dates:
        # universe: assets with a bar AND a usable dfh20 on this date
        ranked = [(r["dfh20"], asset) for asset, m in by_date.items()
                  if (r := m.get(date)) is not None
                  and r["dfh20"] is not None
                  and r["close"] is not None and r["close"] > 0]
        ranked.sort(key=lambda x: (x[0], x[1]))  # score asc, asset tiebreak

        btc_close, btc_sma, btc_ret = btc_gate.get(date, (None, None, None))
        if gate_mode == "ret20":
            gate_on = btc_ret is not None and btc_ret > 0
        else:
            gate_on = (btc_close is not None and btc_sma is not None
                       and btc_close > btc_sma)

        if len(ranked) < MIN_UNIVERSE:
            event = "skipped_insufficient_universe"
            book = []
        elif btc_close is None:
            event = "skipped_no_btc_bar"
            book = []
        elif not gate_on:
            event = "flat_gate_off"
            book = []
        else:
            event = "rebalance"
            book = ([("short", a) for _, a in ranked[:EDGE]]
                    + [("long", a) for _, a in ranked[-EDGE:]])

        # turnover vs the previous book: a changed leg slot trades its 1/4
        # notional once (one-sided); unwind/rebuild counts removed/added legs
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
                    # carry forward the asset's last close at/before date
                    prior = [r for r in series[asset] if r["date"] <= date]
                    exit_bar = prior[-1] if prior else None
                    stale = True
                ret_bps = None
                if (entry_bar and exit_bar and entry_bar["close"]
                        and exit_bar["close"]
                        and entry_bar["close"] > 0 and exit_bar["close"] > 0):
                    ret = exit_bar["close"] / entry_bar["close"] - 1.0
                    if side == "short":
                        ret = -ret
                    ret_bps = ret * 1e4
                    gross += ret_bps / N_LEG_SLOTS
                legs_out.append({
                    "asset": asset, "side": side,
                    "ret_bps": round(ret_bps, 2) if ret_bps is not None else None,
                    "entry_close": entry_bar["close"] if entry_bar else None,
                    "exit_close": exit_bar["close"] if exit_bar else None,
                    "exit_close_date": exit_bar["date"] if exit_bar else None,
                    "exit_basis": exit_bar["basis"] if exit_bar else None,
                    "stale_fill": stale})
            priced = sum(1 for l in legs_out if l["ret_bps"] is not None)
            if priced < len(prev_book):
                gross = None  # incomplete fill: do not book a partial book
        else:
            priced = 0
            gross = 0.0

        day_net = (gross - cost) if gross is not None else None
        if gross is not None:
            cum["gross"] += gross
        cum["cost"] += cost
        if day_net is not None:
            cum["net"] += day_net

        lines.append({
            "schema_version": SCHEMA,
            "sleeve": sleeve_name(gate_mode),
            "decision_date": date,
            "decision_ns": ns_by_date[date],
            "event": event,
            "universe": {"ranked": len(ranked),
                         "min_required": MIN_UNIVERSE,
                         "assets": [a for _, a in ranked]},
            "gate": {"mode": gate_mode,
                     "btc_close": _r(btc_close, 4),
                     "btc_sma20": _r(btc_sma, 4),
                     "btc_ret20": _r(btc_ret, 6),
                     "on": gate_on},
            "book": {"long": sorted(a for s, a in book if s == "long"),
                     "short": sorted(a for s, a in book if s == "short")},
            "book_detail": [{"asset": a, "side": s,
                             "dfh20": _r(_score(ranked, a), 6)}
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
                           "net_bps": _r(cum["net"], 4)},
        })
        prev_book = book
        prev_date = date
    return lines


def _score(ranked, asset):
    for sc, a in ranked:
        if a == asset:
            return sc
    return None


def _r(x, nd=4):
    return round(x, nd) if isinstance(x, (int, float)) else x


def replay_with_lines(args):
    series, extra = load_inputs(args.cohort, args.supplement)
    cohort_feature = {}
    for l in args.cohort.read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        f = feat(r, "dfh20")
        if f is not None:
            cohort_feature[r["id"]] = f
    add_dfh20(series)
    diff, n = crosscheck_dfh20(series, cohort_feature)
    lines = replay(series, gate_mode=args.gate)
    return lines, extra, diff, n


def cmd_run(args):
    lines, extra, diff, n_checked = replay_with_lines(args)
    seen = set()
    if args.ledger.exists():
        for l in args.ledger.read_text().splitlines():
            if not l.strip():
                continue
            seen.add(json.loads(l)["decision_date"])

    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    appended = skipped = rebalances = skips = 0
    with args.ledger.open("a", encoding="utf-8") as stream:
        for line in lines:
            if line["decision_date"] in seen:
                skipped += 1
                continue
            stream.write(json.dumps(line, sort_keys=True) + "\n")
            appended += 1
            rebalances += line["event"] == "rebalance"
            skips += line["event"].startswith("skipped")

    last = lines[-1] if lines else None
    print(json.dumps({
        "ledger": str(args.ledger),
        "sleeve": sleeve_name(args.gate),
        "universe": args.universe,
        "cohort": str(args.cohort),
        "supplement_records_merged": extra,
        "dfh20_recompute_max_abs_diff": diff,
        "dfh20_crosschecked_bars": n_checked,
        "decision_dates": len(lines),
        "dates_already_recorded": skipped,
        "dates_appended": appended,
        "rebalances_appended": rebalances,
        "skipped_days_appended": skips,
        "last_line": ({"decision_date": last["decision_date"],
                       "event": last["event"],
                       "gate_on": last["gate"]["on"],
                       "book": last["book"],
                       "cumulative": last["cumulative"]}
                      if last else None),
    }, indent=2, sort_keys=True))
    return 0


def cmd_snapshot(args):
    lines, extra, diff, n_checked = replay_with_lines(args)
    last = lines[-1] if lines else None
    if last is None:
        print(json.dumps({"schema_version": SCHEMA + "-snapshot",
                          "sleeve": sleeve_name(args.gate), "status": "empty"}))
        return 0
    n_reb = sum(1 for l in lines if l["event"] == "rebalance")
    n_skip = sum(1 for l in lines if l["event"].startswith("skipped"))
    n_flat = sum(1 for l in lines if l["event"] == "flat_gate_off")
    print(json.dumps({
        "schema_version": SCHEMA + "-snapshot",
        "sleeve": sleeve_name(args.gate),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "supplement_records_merged": extra,
        "dfh20_recompute_max_abs_diff": diff,
        "as_of_decision_date": last["decision_date"],
        "gate": last["gate"],
        "event": last["event"],
        "current_book": last["book"],
        "book_detail": last["book_detail"],
        "universe": last["universe"],
        "days": {"total": len(lines), "rebalance": n_reb,
                 "flat_gate_off": n_flat, "skipped": n_skip},
        "cumulative": last["cumulative"],
        "last_realized": last["realized"],
    }, indent=2, sort_keys=True))
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", type=Path, default=COHORT)
    ap.add_argument("--supplement", type=Path, action="append",
                    default=None,
                    help="refresh supplement JSONL (repeatable; merged and "
                         "deduped by id). Defaults: binance_refresh_v1 "
                         "(legacy mark-price bars) + binance_xs_refresh_v1 "
                         "(XS close bars)")
    ap.add_argument("--ledger", type=Path, default=LEDGER)
    ap.add_argument("--universe", choices=("xs_v1", "xs_v2"), default="xs_v1",
                    help="xs_v2 = 30-asset T129 cohort (perp_pit_xs_v2 + the "
                         "xs2 refresh supplement) writing to "
                         "results/forward_ledger_v2_xs2.jsonl; xs_v1 is the "
                         "unchanged default used by the daily chain")
    ap.add_argument("--gate", choices=("sma20", "ret20"), default="sma20",
                    help="BTC gate mode. sma20 (default) preserves the "
                         "historic ledger semantics. ret20 is the frozen "
                         "spec-v1 contract (btc_ret20 > 0); pair it with a "
                         "separate --ledger file to keep the two sleeves "
                         "independent.")
    ap.add_argument("--snapshot", action="store_true",
                    help="print sleeve state as of the latest decision date")
    args = ap.parse_args()
    if args.universe == "xs_v2":
        if args.cohort == COHORT:
            args.cohort = COHORT_XS2
        if args.supplement is None:
            args.supplement = SUPPLEMENTS_XS2
        if args.ledger == LEDGER:
            args.ledger = LEDGER_XS2
    if args.supplement is None:
        args.supplement = SUPPLEMENTS
    return cmd_snapshot(args) if args.snapshot else cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
