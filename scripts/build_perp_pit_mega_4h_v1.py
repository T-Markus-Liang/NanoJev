#!/usr/bin/env python3
"""Build a LEAN 4h PIT record set for the ~300-symbol MEGA cohort (T124).

Companion to ``build_perp_pit_mega_v1.py`` (T112, daily). Same source: the
imported sibling-project archive ``data/rc_futures_v1/`` — 301 symbol dirs
``<BASE>/`` holding ``<BASE>USDT_4h.csv`` (4h bars, columns
``timestamp,open,high,low,close,volume,quote_volume``) and, for 52 symbols,
``funding.csv`` (``timestamp,funding_rate``). Binance-futures-shaped layout
(USDT-M naming, 00/08/16 UTC funding grid for most symbols); upstream
provenance is a copy, not re-verified.

SCOPE (bounded per the task brief):
  * symbols: every dir whose 4h csv has >= MIN_BARS bars;
  * decision bars: bar OPEN inside [EVAL_START, EVAL_END) = 2023-01-01 ..
    2025-12-31 UTC — the overlap with funding coverage, skipping ancient
    thin years;
  * one record per (symbol, 4h bar) with a usable 4h forward label.

Records are intentionally FLAT and lean (task brief: only needed fields) —
this deviates from the nested features{}/label{} shape of the daily builds:

  asset    "BTCUSDT-PERP"
  ts       bar open epoch ms (UTC); decision instant = bar close
  close    bar close price (mark proxy)
  qv       bar quote volume
  fund     last funding rate settled at or before bar close (None if the
           symbol has no funding.csv or no settlement yet)
  hts      hours from bar close to the NEXT actual funding settlement
           (funding.csv schedule; ~0 for a bar closing on a settlement,
           ~4 mid-cell on the 8h grid; None for unfunded symbols)
  htsg     hours from bar close to the next settlement on the ASSUMED
           00/08/16 grid — defined for every symbol (0 or 4)
  m4       mom_4h  = log(close/close[i-1])   one 4h bar return
  m12      mom_12h = log(close/close[i-3])   three 4h bars
  rv       rv_24h  = stdev of the last 6 bar log-returns (i-5..i)
  fpos     share of the trailing 180 STRICTLY-PRIOR bars with fund > 0
           (funded symbols only, needs a full window; None otherwise)
  reg      funding regime from fpos: "pos" >0.6 / "neg" <0.4 / "mix"
  fwd4     10000*(close[i+1]/close[i]-1)  — contiguous next bar required
  fwd12    same over i+3 (None if not contiguous)
  fwd24    same over i+6 (None if not contiguous)

PIT discipline: every feature uses strictly-prior or current-bar data only;
``fund`` advances on settlements with calc_time <= bar close_time (the
settlement stamped 08:00:00.00x is NOT knowable at the 07:59:59.999 close);
``hts`` is a scheduled quantity, knowable in advance. Forward labels may
consume bars just past EVAL_END (a Dec-2025 decision's 24h label lands in
Jan-2026) — that is a label horizon, not look-ahead.

Scope and honesty rules enforced here:
* Real venue-shaped data, but a SECOND-HAND SNAPSHOT of history, not an
  as-of vintage and not a verified venue pull.
* Simulated/paper research only: no orders, no account, no broker.
* Labels are GROSS close-price moves — not returns net of costs, PnL, edge
  or tradable results (fees, spread, slippage, funding cashflows excluded).
* If no usable data exists this writes an honest "no data" summary and
  exits 0 rather than fabricating records.
"""
import argparse
import csv
import datetime as dt
import json
import math
import pathlib
import statistics

SCHEMA = "nanojev-perp-pit-mega-4h-v1"
VENUE = "binance_um"               # venue-shaped: USDT-M perp layout
BAR_MS = 4 * 3_600_000
HOUR_MS = 3_600_000
SETTLE_GRID_MS = 8 * HOUR_MS       # assumed 00/08/16 UTC settlement grid
MIN_BARS = 1000                    # inclusion floor per symbol (task brief)
EVAL_START_MS = int(dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)
                    .timestamp() * 1000)
EVAL_END_MS = int(dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
                  .timestamp() * 1000)
FUND_REGIME_WINDOW = 180           # trailing bars for fpos/reg
POS_HI, POS_LO = 0.6, 0.4          # T94 regime thresholds
EXPECTED_4H_HEADER = ("timestamp", "open", "high", "low", "close",
                      "volume", "quote_volume")


def parse_ts_ms(text):
    """'YYYY-MM-DD[ HH:MM:SS[.ffffff]]' -> UTC epoch ms."""
    d = dt.datetime.fromisoformat(text.strip()).replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp() * 1000)


def load_4h_bars(path):
    """(sorted ts_ms list, {ts: (close, qv)}, err) from a 4h csv."""
    bars = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None or tuple(h.strip() for h in header[:7]) \
                != EXPECTED_4H_HEADER:
            return [], {}, f"unexpected header: {header}"
        for row in reader:
            if len(row) < 7 or not row[0].strip():
                continue
            try:
                ts = parse_ts_ms(row[0])
                bars[ts] = (float(row[4]), float(row[6]))
            except (ValueError, IndexError):
                continue
    return sorted(bars), bars, None


def load_funding(path):
    """funding.csv -> sorted [calc_ms, rate] plus a cadence class.

    cadence = dominant gap in whole hours when >=90% of gaps match it
    ("8h"/"4h"/"1h"), else "mixed"; None when <2 rows."""
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None or header[0].strip() != "timestamp":
            return rows, "unparseable"
        for row in reader:
            if len(row) < 2 or not row[0].strip():
                continue
            try:
                rows.append((parse_ts_ms(row[0]), float(row[1])))
            except (ValueError, IndexError):
                continue
    rows.sort()
    if len(rows) < 2:
        return rows, None
    gaps = {}
    for i in range(1, len(rows)):
        g = round((rows[i][0] - rows[i - 1][0]) / HOUR_MS, 1)
        gaps[g] = gaps.get(g, 0) + 1
    mode, cnt = max(gaps.items(), key=lambda kv: kv[1])
    frac = cnt / (len(rows) - 1)
    if frac >= 0.9 and mode in (1.0, 4.0, 8.0):
        cadence = f"{int(mode)}h"
    elif frac >= 0.9:
        cadence = f"{mode}h"
    else:
        cadence = "mixed"
    return rows, cadence


def build_records(base, ts_list, bars, funding):
    """Emit lean records for decision bars inside the eval window."""
    symbol = f"{base}USDT-PERP"
    records = []
    skipped = {"nonpositive_price": 0, "zero_volume": 0,
               "no_fwd4_label": 0}
    closes = [bars[t][0] for t in ts_list]
    n = len(ts_list)
    fcursor = 0
    last_funding = None
    pos_run = 0                   # trailing-180 count of fund>0 (bars)
    hist_fund_pos = []            # 1/0 per bar for the sliding window
    funded_file = bool(funding)
    for i in range(n):
        ts = ts_list[i]
        close_time = ts + BAR_MS - 1          # Binance close_time convention
        # Advance settled-funding cursor only up to the decision instant.
        while fcursor < len(funding) and funding[fcursor][0] <= close_time:
            last_funding = funding[fcursor]
            fcursor += 1
        fund = last_funding[1] if last_funding is not None else None
        # next ACTUAL settlement is a scheduled quantity (not look-ahead)
        if fcursor < len(funding):
            hts = round((funding[fcursor][0] - close_time) / HOUR_MS, 3)
        else:
            hts = None
        # assumed 00/08/16 grid: distance from the bar-close instant
        htsg = (SETTLE_GRID_MS - ((ts + BAR_MS) % SETTLE_GRID_MS)) \
            % SETTLE_GRID_MS
        htsg = round(htsg / HOUR_MS, 3)

        # sliding trailing-180 funding-positive share over STRICTLY-PRIOR
        # bars i-180..i-1: fold in bar i-1's flag, drop bar i-181's
        if i > 0:
            pos_run += hist_fund_pos[i - 1]
            if i - 1 - FUND_REGIME_WINDOW >= 0:
                pos_run -= hist_fund_pos[i - 1 - FUND_REGIME_WINDOW]
        if funded_file and i >= FUND_REGIME_WINDOW:
            fpos = round(pos_run / FUND_REGIME_WINDOW, 4)
            reg = "pos" if fpos > POS_HI else "neg" if fpos < POS_LO \
                else "mix"
        else:
            fpos = reg = None
        hist_fund_pos.append(1 if (fund is not None and fund > 0) else 0)

        if ts < EVAL_START_MS or ts >= EVAL_END_MS:
            continue

        close, qv = bars[ts]
        if close <= 0:
            skipped["nonpositive_price"] += 1
            continue
        if qv <= 0:
            skipped["zero_volume"] += 1
            continue

        m4 = m12 = rv = None
        if i >= 1 and ts - ts_list[i - 1] == BAR_MS \
                and closes[i - 1] > 0:
            m4 = round(math.log(close / closes[i - 1]), 8)
        if i >= 3 and ts - ts_list[i - 3] == 3 * BAR_MS \
                and closes[i - 3] > 0:
            m12 = round(math.log(close / closes[i - 3]), 8)
        if i >= 6 and ts - ts_list[i - 6] == 6 * BAR_MS:
            window = closes[i - 6:i + 1]
            if all(c > 0 for c in window):
                rets = [math.log(window[k + 1] / window[k])
                        for k in range(6)]
                rv = round(statistics.stdev(rets), 8)

        fwd4 = fwd12 = fwd24 = None
        if i + 1 < n and ts_list[i + 1] - ts == BAR_MS \
                and closes[i + 1] > 0:
            fwd4 = round(10000.0 * (closes[i + 1] / close - 1.0), 4)
        if i + 3 < n and ts_list[i + 3] - ts == 3 * BAR_MS \
                and closes[i + 3] > 0:
            fwd12 = round(10000.0 * (closes[i + 3] / close - 1.0), 4)
        if i + 6 < n and ts_list[i + 6] - ts == 6 * BAR_MS \
                and closes[i + 6] > 0:
            fwd24 = round(10000.0 * (closes[i + 6] / close - 1.0), 4)
        if fwd4 is None:
            skipped["no_fwd4_label"] += 1
            continue

        records.append({
            "asset": symbol, "ts": ts, "close": close, "qv": qv,
            "fund": fund, "hts": hts, "htsg": htsg,
            "m4": m4, "m12": m12, "rv": rv,
            "fpos": fpos, "reg": reg,
            "fwd4": fwd4, "fwd12": fwd12, "fwd24": fwd24})
    return records, skipped


def no_data_summary(output):
    summary = {
        "schema_version": "nanojev-perp-pit-mega-4h-build-v1",
        "status": "NO DATA: data/rc_futures_v1 produced no usable 4h files; "
                  "records were not built and none were fabricated",
        "records": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output.parent / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True)
                            + "\n", encoding="utf-8")
    return summary_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=pathlib.Path,
                        default=pathlib.Path("data/rc_futures_v1"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path(
        "data/perp_pit_mega_4h_v1/records.jsonl"))
    parser.add_argument("--min-bars", type=int, default=MIN_BARS)
    parser.add_argument("--symbols", default=None,
                        help="optional comma-separated base-asset filter "
                             "(default: every dir under the source root)")
    args = parser.parse_args()

    if not args.source_root.is_dir():
        summary_path = no_data_summary(args.output)
        print(json.dumps({"records": 0, "status": "no_data",
                          "summary": str(summary_path)}, sort_keys=True))
        return 0

    universe, inventory = {}, {"dirs": 0, "with_4h": 0, "with_funding": 0,
                             "dirs_without_4h": []}
    for d in sorted(args.source_root.iterdir()):
        if not d.is_dir():
            continue
        inventory["dirs"] += 1
        base = d.name
        f4h = d / f"{base}USDT_4h.csv"
        if (d / "funding.csv").is_file():
            inventory["with_funding"] += 1
        if f4h.is_file():
            inventory["with_4h"] += 1
            universe[base] = f4h
        else:
            inventory["dirs_without_4h"].append(base)
    if args.symbols:
        wanted = {s.strip().upper() for s in args.symbols.split(",")
                  if s.strip()}
        universe = {b: p for b, p in universe.items() if b in wanted}

    all_records, per_symbol, excluded = [], {}, {}
    for base, path in sorted(universe.items()):
        ts_list, bars, err = load_4h_bars(path)
        if err or not ts_list:
            excluded[base] = {"reason": err or "no parseable 4h bars",
                              "bars_4h": 0}
            continue
        if len(ts_list) < args.min_bars:
            excluded[base] = {
                "reason": f"insufficient_history: {len(ts_list)} 4h bars "
                          f"< min_bars={args.min_bars}",
                "bars_4h": len(ts_list),
                "first_bar": dt.datetime.fromtimestamp(
                    ts_list[0] / 1000, dt.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M"),
                "last_bar": dt.datetime.fromtimestamp(
                    ts_list[-1] / 1000, dt.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M")}
            continue
        funding_path = args.source_root / base / "funding.csv"
        funding, cadence = (load_funding(funding_path)
                            if funding_path.is_file() else ([], None))
        records, skipped = build_records(base, ts_list, bars, funding)
        per_symbol[base] = {
            "records": len(records),
            "skipped": skipped,
            "coverage": {
                "first_bar": dt.datetime.fromtimestamp(
                    ts_list[0] / 1000, dt.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M"),
                "last_bar": dt.datetime.fromtimestamp(
                    ts_list[-1] / 1000, dt.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M"),
                "bars_4h": len(ts_list),
                "funding_rows": len(funding),
                "funding_cadence": cadence,
                "has_funding_file": funding_path.is_file()},
            "first": dt.datetime.fromtimestamp(
                records[0]["ts"] / 1000, dt.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M") if records else None,
            "last": dt.datetime.fromtimestamp(
                records[-1]["ts"] / 1000, dt.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M") if records else None}
        all_records.extend(records)

    if not all_records:
        summary_path = no_data_summary(args.output)
        print(json.dumps({"records": 0, "status": "no_data",
                          "summary": str(summary_path)}, sort_keys=True))
        return 0

    all_records.sort(key=lambda r: (r["ts"], r["asset"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in all_records:
            stream.write(json.dumps(record, sort_keys=True,
                                    separators=(",", ":")) + "\n")

    included = sorted(per_symbol)
    funded = sorted(b for b in included
                    if per_symbol[b]["coverage"]["has_funding_file"])
    cadence_classes = {}
    for b in funded:
        c = per_symbol[b]["coverage"]["funding_cadence"]
        cadence_classes.setdefault(c, []).append(b)
    summary = {
        "schema_version": "nanojev-perp-pit-mega-4h-build-v1",
        "record_schema": SCHEMA,
        "status": "MEGA 4h cohort (T124); every rc_futures_v1 dir with "
                  ">= min_bars 4h bars; decision bars restricted to "
                  "2023-01-01..2025-12-31 UTC",
        "source": "data/rc_futures_v1 — imported sibling-project archive "
                  "copy of venue-shaped USDT-M perp CSVs (4h bars for all "
                  "dirs; funding.csv for a subset); NOT a fresh venue fetch",
        "venue": VENUE,
        "granularity": "4h",
        "eval_window_utc": ["2023-01-01T00:00", "2025-12-31T23:59"],
        "min_bars_inclusion_floor": args.min_bars,
        "fields": ["asset", "ts", "close", "qv", "fund", "hts", "htsg",
                   "m4", "m12", "rv", "fpos", "reg", "fwd4", "fwd12",
                   "fwd24"],
        "field_notes": {
            "ts": "bar OPEN epoch ms UTC; decision instant = bar close",
            "fund": "last funding rate with calc_time <= bar close_time "
                    "(settlement stamped 08:00:00.00x is NOT knowable at "
                    "the 07:59:59.999 close); null unfunded/pre-first",
            "hts": "hours to next ACTUAL settlement from funding.csv "
                   "schedule (null for unfunded symbols)",
            "htsg": "hours to next settlement on the assumed 00/08/16 UTC "
                    "grid — 0 for bars closing on the grid, 4 mid-cell; "
                    "defined for every symbol",
            "m4": "log(close/close[i-1]) — one 4h bar return",
            "m12": "log(close/close[i-3]) — 12h momentum",
            "rv": "stdev of the last 6 bar log-returns (~24h)",
            "fpos": "share of trailing-180 strictly-prior bars with "
                    "fund>0 (funded symbols, full window only)",
            "reg": "pos if fpos>0.6, neg if fpos<0.4, else mix (T94 "
                   "thresholds)",
            "fwd4/fwd12/fwd24": "gross forward close return bps over "
                                "1/3/6 contiguous bars; fwd24 may be null "
                                "near series end"},
        "records": len(all_records),
        "symbols_included": included,
        "symbols_included_count": len(included),
        "symbols_excluded": excluded,
        "symbols_excluded_count": len(excluded),
        "symbols_with_funding": funded,
        "symbols_with_funding_count": len(funded),
        "funding_cadence_classes": {k: sorted(v) for k, v in
                                    sorted(cadence_classes.items())},
        "per_symbol": per_symbol,
        "file_inventory": inventory,
        "honesty": {
            "not_a_return": "labels are gross close-price moves; fees, "
                            "spread, slippage, funding cashflows and "
                            "leverage are excluded by construction",
            "not_an_asof_vintage": "second-hand archive copy "
                                   "(rc_futures_v1), not a verified venue "
                                   "pull or as-of vintage",
            "not_live": "no orders, no account, no broker, no trading API",
            "no_profitability_claim": True,
        },
    }
    summary_path = args.output.parent / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True)
                            + "\n", encoding="utf-8")
    print(json.dumps({"records": summary["records"],
                      "symbols": summary["symbols_included_count"],
                      "funded_symbols": summary[
                          "symbols_with_funding_count"],
                      "output": str(args.output),
                      "summary": str(summary_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
