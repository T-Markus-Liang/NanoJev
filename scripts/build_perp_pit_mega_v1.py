#!/usr/bin/env python3
"""Build daily PIT-lite records for the 301-symbol MEGA cross-sectional cohort (T112).

Companion to ``build_xs_v1.py`` (T104, 10 assets). The source is NOT a fresh venue
fetch: it is the imported sibling-project archive ``data/rc_futures_v1/`` — 301
symbol directories ``<BASE>/`` holding ``<BASE>USDT_1d.csv`` (daily bars, columns
``timestamp,open,high,low,close,volume,quote_volume[,taker_buy_quote_volume]``)
and, for ~52 symbols, ``funding.csv`` (``timestamp,funding_rate``, ~3
settlements/day at 00/08/16 UTC -> 8h interval). The funding-settlement layout,
USDT-M naming and metrics columns are Binance-futures shaped; upstream
provenance is a copy, not re-verified — recorded under ``provenance`` below.

One daily record per (symbol, day) is emitted to
``data/perp_pit_mega_v1/records.jsonl`` + ``build_summary.json``. Record-shape
conventions follow ``build_perp_pit_v1.py`` / ``build_xs_v1.py``: a ``features``
dict whose values carry ``value`` / ``event_ns`` / ``available_ns`` /
``fit_cutoff_ns`` / ``source_id`` / ``version``, ``decision_ns`` at bar close +
1 ns, and gross forward close-return labels (1d and 5d, contiguous bars only).

SURVIVORSHIP PROFILE (the point of this cohort): the archive appears to include
delisted/renamed contracts — a symbol whose daily series stops before the
archive tail (2025-12) probably stopped trading. Each record carries
``listed``/``last_bar_date`` metadata and the summary counts early-stoppers
explicitly. ``listed`` is a PROXY: series reaches the archive end month; some
early stops are ticker renames (e.g. MATIC->POL, RNDR->RENDER, GAL->GALXE)
rather than true delistings — both reduce survivorship bias but rename pairs
can double-count an economic asset.

Features emitted (values are null rather than record-dropping when the trailing
contiguous window is unavailable — same pilot-lite rule):
  close, quote_volume, last_funding_rate, funding_interval_hours,
  dfh20  = close / max(close[i-20:i]) - 1   (strictly prior 20 contiguous bars),
  mom20  = close / close[i-20] - 1          (strictly prior 20-bar return),
  vol20  = stdev of the 20 trailing daily log-returns (same contiguous window).

Deviations from ``build_xs_v1.py``:
  * funding is OPTIONAL: the 249 no-funding symbols still emit records with
    ``last_funding_rate``/``funding_interval_hours`` null (xs_v1 skipped
    pre-first-settlement days; here nothing is dropped for missing funding).
  * ``funding_interval_hours`` is INFERRED per settlement (rounded gap to the
    previous settlement, hours) instead of read from an archive field.
  * records carry ``listed`` (bool) and ``last_bar_date`` per symbol.
  * inclusion floor: a symbol needs >= MIN_BARS daily bars to enter the cohort;
    excluded symbols are listed honestly in the summary.

Scope and honesty rules enforced here:
* Real venue-shaped data, but a SECOND-HAND SNAPSHOT of history, not an as-of
  vintage and not a verified venue pull.
* Simulated/paper research only: no orders, no account, no broker.
* Labels are GROSS close-price moves. They are NOT returns net of costs, PnL,
  edge or tradable results: fees, spread, slippage, funding cashflows and
  leverage are excluded.
* If no usable data exists, this script writes an honest "no data"
  build_summary.json and exits 0 rather than fabricating records.
"""
import argparse
import csv
import datetime as dt
import json
import math
import pathlib
import statistics

SCHEMA = "nanojev-financial-pit-mega-v1"
VENUE = "binance_um"          # venue-shaped: USDT-M perp layout, 8h funding
MS = 1_000_000
DAY_MS = 86_400_000
HOUR_MS = 3_600_000
WINDOW = 20                                # dfh20 / mom20 / vol20 trailing bars
MIN_BARS = 200                             # inclusion floor per symbol
LISTED_TAIL_DAYS = 31                      # last_bar within this of archive max -> "listed"
HORIZON_NS = 86_400_000_000_000            # 1 day, from the frozen definition
LABEL_LAG_NS = 604_800_000_000_000         # 7 days, from the protocol
THRESHOLD_BPS = 25.0
EVENT_NAME = "perp_forward_close_return_up_25bps_1d_gross"
FEATURES = ("close", "quote_volume", "last_funding_rate", "funding_interval_hours",
            "dfh20", "mom20", "vol20")
EXPECTED_1D_HEADER = ("timestamp", "open", "high", "low", "close",
                      "volume", "quote_volume")  # optional 8th col ignored


def parse_day_ms(text):
    """'YYYY-MM-DD' -> UTC epoch ms at day open."""
    d = dt.datetime.strptime(text.strip()[:10], "%Y-%m-%d").replace(
        tzinfo=dt.timezone.utc)
    return int(d.timestamp() * 1000)


def parse_ts_ms(text):
    """'YYYY-MM-DD[ HH:MM:SS[.ffffff]]' -> UTC epoch ms."""
    d = dt.datetime.fromisoformat(text.strip()).replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp() * 1000)


def load_daily_bars(path):
    """{day_open_ms: {open,high,low,close,volume,quote_volume}} from a 1d csv."""
    bars = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None or tuple(h.strip() for h in header[:7]) != EXPECTED_1D_HEADER:
            return bars, f"unexpected header: {header}"
        for row in reader:
            if len(row) < 7 or not row[0].strip():
                continue
            try:
                day = parse_day_ms(row[0])
                bars[day] = {"open": float(row[1]), "high": float(row[2]),
                             "low": float(row[3]), "close": float(row[4]),
                             "volume": float(row[5]),
                             "quote_volume": float(row[6])}
            except (ValueError, IndexError):
                continue
    return bars, None


def load_funding(path):
    """funding.csv -> sorted [{calc_ms, rate, interval_hours}]; interval_hours is
    inferred as the rounded gap to the previous settlement (null on the first)."""
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None or header[0].strip() != "timestamp":
            return rows
        for row in reader:
            if len(row) < 2 or not row[0].strip():
                continue
            try:
                rows.append({"calc_ms": parse_ts_ms(row[0]),
                             "rate": float(row[1]), "interval_hours": None})
            except (ValueError, IndexError):
                continue
    rows.sort(key=lambda r: r["calc_ms"])
    for i, row in enumerate(rows):
        if i:
            gap_h = (row["calc_ms"] - rows[i - 1]["calc_ms"]) / HOUR_MS
            if 0 < gap_h <= 48:
                row["interval_hours"] = int(round(gap_h))
    return rows


def build_records(base, bars, funding, last_bar_ms, archive_last_ms):
    days = sorted(bars)
    closes = [bars[day]["close"] for day in days]
    records, skipped = [], {"missing_bar": 0, "zero_volume": 0,
                            "nonpositive_price": 0,
                            "insufficient_feature_window": 0, "no_label_bar": 0,
                            "noncontiguous_label_window": 0}
    symbol = f"{base}USDT"
    last_bar_date = dt.datetime.fromtimestamp(
        last_bar_ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d")
    listed = last_bar_ms >= archive_last_ms - LISTED_TAIL_DAYS * DAY_MS
    funding_cursor = 0
    last_funding = None
    for index, day in enumerate(days):
        day_text = dt.datetime.fromtimestamp(day / 1000, dt.timezone.utc).strftime(
            "%Y-%m-%d")
        close_time = day + DAY_MS - 1     # Binance close_time convention: 23:59:59.999
        # Advance the settled-funding cursor only up to the decision instant.
        while funding_cursor < len(funding) \
                and funding[funding_cursor]["calc_ms"] <= close_time:
            last_funding = funding[funding_cursor]
            funding_cursor += 1
        if index + 1 >= len(days):
            skipped["no_label_bar"] += 1
            continue
        if days[index + 1] - day != DAY_MS:
            # a gap in the archive must not silently become a "1d forward" label
            skipped["noncontiguous_label_window"] += 1
            continue

        close = bars[day]["close"]
        volume = bars[day]["volume"]
        if close <= 0:
            skipped["nonpositive_price"] += 1
            continue
        if volume <= 0:
            skipped["zero_volume"] += 1
            continue

        # Trailing-20 contiguous-bar features; null (not dropped) when unavailable.
        contiguous20 = index >= WINDOW and day - days[index - WINDOW] == WINDOW * DAY_MS
        if contiguous20:
            window = closes[index - WINDOW:index + 1]          # 21 closes
            prior = closes[index - WINDOW:index]               # 20 strictly prior
            dfh20 = close / max(prior) - 1.0 if max(prior) > 0 else None
            mom20 = close / closes[index - WINDOW] - 1.0 \
                if closes[index - WINDOW] > 0 else None
            returns = [math.log(window[i + 1] / window[i])
                       for i in range(len(window) - 1)
                       if window[i] > 0 and window[i + 1] > 0]
            vol20 = statistics.stdev(returns) if len(returns) >= WINDOW else None
        else:
            dfh20 = mom20 = vol20 = None
            skipped["insufficient_feature_window"] += 1

        decision_ns = close_time * MS + 1
        event_ns = close_time * MS
        available_ns = event_ns + 1
        fit_cutoff = event_ns  # trailing window closes on the decision bar

        def feature(value, source_id, version, event=event_ns,
                    avail=available_ns, fit=0):
            return {"value": value, "event_ns": event, "available_ns": avail,
                    "fit_cutoff_ns": fit, "source_id": source_id,
                    "version": version}

        if last_funding is not None:
            f_event = last_funding["calc_ms"] * MS
            f_rate, f_hours = (last_funding["rate"],
                               last_funding["interval_hours"])
        else:
            f_event, f_rate, f_hours = event_ns, None, None
        features = {
            "close": feature(close, "venue_perp_klines",
                             "perp-1d-close-as-mark-proxy-v1"),
            "quote_volume": feature(bars[day]["quote_volume"], "venue_perp_klines",
                                    "perp-1d-quote-volume-v1"),
            "last_funding_rate": feature(f_rate, "venue_funding_rate_history",
                                         "perp-funding-rate-settled-v1",
                                         event=f_event, avail=f_event),
            "funding_interval_hours": feature(
                f_hours, "venue_instrument_or_funding_config",
                "perp-funding-interval-hours-inferred-v1",
                event=f_event, avail=f_event),
            "dfh20": feature(dfh20, "venue_perp_klines",
                             "perp-1d-close-dist-from-20bar-high-v1",
                             fit=fit_cutoff),
            "mom20": feature(mom20, "venue_perp_klines",
                             "perp-1d-close-20bar-momentum-v1", fit=fit_cutoff),
            "vol20": feature(vol20, "venue_perp_klines",
                             "perp-1d-close-realized-vol-20bar-v1",
                             fit=fit_cutoff),
        }
        exit_close = bars[days[index + 1]]["close"]
        if exit_close <= 0:
            skipped["nonpositive_price"] += 1
            continue
        forward_bps = 10000.0 * ((exit_close / close) - 1.0)
        forward_5d_bps = None
        if index + 5 < len(days) and days[index + 5] - day == 5 * DAY_MS \
                and bars[days[index + 5]]["close"] > 0:
            forward_5d_bps = 10000.0 * (
                (bars[days[index + 5]]["close"] / close) - 1.0)
        end_ns = decision_ns + HORIZON_NS

        records.append({
            "schema_version": SCHEMA,
            "id": f"{VENUE}:{symbol}:{day_text}",
            "asset_id": f"{symbol}-PERP",
            "venue": VENUE,
            "decision_ns": decision_ns,
            "universe_available_ns": event_ns,
            "listed": listed,
            "last_bar_date": last_bar_date,
            "features": features,
            "label": {"event": EVENT_NAME,
                      "end_ns": end_ns, "available_ns": end_ns + LABEL_LAG_NS,
                      "outcome": bool(forward_bps >= THRESHOLD_BPS),
                      "forward_return_bps": forward_bps,
                      "forward_return_5d_bps": forward_5d_bps},
        })
    return records, skipped


def no_data_summary(output):
    summary = {
        "schema_version": "nanojev-perp-pit-mega-build-v1",
        "status": "NO DATA: data/rc_futures_v1 produced no usable files; "
                  "records were not built and none were fabricated",
        "records": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output.parent / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    return summary_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=pathlib.Path,
                        default=pathlib.Path("data/rc_futures_v1"))
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("data/perp_pit_mega_v1/records.jsonl"))
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

    # Universe: every dir holding a <BASE>USDT_1d.csv.
    universe = {}
    inventory = {"dirs": 0, "with_1d": 0, "with_4h": 0, "with_1h": 0,
                 "with_funding": 0, "with_metrics": 0,
                 "dirs_without_1d": []}
    for d in sorted(args.source_root.iterdir()):
        if not d.is_dir():
            continue
        inventory["dirs"] += 1
        base = d.name
        f1d = d / f"{base}USDT_1d.csv"
        files = {p.name for p in d.iterdir()}
        for kind, flag in (("4h", "with_4h"), ("1h", "with_1h")):
            if f"{base}USDT_{kind}.csv" in files:
                inventory[flag] += 1
        if "funding.csv" in files:
            inventory["with_funding"] += 1
        if "metrics.csv" in files:
            inventory["with_metrics"] += 1
        if f1d.is_file():
            inventory["with_1d"] += 1
            universe[base] = f1d
        else:
            inventory["dirs_without_1d"].append(base)

    if args.symbols:
        wanted = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        universe = {b: p for b, p in universe.items() if b in wanted}

    # First pass: bar spans (needed for the archive-tail 'listed' proxy).
    spans, excluded = {}, {}
    for base, path in universe.items():
        bars, err = load_daily_bars(path)
        if err or not bars:
            excluded[base] = {"reason": err or "no parseable daily bars",
                              "daily_bars": 0}
            continue
        days = sorted(bars)
        spans[base] = {"bars": bars, "first_ms": days[0], "last_ms": days[-1],
                       "n": len(days)}
    archive_last_ms = max((s["last_ms"] for s in spans.values()), default=0)
    archive_last_date = (
        dt.datetime.fromtimestamp(archive_last_ms / 1000, dt.timezone.utc)
        .strftime("%Y-%m-%d") if archive_last_ms else None)

    all_records, per_symbol = [], {}
    for base, span in sorted(spans.items()):
        if span["n"] < args.min_bars:
            excluded[base] = {
                "reason": f"insufficient_history: {span['n']} daily bars "
                          f"< min_bars={args.min_bars}",
                "daily_bars": span["n"],
                "first_bar": dt.datetime.fromtimestamp(
                    span["first_ms"] / 1000, dt.timezone.utc).strftime("%Y-%m-%d"),
                "last_bar": dt.datetime.fromtimestamp(
                    span["last_ms"] / 1000, dt.timezone.utc).strftime("%Y-%m-%d")}
            continue
        funding_path = args.source_root / base / "funding.csv"
        funding = load_funding(funding_path) if funding_path.is_file() else []
        records, skipped = build_records(base, span["bars"], funding,
                                         span["last_ms"], archive_last_ms)
        positives = sum(1 for r in records if r["label"]["outcome"])
        listed = span["last_ms"] >= archive_last_ms - LISTED_TAIL_DAYS * DAY_MS
        per_symbol[base] = {
            "records": len(records), "positives": positives,
            "positive_rate": (positives / len(records)) if records else None,
            "listed": listed,
            "skipped": skipped,
            "coverage": {
                "first_bar": dt.datetime.fromtimestamp(
                    span["first_ms"] / 1000, dt.timezone.utc).strftime("%Y-%m-%d"),
                "last_bar": dt.datetime.fromtimestamp(
                    span["last_ms"] / 1000, dt.timezone.utc).strftime("%Y-%m-%d"),
                "daily_bars": span["n"],
                "funding_rows": len(funding),
                "has_funding_file": funding_path.is_file()},
            "first": records[0]["id"].rsplit(":", 1)[-1] if records else None,
            "last": records[-1]["id"].rsplit(":", 1)[-1] if records else None}
        all_records.extend(records)

    if not all_records:
        summary_path = no_data_summary(args.output)
        print(json.dumps({"records": 0, "status": "no_data",
                          "summary": str(summary_path)}, sort_keys=True))
        return 0

    all_records.sort(key=lambda row: (row["decision_ns"], row["id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in all_records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    positives = sum(1 for r in all_records if r["label"]["outcome"])
    included = sorted(per_symbol)
    early_stopped = {b: per_symbol[b]["coverage"]["last_bar"]
                     for b in included if not per_symbol[b]["listed"]}
    funded = sorted(b for b in included
                    if per_symbol[b]["coverage"]["has_funding_file"])
    summary = {
        "schema_version": "nanojev-perp-pit-mega-build-v1",
        "status": "MEGA cross-sectional cohort (T112); every rc_futures_v1 dir "
                  "with >= min_bars daily bars",
        "source": "data/rc_futures_v1 — imported sibling-project archive copy "
                  "of venue-shaped USDT-M perp CSVs (1d bars for all dirs; "
                  "funding.csv for a subset); NOT a fresh venue fetch",
        "venue": VENUE,
        "is_pilot": False,
        "granularity": "1d",
        "features": list(FEATURES),
        "event": EVENT_NAME,
        "threshold_bps": THRESHOLD_BPS, "horizon_ns": HORIZON_NS,
        "label_availability_lag_ns": LABEL_LAG_NS,
        "min_bars_inclusion_floor": args.min_bars,
        "records": len(all_records), "positives": positives,
        "positive_rate": (positives / len(all_records)) if all_records else None,
        "archive_last_bar": archive_last_date,
        "symbols_included": included,
        "symbols_included_count": len(included),
        "symbols_excluded": excluded,
        "symbols_excluded_count": len(excluded),
        "per_symbol": per_symbol,
        "file_inventory": inventory,
        "funding": {
            "symbols_with_funding_file": funded,
            "symbols_with_funding_file_count": len(funded),
            "convention": "last_funding_rate = last settlement with "
                          "calc_time <= day close (8h cadence, ~3 rows/day); "
                          "funding_interval_hours inferred from settlement "
                          "gaps; both null on the ~249 no-funding symbols and "
                          "on pre-first-settlement days",
        },
        "delisting_proxy": {
            "definition": "listed = series reaches within "
                          f"{LISTED_TAIL_DAYS}d of the archive tail "
                          f"({archive_last_date}); earlier stop = probably "
                          "delisted OR renamed",
            "archive_last_bar": archive_last_date,
            "early_stopped_count": len(early_stopped),
            "early_stopped": early_stopped,
            "note": "some early stops are ticker renames rather than true "
                    "delistings (e.g. MATIC->POL, RNDR->RENDER, GAL->GALXE); "
                    "both cases reduce single-sided survivorship bias but "
                    "rename pairs can double-count one economic asset",
        },
        "deviations_from_xs_v1_builder": {
            "funding_optional": "no-funding symbols emit records with null "
                "funding features instead of being dropped (xs_v1 skipped "
                "pre-settlement days)",
            "interval_inferred": "funding_interval_hours inferred from "
                "settlement gaps (archive field absent in csv source)",
            "record_metadata": "each record carries listed + last_bar_date",
            "close_as_mark": "same convention: csv close is the mark proxy for "
                "every symbol; no markPriceKlines exist in this source",
        },
        "honesty": {
            "not_a_return": "labels are gross close-price moves; fees, spread, "
                            "slippage, funding cashflows and leverage are "
                            "excluded by construction",
            "not_an_asof_vintage": "second-hand snapshot copy; not evidence of "
                                   "what was knowable at the time; upstream "
                                   "fetch provenance not re-verified",
            "not_live": "no orders, no account, no broker, no trading API",
            "no_profitability_claim": True,
        },
    }
    summary_path = args.output.parent / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(json.dumps({"records": summary["records"],
                      "positives": summary["positives"],
                      "positive_rate": (round(summary["positive_rate"], 4)
                                        if summary["positive_rate"] is not None
                                        else None),
                      "symbols_included": len(included),
                      "symbols_excluded": len(excluded),
                      "early_stopped": len(early_stopped),
                      "with_funding": len(funded),
                      "output": str(args.output),
                      "summary": str(summary_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
