#!/usr/bin/env python3
"""Build daily PIT-lite records for the cross-sectional pilot (T102).

Companion to ``build_perp_pit_v1.py``. It merges two bounded archive slices:
  * the approved 5-symbol slice in ``data/binance_vision_v1/`` (full history), and
  * the 5-symbol x 3-month pilot slice in ``data/binance_xs_pilot_v1/``
    (``fetch_binance_xs_pilot_v1.py``),
and emits daily records whose decision day falls inside the pilot months
(2026-06-01 .. 2026-08-31 by default) for all 10 symbols.

Record-shape conventions follow ``build_perp_pit_v1.py`` / the intraday builder:
a ``features`` dict whose values carry ``value`` / ``event_ns`` / ``available_ns`` /
``fit_cutoff_ns`` / ``source_id`` / ``version``, ``decision_ns`` at bar close + 1 ns,
and a gross forward mark-price label. Two deliberate PILOT-LITE deviations, both
recorded in build_summary.json:
  * ``realized_vol_24bar`` is emitted with ``value: null`` when the trailing
    24-bar contiguous window is not available (new symbols only have pilot-month
    history) instead of dropping the record — the XS arm does not use this
    feature and dropping the record would remove ~25% of new-symbol days.
  * The label additionally carries ``forward_return_bps`` (1d) and
    ``forward_return_5d_bps`` (5d, null when no contiguous 5th forward bar exists)
    so downstream cross-sectional arms do not need to re-read the archives.

Scope and honesty rules enforced here:
* Real venue data, but the archive is a CURRENT SNAPSHOT of history, not an as-of vintage.
* Simulated/paper research only: no orders, no account, no broker.
* Labels are GROSS mark-price moves. They are NOT returns net of costs, PnL, edge or
  tradable results: fees, spread, slippage, funding cashflows and leverage are excluded.
* If no usable data exists, this script writes an honest "no data" build_summary.json
  and exits 0 rather than fabricating records.
"""
import argparse
import datetime as dt
import json
import math
import pathlib
import statistics
import zipfile

SCHEMA = "nanojev-financial-pit-xs-pilot-v1"
VENUE = "binance_um"
MS = 1_000_000
DAY_MS = 86_400_000
HORIZON_NS = 86_400_000_000_000            # 1 day, from the frozen definition
LABEL_LAG_NS = 604_800_000_000_000         # 7 days, from the protocol
THRESHOLD_BPS = 25.0
EVENT_NAME = "perp_forward_mark_return_up_25bps_1d_gross"
FEATURES = ("mark_price", "index_price", "mark_index_basis_bps", "last_funding_rate",
            "funding_interval_hours", "quote_volume", "trade_count", "taker_buy_ratio",
            "realized_vol_24bar")
LEGACY_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
PILOT_SYMBOLS = ("ADAUSDT", "DOGEUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT")


def read_rows(path):
    with zipfile.ZipFile(path) as archive:
        name = archive.namelist()[0]
        return [line for line in archive.read(name).decode().strip().splitlines() if line.strip()]


def load_klines(root, kind, symbol):
    """Daily bars as {open_time_ms: {...}} from every zip under <root>/<kind>/<symbol>/."""
    bars = {}
    kind_dir = root / kind / symbol
    if not kind_dir.is_dir():
        return bars
    for path in sorted(kind_dir.glob("*.zip")):
        for line in read_rows(path):
            parts = line.split(",")
            if parts[0] == "open_time" or len(parts) < 11:
                continue
            open_time = int(parts[0])
            bars[open_time] = {
                "open": float(parts[1]), "high": float(parts[2]),
                "low": float(parts[3]), "close": float(parts[4]),
                "volume": float(parts[5]), "close_time": int(parts[6]),
                "quote_volume": float(parts[7]), "count": int(parts[8]),
                "taker_buy_volume": float(parts[9]),
            }
    return bars


def load_funding(root, symbol):
    rows = []
    funding_dir = root / "fundingRate" / symbol
    if not funding_dir.is_dir():
        return rows
    for path in sorted(funding_dir.glob("*.zip")):
        for line in read_rows(path):
            parts = line.split(",")
            if parts[0] == "calc_time" or len(parts) < 3:
                continue
            rows.append({"calc_time": int(parts[0]), "interval_hours": int(parts[1]),
                         "rate": float(parts[2])})
    rows.sort(key=lambda row: row["calc_time"])
    return rows


def resolve_root(roots, symbol):
    """First archive root that actually holds klines for this symbol."""
    for root in roots:
        kind_dir = root / "klines" / symbol
        if kind_dir.is_dir() and any(kind_dir.glob("*.zip")):
            return root
    return None


def build_records(root, symbol, first_day, last_day):
    klines = load_klines(root, "klines", symbol)
    marks = load_klines(root, "markPriceKlines", symbol)
    indexes = load_klines(root, "indexPriceKlines", symbol)
    funding = load_funding(root, symbol)
    days = sorted(set(klines) & set(marks) & set(indexes))

    closes = [marks[day]["close"] for day in days]
    records, skipped = [], {"missing_bar": 0, "zero_volume": 0, "nonpositive_price": 0,
                            "insufficient_vol_window": 0, "no_label_bar": 0,
                            "no_funding_yet": 0, "outside_range": 0,
                            "noncontiguous_label_window": 0}
    funding_cursor = 0
    last_funding = None
    for index, day in enumerate(days):
        day_text = dt.datetime.fromtimestamp(day / 1000, dt.timezone.utc).strftime("%Y-%m-%d")
        if day_text < first_day or day_text > last_day:
            skipped["outside_range"] += 1
            continue
        close_time = marks[day]["close_time"]
        # Advance the settled-funding cursor only up to the decision instant (no look-ahead).
        while funding_cursor < len(funding) and funding[funding_cursor]["calc_time"] <= close_time:
            last_funding = funding[funding_cursor]
            funding_cursor += 1
        if last_funding is None:
            skipped["no_funding_yet"] += 1
            continue
        if index + 1 >= len(days):
            skipped["no_label_bar"] += 1
            continue
        if days[index + 1] - day != DAY_MS:
            # a gap in the archive must not silently become a "1d forward" label
            skipped["noncontiguous_label_window"] += 1
            continue

        mark_close = marks[day]["close"]
        index_close = indexes[day]["close"]
        volume = klines[day]["volume"]
        if mark_close <= 0 or index_close <= 0:
            skipped["nonpositive_price"] += 1
            continue
        if volume <= 0:
            skipped["zero_volume"] += 1
            continue

        # PILOT-LITE: emit realized_vol_24bar as null rather than dropping the record
        # when the trailing 24-bar contiguous window is unavailable.
        realized_vol = None
        if index >= 24 and day - days[index - 24] == 24 * DAY_MS:
            window = closes[index - 24:index + 1]
            returns = [math.log(window[i + 1] / window[i]) for i in range(len(window) - 1)]
            realized_vol = statistics.stdev(returns)
        else:
            skipped["insufficient_vol_window"] += 1

        decision_ns = close_time * MS + 1
        event_ns = close_time * MS
        available_ns = event_ns + 1
        fit_cutoff = event_ns  # trailing window closes on the decision bar

        def feature(value, source_id, version, event=event_ns, avail=available_ns, fit=0):
            return {"value": value, "event_ns": event, "available_ns": avail,
                    "fit_cutoff_ns": fit, "source_id": source_id, "version": version}

        features = {
            "mark_price": feature(mark_close, "venue_mark_price_klines",
                                  "perp-1d-mark-price-close-v1"),
            "index_price": feature(index_close, "venue_index_price_klines",
                                   "perp-1d-index-price-close-v1"),
            "mark_index_basis_bps": feature((mark_close - index_close) / index_close * 10000.0,
                                            "venue_mark_and_index_price_klines",
                                            "perp-mark-index-basis-bps-v1"),
            "last_funding_rate": feature(last_funding["rate"], "venue_funding_rate_history",
                                         "perp-funding-rate-settled-v1",
                                         event=last_funding["calc_time"] * MS,
                                         avail=last_funding["calc_time"] * MS),
            "funding_interval_hours": feature(last_funding["interval_hours"],
                                              "venue_instrument_or_funding_config",
                                              "perp-funding-interval-hours-v1",
                                              event=last_funding["calc_time"] * MS,
                                              avail=last_funding["calc_time"] * MS),
            "quote_volume": feature(klines[day]["quote_volume"], "venue_perp_klines",
                                    "perp-1d-quote-volume-v1"),
            "trade_count": feature(klines[day]["count"], "venue_perp_klines",
                                   "perp-1d-trade-count-v1"),
            "taker_buy_ratio": feature(klines[day]["taker_buy_volume"] / volume,
                                       "venue_perp_klines", "perp-1d-taker-buy-ratio-v1"),
            "realized_vol_24bar": feature(realized_vol, "venue_mark_price_klines",
                                          "perp-mark-realized-vol-24bar-v1", fit=fit_cutoff),
        }
        exit_close = marks[days[index + 1]]["close"]
        if exit_close <= 0:
            skipped["nonpositive_price"] += 1
            continue
        forward_bps = 10000.0 * ((exit_close / mark_close) - 1.0)
        forward_5d_bps = None
        if index + 5 < len(days) and days[index + 5] - day == 5 * DAY_MS \
                and marks[days[index + 5]]["close"] > 0:
            forward_5d_bps = 10000.0 * ((marks[days[index + 5]]["close"] / mark_close) - 1.0)
        end_ns = decision_ns + HORIZON_NS

        records.append({
            "schema_version": SCHEMA,
            "id": f"{VENUE}:{symbol}:{day_text}",
            "asset_id": f"{symbol}-PERP",
            "venue": VENUE,
            "decision_ns": decision_ns,
            "universe_available_ns": event_ns,
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
        "schema_version": "nanojev-perp-pit-xs-pilot-build-v1",
        "status": "NO DATA: the bounded pilot fetch produced no usable files; "
                  "records were not built and none were fabricated",
        "records": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output.parent / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    return summary_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive-roots", default="data/binance_vision_v1,data/binance_xs_pilot_v1",
                        help="comma-separated archive roots; each symbol is built from the "
                             "first root that holds its klines")
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("data/perp_pit_xs_pilot_v1/records.jsonl"))
    parser.add_argument("--symbols",
                        default=",".join(LEGACY_SYMBOLS + PILOT_SYMBOLS))
    parser.add_argument("--first-day", default="2026-06-01")
    parser.add_argument("--last-day", default="2026-08-31")
    args = parser.parse_args()

    roots = [pathlib.Path(r.strip()) for r in args.archive_roots.split(",") if r.strip()]
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    all_records, per_symbol = [], {}
    for symbol in symbols:
        root = resolve_root(roots, symbol)
        if root is None:
            per_symbol[symbol] = {"records": 0, "status": "no_data",
                                  "archive_root": None,
                                  "skipped": {"no_archive_data": 1}}
            continue
        records, skipped = build_records(root, symbol, args.first_day, args.last_day)
        positives = sum(1 for r in records if r["label"]["outcome"])
        per_symbol[symbol] = {
            "records": len(records), "positives": positives,
            "positive_rate": (positives / len(records)) if records else None,
            "archive_root": str(root),
            "skipped": skipped,
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
    symbols_with_data = sorted(s for s, r in per_symbol.items() if r["records"] > 0)
    summary = {
        "schema_version": "nanojev-perp-pit-xs-pilot-build-v1",
        "status": "PILOT cross-sectional cohort (T102); bounded 10-asset x 3-month slice",
        "source": "Binance public archive, USDT-M perpetuals, daily bars "
                  "(see data/binance_vision_v1 and data/binance_xs_pilot_v1 manifests)",
        "venue": VENUE,
        "is_pilot": True,
        "granularity": "1d",
        "features": list(FEATURES),
        "event": EVENT_NAME,
        "threshold_bps": THRESHOLD_BPS, "horizon_ns": HORIZON_NS,
        "label_availability_lag_ns": LABEL_LAG_NS,
        "records": len(all_records), "positives": positives,
        "positive_rate": (positives / len(all_records)) if all_records else None,
        "first_day": args.first_day, "last_day": args.last_day,
        "symbols_with_data": symbols_with_data,
        "symbols_with_data_count": len(symbols_with_data),
        "per_symbol": per_symbol,
        "pilot_lite_deviations": {
            "realized_vol_24bar_nullable": "emitted as null when the trailing 24-bar "
                "contiguous window is unavailable (new symbols only have pilot-month "
                "history); the record is kept rather than dropped",
            "extra_label_fields": "label carries forward_return_bps (1d) and "
                "forward_return_5d_bps (5d, null without a contiguous 5th forward bar) "
                "for downstream cross-sectional arms",
            "new_symbol_window_limits": "new symbols have funding/kline history only "
                "from 2026-06-01; trailing-window features are bounded by that cutoff",
        },
        "honesty": {
            "not_a_return": "labels are gross mark-price moves; fees, spread, slippage, "
                            "funding cashflows and leverage are excluded by construction",
            "not_an_asof_vintage": "the archive is a current snapshot and has been replaced "
                                   "in place historically; it is not evidence of what was "
                                   "knowable at the time",
            "not_live": "no orders, no account, no broker, no trading API was used",
            "no_profitability_claim": True,
        },
    }
    summary_path = args.output.parent / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(json.dumps({"records": summary["records"], "positives": summary["positives"],
                      "positive_rate": (round(summary["positive_rate"], 4)
                                        if summary["positive_rate"] is not None else None),
                      "symbols_with_data": len(symbols_with_data),
                      "output": str(args.output), "summary": str(summary_path)},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
