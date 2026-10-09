#!/usr/bin/env python3
"""Build a small REAL-DATA point-in-time INTRADAY record set from the Binance archive.

Hourly companion to ``build_perp_pit_v1.py``. It converts the bounded intraday download
produced by ``fetch_binance_intraday_v1.py`` (one symbol, one interval, one month) into
hourly records following the same record-shape conventions: a ``features`` dict whose
values carry ``value`` / ``event_ns`` / ``available_ns`` / ``fit_cutoff_ns`` /
``source_id`` / ``version``, a ``decision_ns`` at bar close + 1 ns, and a gross
forward mark-price label.

This unlocks measurements the daily cohort cannot express:
* lead-lag between intraday features and the next few hours of mark returns;
* funding-timing effects (funding settles intra-day, e.g. every 8h on BTCUSDT).

Scope and honesty rules enforced here:
* Real venue data, but the archive is a CURRENT SNAPSHOT of history, not an as-of vintage.
* Simulated/paper research only: no orders, no account, no broker.
* The label is a GROSS mark-price move. It is NOT a return net of costs, PnL, edge or a
  tradable result: fees, spread, slippage, funding cashflows and leverage are excluded.
* If the intraday fetch produced no data, this script writes an honest "no data"
  build_summary.json and exits 0 rather than fabricating records.
"""
import argparse
import datetime as dt
import json
import math
import pathlib
import statistics
import zipfile

SCHEMA = "nanojev-financial-pit-intraday-v1"
VENUE = "binance_um"
MS = 1_000_000
HOUR_NS = 3_600_000_000_000
HORIZON_HOURS = 4
HORIZON_NS = HORIZON_HOURS * HOUR_NS
LABEL_LAG_NS = 604_800_000_000_000          # keep the v1 7-day label-availability convention
THRESHOLD_BPS = 25.0
EVENT_NAME = f"perp_forward_mark_return_up_25bps_{HORIZON_HOURS}h_gross"
FEATURES = ("mark_price", "index_price", "mark_index_basis_bps", "last_funding_rate",
            "funding_interval_hours", "ns_until_next_funding_settlement",
            "quote_volume", "trade_count", "taker_buy_ratio", "realized_vol_24h")


def read_rows(path):
    with zipfile.ZipFile(path) as archive:
        name = archive.namelist()[0]
        return [line for line in archive.read(name).decode().strip().splitlines() if line.strip()]


def load_klines(root, kind, symbol):
    """Bars as {open_time_ms: {...}} from every zip under <root>/<kind>/<symbol>/."""
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


def build_records(root, symbol):
    klines = load_klines(root, "klines", symbol)
    marks = load_klines(root, "markPriceKlines", symbol)
    indexes = load_klines(root, "indexPriceKlines", symbol)
    funding = load_funding(root, symbol)
    # mark klines are the decision price; use perp klines for volume and index for basis
    hours = sorted(set(klines) & set(marks))
    have_index = bool(indexes)
    if have_index:
        hours = [h for h in hours if h in indexes]

    closes = [marks[h]["close"] for h in hours]
    funding_times = [row["calc_time"] for row in funding]
    records, skipped = [], {"missing_bar": 0, "zero_volume": 0, "nonpositive_price": 0,
                            "insufficient_vol_window": 0, "no_label_bar": 0,
                            "no_funding_yet": 0, "noncontiguous_vol_window": 0,
                            "noncontiguous_label_window": 0}
    funding_cursor = 0
    last_funding = None
    for index, hour in enumerate(hours):
        hour_text = dt.datetime.fromtimestamp(hour / 1000, dt.timezone.utc) \
            .strftime("%Y-%m-%dT%H")
        close_time = marks[hour]["close_time"]
        # Advance the settled-funding cursor only up to the decision instant (no look-ahead).
        while funding_cursor < len(funding) and funding[funding_cursor]["calc_time"] <= close_time:
            last_funding = funding[funding_cursor]
            funding_cursor += 1
        if last_funding is None:
            skipped["no_funding_yet"] += 1
            continue
        if index + HORIZON_HOURS >= len(hours):
            skipped["no_label_bar"] += 1
            continue
        if hours[index + HORIZON_HOURS] - hour != HORIZON_HOURS * 3_600_000:
            # a gap in the archive (e.g. a missing month) must not silently become
            # a "4h forward" label that actually spans days
            skipped["noncontiguous_label_window"] += 1
            continue

        mark_close = marks[hour]["close"]
        volume = klines[hour]["volume"]
        if mark_close <= 0:
            skipped["nonpositive_price"] += 1
            continue
        if volume <= 0:
            skipped["zero_volume"] += 1
            continue
        if index < 24:
            skipped["insufficient_vol_window"] += 1
            continue
        if hour - hours[index - 24] != 24 * 3_600_000:
            skipped["noncontiguous_vol_window"] += 1
            continue

        window = closes[index - 24:index + 1]
        returns = [math.log(window[i + 1] / window[i]) for i in range(len(window) - 1)]
        realized_vol = statistics.stdev(returns)

        # Next scheduled settlement is knowable in advance (funding times are on a fixed
        # grid), so exposing the distance to it is not look-ahead.
        next_funding = next((t for t in funding_times if t > close_time), None)
        decision_ns = close_time * MS + 1
        event_ns = close_time * MS
        available_ns = event_ns + 1
        fit_cutoff = event_ns

        def feature(value, source_id, version, event=event_ns, avail=available_ns, fit=0):
            return {"value": value, "event_ns": event, "available_ns": avail,
                    "fit_cutoff_ns": fit, "source_id": source_id, "version": version}

        features = {
            "mark_price": feature(mark_close, "venue_mark_price_klines",
                                  "perp-1h-mark-price-close-v1"),
            "last_funding_rate": feature(last_funding["rate"], "venue_funding_rate_history",
                                         "perp-funding-rate-settled-v1",
                                         event=last_funding["calc_time"] * MS,
                                         avail=last_funding["calc_time"] * MS),
            "funding_interval_hours": feature(last_funding["interval_hours"],
                                              "venue_instrument_or_funding_config",
                                              "perp-funding-interval-hours-v1",
                                              event=last_funding["calc_time"] * MS,
                                              avail=last_funding["calc_time"] * MS),
            "ns_until_next_funding_settlement": feature(
                (next_funding - close_time) * MS if next_funding is not None else None,
                "venue_funding_rate_schedule", "perp-funding-time-to-settlement-v1"),
            "quote_volume": feature(klines[hour]["quote_volume"], "venue_perp_klines",
                                    "perp-1h-quote-volume-v1"),
            "trade_count": feature(klines[hour]["count"], "venue_perp_klines",
                                   "perp-1h-trade-count-v1"),
            "taker_buy_ratio": feature(klines[hour]["taker_buy_volume"] / volume,
                                       "venue_perp_klines", "perp-1h-taker-buy-ratio-v1"),
            "realized_vol_24h": feature(realized_vol, "venue_mark_price_klines",
                                        "perp-mark-realized-vol-24h-v1", fit=fit_cutoff),
        }
        if have_index:
            index_close = indexes[hour]["close"]
            if index_close <= 0:
                skipped["nonpositive_price"] += 1
                continue
            features["index_price"] = feature(index_close, "venue_index_price_klines",
                                              "perp-1h-index-price-close-v1")
            features["mark_index_basis_bps"] = feature(
                (mark_close - index_close) / index_close * 10000.0,
                "venue_mark_and_index_price_klines", "perp-mark-index-basis-bps-v1")

        exit_close = marks[hours[index + HORIZON_HOURS]]["close"]
        if exit_close <= 0:
            skipped["nonpositive_price"] += 1
            continue
        forward_bps = 10000.0 * ((exit_close / mark_close) - 1.0)
        end_ns = decision_ns + HORIZON_NS

        records.append({
            "schema_version": SCHEMA,
            "id": f"{VENUE}:{symbol}:{hour_text}",
            "asset_id": f"{symbol}-PERP",
            "venue": VENUE,
            "decision_ns": decision_ns,
            "universe_available_ns": event_ns,
            "features": features,
            "label": {"event": EVENT_NAME,
                      "end_ns": end_ns, "available_ns": end_ns + LABEL_LAG_NS,
                      "outcome": bool(forward_bps >= THRESHOLD_BPS),
                      "forward_return_bps": forward_bps},
        })
    return records, skipped


def no_data_summary(output):
    summary = {
        "schema_version": "nanojev-perp-pit-intraday-build-v1",
        "status": "NO DATA: the bounded intraday fetch produced no usable files; "
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
    parser.add_argument("--archive-root", type=pathlib.Path,
                        default=pathlib.Path("data/binance_intraday_v1"))
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("data/perp_pit_intraday_v1/records.jsonl"))
    parser.add_argument("--symbol", default="ALL",
                        help="comma-separated symbols, or ALL (default) to use every "
                             "symbol directory present under <archive-root>/klines")
    args = parser.parse_args()

    if args.symbol.strip().upper() == "ALL":
        klines_root = args.archive_root / "klines"
        symbols = sorted(p.name for p in klines_root.iterdir() if p.is_dir()) \
            if klines_root.is_dir() else []
    else:
        symbols = [s.strip().upper() for s in args.symbol.split(",") if s.strip()]

    all_records, per_symbol = [], {}
    for symbol in symbols:
        kline_dir = args.archive_root / "klines" / symbol
        mark_dir = args.archive_root / "markPriceKlines" / symbol
        if not (kline_dir.is_dir() and any(kline_dir.glob("*.zip"))
                and mark_dir.is_dir() and any(mark_dir.glob("*.zip"))):
            per_symbol[symbol] = {"records": 0, "status": "no_data",
                                  "skipped": {"no_archive_data": 1}}
            continue
        records, skipped = build_records(args.archive_root, symbol)
        positives = sum(1 for r in records if r["label"]["outcome"])
        per_symbol[symbol] = {
            "records": len(records), "positives": positives,
            "positive_rate": (positives / len(records)) if records else None,
            "skipped": skipped,
            "first": records[0]["id"].rsplit(":", 1)[-1] if records else None,
            "last": records[-1]["id"].rsplit(":", 1)[-1] if records else None}
        all_records.extend(records)

    if not all_records:
        summary_path = no_data_summary(args.output)
        print(json.dumps({"records": 0, "status": "no_data",
                          "summary": str(summary_path)}, sort_keys=True))
        return 0

    records = all_records
    records.sort(key=lambda row: (row["decision_ns"], row["id"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    positives = sum(1 for r in records if r["label"]["outcome"])
    summary = {
        "schema_version": "nanojev-perp-pit-intraday-build-v1",
        "status": "PILOT intraday cohort; scaffold for hourly lead-lag / funding-timing work",
        "source": "Binance public archive, USDT-M perpetuals, hourly bars "
                  "(see data/binance_intraday_v1/fetch_manifest.json)",
        "venue": VENUE,
        "is_pilot": True,
        "granularity": "1h",
        "features": list(FEATURES),
        "event": EVENT_NAME,
        "threshold_bps": THRESHOLD_BPS, "horizon_ns": HORIZON_NS,
        "label_availability_lag_ns": LABEL_LAG_NS,
        "records": len(records), "positives": positives,
        "positive_rate": (positives / len(records)) if records else None,
        "per_symbol": per_symbol,
        "honesty": {
            "not_a_return": "the label is a gross mark-price move; fees, spread, slippage, "
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
                      "output": str(args.output), "summary": str(summary_path)},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
