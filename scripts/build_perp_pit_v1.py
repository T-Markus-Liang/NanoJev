#!/usr/bin/env python3
"""Build a REAL-DATA point-in-time cohort from the Binance USDT-M perpetual archive.

This converts downloaded public archive files into the ``nanojev-financial-pit-v1`` record
contract so the UNMODIFIED ``scripts/financial_pit_v1.py`` validator can audit it.

Scope and honesty rules enforced here:
* Real venue data, but the archive is a CURRENT SNAPSHOT of history, not an as-of vintage.
* Simulated/paper research only: no orders, no account, no broker.
* The label is a GROSS mark-price move. It is NOT a return, PnL, edge or tradable result:
  fees, spread, slippage, funding and leverage are excluded by construction.
* The pilot feature set is a DECLARED SUBSET of the R1 allowlist. Three allowlisted
  features cannot be built from this archive (open_interest_level,
  open_interest_log_change_1d need the daily metrics files; liquidation_intensity_1d is
  published by no venue). The R1 allowlist is closed, so this cohort is explicitly a PILOT
  cohort and must not be presented as the frozen R1 cohort.

Decision-time convention: ``decision_ns = close_time_ms * 1_000_000 + 1``, i.e. the venue
bar's recorded close instant plus one nanosecond, per the protocol's literal wording. The
999,999 ns difference from "the first instant after the bar" is immaterial at daily
granularity but is recorded as a pilot assumption for R1.
"""
import argparse
import datetime as dt
import json
import math
import pathlib
import statistics
import zipfile

SCHEMA = "nanojev-financial-pit-v1"
VENUE = "binance_um"
HORIZON_NS = 86_400_000_000_000            # 1 day, from the frozen definition
LABEL_LAG_NS = 604_800_000_000_000         # 7 days, from the protocol
THRESHOLD_BPS = 25.0
MS = 1_000_000
# Verified reproducible against the protocol's own convention (see docs).
DEFINITION_SHA256 = "8796b7f90a0f62f972f0f40e80e07b100f559f47342aece0bf5582fc27392ac9"
EVENT_NAME = "perp_forward_mark_return_up_25bps_1d_gross"
PILOT_FEATURES = ("mark_price", "index_price", "mark_index_basis_bps", "last_funding_rate",
                  "funding_interval_hours", "quote_volume", "trade_count", "taker_buy_ratio",
                  "realized_vol_24bar")
OMITTED_FEATURES = {
    "open_interest_level": "needs the daily metrics archive, not downloaded in this slice",
    "open_interest_log_change_1d": "needs the daily metrics archive, not downloaded in this slice",
    "liquidation_intensity_1d": "no venue in the reviewed set publishes historical liquidations",
}


def read_rows(path):
    with zipfile.ZipFile(path) as archive:
        name = archive.namelist()[0]
        return [line for line in archive.read(name).decode().strip().splitlines() if line.strip()]


def load_klines(root, kind, symbol):
    """Daily bars as {open_time_ms: [open, high, low, close, volume, quote_volume, count, taker_buy_volume]}."""
    bars = {}
    for path in sorted((root / kind / symbol).glob("*.zip")):
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
    for path in sorted((root / "fundingRate" / symbol).glob("*.zip")):
        for line in read_rows(path):
            parts = line.split(",")
            if parts[0] == "calc_time" or len(parts) < 3:
                continue
            rows.append({"calc_time": int(parts[0]), "interval_hours": int(parts[1]),
                         "rate": float(parts[2])})
    rows.sort(key=lambda row: row["calc_time"])
    return rows


def build_records(root, symbol, first_day, last_day):
    klines = load_klines(root, "klines", symbol)
    marks = load_klines(root, "markPriceKlines", symbol)
    indexes = load_klines(root, "indexPriceKlines", symbol)
    funding = load_funding(root, symbol)
    days = sorted(set(klines) & set(marks) & set(indexes))

    closes = [marks[day]["close"] for day in days]
    records, skipped = [], {"missing_bar": 0, "zero_volume": 0, "nonpositive_price": 0,
                            "insufficient_vol_window": 0, "no_label_bar": 0, "no_funding_yet": 0,
                            "outside_range": 0}
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

        mark_close = marks[day]["close"]
        index_close = indexes[day]["close"]
        volume = klines[day]["volume"]
        if mark_close <= 0 or index_close <= 0:
            skipped["nonpositive_price"] += 1
            continue
        if volume <= 0:
            skipped["zero_volume"] += 1
            continue
        if index < 24:
            skipped["insufficient_vol_window"] += 1
            continue

        window = closes[index - 24:index + 1]
        returns = [math.log(window[i + 1] / window[i]) for i in range(len(window) - 1)]
        realized_vol = statistics.stdev(returns)

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
        outcome = (10000.0 * ((exit_close / mark_close) - 1.0)) >= THRESHOLD_BPS
        end_ns = decision_ns + HORIZON_NS

        records.append({
            "schema_version": SCHEMA,
            "id": f"{VENUE}:{symbol}:{day_text}",
            "asset_id": f"{symbol}-PERP",
            "venue": VENUE,
            "decision_ns": decision_ns,
            "universe_available_ns": event_ns,
            "features": features,
            "label": {"event": EVENT_NAME, "definition_sha256": DEFINITION_SHA256,
                      "end_ns": end_ns, "available_ns": end_ns + LABEL_LAG_NS,
                      "outcome": bool(outcome)},
        })
    return records, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive-root", type=pathlib.Path,
                        default=pathlib.Path("data/binance_vision_v1"))
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("data/perp_pit_v1/records.jsonl"))
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT")
    parser.add_argument("--first-day", default="2023-01-01")
    parser.add_argument("--last-day", default="2026-08-31")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    all_records, report = [], {}
    for symbol in symbols:
        records, skipped = build_records(args.archive_root, symbol, args.first_day, args.last_day)
        all_records.extend(records)
        report[symbol] = {"records": len(records), "skipped": skipped,
                          "first": records[0]["id"].rsplit(":", 1)[-1] if records else None,
                          "last": records[-1]["id"].rsplit(":", 1)[-1] if records else None}
    all_records.sort(key=lambda row: (row["asset_id"], row["decision_ns"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in all_records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    positives = sum(1 for r in all_records if r["label"]["outcome"])
    summary = {
        "schema_version": "nanojev-perp-pit-pilot-build-v1",
        "status": "PILOT cohort, not the frozen R1 cohort",
        "source": "Binance public archive, USDT-M perpetuals, daily bars",
        "venue": VENUE,
        "is_pilot": True,
        "pilot_reason": "R1 allowlist is a closed 12-feature set; three features cannot be built "
                        "from this archive, so this cohort declares a documented 9-feature subset.",
        "features": list(PILOT_FEATURES),
        "omitted_features": OMITTED_FEATURES,
        "event": EVENT_NAME, "definition_sha256": DEFINITION_SHA256,
        "threshold_bps": THRESHOLD_BPS, "horizon_ns": HORIZON_NS,
        "label_availability_lag_ns": LABEL_LAG_NS,
        "records": len(all_records), "positives": positives,
        "positive_rate": (positives / len(all_records)) if all_records else None,
        "first_day": args.first_day, "last_day": args.last_day,
        "per_symbol": report,
        "honesty": {
            "not_a_return": "the label is a gross mark-price move; fees, spread, slippage, "
                            "funding and leverage are excluded by construction",
            "not_an_asof_vintage": "the archive is a current snapshot and has been replaced in "
                                   "place historically; it is not evidence of what was knowable",
            "not_live": "no orders, no account, no broker, no trading API was used",
            "no_profitability_claim": True,
        },
    }
    summary_path = args.output.parent / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"records": summary["records"], "positives": summary["positives"],
                      "positive_rate": round(summary["positive_rate"], 4),
                      "output": str(args.output), "summary": str(summary_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
