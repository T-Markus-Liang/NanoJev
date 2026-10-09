#!/usr/bin/env python3
"""Build daily PIT-lite records for the FULL 10-asset cross-sectional cohort (T104).

Companion to ``build_xs_pilot_v1.py``. It merges two archive slices over their FULL
available span (no pilot-month window):
  * the approved 5-symbol slice in ``data/binance_vision_v1/``
    (BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT), and
  * the T104 full-history 5-symbol slice in ``data/binance_xs_v1/``
    (ADAUSDT, DOGEUSDT, LINKUSDT, LTCUSDT, DOTUSDT; klines + fundingRate only,
    ``fetch_binance_xs_full_v1.py``),
and emits one daily record per (symbol, day) to
``data/perp_pit_xs_v1/records.jsonl`` + ``build_summary.json``.

Record-shape conventions follow ``build_perp_pit_v1.py`` / ``build_xs_pilot_v1.py``:
a ``features`` dict whose values carry ``value`` / ``event_ns`` / ``available_ns`` /
``fit_cutoff_ns`` / ``source_id`` / ``version``, ``decision_ns`` at bar close + 1 ns,
and gross forward close-return labels (1d and 5d, contiguous bars only).

DELIBERATE DEVIATION from the pilot builder — close-as-mark:
  the T104 archive slice does NOT include markPriceKlines/indexPriceKlines (bounded
  fetch: the XS arms only need close + volume + funding). The ``close`` feature is
  the regular klines close used AS THE MARK PROXY for every symbol — including the
  legacy five, so all 10 assets share one price basis and cross-sectional ranks are
  not contaminated by a mixed close/mark basis. The deviation between klines close
  and markPriceKlines close is quantified on the legacy symbols (where both exist)
  and reported in build_summary.json under ``close_vs_mark_deviation``; for the new
  symbols the deviation is unmeasured on this archive slice (daily mark vs last
  differs by small basis noise, typically single-digit bps).

Features emitted (values are null rather than record-dropping when the trailing
contiguous window is unavailable — same pilot-lite rule, recorded in the summary):
  close, quote_volume, last_funding_rate, funding_interval_hours,
  dfh20  = close / max(close[i-20:i]) - 1   (strictly prior 20 contiguous bars),
  mom20  = close / close[i-20] - 1          (strictly prior 20-bar return),
  vol20  = stdev of the 20 trailing daily log-returns (same contiguous window).

Scope and honesty rules enforced here:
* Real venue data, but the archive is a CURRENT SNAPSHOT of history, not an as-of vintage.
* Simulated/paper research only: no orders, no account, no broker.
* Labels are GROSS close-price moves. They are NOT returns net of costs, PnL, edge or
  tradable results: fees, spread, slippage, funding cashflows and leverage are excluded.
* If no usable data exists, this script writes an honest "no data" build_summary.json
  and exits 0 rather than fabricating records. Partial fetch coverage is reported
  per symbol under ``coverage`` instead of being hidden.
"""
import argparse
import datetime as dt
import json
import math
import pathlib
import statistics
import zipfile

SCHEMA = "nanojev-financial-pit-xs-v1"
VENUE = "binance_um"
MS = 1_000_000
DAY_MS = 86_400_000
WINDOW = 20                                # dfh20 / mom20 / vol20 trailing bars
HORIZON_NS = 86_400_000_000_000            # 1 day, from the frozen definition
LABEL_LAG_NS = 604_800_000_000_000         # 7 days, from the protocol
THRESHOLD_BPS = 25.0
EVENT_NAME = "perp_forward_close_return_up_25bps_1d_gross"
FEATURES = ("close", "quote_volume", "last_funding_rate", "funding_interval_hours",
            "dfh20", "mom20", "vol20")
LEGACY_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
NEW_SYMBOLS = ("ADAUSDT", "DOGEUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT")


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


def close_vs_mark_deviation(root, symbol):
    """|mark_close/klines_close - 1| in bps over shared days, where marks exist."""
    marks = load_klines(root, "markPriceKlines", symbol)
    if not marks:
        return None
    klines = load_klines(root, "klines", symbol)
    deviations = []
    for day in sorted(set(klines) & set(marks)):
        k_close, m_close = klines[day]["close"], marks[day]["close"]
        if k_close > 0 and m_close > 0:
            deviations.append(abs(m_close / k_close - 1.0) * 10000.0)
    if not deviations:
        return None
    deviations.sort()
    n = len(deviations)
    return {"days": n,
            "mean_bps": round(sum(deviations) / n, 3),
            "median_bps": round(deviations[n // 2], 3),
            "p95_bps": round(deviations[min(n - 1, int(0.95 * n))], 3),
            "max_bps": round(deviations[-1], 3)}


def build_records(root, symbol, first_day, last_day):
    klines = load_klines(root, "klines", symbol)
    funding = load_funding(root, symbol)
    days = sorted(klines)

    closes = [klines[day]["close"] for day in days]
    records, skipped = [], {"missing_bar": 0, "zero_volume": 0, "nonpositive_price": 0,
                            "insufficient_feature_window": 0, "no_label_bar": 0,
                            "no_funding_yet": 0, "outside_range": 0,
                            "noncontiguous_label_window": 0}
    funding_cursor = 0
    last_funding = None
    for index, day in enumerate(days):
        day_text = dt.datetime.fromtimestamp(day / 1000, dt.timezone.utc).strftime("%Y-%m-%d")
        if (first_day and day_text < first_day) or (last_day and day_text > last_day):
            skipped["outside_range"] += 1
            continue
        close_time = klines[day]["close_time"]
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

        close = klines[day]["close"]
        volume = klines[day]["volume"]
        if close <= 0:
            skipped["nonpositive_price"] += 1
            continue
        if volume <= 0:
            skipped["zero_volume"] += 1
            continue

        # Trailing-20 contiguous-bar features; null (not dropped) when unavailable.
        contiguous20 = index >= WINDOW and day - days[index - WINDOW] == WINDOW * DAY_MS
        if contiguous20:
            window = closes[index - WINDOW:index + 1]          # 21 closes, strictly prior high
            prior = closes[index - WINDOW:index]               # 20 strictly prior closes
            dfh20 = close / max(prior) - 1.0 if max(prior) > 0 else None
            mom20 = close / closes[index - WINDOW] - 1.0 \
                if closes[index - WINDOW] > 0 else None
            returns = [math.log(window[i + 1] / window[i])
                       for i in range(len(window) - 1) if window[i] > 0 and window[i + 1] > 0]
            vol20 = statistics.stdev(returns) if len(returns) >= WINDOW else None
        else:
            dfh20 = mom20 = vol20 = None
            skipped["insufficient_feature_window"] += 1

        decision_ns = close_time * MS + 1
        event_ns = close_time * MS
        available_ns = event_ns + 1
        fit_cutoff = event_ns  # trailing window closes on the decision bar

        def feature(value, source_id, version, event=event_ns, avail=available_ns, fit=0):
            return {"value": value, "event_ns": event, "available_ns": avail,
                    "fit_cutoff_ns": fit, "source_id": source_id, "version": version}

        features = {
            "close": feature(close, "venue_perp_klines",
                             "perp-1d-close-as-mark-proxy-v1"),
            "quote_volume": feature(klines[day]["quote_volume"], "venue_perp_klines",
                                    "perp-1d-quote-volume-v1"),
            "last_funding_rate": feature(last_funding["rate"], "venue_funding_rate_history",
                                         "perp-funding-rate-settled-v1",
                                         event=last_funding["calc_time"] * MS,
                                         avail=last_funding["calc_time"] * MS),
            "funding_interval_hours": feature(last_funding["interval_hours"],
                                              "venue_instrument_or_funding_config",
                                              "perp-funding-interval-hours-v1",
                                              event=last_funding["calc_time"] * MS,
                                              avail=last_funding["calc_time"] * MS),
            "dfh20": feature(dfh20, "venue_perp_klines",
                             "perp-1d-close-dist-from-20bar-high-v1", fit=fit_cutoff),
            "mom20": feature(mom20, "venue_perp_klines",
                             "perp-1d-close-20bar-momentum-v1", fit=fit_cutoff),
            "vol20": feature(vol20, "venue_perp_klines",
                             "perp-1d-close-realized-vol-20bar-v1", fit=fit_cutoff),
        }
        exit_close = klines[days[index + 1]]["close"]
        if exit_close <= 0:
            skipped["nonpositive_price"] += 1
            continue
        forward_bps = 10000.0 * ((exit_close / close) - 1.0)
        forward_5d_bps = None
        if index + 5 < len(days) and days[index + 5] - day == 5 * DAY_MS \
                and klines[days[index + 5]]["close"] > 0:
            forward_5d_bps = 10000.0 * ((klines[days[index + 5]]["close"] / close) - 1.0)
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
    coverage = {"first_bar": (dt.datetime.fromtimestamp(days[0] / 1000, dt.timezone.utc)
                              .strftime("%Y-%m-%d") if days else None),
                "last_bar": (dt.datetime.fromtimestamp(days[-1] / 1000, dt.timezone.utc)
                             .strftime("%Y-%m-%d") if days else None),
                "daily_bars": len(days),
                "funding_rows": len(funding)}
    return records, skipped, coverage


def no_data_summary(output):
    summary = {
        "schema_version": "nanojev-perp-pit-xs-build-v1",
        "status": "NO DATA: the archives produced no usable files; "
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
    parser.add_argument("--archive-roots", default="data/binance_vision_v1,data/binance_xs_v1",
                        help="comma-separated archive roots; each symbol is built from the "
                             "first root that holds its klines")
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("data/perp_pit_xs_v1/records.jsonl"))
    parser.add_argument("--symbols",
                        default=",".join(LEGACY_SYMBOLS + NEW_SYMBOLS))
    parser.add_argument("--first-day", default=None,
                        help="optional YYYY-MM-DD lower bound (default: full span)")
    parser.add_argument("--last-day", default=None,
                        help="optional YYYY-MM-DD upper bound (default: full span)")
    args = parser.parse_args()

    roots = [pathlib.Path(r.strip()) for r in args.archive_roots.split(",") if r.strip()]
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    all_records, per_symbol, deviations = [], {}, {}
    for symbol in symbols:
        root = resolve_root(roots, symbol)
        if root is None:
            per_symbol[symbol] = {"records": 0, "status": "no_data",
                                  "archive_root": None,
                                  "skipped": {"no_archive_data": 1}}
            continue
        dev = close_vs_mark_deviation(root, symbol)
        if dev is not None:
            deviations[symbol] = dev
        records, skipped, coverage = build_records(root, symbol,
                                                   args.first_day, args.last_day)
        positives = sum(1 for r in records if r["label"]["outcome"])
        per_symbol[symbol] = {
            "records": len(records), "positives": positives,
            "positive_rate": (positives / len(records)) if records else None,
            "archive_root": str(root),
            "skipped": skipped,
            "coverage": coverage,
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
        "schema_version": "nanojev-perp-pit-xs-build-v1",
        "status": "FULL-SPAN cross-sectional cohort (T104); 10 assets over their "
                  "complete archive history",
        "source": "Binance public archive, USDT-M perpetuals, daily bars "
                  "(see data/binance_vision_v1 and data/binance_xs_v1 manifests)",
        "venue": VENUE,
        "is_pilot": False,
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
        "close_vs_mark_deviation": {
            "definition": "|markPriceKlines close / klines close - 1| in bps per shared "
                          "day; measured only where mark archives exist (legacy slice)",
            "per_symbol": deviations,
            "unmeasured_symbols": sorted(set(per_symbol) - set(deviations)),
            "note": "new symbols have no markPriceKlines in the bounded T104 slice; "
                    "close is used as the mark proxy for ALL 10 symbols so the "
                    "cross-section shares one price basis",
        },
        "deviations_from_pilot_builder": {
            "close_as_mark": "features.close is the regular klines close used as the "
                "mark proxy for every symbol; markPriceKlines/indexPriceKlines are "
                "absent from the T104 slice by design (bounded fetch)",
            "feature_set": "dfh20/mom20/vol20 computed here on a strictly-prior "
                "20-contiguous-bar window; emitted as null rather than dropping the "
                "record when the window is unavailable",
            "full_span": "no pilot-month window; each symbol spans its complete "
                "archive coverage (see per_symbol.coverage)",
            "label_basis": "forward_return_bps / forward_return_5d_bps are close-to-"
                "close moves (contiguous daily bars only), not mark-to-mark",
        },
        "honesty": {
            "not_a_return": "labels are gross close-price moves; fees, spread, slippage, "
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
