#!/usr/bin/env python3
"""Build daily PIT-lite records for the 30-asset cross-sectional cohort (T129).

Companion to ``build_xs_v1.py`` (T104, 10 assets) and ``build_perp_pit_mega_v1.py``
(T112, 277 assets). The universe is ``research/live_universe_v2.json``: the top-30
mega-cohort symbols by median 2024-2025 quote_volume (which already contains all
10 xs_v1 symbols).

Each symbol's FULL history is assembled from two source families, merged per day:

  * ``data/rc_futures_v1/<BASE>/<BASE>USDT_1d.csv`` — the second-hand mega
    archive, daily bars through 2025-12-31 (provenance caveat: an imported
    sibling-project copy, not a verified venue pull);
  * the venue zip archives, in precedence order ``data/binance_vision_v1``,
    ``data/binance_xs_v1``, ``data/binance_xs2_v1`` — monthly klines/fundingRate
    zips. xs2 covers the 20 new symbols' 2026 tail (``fetch_binance_xs2_v1.py``);
    vision/xs_v1 cover the legacy 10 through 2026-08.

  Zip bars override csv bars on the same day (the zip is the primary venue
  fetch; the csv is a second-hand copy). The seam at 2025-12-31 -> 2026-01-01
  is handled by building on the merged series, so dfh20/mom20/vol20 windows and
  forward labels are contiguous across sources.

Funding is likewise merged: rc_futures ``funding.csv`` rows (interval INFERRED
from settlement gaps, mega convention) plus zip ``fundingRate`` rows (explicit
interval column), deduplicated by calc_time with the zip row winning. 13 of the
20 new symbols have no rc_futures funding.csv, so their pre-2026 records carry
null funding features — same null-not-drop rule as the mega builder (xs_v1
dropped pre-first-settlement days; xs_v2 emits them with nulls).

Record shape is IDENTICAL to xs_v1 (``nanojev-financial-pit-xs-v1``):
close is the regular klines close used as the mark proxy for every symbol;
close-vs-mark deviation is quantified only where markPriceKlines archives
exist (the 5 legacy vision symbols) — it is UNMEASURED for the 20 new symbols
and for the rc_futures-only portion of legacy history.

Scope and honesty rules (unchanged): current snapshots, not as-of vintages;
simulated/paper research only — no orders, no account, no broker; labels are
GROSS close-price moves, not net returns; partial coverage is reported per
symbol instead of being hidden; no usable data -> honest no-data summary.
"""
import argparse
import csv
import datetime as dt
import json
import math
import pathlib
import statistics
import zipfile

SCHEMA = "nanojev-financial-pit-xs-v1"   # same record schema as xs_v1
VENUE = "binance_um"
MS = 1_000_000
DAY_MS = 86_400_000
HOUR_MS = 3_600_000
WINDOW = 20                                # dfh20 / mom20 / vol20 trailing bars
HORIZON_NS = 86_400_000_000_000            # 1 day, from the frozen definition
LABEL_LAG_NS = 604_800_000_000_000         # 7 days, from the protocol
THRESHOLD_BPS = 25.0
EVENT_NAME = "perp_forward_close_return_up_25bps_1d_gross"
FEATURES = ("close", "quote_volume", "last_funding_rate", "funding_interval_hours",
            "dfh20", "mom20", "vol20")
RC_ROOT = pathlib.Path("data/rc_futures_v1")
ZIP_ROOTS = (pathlib.Path("data/binance_vision_v1"),
             pathlib.Path("data/binance_xs_v1"),
             pathlib.Path("data/binance_xs2_v1"))
UNIVERSE_FILE = pathlib.Path("research/live_universe_v2.json")
EXPECTED_1D_HEADER = ("timestamp", "open", "high", "low", "close",
                      "volume", "quote_volume")
# Universe symbol -> venue archive ticker (verified 2026-09-24): PEPE/SHIB/BONK
# monthly zips only exist under the 1000- tickers in 2026; rc_futures csvs are
# already in the same per-1000 units, so the merged series is contiguous.
VENUE_ALIASES = {"PEPEUSDT": "1000PEPEUSDT",
                 "SHIBUSDT": "1000SHIBUSDT",
                 "BONKUSDT": "1000BONKUSDT"}


def load_universe(path=UNIVERSE_FILE):
    doc = json.loads(path.read_text(encoding="utf-8"))
    return list(doc["symbols"])


def read_rows(path):
    with zipfile.ZipFile(path) as archive:
        name = archive.namelist()[0]
        return [line for line in archive.read(name).decode().strip().splitlines()
                if line.strip()]


def _zip_dirs(root, kind, symbol):
    """Candidate dirs for a symbol's zips: canonical name plus venue alias."""
    yield root / kind / symbol
    alias = VENUE_ALIASES.get(symbol)
    if alias:
        yield root / kind / alias


def load_zip_klines(root, symbol):
    """Daily bars {open_time_ms: {...}} from zips under <root>/klines/<sym>/."""
    bars = {}
    for kind_dir in _zip_dirs(root, "klines", symbol):
        if not kind_dir.is_dir():
            continue
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
                    "quote_volume": float(parts[7]),
                }
    return bars


def load_csv_bars(base):
    """{open_time_ms: {...}} from rc_futures <BASE>USDT_1d.csv; close_time is
    synthesized as open + 1d - 1ms (Binance convention), like the mega build."""
    path = RC_ROOT / base / f"{base}USDT_1d.csv"
    bars = {}
    if not path.is_file():
        return bars
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None or tuple(h.strip() for h in header[:7]) \
                != EXPECTED_1D_HEADER:
            return bars
        for row in reader:
            if len(row) < 7 or not row[0].strip():
                continue
            try:
                day = int(dt.datetime.strptime(
                    row[0].strip()[:10], "%Y-%m-%d").replace(
                    tzinfo=dt.timezone.utc).timestamp() * 1000)
                bars[day] = {
                    "open": float(row[1]), "high": float(row[2]),
                    "low": float(row[3]), "close": float(row[4]),
                    "volume": float(row[5]),
                    "close_time": day + DAY_MS - 1,
                    "quote_volume": float(row[6])}
            except (ValueError, IndexError):
                continue
    return bars


def load_zip_funding(root, symbol):
    """fundingRate zips -> {calc_time_ms: {rate, interval_hours}}."""
    rows = {}
    for funding_dir in _zip_dirs(root, "fundingRate", symbol):
        if not funding_dir.is_dir():
            continue
        for path in sorted(funding_dir.glob("*.zip")):
            for line in read_rows(path):
                parts = line.split(",")
                if parts[0] == "calc_time" or len(parts) < 3:
                    continue
                rows[int(parts[0])] = {"rate": float(parts[2]),
                                       "interval_hours": int(parts[1])}
    return rows


def load_csv_funding(base):
    """rc_futures funding.csv -> {calc_time_ms: {rate, interval_hours}};
    interval INFERRED as the rounded gap to the previous settlement."""
    path = RC_ROOT / base / "funding.csv"
    rows = []
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None or header[0].strip() != "timestamp":
            return {}
        for row in reader:
            if len(row) < 2 or not row[0].strip():
                continue
            try:
                t = int(dt.datetime.fromisoformat(row[0].strip()).replace(
                    tzinfo=dt.timezone.utc).timestamp() * 1000)
                rows.append({"calc_ms": t, "rate": float(row[1])})
            except (ValueError, IndexError):
                continue
    rows.sort(key=lambda r: r["calc_ms"])
    out = {}
    for i, row in enumerate(rows):
        interval = None
        if i:
            gap_h = (row["calc_ms"] - rows[i - 1]["calc_ms"]) / HOUR_MS
            if 0 < gap_h <= 48:
                interval = int(round(gap_h))
        out[row["calc_ms"]] = {"rate": row["rate"], "interval_hours": interval}
    return out


def merged_bars(symbol):
    """Merged day -> bar across rc_futures csv + zip roots (zips win)."""
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    bars = load_csv_bars(base)
    for root in ZIP_ROOTS:
        bars.update(load_zip_klines(root, symbol))
    return bars


def merged_funding(symbol):
    """Merged sorted [{calc_ms, rate, interval_hours}] across csv + zip roots
    (zip rows win on a shared calc_time)."""
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    funding = load_csv_funding(base)
    for root in ZIP_ROOTS:
        funding.update(load_zip_funding(root, symbol))
    return [{"calc_ms": t, **funding[t]} for t in sorted(funding)]


def close_vs_mark_deviation(symbol):
    """|mark_close/klines_close - 1| bps over shared days, where mark archives
    exist (only the 5 legacy symbols in data/binance_vision_v1)."""
    root = ZIP_ROOTS[0]
    marks = {}
    kind_dir = root / "markPriceKlines" / symbol
    if kind_dir.is_dir():
        for path in sorted(kind_dir.glob("*.zip")):
            for line in read_rows(path):
                parts = line.split(",")
                if parts[0] == "open_time" or len(parts) < 11:
                    continue
                marks[int(parts[0])] = float(parts[4])
    klines = load_zip_klines(root, symbol)
    if not marks or not klines:
        return None
    deviations = []
    for day in sorted(set(klines) & set(marks)):
        k_close, m_close = klines[day]["close"], marks[day]
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


def build_records(symbol, first_day, last_day):
    klines = merged_bars(symbol)
    funding = merged_funding(symbol)
    days = sorted(klines)
    closes = [klines[day]["close"] for day in days]
    records, skipped = [], {"missing_bar": 0, "zero_volume": 0,
                            "nonpositive_price": 0,
                            "insufficient_feature_window": 0, "no_label_bar": 0,
                            "no_funding_yet": 0, "outside_range": 0,
                            "noncontiguous_label_window": 0}
    funding_cursor = 0
    last_funding = None
    for index, day in enumerate(days):
        day_text = dt.datetime.fromtimestamp(day / 1000, dt.timezone.utc).strftime(
            "%Y-%m-%d")
        if (first_day and day_text < first_day) or (last_day and day_text > last_day):
            skipped["outside_range"] += 1
            continue
        close_time = klines[day]["close_time"]
        while funding_cursor < len(funding) \
                and funding[funding_cursor]["calc_ms"] <= close_time:
            last_funding = funding[funding_cursor]
            funding_cursor += 1
        # xs_v2 follows the MEGA rule: missing funding emits null features,
        # it does not drop the record (13/20 new symbols have no rc funding.csv).
        if index + 1 >= len(days):
            skipped["no_label_bar"] += 1
            continue
        if days[index + 1] - day != DAY_MS:
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

        contiguous20 = index >= WINDOW and day - days[index - WINDOW] == WINDOW * DAY_MS
        if contiguous20:
            window = closes[index - WINDOW:index + 1]
            prior = closes[index - WINDOW:index]
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
        fit_cutoff = event_ns

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
            "quote_volume": feature(klines[day]["quote_volume"],
                                    "venue_perp_klines",
                                    "perp-1d-quote-volume-v1"),
            "last_funding_rate": feature(f_rate, "venue_funding_rate_history",
                                         "perp-funding-rate-settled-v1",
                                         event=f_event, avail=f_event),
            "funding_interval_hours": feature(
                f_hours, "venue_instrument_or_funding_config",
                "perp-funding-interval-hours-v1",
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
        exit_close = klines[days[index + 1]]["close"]
        if exit_close <= 0:
            skipped["nonpositive_price"] += 1
            continue
        forward_bps = 10000.0 * ((exit_close / close) - 1.0)
        forward_5d_bps = None
        if index + 5 < len(days) and days[index + 5] - day == 5 * DAY_MS \
                and klines[days[index + 5]]["close"] > 0:
            forward_5d_bps = 10000.0 * (
                (klines[days[index + 5]]["close"] / close) - 1.0)
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
        "schema_version": "nanojev-perp-pit-xs-build-v2",
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
    parser.add_argument("--universe-file", type=pathlib.Path,
                        default=UNIVERSE_FILE,
                        help="live_universe_v2.json; its symbols list is the cohort")
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("data/perp_pit_xs_v2/records.jsonl"))
    parser.add_argument("--symbols", default=None,
                        help="optional comma-separated override (default: the "
                             "universe file's symbol list)")
    parser.add_argument("--first-day", default=None)
    parser.add_argument("--last-day", default=None)
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = load_universe(args.universe_file)

    all_records, per_symbol, deviations = [], {}, {}
    for symbol in symbols:
        bars = merged_bars(symbol)
        if not bars:
            per_symbol[symbol] = {"records": 0, "status": "no_data",
                                  "skipped": {"no_archive_data": 1}}
            continue
        dev = close_vs_mark_deviation(symbol)
        if dev is not None:
            deviations[symbol] = dev
        records, skipped, coverage = build_records(symbol,
                                                   args.first_day, args.last_day)
        positives = sum(1 for r in records if r["label"]["outcome"])
        per_symbol[symbol] = {
            "records": len(records), "positives": positives,
            "positive_rate": (positives / len(records)) if records else None,
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
        "schema_version": "nanojev-perp-pit-xs-build-v2",
        "status": "FULL-SPAN 30-asset cross-sectional cohort (T129); "
                  "live_universe_v2 top-30 by median 2024-2025 quote_volume",
        "source": "rc_futures_v1 daily csvs (through 2025-12, second-hand copy) "
                  "merged with venue zip archives binance_vision_v1 / "
                  "binance_xs_v1 / binance_xs2_v1 (zips win per day)",
        "universe_file": str(args.universe_file),
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
            "definition": "|markPriceKlines close / klines close - 1| in bps per "
                          "shared day; measured only where mark archives exist "
                          "(the 5 legacy vision symbols)",
            "per_symbol": deviations,
            "unmeasured_symbols": sorted(set(per_symbol) - set(deviations)),
            "note": "the 20 new symbols and the rc_futures-only history have no "
                    "markPriceKlines; close is the mark proxy for ALL 30 symbols "
                    "so the cross-section shares one price basis",
        },
        "deviations_from_xs_v1_builder": {
            "merged_sources": "bars/funding merge rc_futures csv + zip archives "
                "per day (zips win); the 2025-12-31 -> 2026-01-01 seam yields "
                "contiguous windows and labels",
            "funding_null_not_drop": "records with no settled funding yet emit "
                "null funding features instead of being dropped (mega rule; "
                "13/20 new symbols have no rc_futures funding.csv pre-2026)",
            "close_as_mark": "unchanged: regular klines close as the mark proxy "
                "for every symbol; unmeasured for the 20 new symbols",
            "label_basis": "forward_return_bps / forward_return_5d_bps are "
                "close-to-close moves (contiguous daily bars only)",
        },
        "honesty": {
            "not_a_return": "labels are gross close-price moves; fees, spread, "
                            "slippage, funding cashflows and leverage are "
                            "excluded by construction",
            "not_an_asof_vintage": "both sources are current snapshots of "
                                   "history, not as-of vintages; rc_futures_v1 "
                                   "is a second-hand copy, not a verified pull",
            "not_live": "no orders, no account, no broker, no trading API used",
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
