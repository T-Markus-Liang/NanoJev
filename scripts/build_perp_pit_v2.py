#!/usr/bin/env python3
"""Build the REAL-DATA R1 point-in-time cohort (protocol v2, frozen 2026-09-20).

Converts downloaded public archive files into the ``nanojev-financial-pit-v1`` record
contract so the UNMODIFIED ``scripts/financial_pit_v1.py`` validator can audit it.

Differences from the pilot builder (build_perp_pit_v1.py):
* 11-feature closed schema: adds ``open_interest_level`` and
  ``open_interest_log_change_1d`` from the Binance daily metrics archive; the closed
  schema dropped ``liquidation_intensity_1d`` per owner decision D1-a/D1.2.
* ``label.definition_sha256`` is the R1-rebased definition hash
  ``cffd49217c95e83bedffc05f43f018964758cc58ba153f3963fc221d1f37124c``
  (USDT-margined linear perpetuals only).
* Venue scope is Binance USD-M only: Bybit's public kline carries no trade_count or
  taker-buy decomposition, Hyperliquid has no historical OI series, and Aster was
  dropped by owner decision. Venue holdout is declared unsatisfiable in V1.
* Decision spacing complies literally with the frozen label-spacing rule:
  consecutive decisions per instrument are >= horizon_ns + embargo_ns apart
  (86,400e9 + 3,600e9 ns), so at most every other daily bar emits a decision.
* Trailing-30-bar median quote-volume liquidity screen (>= 1M USDT-equivalent),
  evaluated causally at each decision.

OI contract (protocol v2 ``open_interest_contract``): ``open_interest_level`` is the
most recent ``sum_open_interest_value`` (USD notional, single-side) row with
observation_ns < decision_ns; the log change uses the same as-of rule 86400e9 ns
earlier. ``metrics.create_time`` is a UTC datetime string, not an integer epoch —
it is parsed on its own path. Missing/nonpositive OI excludes the row.
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
EMBARGO_NS = 3_600_000_000_000             # 1 hour, from the frozen core
MIN_SPACING_NS = HORIZON_NS + EMBARGO_NS   # literal label-spacing rule
LABEL_LAG_NS = 604_800_000_000_000         # 7 days, from the protocol
THRESHOLD_BPS = 25.0
MS = 1_000_000
MIN_MEDIAN_QUOTE_VOLUME = 1_000_000.0      # liquidity screen, USDT-equivalent
LIQUIDITY_WINDOW = 30
VOL_WINDOW = 24
DEFINITION_SHA256 = "cffd49217c95e83bedffc05f43f018964758cc58ba153f3963fc221d1f37124c"
EVENT_NAME = "perp_forward_mark_return_up_25bps_1d_gross"
FEATURES = ("mark_price", "index_price", "mark_index_basis_bps", "last_funding_rate",
            "funding_interval_hours", "open_interest_level", "open_interest_log_change_1d",
            "quote_volume", "trade_count", "taker_buy_ratio", "realized_vol_24bar")


def read_rows(path):
    with zipfile.ZipFile(path) as archive:
        name = archive.namelist()[0]
        return [line for line in archive.read(name).decode().strip().splitlines() if line.strip()]


def load_klines(root, kind, symbol):
    """Daily bars as {open_time_ms: {...}} from the Binance monthly archive."""
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


def load_metrics(metrics_roots, symbol):
    """Daily metrics files -> sorted [(obs_ns, sum_open_interest_value)].

    ``create_time`` is a UTC datetime string (``YYYY-MM-DD HH:MM:SS``), parsed on its
    own path — never through the integer-epoch branch used by klines/funding.
    """
    rows = []
    for root in metrics_roots:
        directory = root / "metrics" / symbol
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.zip")):
            for line in read_rows(path):
                parts = line.split(",")
                if parts[0] == "create_time" or len(parts) < 4:
                    continue
                stamp = dt.datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=dt.timezone.utc)
                obs_ns = int(stamp.timestamp() * 1_000_000_000)
                value = float(parts[3])  # sum_open_interest_value, USD notional
                rows.append((obs_ns, value))
    rows.sort()
    return rows


def last_before(rows, bound_ns):
    """Most recent (obs_ns, value) with obs_ns < bound_ns, or None."""
    lo, hi = 0, len(rows)
    while lo < hi:
        mid = (lo + hi) // 2
        if rows[mid][0] < bound_ns:
            lo = mid + 1
        else:
            hi = mid
    return rows[lo - 1] if lo else None


def build_records(root, metrics_roots, symbol, first_day, last_day):
    klines = load_klines(root, "klines", symbol)
    marks = load_klines(root, "markPriceKlines", symbol)
    indexes = load_klines(root, "indexPriceKlines", symbol)
    funding = load_funding(root, symbol)
    oi = load_metrics(metrics_roots, symbol)
    days = sorted(set(klines) & set(marks) & set(indexes))

    closes = [marks[day]["close"] for day in days]
    records, skipped = [], {"missing_bar": 0, "zero_volume": 0, "nonpositive_price": 0,
                            "insufficient_vol_window": 0, "no_label_bar": 0,
                            "no_funding_yet": 0, "outside_range": 0,
                            "liquidity_screen": 0, "no_oi_current": 0,
                            "no_oi_prior": 0, "nonpositive_oi": 0, "spacing": 0}
    funding_cursor = 0
    last_funding = None
    last_decision_ns = None
    quote_history = []
    for index, day in enumerate(days):
        day_text = dt.datetime.fromtimestamp(day / 1000, dt.timezone.utc).strftime("%Y-%m-%d")
        quote_history.append(klines[day]["quote_volume"])
        if day_text < first_day or day_text > last_day:
            skipped["outside_range"] += 1
            continue
        close_time = marks[day]["close_time"]
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
        if index < VOL_WINDOW:
            skipped["insufficient_vol_window"] += 1
            continue

        decision_ns = close_time * MS + 1
        # Literal label-spacing rule: >= horizon + embargo between decisions.
        if last_decision_ns is not None and decision_ns - last_decision_ns < MIN_SPACING_NS:
            skipped["spacing"] += 1
            continue

        # Causal liquidity screen: trailing 30-bar median quote volume.
        window_qv = quote_history[-LIQUIDITY_WINDOW:]
        if len(window_qv) < LIQUIDITY_WINDOW:
            skipped["liquidity_screen"] += 1
            continue
        if statistics.median(window_qv) < MIN_MEDIAN_QUOTE_VOLUME:
            skipped["liquidity_screen"] += 1
            continue

        # Open interest: last observation strictly before decision_ns, and the
        # same rule applied at decision_ns - 1 day.
        oi_now = last_before(oi, decision_ns)
        if oi_now is None:
            skipped["no_oi_current"] += 1
            continue
        oi_prev = last_before(oi, decision_ns - HORIZON_NS)
        if oi_prev is None:
            skipped["no_oi_prior"] += 1
            continue
        if oi_now[1] <= 0 or oi_prev[1] <= 0:
            skipped["nonpositive_oi"] += 1
            continue
        oi_log_change = math.log(oi_now[1] / oi_prev[1])

        window = closes[index - VOL_WINDOW:index + 1]
        returns = [math.log(window[i + 1] / window[i]) for i in range(len(window) - 1)]
        realized_vol = statistics.stdev(returns)

        event_ns = close_time * MS
        available_ns = event_ns + 1

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
            "open_interest_level": feature(oi_now[1], "venue_metrics_daily_archive",
                                           "perp-oi-usd-notional-asof-v1",
                                           event=oi_now[0], avail=oi_now[0]),
            "open_interest_log_change_1d": feature(oi_log_change,
                                                   "venue_metrics_daily_archive",
                                                   "perp-oi-log-change-1d-asof-v1",
                                                   event=oi_now[0], avail=oi_now[0],
                                                   fit=oi_now[0]),
            "quote_volume": feature(klines[day]["quote_volume"], "venue_perp_klines",
                                    "perp-1d-quote-volume-v1"),
            "trade_count": feature(klines[day]["count"], "venue_perp_klines",
                                   "perp-1d-trade-count-v1"),
            "taker_buy_ratio": feature(klines[day]["taker_buy_volume"] / volume,
                                       "venue_perp_klines", "perp-1d-taker-buy-ratio-v1"),
            "realized_vol_24bar": feature(realized_vol, "venue_mark_price_klines",
                                          "perp-mark-realized-vol-24bar-v1", fit=event_ns),
        }
        exit_close = marks[days[index + 1]]["close"]
        if exit_close <= 0:
            skipped["nonpositive_price"] += 1
            continue
        outcome = (10000.0 * ((exit_close / mark_close) - 1.0)) >= THRESHOLD_BPS
        end_ns = decision_ns + HORIZON_NS
        last_decision_ns = decision_ns

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
    parser.add_argument("--metrics-roots", type=pathlib.Path, nargs="+",
                        default=[pathlib.Path("data/binance_metrics_v1")],
                        help="roots each containing metrics/{SYMBOL}/*.zip; per-symbol "
                             "fetch roots are expanded automatically")
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("data/perp_pit_v2/records.jsonl"))
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT")
    parser.add_argument("--first-day", default="2023-01-01")
    parser.add_argument("--last-day", default="2026-08-31")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    metrics_roots = []
    for root in args.metrics_roots:
        metrics_roots.append(root)
        for symbol in symbols:  # per-symbol fetch roots also work
            candidate = root / symbol
            if candidate.is_dir():
                metrics_roots.append(candidate)

    all_records, report = [], {}
    for symbol in symbols:
        records, skipped = build_records(args.archive_root, metrics_roots, symbol,
                                         args.first_day, args.last_day)
        all_records.extend(records)
        report[symbol] = {"records": len(records), "skipped": skipped,
                          "first": records[0]["id"].rsplit(":", 1)[-1] if records else None,
                          "last": records[-1]["id"].rsplit(":", 1)[-1] if records else None}
    all_records.sort(key=lambda row: (row["asset_id"], row["decision_ns"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:  # never overwrite evidence
        for record in all_records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    positives = sum(1 for r in all_records if r["label"]["outcome"])
    summary = {
        "schema_version": "nanojev-perp-pit-r1-build-v1",
        "status": "R1 candidate cohort under frozen protocol v2; not training-ready until "
                  "the unmodified validator accepts it and the freeze receipt is recorded",
        "source": "Binance public archive (klines, markPriceKlines, indexPriceKlines, "
                  "fundingRate, daily metrics), USDT-M perpetuals, daily bars",
        "venues": [VENUE],
        "venue_note": "Binance only: Bybit lacks trade_count/taker_buy_ratio in its public "
                      "kline payload; Hyperliquid has no historical OI series; Aster was "
                      "dropped by owner decision 2026-09-20.",
        "is_pilot": False,
        "features": list(FEATURES),
        "dropped_features": {"liquidation_intensity_1d": "no uniform historical source; "
                              "dropped per owner D1.2 rather than imputed"},
        "decision_spacing_ns": MIN_SPACING_NS,
        "liquidity_screen": {"rule": "median trailing 30-bar quote volume >= 1M USDT",
                             "window_bars": LIQUIDITY_WINDOW},
        "event": EVENT_NAME, "definition_sha256": DEFINITION_SHA256,
        "protocol": "research/financial_experiment_protocol_v2.json",
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
            "venue_holdout_limitation": "single-venue V1 cohort; venue holdout declared "
                                        "unsatisfiable, deferred to a registered extension",
        },
    }
    summary_path = args.output.parent / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"records": summary["records"], "positives": summary["positives"],
                      "positive_rate": round(summary["positive_rate"], 4) if all_records else None,
                      "output": str(args.output), "summary": str(summary_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
