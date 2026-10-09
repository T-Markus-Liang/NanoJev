#!/usr/bin/env python3
"""Offline-safe incremental refresh of the perp_pit_v1 cohort tail.

Extends the ledger input with recent public Binance USD-M data so
``financial_forward_ledger_v1.py`` can keep stepping forward. MINIMAL scope:
daily mark close + index close (for the basis feature) + latest settled
funding per symbol — enough to evaluate the frozen signal on new bars.

Reads the existing cohort only to find each symbol's last bar; writes new
PIT-lite records to ``data/binance_refresh_v1/records.jsonl`` (the cohort
file itself is never modified). Records carry the signal-relevant feature
subset (mark_price, index_price, mark_index_basis_bps, last_funding_rate,
funding_interval_hours); features not needed by the signal are omitted.

OFFLINE-SAFE: if data.binance.vision is unreachable (direct and via the
NANOJEV_PROXY egress), the script prints a clear message and exits 0 with
no writes beyond the manifest-free no-op.

READ-ONLY PUBLIC ARCHIVE ONLY — same licence posture as
fetch_binance_vision_v1.py: CC BY-NC-SA 4.0, research use, no orders, no
account, no API key, current snapshots not as-of vintages.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import urllib.error
import urllib.request
import zipfile
import io

MONTHLY_BASE = "https://data.binance.vision/data/futures/um/monthly"
DAILY_BASE = "https://data.binance.vision/data/futures/um/daily"
SOURCE_TERMS = "https://data.binance.vision/Binance_Vision-Terms_of_Use.pdf"
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
VENUE = "binance_um"
MS = 1_000_000
# Same local egress proxy convention as scripts/fetch_venue_perp_v1.py.
DEFAULT_PROXY = os.environ.get("NANOJEV_PROXY", "http://127.0.0.1:7890")

DAILY_URL = {k: f"{DAILY_BASE}/{k}/{{sym}}/1d/{{sym}}-1d-{{date}}.zip"
             for k in ("markPriceKlines", "indexPriceKlines")}
FUNDING_URL = f"{MONTHLY_BASE}/fundingRate/{{sym}}/{{sym}}-fundingRate-{{month}}.zip"


def openers():
    yield "direct", urllib.request.build_opener()
    if DEFAULT_PROXY:
        yield DEFAULT_PROXY, urllib.request.build_opener(
            urllib.request.ProxyHandler(
                {"http": DEFAULT_PROXY, "https": DEFAULT_PROXY}))


def fetch(url, timeout=30):
    """Return (status, payload) or raise urllib.error.URLError on no route."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "nanojev-research/1.0"})
    last_error = None
    for name, opener in openers():
        try:
            with opener.open(request, timeout=timeout) as response:
                return response.status, response.read(), name
        except urllib.error.HTTPError as error:
            # HTTP status reached the wire: host is reachable.
            return error.code, error.read(), name
        except Exception as error:  # noqa: BLE001 - try the next route
            last_error = error
    raise urllib.error.URLError(f"all routes failed: {last_error}")


def zip_rows(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        name = archive.namelist()[0]
        return [l for l in archive.read(name).decode().splitlines() if l.strip()]


def parse_klines(payload):
    """open_time_ms -> {close, close_time} from a daily kline zip."""
    bars = {}
    for line in zip_rows(payload):
        parts = line.split(",")
        if parts[0] == "open_time" or len(parts) < 7:
            continue
        bars[int(parts[0])] = {"close": float(parts[4]),
                               "close_time": int(parts[6])}
    return bars


def parse_funding(payload):
    rows = []
    for line in zip_rows(payload):
        parts = line.split(",")
        if parts[0] == "calc_time" or len(parts) < 3:
            continue
        rows.append({"calc_time": int(parts[0]),
                     "interval_hours": int(parts[1]),
                     "rate": float(parts[2])})
    rows.sort(key=lambda r: r["calc_time"])
    return rows


def cohort_tail(path):
    """symbol -> last bar date (YYYY-MM-DD) from cohort record ids."""
    tail = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rid = json.loads(line)["id"]  # binance_um:SYM:YYYY-MM-DD
        _, sym, date = rid.split(":")
        if date > tail.get(sym, ""):
            tail[sym] = date
    return tail


def months_between(first_day, last_day):
    year, month = first_day.year, first_day.month
    while (year, month) <= (last_day.year, last_day.month):
        yield f"{year:04d}-{month:02d}"
        month += 1
        if month == 13:
            year, month = year + 1, 1


def feature(value, source_id, version, event_ns, avail_ns):
    return {"value": value, "event_ns": event_ns, "available_ns": avail_ns,
            "fit_cutoff_ns": 0, "source_id": source_id, "version": version}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", type=pathlib.Path,
                    default=pathlib.Path("data/perp_pit_v1/records.jsonl"))
    ap.add_argument("--output-dir", type=pathlib.Path,
                    default=pathlib.Path("data/binance_refresh_v1"))
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--days", type=int, default=45,
                    help="max days back to fetch when the cohort tail is old")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    tail = cohort_tail(args.cohort)
    today = dt.datetime.now(dt.timezone.utc).date()
    yesterday = today - dt.timedelta(days=1)

    # Connectivity probe on a file that must exist (first needed mark zip).
    wanted = {s: max(dt.date.fromisoformat(tail.get(s, "2000-01-01"))
                     + dt.timedelta(days=1), yesterday
                     - dt.timedelta(days=args.days - 1))
              for s in symbols}
    wanted = {s: d for s, d in wanted.items() if d <= yesterday}
    if not wanted:
        print(json.dumps({"status": "up_to_date",
                          "note": "cohort tail already covers yesterday; "
                                  "nothing to refresh"}))
        return 0
    first_sym = sorted(wanted)[0]
    probe_url = DAILY_URL["markPriceKlines"].format(
        sym=first_sym, date=wanted[first_sym].isoformat())
    try:
        status, _, route = fetch(probe_url)
    except Exception as error:  # noqa: BLE001
        print(json.dumps({"status": "offline_noop",
                          "note": "data.binance.vision unreachable direct and "
                                  "via egress proxy; no writes performed",
                          "error": f"{type(error).__name__}: {error}"}))
        return 0
    if status >= 500:
        print(json.dumps({"status": "upstream_error_noop",
                          "note": "archive reachable but returned a server "
                                  "error on probe; no writes performed",
                          "http_status": status}))
        return 0

    files, missing, records = [], [], []
    for symbol in symbols:
        if symbol not in wanted:
            continue
        first_day = wanted[symbol]
        # funding rows for the months spanned by the refresh range
        funding = []
        for month in months_between(first_day, yesterday):
            url = FUNDING_URL.format(sym=symbol, month=month)
            try:
                status, payload, via = fetch(url)
            except Exception as error:  # noqa: BLE001 - per-file, not fatal
                missing.append({"url": url,
                                "error": f"{type(error).__name__}"})
                continue
            if status != 200:
                missing.append({"url": url, "http_status": status})
                continue
            funding.extend(parse_funding(payload))
            files.append({"url": url, "http_status": status, "route": via,
                          "bytes": len(payload),
                          "sha256": hashlib.sha256(payload).hexdigest()})
        funding.sort(key=lambda r: r["calc_time"])

        cursor = first_day
        while cursor <= yesterday:
            date_text = cursor.isoformat()
            bars = {}
            for kind in ("markPriceKlines", "indexPriceKlines"):
                url = DAILY_URL[kind].format(sym=symbol, date=date_text)
                try:
                    status, payload, via = fetch(url)
                except Exception as error:  # noqa: BLE001
                    missing.append({"url": url,
                                    "error": f"{type(error).__name__}"})
                    status = None
                if status == 200:
                    bars[kind] = parse_klines(payload)
                    files.append({"url": url, "http_status": status,
                                  "route": via, "bytes": len(payload),
                                  "sha256":
                                  hashlib.sha256(payload).hexdigest()})
                elif status is not None:
                    missing.append({"url": url, "http_status": status})
            cursor += dt.timedelta(days=1)
            if "markPriceKlines" not in bars or "indexPriceKlines" not in bars:
                continue
            # daily files hold a single bar
            mk = next(iter(bars["markPriceKlines"].values()))
            ix = next(iter(bars["indexPriceKlines"].values()))
            close_time = mk["close_time"]
            mark_close, index_close = mk["close"], ix["close"]
            if mark_close <= 0 or index_close <= 0:
                continue
            last_funding = None
            for row in funding:
                if row["calc_time"] <= close_time:
                    last_funding = row
                else:
                    break
            if last_funding is None:
                continue
            decision_ns = close_time * MS + 1
            event_ns = close_time * MS
            avail_ns = event_ns + 1
            records.append({
                "schema_version": "nanojev-financial-pit-v1",
                "id": f"{VENUE}:{symbol}:{date_text}",
                "asset_id": f"{symbol}-PERP",
                "venue": VENUE,
                "decision_ns": decision_ns,
                "universe_available_ns": event_ns,
                "features": {
                    "mark_price": feature(
                        mark_close, "venue_mark_price_klines",
                        "perp-1d-mark-price-close-v1", event_ns, avail_ns),
                    "index_price": feature(
                        index_close, "venue_index_price_klines",
                        "perp-1d-index-price-close-v1", event_ns, avail_ns),
                    "mark_index_basis_bps": feature(
                        (mark_close - index_close) / index_close * 10000.0,
                        "venue_mark_and_index_price_klines",
                        "perp-mark-index-basis-bps-v1", event_ns, avail_ns),
                    "last_funding_rate": feature(
                        last_funding["rate"], "venue_funding_rate_history",
                        "perp-funding-rate-settled-v1",
                        last_funding["calc_time"] * MS,
                        last_funding["calc_time"] * MS),
                    "funding_interval_hours": feature(
                        last_funding["interval_hours"],
                        "venue_instrument_or_funding_config",
                        "perp-funding-interval-hours-v1",
                        last_funding["calc_time"] * MS,
                        last_funding["calc_time"] * MS),
                },
                "label": None,
                "refresh_supplement": True,
            })

    records.sort(key=lambda r: (r["asset_id"], r["decision_ns"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "records.jsonl"
    with out.open("w", encoding="utf-8") as stream:
        for r in records:
            stream.write(json.dumps(r, sort_keys=True) + "\n")

    manifest = {
        "schema_version": "nanojev-binance-refresh-v1",
        "purpose": "extend forward-ledger input; paper research only, no orders",
        "source": "Binance public data archive (data.binance.vision)",
        "source_terms": SOURCE_TERMS,
        "licence": "CC BY-NC-SA 4.0 (non-commercial); research use (S4.1)",
        "point_in_time_caveat": "CURRENT SNAPSHOTS of history, not as-of vintages",
        "cohort_tail_per_symbol": tail,
        "refresh_range": {s: [wanted[s].isoformat(), yesterday.isoformat()]
                          for s in wanted},
        "records_written": len(records),
        "file_count": len(files),
        "missing": missing,
        "files": files,
        "output": str(out),
    }
    mpath = args.output_dir / "refresh_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")
    print(json.dumps({"status": "refreshed",
                      "records_written": len(records),
                      "missing": len(missing),
                      "output": str(out),
                      "manifest": str(mpath)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
