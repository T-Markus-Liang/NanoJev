#!/usr/bin/env python3
"""Offline-safe incremental refresh of the perp_pit_xs_v1 cohort tail (T110).

Companion to ``refresh_binance_cohort_v1.py`` — that job only covers the 5
LEGACY symbols (BTC/ETH/SOL/BNB/XRP, mark+index basis records). The XS cohort
``data/perp_pit_xs_v1/records.jsonl`` added 5 more symbols in T104
(ADA/DOGE/DOT/LINK/LTC) whose refresh was never wired; this script is their
daily tail so ``financial_forward_ledger_v2.py`` can keep ranking a >=6 asset
universe past the cohort end (2026-08-30).

SCOPE: the 5 new symbols only, klines close + fundingRate — the XS feature
basis is the regular klines close used as the mark proxy (build_xs_v1.py
convention; no markPriceKlines for the new symbols). Writes PIT-lite records
to ``data/binance_xs_refresh_v1/records.jsonl`` (the cohort file itself is
never modified; the v2 ledger merges this file as a supplement deduped by id).

ARCHIVE SHAPE (verified 2026-09-24):
  * monthly klines/fundingRate zips exist only for CLOSED months — the
    current month 404s, so the tail is fetched via DAILY kline zips
    {DAILY_BASE}/klines/{SYM}/1d/{SYM}-1d-YYYY-MM-DD.zip
  * fundingRate is monthly-only (daily funding zips 404); the current
    month's file is absent until month close, so refreshed bars carry the
    ``close`` feature always and ``last_funding_rate`` /
    ``funding_interval_hours`` only when a published monthly file covers the
    bar's close_time. The v2 dfh sleeve does not consume funding.
  * fetch bound: (cohort_tail+1 .. yesterday) daily kline zips per symbol
    (default --days 45 cap) + one monthly funding zip per spanned month per
    symbol. At a fresh cohort tail this is ~5 funding zips + ~5*N tiny daily
    zips (~350 bytes each).

OFFLINE-SAFE: if data.binance.vision is unreachable (direct and via the
NANOJEV_PROXY egress), prints a clear message and exits 0 with no writes.

LAUNCHD WIRING: this script is appended to the forward-ledger daily chain in
``deploy/launchd/ai.nanojev.forward-ledger.plist`` ProgramArguments; the
installed copy under ~/Library/LaunchAgents is only refreshed by re-running
``deploy/launchd/install.sh`` (owner action).

READ-ONLY PUBLIC ARCHIVE ONLY — same licence posture as
fetch_binance_xs_full_v1.py: CC BY-NC-SA 4.0, research use, no orders, no
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
# The T104 expansion symbols — the 5 symbols NOT covered by
# refresh_binance_cohort_v1.py (which owns the legacy set).
DEFAULT_SYMBOLS = ("ADAUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT")
VENUE = "binance_um"
MS = 1_000_000
# Same local egress proxy convention as scripts/refresh_binance_cohort_v1.py.
DEFAULT_PROXY = os.environ.get("NANOJEV_PROXY", "http://127.0.0.1:7890")

DAILY_KLINE_URL = f"{DAILY_BASE}/klines/{{sym}}/1d/{{sym}}-1d-{{date}}.zip"
FUNDING_URL = f"{MONTHLY_BASE}/fundingRate/{{sym}}/{{sym}}-fundingRate-{{month}}.zip"


def openers():
    yield "direct", urllib.request.build_opener()
    if DEFAULT_PROXY:
        yield DEFAULT_PROXY, urllib.request.build_opener(
            urllib.request.ProxyHandler(
                {"http": DEFAULT_PROXY, "https": DEFAULT_PROXY}))


def fetch(url, timeout=30):
    """Return (status, payload, route) or raise urllib.error.URLError."""
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


def parse_daily_kline(payload):
    """Single daily bar from a 1d kline zip -> {close, close_time}."""
    for line in zip_rows(payload):
        parts = line.split(",")
        if parts[0] == "open_time" or len(parts) < 7:
            continue
        return {"close": float(parts[4]), "close_time": int(parts[6])}
    return None


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
                    default=pathlib.Path("data/perp_pit_xs_v1/records.jsonl"))
    ap.add_argument("--output-dir", type=pathlib.Path,
                    default=pathlib.Path("data/binance_xs_refresh_v1"))
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS),
                    help="the 5 T104-expansion symbols by default; the legacy "
                         "5 are owned by refresh_binance_cohort_v1.py")
    ap.add_argument("--days", type=int, default=45,
                    help="max days back to fetch when the cohort tail is old")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    tail = cohort_tail(args.cohort)
    today = dt.datetime.now(dt.timezone.utc).date()
    yesterday = today - dt.timedelta(days=1)

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

    # Connectivity probe on a file that must exist (first needed kline zip).
    first_sym = sorted(wanted)[0]
    probe_url = DAILY_KLINE_URL.format(
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
        # funding rows for the months spanned by the refresh range (monthly
        # files exist only for closed months; current-month 404s are normal).
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
            cursor += dt.timedelta(days=1)
            url = DAILY_KLINE_URL.format(sym=symbol, date=date_text)
            try:
                status, payload, via = fetch(url)
            except Exception as error:  # noqa: BLE001
                missing.append({"url": url,
                                "error": f"{type(error).__name__}"})
                continue
            if status != 200:
                missing.append({"url": url, "http_status": status})
                continue
            files.append({"url": url, "http_status": status, "route": via,
                          "bytes": len(payload),
                          "sha256": hashlib.sha256(payload).hexdigest()})
            bar = parse_daily_kline(payload)
            if bar is None or bar["close"] <= 0:
                missing.append({"url": url, "note": "no parseable daily bar"})
                continue
            close_time = bar["close_time"]
            decision_ns = close_time * MS + 1
            event_ns = close_time * MS
            avail_ns = event_ns + 1
            feats = {
                "close": feature(bar["close"], "venue_perp_klines",
                                 "perp-1d-close-as-mark-proxy-v1",
                                 event_ns, avail_ns),
            }
            last_funding = None
            for row in funding:
                if row["calc_time"] <= close_time:
                    last_funding = row
                else:
                    break
            if last_funding is not None:
                feats["last_funding_rate"] = feature(
                    last_funding["rate"], "venue_funding_rate_history",
                    "perp-funding-rate-settled-v1",
                    last_funding["calc_time"] * MS,
                    last_funding["calc_time"] * MS)
                feats["funding_interval_hours"] = feature(
                    last_funding["interval_hours"],
                    "venue_instrument_or_funding_config",
                    "perp-funding-interval-hours-v1",
                    last_funding["calc_time"] * MS,
                    last_funding["calc_time"] * MS)
            records.append({
                "schema_version": "nanojev-financial-pit-xs-v1",
                "id": f"{VENUE}:{symbol}:{date_text}",
                "asset_id": f"{symbol}-PERP",
                "venue": VENUE,
                "decision_ns": decision_ns,
                "universe_available_ns": event_ns,
                "features": feats,
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
        "schema_version": "nanojev-binance-xs-refresh-v1",
        "purpose": "extend the perp_pit_xs_v1 tail for the 5 T104-expansion "
                   "symbols so forward_ledger_v2 keeps a >=6 asset universe; "
                   "paper research only, no orders",
        "source": "Binance public data archive (data.binance.vision)",
        "source_terms": SOURCE_TERMS,
        "licence": "CC BY-NC-SA 4.0 (non-commercial); research use (S4.1)",
        "point_in_time_caveat": "CURRENT SNAPSHOTS of history, not as-of vintages",
        "archive_note": "monthly zips exist only for closed months; the tail "
                        "uses DAILY kline zips. fundingRate is monthly-only, "
                        "so bars in the open month carry close-only features.",
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
