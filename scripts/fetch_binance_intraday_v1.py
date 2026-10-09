#!/usr/bin/env python3
"""Bounded, provenance-recording fetch of Binance INTRADAY public archive files.

READ-ONLY PUBLIC ARCHIVE ONLY. This script performs plain HTTP GETs of static files from
data.binance.vision. It never authenticates, never calls a trading/REST order endpoint,
never opens a websocket, and never contacts a broker. Nothing here can place an order.

This is the deliberately SMALL intraday companion to ``fetch_binance_vision_v1.py``: a
bounded set of symbol-months (comma-separated ``--symbol`` / ``--month`` lists), four file
kinds per symbol-month. Its purpose is to unblock hourly lead-lag / funding-timing
research that the daily cohort cannot express. New file entries are APPENDED to the
existing fetch_manifest.json; prior entries are kept.

Licence: the Binance Vision Dataset Terms v1.0 put these datasets under CC BY-NC-SA 4.0
(non-commercial). Research use is permitted (S4.1); live proprietary trading execution and
automated commercial order generation are prohibited (S4.2). Downloaded files are kept in
the local ignored ``data/`` directory and are not redistributed.

Point-in-time caveat recorded in the manifest: these are CURRENT SNAPSHOTS of history, not
as-of vintages. Binance documents that archived files have been replaced in place, so a
file downloaded today is not evidence of what was knowable at the time.
"""
import argparse
import datetime as dt
import hashlib
import json
import pathlib
import time
import urllib.error
import urllib.request

MONTHLY_BASE = "https://data.binance.vision/data/futures/um/monthly"
SOURCE_TERMS = "https://data.binance.vision/Binance_Vision-Terms_of_Use.pdf"
UA = {"User-Agent": "nanojev-research/1.0"}
MAX_FILES = 4          # hard bound: never fetch more than this many files per symbol-month
MAX_MONTH_STEPBACK = 3 # if the auto-picked month 404s, step back at most this far
MAX_CONSECUTIVE_NETWORK_FAILURES = 4  # abort honestly rather than hammer a dead network


class AbortFetch(Exception):
    """Raised when repeated non-HTTP network failures make further fetching pointless."""


def most_recent_complete_month(today=None):
    """YYYY-MM of the last fully elapsed UTC month."""
    today = today or dt.datetime.now(dt.timezone.utc).date()
    first_of_month = today.replace(day=1)
    last_of_prev = first_of_month - dt.timedelta(days=1)
    return f"{last_of_prev.year:04d}-{last_of_prev.month:02d}"


def step_back_month(month_text, n):
    year, month = (int(x) for x in month_text.split("-"))
    month -= n
    while month < 1:
        month += 12
        year -= 1
    return f"{year:04d}-{month:02d}"


def url_for(kind, symbol, interval, month):
    if kind == "fundingRate":
        return f"{MONTHLY_BASE}/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"
    return f"{MONTHLY_BASE}/{kind}/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"


def fetch(url, timeout=60):
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read()


def plan_files(symbol, interval, month):
    """Ordered (slot, kind, url) candidates. The 'funding' slot has a fallback kind:
    if the monthly fundingRate zip is absent, premiumIndexKlines at the bar interval
    is tried instead (it carries mark/index data usable for funding research)."""
    return [
        ("kline", "klines", url_for("klines", symbol, interval, month)),
        ("mark_kline", "markPriceKlines", url_for("markPriceKlines", symbol, interval, month)),
        ("index_kline", "indexPriceKlines", url_for("indexPriceKlines", symbol, interval, month)),
        ("funding", "fundingRate", url_for("fundingRate", symbol, interval, month)),
    ]


def funding_fallback(symbol, interval, month):
    return ("funding", "premiumIndexKlines",
            url_for("premiumIndexKlines", symbol, interval, month))


def fetch_symbol_month(symbol, interval, month, output_dir, fetched_at, failures):
    """Fetch one symbol-month, stepping back if the month is not archived yet.

    ``failures`` is a one-element list counting CONSECUTIVE non-HTTP network errors
    across the whole run; hitting MAX_CONSECUTIVE_NETWORK_FAILURES raises AbortFetch
    rather than fabricating data or hammering a dead network.
    Returns (entries, missing, chosen_month).
    """
    missing = []
    for back in range(MAX_MONTH_STEPBACK + 1):
        candidate = month if back == 0 else step_back_month(month, back)
        if back:
            missing.append({"symbol": symbol,
                            "note": f"kline month {step_back_month(month, back - 1)} "
                                    f"not archived; stepping back to {candidate}"})
        plan = plan_files(symbol, interval, candidate)
        month_missing = []
        month_entries = []
        for slot, kind, url in plan[:MAX_FILES]:
            target = output_dir / kind / symbol / url.rsplit("/", 1)[-1]
            if target.exists():
                payload, status = target.read_bytes(), 200
            else:
                try:
                    status, payload = fetch(url)
                    failures[0] = 0
                    time.sleep(0.4)  # be polite: bounded sample, no burst
                except urllib.error.HTTPError as error:
                    failures[0] = 0  # a real HTTP answer: the network is fine
                    month_missing.append({"symbol": symbol, "url": url,
                                          "http_status": error.code,
                                          "slot": slot, "kind": kind})
                    continue
                except Exception as error:  # noqa: BLE001 - network failure is recorded, not fatal
                    failures[0] += 1
                    month_missing.append({"symbol": symbol, "url": url, "slot": slot,
                                          "kind": kind,
                                          "error": f"{type(error).__name__}: {str(error)[:160]}"})
                    if failures[0] >= MAX_CONSECUTIVE_NETWORK_FAILURES:
                        raise AbortFetch(f"{failures[0]} consecutive network failures")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            month_entries.append({"symbol": symbol, "slot": slot, "kind": kind, "url": url,
                                  "local_path": str(target), "http_status": status,
                                  "bytes": len(payload), "fetched_at": fetched_at,
                                  "sha256": hashlib.sha256(payload).hexdigest()})
        # Funding fallback: only tried if the fundingRate zip specifically 404'd.
        if any(m["slot"] == "funding" for m in month_missing) and \
                not any(e["slot"] == "funding" for e in month_entries):
            slot, kind, url = funding_fallback(symbol, interval, candidate)
            target = output_dir / kind / symbol / url.rsplit("/", 1)[-1]
            try:
                status, payload = fetch(url)
                failures[0] = 0
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                month_entries.append({"symbol": symbol, "slot": slot, "kind": kind,
                                      "url": url, "local_path": str(target),
                                      "http_status": status, "bytes": len(payload),
                                      "fetched_at": fetched_at,
                                      "sha256": hashlib.sha256(payload).hexdigest(),
                                      "note": "fallback: fundingRate monthly zip unavailable"})
            except urllib.error.HTTPError as error:
                failures[0] = 0
                month_missing.append({"symbol": symbol, "url": url, "slot": slot,
                                      "kind": kind, "http_status": error.code})
            except Exception as error:  # noqa: BLE001
                failures[0] += 1
                month_missing.append({"symbol": symbol, "url": url, "slot": slot,
                                      "kind": kind,
                                      "error": f"{type(error).__name__}: {str(error)[:160]}"})
                if failures[0] >= MAX_CONSECUTIVE_NETWORK_FAILURES:
                    raise AbortFetch(f"{failures[0]} consecutive network failures")
        missing.extend(month_missing)
        # The core kline is the hard requirement; if it 404'd the month is not archived yet.
        if any(e["slot"] == "kline" for e in month_entries):
            return month_entries, missing, candidate
        if not any(m["slot"] == "kline" and m.get("http_status") == 404
                   for m in month_missing):
            # network failure (not a 404) on the kline: do not walk back months
            return month_entries, missing, None
    missing.append({"symbol": symbol,
                    "note": f"no archived kline month found within "
                            f"{MAX_MONTH_STEPBACK} months before {month}"})
    return [], missing, None


def entry_symbol(entry):
    """Symbol for a manifest file entry (older entries have no 'symbol' key)."""
    return entry.get("symbol") or pathlib.PurePath(entry["local_path"]).parent.name


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=pathlib.Path("data/binance_intraday_v1"))
    parser.add_argument("--symbol", default="BTCUSDT",
                        help="comma-separated symbols, e.g. BTCUSDT,ETHUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--month", default=None,
                        help="comma-separated YYYY-MM list; defaults to the most recent "
                             "complete UTC month")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbol.split(",") if s.strip()]
    months = [m.strip() for m in
              (args.month or most_recent_complete_month()).split(",") if m.strip()]
    fetched_at = dt.datetime.now(dt.timezone.utc).isoformat()

    all_entries, all_missing, fetched_months = [], [], {}
    failures = [0]
    aborted = None
    for month in months:
        for symbol in symbols:
            try:
                entries, missing, chosen = fetch_symbol_month(
                    symbol, args.interval, month, args.output_dir, fetched_at, failures)
            except AbortFetch as error:
                aborted = f"aborted at {symbol} {month}: {error}"
                all_missing.append({"symbol": symbol, "note": aborted})
                break
            all_entries.extend(entries)
            all_missing.extend(missing)
            fetched_months.setdefault(symbol, {})[month] = chosen
        if aborted:
            break

    # Append to the existing manifest: prior file entries are kept, new ones appended
    # (deduplicated by url so re-runs stay idempotent).
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "fetch_manifest.json"
    prior_files, prior_missing = [], []
    if manifest_path.exists():
        try:
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            prior_files = prior.get("files", [])
            prior_missing = prior.get("missing", [])
        except Exception:  # noqa: BLE001 - a corrupt manifest is replaced, not fatal
            pass
    known_urls = {e["url"] for e in prior_files}
    files = prior_files + [e for e in all_entries if e["url"] not in known_urls]
    all_symbols = sorted({entry_symbol(e) for e in files})

    manifest = {
        "schema_version": "nanojev-binance-intraday-fetch-v1",
        "purpose": "offline paper-trading research only; no orders, no broker, no trading API",
        "source": "Binance public data archive (data.binance.vision)",
        "source_terms": SOURCE_TERMS,
        "licence": "CC BY-NC-SA 4.0 (non-commercial). Research use permitted (S4.1); live "
                   "proprietary trading execution prohibited (S4.2).",
        "point_in_time_caveat": "CURRENT SNAPSHOTS of history, not as-of vintages. Binance "
                                "documents in-place file replacement, so these bytes are not "
                                "evidence of what was knowable at the time.",
        "symbols": all_symbols,
        "interval": args.interval,
        "requested_months": months,
        "fetched_months": fetched_months,
        "max_files_bound": MAX_FILES,
        "aborted": aborted,
        "file_count": len(files),
        "total_bytes": sum(e["bytes"] for e in files),
        "fetched_at": fetched_at,
        "missing": prior_missing + all_missing,
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    have_klines = {entry_symbol(e) for e in files if e["slot"] == "kline"}
    ok = aborted is None and all(symbol in have_klines for symbol in symbols)
    print(json.dumps({"ok": ok, "aborted": aborted,
                      "fetched_months": fetched_months,
                      "new_files": len(files) - len(prior_files),
                      "file_count": len(files),
                      "total_bytes": manifest["total_bytes"],
                      "missing": len(all_missing),
                      "manifest": str(manifest_path)}, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
