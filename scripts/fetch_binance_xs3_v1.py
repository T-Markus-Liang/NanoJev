#!/usr/bin/env python3
"""Bounded 2026 TAIL fetch for the ~20 NEW live-universe-v3 symbols (T151).

READ-ONLY PUBLIC ARCHIVE ONLY. Same posture as ``fetch_binance_xs2_v1.py``
(T129) / ``fetch_binance_xs_full_v1.py``: plain HTTP GETs of static files from
data.binance.vision — no auth, no trading endpoint, no websocket, no broker.
Nothing here can place an order.

This is the T151 universe expansion: the 20 NEW symbols selected in
``research/live_universe_v3.json`` (the most recent live rc_futures_v1
listings, chosen to revive the T148/v5 young pool). Their history through
2025-12 comes from data/rc_futures_v1 (mega archive); this fetch supplies the
2026 tail so ``build_xs_v3.py`` can extend them to the cohort end and
``refresh_binance_xs3_v1.py`` can keep them alive:

  * ``klines`` (1d interval) and ``fundingRate`` — MONTHLY zips, months
    2026-01 .. the current month inclusive. Monthly zips exist only for
    CLOSED months, so the open month 404s and is recorded in ``missing``
    (its bars arrive via daily zips in the xs3 refresh job instead).
    Plan size = 20 x 9 x 2 = 360 files (hard cap 450).

Honesty rules (unchanged): HTTP 404s are recorded in the manifest and are NOT
fatal (some symbols may be delisted in 2026 — partial coverage is reported);
repeated non-HTTP network failures abort the run honestly after 4 consecutive
failures. Idempotent: existing files are served from cache and manifest
entries are deduplicated by url.

Licence: CC BY-NC-SA 4.0 per the Binance Vision Dataset Terms v1.0
(non-commercial, research use S4.1). Files stay in the local ignored
``data/`` directory and are not redistributed. These are CURRENT SNAPSHOTS of
history, not as-of vintages.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import time
import urllib.error
import urllib.request

MONTHLY_BASE = "https://data.binance.vision/data/futures/um/monthly"
SOURCE_TERMS = "https://data.binance.vision/Binance_Vision-Terms_of_Use.pdf"
UA = {"User-Agent": "nanojev-research/1.0"}
UNIVERSE_FILE = pathlib.Path("research/live_universe_v3.json")
# Same local egress proxy convention as scripts/refresh_binance_cohort_v1.py —
# the direct route intermittently dies with SSL EOF on this host.
DEFAULT_PROXY = os.environ.get("NANOJEV_PROXY", "http://127.0.0.1:7890")

# Universe symbol -> venue archive ticker where they differ (carried over from
# xs2; none of the 20 xs3 symbols are 1000- tickers, verified via fetch 404s).
VENUE_ALIASES = {"PEPEUSDT": "1000PEPEUSDT",
                 "SHIBUSDT": "1000SHIBUSDT",
                 "BONKUSDT": "1000BONKUSDT"}

# The 20 NEW symbols of live_universe_v3 (xs_v2's 30 are already covered by
# the vision/xs_v1/xs2 archives + their refresh jobs).
XS3_SYMBOLS = ("POWERUSDT", "BEATUSDT", "MMTUSDT", "RIVERUSDT", "GIGGLEUSDT",
               "4USDT", "EVAAUSDT", "LIGHTUSDT", "BLESSUSDT", "ASTERUSDT",
               "0GUSDT", "STBLUSDT", "AVNTUSDT", "LINEAUSDT", "WLFIUSDT",
               "XPLUSDT", "PROVEUSDT", "ERAUSDT", "CUSDT", "MUSDT")
FIRST_MONTH = (2026, 1)          # mega cohort covers through 2025-12-31
FULL_KINDS = ("klines", "fundingRate")
MAX_FILES = 450                  # bound: 20 syms x 9 months x 2 kinds = 360
MAX_CONSECUTIVE_NETWORK_FAILURES = 4


class AbortFetch(Exception):
    """Raised when repeated non-HTTP network failures make further fetching pointless."""


def month_range(first, last):
    """Inclusive YYYY-MM list from first=(y,m) to last=(y,m)."""
    y, m = first
    months = []
    while (y, m) <= last:
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return months


def default_months():
    """2026-01 .. current UTC month inclusive. The current month's monthly
    zips 404 until it closes (recorded in missing, not fatal)."""
    today = dt.datetime.now(dt.timezone.utc)
    return month_range(FIRST_MONTH, (today.year, today.month))


def url_for(kind, symbol, month):
    venue_sym = VENUE_ALIASES.get(symbol, symbol)
    if kind == "fundingRate":
        return (f"{MONTHLY_BASE}/fundingRate/{venue_sym}/"
                f"{venue_sym}-fundingRate-{month}.zip")
    return (f"{MONTHLY_BASE}/{kind}/{venue_sym}/1d/"
            f"{venue_sym}-1d-{month}.zip")


def openers():
    yield "direct", urllib.request.build_opener()
    if DEFAULT_PROXY:
        yield DEFAULT_PROXY, urllib.request.build_opener(
            urllib.request.ProxyHandler(
                {"http": DEFAULT_PROXY, "https": DEFAULT_PROXY}))


def fetch(url, timeout=60):
    """GET url direct first, then via the local egress proxy. Raises the last
    error if every route fails; HTTP errors propagate as HTTPError."""
    request = urllib.request.Request(url, headers=UA)
    last_error = None
    for _name, opener in openers():
        try:
            with opener.open(request, timeout=timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError:
            raise  # a real HTTP answer: do not retry the same status
        except Exception as error:  # noqa: BLE001 - try the next route
            last_error = error
    raise urllib.error.URLError(f"all routes failed: {last_error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=pathlib.Path("data/binance_xs3_v1"))
    parser.add_argument("--symbols", default=",".join(XS3_SYMBOLS),
                        help="comma-separated symbols (bounded T151 set by default: "
                             "the 20 new live_universe_v3 members)")
    parser.add_argument("--months", default=",".join(default_months()),
                        help="comma-separated YYYY-MM list (default 2026-01..current "
                             "month; open-month monthly zips 404 honestly)")
    parser.add_argument("--kinds", default=",".join(FULL_KINDS),
                        help="bounded: klines (1d) and fundingRate only")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit without fetching")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    months = [m.strip() for m in args.months.split(",") if m.strip()]
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    for kind in kinds:
        if kind not in FULL_KINDS:
            raise SystemExit(f"kind not in the T151 bound: {kind} "
                             f"(only {', '.join(FULL_KINDS)} are approved)")
    planned = len(symbols) * len(months) * len(kinds)
    print(json.dumps({"planned_files": planned, "hard_cap": MAX_FILES,
                      "symbols": len(symbols), "months": len(months),
                      "kinds": kinds,
                      "span": f"{months[0]}..{months[-1]}" if months else None},
                     sort_keys=True))
    if planned > MAX_FILES:
        raise SystemExit(f"bound exceeded: {planned} files planned > {MAX_FILES} approved "
                         "for this expansion; a larger fetch needs explicit approval")
    if args.dry_run:
        return 0

    fetched_at = dt.datetime.now(dt.timezone.utc).isoformat()
    entries, missing = [], []
    downloaded, cached = 0, 0
    failures = [0]
    aborted = None

    for kind in kinds:
        if aborted:
            break
        for symbol in symbols:
            if aborted:
                break
            for month in months:
                url = url_for(kind, symbol, month)
                venue_sym = VENUE_ALIASES.get(symbol, symbol)
                target = (args.output_dir / kind / venue_sym
                          / url.rsplit("/", 1)[-1])
                if target.exists():
                    payload, status = target.read_bytes(), 200
                    cached += 1
                else:
                    try:
                        status, payload = fetch(url)
                        failures[0] = 0
                        time.sleep(0.4)  # be polite: bounded sample, no burst
                    except urllib.error.HTTPError as error:
                        failures[0] = 0  # a real HTTP answer: the network is fine
                        missing.append({"symbol": symbol, "kind": kind, "month": month,
                                        "url": url, "http_status": error.code})
                        continue
                    except Exception as error:  # noqa: BLE001 - recorded, not fatal
                        failures[0] += 1
                        missing.append({"symbol": symbol, "kind": kind, "month": month,
                                        "url": url,
                                        "error": f"{type(error).__name__}: {str(error)[:160]}"})
                        if failures[0] >= MAX_CONSECUTIVE_NETWORK_FAILURES:
                            aborted = (f"aborted at {symbol} {kind} {month}: "
                                       f"{failures[0]} consecutive network failures")
                            missing.append({"note": aborted})
                            break
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(payload)
                    downloaded += 1
                entries.append({"symbol": symbol, "venue_symbol": venue_sym,
                                "kind": kind, "month": month,
                                "url": url, "local_path": str(target),
                                "http_status": status, "bytes": len(payload),
                                "fetched_at": fetched_at,
                                "sha256": hashlib.sha256(payload).hexdigest()})

    # Append to the existing manifest (idempotent re-runs, deduplicated by url).
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
    files = prior_files + [e for e in entries if e["url"] not in known_urls]

    symbols_with_klines = sorted({e["symbol"] for e in files if e["kind"] == "klines"})
    manifest = {
        "schema_version": "nanojev-binance-xs3-fetch-v1",
        "purpose": "bounded 2026 tail fetch for the 20 NEW live_universe_v3 "
                   "symbols (T151 — recent listings to revive the v5 young "
                   "pool); history through 2025-12 comes from "
                   "data/rc_futures_v1; offline paper-trading research only; "
                   "no orders, no broker, no trading API",
        "source": "Binance public data archive (data.binance.vision)",
        "source_terms": SOURCE_TERMS,
        "licence": "CC BY-NC-SA 4.0 (non-commercial). Research use permitted (S4.1); "
                   "live proprietary trading execution prohibited (S4.2).",
        "point_in_time_caveat": "CURRENT SNAPSHOTS of history, not as-of vintages. "
                                "Binance documents in-place file replacement, so "
                                "these bytes are not evidence of what was knowable "
                                "at the time.",
        "bound": {"max_files": MAX_FILES, "planned_files": planned,
                  "note": "20 new symbols x 9 months (2026-01..current) x 2 kinds "
                          "(klines 1d + fundingRate). The open month's monthly "
                          "zips 404 by design — its bars come from daily zips via "
                          "refresh_binance_xs3_v1.py"},
        "universe_file": str(UNIVERSE_FILE),
        "venue_aliases": VENUE_ALIASES,
        "symbols": symbols,
        "months": months,
        "kinds": kinds,
        "symbols_with_klines": symbols_with_klines,
        "aborted": aborted,
        "downloaded": downloaded,
        "served_from_cache": cached,
        "file_count": len(files),
        "total_bytes": sum(e["bytes"] for e in files),
        "fetched_at": fetched_at,
        "missing": prior_missing + missing,
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    ok = aborted is None and len(symbols_with_klines) >= len(symbols)
    print(json.dumps({"ok": ok, "aborted": aborted,
                      "symbols_with_klines": symbols_with_klines,
                      "new_files": len(files) - len(prior_files),
                      "file_count": len(files),
                      "total_bytes": manifest["total_bytes"],
                      "missing": len(missing),
                      "manifest": str(manifest_path)}, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
