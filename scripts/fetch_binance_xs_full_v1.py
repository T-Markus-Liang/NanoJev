#!/usr/bin/env python3
"""Bounded FULL-HISTORY cross-sectional fetch of Binance public archive files (T104).

READ-ONLY PUBLIC ARCHIVE ONLY. This script performs plain HTTP GETs of static files from
data.binance.vision. It never authenticates, never calls a trading/REST order endpoint,
never opens a websocket, and never contacts a broker. Nothing here can place an order.

This is the BOUNDED full-history expansion approved for T104: the same 5 new symbols
as the T102 pilot x months 2023-01..2026-08 x TWO file kinds only:
  * ``klines`` (1d interval: close + volume -> dfh, momentum, vol features), and
  * ``fundingRate``.
``markPriceKlines`` / ``indexPriceKlines`` are deliberately NOT fetched: the approved
XS arms (funding-rank / dfh-rank) use close as the mark proxy, and the deviation is
quantified on the legacy symbols at build time. Plan size = 5 x 44 x 2 = 440 files
(hard cap 500), ~2 KB each (~1 MB total), written to ``data/binance_xs_v1/`` — a
separate directory from both the approved 5-symbol slice and the T102 pilot slice.

URL conventions verified against ``data/binance_vision_v1/fetch_manifest.json`` and
the T102 pilot manifest:
  klines (monthly, 1d interval):
    {MONTHLY_BASE}/klines/{SYM}/1d/{SYM}-1d-YYYY-MM.zip
  fundingRate (monthly):
    {MONTHLY_BASE}/fundingRate/{SYM}/{SYM}-fundingRate-YYYY-MM.zip

Honesty rules: HTTP 404s are recorded in the manifest (some symbols may list later
than 2023-01) and are NOT fatal. Repeated non-HTTP network failures abort the run
honestly (after 4 consecutive failures) rather than hammering a dead network; partial
results are still written to the manifest. Idempotent: existing files are served
from cache and manifest entries are deduplicated by url.

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

FULL_SYMBOLS = ("ADAUSDT", "DOGEUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT")
FIRST_MONTH = (2023, 1)
LAST_MONTH = (2026, 8)
FULL_KINDS = ("klines", "fundingRate")
MAX_FILES = 500  # hard bound for this expansion: plan is 5 x 44 x 2 = 440
MAX_CONSECUTIVE_NETWORK_FAILURES = 4  # abort honestly rather than hammer a dead network


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


def url_for(kind, symbol, month):
    if kind == "fundingRate":
        return f"{MONTHLY_BASE}/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"
    return f"{MONTHLY_BASE}/{kind}/{symbol}/1d/{symbol}-1d-{month}.zip"


def fetch(url, timeout=60):
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=pathlib.Path("data/binance_xs_v1"))
    parser.add_argument("--symbols", default=",".join(FULL_SYMBOLS),
                        help="comma-separated symbols (bounded T104 set by default)")
    parser.add_argument("--months", default=",".join(month_range(FIRST_MONTH, LAST_MONTH)),
                        help="comma-separated YYYY-MM list")
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
            raise SystemExit(f"kind not in the T104 bound: {kind} "
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
                target = args.output_dir / kind / symbol / url.rsplit("/", 1)[-1]
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
                entries.append({"symbol": symbol, "kind": kind, "month": month,
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
        "schema_version": "nanojev-binance-xs-fetch-v1",
        "purpose": "bounded full-history cross-sectional expansion (T104); offline "
                   "paper-trading research only; no orders, no broker, no trading API",
        "source": "Binance public data archive (data.binance.vision)",
        "source_terms": SOURCE_TERMS,
        "licence": "CC BY-NC-SA 4.0 (non-commercial). Research use permitted (S4.1); live "
                   "proprietary trading execution prohibited (S4.2).",
        "point_in_time_caveat": "CURRENT SNAPSHOTS of history, not as-of vintages. Binance "
                                "documents in-place file replacement, so these bytes are not "
                                "evidence of what was knowable at the time.",
        "bound": {"max_files": MAX_FILES, "planned_files": planned,
                  "note": "5 new symbols x 44 months (2023-01..2026-08) x 2 kinds "
                          "(klines 1d + fundingRate). markPriceKlines/indexPriceKlines "
                          "deliberately excluded — the approved XS arms use close as "
                          "the mark proxy; no 15-asset expansion approved"},
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
