#!/usr/bin/env python3
"""Bounded, provenance-recording fetch of Binance public archive files.

READ-ONLY PUBLIC ARCHIVE ONLY. This script performs plain HTTP GETs of static files from
data.binance.vision. It never authenticates, never calls a trading/REST order endpoint,
never opens a websocket, and never contacts a broker. Nothing here can place an order.

Licence: the Binance Vision Dataset Terms v1.0 put these datasets under CC BY-NC-SA 4.0
(non-commercial). Research use is permitted (S4.1); live proprietary trading execution and
automated commercial order generation are prohibited (S4.2). Downloaded files are kept in
the local ignored ``data/`` directory and are not redistributed.

Point-in-time caveat recorded in the manifest: these are CURRENT SNAPSHOTS of history, not
as-of vintages. Binance documents that archived files have been replaced in place, so a
file downloaded today is not evidence of what was knowable at the time.
"""
import argparse
import hashlib
import json
import pathlib
import urllib.error
import urllib.request

BASE = "https://data.binance.vision/data/futures/um/monthly"
SOURCE_TERMS = "https://data.binance.vision/Binance_Vision-Terms_of_Use.pdf"
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
KINDS = {
    "klines": "{sym}/1d/{sym}-1d-{month}.zip",
    "markPriceKlines": "{sym}/1d/{sym}-1d-{month}.zip",
    "indexPriceKlines": "{sym}/1d/{sym}-1d-{month}.zip",
    "fundingRate": "{sym}/{sym}-fundingRate-{month}.zip",
}


def url_for(kind, symbol, month):
    return f"{BASE}/{kind}/{KINDS[kind].format(sym=symbol, month=month)}"


def months(first, last):
    year, month = first
    end_year, end_month = last
    while (year, month) <= (end_year, end_month):
        yield f"{year:04d}-{month:02d}"
        month += 1
        if month == 13:
            year, month = year + 1, 1


def fetch(url, timeout=60):
    request = urllib.request.Request(url, headers={"User-Agent": "nanojev-research/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("data/binance_vision_v1"))
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--first-month", default="2023-01")
    parser.add_argument("--last-month", default="2026-08")
    parser.add_argument("--kinds", default=",".join(KINDS))
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    for kind in kinds:
        if kind not in KINDS:
            raise SystemExit(f"unknown kind: {kind}")
    first = tuple(int(x) for x in args.first_month.split("-"))
    last = tuple(int(x) for x in args.last_month.split("-"))

    entries, missing, downloaded, cached = [], [], 0, 0
    for kind in kinds:
        for symbol in symbols:
            for month in months(first, last):
                url = url_for(kind, symbol, month)
                target = args.output_dir / kind / symbol / url.rsplit("/", 1)[-1]
                if target.exists():
                    payload = target.read_bytes()
                    status, cached = 200, cached + 1
                else:
                    try:
                        status, payload = fetch(url)
                    except urllib.error.HTTPError as error:
                        missing.append({"url": url, "http_status": error.code})
                        continue
                    except Exception as error:  # noqa: BLE001 - network failure is recorded, not fatal
                        missing.append({"url": url, "error": f"{type(error).__name__}"})
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(payload)
                    downloaded += 1
                entries.append({"url": url, "local_path": str(target), "http_status": status,
                                "bytes": len(payload),
                                "sha256": hashlib.sha256(payload).hexdigest()})

    manifest = {
        "schema_version": "nanojev-binance-vision-fetch-v1",
        "purpose": "offline paper-trading research only; no orders, no broker, no trading API",
        "source": "Binance public data archive (data.binance.vision)",
        "source_terms": SOURCE_TERMS,
        "licence": "CC BY-NC-SA 4.0 (non-commercial). Research use permitted (S4.1); live "
                   "proprietary trading execution prohibited (S4.2).",
        "point_in_time_caveat": "CURRENT SNAPSHOTS of history, not as-of vintages. Binance "
                                "documents in-place file replacement, so these bytes are not "
                                "evidence of what was knowable at the time.",
        "symbols": symbols,
        "kinds": kinds,
        "first_month": args.first_month,
        "last_month": args.last_month,
        "downloaded": downloaded,
        "served_from_cache": cached,
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "missing": missing,
        "files": entries,
    }
    out = args.output_dir / "fetch_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in
                      ("downloaded", "served_from_cache", "file_count", "total_bytes")}, sort_keys=True))
    print(json.dumps({"missing_count": len(missing), "manifest": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
