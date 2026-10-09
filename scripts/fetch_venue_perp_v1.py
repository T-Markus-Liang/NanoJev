#!/usr/bin/env python3
"""Bounded, provenance-recording fetch of REAL perpetual-contract data from venue APIs.

READ-ONLY PUBLIC MARKET DATA ONLY. This script performs plain HTTPS GET/POST calls to
public market-data endpoints. It never authenticates, never sends an API key, never calls
an order/position/account endpoint, never opens a private websocket, and never contacts a
broker. Nothing here can place an order or read an account.

Licence status recorded in the manifest: Binance Vision terms were read in full (CC BY-NC-SA
4.0, non-commercial, live execution prohibited by S4.2). Bybit / Aster / Hyperliquid terms
pages were NOT reachable from this environment, so their permissions are recorded as
UNVERIFIED. The human project owner has explicitly authorised connecting these data sources
for non-commercial research; that authorisation is a project decision, not a legal
determination, and commercial or live-execution use remains unlicensed and out of scope.

Venue hosts: api.bybit.com is DNS-poisoned in this environment (resolves to an unrelated
Brazilian address). The official regional mirror api.bybit.nl returns real API data and is
used instead; the DNS observation is recorded in the manifest. Aster's API host
(fapi.asterdex.com) is unreachable on every probed variant, so Aster cannot be connected
from here.
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

# The OFFICIAL host works once traffic goes through the egress proxy; the earlier regional
# mirror (api.bybit.nl) is kept only as a documented fallback.
BYBIT_BASE = "https://api.bybit.com"
BYBIT_MIRROR_FALLBACK = "https://api.bybit.nl"
HYPERLIQUID_INFO = "https://api.hyperliquid.xyz/info"
ASTER_BASE = "https://fapi.asterdex.com"
DEFAULT_BYBIT = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
DEFAULT_HYPERLIQUID = ("BTC", "ETH", "SOL", "BNB", "XRP")
DEFAULT_ASTER = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
DAY_MS = 86_400_000
UA = {"User-Agent": "nanojev-research/1.0", "Accept": "application/json"}

# Local egress proxy. Needed because this environment DNS-poisons api.bybit.com and drops
# direct connections to fapi.asterdex.com; routing through the proxy reaches both. Override
# with --proxy or NANOJEV_PROXY, or pass --proxy "" to force direct connections.
DEFAULT_PROXY = os.environ.get("NANOJEV_PROXY", "http://127.0.0.1:7890")
PROXY = DEFAULT_PROXY


def _opener():
    if not PROXY:
        return urllib.request.build_opener()
    return urllib.request.build_opener(urllib.request.ProxyHandler(
        {"http": PROXY, "https": PROXY}))


def ms(date_text):
    return int(dt.datetime.strptime(date_text, "%Y-%m-%d")
               .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


def http_json(url, payload=None, timeout=45):
    data = None if payload is None else json.dumps(payload).encode()
    headers = dict(UA)
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with _opener().open(request, timeout=timeout) as response:
        return response.status, json.loads(response.read())


# --------------------------------------------------------------------------- bybit

def bybit_get(path):
    status, body = http_json(f"{BYBIT_BASE}{path}")
    if status != 200 or body.get("retCode") != 0:
        raise RuntimeError(f"bybit error {status} {body.get('retMsg')}")
    return body["result"]


def bybit_series(symbol, series, start_ms, end_ms, interval="D", page_param="start"):
    """Bybit public kline families.

    Bybit returns the MOST RECENT ``limit`` rows inside the requested range, so paging
    forward by ``start`` silently truncates history at one page. Page BACKWARD by moving
    ``end`` to just before the oldest row received, keeping ``start`` fixed.
    """
    path_by_series = {
        "kline": "/v5/market/kline",
        "mark": "/v5/market/mark-price-kline",
        "index": "/v5/market/index-price-kline",
    }
    rows, cursor_end = [], end_ms
    while cursor_end > start_ms:
        query = (f"{path_by_series[series]}?category=linear&symbol={symbol}"
                 f"&interval={interval}&start={start_ms}&end={cursor_end}&limit=1000")
        result = bybit_get(query)
        batch = result.get("list") or []
        if not batch:
            break
        rows.extend(batch)
        oldest = min(int(row[0]) for row in batch)
        if oldest <= start_ms or oldest >= cursor_end:
            break
        cursor_end = oldest - 1
        time.sleep(0.12)
    deduped = {int(row[0]): row for row in rows}
    return [deduped[key] for key in sorted(deduped)]


def bybit_funding(symbol, start_ms, end_ms):
    """Funding history, paged BACKWARD for the same reason as the kline families."""
    rows, cursor_end = [], end_ms
    while cursor_end > start_ms:
        result = bybit_get(f"/v5/market/funding/history?category=linear&symbol={symbol}"
                           f"&startTime={start_ms}&endTime={cursor_end}&limit=200")
        batch = result.get("list") or []
        if not batch:
            break
        rows.extend(batch)
        oldest = min(int(row["fundingRateTimestamp"]) for row in batch)
        if oldest <= start_ms or oldest >= cursor_end:
            break
        cursor_end = oldest - 1
        time.sleep(0.12)
    deduped = {int(row["fundingRateTimestamp"]): row for row in rows}
    return [deduped[key] for key in sorted(deduped)]


def bybit_open_interest(symbol, start_ms, end_ms):
    """Open interest, paged BACKWARD for the same reason as the kline families."""
    rows, cursor_end = [], end_ms
    while cursor_end > start_ms:
        result = bybit_get(f"/v5/market/open-interest?category=linear&symbol={symbol}"
                           f"&intervalTime=1d&startTime={start_ms}&endTime={cursor_end}&limit=200")
        batch = result.get("list") or []
        if not batch:
            break
        rows.extend(batch)
        oldest = min(int(row["timestamp"]) for row in batch)
        if oldest <= start_ms or oldest >= cursor_end:
            break
        cursor_end = oldest - 1
        time.sleep(0.12)
    deduped = {int(row["timestamp"]): row for row in rows}
    return [deduped[key] for key in sorted(deduped)]


# ---------------------------------------------------------------------- hyperliquid

def hl_info(payload):
    status, body = http_json(HYPERLIQUID_INFO, payload)
    if status != 200:
        raise RuntimeError(f"hyperliquid error {status}")
    return body


def hl_funding_all(coin, start_ms, end_ms):
    """Hyperliquid funding is hourly and capped per response, so page on time."""
    rows, cursor = [], start_ms
    while cursor < end_ms:
        batch = hl_info({"type": "fundingHistory", "coin": coin,
                         "startTime": cursor, "endTime": end_ms})
        if not batch:
            break
        rows.extend(batch)
        newest = max(int(row["time"]) for row in batch)
        if newest < cursor:
            break
        cursor = newest + 1
        time.sleep(0.15)
    rows.sort(key=lambda row: int(row["time"]))
    return rows


def hl_candles_all(coin, start_ms, end_ms):
    """candleSnapshot is capped at the most recent 5000 candles; page to be safe."""
    rows, cursor = [], start_ms
    while cursor < end_ms:
        batch = hl_info({"type": "candleSnapshot", "req": {
            "coin": coin, "interval": "1d", "startTime": cursor, "endTime": end_ms}})
        if not batch:
            break
        rows.extend(batch)
        newest = max(int(row["t"]) for row in batch)
        if newest < cursor:
            break
        cursor = newest + 1
        time.sleep(0.15)
    deduped = {int(row["t"]): row for row in rows}
    return [deduped[key] for key in sorted(deduped)]


# ---------------------------------------------------------------------------- aster

def aster_get(path):
    status, body = http_json(f"{ASTER_BASE}{path}")
    if status != 200:
        raise RuntimeError(f"aster error {status}")
    return body


def aster_series(symbol, series, start_ms, end_ms, interval="1d"):
    """Binance-shaped kline families. These return rows oldest-first from ``startTime``,
    so paging FORWARD by the last open time is correct here (unlike Bybit)."""
    path_by_series = {
        "kline": "/fapi/v1/klines",
        "mark": "/fapi/v1/markPriceKlines",
        "index": "/fapi/v1/indexPriceKlines",
    }
    key = "pair" if series == "index" else "symbol"
    rows, cursor = [], start_ms
    while cursor < end_ms:
        query = (f"{path_by_series[series]}?{key}={symbol}&interval={interval}"
                 f"&startTime={cursor}&endTime={end_ms}&limit=1500")
        batch = aster_get(query)
        if not batch:
            break
        rows.extend(batch)
        newest = max(int(row[0]) for row in batch)
        if newest < cursor:
            break
        cursor = newest + 1
        time.sleep(0.12)
    deduped = {int(row[0]): row for row in rows}
    return [deduped[k] for k in sorted(deduped)]


def aster_funding(symbol, start_ms, end_ms):
    """Settled funding. Aster rows carry fundingTime and fundingRate but NO interval field,
    so the interval must come from fundingRateConfig and is recorded separately."""
    rows, cursor = [], start_ms
    while cursor < end_ms:
        batch = aster_get(f"/fapi/v1/fundingRate?symbol={symbol}&startTime={cursor}"
                          f"&endTime={end_ms}&limit=1000")
        if not batch:
            break
        rows.extend(batch)
        newest = max(int(row["fundingTime"]) for row in batch)
        if newest < cursor:
            break
        cursor = newest + 1
        time.sleep(0.12)
    deduped = {int(row["fundingTime"]): row for row in rows}
    return [deduped[k] for k in sorted(deduped)]


# ---------------------------------------------------------------------------- fetch

def collect(venue, symbols, start_ms, end_ms):
    """Return {symbol: {series_name: payload}} plus a per-series error list."""
    series, errors = {}, []
    for symbol in symbols:
        entry = {}
        if venue == "bybit":
            jobs = (("kline", lambda s=symbol: bybit_series(s, "kline", start_ms, end_ms)),
                    ("mark", lambda s=symbol: bybit_series(s, "mark", start_ms, end_ms)),
                    ("index", lambda s=symbol: bybit_series(s, "index", start_ms, end_ms)),
                    ("funding", lambda s=symbol: bybit_funding(s, start_ms, end_ms)),
                    ("open_interest", lambda s=symbol: bybit_open_interest(s, start_ms, end_ms)),
                    ("instruments", lambda s=symbol: bybit_get(
                        f"/v5/market/instruments-info?category=linear&symbol={s}")["list"]))
        elif venue == "aster":
            # Aster exposes NO open-interest endpoint at all (recorded as unavailable, not
            # as an error), and its funding rows carry no interval field.
            jobs = (("kline", lambda s=symbol: aster_series(s, "kline", start_ms, end_ms)),
                    ("mark", lambda s=symbol: aster_series(s, "mark", start_ms, end_ms)),
                    ("index", lambda s=symbol: aster_series(s, "index", start_ms, end_ms)),
                    ("funding", lambda s=symbol: aster_funding(s, start_ms, end_ms)),
                    ("funding_info", lambda s=symbol: aster_get(
                        f"/fapi/v1/fundingInfo?symbol={s}")),
                    ("premium_index", lambda s=symbol: aster_get(
                        f"/fapi/v1/premiumIndex?symbol={s}")),
                    ("instruments", lambda s=symbol: [
                        item for item in aster_get("/fapi/v1/exchangeInfo")["symbols"]
                        if item.get("symbol") == s]))
        else:
            jobs = (("candles", lambda s=symbol: hl_candles_all(s, start_ms, end_ms)),
                    ("funding", lambda s=symbol: hl_funding_all(s, start_ms, end_ms)),
                    ("asset_ctxs", lambda s=symbol: hl_info({"type": "metaAndAssetCtxs"})))
        for name, job in jobs:
            try:
                entry[name] = job()
            except Exception as error:  # noqa: BLE001 - a failed series is recorded, not fatal
                errors.append({"symbol": symbol, "series": name,
                               "error": f"{type(error).__name__}: {str(error)[:160]}"})
            time.sleep(0.15)
        series[symbol] = entry
    return series, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--venue", choices=("bybit", "hyperliquid", "aster"), required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("data/venue_perp_v1"))
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-09-19")
    parser.add_argument("--proxy", default=None,
                        help="egress proxy; defaults to NANOJEV_PROXY or http://127.0.0.1:7890. "
                             "Pass an empty string to force direct connections.")
    args = parser.parse_args()

    global PROXY
    if args.proxy is not None:
        PROXY = args.proxy

    defaults = {"bybit": DEFAULT_BYBIT, "aster": DEFAULT_ASTER,
                "hyperliquid": DEFAULT_HYPERLIQUID}[args.venue]
    symbols = ([s.strip() for s in args.symbols.split(",") if s.strip()]
               if args.symbols else list(defaults))
    start_ms, end_ms = ms(args.start), ms(args.end)
    series, errors = collect(args.venue, symbols, start_ms, end_ms)

    target = args.output_dir / args.venue
    target.mkdir(parents=True, exist_ok=True)
    files = []
    for symbol, payloads in series.items():
        for name, payload in payloads.items():
            path = target / f"{symbol}.{name}.json"
            blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            path.write_bytes(blob)
            files.append({"symbol": symbol, "series": name, "local_path": str(path),
                          "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
                          "records": len(payload) if isinstance(payload, list) else None})

    manifest = {
        "schema_version": "nanojev-venue-perp-fetch-v1",
        "venue": args.venue,
        "purpose": "offline paper-trading research only; no orders, no account access, no API key",
        "base_url": BYBIT_BASE if args.venue == "bybit" else HYPERLIQUID_INFO,
        "requested_base": "https://api.bybit.com" if args.venue == "bybit" else HYPERLIQUID_INFO,
        "host_note": ("api.bybit.com is DNS-poisoned for DIRECT connections in this environment "
                      "(resolves to an unrelated address and never connects); through the egress "
                      "proxy the official host serves the v5 API. api.bybit.nl is a documented "
                      "fallback." if args.venue == "bybit" else "direct"),
        "symbols": symbols, "start": args.start, "end": args.end,
        "egress_proxy": PROXY or "direct",
        "egress_note": ("api.bybit.com is DNS-poisoned in this environment and "
                        "fapi.asterdex.com refuses direct connections; routing through the "
                        "local egress proxy reaches both. Recorded so the fetch is repeatable."
                        if PROXY else "direct connections"),
        "licence_status": ("UNVERIFIED from this environment. The human project owner authorised "
                           "connecting these sources for non-commercial research; that is a project "
                           "decision, not a legal determination. Commercial or live-execution use "
                           "remains unlicensed."),
        "point_in_time_caveat": ("Vendor APIs return the CURRENT view of history and can be revised. "
                                 "This is not an as-of vintage and does not prove what was knowable "
                                 "at the time."),
        "file_count": len(files), "total_bytes": sum(f["bytes"] for f in files),
        "errors": errors, "files": files,
    }
    out = target / "fetch_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"venue": args.venue, "file_count": manifest["file_count"],
                      "total_bytes": manifest["total_bytes"], "errors": len(errors),
                      "manifest": str(out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
