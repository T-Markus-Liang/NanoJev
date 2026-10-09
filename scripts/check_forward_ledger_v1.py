#!/usr/bin/env python3
"""T90: health probe for the forward paper-trade ledger.

Read-only. Scans ``results/forward_ledger_v1.jsonl`` plus the Binance
refresh supplement to answer one question: is the ledger still stepping
forward on fresh daily bars?

Exit codes:
  0  healthy — newest ledger decision bar is within --max-age-hours of now
  1  unhealthy — ledger missing/empty, or newest bar is stale

Prints one JSON status object either way
(``status``: ``ok`` | ``stale`` | ``missing`` | ``empty``).
"""

import argparse
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "results/forward_ledger_v1.jsonl"
SUPPLEMENT = ROOT / "data/binance_refresh_v1/records.jsonl"
REFRESH_MANIFEST = ROOT / "data/binance_refresh_v1/refresh_manifest.json"
NS = 1_000_000_000
DEFAULT_MAX_AGE_HOURS = 48.0


def scan_jsonl(path):
    """(max_decision_ns, {asset: last record date}, n_lines, n_bad)."""
    max_ns = None
    last_date = {}
    n_lines = n_bad = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        n_lines += 1
        try:
            row = json.loads(raw)
            ns = int(row["decision_ns"])
        except (ValueError, KeyError, TypeError):
            n_bad += 1
            continue
        if max_ns is None or ns > max_ns:
            max_ns = ns
        rid = row.get("record_id") or row.get("id") or ""
        parts = rid.split(":")
        date = parts[-1] if len(parts) >= 2 else ""
        asset = row.get("asset_id") or (parts[-2] if len(parts) >= 2 else "?")
        if date > last_date.get(asset, ""):
            last_date[asset] = date
    return max_ns, last_date, n_lines, n_bad


def refresh_manifest_status(path):
    """Small status dict from the last refresh manifest, if present."""
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {"unreadable": True}
    return {
        "records_written": manifest.get("records_written"),
        "missing_files": len(manifest.get("missing") or []),
        "refresh_range": manifest.get("refresh_range"),
        "manifest_mtime_utc": dt.datetime.fromtimestamp(
            path.stat().st_mtime, dt.timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", type=Path, default=LEDGER)
    ap.add_argument("--supplement", type=Path, default=SUPPLEMENT)
    ap.add_argument("--manifest", type=Path, default=REFRESH_MANIFEST)
    ap.add_argument("--max-age-hours", type=float,
                    default=DEFAULT_MAX_AGE_HOURS,
                    help="ledger is stale when the newest bar is older "
                         "than this (default 48h; daily job + 1d slack)")
    args = ap.parse_args()

    now = dt.datetime.now(dt.timezone.utc)
    status = {"schema_version": "nanojev-forward-ledger-check-v1",
              "checked_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "ledger": str(args.ledger),
              "max_age_hours": args.max_age_hours}

    if not args.ledger.exists():
        status["status"] = "missing"
        status["note"] = "ledger file not found; forward ledger has never run"
        print(json.dumps(status, indent=2, sort_keys=True))
        return 1

    max_ns, last_date, n_lines, n_bad = scan_jsonl(args.ledger)
    if max_ns is None:
        status["status"] = "empty"
        status["note"] = "ledger file exists but holds no parseable bars"
        status["lines"] = n_lines
        print(json.dumps(status, indent=2, sort_keys=True))
        return 1

    latest_utc = dt.datetime.fromtimestamp(max_ns / NS, dt.timezone.utc)
    age_h = (now - latest_utc).total_seconds() / 3600.0
    status["latest_decision_utc"] = latest_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    status["latest_bar_date"] = max(last_date.values()) if last_date else None
    status["last_bar_per_asset"] = last_date
    status["age_hours"] = round(age_h, 2)
    status["lines"] = n_lines
    if n_bad:
        status["unparseable_lines"] = n_bad

    if args.supplement.exists():
        sup_ns, sup_dates, sup_n, _ = scan_jsonl(args.supplement)
        status["supplement"] = {
            "records": sup_n,
            "latest_bar_date": max(sup_dates.values()) if sup_dates else None,
            "ahead_of_ledger": bool(
                sup_ns is not None and sup_ns > max_ns),
        }
    else:
        status["supplement"] = {"records": 0, "latest_bar_date": None,
                                "ahead_of_ledger": False}
    manifest = refresh_manifest_status(args.manifest)
    if manifest:
        status["last_refresh"] = manifest

    if age_h <= args.max_age_hours:
        status["status"] = "ok"
        exit_code = 0
    else:
        status["status"] = "stale"
        status["note"] = ("newest ledger bar is older than "
                          f"{args.max_age_hours:g}h — the daily refresh+"
                          "ledger job is not landing (offline? launchd job "
                          "not installed? archive gap?)")
        exit_code = 1

    print(json.dumps(status, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
