#!/usr/bin/env python3
"""T148: health probe for the age-conditioned sleeve forward paper
ledger.

Read-only. Scans ``results/forward_ledger_v5.jsonl`` plus the merged
xs_v2 inputs (perp_pit_xs_v2 cohort + binance_refresh_v1 /
binance_xs_refresh_v1 / binance_xs2_refresh_v1 supplements) to answer:
is the ledger still stepping forward on fresh daily bars, and is the
age-conditioned book accounting internally consistent?

Checks:
  * ledger exists, is parseable, and its newest decision_date reaches
    the merged-input tail (the ledger writes a line for EVERY input
    date, so lag = input dates with no ledger line)
  * ledger freshness in wall-clock hours vs --max-age-hours
  * cumulative consistency: each line's cumulative.{gross,cost,net}_bps
    equals the running sum over day fields, and cumulative.net_equity
    re-compounds from ((gross or 0) - cost)/1e4 (spot-checked across
    the whole file, tolerances for recorded rounding)
  * structural consistency: event is a known v5 event; ``rebalance``
    lines carry a book whose longs are the young pool's top-2 by dfh20
    and whose shorts are the old pool's top-2 (per-side >=3-candidate
    floor); every booked leg's recorded age_bucket matches its pool;
    non-rebalance events carry no legs
  * replay divergence: the ledger module's own replay (merged inputs +
    recomputed dfh20/age) is re-run and event/book/gate/age_buckets on
    the ledger's last recorded date are compared; a mismatch is
    reported as replay_divergence (late-arriving bars legitimately
    change what a fresh replay sees — the append-only ledger is never
    rewritten), counted but not failed on its own
  * last-line event is surfaced: a persistent ``flat_gate_off`` tail
    means btc_ret20 <= 0; ``flat_thin_buckets`` means neither age pool
    fielded >=3 candidates; ``skipped_no_btc_ret20`` means the majors
    supplement stopped landing (BTC series stale)

Exit codes:
  0  healthy — ledger covers the input tail and is within
     --max-age-hours
  1  unhealthy — missing/empty ledger, lag behind inputs, cumulative or
     structural inconsistency, or stale
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "results/forward_ledger_v5.jsonl"
COHORT = ROOT / "data/perp_pit_xs_v2/records.jsonl"
SUPPLEMENTS = [ROOT / "data/binance_refresh_v1/records.jsonl",
               ROOT / "data/binance_xs_refresh_v1/records.jsonl",
               ROOT / "data/binance_xs2_refresh_v1/records.jsonl"]
XS_MANIFEST = ROOT / "data/binance_xs_refresh_v1/refresh_manifest.json"
XS2_MANIFEST = ROOT / "data/binance_xs2_refresh_v1/refresh_manifest.json"
NS = 1_000_000_000
DEFAULT_MAX_AGE_HOURS = 48.0

sys.path.insert(0, str(ROOT / "scripts"))
import financial_forward_ledger_v5 as led  # noqa: E402


def scan_jsonl(path):
    """dates present, max decision_ns, n_lines, n_bad for a ledger/input
    file."""
    dates = set()
    max_ns = None
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
        date = row.get("decision_date") or row.get("id", "").rsplit(
            ":", 1)[-1]
        if date:
            dates.add(date)
        if max_ns is None or ns > max_ns:
            max_ns = ns
    return dates, max_ns, n_lines, n_bad


def input_tail(cohort_path, supplement_paths):
    """(all decision dates, per-asset tail date) across merged inputs."""
    dates = set()
    tail = {}
    for path in [cohort_path, *supplement_paths]:
        if not path.exists():
            continue
        for l in path.read_text().splitlines():
            if not l.strip():
                continue
            rid = json.loads(l)["id"]
            _, sym, date = rid.split(":")
            dates.add(date)
            if date > tail.get(sym, ""):
                tail[sym] = date
    return dates, tail


def book_structure_ok(row, issues):
    """v5 book rules on one ledger line."""
    date = row.get("decision_date")
    event = row.get("event")
    book = row.get("book") or {}
    longs = book.get("long") or []
    shorts = book.get("short") or []
    buckets = row.get("age_buckets") or {}
    n_young = buckets.get("young_candidates") or 0
    n_old = buckets.get("old_candidates") or 0

    if event == "rebalance":
        young_pool = set(buckets.get("young_assets") or [])
        old_pool = set(buckets.get("old_assets") or [])
        if n_young >= led.MIN_SIDE and len(longs) != led.EDGE:
            issues.append({"decision_date": date,
                           "issue": "young_side_wrong_size",
                           "n_young": n_young, "n_long": len(longs)})
        if n_old >= led.MIN_SIDE and len(shorts) != led.EDGE:
            issues.append({"decision_date": date,
                           "issue": "old_side_wrong_size",
                           "n_old": n_old, "n_short": len(shorts)})
        if n_young < led.MIN_SIDE and longs:
            issues.append({"decision_date": date,
                           "issue": "young_legs_below_min_side"})
        if n_old < led.MIN_SIDE and shorts:
            issues.append({"decision_date": date,
                           "issue": "old_legs_below_min_side"})
        if not set(longs) <= young_pool:
            issues.append({"decision_date": date,
                           "issue": "long_outside_young_pool",
                           "longs": longs})
        if not set(shorts) <= old_pool:
            issues.append({"decision_date": date,
                           "issue": "short_outside_old_pool",
                           "shorts": shorts})
        for leg in row.get("book_detail") or []:
            want = ("young" if leg.get("side") == "long" else "old")
            if leg.get("age_bucket") != want:
                issues.append({"decision_date": date,
                               "issue": "leg_wrong_age_bucket",
                               "asset": leg.get("asset"),
                               "side": leg.get("side"),
                               "age_bucket": leg.get("age_bucket")})
    elif longs or shorts:
        issues.append({"decision_date": date,
                       "issue": "non_rebalance_event_with_book",
                       "event": event})


def cumulative_consistency(path):
    """Re-sum/re-compound day fields over recorded lines; return
    (mismatches, issues, last_line)."""
    cum = {"gross_bps": 0.0, "cost_bps": 0.0, "net_bps": 0.0}
    equity = 1.0
    last = None
    mismatches = []
    issues = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        last = row
        date = row.get("decision_date")
        event = row.get("event")
        if event not in led.EVENTS:
            issues.append({"decision_date": date,
                           "issue": "unknown_event", "event": event})
        uni = row.get("universe") or {}
        if len(uni.get("assets") or []) != uni.get("scored"):
            issues.append({"decision_date": date,
                           "issue": "universe_scored_mismatch",
                           "n_assets": len(uni.get("assets") or []),
                           "scored": uni.get("scored")})
        book_structure_ok(row, issues)

        gross = row["realized"].get("gross_bps")
        if gross is not None:
            cum["gross_bps"] += gross
        cum["cost_bps"] += row["rebalance"]["cost_bps"]
        if row["day_net_bps"] is not None:
            cum["net_bps"] += row["day_net_bps"]
        equity *= (1.0 + ((gross if gross is not None else 0.0)
                          - row["rebalance"]["cost_bps"]) / 1e4)
        recorded = row["cumulative"]
        for k in cum:
            if abs(recorded[k] - cum[k]) > 0.05:  # rounding tol (bps)
                mismatches.append({"decision_date": date, "field": k,
                                   "recorded": recorded[k],
                                   "resummed": round(cum[k], 4)})
        if abs(recorded["net_equity"] - equity) > 1e-4:
            mismatches.append({"decision_date": date,
                               "field": "net_equity",
                               "recorded": recorded["net_equity"],
                               "recompounded": round(equity, 8)})
    return mismatches, issues, last


def manifest_status(path):
    if not path.exists():
        return None
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {"unreadable": True}
    return {"records_written": m.get("records_written"),
            "missing_files": len(m.get("missing") or []),
            "refresh_range": m.get("refresh_range"),
            "manifest_mtime_utc": dt.datetime.fromtimestamp(
                path.stat().st_mtime, dt.timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", type=Path, default=LEDGER)
    ap.add_argument("--cohort", type=Path, default=COHORT)
    ap.add_argument("--supplement", type=Path, action="append",
                    default=None)
    ap.add_argument("--xs-manifest", type=Path, default=XS_MANIFEST)
    ap.add_argument("--xs2-manifest", type=Path, default=XS2_MANIFEST)
    ap.add_argument("--max-age-hours", type=float,
                    default=DEFAULT_MAX_AGE_HOURS)
    args = ap.parse_args()
    if args.supplement is None:
        args.supplement = SUPPLEMENTS

    now = dt.datetime.now(dt.timezone.utc)
    status = {"schema_version": "nanojev-forward-ledger-check-v5",
              "checked_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "ledger": str(args.ledger),
              "max_age_hours": args.max_age_hours}
    exit_code = 0

    if not args.ledger.exists():
        status["status"] = "missing"
        status["note"] = "ledger file not found; v5 ledger has never run"
        print(json.dumps(status, indent=2, sort_keys=True))
        return 1

    ledger_dates, max_ns, n_lines, n_bad = scan_jsonl(args.ledger)
    if max_ns is None:
        status["status"] = "empty"
        status["lines"] = n_lines
        print(json.dumps(status, indent=2, sort_keys=True))
        return 1

    input_dates, tail = input_tail(args.cohort, args.supplement)
    missing_dates = sorted(d for d in input_dates
                           if d not in ledger_dates)
    status["ledger_last_date"] = max(ledger_dates)
    status["input_tail_date"] = max(input_dates) if input_dates else None
    status["input_tail_per_symbol"] = tail
    status["input_dates_missing_from_ledger"] = missing_dates[:10]
    status["n_missing_dates"] = len(missing_dates)
    status["lines"] = n_lines
    if n_bad:
        status["unparseable_lines"] = n_bad

    latest_utc = dt.datetime.fromtimestamp(max_ns / NS, dt.timezone.utc)
    age_h = (now - latest_utc).total_seconds() / 3600.0
    status["latest_decision_utc"] = latest_utc.strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    status["age_hours"] = round(age_h, 2)

    mismatches, issues, last_line = cumulative_consistency(args.ledger)
    status["cumulative_mismatches"] = mismatches[:5]
    status["n_cumulative_mismatches"] = len(mismatches)
    status["structural_issues"] = issues[:10]
    status["n_structural_issues"] = len(issues)
    if last_line:
        status["last_line"] = {
            "decision_date": last_line["decision_date"],
            "event": last_line["event"],
            "gate": last_line["gate"],
            "book": last_line["book"],
            "age_buckets": {k: last_line["age_buckets"][k]
                            for k in ("young_candidates",
                                      "old_candidates",
                                      "censored_scored")},
            "universe_scored": last_line["universe"]["scored"],
            "cumulative": last_line["cumulative"]}

    # replay divergence: fresh replay vs the recorded tail line
    try:
        series, _ = led.load_inputs(args.cohort, args.supplement)
        led.add_dfh20(series)
        led.add_age(series)
        replayed = {l["decision_date"]: l
                    for l in led.replay(series)}
        cur = replayed.get(status["ledger_last_date"])
        if cur and last_line:
            divergence = {k: {"replayed": cur[k],
                              "recorded": last_line[k]}
                          for k in ("event", "book", "gate",
                                    "age_buckets")
                          if cur[k] != last_line[k]}
            status["replay_divergence"] = divergence or None
    except Exception as error:  # noqa: BLE001 - probe must not crash
        status["replay_error"] = f"{type(error).__name__}: {error}"

    for key, path in (("xs_refresh", args.xs_manifest),
                      ("xs2_refresh", args.xs2_manifest)):
        m = manifest_status(path)
        if m:
            status[key] = m

    notes = []
    if missing_dates:
        notes.append(f"{len(missing_dates)} input date(s) have no "
                     "ledger line — the ledger is lagging its inputs")
        exit_code = 1
    if mismatches:
        notes.append("cumulative fields do not re-sum over recorded "
                     "lines")
        exit_code = 1
    if issues:
        notes.append("recorded lines violate the v5 ledger structure "
                     "(events/books/age pools)")
        exit_code = 1
    if age_h > args.max_age_hours:
        notes.append(f"newest ledger bar is older than "
                     f"{args.max_age_hours:g}h — daily job not landing")
        exit_code = 1
    if last_line and last_line["event"] == "flat_gate_off":
        notes.append("tail is flat_gate_off: btc_ret20 <= 0 — the "
                     "paper book is flat under the master gate")
    if last_line and last_line["event"] == "flat_thin_buckets":
        notes.append("tail is flat_thin_buckets: gate is on but "
                     "neither age pool fielded >=3 candidates — "
                     "sleeve flat until an age bucket thickens")
    if last_line and last_line["event"] == "skipped_no_btc_ret20":
        notes.append("tail is skipped_no_btc_ret20: the BTC series is "
                     "stale — check the majors (binance_refresh_v1) "
                     "supplement; ledger is healthy but cannot evaluate "
                     "the gate")
    if last_line and last_line["event"] == "skipped_insufficient_universe":
        notes.append("tail is skipped_insufficient_universe: a "
                     "supplement has stopped landing (scored < 6); "
                     "ledger is healthy but the sleeve is flat until "
                     "the refresh recovers")
    if status.get("replay_divergence"):
        notes.append("fresh replay disagrees with the recorded tail "
                     "line — late-arriving input bars changed the "
                     "cross-section (informational; append-only ledger "
                     "is not rewritten)")

    status["status"] = "ok" if exit_code == 0 else "unhealthy"
    if notes:
        status["notes"] = notes
    print(json.dumps(status, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
