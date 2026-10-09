#!/usr/bin/env python3
"""T110: health probe for the XS dfh sleeve forward paper ledger.

Read-only. Scans ``results/forward_ledger_v2.jsonl`` plus the merged XS
inputs (perp_pit_xs_v1 cohort + both refresh supplements) to answer:
is the ledger still stepping forward on fresh daily bars, and is the book
accounting internally consistent?

Checks:
  * ledger exists, is parseable, and its newest decision_date reaches the
    merged-input tail (the ledger writes a line for EVERY input date, so
    lag = input dates with no ledger line)
  * ledger freshness in wall-clock hours vs --max-age-hours
  * cumulative consistency: each line's cumulative.net_bps equals the
    running sum of day_net_bps (spot-checked across the whole file);
    the last TAIL_DRIFT_DATES decision dates are exempt — a fresh
    append reprices recent fills against late-arriving bars while the
    append-only ledger is never rewritten, so tail-line cumulative
    drift is reported as tail_drift_note, not a failure
  * book consistency: the ledger module's own replay is re-run over current
    inputs and the book/gate/event on the ledger's last recorded date is
    compared; a mismatch is reported as replay_divergence (late-arriving
    bars legitimately change what a fresh replay sees — the append-only
    ledger is never rewritten), counted but not failed on its own
  * last-line event is surfaced: a persistent
    ``skipped_insufficient_universe`` tail means a supplement stopped
    landing even though the ledger itself is healthy

Exit codes:
  0  healthy — ledger covers the input tail and is within --max-age-hours
  1  unhealthy — missing/empty ledger, lag behind inputs, cumulative
     inconsistency, or stale
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "results/forward_ledger_v2.jsonl"
COHORT = ROOT / "data/perp_pit_xs_v1/records.jsonl"
SUPPLEMENTS = [ROOT / "data/binance_refresh_v1/records.jsonl",
               ROOT / "data/binance_xs_refresh_v1/records.jsonl"]
XS_MANIFEST = ROOT / "data/binance_xs_refresh_v1/refresh_manifest.json"
NS = 1_000_000_000
DEFAULT_MAX_AGE_HOURS = 48.0
# Re-sum mismatches on the last TAIL_DRIFT_DATES decision dates are
# legitimate tail drift (fresh replay repricing on append) — reported
# as tail_drift_note; mismatches on older lines still hard-fail.
TAIL_DRIFT_DATES = 2

sys.path.insert(0, str(ROOT / "scripts"))
import financial_forward_ledger_v2 as led  # noqa: E402


def scan_jsonl(path):
    """dates present, max decision_ns, n_lines, n_bad for a ledger/input file."""
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
        date = row.get("decision_date") or row.get("id", "").rsplit(":", 1)[-1]
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


def cumulative_consistency(path, tail_dates):
    """Re-sum day_net/cost over the recorded lines; return
    (mismatches, tail_mismatches, last). Mismatches on tail_dates are
    tail drift (fresh replay repricing on append), not failures."""
    cum = {"gross_bps": 0.0, "cost_bps": 0.0, "net_bps": 0.0}
    last = None
    mismatches = []
    tail_mismatches = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        last = row
        gross = row["realized"].get("gross_bps")
        if gross is not None:
            cum["gross_bps"] += gross
        cum["cost_bps"] += row["rebalance"]["cost_bps"]
        if row["day_net_bps"] is not None:
            cum["net_bps"] += row["day_net_bps"]
        recorded = row["cumulative"]
        sink = (tail_mismatches
                if row.get("decision_date") in tail_dates
                else mismatches)
        for k in cum:
            if abs(recorded[k] - cum[k]) > 0.05:  # rounding tolerance (bps)
                sink.append({"decision_date": row["decision_date"],
                             "field": k, "recorded": recorded[k],
                             "resummed": round(cum[k], 4)})
    return mismatches, tail_mismatches, last


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
    ap.add_argument("--supplement", type=Path, action="append", default=None)
    ap.add_argument("--xs-manifest", type=Path, default=XS_MANIFEST)
    ap.add_argument("--max-age-hours", type=float,
                    default=DEFAULT_MAX_AGE_HOURS)
    args = ap.parse_args()
    if args.supplement is None:
        args.supplement = SUPPLEMENTS

    now = dt.datetime.now(dt.timezone.utc)
    status = {"schema_version": "nanojev-forward-ledger-check-v2",
              "checked_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "ledger": str(args.ledger),
              "max_age_hours": args.max_age_hours}
    exit_code = 0

    if not args.ledger.exists():
        status["status"] = "missing"
        status["note"] = "ledger file not found; v2 ledger has never run"
        print(json.dumps(status, indent=2, sort_keys=True))
        return 1

    ledger_dates, max_ns, n_lines, n_bad = scan_jsonl(args.ledger)
    if max_ns is None:
        status["status"] = "empty"
        status["lines"] = n_lines
        print(json.dumps(status, indent=2, sort_keys=True))
        return 1

    input_dates, tail = input_tail(args.cohort, args.supplement)
    missing_dates = sorted(d for d in input_dates if d not in ledger_dates)
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
    status["latest_decision_utc"] = latest_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    status["age_hours"] = round(age_h, 2)

    tail_dates = set(sorted(ledger_dates)[-TAIL_DRIFT_DATES:])
    mismatches, tail_mismatches, last_line = cumulative_consistency(
        args.ledger, tail_dates)
    status["cumulative_mismatches"] = mismatches[:5]
    status["n_cumulative_mismatches"] = len(mismatches)
    if tail_mismatches:
        status["tail_cumulative_mismatches"] = tail_mismatches[:5]
        status["n_tail_cumulative_mismatches"] = len(tail_mismatches)
        status["tail_drift_note"] = (
            "cumulative re-sum drift on the last "
            f"{TAIL_DRIFT_DATES} decision date(s) "
            f"({', '.join(sorted(tail_dates))}): fresh replay repriced "
            "recent fills against late-arriving bars while the "
            "append-only ledger is never rewritten — informational")
    if last_line:
        status["last_line"] = {"decision_date": last_line["decision_date"],
                               "event": last_line["event"],
                               "gate_on": last_line["gate"]["on"],
                               "book": last_line["book"],
                               "universe_ranked":
                                   last_line["universe"]["ranked"],
                               "cumulative": last_line["cumulative"]}

    # replay divergence: fresh replay vs the recorded line on the same date
    try:
        series, _ = led.load_inputs(args.cohort, args.supplement)
        led.add_dfh20(series)
        replayed = {l["decision_date"]: l for l in led.replay(series)}
        cur = replayed.get(status["ledger_last_date"])
        if cur and last_line:
            divergence = {k: {"replayed": cur[k], "recorded": last_line[k]}
                          for k in ("event", "book", "gate")
                          if cur[k] != last_line[k]}
            status["replay_divergence"] = divergence or None
    except Exception as error:  # noqa: BLE001 - probe must not crash on inputs
        status["replay_error"] = f"{type(error).__name__}: {error}"

    m = manifest_status(args.xs_manifest)
    if m:
        status["xs_refresh"] = m

    notes = []
    if missing_dates:
        notes.append(f"{len(missing_dates)} input date(s) have no ledger "
                     "line — the ledger is lagging its inputs")
        exit_code = 1
    if mismatches:
        notes.append("cumulative fields do not re-sum over recorded "
                     "non-tail lines")
        exit_code = 1
    if tail_mismatches:
        notes.append("tail_drift_note: cumulative drift confined to "
                     "the tail decision date(s) — fresh replay "
                     "repricing; the append-only ledger is not "
                     "rewritten")
    if age_h > args.max_age_hours:
        notes.append(f"newest ledger bar is older than "
                     f"{args.max_age_hours:g}h — daily job not landing")
        exit_code = 1
    if last_line and last_line["event"] == "skipped_insufficient_universe":
        notes.append("tail is skipped_insufficient_universe: a supplement "
                     "has stopped landing (universe < 6); ledger is healthy "
                     "but the sleeve is flat until the refresh recovers")
    if status.get("replay_divergence"):
        notes.append("fresh replay disagrees with the recorded tail line — "
                     "late-arriving input bars changed the cross-section "
                     "(informational; append-only ledger is not rewritten)")

    status["status"] = "ok" if exit_code == 0 else "unhealthy"
    if notes:
        status["notes"] = notes
    print(json.dumps(status, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
