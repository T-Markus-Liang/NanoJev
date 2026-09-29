#!/usr/bin/env python3
"""T120: health probe for the rolling-campaign sleeve forward ledger.

Read-only. Scans ``results/forward_ledger_v3.jsonl`` plus the merged XS
inputs (perp_pit_xs_v1 cohort + both refresh supplements) to answer:
is the ledger still stepping forward on fresh daily bars, and is the
campaign state machine internally consistent?

Checks:
  * ledger exists, is parseable, and its newest decision_date reaches
    the merged-input tail (the ledger writes a line for EVERY input
    date, so lag = input dates with no ledger line)
  * ledger freshness in wall-clock hours vs --max-age-hours
  * state-transition integrity across recorded lines: from a flat
    end-of-day state only scan/flat_gate_off/enter may follow; from an
    open campaign only hold/add/exit; enter lines carry an entry fill
    and set position.asset; exit lines carry a reason from the frozen
    T119 exit set; hold/add lines keep the same asset; campaigns_closed
    is nondecreasing and increments exactly on exit lines; equity is
    present and finite on every line
  * replay divergence: the ledger module's own replay is re-run over
    current inputs and event/position/equity on the ledger's last
    recorded date are compared; a mismatch is reported as
    replay_divergence (late-arriving bars legitimately change what a
    fresh replay sees — the append-only ledger is never rewritten),
    counted but not failed on its own
  * last-line event is surfaced: a persistent ``flat_gate_off`` tail
    simply means the BTC regime gate is off

Exit codes:
  0  healthy — ledger covers the input tail and is within
     --max-age-hours
  1  unhealthy — missing/empty ledger, lag behind inputs, state-machine
     inconsistency, or stale
"""

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "results/forward_ledger_v3.jsonl"
COHORT = ROOT / "data/perp_pit_xs_v1/records.jsonl"
SUPPLEMENTS = [ROOT / "data/binance_refresh_v1/records.jsonl",
               ROOT / "data/binance_xs_refresh_v1/records.jsonl"]
XS_MANIFEST = ROOT / "data/binance_xs_refresh_v1/refresh_manifest.json"
NS = 1_000_000_000
DEFAULT_MAX_AGE_HOURS = 48.0

sys.path.insert(0, str(ROOT / "scripts"))
import financial_forward_ledger_v3 as led  # noqa: E402

FLAT_EVENTS = {"scan", "flat_gate_off", "enter"}
OPEN_EVENTS = {"hold", "add", "exit"}


def scan_jsonl(path):
    """dates present, max decision_ns, n_lines, n_bad."""
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


def state_integrity(path):
    """Replay the recorded event/position stream; return (issues, last).

    The ledger lines carry the end-of-day state, so the state machine
    can be checked without re-reading market data.
    """
    issues = []
    last = None
    prev_open = False
    prev_asset = None
    prev_closed = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        last = row
        date = row.get("decision_date")
        event = row.get("event")
        pos = row.get("position")
        eq = row.get("equity")
        closed = row.get("campaigns_closed")

        if event not in led.EVENTS:
            issues.append({"date": date, "issue": "unknown_event",
                           "event": event})
        if not isinstance(eq, (int, float)) or not math.isfinite(eq):
            issues.append({"date": date, "issue": "bad_equity",
                           "equity": eq})
        if not isinstance(closed, int) or closed < prev_closed:
            issues.append({"date": date, "issue": "campaigns_closed_"
                           "not_nondecreasing",
                           "campaigns_closed": closed})
        if event == "exit" and closed != prev_closed + 1:
            issues.append({"date": date, "issue": "exit_without_"
                           "campaign_close"})
        if event != "exit" and closed != prev_closed:
            issues.append({"date": date, "issue": "campaigns_closed_"
                           "changed_without_exit"})

        if prev_open:
            if event not in OPEN_EVENTS:
                issues.append({"date": date, "issue": "open_state_"
                               "illegal_event", "event": event})
            if event in ("hold", "add"):
                if pos is None or pos.get("asset") != prev_asset:
                    issues.append({"date": date, "issue": "open_state_"
                                   "asset_break",
                                   "prev_asset": prev_asset,
                                   "position": pos})
            if event == "exit":
                reason = (row.get("exit") or {}).get("reason")
                if reason not in led.EXIT_REASONS:
                    issues.append({"date": date, "issue": "bad_exit_"
                                   "reason", "reason": reason})
                re = row.get("same_day_reentry")
                if pos is not None and (
                        re is None
                        or pos.get("asset") != re.get("asset")):
                    issues.append({"date": date, "issue": "post_exit_"
                                   "position_not_reentry"})
        else:
            if event not in FLAT_EVENTS:
                issues.append({"date": date, "issue": "flat_state_"
                               "illegal_event", "event": event})
            if event == "enter":
                if (pos is None or row.get("fill") is None
                        or pos.get("asset")
                        != row["fill"].get("asset")):
                    issues.append({"date": date, "issue": "enter_"
                                   "without_position"})
            elif pos is not None:
                issues.append({"date": date, "issue": "flat_event_"
                               "with_position", "event": event})

        prev_open = pos is not None
        prev_asset = pos.get("asset") if pos else None
        prev_closed = closed if isinstance(closed, int) else prev_closed
    return issues, last


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
    ap.add_argument("--max-age-hours", type=float,
                    default=DEFAULT_MAX_AGE_HOURS)
    args = ap.parse_args()
    if args.supplement is None:
        args.supplement = SUPPLEMENTS

    now = dt.datetime.now(dt.timezone.utc)
    status = {"schema_version": "nanojev-forward-ledger-check-v3",
              "checked_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "ledger": str(args.ledger),
              "max_age_hours": args.max_age_hours}
    exit_code = 0

    if not args.ledger.exists():
        status["status"] = "missing"
        status["note"] = "ledger file not found; v3 ledger has never run"
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

    issues, last_line = state_integrity(args.ledger)
    status["state_integrity_issues"] = issues[:10]
    status["n_state_integrity_issues"] = len(issues)
    if last_line:
        status["last_line"] = {
            "decision_date": last_line["decision_date"],
            "event": last_line["event"],
            "gate": last_line["gate"],
            "position": last_line["position"],
            "equity": last_line["equity"],
            "campaigns_closed": last_line["campaigns_closed"],
        }

    # replay divergence: fresh replay vs the recorded tail line
    try:
        series, _ = led.load_inputs(args.cohort, args.supplement)
        led.add_bar_features(series)
        replayed, _ = led.replay(series)
        by_date = {l["decision_date"]: l for l in replayed}
        cur = by_date.get(status["ledger_last_date"])
        if cur and last_line:
            divergence = {k: {"replayed": cur[k],
                              "recorded": last_line[k]}
                          for k in ("event", "position", "equity")
                          if cur[k] != last_line[k]}
            status["replay_divergence"] = divergence or None
    except Exception as error:  # noqa: BLE001 - probe must not crash
        status["replay_error"] = f"{type(error).__name__}: {error}"

    m = manifest_status(args.xs_manifest)
    if m:
        status["xs_refresh"] = m

    notes = []
    if missing_dates:
        notes.append(f"{len(missing_dates)} input date(s) have no "
                     "ledger line — the ledger is lagging its inputs")
        exit_code = 1
    if issues:
        notes.append("recorded lines violate the campaign state "
                     "machine transitions")
        exit_code = 1
    if age_h > args.max_age_hours:
        notes.append(f"newest ledger bar is older than "
                     f"{args.max_age_hours:g}h — daily job not landing")
        exit_code = 1
    if last_line and last_line["event"] == "flat_gate_off":
        notes.append("tail is flat_gate_off: BTC ret20 gate is off — "
                     "no campaigns can open until the regime turns")
    if status.get("replay_divergence"):
        notes.append("fresh replay disagrees with the recorded tail "
                     "line — late-arriving input bars changed the "
                     "state machine path (informational; append-only "
                     "ledger is not rewritten)")

    status["status"] = "ok" if exit_code == 0 else "unhealthy"
    if notes:
        status["notes"] = notes
    print(json.dumps(status, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
