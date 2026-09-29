#!/usr/bin/env python3
"""T149: aggregate health probe for ALL forward paper-trade ledgers.

Read-only w.r.t. ledgers and inputs. The launchd daily chain currently
writes seven ledgers:

  results/forward_ledger_v1.jsonl    spec v1,   perp_pit_v1 universe (5)
  results/forward_ledger_v2.jsonl    XS dfh,    xs_v1 universe (10)
  results/forward_ledger_v3.jsonl    campaign,  xs_v1
  results/forward_ledger_v4.jsonl    ridge_all, xs_v1
  results/forward_ledger_v2_xs2.jsonl  XS dfh,  xs_v2 universe (30)
  results/forward_ledger_v3_xs2.jsonl  campaign, xs_v2
  results/forward_ledger_v4_xs2_ridge_min3_top30.jsonl  ridge_min3_top30,
                                       xs_v2

Two files on disk are NOT written by that chain:

  results/forward_ledger_v4_xs2.jsonl — ORPHAN: earlier ``--universe
      xs_v2`` runs with the default ``ridge_all`` model; superseded by
      the ``ridge_min3_top30`` variant. Reported as ``orphan`` and
      excluded from the exit code so a permanently-stale-by-design
      file does not alarm forever.
  results/forward_ledger_v5.jsonl — T148 age-conditioned xs_v2 book;
      exists and receives manual/ad-hoc writes but is not yet in the
      launchd chain. Monitored (stale = alarm) so the sleeve cannot go
      silently dark if it does get scheduled.

Per-ledger checks:
  * exists, non-empty, parseable
  * coverage: input decision dates (cohort + refresh supplements) that
    have no ledger line; lag = input_tail - ledger_last
  * freshness: wall-clock age of the newest decision_ns vs
    --max-age-hours
  * growth: lines/dates vs the previous run of this probe (state kept in
    results/forward_ledgers_all_v1_state.json); a shrinking ledger is a
    failure, stagnation while inputs grew is caught by coverage
  * event sanity: every event in the family vocabulary; tail-30 active
    fraction reported; zero active events over the last 30 dates is a
    note (flat regimes are legitimate), an all-skip ledger or unknown
    events are failures
  * equity sanity: v2/v4 families re-sum/re-compound the recorded
    cumulative block (same tolerances as check_forward_ledger_v4.py);
    the last TAIL_DRIFT_DATES decision dates are exempt — a fresh
    append reprices recent fills against late-arriving bars while the
    append-only ledger is never rewritten, so tail-line cumulative
    drift is reported as tail_drift_note, not a failure;
    v3 equity must be finite and in (0, 100]; v1 signals must be finite
    where non-null

Exit codes:
  0  every chain-active ledger covers its input tail and is fresh
  1  any chain-active ledger missing/empty/lagging/stale/inconsistent

Output: a compact per-ledger status table, then notes. ``--json``
appends the full machine-readable detail.
"""

import argparse
import datetime as dt
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NS = 1_000_000_000
DEFAULT_MAX_AGE_HOURS = 48.0
STATE_FILE = ROOT / "results/forward_ledgers_all_v1_state.json"
# A fresh ledger append reprices recent fills against late-arriving
# bars, but the append-only file is never rewritten — so the last 1-2
# decision dates can legitimately carry a cumulative-base shift.
# Re-sum mismatches on those tail dates become tail_drift_note, not
# failures; mismatches on older lines still hard-fail.
TAIL_DRIFT_DATES = 2

# ----------------------------------------------------------- inputs
COHORT_V1 = ROOT / "data/perp_pit_v1/records.jsonl"
COHORT_XS1 = ROOT / "data/perp_pit_xs_v1/records.jsonl"
COHORT_XS2 = ROOT / "data/perp_pit_xs_v2/records.jsonl"
SUP_LEGACY = ROOT / "data/binance_refresh_v1/records.jsonl"
SUP_XS1 = ROOT / "data/binance_xs_refresh_v1/records.jsonl"
SUP_XS2 = ROOT / "data/binance_xs2_refresh_v1/records.jsonl"

INPUTS_V1 = [COHORT_V1, SUP_LEGACY]
INPUTS_XS1 = [COHORT_XS1, SUP_LEGACY, SUP_XS1]
INPUTS_XS2 = [COHORT_XS2, SUP_LEGACY, SUP_XS1, SUP_XS2]

# --------------------------------------------------------- registry
# family: v1 = per-asset signal lines; v2 = dfh book; v3 = campaign state
# machine; v4 = ridge book. chain=False -> file exists on disk but the
# current launchd chain no longer writes it.
LEDGERS = [
    {"name": "v1", "family": "v1", "chain": True,
     "path": ROOT / "results/forward_ledger_v1.jsonl",
     "inputs": INPUTS_V1,
     "desc": "spec v1 funding/basis sleeve (5 assets)"},
    {"name": "v2", "family": "v2", "chain": True,
     "path": ROOT / "results/forward_ledger_v2.jsonl",
     "inputs": INPUTS_XS1,
     "desc": "XS dfh20 book, xs_v1 (10 assets)"},
    {"name": "v3", "family": "v3", "chain": True,
     "path": ROOT / "results/forward_ledger_v3.jsonl",
     "inputs": INPUTS_XS1,
     "desc": "rolling campaign, xs_v1"},
    {"name": "v4", "family": "v4", "chain": True,
     "path": ROOT / "results/forward_ledger_v4.jsonl",
     "inputs": INPUTS_XS1,
     "desc": "ridge_all book, xs_v1"},
    {"name": "v2_xs2", "family": "v2", "chain": True,
     "path": ROOT / "results/forward_ledger_v2_xs2.jsonl",
     "inputs": INPUTS_XS2,
     "desc": "XS dfh20 book, xs_v2 (30 assets)"},
    {"name": "v3_xs2", "family": "v3", "chain": True,
     "path": ROOT / "results/forward_ledger_v3_xs2.jsonl",
     "inputs": INPUTS_XS2,
     "desc": "rolling campaign, xs_v2"},
    {"name": "v4_xs2", "family": "v4", "chain": False,
     "path": ROOT / "results/forward_ledger_v4_xs2.jsonl",
     "inputs": INPUTS_XS2,
     "desc": "ridge_all book, xs_v2 — ORPHAN: superseded by "
             "ridge_min3_top30 variant in the daily chain"},
    {"name": "v4_xs2_ridge_min3_top30", "family": "v4", "chain": True,
     "path": ROOT / "results/forward_ledger_v4_xs2_ridge_min3_top30.jsonl",
     "inputs": INPUTS_XS2,
     "desc": "ridge_min3_top30 book, xs_v2"},
    {"name": "v5", "family": "v5", "chain": True,
     "path": ROOT / "results/forward_ledger_v5.jsonl",
     "inputs": INPUTS_XS2,
     "desc": "T148 age-conditioned book, xs_v2 — not yet in the "
             "launchd chain; written by manual/ad-hoc runs"},
]

EVENT_VOCAB = {
    "v1": {None, "enter_long", "exit_time_stop", "exit_regime_off"},
    "v2": {"rebalance", "flat_gate_off", "skipped_insufficient_universe",
           "skipped_no_btc_bar"},
    "v3": {"scan", "flat_gate_off", "enter", "add", "exit", "hold"},
    "v4": {"score", "score_flat_gate_off",
           "skipped_insufficient_universe", "skipped_no_btc_ret20"},
    "v5": {"rebalance", "flat_gate_off", "flat_thin_buckets",
           "skipped_insufficient_universe", "skipped_no_btc_ret20"},
}
STATE_VOCAB_V1 = {"warmup", "flat", "long"}


def active_event(family, event):
    """Is this event 'doing something' (vs flat/skipped)?"""
    if family == "v1":
        return event is not None
    if family in ("v2", "v5"):
        return event == "rebalance"
    if family == "v3":
        return event in ("enter", "add", "exit", "hold")
    return event in ("score", "score_flat_gate_off")  # v4


# ------------------------------------------------------------- scan
def row_date(row):
    if "decision_date" in row:
        return row["decision_date"]
    rid = row.get("record_id") or row.get("id") or ""
    return rid.rsplit(":", 1)[-1]


def scan_ledger(path):
    """Single pass: dates, max decision_ns, lines, bad lines, events,
    last line."""
    dates = set()
    max_ns = None
    n_lines = n_bad = 0
    events = {}
    last = None
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
        last = row
        d = row_date(row)
        if d:
            dates.add(d)
        if max_ns is None or ns > max_ns:
            max_ns = ns
        ev = row.get("event")
        events[ev] = events.get(ev, 0) + 1
    return {"dates": dates, "max_ns": max_ns, "lines": n_lines,
            "bad": n_bad, "events": events, "last": last}


def input_dates(paths):
    """All decision dates across cohort + supplement inputs."""
    dates = set()
    for p in paths:
        if not p.exists():
            continue
        for l in p.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            dates.add(json.loads(l)["id"].rsplit(":", 1)[-1])
    return dates


def equity_check(family, path, tail_dates):
    """Per-family cumulative-equity sanity -> (issues, tail_issues,
    last_equity). Re-sum/re-compound mismatches whose decision_date is
    in tail_dates go to tail_issues (legitimate repricing drift); the
    rest are hard inconsistencies."""
    issues = []
    tail_issues = []
    last_equity = None
    if family in ("v2", "v4", "v5"):
        cum = {"gross_bps": 0.0, "cost_bps": 0.0, "net_bps": 0.0}
        equity = 1.0
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
                rec = row["cumulative"]
            except (ValueError, KeyError):
                continue
            gross = row.get("realized", {}).get("gross_bps")
            cost = row.get("rebalance", {}).get("cost_bps") or 0.0
            if gross is not None:
                cum["gross_bps"] += gross
            cum["cost_bps"] += cost
            if row.get("day_net_bps") is not None:
                cum["net_bps"] += row["day_net_bps"]
            equity *= 1.0 + ((gross if gross is not None else 0.0)
                             - cost) / 1e4
            sink = (tail_issues if row_date(row) in tail_dates
                    else issues)
            for k, tol in (("gross_bps", 0.05), ("cost_bps", 0.05),
                           ("net_bps", 0.05)):
                if k in rec and abs(rec[k] - cum[k]) > tol:
                    sink.append({"date": row_date(row), "field": k,
                                 "recorded": rec[k],
                                 "resummed": round(cum[k], 4)})
            if rec.get("net_equity") is not None:
                if abs(rec["net_equity"] - equity) > 1e-4:
                    sink.append({"date": row_date(row),
                                 "field": "net_equity",
                                 "recorded": rec["net_equity"],
                                 "recompounded": round(equity, 8)})
            last_equity = rec.get("net_equity")
        if (last_equity is not None
                and not (1e-3 < last_equity < 1e3)):
            issues.append({"field": "net_equity", "issue": "absurd_value",
                           "value": last_equity})
    elif family == "v3":
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            eq = row.get("equity")
            if (not isinstance(eq, (int, float)) or not math.isfinite(eq)
                    or not (0.0 < eq <= 100.0)):
                issues.append({"date": row_date(row),
                               "issue": "bad_equity", "equity": eq})
            else:
                last_equity = eq
    else:  # v1: no equity; check signal fields are finite when present
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            sig = row.get("signal") or {}
            for k, v in sig.items():
                if v is not None and (
                        not isinstance(v, (int, float))
                        or not math.isfinite(v)):
                    issues.append({"date": row_date(row),
                                   "issue": "nonfinite_signal",
                                   "field": k, "value": v})
    return issues, tail_issues, last_equity


def load_state(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True),
                    encoding="utf-8")


# ------------------------------------------------------------- main
def check_one(spec, prev, now, max_age_hours):
    """Run every check for one ledger; return a status dict."""
    s = {"name": spec["name"], "family": spec["family"],
         "chain": spec["chain"], "desc": spec["desc"],
         "ledger": str(spec["path"])}
    fails = []
    notes = []

    if not spec["path"].exists():
        s.update(status="missing", fails=["ledger file not found"])
        return s, []

    scan = scan_ledger(spec["path"])
    if scan["max_ns"] is None:
        s.update(status="empty", lines=scan["lines"],
                 fails=["no parseable lines"])
        return s, []

    s["lines"] = scan["lines"]
    s["dates"] = len(scan["dates"])
    s["ledger_last_date"] = max(scan["dates"])
    if scan["bad"]:
        notes.append(f"{scan['bad']} unparseable line(s)")
        s["unparseable_lines"] = scan["bad"]

    latest_utc = dt.datetime.fromtimestamp(scan["max_ns"] / NS,
                                           dt.timezone.utc)
    age_h = (now - latest_utc).total_seconds() / 3600.0
    s["latest_decision_utc"] = latest_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    s["age_hours"] = round(age_h, 1)

    # coverage vs merged inputs
    idates = input_dates(spec["inputs"])
    missing = sorted(d for d in idates if d not in scan["dates"])
    s["input_tail_date"] = max(idates) if idates else None
    s["n_missing_input_dates"] = len(missing)
    if missing:
        s["missing_input_dates_head"] = missing[:5]
        lag = (dt.date.fromisoformat(s["input_tail_date"])
               - dt.date.fromisoformat(s["ledger_last_date"])).days
        s["lag_days"] = lag
        fails.append(f"{len(missing)} input date(s) have no ledger line "
                     f"(lag {lag}d)")
    else:
        s["lag_days"] = 0

    # freshness
    if age_h > max_age_hours:
        fails.append(f"newest decision bar is {age_h:.1f}h old "
                     f"(> {max_age_hours:g}h)")

    # growth vs last probe run
    p = prev.get(spec["name"]) or {}
    if p:
        d_lines = scan["lines"] - p.get("lines", 0)
        d_dates = len(scan["dates"]) - p.get("dates", 0)
        s["lines_delta"] = d_lines
        s["dates_delta"] = d_dates
        if d_lines < 0:
            fails.append(f"ledger shrank by {-d_lines} line(s) since "
                         "last check — truncation/rewrite?")

    # event sanity
    vocab = EVENT_VOCAB[spec["family"]]
    unknown = {e: n for e, n in scan["events"].items() if e not in vocab}
    s["events"] = {str(k): v for k, v in sorted(
        scan["events"].items(), key=lambda kv: -kv[1])}
    if unknown:
        fails.append(f"unknown event(s): "
                     f"{ {str(k): v for k, v in unknown.items()} }")
    if spec["family"] == "v1":
        bad_states = [r for r in _iter_rows(spec["path"])
                      if r.get("state") not in STATE_VOCAB_V1]
        if bad_states:
            fails.append(f"{len(bad_states)} line(s) with unknown state")
    tail = sorted(scan["dates"])[-30:]
    tail_set = set(tail)
    n_tail = n_tail_active = 0
    for r in _iter_rows(spec["path"]):
        if row_date(r) in tail_set:
            n_tail += 1
            if active_event(spec["family"], r.get("event")):
                n_tail_active += 1
    s["tail30_active_frac"] = (round(n_tail_active / n_tail, 3)
                               if n_tail else None)
    if n_tail and n_tail_active == 0:
        if not any(active_event(spec["family"], e)
                   for e in scan["events"]):
            fails.append("ledger has never emitted an active event — "
                         "pathological")
        else:
            notes.append("no active events in the last 30 decision "
                         "dates (flat/skip streak — could be a stuck "
                         "gate or dead supplement)")
    s["last_event"] = (scan["last"] or {}).get("event")
    if spec["family"] == "v1" and scan["last"]:
        s["last_state"] = scan["last"].get("state")

    # equity sanity — cumulative re-sum is checked on non-tail lines
    # only; the last TAIL_DRIFT_DATES decision dates may legitimately
    # drift (fresh replay repricing on append) and become a note
    tail_dates = set(sorted(scan["dates"])[-TAIL_DRIFT_DATES:])
    eq_issues, tail_eq_issues, last_eq = equity_check(
        spec["family"], spec["path"], tail_dates)
    s["last_equity"] = last_eq
    s["n_equity_issues"] = len(eq_issues)
    if eq_issues:
        s["equity_issues_head"] = eq_issues[:5]
        fails.append(f"{len(eq_issues)} cumulative/equity "
                     "inconsistencies on non-tail lines")
    if tail_eq_issues:
        s["n_tail_drift_issues"] = len(tail_eq_issues)
        s["tail_drift_issues_head"] = tail_eq_issues[:5]
        s["tail_drift_note"] = (
            f"{len(tail_eq_issues)} cumulative/equity mismatch(es) on "
            f"the last {TAIL_DRIFT_DATES} decision date(s) "
            f"({', '.join(sorted(tail_dates))}): the fresh replay "
            "repriced recent fills against late-arriving bars while "
            "the append-only ledger is never rewritten — "
            "informational, not a failure")
        notes.append(s["tail_drift_note"])

    s["fails"] = fails
    s["notes"] = notes
    if not spec["chain"]:
        s["status"] = "orphan"
        notes.insert(0, "not written by the current launchd chain "
                        "(see desc); reported but excluded from the "
                        "exit code")
    else:
        s["status"] = "ok" if not fails else "unhealthy"
    new_prev = {"lines": scan["lines"], "dates": len(scan["dates"]),
                "last_date": s["ledger_last_date"],
                "checked_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ")}
    return s, [(spec["name"], new_prev)]


def _iter_rows(path):
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            yield json.loads(raw)
        except ValueError:
            continue


def fmt_row(s):
    lag = s.get("lag_days")
    lag_s = "-" if lag is None else (f"{lag}d" if lag else "0")
    age = s.get("age_hours")
    delta = s.get("lines_delta")
    delta_s = "-" if delta is None else f"{delta:+d}"
    eq = s.get("last_equity")
    eq_s = "-" if eq is None else f"{eq:.4f}"
    act = s.get("tail30_active_frac")
    act_s = "-" if act is None else f"{act:.2f}"
    return (f"{s['name']:<26} {s['status']:<10} "
            f"{s.get('ledger_last_date', '-'):<10} {lag_s:<4} "
            f"{(f'{age:.1f}' if age is not None else '-'):<6} "
            f"{s.get('lines', '-'):<6} {delta_s:<6} {act_s:<5} "
            f"{eq_s:<8} {str(s.get('last_event')):<32}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-age-hours", type=float,
                    default=DEFAULT_MAX_AGE_HOURS,
                    help="ledger is stale when the newest bar is older "
                         "than this (default 48h; daily job + 1d "
                         "slack)")
    ap.add_argument("--state-file", type=Path, default=STATE_FILE,
                    help="probe state for growth checks")
    ap.add_argument("--no-state", action="store_true",
                    help="do not read or write the growth state file")
    ap.add_argument("--json", action="store_true",
                    help="also print the full JSON detail after the "
                         "status table")
    args = ap.parse_args()

    now = dt.datetime.now(dt.timezone.utc)
    prev = {} if args.no_state else load_state(args.state_file)
    prev_ledgers = prev.get("ledgers", {})

    detail = {"schema_version": "nanojev-forward-ledgers-check-all-v1",
              "checked_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "max_age_hours": args.max_age_hours,
              "ledgers": []}
    new_ledgers = {}
    exit_code = 0

    for spec in LEDGERS:
        s, state_entry = check_one(spec, prev_ledgers, now,
                                   args.max_age_hours)
        detail["ledgers"].append(s)
        new_ledgers.update(state_entry)
        if spec["chain"] and s["status"] != "ok":
            exit_code = 1

    # -------------------------------------------------- status table
    print(f"forward-ledger fleet @ "
          f"{now.strftime('%Y-%m-%dT%H:%M:%SZ')}  "
          f"(max_age {args.max_age_hours:g}h)")
    hdr = (f"{'ledger':<26} {'status':<10} {'last_date':<10} {'lag':<4} "
           f"{'age_h':<6} {'lines':<6} {'Δlines':<6} {'act30':<5} "
           f"{'equity':<8} {'last_event':<32}")
    print(hdr)
    print("-" * len(hdr))
    for s in detail["ledgers"]:
        print(fmt_row(s))
        for f_ in s.get("fails", []):
            print(f"    FAIL: {f_}")
        for n in s.get("notes", []):
            print(f"    note: {n}")

    n_bad = sum(1 for s in detail["ledgers"]
                if s["status"] not in ("ok", "orphan"))
    n_orphan = sum(1 for s in detail["ledgers"]
                   if s["status"] == "orphan")
    detail["summary"] = {
        "status": "ok" if exit_code == 0 else "unhealthy",
        "n_ledgers": len(detail["ledgers"]),
        "n_unhealthy": n_bad,
        "n_orphan": n_orphan}
    print(f"summary: {detail['summary']['status']} — "
          f"{len(detail['ledgers'])} ledgers scanned, "
          f"{n_bad} unhealthy, {n_orphan} orphan(s) "
          "(excluded from exit code)")

    if args.json:
        print(json.dumps(detail, indent=2, sort_keys=True))

    if not args.no_state:
        save_state(args.state_file,
                   {"schema_version": "nanojev-forward-ledgers-state-v1",
                    "checked_at_utc":
                        now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "ledgers": new_ledgers})
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
