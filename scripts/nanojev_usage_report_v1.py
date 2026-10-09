#!/usr/bin/env python3
"""Usage report for the local NanoJev eval/route wrappers.

Reads the content-free JSONL log written by ``nanojev-eval`` and
``jev-route`` (default ``~/.local/state/nanojev-eval/log.jsonl``) and prints
a compact summary:

  * total events, broken down by tool (nanojev-eval calls, jev-route
    routing decisions, feedback notes)
  * local-vs-official split of routed traffic
  * ``--mode quality`` escalation rate (local confidence below threshold)
  * feedback entries verbatim
  * log time span (implied uptime of observed usage)

Read-only. Standard library only. Unknown/malformed lines are counted and
skipped, never fatal.

Usage:
    python3 scripts/nanojev_usage_report_v1.py [--log PATH]
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG = Path.home() / ".local/state/nanojev-eval/log.jsonl"


def parse_ts(value):
    """Parse ``2026-09-24T01:23:01Z``-style timestamps; ``None`` on failure."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def fmt_span(first, last):
    """Human-readable span like ``1h 44m`` or ``3d 2h``."""
    if first is None or last is None:
        return "unknown"
    seconds = max(0, int((last - first).total_seconds()))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes = seconds // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def pct(part, whole):
    return f"{100.0 * part / whole:.1f}%" if whole else "n/a"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", default=str(DEFAULT_LOG),
                        help="path to log.jsonl (default: %(default)s)")
    args = parser.parse_args()
    log_path = Path(args.log).expanduser()

    if not log_path.exists():
        print(f"no log at {log_path} — nothing recorded yet")
        return 0

    calls = []            # type == "call" (nanojev-eval)
    routes = []           # type == "route" (jev-route)
    feedback = []         # type == "agent_feedback"
    other = Counter()     # any other "type" value
    malformed = 0
    timestamps = []

    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            ts = parse_ts(event.get("timestamp"))
            if ts is not None:
                timestamps.append(ts)
            kind = event.get("type")
            if kind == "call":
                calls.append(event)
            elif kind == "route":
                routes.append(event)
            elif kind == "agent_feedback":
                feedback.append(event)
            else:
                other[str(kind)] += 1

    total = len(calls) + len(routes) + len(feedback) + sum(other.values())

    print(f"nanojev usage report — {log_path}")
    print("=" * 60)
    if timestamps:
        first, last = min(timestamps), max(timestamps)
        span = fmt_span(first, last)
        first_s = first.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        last_s = last.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"log span: {first_s} → {last_s} "
              f"(implied uptime span: {span})")
    print(f"events: {total} parsed"
          + (f", {malformed} malformed lines skipped" if malformed else ""))
    print()

    # --- calls by tool ---------------------------------------------------
    print("calls by tool")
    print(f"  nanojev-eval (call events):      {len(calls)}")
    if calls:
        endpoints = Counter(c.get("endpoint", "?") for c in calls)
        backends = Counter(c.get("backend", "-") for c in calls)
        errors = sum(1 for c in calls if c.get("status") != 200)
        elapsed = [c["elapsed_ms"] for c in calls
                   if isinstance(c.get("elapsed_ms"), (int, float))]
        print(f"    endpoints: " + ", ".join(
            f"{k}={v}" for k, v in sorted(endpoints.items())))
        print(f"    backends:  " + ", ".join(
            f"{k}={v}" for k, v in sorted(backends.items())))
        if elapsed:
            print(f"    elapsed_ms: avg={sum(elapsed)/len(elapsed):.0f} "
                  f"min={min(elapsed):.0f} max={max(elapsed):.0f}")
        if errors:
            print(f"    non-200 responses: {errors}")
    print(f"  jev-route (route events):        {len(routes)}")
    if routes:
        modes = Counter(r.get("mode") or "ratio" for r in routes)
        print(f"    modes: " + ", ".join(
            f"{k}={v}" for k, v in sorted(modes.items())))
    print(f"  feedback notes (agent_feedback): {len(feedback)}")
    for kind, count in sorted(other.items()):
        print(f"  other type={kind}: {count}")
    print()

    # --- local vs official ----------------------------------------------
    if routes:
        actual = Counter(r.get("actual", "?") for r in routes)
        local = actual.get("local", 0)
        official = actual.get("official", 0)
        print("local vs official (route.actual)")
        print(f"  local:    {local} ({pct(local, len(routes))})")
        print(f"  official: {official} ({pct(official, len(routes))})")
        for key in sorted(actual):
            if key not in ("local", "official"):
                print(f"  {key}: {actual[key]}")
        print()

    # --- quality-mode escalation ----------------------------------------
    quality = [r for r in routes if (r.get("mode") or "ratio") == "quality"]
    if quality:
        escalated = [r for r in quality
                     if r.get("actual") == "official"]
        print("quality mode (--mode quality)")
        print(f"  routed: {len(quality)}, escalated to official: "
              f"{len(escalated)} ({pct(len(escalated), len(quality))})")
        confidences = [r["local_confidence"] for r in quality
                       if isinstance(r.get("local_confidence"), (int, float))]
        if confidences:
            print(f"  local_confidence: min={min(confidences):.4f} "
                  f"avg={sum(confidences)/len(confidences):.4f} "
                  f"max={max(confidences):.4f}")
        print()

    # --- feedback verbatim ----------------------------------------------
    print("feedback entries")
    if feedback:
        for event in feedback:
            stamp = event.get("timestamp", "?")
            print(f"  [{stamp}] {event.get('note', '')}")
    else:
        print("  (none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
