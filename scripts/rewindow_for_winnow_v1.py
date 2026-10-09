#!/usr/bin/env python3
"""Optional winnow-compatible re-windowing for real-context eval candidates.

Implements the recovery path of docs/WINNOW_CONTEXT_LIMIT_V1.md (option (a)
subset). The default policy stays "accept partial coverage" — this script
builds a *parallel* candidate file; nothing here switches it on.

  - Reads data/real_context_eval_v1/candidates.jsonl.
  - Measures every record's *winnow* prefix tokens via
    POST /v1/winnow/inspect (tokenize-only, ~10 ms). Calls are sequential —
    kev wedged under concurrency 8 earlier and winnow may behave similarly;
    547 records is seconds of work either way. Localhost URLs only.
  - Records whose prefix exceeds TARGET_PREFIX_TOKENS (8,000; safety margin
    under the measured 8,096 ceiling = 8,192 ctx - 96-token noul suffix) are
    re-windowed: the OLDEST non-protected conversation entries are dropped
    first, re-measuring after each drop, until the prefix fits or only
    protected entries remain.
  - Protected entries mirror the "must" set of
    build_real_context_eval_v1._window_indices: the candidate segment itself
    (state.candidate_pointer), the final user turn, and every
    control/system-role entry — which covers the "/elided/N" sidecar
    pointers the builder inserts. state.user_messages_in_order is a sidecar
    and is preserved verbatim.
  - Dropped runs are re-rendered as {"pointer": "/elided/k", "role":
    "control", "content": "<elided N earlier segments>"} entries, matching
    _render_window; marker numbering continues after any existing /elided
    pointers.
  - A record that still cannot fit is written (minimally windowed) with
    meta.winnow_overflow = true — fail-open to winnow abstention, which the
    review queue already maps to "uncertain" -> human review.
  - Output: data/real_context_eval_v1/candidates_winnow_fit.jsonl with the
    same record_ids in the same order; meta.winnow_rewindow carries the
    provenance note (measured/final prefix tokens, dropped pointers).

Usage: python3 scripts/rewindow_for_winnow_v1.py \
    [--candidates PATH] [--output PATH] [--inspect-url URL] \
    [--target 8000] [--limit N]
"""
import argparse
from http.client import HTTPConnection
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CANDIDATES = ROOT / "data" / "real_context_eval_v1" / "candidates.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "real_context_eval_v1" / "candidates_winnow_fit.jsonl"
DEFAULT_INSPECT_URL = "http://127.0.0.1:8091/v1/winnow/inspect"

# Measured ceiling is prefix <= 8,096 (docs/WINNOW_CONTEXT_LIMIT_V1.md);
# 8,000 leaves margin for suffix variation if the question text ever changes.
TARGET_PREFIX_TOKENS = 8_000

PROTECTED_ROLES = ("control", "system")
ELIDED_POINTER = re.compile(r"^/elided/(\d+)$")
LOCALHOSTS = {"127.0.0.1", "localhost", "::1"}


def serialized(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False)


def candidate_index(conversation, candidate_pointer):
    for i, entry in enumerate(conversation):
        if entry.get("pointer") == candidate_pointer:
            return i
    return None


def final_user_index(conversation):
    return max((i for i, e in enumerate(conversation) if e.get("role") == "user"),
               default=None)


def protected_indices(conversation, candidate_pointer):
    """Indices that re-windowing must never drop: control/system-role entries
    (including /elided/N sidecar pointers), the candidate segment, and the
    final user turn. Returns None when the candidate pointer does not resolve
    (re-windowing would risk dropping the candidate -> caller fails open)."""
    cand = candidate_index(conversation, candidate_pointer)
    if cand is None:
        return None
    protected = {i for i, e in enumerate(conversation)
                 if e.get("role") in PROTECTED_ROLES}
    protected.add(cand)
    last_user = final_user_index(conversation)
    if last_user is not None:
        protected.add(last_user)
    return protected


def render_window(entries, dropped):
    """Re-render ``entries`` with the indices in ``dropped`` removed, inserting
    "<elided N earlier segments>" control markers at every gap (same shape as
    build_real_context_eval_v1._render_window). New /elided pointers are
    numbered after the highest existing one."""
    next_marker = 1 + max(
        (int(m.group(1)) for e in entries
         for m in (ELIDED_POINTER.match(str(e.get("pointer", ""))),) if m),
        default=-1)
    out, run = [], 0
    for i, entry in enumerate(entries):
        if i in dropped:
            run += 1
            continue
        if run:
            out.append({"pointer": f"/elided/{next_marker}",
                        "role": "control",
                        "content": f"<elided {run} earlier segments>"})
            next_marker += 1
            run = 0
        out.append(entry)
    if run:
        out.append({"pointer": f"/elided/{next_marker}",
                    "role": "control",
                    "content": f"<elided {run} earlier segments>"})
    return out


def rewindow_request(request, measure, target=TARGET_PREFIX_TOKENS):
    """Shrink ``request.state.conversation`` until ``measure(request)`` (winnow
    prefix tokens) is <= ``target``. ``measure`` is any request-dict -> int
    callable so tests can inject a fake counter.

    Returns (request, info); info lists the dropped pointers and flags
    overflow when the protected floor still exceeds the target."""
    state = json.loads(request["state"])
    conversation = state.get("conversation") or []
    tokens = measure(request)
    info = {"target_prefix_tokens": target,
            "measured_prefix_tokens": tokens,
            "rewindowed": False,
            "dropped_pointers": [],
            "final_prefix_tokens": tokens,
            "overflow": False}
    if tokens <= target:
        return request, info

    protected = protected_indices(conversation, state.get("candidate_pointer"))
    if protected is None:
        # Candidate pointer does not resolve; dropping blindly could remove
        # the candidate itself. Fail open to abstention, request untouched.
        info["overflow"] = True
        info["reason"] = "candidate_pointer_not_in_conversation"
        return request, info

    removable = [i for i in range(len(conversation)) if i not in protected]
    dropped = set()
    current = request
    while tokens > target and removable:
        drop = removable.pop(0)  # oldest non-protected first
        dropped.add(drop)
        new_state = dict(state)
        new_state["conversation"] = render_window(conversation, dropped)
        current = dict(request)
        current["state"] = serialized(new_state)
        tokens = measure(current)
    info["rewindowed"] = bool(dropped)
    info["dropped_pointers"] = [conversation[i].get("pointer")
                              for i in sorted(dropped)]
    info["final_prefix_tokens"] = tokens
    info["overflow"] = tokens > target
    if info["overflow"]:
        info["reason"] = "protected_floor_exceeds_target"
    return current, info


def make_measurer(url, timeout=30.0, retries=3):
    """Sequential POST /v1/winnow/inspect measurer. Localhost only — the
    endpoint needs no key but is unauthenticated, so refuse remote hosts."""
    parts = urlparse(url)
    if parts.scheme != "http" or parts.hostname not in LOCALHOSTS:
        raise ValueError(f"inspect endpoint must be plain http localhost: {url}")

    def measure(request):
        body = json.dumps(request).encode("utf-8")
        last = None
        for attempt in range(retries):
            conn = HTTPConnection(parts.hostname, parts.port, timeout=timeout)
            try:
                conn.request("POST", parts.path, body=body,
                             headers={"Content-Type": "application/json"})
                resp = conn.getresponse()
                data = json.loads(resp.read(8_000_000))
                if resp.status == 200:
                    return data["prefix_tokens"]
                last = f"status {resp.status}: {str(data)[:200]}"
            except Exception as exc:  # noqa: BLE001 - report and retry
                last = f"{type(exc).__name__}: {exc}"
            finally:
                conn.close()
            time.sleep(attempt + 1)
        raise RuntimeError(f"inspect failed after {retries} attempts: {last}")

    return measure


def rewindow_records(records, measure, target=TARGET_PREFIX_TOKENS):
    """Apply rewindow_request to each record; returns (out_records, summary)."""
    out_records, summary = [], {"records": 0, "over_target": 0,
                                "rewindowed": 0, "fit": 0, "overflow": 0}
    for rec in records:
        summary["records"] += 1
        new_request, info = rewindow_request(rec["request"], measure, target)
        meta = dict(rec.get("meta") or {})
        meta["winnow_rewindow"] = {
            "script": "scripts/rewindow_for_winnow_v1.py",
            "policy": "optional_recovery_path_partial_coverage_default",
            **info,
        }
        if info["measured_prefix_tokens"] > target:
            summary["over_target"] += 1
        if info["rewindowed"]:
            summary["rewindowed"] += 1
            meta["windowed"] = True
        if info["overflow"]:
            summary["overflow"] += 1
            meta["winnow_overflow"] = True
        else:
            summary["fit"] += 1
        out = dict(rec)
        out["request"] = new_request
        out["meta"] = meta
        out_records.append(out)
    return out_records, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--inspect-url", default=DEFAULT_INSPECT_URL)
    ap.add_argument("--target", type=int, default=TARGET_PREFIX_TOKENS)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    records = [json.loads(line)
               for line in args.candidates.open(encoding="utf-8")
               if line.strip()]
    if args.limit:
        records = records[: args.limit]
    measure = make_measurer(args.inspect_url)

    out_records, summary = rewindow_records(records, measure, args.target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for rec in out_records:
            stream.write(serialized(rec) + "\n")
    print(json.dumps({"event": "complete", "output": str(args.output),
                      **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
