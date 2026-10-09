#!/usr/bin/env python3
"""Live fast/strong cascade check: Kev-4B (8092) → Winnow-12B Q8 (8091).

Runs synthetic scoring payloads through ``CascadeScorer`` against both live
services. Confident fast answers stay on the fast path; uncertain or failing
ones route to Winnow. Loopback only; no provider calls.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from context_gate_v1 import parse_segments, scoring_payload
from scorer_adapters_v1 import (CascadeScorer, ScorerError,
                                SystemOneHTTPScorer)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cascade_live_v1.json"
FAST_URL = "http://127.0.0.1:8092"
STRONG_URL = "http://127.0.0.1:8091"
FAST_MODEL = "kev-latest"
STRONG_MODEL = "Winnow-12B"


def sha256_text(data):
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def request_body():
    return {"model": "test", "messages": [
        {"role": "system", "content": "Never remove constraints"},
        {"role": "assistant",
         "content": "An unrelated old weather forecast for last year."},
        {"role": "assistant",
         "content": "SKU 123 is stored in warehouse zone B, aisle 4."},
        {"role": "user", "content": "Find the warehouse status for SKU 123."}]}


def run_cascade(fast_url=FAST_URL, strong_url=STRONG_URL):
    failures = []
    raw = json.dumps(request_body()).encode()
    segments = parse_segments(raw, "openai_chat")
    candidates = [s for s in segments
                  if s.pointer in ("/messages/1/content", "/messages/2/content")]
    payload = scoring_payload(segments, candidates)

    fast = SystemOneHTTPScorer(fast_url, timeout=30.0, model_id=FAST_MODEL)
    strong = SystemOneHTTPScorer(strong_url, timeout=30.0, model_id=STRONG_MODEL)

    # 1) Live cascade: confident fast answers stay on Kev-4B.
    cascade = CascadeScorer(fast, strong, fast_threshold=0.95)
    started = time.perf_counter()
    result = cascade(payload)
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    meta = result.get("checkpoint", {})
    scores = {state["id"]: state["answers"]["irrelevant"]["probabilities"]["true"]
              for state in result["states"]}
    if scores.get("segment_0", 0) < 0.9:
        failures.append("stale weather segment should score high irrelevant")
    if scores.get("segment_1", 1) > 0.1:
        failures.append("required evidence segment should be retained")
    # 2) Strong-fallback path: fast scorer that always raises forces Winnow.
    def dead_fast(_payload):
        raise ScorerError("simulated fast outage")
    fallback = CascadeScorer(dead_fast, strong, fast_threshold=0.95)
    started = time.perf_counter()
    fallback_result = fallback(payload)
    fallback_latency = round((time.perf_counter() - started) * 1000, 1)
    fb_meta = fallback_result.get("checkpoint", {})
    fb_scores = {state["id"]: state["answers"]["irrelevant"]["probabilities"]["true"]
                 for state in fallback_result["states"]}
    if len(fb_meta.get("strong_paths", [])) != len(candidates):
        failures.append("dead fast path should route all states to strong")
    if fb_scores.get("segment_1", 1) > 0.1:
        failures.append("strong path should retain required evidence")
    return {
        "schema_version": "nanojev-cascade-live-v1",
        "status": "cascade_live_pass" if not failures else "cascade_live_fail",
        "provider_calls": 0,
        "fast": {"url": fast_url, "model": FAST_MODEL},
        "strong": {"url": strong_url, "model": STRONG_MODEL},
        "cascade": {"latency_ms": latency_ms,
                    "fast_paths": meta.get("fast_paths"),
                    "strong_paths": meta.get("strong_paths"),
                    "fast_errors": meta.get("fast_errors"),
                    "scores": scores},
        "fallback": {"latency_ms": fallback_latency,
                     "fast_paths": fb_meta.get("fast_paths"),
                     "strong_paths": fb_meta.get("strong_paths"),
                     "fast_errors": fb_meta.get("fast_errors"),
                     "scores": fb_scores},
        "failures": failures,
        "ok": not failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast-url", default=FAST_URL)
    parser.add_argument("--strong-url", default=STRONG_URL)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    receipt = run_cascade(args.fast_url, args.strong_url)
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "sha256": sha256_text(encoded), "ok": receipt["ok"],
                      "status": receipt["status"],
                      "cascade_latency_ms": receipt["cascade"]["latency_ms"],
                      "fallback_latency_ms": receipt["fallback"]["latency_ms"]},
                     indent=2))
    if not receipt["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
