#!/usr/bin/env python3
"""Winnow-12B Q8 concurrency and cache-isolation stress check.

Sends parallel `/v1/systemone` requests with distinct states and checks:
determinism (same request → identical probability), isolation (different
states don't contaminate each other's answers), and latency under parallel
load. Loopback only; no provider calls. Read-only for the server.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import time
from http.client import HTTPConnection
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "winnow_stress_v1.json"
WINNOW_URL = "http://127.0.0.1:8091"

STATES = [
    {"ticket": "I was charged twice. Please refund the duplicate payment."},
    {"ticket": "The app crashes on launch after the latest update."},
    {"ticket": "How do I export my data as CSV?"},
    {"ticket": "My order arrived damaged and I need a replacement."},
]
QUESTION = {"type": "noul",
            "instructions": "Does the customer request a refund or replacement?"}
EXPECTED_HIGH = {0, 3}  # refund/replacement tickets


def post_systemone(host, port, state, timeout=120.0):
    payload = json.dumps({"model": "Winnow-12B", "state": state,
                          "questions": {"refund": QUESTION}})
    connection = HTTPConnection(host, port, timeout=timeout)
    started = time.perf_counter()
    try:
        connection.request("POST", "/v1/systemone", body=payload,
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        body = json.loads(response.read(2_000_000))
        return {"status": response.status,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "noul": (body.get("answers") or {}).get("refund", {}).get("noul"),
                "usage": body.get("usage")}
    finally:
        connection.close()


def sha256_text(data):
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def run_stress(url=WINNOW_URL, workers=4, repeats=5):
    from urllib.parse import urlsplit
    parts = urlsplit(url)
    host, port = parts.hostname, parts.port
    failures = []

    status, _ = post_systemone(host, port, {"task": "health"}), None
    # Determinism: same request three times must give identical probabilities.
    baseline = [post_systemone(host, port, STATES[0])["noul"] for _ in range(3)]
    deterministic = len(set(baseline)) == 1
    if not deterministic:
        failures.append(f"non-deterministic answers: {baseline}")

    # Parallel: every state repeated `repeats` times across `workers` threads.
    tasks = [(index, rep) for index in range(len(STATES)) for rep in range(repeats)]
    latencies, by_state = [], {i: [] for i in range(len(STATES))}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(post_systemone, host, port, STATES[i]): i
                   for i, _ in tasks}
        for future, index in futures.items():
            result = future.result()
            latencies.append(result["latency_ms"])
            if result["status"] != 200 or result["noul"] is None:
                failures.append(f"state {index}: bad response {result['status']}")
                continue
            by_state[index].append(result["noul"])

    # Isolation: per-state answers consistent across concurrent calls, and
    # high/low groups are separated the right way.
    consistent = True
    separated = True
    for index, values in by_state.items():
        if len(set(values)) > 1:
            consistent = False
        mean = sum(values) / len(values) if values else None
        if mean is not None and ((index in EXPECTED_HIGH) != (mean > 0.5)):
            separated = False
    if not consistent:
        failures.append("per-state answers varied across concurrent calls")
    if not separated:
        failures.append("refund/replacement states not separated from others")

    latencies.sort()
    count = len(latencies)
    return {
        "schema_version": "nanojev-winnow-stress-v1",
        "status": "winnow_stress_pass" if not failures else "winnow_stress_fail",
        "provider_calls": 0,
        "scorer_url": url,
        "workers": workers,
        "total_calls": count,
        "deterministic": deterministic,
        "consistent_under_concurrency": consistent,
        "isolated": separated,
        "state_means": {str(i): round(sum(v) / len(v), 4)
                        for i, v in by_state.items() if v},
        "latency_ms": {"min": latencies[0] if latencies else None,
                       "p50": latencies[count // 2] if latencies else None,
                       "p95": latencies[int(count * 0.95)] if latencies else None,
                       "max": latencies[-1] if latencies else None},
        "failures": failures,
        "ok": not failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer-url", default=WINNOW_URL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    receipt = run_stress(args.scorer_url, args.workers, args.repeats)
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "sha256": sha256_text(encoded), "ok": receipt["ok"],
                      "status": receipt["status"],
                      "latency": receipt["latency_ms"]}, indent=2))
    if not receipt["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
