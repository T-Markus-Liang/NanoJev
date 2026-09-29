#!/usr/bin/env python3
"""Score a valen-format eval.jsonl against any /v1/systemone backend.

Posts each record's ``request`` body verbatim to the endpoint and records
the ``irrelevant`` noul probability + latency. Works for local backends
(127.0.0.1:8091/8092/8093/8094, or the 8876 unified service with
``?backend=...``) and for the official TypeSafe API when ``--api-key-file``
is given.

    # local winnow on v3 eval
    python3 scripts/eval_systemone_backend_v1.py \
        --url http://127.0.0.1:8091/v1/systemone \
        --data data/valen_nano_v3/eval.jsonl \
        --output results/winnow_v3_preds.jsonl

    # official Jev (paid calls; keep outputs local-only)
    python3 scripts/eval_systemone_backend_v1.py \
        --url https://api.typesafe.ai/v1/systemone \
        --api-key-file ~/.config/jev-eval/apikey \
        --data data/valen_nano_v3/eval.jsonl \
        --output results/jev_official_valen_nano_v3_preds.jsonl

Local-only receipt: raw provider probabilities must not be published.
"""
import argparse
import json
import time
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from urllib.parse import urlparse


def post(url, body, headers, timeout):
    parts = urlparse(url)
    conn_cls = HTTPSConnection if parts.scheme == "https" else HTTPConnection
    conn = conn_cls(parts.hostname, parts.port, timeout=timeout)
    path = parts.path + (f"?{parts.query}" if parts.query else "")
    started = time.perf_counter()
    try:
        conn.request("POST", path, body=body,
                     headers={"Content-Type": "application/json", **headers})
        resp = conn.getresponse()
        data = json.loads(resp.read(8_000_000))
        ms = (time.perf_counter() - started) * 1000
        return resp.status, data, ms
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--api-key-file", type=Path, default=None)
    ap.add_argument("--inject-model", default=None,
                    help="merge {\"model\": X} into each request body "
                         "(required by the official TypeSafe API)")
    ap.add_argument("--concurrency-note", action="store_true")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retry", type=int, default=3)
    args = ap.parse_args()

    headers = {}
    if args.api_key_file:
        headers["Authorization"] = (
            f"Bearer {args.api_key_file.read_text().strip()}")

    records = [json.loads(l) for l in args.data.open() if l.strip()]
    if args.limit:
        records = records[: args.limit]

    done = set()
    if args.output.exists():
        for l in args.output.open():
            done.add(json.loads(l)["i"])
        mode = "a"
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        mode = "w"

    out = args.output.open(mode)
    correct = n_done = timeouts = 0
    for i, rec in enumerate(records):
        if i in done:
            n_done += 1
            continue
        req = dict(rec["request"])
        if args.inject_model:
            req["model"] = args.inject_model
        body = json.dumps(req)
        status, data, ms = 0, None, 0.0
        for attempt in range(args.retry):
            try:
                status, data, ms = post(args.url, body, headers, args.timeout)
                if status == 200:
                    break
            except Exception:
                status = 0
            time.sleep(5 * (attempt + 1))
        row = {"i": i, "ms": round(ms, 1)}
        if status == 200 and data and "answers" in data:
            ans = data["answers"].get("irrelevant") or {}
            row["noul"] = ans.get("noul")
            row["model"] = data.get("model")
        else:
            row["noul"] = None
            row["error"] = f"status {status}"
            timeouts += 1
        tgt = ((rec.get("targets") or {}).get("irrelevant") or {}).get(
            "probabilities") or {}
        row["target"] = tgt.get("true")
        out.write(json.dumps(row) + "\n")
        out.flush()
        if (i + 1) % 50 == 0:
            print(json.dumps({"event": "progress", "records": i + 1,
                              "errors": timeouts}), flush=True)
    print(json.dumps({"event": "complete", "errors": timeouts,
                      "skipped_resumed": n_done}))


if __name__ == "__main__":
    main()
