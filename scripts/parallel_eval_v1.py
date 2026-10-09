#!/usr/bin/env python3
"""Parallel shard driver over multiple valen-head sidecars.

eval_systemone_backend_v1.py is sequential; the model is single-sequence
per request and MPS underutilizes with one client. This shards pending row
indices across N worker threads, each pinned to its own sidecar port, and
appends result rows with the GLOBAL index i (same row schema). Safe to
resume: indices already present in --output are skipped.

    python3 scripts/parallel_eval_v1.py \
        --data data/real_context_eval_v1/eval.jsonl \
        --output results/lora_v5_frozen_main.jsonl \
        --ports 8095,8098,8099,8100 --timeout 90
"""
import argparse
import json
import threading
import time
import urllib.request
from pathlib import Path


def post(port, body, timeout):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/systemone",
        data=body.encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode()), (time.perf_counter() - t0) * 1000


def worker(port, jobs, data, out_path, timeout, retry, lock, stats):
    with open(out_path, "a") as out:
        for i in jobs:
            rec = data[i]
            body = json.dumps(rec["request"])
            status_ok, ans, ms = False, None, 0.0
            for attempt in range(retry):
                try:
                    data_resp, ms = post(port, body, timeout)
                    status_ok = True
                    break
                except Exception:
                    time.sleep(5 * (attempt + 1))
            row = {"i": i, "ms": round(ms, 1)}
            if status_ok and "answers" in data_resp:
                a = data_resp["answers"].get("irrelevant") or {}
                row["noul"] = a.get("noul")
                row["model"] = data_resp.get("model")
            else:
                row["noul"] = None
                row["error"] = "request_failed"
                with lock:
                    stats["errors"] += 1
            tgt = ((rec.get("targets") or {}).get("irrelevant") or {}
                   ).get("probabilities") or {}
            row["target"] = tgt.get("true")
            with lock:  # single-writer-per-file guarantee
                out.write(json.dumps(row) + "\n")
                out.flush()
            with lock:
                stats["done"] += 1
                if stats["done"] % 25 == 0:
                    print(json.dumps({"progress": stats["done"],
                                      "of": stats["total"],
                                      "errors": stats["errors"]}),
                          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ports", required=True, help="comma list")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--retry", type=int, default=3)
    args = ap.parse_args()

    data = [json.loads(l) for l in args.data.open() if l.strip()]
    done = set()
    if args.output.exists():
        done = {json.loads(l)["i"] for l in args.output.open()}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    pending = [i for i in range(len(data)) if i not in done]
    ports = [int(p) for p in args.ports.split(",")]
    shards = [pending[k::len(ports)] for k in range(len(ports))]
    stats = {"done": 0, "total": len(pending), "errors": 0}
    lock = threading.Lock()
    print(json.dumps({"pending": len(pending), "ports": ports}), flush=True)

    threads = [threading.Thread(target=worker,
                                args=(p, s, data, args.output, args.timeout,
                                      args.retry, lock, stats))
               for p, s in zip(ports, shards)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(json.dumps({"event": "complete", **stats}), flush=True)


if __name__ == "__main__":
    main()
