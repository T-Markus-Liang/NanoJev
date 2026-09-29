#!/usr/bin/env python3
"""nanojev-eval — local twin of jev-eval for the unified nanojev service.

Same input/output shape as `jev-eval` so agents can call local and official
Jev in parallel:

    nanojev_eval.py  input.json            # or stdin
    nanojev_eval.py  --backend cascade < input.json
    nanojev_eval.py  --context-gate < payload.json

Input  (systemone mode): {"state": ..., "questions": {id: {type, instructions, ...}}}
Output: {"answers": ..., "usage": ..., "model": ..., "backend": ...,
         "elapsed_ms": ..., "source": "nanojev-local"}

Input  (--context-gate): {"request": <wire body>, "wire_format"?, "sidecar"?,
                          "backend"?, "threshold"?, "max_scorer_payload_bytes"?}
Output: the content-free shadow receipt (pointers, reason codes, scores).

No API key. Loopback only. Advisory semantics: outputs never authorize actions.
"""
import json
import os
import subprocess
import sys
import time
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_URL = os.environ.get("NANOJEV_URL", "http://127.0.0.1:8876")
BACKENDS = ("winnow", "kev", "cascade")
SKILL = "/Users/markus/.codex/skills/nanojev-local-decider/scripts/nanojev_skill.py"
LOG_PATH = Path.home() / ".local/state/nanojev-eval/log.jsonl"


def log_event(event):
    """Append a content-free usage/feedback line (never payload text)."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        event["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False,
                                    allow_nan=False) + "\n")
    except OSError:
        pass


def fail(message):
    print(json.dumps({"error": message}))
    sys.exit(1)


def post(url, path, body, timeout=60.0):
    parsed = urlparse(url)
    connection = HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    try:
        connection.request("POST", path, body=json.dumps(body),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.read(4_000_000)
    finally:
        connection.close()


def ensure_service(url):
    parsed = urlparse(url)
    connection = HTTPConnection(parsed.hostname, parsed.port, timeout=2)
    try:
        connection.request("GET", "/api/health")
        if connection.getresponse().status == 200:
            return
    except Exception:
        pass
    finally:
        connection.close()
    try:
        subprocess.run([sys.executable, SKILL, "health", "--start"],
                       check=True, capture_output=True, timeout=240)
    except Exception:
        fail("nanojev service unreachable; run: bash scripts/start_local_stack.sh")


def read_input(argv):
    args, skip = [], False
    for index, arg in enumerate(argv):
        if skip:
            skip = False
            continue
        if arg in ("--backend", "--url"):
            skip = True
            continue
        if arg.startswith("--"):
            continue
        args.append(arg)
    if args:
        with open(args[-1], "r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.load(sys.stdin)


def backend_arg(argv):
    for index, arg in enumerate(argv):
        if arg == "--backend" and index + 1 < len(argv):
            backend = argv[index + 1]
            if backend not in BACKENDS:
                fail(f"backend must be one of {BACKENDS}")
            return backend
    return os.environ.get("NANOJEV_BACKEND", "winnow")


def main():
    argv = sys.argv[1:]
    if "--feedback" in argv:
        index = argv.index("--feedback")
        note = argv[index + 1] if index + 1 < len(argv) else None
        if not isinstance(note, str) or not note.strip():
            fail("--feedback requires a note; keep it free of secrets/prompts")
        log_event({"type": "agent_feedback", "note": note.strip()[:2000]})
        print(json.dumps({"recorded": True, "log": str(LOG_PATH)}))
        return
    context_gate = "--context-gate" in argv
    backend = backend_arg(argv)
    url = DEFAULT_URL
    for index, arg in enumerate(argv):
        if arg == "--url" and index + 1 < len(argv):
            url = argv[index + 1]
    try:
        payload = read_input(argv)
    except Exception:
        fail("input must be JSON ({state, questions} or context-gate payload)")
    ensure_service(url)
    started = time.perf_counter()
    if context_gate:
        if not isinstance(payload, dict) or not isinstance(payload.get("request"), dict):
            fail("context-gate input requires a 'request' object")
        status, body = post(url, "/v1/context-gate", payload)
        result = json.loads(body or b"{}")
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
        result["source"] = "nanojev-local"
        log_event({"type": "call", "endpoint": "context-gate",
                   "status": status, "elapsed_ms": result["elapsed_ms"],
                   "gate_status": result.get("status"),
                   "gate_reason": result.get("reason")})
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(0 if status == 200 else 1)
    if not isinstance(payload, dict) or "state" not in payload or \
            not isinstance(payload.get("questions"), dict):
        fail("input must be {state, questions:{id:{type,instructions,...}}}")
    status, body = post(url, f"/v1/systemone?backend={backend}", payload)
    if status != 200:
        fail(f"service returned {status}: {body[:200].decode('utf-8', 'replace')}")
    result = json.loads(body)
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    result["backend"] = backend
    result["source"] = "nanojev-local"
    log_event({"type": "call", "endpoint": "systemone", "backend": backend,
               "status": status, "elapsed_ms": result["elapsed_ms"],
               "model": result.get("model")})
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
