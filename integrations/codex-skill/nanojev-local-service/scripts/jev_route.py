#!/usr/bin/env python3
"""jev-route — deterministic traffic split between official Jev and local NanoJev.

Same input/output shape as jev-eval. A payload hash decides the path, so the
same request always routes the same way (reproducible, auditable):

    jev-route < input.json                    # default: 30% local
    jev-route --local-ratio 0.3 < input.json  # explicit ratio
    JEV_ROUTE_LOCAL_RATIO=0.5 jev-route < input.json

Routing rule (mode=ratio): sha256(canonical payload) % 1000 < ratio*1000
→ local. Deterministic, reproducible, auditable.

Mode=quality: local answers first; when any answer's top probability is
below --escalate-below (default 0.95), the request escalates to official
Jev (`route.escalated: true`). This keeps confident traffic local and
sends only uncertain cases upstream — quality floor without a fixed ratio.

When the chosen local path is unreachable, falls back to official
(`fallback: true`). Official calls go through `jev-eval`, which handles
keys/routing per global rules. All decisions log to
~/.local/state/nanojev-eval/log.jsonl (content-free).
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

JEV_EVAL = os.environ.get("JEV_EVAL", str(Path.home() / ".local/bin/jev-eval"))
NANOJEV_EVAL = os.environ.get("NANOJEV_EVAL",
                              str(Path.home() / ".local/bin/nanojev-eval"))
LOG_PATH = Path.home() / ".local/state/nanojev-eval/log.jsonl"


def fail(message):
    print(json.dumps({"error": message}))
    sys.exit(1)


def log_event(event):
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        event["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False,
                                    allow_nan=False) + "\n")
    except OSError:
        pass


def route_bucket(payload):
    """Deterministic bucket in [0,1000) for this payload."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return int(hashlib.sha256(canonical.encode("utf-8")).hexdigest(), 16) % 1000


def run_cli(binary, payload):
    started = time.perf_counter()
    proc = subprocess.run([binary], input=json.dumps(payload),
                          capture_output=True, text=True, timeout=120)
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    if proc.returncode != 0:
        raise RuntimeError((proc.stdout or proc.stderr or
                            f"exit {proc.returncode}")[:200])
    result = json.loads(proc.stdout)
    result["elapsed_ms"] = elapsed
    return result


def answer_confidence(result):
    """Min top-probability across answers; None if the shape is unknown."""
    answers = result.get("answers")
    if not isinstance(answers, dict) or not answers:
        return None
    weakest = 1.0
    for answer in answers.values():
        if not isinstance(answer, dict):
            return None
        if isinstance(answer.get("noul"), (int, float)):
            value = answer["noul"]
            conf = max(value, 1 - value)
        elif isinstance(answer.get("probabilities"), dict) and \
                answer["probabilities"]:
            conf = max(answer["probabilities"].values())
        elif isinstance(answer.get("confidence"), (int, float)):
            conf = answer["confidence"]
        else:
            return None
        weakest = min(weakest, conf)
    return weakest


def main():
    argv = sys.argv[1:]
    mode = "ratio"
    for index, arg in enumerate(argv):
        if arg == "--mode" and index + 1 < len(argv):
            mode = argv[index + 1]
    if mode not in ("ratio", "quality"):
        fail("--mode must be ratio or quality")
    threshold = float(os.environ.get("JEV_ROUTE_ESCALATE_BELOW", "0.95"))
    for index, arg in enumerate(argv):
        if arg == "--escalate-below" and index + 1 < len(argv):
            threshold = float(argv[index + 1])
    ratio = float(os.environ.get("JEV_ROUTE_LOCAL_RATIO", "0.3"))
    for index, arg in enumerate(argv):
        if arg == "--local-ratio" and index + 1 < len(argv):
            ratio = float(argv[index + 1])
    if not 0.0 <= ratio <= 1.0:
        fail("--local-ratio must be in [0,1]")
    files = [a for i, a in enumerate(argv)
             if not a.startswith("--")
             and (i == 0 or argv[i - 1] not in
                  ("--local-ratio", "--mode", "--escalate-below"))]
    try:
        raw = Path(files[0]).read_text(encoding="utf-8") if files \
            else sys.stdin.read()
        payload = json.loads(raw)
    except Exception:
        fail("input must be JSON {state, questions}")

    if mode == "quality":
        result, route = run_quality(payload, threshold)
    else:
        result, route = run_ratio(payload, ratio)
    result["route"] = route
    log_event({"type": "route", "mode": mode, "planned": route["planned"],
               "actual": route["actual"],
               "elapsed_ms": result["elapsed_ms"],
               "local_ratio": route.get("local_ratio"),
               "local_confidence": route.get("local_confidence")})
    print(json.dumps(result, ensure_ascii=False))


def run_ratio(payload, ratio):
    go_local = route_bucket(payload) < ratio * 1000
    planned = "local" if go_local else "official"
    binary = NANOJEV_EVAL if go_local else JEV_EVAL
    try:
        result = run_cli(binary, payload)
        actual, fallback = planned, False
    except Exception:
        if not go_local:
            raise
        result = run_cli(JEV_EVAL, payload)   # local down → official
        actual, fallback = "official", True
    return result, {"mode": "ratio", "planned": planned, "actual": actual,
                    "local_ratio": ratio, "fallback": fallback}


def run_quality(payload, threshold):
    try:
        local = run_cli(NANOJEV_EVAL, payload)
    except Exception:
        result = run_cli(JEV_EVAL, payload)
        return result, {"mode": "quality", "planned": "local",
                        "actual": "official", "fallback": True}
    confidence = answer_confidence(local)
    if confidence is not None and confidence >= threshold:
        return local, {"mode": "quality", "planned": "local",
                       "actual": "local", "escalated": False,
                       "local_confidence": confidence,
                       "escalate_below": threshold}
    result = run_cli(JEV_EVAL, payload)
    return result, {"mode": "quality", "planned": "local",
                    "actual": "official", "escalated": True,
                    "local_confidence": confidence,
                    "escalate_below": threshold}


if __name__ == "__main__":
    main()
