#!/usr/bin/env python3
"""Daily health check for local NanoJev/scorer services.

Read-only probes only — it never starts, stops, or reconfigures anything.
The NanoJev lifecycle service is the only required service; legacy scorer
endpoints are reported for information. ``--smoke`` additionally posts one
tiny evaluate request to prove the loaded model actually answers.
"""

import argparse
import json
import sys
import time
from http.client import HTTPConnection
from urllib.parse import urlsplit

NANOJEV_PORTS = [8765] + list(range(8876, 8891))
NANOJEV_HEALTH = "/api/health"
NANOJEV_EVALUATE = "/api/evaluate"
OPTIONAL_ENDPOINTS = [
    ("winnow_backend_8091", "http://127.0.0.1:8091/health"),
    ("kev_backend_8092", "http://127.0.0.1:8092/v1/models"),
]
RESTART_HINT = ("python3 /Users/markus/.codex/skills/nanojev-local-decider/"
                "scripts/nanojev_skill.py health --start")

SMOKE_PAYLOAD = {"states": [{
    "id": "health_smoke",
    "state": "ping",
    "questions": {"smoke": {"type": "boolean",
                            "instructions": "Is this a health smoke request?"}},
}]}


def http_json(host, port, path, payload=None, timeout=2.0):
    """Return ``(status, parsed_body_or_None)``; ``(None, None)`` when unreachable."""
    connection = HTTPConnection(host, port, timeout=timeout)
    try:
        if payload is None:
            connection.request("GET", path)
        else:
            connection.request("POST", path, body=json.dumps(payload),
                               headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        body = response.read(65_536)
        try:
            parsed = json.loads(body)
        except (ValueError, UnicodeError):
            parsed = None
        return response.status, parsed
    except Exception:
        return None, None
    finally:
        connection.close()


def probe_url(url, timeout=2.0):
    parts = urlsplit(url)
    started = time.perf_counter()
    status, body = http_json(parts.hostname, parts.port, parts.path or "/", timeout=timeout)
    return {"status_code": status, "body": body,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "up": status == 200}


def find_nanojev(timeout=2.0):
    """Probe the known NanoJev service ports; first ready answer wins."""
    for port in NANOJEV_PORTS:
        result = probe_url(f"http://127.0.0.1:{port}{NANOJEV_HEALTH}", timeout)
        if result["up"] and isinstance(result["body"], dict) and result["body"].get("ready"):
            return port, result
    return None, None


def smoke_evaluate(port, timeout=30.0):
    started = time.perf_counter()
    status, body = http_json("127.0.0.1", port, NANOJEV_EVALUATE,
                             payload=SMOKE_PAYLOAD, timeout=timeout)
    ok = status == 200 and isinstance(body, dict) and "states" in body
    return {"ran": True, "ok": ok,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1)}


DETERMINISM_PAYLOAD = {"state": {"probe": "determinism check"},
                       "questions": {"stable": {"type": "noul",
                                                "instructions": "Is this a determinism probe?"}}}


def determinism_probe(port, timeout=30.0):
    """POST the same /v1/systemone request twice; answers must be identical."""
    started = time.perf_counter()
    results = []
    for _ in range(2):
        status, body = http_json("127.0.0.1", port, "/v1/systemone",
                                 payload=DETERMINISM_PAYLOAD, timeout=timeout)
        if status != 200 or not isinstance(body, dict):
            return {"ran": True, "ok": False,
                    "error": f"status {status}",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
        results.append(body.get("answers"))
    return {"ran": True,
            "ok": results[0] == results[1],
            "identical": results[0] == results[1],
            "latency_ms": round((time.perf_counter() - started) * 1000, 1)}


def collect(timeout=2.0, smoke=False, smoke_timeout=30.0, determinism=False):
    services = []
    port, nanojev = find_nanojev(timeout)
    services.append({
        "name": "nanojev", "required": True,
        "url": f"http://127.0.0.1:{port}{NANOJEV_HEALTH}" if port else "none found (ports 8765, 8876-8890)",
        "status": "up" if nanojev else "down",
        "status_code": (nanojev or {}).get("status_code"),
        "latency_ms": (nanojev or {}).get("latency_ms"),
    })
    for name, url in OPTIONAL_ENDPOINTS:
        result = probe_url(url, timeout)
        services.append({"name": name, "required": False, "url": url,
                         "status": "up" if result["up"] else "down",
                         "status_code": result["status_code"],
                         "latency_ms": result["latency_ms"]})
    smoke_result = {"ran": False, "ok": None}
    if smoke and port is not None:
        smoke_result = smoke_evaluate(port, smoke_timeout)
    elif smoke:
        smoke_result = {"ran": True, "ok": False, "error": "service_down"}
    determinism_result = {"ran": False, "ok": None}
    if determinism and port is not None:
        determinism_result = determinism_probe(port, smoke_timeout)
    elif determinism:
        determinism_result = {"ran": True, "ok": False, "error": "service_down"}
    ok = all(service["status"] == "up" for service in services if service["required"])
    if smoke_result.get("ok") is False or determinism_result.get("ok") is False:
        ok = False
    return {
        "schema_version": "nanojev-local-services-health-v1",
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ok": ok,
        "nanojev_port": port,
        "services": services,
        "smoke": smoke_result,
        "determinism": determinism_result,
        "restart_hint": RESTART_HINT,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="also POST one tiny evaluate request to the NanoJev service")
    parser.add_argument("--determinism", action="store_true",
                        help="POST the same /v1/systemone probe twice and require identical answers")
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--smoke-timeout", type=float, default=30.0)
    parser.add_argument("--output", type=str, default=None,
                        help="optional JSON receipt path")
    args = parser.parse_args()
    receipt = collect(timeout=args.timeout, smoke=args.smoke,
                      smoke_timeout=args.smoke_timeout,
                      determinism=args.determinism)
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(encoded)
    print(encoded, end="")
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
