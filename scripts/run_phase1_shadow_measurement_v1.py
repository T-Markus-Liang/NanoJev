#!/usr/bin/env python3
"""Phase1 shadow measurement — owner-granted synthetic scope.

Executes the Phase1 shadow within the bounded scope grant
(``research/phase1_shadow_scope_synthetic_v1.json``): synthetic fixture
traffic through the loopback gateway in shadow mode, scored by the live
Winnow-12B Q8 service. Original bytes always go upstream; nothing is applied;
no provider calls. The runner validates the grant before sending a single
request and stops if any scope term is violated.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from scorer_adapters_v1 import SystemOneHTTPScorer
from run_active_mode_phase0_v1 import run_case

ROOT = Path(__file__).resolve().parent.parent
GRANT = ROOT / "research" / "phase1_shadow_scope_synthetic_v1.json"
OUT = ROOT / "results" / "phase1_shadow_measurement_v1.json"
WINNOW_URL = "http://127.0.0.1:8091"
WINNOW_MODEL = "Winnow-12B"


def sha256_text(data):
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def load_grant(path):
    grant = json.loads(Path(path).read_text(encoding="utf-8"))
    failures = []
    scope = grant.get("scope") or {}
    if grant.get("status") != "granted_synthetic_scope_only":
        failures.append("grant status is not a synthetic-scope grant")
    for field in ("granted_at", "granted_by", "expires_at"):
        if not grant.get(field):
            failures.append(f"grant missing {field}")
    if scope.get("provider_calls_allowed") is not False:
        failures.append("scope must forbid provider calls")
    if scope.get("active_filtering_allowed") is not False:
        failures.append("scope must forbid active filtering")
    if scope.get("reduced_bytes_may_be_sent") is not False:
        failures.append("scope must forbid sending reduced bytes")
    if scope.get("real_user_data_allowed") is not False:
        failures.append("scope must forbid real user data")
    if scope.get("mode") != "shadow":
        failures.append("scope mode must be shadow")
    max_requests = scope.get("max_requests")
    if type(max_requests) is not int or not 1 <= max_requests <= 25:
        failures.append("scope.max_requests must be an int in [1,25]")
    traffic = scope.get("traffic_source")
    if not isinstance(traffic, str) or not (ROOT / traffic.split(" ")[0]).is_file():
        failures.append("scope traffic_source must name an existing file")
    return grant, scope, failures


class CountingScorer:
    def __init__(self, scorer):
        self.scorer = scorer
        self.calls = 0
        self.latencies = []

    def __call__(self, payload):
        self.calls += 1
        started = time.perf_counter()
        result = self.scorer(payload)
        self.latencies.append(round((time.perf_counter() - started) * 1000, 1))
        return result


def run_measurement(grant_path=GRANT):
    grant, scope, failures = load_grant(grant_path)
    cases_out, aggregates = [], {"unsafe_proposals": 0, "changed_bytes": 0,
                                 "restore_headers": 0}
    if failures:
        return {"schema_version": "nanojev-phase1-shadow-measurement-v1",
                "status": "scope_grant_invalid", "provider_calls": 0,
                "failures": failures, "ok": False}

    manifest_path = ROOT / scope["traffic_source"].split(" ")[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest.get("cases") or []
    if len(cases) > scope["max_requests"]:
        failures.append(f"traffic has {len(cases)} cases > budget "
                        f"{scope['max_requests']}")
        return {"schema_version": "nanojev-phase1-shadow-measurement-v1",
                "status": "scope_budget_exceeded", "provider_calls": 0,
                "failures": failures, "ok": False}

    scorer = CountingScorer(SystemOneHTTPScorer(
        WINNOW_URL, timeout=30.0, model_id=WINNOW_MODEL))
    proposal_totals = {"proposed": 0, "retained": 0}
    for case in cases:
        wire = case.get("wire_format") or "openai_chat"
        raw = json.dumps(case["body"], ensure_ascii=False).encode("utf-8")
        sidecar = case.get("sidecar") or {}
        eligible = {pointer for pointer, meta in
                    (sidecar.get("segments") or {}).items()
                    if isinstance(meta, dict) and meta.get("eligible") is True}
        result, sent, manifest_header = run_case(
            case["case_id"], mode="shadow", wire=wire, scorer=scorer,
            sidecar=sidecar, raw=raw,
            path="/v1/messages" if wire == "anthropic_messages"
            else "/v1/responses" if wire == "openai_responses"
            else "/v1/chat/completions")
        proposed = result.get("proposed_pointers") or []
        unsafe = [pointer for pointer in proposed if pointer not in eligible]
        aggregates["unsafe_proposals"] += len(unsafe)
        if sent != raw:
            aggregates["changed_bytes"] += 1
        if manifest_header is not None:
            aggregates["restore_headers"] += 1
        proposal_totals["proposed" if proposed else "retained"] += 1
        cases_out.append({
            "case_id": case["case_id"], "wire_format": wire,
            "gate_reason": result.get("gate_reason"),
            "forward_reason": result.get("forward_reason"),
            "forwarded_unchanged": sent == raw,
            "restore_header_present": manifest_header is not None,
            "proposed_pointers": proposed,
            "unsafe_proposals": unsafe,
            "expected_gate_reason": (case.get("expected_gate") or {}).get("reason"),
            "expected_all_retain": (case.get("expected_gate") or {}).get("all_retain"),
            "receipt_sha256": result.get("receipt_sha256"),
        })
    for case_result in cases_out:
        if case_result["unsafe_proposals"]:
            failures.append(f"{case_result['case_id']}: unsafe proposal "
                            f"{case_result['unsafe_proposals']}")
        if case_result["forwarded_unchanged"] is not True:
            failures.append(f"{case_result['case_id']}: bytes changed upstream")
        if case_result["restore_header_present"]:
            failures.append(f"{case_result['case_id']}: shadow emitted "
                            "restore manifest")
    return {
        "schema_version": "nanojev-phase1-shadow-measurement-v1",
        "status": "phase1_shadow_measurement_pass" if not failures
        else "phase1_shadow_measurement_fail",
        "provider_calls": 0,
        "grant": {"artifact_id": grant.get("artifact_id"),
                  "granted_at": grant.get("granted_at"),
                  "granted_by": grant.get("granted_by"),
                  "expires_at": grant.get("expires_at"),
                  "max_requests": scope.get("max_requests")},
        "traffic_manifest": str(manifest_path.relative_to(ROOT)),
        "scorer": {"kind": "systemone", "url": WINNOW_URL, "model": WINNOW_MODEL},
        "scorer_calls": scorer.calls,
        "scorer_latency_ms": {"min": min(scorer.latencies) if scorer.latencies else None,
                              "mean": round(sum(scorer.latencies) / len(scorer.latencies), 1)
                              if scorer.latencies else None,
                              "max": max(scorer.latencies) if scorer.latencies else None},
        "requests_measured": len(cases_out),
        "cases": cases_out,
        "proposal_totals": proposal_totals,
        "aggregates": aggregates,
        "failures": failures,
        "ok": not failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grant", type=Path, default=GRANT)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    receipt = run_measurement(args.grant)
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "sha256": sha256_text(encoded), "ok": receipt["ok"],
                      "status": receipt["status"],
                      "requests": receipt.get("requests_measured")},
                     indent=2))
    if not receipt["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
