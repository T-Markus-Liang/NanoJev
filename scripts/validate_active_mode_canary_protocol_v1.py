#!/usr/bin/env python3
"""Validate the active-mode canary protocol without enabling active filtering."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTOCOL = ROOT / "research" / "active_mode_canary_protocol_v1.json"
SCHEMA = "nanojev-active-mode-canary-protocol-v1"
BASELINE_HEADER = "x-nanojev-baseline-provider-prompt-tokens"
SIDECAR_HEADER = "x-nanojev-sidecar"
KILL_SWITCH_ENV = "NANOJEV_GATEWAY_KILL_SWITCH"
PHASE0_REQUIRED_CASES = [
    "active_eligible_drop",
    "provider_accounting_baseline",
    "anthropic_eligible_drop",
    "responses_eligible_drop",
    "unsupported_embeddings",
    "active_no_reduction",
    "shadow_control",
    "kill_switch",
    "scorer_error",
    "malformed_sidecar",
    "dependency_closure",
    "uncertain_score",
    "invalid_score_response",
    "scorer_timeout",
    "restore_manifest_too_large",
    "oversized_request_body",
    "internal_header_stripping",
]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message, failures):
    if not condition:
        failures.append(message)


def validate_protocol(path=DEFAULT_PROTOCOL, root=ROOT):
    path = Path(path)
    root = Path(root)
    failures = []
    protocol = json.loads(path.read_text(encoding="utf-8"))
    require(protocol.get("schema_version") == SCHEMA,
            "unsupported active-mode canary schema", failures)
    require(protocol.get("status") == "protocol_valid_not_authorized",
            "protocol status must remain protocol_valid_not_authorized", failures)

    owner = protocol.get("owner_authorization") or {}
    require(owner.get("required") is True,
            "owner authorization must be required", failures)
    require(owner.get("granted") is False,
            "owner authorization cannot be granted inside the protocol", failures)

    review = protocol.get("review_gate") or {}
    decision_path = root / review.get("decision_artifact", "")
    replay_path = root / review.get("replay_artifact", "")
    try:
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        require(decision.get("status") == review.get("required_status"),
                "Track A decision status mismatch", failures)
        require(sha256_file(decision_path) == review.get("decision_sha256"),
                "Track A decision hash mismatch", failures)
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        require(replay.get("ok") is review.get("required_replay_ok"),
                "Track A replay status mismatch", failures)
        require(sha256_file(replay_path) == review.get("replay_sha256"),
                "Track A replay hash mismatch", failures)
    except (OSError, ValueError) as error:
        failures.append(f"review artifact validation failed: {error}")

    cohorts = protocol.get("cohorts") or {}
    for name in ("development", "heldout_synthetic"):
        cohort = cohorts.get(name) or {}
        manifest_path = root / cohort.get("path", "")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            require(manifest.get("case_count") == cohort.get("case_count"),
                    f"{name}: case count mismatch", failures)
            require(sha256_file(manifest_path) == cohort.get("sha256"),
                    f"{name}: manifest hash mismatch", failures)
        except (OSError, ValueError) as error:
            failures.append(f"{name}: cohort validation failed: {error}")
    require((cohorts.get("real_context_holdout") or {}).get("status") == "pending",
            "real-context holdout must remain pending in v1", failures)

    policy = protocol.get("threshold_policy") or {}
    require(policy.get("production_threshold") is None,
            "production threshold must remain null", failures)
    require(policy.get("diagnostic_threshold") == 0.90,
            "canary diagnostic threshold must be 0.90", failures)

    accounting = protocol.get("provider_accounting") or {}
    require(accounting.get("actual_savings_requires_paired_provider_usage") is True,
            "actual savings must require paired provider usage", failures)
    require(accounting.get("baseline_header") == BASELINE_HEADER,
            "provider baseline header mismatch", failures)
    require(accounting.get("no_local_estimate_is_billing") is True,
            "local estimates must be excluded from billing claims", failures)

    safety = protocol.get("safety") or {}
    require(safety.get("fail_open_required") is True,
            "fail-open must be required", failures)
    require(safety.get("kill_switch_env") == KILL_SWITCH_ENV,
            "kill-switch environment variable mismatch", failures)
    require(safety.get("restore_manifest_header_max_bytes") == 4096,
            "restore-manifest header budget mismatch", failures)
    require(safety.get("content_free_receipts_required") is True,
            "content-free receipts must be required", failures)
    require(safety.get("raw_prompt_response_logging_forbidden") is True,
            "raw prompt/response logging must be forbidden", failures)
    require(safety.get("protected_segment_veto_required") is True,
            "protected-segment veto must be required", failures)
    require(safety.get("dependency_closure_required") is True,
            "dependency closure must be required", failures)
    require(safety.get("round_trip_verification_required") is True,
            "round-trip verification must be required", failures)
    require(safety.get("sidecar_header") == SIDECAR_HEADER,
            "trusted sidecar header mismatch", failures)
    require(type(safety.get("max_requests_per_phase")) is int and
            0 < safety.get("max_requests_per_phase") <= 25,
            "max_requests_per_phase must be a bounded integer <=25", failures)
    require(type(safety.get("max_body_bytes")) is int and
            safety.get("max_body_bytes") >= 1024,
            "max_body_bytes must be a positive bounded integer", failures)

    phases = protocol.get("phases") or []
    require([phase.get("id") for phase in phases] == [
        "phase0_loopback_fake_upstream",
        "phase1_shadow_provider_measure",
        "phase2_paired_canary",
    ], "phase order mismatch", failures)
    for phase in phases:
        require(phase.get("provider_calls_allowed") is False,
                f"{phase.get('id')}: provider calls must be disabled in protocol", failures)
    require(phases[0].get("upstream") == "loopback_fake_only",
            "phase0 must use loopback fake upstream only", failures)
    require(phases[0].get("requires_owner_authorization") is False,
            "phase0 loopback preflight does not require owner authorization", failures)
    require(phases[0].get("required_cases") == PHASE0_REQUIRED_CASES,
            "phase0 required case coverage mismatch", failures)
    for phase in phases[1:]:
        require(phase.get("requires_owner_authorization") is True,
                f"{phase.get('id')}: owner authorization must be required", failures)

    stops = set(protocol.get("stop_conditions") or [])
    for phrase in (
        "any protected segment is proposed for removal",
        "any required evidence string is missing from reduced request",
        "any paired downstream answer regresses",
        "any restore-manifest reconstruction fails",
        "receipt contains raw prompt, response, credential, or tool-output text",
        "owner revokes authorization or requests stop",
    ):
        require(phrase in stops, f"missing stop condition: {phrase}", failures)

    rollback = protocol.get("rollback") or {}
    require(KILL_SWITCH_ENV in str(rollback.get("primary")),
            "rollback must use the deterministic kill switch", failures)
    require(rollback.get("fallback_mode") == "shadow",
            "rollback fallback mode must be shadow", failures)
    require(rollback.get("expected_recovery_state") == "all requests forwarded unchanged",
            "rollback recovery state mismatch", failures)

    scorers = protocol.get("scorers") or {}
    require((scorers.get("strong_path") or {}).get("name") == "Winnow-12B",
            "strong path must remain Winnow-12B", failures)
    require((scorers.get("optional_fast_path") or {}).get("name") == "Kev-4B",
            "optional fast path must remain Kev-4B", failures)

    prohibited = set(protocol.get("prohibited") or [])
    for phrase in (
        "active filtering before explicit owner operational authorization",
        "production threshold selection",
        "provider billing savings claim without paired provider-reported usage",
        "remote scorer URLs",
    ):
        require(phrase in prohibited, f"missing prohibition: {phrase}", failures)

    return {
        "schema_version": "nanojev-active-mode-canary-preflight-v1",
        "protocol_path": str(path.resolve()),
        "protocol_sha256": sha256_file(path),
        "status": "protocol_valid_not_authorized" if not failures else "protocol_invalid",
        "active_filtering_allowed": False,
        "provider_calls_allowed": False,
        "checks": {
            "schema": True,
            "review_gate": True,
            "cohorts": True,
            "threshold_policy": True,
            "provider_accounting": True,
            "safety": True,
            "phases": True,
            "stop_conditions": True,
            "rollback": True,
            "scorer_identity": True,
        },
        "failures": failures,
        "ok": not failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = validate_protocol(args.protocol)
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
        print(json.dumps({"output": str(args.output),
                          "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                          "ok": receipt["ok"], "status": receipt["status"]}, indent=2))
    else:
        print(encoded, end="")
    if not receipt["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
