#!/usr/bin/env python3
"""T10/A5 filter-versus-rebuild protocol, preflight, and synthetic controls.

This module is the frozen contract layer for the A5 comparison. It:

* validates the frozen protocol ``research/nanojev_v2_t10_a5_filter_vs_rebuild_protocol_v1.json``;
* binds the protocol to the frozen A4 manifest by exact SHA-256 and never copies
  fixture content into the protocol or into any receipt;
* derives/verifies the fixture-kind-to-task-family map and the protected families;
* runs a read-only preflight that fails closed on schema, hash, enum,
  authorization, network, raw-content, and path mismatches;
* emits synthetic placeholder receipts only (no provider, model, or network call);
* exposes a fail-closed aggregate gate in which any protected-family loss blocks
  an aggregate win.

The A4 manifest and ``scripts/build_tool_history_fixtures_v1.py`` are consumed
read-only and are never modified. No result is written unless the caller supplies
``--output``, and that write is deterministic and exclusive-create.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


PROTOCOL_SCHEMA = "nanojev-t10-a5-filter-vs-rebuild-protocol-v1"
RECEIPT_SCHEMA = "nanojev-t10-a5-synthetic-receipt-v1"
REPORT_SCHEMA = "nanojev-t10-a5-preflight-report-v1"
BUNDLE_SCHEMA = "nanojev-t10-a5-synthetic-bundle-v1"

STATUS_FROZEN = "frozen_before_measurement"
STATUS_OK = "protocol_valid_not_authorized"
STATUS_BLOCKED = "blocked_protocol_invalid"

EXECUTION_MODE = "synthetic_preflight_control_only"

EXPECTED_A4_REL_PATH = "research/tool_history_fixture_manifest_v1.json"
FROZEN_A4_MANIFEST_SHA256 = "0c8253c9424245315d3aaf1501a6c683b8cda39bc494f8c32005ab219adf89e0"
A4_MANIFEST_SCHEMA = "nanojev-tool-history-fixture-manifest-v1"
A4_SEED = 20260919
A4_CASE_COUNT = 11

ARMS = ("unfiltered_control", "safe_dedup", "relevance_filter", "abstractive_summary", "retrieval_rebuild")
METRICS = ("dependency_pair_retention", "evidence_fidelity", "protected_segment_deletion",
           "downstream_task_success", "restore_cost", "end_to_end_cost")
COST_METRICS = ("restore_cost", "end_to_end_cost")
METRIC_UNIT = "provider_tokens"
HARD_BLOCKING_FLAGS = ("non_repeatable", "credential_bearing", "contains_error", "stale_evidence",
                       "correction_dependent", "volatile_source")
AUTHORIZATION_FLAGS = ("network_access", "provider_calls", "model_load", "training", "quantization",
                       "deployment", "active_pruning", "result_publication", "adoption")
EXPECTED_FAMILIES = ("ambiguous_linkage", "correction_dependency", "credential_handling", "error_evidence",
                     "parallel_tool_use", "side_effect_safety", "snapshot_consistency", "tool_lookup")
REQUIRED_ACCEPTANCE = {
    "max_success_regression": 0.005,
    "min_median_token_reduction": 0.3,
    "protected_segment_deletion": 0,
    "require_paired_confidence_intervals": True,
    "report_per_task_family": True,
    "min_task_families": len(EXPECTED_FAMILIES),
}

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTOCOL = ROOT / "research" / "nanojev_v2_t10_a5_filter_vs_rebuild_protocol_v1.json"
DEFAULT_A4_MANIFEST = ROOT / EXPECTED_A4_REL_PATH

NULL_FIELDS = {
    "provider": ("provider_id", "prompt_tokens", "completion_tokens"),
    "quality": ("dependency_pair_retention", "evidence_fidelity", "downstream_task_success"),
    "physical": ("protected_segment_deletion", "wall_time_ms", "peak_memory_bytes"),
    "cost": ("restore_cost", "end_to_end_cost", "cost_basis", "currency"),
}


def serialized(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_protocol(path=DEFAULT_PROTOCOL):
    return load_json(path)


def _fail(errors, message):
    errors.append(message)


def derive_protected_families(a4_manifest, fixture_family_map):
    """Families are protected when any of their A4 kinds has a hard blocking flag
    or unresolved pairing. This is derived from the frozen A4 manifest, not asserted."""
    protected = set()
    for case in a4_manifest.get("cases", []):
        if not isinstance(case, dict):
            continue
        family = fixture_family_map.get(case.get("kind"))
        if family is None:
            continue
        unsafe = case.get("pairing_resolved") is not True
        for pair in case.get("pairing", []) or []:
            if not isinstance(pair, dict):
                unsafe = True
                continue
            if any(pair.get(flag) for flag in HARD_BLOCKING_FLAGS):
                unsafe = True
        if unsafe:
            protected.add(family)
    return protected


def _validate_source_binding(protocol, a4_manifest, a4_bytes, errors):
    binding = protocol.get("source_binding")
    if not isinstance(binding, dict):
        _fail(errors, "source_binding must be an object")
        return
    if binding.get("a4_manifest_path") != EXPECTED_A4_REL_PATH:
        _fail(errors, "source_binding.a4_manifest_path mismatch")
    if binding.get("a4_manifest_schema_version") != A4_MANIFEST_SCHEMA:
        _fail(errors, "source_binding.a4_manifest_schema_version mismatch")
    if binding.get("a4_manifest_sha256") != FROZEN_A4_MANIFEST_SHA256:
        _fail(errors, "source_binding.a4_manifest_sha256 does not match the frozen A4 digest")
    if binding.get("a4_manifest_seed") != A4_SEED:
        _fail(errors, "source_binding.a4_manifest_seed mismatch")
    if binding.get("a4_case_count") != A4_CASE_COUNT:
        _fail(errors, "source_binding.a4_case_count mismatch")
    if binding.get("read_only") is not True:
        _fail(errors, "source_binding.read_only must be true")
    if a4_bytes is not None:
        if sha256_hex(a4_bytes) != FROZEN_A4_MANIFEST_SHA256:
            _fail(errors, "A4 manifest bytes do not match the frozen SHA-256")
    if a4_manifest is not None:
        if a4_manifest.get("schema_version") != A4_MANIFEST_SCHEMA:
            _fail(errors, "A4 manifest schema_version mismatch")
        if a4_manifest.get("seed") != A4_SEED:
            _fail(errors, "A4 manifest seed mismatch")
        cases = a4_manifest.get("cases")
        if not isinstance(cases, list) or len(cases) != A4_CASE_COUNT:
            _fail(errors, "A4 manifest case count mismatch")


def _validate_families(protocol, a4_manifest, errors):
    family_map = protocol.get("fixture_family_map")
    if not isinstance(family_map, dict):
        _fail(errors, "fixture_family_map must be an object")
        return
    kinds = [case.get("kind") for case in a4_manifest.get("cases", []) if isinstance(case, dict)]
    if sorted(family_map) != sorted(kinds):
        _fail(errors, "fixture_family_map must cover exactly the frozen A4 kinds")
    if any(not isinstance(family, str) or not family for family in family_map.values()):
        _fail(errors, "fixture_family_map values must be non-empty strings")

    families = protocol.get("task_families")
    if not isinstance(families, list) or not families:
        _fail(errors, "task_families must be a non-empty list")
        return
    declared = {}
    for family in families:
        if not isinstance(family, dict):
            _fail(errors, "task_families entries must be objects")
            continue
        family_id = family.get("family_id")
        if not isinstance(family_id, str) or not family_id:
            _fail(errors, "task family_id must be a non-empty string")
            continue
        if family_id in declared:
            _fail(errors, f"duplicate task family {family_id!r}")
        if type(family.get("protected")) is not bool:
            _fail(errors, f"task family {family_id!r} protected must be a boolean")
        declared[family_id] = family
    if sorted(declared) != sorted(EXPECTED_FAMILIES):
        _fail(errors, "task_families must declare exactly the frozen family set")
    covered = {}
    for kind, family_id in family_map.items():
        covered.setdefault(family_id, set()).add(kind)
    for family_id, family in declared.items():
        if set(family.get("kinds", [])) != covered.get(family_id, set()):
            _fail(errors, f"task family {family_id!r} kinds do not match fixture_family_map")

    derived = derive_protected_families(a4_manifest, family_map)
    declared_protected = {family_id for family_id, family in declared.items() if family.get("protected")}
    if derived != declared_protected:
        _fail(errors, "protected task families do not match the A4-derived protection set")
    protected = protocol.get("protected_families")
    if not isinstance(protected, list) or sorted(protected) != sorted(derived):
        _fail(errors, "protected_families must equal the A4-derived protection set")


def _validate_arms(protocol, errors):
    arms = protocol.get("arms")
    if not isinstance(arms, list) or len(arms) != len(ARMS):
        _fail(errors, f"arms must declare exactly {list(ARMS)}")
        return
    ids = [arm.get("arm_id") if isinstance(arm, dict) else None for arm in arms]
    if tuple(ids) != ARMS:
        _fail(errors, f"arms must be exactly {list(ARMS)} in order")
    for index, arm in enumerate(arms):
        if not isinstance(arm, dict):
            _fail(errors, "arm entries must be objects")
            continue
        if arm.get("order") != index:
            _fail(errors, f"arm {arm.get('arm_id')!r} order must be {index}")
        if type(arm.get("lossy")) is not bool:
            _fail(errors, f"arm {arm.get('arm_id')!r} lossy must be a boolean")
        if arm.get("fail_open") is not True:
            _fail(errors, f"arm {arm.get('arm_id')!r} must fail open")
        if not isinstance(arm.get("transformation_class"), str) or not arm["transformation_class"]:
            _fail(errors, f"arm {arm.get('arm_id')!r} needs a transformation_class")


def _validate_metrics(protocol, errors):
    metrics = protocol.get("metrics")
    if not isinstance(metrics, list):
        _fail(errors, "metrics must be a list")
        return
    ids = [metric.get("metric_id") if isinstance(metric, dict) else None for metric in metrics]
    if sorted(ids) != sorted(METRICS):
        _fail(errors, f"metrics must be exactly {list(METRICS)}")
    for metric in metrics:
        if not isinstance(metric, dict):
            _fail(errors, "metric entries must be objects")
            continue
        metric_id = metric.get("metric_id")
        if metric_id in COST_METRICS and metric.get("unit") != METRIC_UNIT:
            _fail(errors, f"cost metric {metric_id!r} unit must be {METRIC_UNIT!r}")
        if metric_id == "protected_segment_deletion" and metric.get("must_be_zero") is not True:
            _fail(errors, "protected_segment_deletion must be must_be_zero")
    cost = protocol.get("cost_contract")
    if not isinstance(cost, dict):
        _fail(errors, "cost_contract must be an object")
        return
    if cost.get("cost_basis") != "provider_reported_or_tokenizer":
        _fail(errors, "cost_contract.cost_basis must be provider_reported_or_tokenizer")
    for flag in ("character_estimates_allowed", "manual_character_counts_allowed"):
        if cost.get(flag) is not False:
            _fail(errors, f"cost_contract.{flag} must be false")
    for flag in ("provider_prompt_tokens_required", "provider_completion_tokens_required",
                 "tokenizer_id_required_for_tokenizer_counts"):
        if cost.get(flag) is not True:
            _fail(errors, f"cost_contract.{flag} must be true")


def _validate_acceptance(protocol, errors):
    acceptance = protocol.get("acceptance")
    if not isinstance(acceptance, dict):
        _fail(errors, "acceptance must be an object")
        return
    for key, expected in REQUIRED_ACCEPTANCE.items():
        if key not in acceptance:
            _fail(errors, f"acceptance.{key} is required")
            continue
        value = acceptance[key]
        if type(expected) is bool:
            if type(value) is not bool or value is not expected:
                _fail(errors, f"acceptance.{key} must be exactly {str(expected).lower()}")
        elif type(expected) is int:
            if type(value) is not int or value != expected:
                _fail(errors, f"acceptance.{key} must be exactly {expected}")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != expected:
                _fail(errors, f"acceptance.{key} must be exactly {expected}")


def _validate_safety_and_auth(protocol, errors):
    safety = protocol.get("safety_contract")
    if not isinstance(safety, dict):
        _fail(errors, "safety_contract must be an object")
    else:
        expected = {
            "fail_open_action": "forward_original_bytes",
            "fail_open_on_every_arm": True,
            "summary_fallback_substituted_silently": False,
            "summary_fallback_is_separate_policy": True,
            "applied": False,
            "shadow_only": True,
            "synthetic_only": True,
            "protected_segment_deletion_must_be_zero": True,
            "protected_family_loss_blocks_aggregate_win": True,
            "equivalence_claims_forbidden": True,
            "active_pruning": False,
        }
        for key, value in expected.items():
            if safety.get(key) != value:
                _fail(errors, f"safety_contract.{key} must be {value!r}")
    authorization = protocol.get("authorization")
    if not isinstance(authorization, dict):
        _fail(errors, "authorization must be an object")
    else:
        if set(authorization) != set(AUTHORIZATION_FLAGS):
            _fail(errors, "authorization must declare exactly the frozen flag set")
        for flag in AUTHORIZATION_FLAGS:
            if authorization.get(flag) is not False:
                _fail(errors, f"authorization.{flag} must be false")
    for key in ("measurement_evidence", "model_loaded"):
        if protocol.get(key) is not False:
            _fail(errors, f"{key} must be false")
    if protocol.get("network_model_calls") != 0:
        _fail(errors, "network_model_calls must be 0")
    if protocol.get("synthetic_only") is not True:
        _fail(errors, "synthetic_only must be true")
    if protocol.get("execution_mode") != EXECUTION_MODE:
        _fail(errors, f"execution_mode must be {EXECUTION_MODE!r}")


def _validate_raw_content(protocol, a4_manifest, errors):
    """Fail closed if any frozen fixture leak token appears in the protocol."""
    rendered = serialized(protocol)
    for case in a4_manifest.get("cases", []):
        if not isinstance(case, dict):
            continue
        for token in case.get("leak_tokens", []) or []:
            if isinstance(token, str) and token and token in rendered:
                _fail(errors, f"protocol leaks raw fixture content for case {case.get('case_id')!r}")


def validate_protocol(protocol, a4_manifest=None, a4_bytes=None):
    """Return a list of integrity errors; an empty list means the protocol is sound."""
    errors = []
    if not isinstance(protocol, dict):
        return ["protocol must be a JSON object"]
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        _fail(errors, f"schema_version must be {PROTOCOL_SCHEMA!r}")
    if protocol.get("status") != STATUS_FROZEN:
        _fail(errors, f"status must be {STATUS_FROZEN!r}")
    if protocol.get("frozen") is not True:
        _fail(errors, "frozen must be true")
    if not isinstance(protocol.get("protocol_id"), str) or not protocol["protocol_id"]:
        _fail(errors, "protocol_id must be a non-empty string")
    if a4_manifest is None:
        if a4_bytes is None:
            try:
                a4_bytes = DEFAULT_A4_MANIFEST.read_bytes()
            except OSError as error:
                _fail(errors, f"A4 manifest unreadable: {error}")
                return errors
        try:
            a4_manifest = json.loads(a4_bytes.decode("utf-8"))
        except (ValueError, UnicodeError) as error:
            _fail(errors, f"A4 manifest unreadable: {error}")
            return errors
    _validate_source_binding(protocol, a4_manifest, a4_bytes, errors)
    _validate_families(protocol, a4_manifest, errors)
    _validate_arms(protocol, errors)
    _validate_metrics(protocol, errors)
    _validate_acceptance(protocol, errors)
    _validate_safety_and_auth(protocol, errors)
    _validate_raw_content(protocol, a4_manifest, errors)
    return errors


def preflight(protocol_path=DEFAULT_PROTOCOL, a4_manifest_path=DEFAULT_A4_MANIFEST):
    """Read-only preflight. Never writes anything; fails closed on any mismatch."""
    errors = []
    canonical = (ROOT / EXPECTED_A4_REL_PATH).resolve()
    target = Path(a4_manifest_path).resolve()
    if target != canonical:
        _fail(errors, "a4_manifest_path_mismatch")
        target = canonical
    a4_bytes = None
    if not target.exists():
        _fail(errors, "a4_manifest_missing")
    else:
        a4_bytes = target.read_bytes()
    try:
        protocol = load_protocol(protocol_path)
        protocol_bytes = Path(protocol_path).read_bytes()
    except (OSError, ValueError, UnicodeError) as error:
        protocol = {"protocol_id": None}
        protocol_bytes = b""
        _fail(errors, f"protocol_unreadable: {error}")
    a4_manifest = None
    if a4_bytes is not None:
        try:
            a4_manifest = json.loads(a4_bytes.decode("utf-8"))
        except (ValueError, UnicodeError):
            _fail(errors, "a4_manifest_unreadable")
    errors.extend(validate_protocol(protocol, a4_manifest=a4_manifest, a4_bytes=a4_bytes))
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": STATUS_BLOCKED if errors else STATUS_OK,
        "protocol_id": protocol.get("protocol_id") if isinstance(protocol, dict) else None,
        "protocol_sha256": sha256_hex(protocol_bytes),
        "a4_manifest": {
            "path": EXPECTED_A4_REL_PATH,
            "sha256": FROZEN_A4_MANIFEST_SHA256,
            "observed_sha256": sha256_hex(a4_bytes) if a4_bytes is not None else None,
            "schema_version": a4_manifest.get("schema_version") if isinstance(a4_manifest, dict) else None,
            "seed": a4_manifest.get("seed") if isinstance(a4_manifest, dict) else None,
            "case_count": len(a4_manifest.get("cases", [])) if isinstance(a4_manifest, dict) else None,
        },
        "arms": list(ARMS),
        "metrics": list(METRICS),
        "families": sorted(EXPECTED_FAMILIES),
        "protected_families": sorted(derive_protected_families(a4_manifest, protocol.get("fixture_family_map", {})))
                             if isinstance(protocol, dict) and isinstance(protocol.get("fixture_family_map"), dict)
                             and isinstance(a4_manifest, dict) else [],
        "synthetic_only": True,
        "measurement_evidence": False,
        "network_model_calls": 0,
        "model_loaded": False,
        "authorization": {flag: False for flag in AUTHORIZATION_FLAGS},
        "wrote_output": False,
        "errors": errors,
    }
    return report


def synthetic_receipt(arm_id, order, protocol_sha256=None):
    """A deterministic placeholder. It carries no provider, quality, physical, or cost value."""
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "receipt_id": f"synthetic-{arm_id}-placeholder",
        "receipt_type": "synthetic_placeholder",
        "arm_id": arm_id,
        "arm_order": order,
        "protocol_sha256": protocol_sha256,
        "applied": False,
        "synthetic_only": True,
        "measurement_evidence": False,
        "model_loaded": False,
        "network_model_calls": 0,
        "fail_open": {
            "action": "forward_original_bytes",
            "forwarded_unchanged": True,
            "summary_substituted": False,
        },
        "provider": {field: None for field in NULL_FIELDS["provider"]},
        "quality": {field: None for field in NULL_FIELDS["quality"]},
        "physical": {field: None for field in NULL_FIELDS["physical"]},
        "cost": {field: None for field in NULL_FIELDS["cost"]},
        "authorization": {flag: False for flag in AUTHORIZATION_FLAGS},
        "paired_over_families": sorted(EXPECTED_FAMILIES),
        "notes": "Placeholder only; not measurement evidence; never authorization.",
    }
    receipt["receipt_sha256"] = None
    receipt["receipt_sha256"] = sha256_hex(canonical_bytes(receipt))
    return receipt


def synthetic_receipts(protocol=None, protocol_sha256=None):
    """One deterministic placeholder receipt per frozen arm."""
    arms = protocol.get("arms") if isinstance(protocol, dict) else None
    if not isinstance(arms, list):
        arms = [{"arm_id": arm, "order": index} for index, arm in enumerate(ARMS)]
    return [synthetic_receipt(arm["arm_id"], arm["order"], protocol_sha256) for arm in arms]


def build_bundle(protocol_path=DEFAULT_PROTOCOL, a4_manifest_path=DEFAULT_A4_MANIFEST):
    """Return the deterministic preflight report plus placeholder receipts."""
    report = preflight(protocol_path, a4_manifest_path)
    try:
        protocol = load_protocol(protocol_path)
    except (OSError, ValueError, UnicodeError):
        protocol = None
    return {
        "schema_version": BUNDLE_SCHEMA,
        "status": report["status"],
        "protocol_sha256": report["protocol_sha256"],
        "preflight": report,
        "receipts": synthetic_receipts(protocol, report["protocol_sha256"]),
        "synthetic_only": True,
        "measurement_evidence": False,
        "authorization": {flag: False for flag in AUTHORIZATION_FLAGS},
    }


def aggregate_gate(protocol, family_results):
    """Fail-closed aggregate decision over per-family results.

    ``family_results`` maps family_id -> {
        "measurement_evidence": bool,
        "paired_confidence_intervals": bool,
        "protected_segment_deletion": int,
        "success_regression": float,        # positive means worse than control
        "token_reduction_fraction": float,  # positive means fewer tokens than control
    }

    Any protected-family loss (protected-segment deletion or a success regression
    beyond ``acceptance.max_success_regression``) blocks an aggregate win outright.
    Missing families or absent measurement evidence also block. The synthetic
    receipts in this package are never measurement evidence, so this gate does not
    authorize anything by itself.
    """
    accepted = protocol.get("acceptance", {}) if isinstance(protocol, dict) else {}
    max_regression = accepted.get("max_success_regression", 0.0)
    min_reduction = accepted.get("min_median_token_reduction", 1.0)
    protected = set(protocol.get("protected_families", [])) if isinstance(protocol, dict) else set()
    families = [family["family_id"] for family in protocol.get("task_families", [])] \
        if isinstance(protocol, dict) else []
    blocked_by = []
    measured = True
    reductions = []
    for family in families:
        result = family_results.get(family)
        if not isinstance(result, dict):
            blocked_by.append({"family": family, "reason": "missing_family_result"})
            measured = False
            continue
        deletion = result.get("protected_segment_deletion")
        regression = result.get("success_regression")
        reduction = result.get("token_reduction_fraction")
        if result.get("measurement_evidence") is not True:
            blocked_by.append({"family": family, "reason": "measurement_evidence_absent"})
            measured = False
        if result.get("paired_confidence_intervals") is not True:
            blocked_by.append({"family": family, "reason": "paired_confidence_intervals_absent"})
            measured = False
        if type(deletion) is not int or deletion < 0:
            blocked_by.append({"family": family, "reason": "invalid_protected_segment_deletion"})
        elif family in protected and deletion > 0:
            blocked_by.append({"family": family, "reason": "protected_family_segment_deletion"})
        if isinstance(regression, bool) or not isinstance(regression, (int, float)):
            blocked_by.append({"family": family, "reason": "invalid_success_regression"})
        elif family in protected and regression > max_regression:
            blocked_by.append({"family": family, "reason": "protected_family_success_regression"})
        if isinstance(reduction, bool) or not isinstance(reduction, (int, float)):
            blocked_by.append({"family": family, "reason": "invalid_token_reduction_fraction"})
        else:
            reductions.append(float(reduction))
    if any(entry["reason"].startswith("protected_family") for entry in blocked_by):
        reason = "protected_family_loss"
    elif blocked_by:
        reason = "gate_incomplete"
    elif not measured:
        reason = "measurement_evidence_absent"
    else:
        reason = "no_measurement_evidence"
    aggregate_win = False
    if not blocked_by and measured:
        ordered = sorted(reductions)
        middle = len(ordered) // 2
        median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
        aggregate_win = median >= min_reduction and all(
            isinstance(family_results[family].get("success_regression"), (int, float))
            and family_results[family]["success_regression"] <= max_regression for family in families)
    return {
        "aggregate_win": aggregate_win,
        "blocked": bool(blocked_by),
        "reason": reason,
        "blocked_by": blocked_by,
        "measurement_evidence": measured and bool(families),
    }


def write_exclusive(path, payload):
    """Deterministic canonical write that refuses to overwrite an existing file."""
    data = canonical_bytes(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "xb") as handle:
        handle.write(data)
    return data


def self_test(protocol_path=DEFAULT_PROTOCOL, a4_manifest_path=DEFAULT_A4_MANIFEST):
    report = preflight(protocol_path, a4_manifest_path)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL,
                        help="frozen A5 protocol JSON (read-only)")
    parser.add_argument("--a4-manifest", type=Path, default=DEFAULT_A4_MANIFEST,
                        help="frozen A4 manifest JSON (read-only)")
    parser.add_argument("--output", type=Path, help="write the synthetic bundle with exclusive create")
    parser.add_argument("--stdout", action="store_true", help="print the synthetic bundle to stdout")
    parser.add_argument("--self-test", action="store_true", help="validate the protocol and A4 binding without writing")
    args = parser.parse_args(argv)
    selected = sum([args.self_test, args.output is not None, args.stdout])
    if selected != 1:
        parser.error("choose exactly one of --self-test, --output, --stdout")
    if args.self_test:
        report = self_test(args.protocol, args.a4_manifest)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["status"] == STATUS_OK else 1
    bundle = build_bundle(args.protocol, args.a4_manifest)
    if args.stdout:
        print(json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if bundle["status"] == STATUS_OK else 1
    try:
        write_exclusive(args.output, bundle)
    except FileExistsError:
        print(json.dumps({"status": "failed", "error": "output_exists"}, indent=2))
        return 2
    print(json.dumps({"status": bundle["status"], "output": str(args.output)}, indent=2))
    return 0 if bundle["status"] == STATUS_OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
