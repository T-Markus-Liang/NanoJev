#!/usr/bin/env python3
"""Fail-closed validator for the T13 risk-gate amendment protocol.

T13 is an additive arm on the T12 real-data validation protocol. This validator
pins the T13 protocol file, re-runs the *entire* T12 preflight (every upstream
hash, the re-derived R1 folds, the point-in-time regime masks, the execution
policy values, the instrument-group partition, and the 45-cell plan), then adds
the T13-specific checks: the gate module and adapter hashes, the declared
limits against the frozen R1 ``limits.*`` set and the ``RiskLimits`` defaults,
and the fail-closed authorization boundary. It emits a preflight receipt only;
it performs no replay and no fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_nanojev_v2 import file_identity  # noqa: E402
import financial_risk_v1 as risk  # noqa: E402
import financial_real_data_risk_v1 as gated  # noqa: E402
import validate_financial_validation_protocol_v1 as t12_validator  # noqa: E402

SCHEMA = "nanojev-financial-risk-gate-protocol-v1"
EXPECTED_PROTOCOL_SHA256 = "c2103fd5540a5ccd7a13de3b6339d1e1867cd1b963864ace6d733e6abc15ae32"

T12_PROTOCOL = ROOT / "research/financial_validation_protocol_v1.json"
BOUND_RECEIPT = ROOT / "results/financial_pit_r1_bound_receipt_20260920_v5.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _under_root(path):
    path = Path(path).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError:
        raise ValueError(f"artifact path escapes project root: {path}") from None
    return path


def _verify_pinned(relative_path, expected_sha256, label):
    path = _under_root(ROOT / relative_path)
    if _sha256(path) != expected_sha256:
        raise ValueError(f"{label} hash mismatch: {relative_path}")
    return file_identity(path)


def _check_t12_binding(amendment):
    """Return the resolved bound T12 protocol path; raises on any drift."""
    t12_protocol_path = _under_root(ROOT / amendment["protocol_path"])
    if t12_protocol_path != T12_PROTOCOL.resolve():
        raise ValueError("T13 protocol does not bind the T12 protocol file")
    if _sha256(t12_protocol_path) != amendment["protocol_sha256"]:
        raise ValueError("bound T12 protocol hash mismatch")
    return t12_protocol_path


def _check_authorization_block(protocol):
    auth = protocol["authorization"]
    for flag in ("real_replay_performed", "replay_authorized", "measurement_authorized",
                 "training_authorized", "order_submission_authorized",
                 "live_trading_authorized"):
        if auth.get(flag) is not False:
            raise ValueError(f"protocol authorization flag {flag} is not fail-closed")
    if auth.get("network_model_calls") != 0:
        raise ValueError("protocol may not declare network model calls")


def _check_gate_declaration(protocol):
    gate = protocol["risk_gate"]
    identities = {
        "engine_module": _verify_pinned(gate["engine_module"],
                                        gate["engine_module_sha256"], "risk engine module"),
        "adapter_module": _verify_pinned(gate["adapter_module"],
                                         gate["adapter_module_sha256"], "gate adapter module"),
    }
    if gate.get("engine") != gated.GATE_NAME:
        raise ValueError("declared gate engine is not financial_risk_v1.RiskEngine")
    if gate.get("policy_version") != risk.RISK_POLICY_VERSION:
        raise ValueError("declared gate policy version differs from the engine's")
    frozen = gated.load_frozen_risk_limits()
    declared = gate["limits"]
    if declared != frozen.to_dict():
        raise ValueError("declared limits differ from the frozen R1 limits.* set")
    if declared != risk.RiskLimits().to_dict():
        raise ValueError("declared limits differ from the RiskLimits defaults")
    if gate.get("latch_kill_switch_on_block"):
        raise ValueError("kill-switch latching must stay disabled in T13")
    return identities


def validate(protocol_path, receipt_path=BOUND_RECEIPT):
    protocol_path = _under_root(protocol_path)
    receipt_path = _under_root(receipt_path)
    if EXPECTED_PROTOCOL_SHA256.startswith("REPLACE"):
        raise ValueError("validator is unpinned: EXPECTED_PROTOCOL_SHA256 placeholder")
    if _sha256(protocol_path) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("T13 protocol hash is not the pinned review artifact")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != SCHEMA:
        raise ValueError("wrong T13 protocol schema")
    if protocol.get("status") != "frozen_preflight_not_replay_authorized":
        raise ValueError("protocol is not the frozen preflight version")

    # T12 binding: the amendment must name the pinned T12 protocol and receipt,
    # and the whole T12 preflight must still pass on the real artifacts.
    amendment = protocol["amendment_of"]
    t12_protocol_path = _check_t12_binding(amendment)
    _verify_pinned(amendment["t12_run_receipt_path"],
                   amendment["t12_run_receipt_sha256"], "T12 run receipt")
    t12_preflight = t12_validator.validate(t12_protocol_path, receipt_path)

    _check_authorization_block(protocol)
    gate_identities = _check_gate_declaration(protocol)

    scope = protocol["replay_scope"]
    if scope.get("probability_policy_interface") != (
            "unchanged: declared_fail_closed_inactive; no probability-driven cell may run"):
        raise ValueError("probability-policy interface declaration drifted")

    return {
        "schema_version": "nanojev-financial-risk-gate-preflight-v1",
        "status": "protocol_valid_not_replay_authorized",
        "protocol": file_identity(protocol_path),
        "binds_t12": {"protocol": file_identity(t12_protocol_path),
                      "run_receipt_sha256": amendment["t12_run_receipt_sha256"]},
        "t12_preflight_status": t12_preflight["status"],
        "t12_upstream_artifacts": t12_preflight["upstream_artifacts"],
        "gate_modules": gate_identities,
        "gate": {"engine": protocol["risk_gate"]["engine"],
                 "policy_version": protocol["risk_gate"]["policy_version"],
                 "limits": protocol["risk_gate"]["limits"],
                 "lifecycle": "fresh_risk_engine_per_replay"},
        "t11_context": t12_preflight["t11_context"],
        "dataset": t12_preflight["dataset"],
        "event_definition_sha256": t12_preflight["event_definition_sha256"],
        "instrument_groups": t12_preflight["instrument_groups"],
        "mask_stats": t12_preflight["mask_stats"],
        "masks_sha256": t12_preflight["masks_sha256"],
        "cell_plan": t12_preflight["cell_plan"],
        "cell_count": t12_preflight["cell_count"],
        "authorization": {"real_replay_performed": False, "replay_authorized": False,
                          "measurement_authorized": False, "training_authorized": False,
                          "network_model_calls": 0},
        "blockers": ["separate nanojev-financial-validation-authorization-v1 receipt bound to "
                     "the T13 protocol hash required before any replay",
                     "venue holdout unsatisfiable in single-venue V1 cohort"],
        "scope": ("Protocol and data-bound preflight only; no replay, no fit, no model quality, "
                  "profitability, or deployment evidence. The gate adds deterministic guard "
                  "evaluation to the same unmodified run_backtest machinery."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "research/financial_risk_gate_protocol_v1.json")
    parser.add_argument("--receipt", type=Path, default=BOUND_RECEIPT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.protocol, args.receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "cell_count": result["cell_count"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
