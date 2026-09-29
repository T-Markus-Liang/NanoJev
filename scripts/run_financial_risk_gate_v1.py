#!/usr/bin/env python3
"""Fail-closed T13 runner: real cohort through the gated backtest harness.

Without a ``nanojev-financial-validation-authorization-v1`` receipt bound to
the pinned T13 protocol hash (decision ``approved_for_real_replay``) the runner
emits a blocked, content-free receipt and exits 2. With it, the frozen R1
cohort is replayed per (fold, instrument, reference policy) cell through
``financial_backtest_v1.run_backtest`` with a fresh ``financial_risk_v1.
RiskEngine`` (frozen R1 ``limits.*``) per replay, and gate verdicts are
reported per fold, per instrument, and per regime stratum — descriptively,
with no headline and no profitability claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_nanojev_v2 import file_identity  # noqa: E402
import financial_baselines_v1 as baselines  # noqa: E402
import financial_real_data_risk_v1 as gated  # noqa: E402
import validate_financial_risk_gate_protocol_v1 as protocol_validator  # noqa: E402

PROTOCOL_PATH = ROOT / "research/financial_risk_gate_protocol_v1.json"
BOUND_RECEIPT = ROOT / "results/financial_pit_r1_bound_receipt_20260920_v5.json"
AUTH_SCHEMA = "nanojev-financial-validation-authorization-v1"
RUN_SCHEMA = "nanojev-financial-realdata-risk-gate-v1"

CODE_FILES = (
    "scripts/financial_real_data_risk_v1.py",
    "scripts/run_financial_risk_gate_v1.py",
    "scripts/validate_financial_risk_gate_protocol_v1.py",
    "scripts/financial_real_data_path_v1.py",
    "scripts/run_financial_validation_v1.py",
    "scripts/validate_financial_validation_protocol_v1.py",
    "scripts/financial_backtest_v1.py",
    "scripts/financial_simulator_v1.py",
    "scripts/financial_risk_v1.py",
    "scripts/financial_baselines_v1.py",
    "scripts/financial_pit_v1.py",
    "scripts/benchmark_nanojev_v2.py",
)

LIMITATIONS = [
    "Bid and ask are collapsed to the record mark price; the cohort carries no order book. "
    "All execution costs are frozen assumed parameters, not venue measurements.",
    "Funding applies the decision-time last-settled rate at each in-window 8h boundary; the "
    "rate that actually settled at each boundary is not observable at record granularity.",
    "The equity curve is sampled at retained decision bars and funding boundaries only; "
    "intra-day excursions and gap-day marks are not represented.",
    "Venue holdout is unsatisfiable in the Binance-only V1 cohort; no venue generalization "
    "is claimed or implied.",
    "Gate limits are the frozen R1 provisional values pending review gate R1, not measured "
    "venue or broker constraints. min_position_quantity=0 makes every net-short intent out "
    "of bounds, so unit_short cells report a ~100% risk-block rate: a mechanical property "
    "of the frozen spot-style parameter set, not a market signal.",
    "A reduce verdict still executes at the allowed quantity; reduced exposure is reported "
    "as reduced_quantity_total in base units, never as avoided loss.",
    "Gate-blocked or gate-reduced cells have truncated exposure, so their equity curves "
    "diverge from the ungated T12 curves by construction; both remain descriptive-only.",
    "Mechanical reference policies validate the harness and the gate plumbing, not a "
    "strategy. T11 found no candidate beating the base rate on test; nothing here may be "
    "described as edge, profitability, or a tradable result.",
    "Gross-event metrics, gate verdict counts, and simulated net PnL are different objects "
    "and never share a column.",
]


def _under_root(path):
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        raise ValueError("path escapes project root") from None
    return resolved


def _check_authorization(path, protocol_sha256):
    if path is None:
        return False, {"status": "missing", "reason": "validation_authorization_receipt_required"}
    path = _under_root(path)
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid validation authorization receipt: {exc}") from None
    if receipt.get("schema_version") != AUTH_SCHEMA:
        raise ValueError("wrong validation authorization receipt schema")
    if receipt.get("protocol_sha256") != protocol_sha256:
        raise ValueError("validation authorization is bound to a different T13 protocol")
    if receipt.get("decision") != "approved_for_real_replay":
        raise ValueError("authorization decision is not approved_for_real_replay")
    if receipt.get("replay_authorized") is not True:
        raise ValueError("replay_authorized flag is not true")
    if receipt.get("measurement_authorized") is not True:
        raise ValueError("measurement authorization flag is not true")
    if receipt.get("training_authorized") is True:
        raise ValueError("a replay authorization must not grant training")
    if receipt.get("order_submission_authorized") is not False:
        raise ValueError("authorization may not permit order submission")
    if receipt.get("live_trading_authorized") is not False:
        raise ValueError("authorization may not permit live trading")
    if receipt.get("network_model_calls") != 0:
        raise ValueError("authorization cannot permit network model calls")
    reviewer = receipt.get("independent_reviewer")
    if not isinstance(reviewer, dict) or not reviewer.get("id") \
            or not reviewer.get("reviewed_utc"):
        raise ValueError("reviewer identity and timestamp are required")
    return True, {
        "status": "approved_for_real_replay",
        "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reviewer_id": reviewer["id"],
        "reviewed_utc": reviewer["reviewed_utc"],
        "independence": reviewer.get("independence"),
    }


def _code_hashes():
    return {name: file_identity(ROOT / name) for name in CODE_FILES}


def run(protocol_path=PROTOCOL_PATH, receipt_path=BOUND_RECEIPT, authorization_path=None):
    started = time.time()
    preflight = protocol_validator.validate(protocol_path, receipt_path)
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    protocol_sha256 = hashlib.sha256(Path(protocol_path).read_bytes()).hexdigest()
    authorized, auth = _check_authorization(authorization_path, protocol_sha256)
    common = {
        "schema_version": RUN_SCHEMA,
        "protocol_sha256": protocol_sha256,
        "binds_t12": preflight["binds_t12"],
        "upstream_preflight_status": preflight["status"],
        "t12_preflight_status": preflight["t12_preflight_status"],
        "authorization": auth,
        "network_model_calls": 0,
        "replay_authorized": bool(authorized),
        "measurement_authorized": bool(authorized),
        "training_authorized": False,
        "order_submission_authorized": False,
        "live_trading_authorized": False,
        "real_replay_performed": False,
        "headline_allowed": False,
    }
    if not authorized:
        common.update({
            "status": "blocked_replay_authorization_missing",
            "replay_performed": False,
            "scope": "No cohort row reached the gated harness; protocol/preflight only.",
            "cell_plan": preflight["cell_plan"],
            "folds": [],
        })
        return common

    t12_protocol = json.loads(
        (ROOT / protocol["amendment_of"]["protocol_path"]).read_text(encoding="utf-8"))
    _, rows, folds = baselines.load_bound_cohort(_under_root(receipt_path))
    groups = baselines.instrument_groups(rows)
    upstream = t12_protocol["upstream_r1"]
    provenance = {"dataset_sha256": upstream["dataset_sha256"],
                  "bound_receipt_sha256": upstream["bound_receipt_sha256"],
                  "event_definition_sha256": upstream["event_definition_sha256"]}
    gate_factory = gated.frozen_gate_factory()
    result = gated.run_gated_validation(
        rows, folds, json.loads((ROOT / upstream["protocol_path"]).read_text(encoding="utf-8")),
        groups, gate_factory=gate_factory, provenance=provenance)
    gate_limits = gate_factory().limits.to_dict()
    common.update({
        "status": "gated_replay_completed_not_promoted",
        "replay_performed": True,
        "real_replay_performed": True,
        "scope": ("T13 real-data risk-gate replay only: the frozen R1 RiskEngine guard set on "
                  "the same 45 mechanical-policy cells as T12, through the unmodified "
                  "financial_backtest_v1 machinery. Descriptive per-cell gate analytics; no "
                  "fitting, selection, promotion, edge, or profitability claim."),
        "risk_gate": {"engine": gated.GATE_NAME,
                      "policy_version": gated.GATE_POLICY_VERSION,
                      "limits_source": gated.FROZEN_PARAMETERS_PATH,
                      "limits": gate_limits,
                      "lifecycle": "fresh_risk_engine_per_replay",
                      "determinism_rule": ("two independent run_backtest calls per cell, fresh "
                                           "gate each; receipts must be byte-identical")},
        "input_hashes": {
            "t13_protocol_sha256": protocol_sha256,
            "t12_protocol_sha256": protocol["amendment_of"]["protocol_sha256"],
            "t12_run_receipt_sha256": protocol["amendment_of"]["t12_run_receipt_sha256"],
            "r1_protocol_sha256": upstream["protocol_sha256"],
            "r1_bound_receipt_sha256": upstream["bound_receipt_sha256"],
            "dataset_sha256": upstream["dataset_sha256"],
            "pit_validator_sha256": upstream["validator_sha256"],
            "pit_core_sha256": upstream["core_sha256"],
            "frozen_parameters_sha256": upstream["frozen_parameters_sha256"],
            "t11_fit_receipt_sha256": t12_protocol["upstream_t11"]["fit_receipt_sha256"],
            "event_definition_sha256": upstream["event_definition_sha256"],
        },
        "code_hashes": _code_hashes(),
        "cohort": {"records": len(rows), "instruments": t12_protocol["cohort"]["instruments"],
                   "venue": t12_protocol["cohort"]["venue"],
                   "instrument_groups": groups},
        "masks_sha256": result["masks_sha256"],
        "mask_stats": result["mask_stats"],
        "gate_summary": result["gate_summary"],
        "folds": result["folds"],
        "t11_context": preflight["t11_context"],
        "honesty_gates": dict(protocol["honesty_gates"], headline_allowed=False),
        "limitations": LIMITATIONS,
        "runtime": {"wall_seconds": round(time.time() - started, 6),
                    "python": sys.version.split()[0]},
    })
    return common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--receipt", type=Path, default=BOUND_RECEIPT)
    parser.add_argument("--authorization-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.protocol, args.receipt, args.authorization_receipt)
    args.output = _under_root(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"],
                      "replay_performed": result["replay_performed"],
                      "folds": len(result.get("folds", []))}, ensure_ascii=False))
    return 0 if result["replay_performed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
