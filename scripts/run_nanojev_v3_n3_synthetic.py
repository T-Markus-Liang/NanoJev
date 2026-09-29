#!/usr/bin/env python3
"""Run the N3 readout controls against a deterministic synthetic fixture only.

This runner is deliberately not a model benchmark.  It exercises the protocol's
pairing and fail-closed controls without loading a checkpoint, reading a dataset,
or making a network call.  A passing report is a harness smoke receipt, never a
quality, training, deployment, or promotion result.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path
import sys

import validate_nanojev_v3_n3_readout_protocol_v1 as protocol_validator


HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = HERE.parent / "research" / "nanojev_v3_n3_readout_protocol_v1.json"
REPORT_STATUS = "synthetic_controls_passed_not_model_evidence"


class SyntheticRunError(RuntimeError):
    """A protocol or synthetic control failed closed."""


@dataclass(frozen=True)
class SyntheticCase:
    case_id: str
    split: str
    question_type: str
    options: tuple[str, ...]
    label: str | None
    protected: bool = False
    ood: bool = False


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_value(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _arm_info(protocol):
    return {arm["arm_id"]: arm for arm in protocol["paired_arms"]}


def synthetic_cases(protocol):
    """Return a small fixture covering every split and required question family."""
    cases = [
        SyntheticCase("synthetic-train-choice", "train", "choice", ("a", "b", "c"), "a"),
        SyntheticCase("synthetic-dev-boolean", "dev", "boolean", ("true", "false"), "true"),
        SyntheticCase("synthetic-calibration-score", "calibration", "score", ("0", "1", "2"), "0"),
        SyntheticCase("synthetic-test-choice", "test", "choice", ("a", "b", "c"), "b"),
        SyntheticCase("synthetic-ood-boolean", "ood", "boolean", ("true", "false"), None, ood=True),
    ]
    existing = {case.case_id for case in cases}
    for declaration in protocol["protected_cases"]:
        case_id = declaration["case_id"]
        if case_id in existing:
            continue
        split = declaration["split"]
        if split == "ood":
            case = SyntheticCase(case_id, split, "boolean", ("true", "false"), None,
                                 protected=True, ood=True)
        else:
            case = SyntheticCase(case_id, split, "choice", ("a", "b", "c"), "b", protected=True)
        cases.append(case)
        existing.add(case_id)
    return tuple(cases)


def fixture_digest(cases):
    return sha256_value([asdict(case) for case in cases])


def _probabilities(case):
    if case.ood:
        share = 1.0 / len(case.options)
        return {option: share for option in case.options}
    remainder = 0.08 / (len(case.options) - 1)
    return {
        option: 0.92 if option == case.label else remainder
        for option in case.options
    }


def _confidence(probabilities):
    return max(probabilities.values())


def _receipt_without_hash(*, arm_id, arm_role, case, control_id, option_order,
                          option_permutation_id, label_assignment, label_permutation_id,
                          predicted_label, probabilities, coverage, correct, confidence):
    constant_true_correct = None
    constant_false_correct = None
    if case.label is not None:
        constant_true_correct = case.label == "true"
        constant_false_correct = case.label == "false"
    return {
        "arm_id": arm_id,
        "arm_role": arm_role,
        "split": case.split,
        "case_id": case.case_id,
        "question_type": case.question_type,
        "option_order": list(option_order),
        "option_permutation_id": option_permutation_id,
        "label_permutation_id": label_permutation_id,
        "label_assignment": label_assignment,
        "predicted_label": predicted_label,
        "probabilities": probabilities,
        "confidence": confidence,
        "coverage": coverage,
        "correct": correct,
        "constant_true_correct": constant_true_correct,
        "constant_false_correct": constant_false_correct,
        "control_id": control_id,
        "protected_case": case.protected,
        "latency_ms": None,
        "memory_bytes": None,
        "network_model_calls": 0,
        "receipt_sha256": None,
        "training_authorized": False,
        "deployment_authorized": False,
    }


def _with_receipt_hash(row):
    payload = dict(row)
    payload["receipt_sha256"] = None
    row["receipt_sha256"] = sha256_value(payload)
    return row


def _option_orders(options):
    # itertools.permutations preserves the input identity order first.
    return tuple(itertools.permutations(options))


def _rotated_label(case):
    index = case.options.index(case.label)
    return case.options[(index + 1) % len(case.options)]


def _make_option_rows(arm_id, arm_role, case):
    rows = []
    for permutation_index, order in enumerate(_option_orders(case.options)):
        probabilities = _probabilities(case)
        predicted = None if case.ood else case.label
        coverage = not case.ood
        confidence = 0.0 if case.ood else _confidence(probabilities)
        correct = None if case.ood else predicted == case.label
        probabilities = ({option: (1.0 / len(case.options)) for option in case.options}
                         if predicted is None else
                         {option: (1.0 if option == predicted else 0.0)
                          for option in case.options})
        rows.append(_with_receipt_hash(_receipt_without_hash(
            arm_id=arm_id,
            arm_role=arm_role,
            case=case,
            control_id="option_permutation",
            option_order=order,
            option_permutation_id=f"option-{permutation_index:03d}",
            label_assignment=case.label,
            label_permutation_id="label-identity",
            predicted_label=predicted,
            probabilities=probabilities,
            coverage=coverage,
            correct=correct,
            confidence=confidence,
        )))
    return rows


def _make_label_row(arm_id, arm_role, case):
    if case.ood:
        return []
    probabilities = _probabilities(case)
    return [_with_receipt_hash(_receipt_without_hash(
        arm_id=arm_id,
        arm_role=arm_role,
        case=case,
        control_id="label_permutation",
        option_order=case.options,
        option_permutation_id="option-000",
        label_assignment=_rotated_label(case),
        label_permutation_id="label-rotated-001",
        predicted_label=case.label,
        probabilities=probabilities,
        coverage=True,
        correct=False,
        confidence=_confidence(probabilities),
    ))]


def _make_baseline_rows(arm_id, arm_role, case):
    # Constant true/false are meaningful for Boolean cases.  The other required
    # baselines are represented on the same Boolean cases, preserving paired fields.
    if case.ood or case.question_type != "boolean":
        return []
    majority = "true"
    rows = []
    for baseline in ("constant_true", "constant_false", "constant_majority", "abstain_all"):
        predicted = None if baseline == "abstain_all" else (
            majority if baseline == "constant_majority" else baseline.removeprefix("constant_"))
        coverage = predicted is not None
        confidence = 0.0 if predicted is None else 1.0
        if predicted is None:
            probabilities = {option: 1.0 / len(case.options) for option in case.options}
        else:
            probabilities = {
                option: (1.0 if option == predicted else 0.0)
                for option in case.options
            }
        rows.append(_with_receipt_hash(_receipt_without_hash(
            arm_id=arm_id,
            arm_role=arm_role,
            case=case,
            control_id="constant_baselines",
            option_order=case.options,
            option_permutation_id="option-000",
            label_assignment=case.label,
            label_permutation_id=f"baseline-{baseline}",
            predicted_label=predicted,
            probabilities=probabilities,
            coverage=coverage,
            correct=None if not coverage else predicted == case.label,
            confidence=confidence,
        )))
    return rows


def build_receipts(protocol, cases=None):
    cases = synthetic_cases(protocol) if cases is None else tuple(cases)
    rows = []
    for arm in protocol["paired_arms"]:
        for case in cases:
            rows.extend(_make_option_rows(arm["arm_id"], arm["role"], case))
            rows.extend(_make_label_row(arm["arm_id"], arm["role"], case))
            rows.extend(_make_baseline_rows(arm["arm_id"], arm["role"], case))
    return rows


def _group(rows, *keys):
    result = {}
    for row in rows:
        key = tuple(row[key_name] for key_name in keys)
        result.setdefault(key, []).append(row)
    return result


def validate_receipts(protocol, cases, receipts):
    required = set(protocol["paired_receipt_fields"])
    if not receipts:
        raise SyntheticRunError("no synthetic receipts were produced")
    for row in receipts:
        if set(row) != required:
            raise SyntheticRunError("receipt field set does not exactly match the protocol")
        expected_hash = row["receipt_sha256"]
        candidate = dict(row)
        candidate["receipt_sha256"] = None
        if expected_hash != sha256_value(candidate):
            raise SyntheticRunError(f"receipt hash mismatch for {row['case_id']}")
        if row["network_model_calls"] != 0:
            raise SyntheticRunError("synthetic receipt reports a network model call")
        if row["training_authorized"] or row["deployment_authorized"]:
            raise SyntheticRunError("synthetic receipt contains authorization")
        if any(key in row for key in ("state", "instructions", "criteria", "raw_text")):
            raise SyntheticRunError("synthetic receipt contains raw input content")
        options = set(row["option_order"])
        probabilities = row["probabilities"]
        if set(probabilities) != options or any(
                not isinstance(value, (int, float)) or not 0 <= value <= 1
                for value in probabilities.values()):
            raise SyntheticRunError(f"invalid probability vector for {row['case_id']}")
        if abs(sum(probabilities.values()) - 1.0) > 1e-9:
            raise SyntheticRunError(f"probabilities are not normalized for {row['case_id']}")

    arm_ids = tuple(arm["arm_id"] for arm in protocol["paired_arms"])
    case_by_id = {case.case_id: case for case in cases}
    option_rows = [row for row in receipts if row["control_id"] == "option_permutation"]
    option_groups = _group(option_rows, "arm_id", "case_id")
    for arm_id in arm_ids:
        for case in cases:
            rows = option_groups.get((arm_id, case.case_id), [])
            expected = len(_option_orders(case.options))
            if len(rows) != expected:
                raise SyntheticRunError(f"option permutation count mismatch for {arm_id}/{case.case_id}")
            if rows[0]["option_order"] != list(case.options):
                raise SyntheticRunError(f"identity order is not first for {arm_id}/{case.case_id}")
            selected = {row["predicted_label"] for row in rows}
            if len(selected) != 1:
                raise SyntheticRunError(f"semantic option selection is unstable for {arm_id}/{case.case_id}")
            if not case.ood and next(iter(selected)) != case.label:
                raise SyntheticRunError(f"synthetic arm missed its target for {arm_id}/{case.case_id}")

    label_rows = [row for row in receipts if row["control_id"] == "label_permutation"]
    label_groups = _group(label_rows, "arm_id", "case_id")
    for key, rows in label_groups.items():
        if len(rows) != 1:
            raise SyntheticRunError(f"label permutation is not singular for {key}")
        original = option_groups[key][0]
        if rows[0]["correct"] and not original["correct"]:
            raise SyntheticRunError(f"label permutation improved accuracy for {key}")

    type_counts = {name: sum(case.question_type == name for case in cases)
                   for name in ("boolean", "score", "choice")}
    if type_counts["boolean"] == 0 or type_counts["score"] == 0:
        raise SyntheticRunError("Boolean and Score coverage is incomplete")

    ood_rows = [row for row in option_rows if case_by_id[row["case_id"]].ood]
    if not ood_rows or any(row["coverage"] or row["predicted_label"] is not None for row in ood_rows):
        raise SyntheticRunError("OOD cases did not abstain")

    protected_rows = [row for row in option_rows if row["protected_case"]]
    confident_protected_errors = sum(
        bool(row["coverage"] and row["confidence"] >= 0.9 and row["correct"] is False)
        for row in protected_rows
    )
    if confident_protected_errors:
        raise SyntheticRunError("protected-case confident error detected")

    baseline_names = set(protocol["constant_baselines"])
    baseline_rows = [row for row in receipts if row["control_id"] == "constant_baselines"]
    observed_baselines = {
        row["label_permutation_id"].removeprefix("baseline-") for row in baseline_rows
    }
    if not baseline_names.issubset(observed_baselines):
        raise SyntheticRunError("a required constant baseline has no receipt")

    return {
        "option_permutation": {"passed": True, "rows": len(option_rows)},
        "label_permutation": {"passed": True, "rows": len(label_rows)},
        "boolean_coverage": {"passed": type_counts["boolean"] > 0, "cases": type_counts["boolean"]},
        "score_coverage": {"passed": type_counts["score"] > 0, "cases": type_counts["score"]},
        "ood_cases": {"passed": True, "rows": len(ood_rows), "abstained": len(ood_rows)},
        "protected_cases": {
            "passed": True,
            "rows": len(protected_rows),
            "confident_errors": confident_protected_errors,
        },
        "constant_baselines": {
            "passed": True,
            "names": sorted(observed_baselines),
            "rows": len(baseline_rows),
        },
    }


def run_protocol(protocol, source="<memory>"):
    validation = protocol_validator.validate_protocol(protocol, source=source)
    if not validation["valid"]:
        raise SyntheticRunError(json.dumps({"protocol_blocked": validation["block_reasons"]}, sort_keys=True))
    cases = synthetic_cases(protocol)
    receipts = build_receipts(protocol, cases)
    controls = validate_receipts(protocol, cases, receipts)
    report = {
        "report_version": "nanojev-v3-n3-synthetic-report-v1",
        "status": REPORT_STATUS,
        "protocol_schema_version": protocol["schema_version"],
        "protocol_sha256": sha256_value(protocol),
        "fixture_sha256": fixture_digest(cases),
        "synthetic_only": True,
        "source": str(source),
        "case_count": len(cases),
        "arm_count": len(protocol["paired_arms"]),
        "receipt_count": len(receipts),
        "control_results": controls,
        "receipts": receipts,
        "network_model_calls": 0,
        "model_loaded": False,
        "training_authorized": False,
        "deployment_authorized": False,
        "production_pruning_authorized": False,
        "training_performed": False,
        "deployment_performed": False,
        "report_sha256": None,
    }
    report["report_sha256"] = sha256_value({key: value for key, value in report.items()
                                             if key != "report_sha256"})
    return report


def load_and_run(path):
    try:
        protocol = protocol_validator.load_protocol(path)
    except (OSError, ValueError) as error:
        raise SyntheticRunError(f"protocol unreadable: {error}") from error
    return run_protocol(protocol, source=path)


def _render(report):
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        report = load_and_run(args.protocol)
        rendered = _render(report)
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
        sys.stdout.write(rendered)
        return 0
    except (OSError, SyntheticRunError, TypeError, ValueError) as error:
        blocked = {
            "report_version": "nanojev-v3-n3-synthetic-report-v1",
            "status": "synthetic_controls_blocked",
            "error": str(error),
            "network_model_calls": 0,
            "model_loaded": False,
            "training_authorized": False,
            "deployment_authorized": False,
            "production_pruning_authorized": False,
            "training_performed": False,
            "deployment_performed": False,
        }
        sys.stdout.write(_render(blocked))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
