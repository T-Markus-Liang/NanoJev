#!/usr/bin/env python3
"""Fail-closed structural validator for the versioned T9d FP32/MPS protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "nanojev-v2-t9d-fp32-mps-protocol-v2"


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("protocol root must be an object")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(protocol: dict, root: Path) -> list[str]:
    errors: list[str] = []
    if protocol.get("schema_version") != SCHEMA:
        errors.append("wrong schema_version")
    if protocol.get("protocol_version") != "v2":
        errors.append("protocol_version must be v2")
    if protocol.get("status") != "versioned_pending_independent_review_and_clean_corpus":
        errors.append("protocol must remain pending review and clean corpus")
    runtime = protocol.get("runtime", {})
    for key, expected in (("device", "mps"), ("precision", "fp32"), ("parameter_storage", "float32")):
        if runtime.get(key) != expected:
            errors.append(f"runtime.{key} must be {expected}")
    recipe = protocol.get("recipe", {})
    expected_recipe = {
        "training_seeds": [17, 18, 19], "steps": 300, "eval_every": 50,
        "batch_questions": 16, "microbatch_questions": 4, "max_microbatch_tokens": 16384,
        "max_length": 2048, "backbone_lr": 0.00002, "head_lr": 0.0002,
        "objective": "gold_distribution", "loss": "ce",
    }
    for key, expected in expected_recipe.items():
        if recipe.get(key) != expected:
            errors.append(f"recipe.{key} mismatch")
    auth = protocol.get("authorization", {})
    for key in ("training_authorized", "measurement_authorized", "checkpoint_promotion_authorized",
                "deployment_authorized", "active_context_removal_authorized", "live_trading_authorized"):
        if auth.get(key) is not False:
            errors.append(f"authorization.{key} must be false")
    for rel, key in (("scripts/train_pipeline_decisions.py", "trainer_sha256"),
                     ("scripts/predict_toy_decisions.py", "predictor_sha256")):
        path = root / rel
        if not path.is_file():
            errors.append(f"missing bound file: {rel}")
        elif runtime.get(key) != sha256(path):
            errors.append(f"hash mismatch: {rel}")
    maze = protocol.get("data_and_split_contract", {}).get("maze_split_sha256", {})
    for name, expected in maze.items():
        path = root / "dataset/games_v4/data/local_maze_v1" / name
        if not path.is_file():
            errors.append(f"missing maze split: {name}")
        elif sha256(path) != expected:
            errors.append(f"hash mismatch: {name}")
    gates = protocol.get("paired_gates", {})
    if gates.get("g4_report") != ["accuracy", "nll", "brier", "ece"]:
        errors.append("G4 must report accuracy, nll, brier and ece")
    if gates.get("g4_pairing") != "same cohort and matched initialization":
        errors.append("G4 pairing rule missing")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("research/nanojev_v2_t9d_fp32_mps_protocol_v2.json"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    errors = validate(load(args.root / args.protocol if not args.protocol.is_absolute() else args.protocol), args.root.resolve())
    result = {"schema_version": SCHEMA, "status": "passed" if not errors else "failed", "errors": errors,
              "training_authorized": False, "measurement_authorized": False}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
