#!/usr/bin/env python3
"""Deterministic Track A evidence replay.

This script packages the local checks a reviewer should rerun before any
active-mode discussion. It does not start scorer servers, call a provider, or
apply a reduction. It validates manifests/policies, reruns the deterministic
fixture/evaluator tests, checks result invariants, and writes a hash receipt.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

JSON_FILES = [
    "research/context_filter_value_fixture_manifest_v2.json",
    "research/context_filter_value_fixture_manifest_v3.json",
    "research/context_filter_threshold_policy_v1.json",
    "research/nanojev_vnext_architecture_decision_v1.json",
    "research/oss_jev_reproduction_matrix_v1.json",
    "results/scorer_stress_receipt_v1.json",
]

SELFTEST_FILE = "results/filter_value_v2_selftest_v1.json"
V3_SELFTEST_FILE = "results/filter_value_v3_selftest_v1.json"

RESULT_FILES = [
    "results/filter_value_v2_control_dedup_localgen_qwen3_0p6b_v1.json",
    "results/filter_value_v2_control_dedup_localgen_qwen25_3b_v1.json",
    "results/filter_value_v2_control_dedup_localgen_qwen35_4b_v1.json",
    "results/filter_value_v2_winnow_t090_localgen_qwen3_0p6b_v1.json",
    "results/filter_value_v2_winnow_t090_localgen_qwen25_3b_v1.json",
    "results/filter_value_v2_winnow_t090_localgen_qwen35_4b_v1.json",
    "results/filter_value_v2_winnow_t090_v1.json",
    "results/filter_value_v3_control_dedup_localgen_qwen35_4b_v1.json",
    "results/filter_value_v3_winnow_t090_v1.json",
    "results/filter_value_v3_winnow_t090_localgen_qwen35_4b_v1.json",
    "results/filter_value_v2_kev4b_t090_v1.json",
    "results/filter_value_v2_kev9b_t090_v1.json",
    "results/filter_value_v2_cascade_reflex_t090_f095_v1.json",
    "results/filter_value_v2_cascade_semif_t090_f080_v1.json",
    "results/filter_value_v2_cascade_decider2b_t090_f090_v1.json",
    "results/filter_value_v2_cascade_kev4b_t090_f095_v1.json",
]

REQUIRED_DOCS = [
    "docs/CONTEXT_FILTER_VALUE_V2.md",
    "docs/CONTEXT_FILTER_VALUE_V3.md",
    "docs/CONTEXT_FILTER_THRESHOLD_POLICY_V1.md",
    "docs/FAST_PATH_CASCADE_COMPARISON_V1.md",
    "docs/NANOJEV_VNEXT_ARCHITECTURE_V1.md",
    "docs/TRACK_A_REVIEW_CHECKLIST_V1.md",
    "docs/TRACK_A_REVIEW_DECISION_V1.md",
]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_receipt(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def run_command(command, env=None, expect_fail=False):
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    if expect_fail:
        ok = proc.returncode != 0
    else:
        ok = proc.returncode == 0
    return {"command": command, "ok": ok, "returncode": proc.returncode,
            "expected_fail": expect_fail, "elapsed_ms": elapsed_ms,
            "output_tail": proc.stdout[-4000:]}


def check_result_invariants(path, expected_unsafe=None):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_unsafe = expected_unsafe or {}
    failures = []
    for arm in result.get("results", []):
        name = arm.get("arm")
        totals = arm.get("totals", {})
        expected = expected_unsafe.get(name, 0)
        if totals.get("unsafe_removals") != expected:
            failures.append(f"{name}: unsafe_removals={totals.get('unsafe_removals')}, expected={expected}")
        if totals.get("paired_answer_regressions") != expected:
            failures.append(f"{name}: paired_answer_regressions={totals.get('paired_answer_regressions')}, expected={expected}")
        if totals.get("round_trip_ok") != totals.get("round_trip_attempted"):
            failures.append(f"{name}: incomplete round trips")
    return {"path": str(path), "sha256": sha256_file(path), "failures": failures}


def review(root=ROOT, run_tests=True):
    receipt = {"schema_version": "nanojev-track-a-review-replay-v1",
               "scope": "local deterministic evidence replay; no active filtering; no provider call",
               "commands": [], "files": [], "result_invariants": [], "failures": []}

    for path in JSON_FILES + REQUIRED_DOCS:
        try:
            receipt["files"].append(file_receipt(root / path))
        except FileNotFoundError as error:
            receipt["failures"].append(str(error))

    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "scripts") + os.pathsep + env.get("PYTHONPATH", "")
    commands = [
        [sys.executable, "scripts/build_filter_value_fixtures_v2.py", "--check"],
        [sys.executable, "scripts/build_filter_value_fixtures_v3.py", "--check"],
        [sys.executable, "-m", "json.tool", "research/context_filter_threshold_policy_v1.json"],
        [sys.executable, "-m", "json.tool", "research/context_filter_value_fixture_manifest_v2.json"],
        [sys.executable, "-m", "json.tool", "research/context_filter_value_fixture_manifest_v3.json"],
        [sys.executable, "-m", "json.tool", "research/nanojev_vnext_architecture_decision_v1.json"],
        [sys.executable, "-m", "json.tool", "research/oss_jev_reproduction_matrix_v1.json"],
        [sys.executable, "-m", "json.tool", "results/scorer_stress_receipt_v1.json"],
    ]
    if run_tests:
        commands.append([sys.executable, "-m", "unittest",
                         "scripts.test_filter_value_v1",
                         "scripts.test_local_main_model_evaluator_v1",
                         "scripts.test_scorer_stress_v1"])
    for command in commands:
        check = run_command(command, env=env)
        receipt["commands"].append(check)
        if not check["ok"]:
            receipt["failures"].append("command failed: " + " ".join(command))

    negative = [sys.executable, "scripts/benchmark_filter_value_v1.py",
                "--manifest", "research/context_filter_value_fixture_manifest_v2.json",
                "--threshold-policy", "research/context_filter_threshold_policy_v1.json",
                "--review-mode", "--threshold", "0.80", "--arms", "control",
                "--output", "/tmp/nanojev-undeclared-threshold-review.json"]
    check = run_command(negative, env=env, expect_fail=True)
    receipt["commands"].append(check)
    if not check["ok"] or "undeclared diagnostic threshold" not in check["output_tail"]:
        receipt["failures"].append("review-mode undeclared-threshold negative control failed")

    try:
        selftest = check_result_invariants(root / SELFTEST_FILE,
                                           expected_unsafe={"stub": 1})
        receipt["result_invariants"].append(selftest)
        receipt["failures"].extend(selftest["failures"])
    except FileNotFoundError as error:
        receipt["failures"].append(str(error))

    try:
        v3_selftest = check_result_invariants(root / V3_SELFTEST_FILE,
                                              expected_unsafe={"stub": 1})
        receipt["result_invariants"].append(v3_selftest)
        receipt["failures"].extend(v3_selftest["failures"])
    except FileNotFoundError as error:
        receipt["failures"].append(str(error))

    for path in RESULT_FILES:
        try:
            check = check_result_invariants(root / path)
            receipt["result_invariants"].append(check)
            receipt["failures"].extend(check["failures"])
        except FileNotFoundError as error:
            receipt["failures"].append(str(error))
    receipt["ok"] = not receipt["failures"]
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "track_a_review_replay_v1.json")
    parser.add_argument("--skip-tests", action="store_true",
                        help="hash/check manifests and result invariants without rerunning unit tests")
    args = parser.parse_args()
    receipt = review(run_tests=not args.skip_tests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()),
                      "sha256": sha256_file(args.output),
                      "ok": receipt["ok"],
                      "failure_count": len(receipt["failures"]),
                      "failures": receipt["failures"]}, indent=2))
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
