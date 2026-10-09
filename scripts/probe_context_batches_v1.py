#!/usr/bin/env python3
"""T8 real-loopback parity probe; synthetic input only, no active filtering or training.

Compare the production 32-state partition with a 16-state reference partition. Both
respect the existing HTTP service's 32-state cap and keep the entire original context.
The patch affects only this probe process, never the persistent server/configuration.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import context_gate_v1 as gate
from context_gate_local import LoggedScorer, LoopbackPredictor


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints/local_atomic_seed17/variants/local_atomic_seed17"
PROTOCOL = {
    "schema_version": "nanojev-context-batch-probe-v1",
    "cases": [["openai_chat", 32], ["openai_chat", 33], ["openai_chat", 65],
              ["openai_responses", 33], ["anthropic_messages", 33]],
    "batch_sizes": [32, 16], "threshold": 0.99, "absolute_probability_tolerance": 1e-6,
    "temperature": 1.0, "inputs": "synthetic indexed notes and a protected final user request",
    "production_activation_authorized": False, "no_model_quality_or_speedup_claim": True,
}


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fixture(wire, count):
    key = "input" if wire == "openai_responses" else "messages"
    body = {"model": "synthetic-only", key: [
        {"role": "assistant", "content": f"Note {i}."} for i in range(count)] + [
        {"role": "user", "content": "What is the final note? Retain evidence needed to answer."}]}
    if wire == "anthropic_messages":
        body["max_tokens"] = 64
    pointers = [f"/{key}/{i}/content" for i in range(count)]
    raw = gate.serialized(body).encode()
    return raw, {"segments": {p: {"eligible": True} for p in pointers}}, pointers


def compare(left, right, pointers):
    a = {s["pointer"]: s for s in left["segments"]}
    b = {s["pointer"]: s for s in right["segments"]}
    missing = [p for p in pointers if a[p]["p_irrelevant"] is None or b[p]["p_irrelevant"] is None]
    delta = None if missing else max(abs(a[p]["p_irrelevant"] - b[p]["p_irrelevant"]) for p in pointers)
    plan = lambda receipt: [s["pointer"] for s in receipt["segments"] if s["suggestion"] == "drop"]
    return {"all_candidates_scored": not missing, "missing_candidate_count": len(missing),
            "max_abs_probability_delta": delta, "plans_identical": plan(left) == plan(right),
            "plan_sha256_32": gate.fingerprint(plan(left)), "plan_sha256_16": gate.fingerprint(plan(right)),
            "passed": not missing and delta <= PROTOCOL["absolute_probability_tolerance"]
                      and plan(left) == plan(right)}


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def run(output, url, checkpoint):
    predictor = LoopbackPredictor(url, timeout=30)  # rejects non-loopback URLs before any output
    paths = [Path(__file__), ROOT / "scripts/context_gate_v1.py", ROOT / "scripts/context_gate_local.py",
             ROOT / "scripts/main_model_gateway_v1.py", ROOT / "scripts/scorer_adapters_v1.py",
             ROOT / "integrations/codex-skill/nanojev-local-decider/scripts/nanojev_skill.py",
             checkpoint / "config.json", checkpoint / "best.safetensors"]
    identities = {str(p): sha256_file(p) for p in paths}
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "protocol.json", PROTOCOL)
    usage = output / "usage.jsonl"
    logged = LoggedScorer(predictor, checkpoint, usage)
    rows, executions = [], []

    def scorer(payload):
        result = logged(payload)
        execution = result.get("execution", {})
        if execution.get("network_model_calls") != 0 or execution.get("device") != "mps" or execution.get("precision") != "fp32":
            raise ValueError("unexpected local runtime; do not treat this as a verified arm")
        if result.get("temperature", {}).get("value") != PROTOCOL["temperature"]:
            raise ValueError("unexpected temperature")
        executions.append({"event_id": result["context_gate_usage_event_id"],
                           "checkpoint": result["checkpoint"], "execution": execution,
                           "request_sha256": gate.fingerprint(payload)})
        return result

    for wire, count in PROTOCOL["cases"]:
        raw, sidecar, pointers = fixture(wire, count)
        arms = {}
        for size in PROTOCOL["batch_sizes"]:
            with patch.object(gate, "MAX_SCORED", size):
                forwarded, receipt = gate.shadow_request(raw, wire, sidecar, scorer,
                                                        threshold=PROTOCOL["threshold"])
            if forwarded is not raw or receipt["actual_removed_segments"] != 0 or receipt["actual_removed_tokens"] != 0:
                raise AssertionError("shadow invariant violated")
            arms[size] = receipt
            write_new(output / f"{wire}_{count}_batch{size}.json", receipt)
        parity = compare(arms[32], arms[16], pointers)
        rows.append({"wire_format": wire, "candidate_count": count,
                     "fixture_sha256": gate.fingerprint({"raw_sha256": gate.fingerprint(raw), "sidecar": sidecar}),
                     "batch_counts": {str(k): len(v.get("batches", [])) for k, v in arms.items()},
                     "reasons": {str(k): v["reason"] for k, v in arms.items()},
                     "proposed_drops": sum(s["suggestion"] == "drop" for s in arms[32]["segments"]), **parity})
        print(json.dumps(rows[-1]), flush=True)
    unchanged = identities == {str(p): sha256_file(p) for p in paths}
    # Cross-arm identity comparison is independent of the core's within-request check.
    fingerprints = {gate.fingerprint(e["checkpoint"]) for e in executions}
    passed = all(row["passed"] for row in rows) and unchanged and len(fingerprints) == 1
    report = {"schema_version": PROTOCOL["schema_version"], "status": "passed" if passed else "failed",
              "protocol_sha256": sha256_file(output / "protocol.json"), "source_hashes": identities,
              "source_files_unchanged": unchanged, "model_fingerprint_count": len(fingerprints),
              "rows": rows, "executions": executions, "actual_removed_tokens": 0,
              "usage_sha256": sha256_file(usage) if usage.exists() else None,
              "limitations": ["Constructed inputs, not a relevance-quality or downstream safety evaluation",
                  "Times in raw receipts are single observations, not speedup or tail-latency evidence",
                  "Checkpoint metadata is a server claim; local file hashes are not remote weight attestation",
                  "32/16 diagnostic partition only; service limits, threshold and checkpoint unchanged"]}
    write_new(output / "report.json", report)
    return 0 if passed else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8876")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    return run(args.output_dir, args.url, args.checkpoint.resolve(strict=True))


if __name__ == "__main__":
    raise SystemExit(main())
