#!/usr/bin/env python3
"""Run a local NanoJev checkpoint on raw V2 decision records and summarize targets."""

import argparse
from collections import Counter, defaultdict
import json
import time
from pathlib import Path

from benchmark_nanojev_v2 import (
    benchmark_manifest, canonical_json, file_identity, hardware_descriptor, peak_rss_bytes, percentile,
    resolve_input_files, sha256_bytes,
)
from evaluate_pipeline_decisions import evaluate
from predict_toy_decisions import DecisionPredictor
from train_pipeline_decisions import candidate_ids, read_training_records, validate_training_row, SPLITS


def load_rows(path, splits):
    if splits - set(SPLITS):
        raise ValueError("unknown requested split")
    selected_files = resolve_input_files(path, splits)
    rows, _ = read_training_records(Path(path))
    if splits:
        rows = [row for row in rows if row["split"] in splits]
    if not rows:
        raise ValueError("input contains no rows for the requested splits")
    if splits and splits - {row["split"] for row in rows}:
        raise ValueError("input does not contain every requested split")
    return rows, selected_files


def target_mapping(row, qid, question):
    ids = candidate_ids(question)
    target = validate_training_row(row)[qid]
    if target["gold_probs"] is not None:
        return dict(zip(ids, target["gold_probs"])), target["gold_probs_kind"], target["gold_label_kind"]
    if target["gold_index"] is not None:
        if target["gold_label_kind"] != "deterministic_truth":
            raise ValueError("hard-only targets require explicit deterministic_truth; observed or compatibility labels are not truth distributions")
        index = target["gold_index"]
        return {key: float(index == i) for i, key in enumerate(ids)}, "deterministic_truth", target["gold_label_kind"]
    raise ValueError("raw benchmark requires an independent gold target for every question")


def prediction_row(row, qid, question, answer):
    target, target_kind, label_kind = target_mapping(row, qid, question)
    output = {
        "id": f"{row['id']}:{qid}",
        "state_id": row["state_id"],
        "split": row["split"],
        "family_id": row["family_id"],
        "type": question["type"],
        "candidate_ids": candidate_ids(question),
        "student_probs": answer["probabilities"],
        "gold_probs_kind": target_kind,
        "gold_label_kind": label_kind,
        "metadata": row.get("metadata", {}),
    }
    if target is not None:
        output["gold_probs"] = target
    return output


def run(args):
    if args.batch_states < 1 or args.batch_questions < 0:
        raise ValueError("invalid batch limits")
    splits = {item.strip() for item in args.splits.split(",") if item.strip()} if args.splits else set()
    rows, files = load_rows(args.input, splits)
    for row in rows:
        for qid, question in row["questions"].items():
            target_mapping(row, qid, question)
    # Hash the actual evaluator and its direct metric/loader dependencies before inference.
    manifest = benchmark_manifest(args.input, args.checkpoint, args.device,
                                  script_path=__file__, split_filter=splits)
    manifest["evaluation_code"] = [file_identity(Path(__file__).with_name(name)) for name in (
        "evaluate_pipeline_decisions.py", "train_pipeline_decisions.py", "predict_toy_decisions.py",
        "train_toy_decisions.py", "benchmark_nanojev_v2.py",
    )]
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = sha256_bytes(canonical_json(manifest).encode("utf-8"))
    load_started = time.perf_counter()
    engine = DecisionPredictor(args.checkpoint, device_name=args.device,
                               precision=args.precision, max_length=args.max_length)
    load_ms = (time.perf_counter() - load_started) * 1000
    manifest["hardware"] = hardware_descriptor(engine.device)
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = sha256_bytes(canonical_json(manifest).encode("utf-8"))
    predictions = []
    request_ms = []
    counters = Counter()
    started = time.perf_counter()
    for start in range(0, len(rows), args.batch_states):
        group = rows[start:start + args.batch_states]
        payload = {"states": [{key: row[key] for key in ("id", "state", "questions")} for row in group]}
        request_started = time.perf_counter()
        result = engine.predict(payload, batch_questions=args.batch_questions)
        request_ms.append((time.perf_counter() - request_started) * 1000)
        for key in ("forward_passes", "network_model_calls", "candidate_paths", "questions", "states"):
            counters[key] += result["execution"][key]
        if len(result["states"]) != len(group):
            raise ValueError("predictor returned wrong number of states")
        by_id = {state["id"]: state for state in result["states"]}
        if set(by_id) != {row["id"] for row in group}:
            raise ValueError("predictor returned mismatched state IDs")
        for row in group:
            answers = by_id[row["id"]]["answers"]
            if set(answers) != set(row["questions"]):
                raise ValueError("predictor returned mismatched question IDs")
            for qid, question in row["questions"].items():
                predictions.append(prediction_row(row, qid, question, answers[qid]))
    elapsed_ms = (time.perf_counter() - started) * 1000
    report = evaluate(predictions)
    prediction_text = "".join(canonical_json(row) + "\n" for row in predictions)
    grouped = defaultdict(list)
    for row in predictions:
        grouped[row["metadata"].get("challenge_variant", "original")].append(row)
    report.update({
        "schema_version": "nanojev-raw-dataset-v2-report-v2",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "input": str(Path(args.input).resolve()),
        "input_files": [str(path.resolve()) for path in files],
        "evaluated_records_sha256": sha256_bytes(canonical_json(rows).encode("utf-8")),
        "predictions_sha256": sha256_bytes(prediction_text.encode("utf-8")),
        "arguments": {"splits": sorted(splits), "batch_states": args.batch_states,
                      "batch_questions": args.batch_questions, "max_length": engine.limit},
        "by_variant": {key: evaluate(group) for key, group in sorted(grouped.items())},
        "coverage": {split: {"states": sum(row["split"] == split for row in rows),
                             "source_groups": len({row.get("metadata", {}).get("source_group_id", row["state_id"])
                                                   for row in rows if row["split"] == split})}
                     for split in sorted({row["split"] for row in rows})},
        "execution": {
            "device": str(engine.device), "precision": engine.precision, "load_ms": load_ms,
            "requests": len(request_ms), "cold_request_ms": request_ms[0],
            "warm_requests": len(request_ms[1:]),
            "warm_request_p50_ms": percentile(request_ms[1:], 0.5),
            "warm_request_p95_ms": percentile(request_ms[1:], 0.95),
            "warm_request_p99_ms": percentile(request_ms[1:], 0.99),
            "questions_per_second": len(predictions) / (elapsed_ms / 1000),
            "elapsed_ms": elapsed_ms, "peak_rss_bytes": peak_rss_bytes(), **dict(counters),
            "latency_scope": "Batched predict call including tokenization and host output conversion; not per-question compute. Throughput includes first request and target serialization, excludes model load and provenance hashing.",
        },
        "manifest": manifest,
        "prediction_count": len(predictions),
    })
    if args.predictions:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        args.predictions.write_text(prediction_text, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--splits", default="test,ood")
    parser.add_argument("--batch-states", type=int, default=8)
    parser.add_argument("--batch-questions", type=int, default=0)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16"), default="auto")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_states < 1 or args.batch_questions < 0:
        parser.error("batch sizes must be positive; batch-questions may be zero")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "predictions": result["prediction_count"], "device": result["execution"]["device"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
