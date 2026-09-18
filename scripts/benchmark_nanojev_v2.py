#!/usr/bin/env python3
"""Run the fixed V2 quality and runtime gates for a local NanoJev checkpoint."""
import argparse
import json
import math
import time
from pathlib import Path

from predict_toy_decisions import DecisionPredictor


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def mean(values):
    return math.fsum(values) / len(values) if values else None


def probability_metrics(rows):
    correct, nll, brier, confidences = [], [], [], []
    invalid = 0
    for row in rows:
        probabilities = row["probabilities"]
        values = list(probabilities.values())
        if not values or not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
            invalid += 1
            continue
        if abs(math.fsum(values) - 1.0) > 1e-5:
            invalid += 1
            continue
        gold_index = row["gold_index"]
        prediction = max(range(len(values)), key=values.__getitem__)
        correct_value = int(prediction == gold_index)
        correct.append(correct_value)
        nll.append(-math.log(max(values[gold_index], 1e-30)))
        brier.append(math.fsum((value - int(index == gold_index)) ** 2
                                for index, value in enumerate(values)))
        confidences.append((max(values), correct_value))

    bins = [{"n": 0, "confidence": 0.0, "correct": 0.0} for _ in range(10)]
    for confidence, correct_value in confidences:
        bucket = min(9, int(confidence * 10))
        bins[bucket]["n"] += 1
        bins[bucket]["confidence"] += confidence
        bins[bucket]["correct"] += correct_value
    ece = 0.0
    for bucket in bins:
        if bucket["n"]:
            confidence = bucket["confidence"] / bucket["n"]
            accuracy = bucket["correct"] / bucket["n"]
            ece += bucket["n"] / len(confidences) * abs(confidence - accuracy)

    return {
        "questions": len(correct),
        "accuracy": mean(correct),
        "nll": mean(nll),
        "brier": mean(brier),
        "ece": ece if confidences else None,
        "invalid_outputs": invalid,
    }


def gold_index(question, answer):
    typ = question["type"]
    if typ == "boolean":
        return 1 if question["gold"] else 0
    if typ == "choice":
        return list(question["criteria"]).index(question["gold"])
    return int(question["gold"])


def load_rows(path, split_filter):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if split_filter and row.get("split") not in split_filter:
            continue
        rows.append(row)
    if not rows:
        raise ValueError("input contains no rows for the requested split")
    return rows


def run(args):
    rows = load_rows(args.input, set(args.splits.split(",")) if args.splits else set())
    load_started = time.perf_counter()
    engine = DecisionPredictor(args.checkpoint, device_name=args.device,
                               precision=args.precision, max_length=args.max_length)
    load_ms = (time.perf_counter() - load_started) * 1000
    predictions = {split: [] for split in (args.splits.split(",") if args.splits else ["all"])}
    request_ms = []
    forward_passes = 0
    network_calls = 0
    started = time.perf_counter()

    for start in range(0, len(rows), args.batch_states):
        group = rows[start:start + args.batch_states]
        payload = {"states": [{"id": row["id"], "state": row["state"],
                               "questions": row["questions"]} for row in group]}
        request_started = time.perf_counter()
        result = engine.predict(payload, batch_questions=args.batch_questions)
        request_ms.append((time.perf_counter() - request_started) * 1000)
        forward_passes += result["execution"]["forward_passes"]
        network_calls += result["execution"]["network_model_calls"]
        by_id = {state["id"]: state for state in result["states"]}
        for row in group:
            split = row.get("split", "all")
            destination = predictions.setdefault(split, [])
            answers = by_id[row["id"]]["answers"]
            for qid, question in row["questions"].items():
                answer = answers[qid]
                destination.append({
                    "probabilities": answer["probabilities"],
                    "gold_index": gold_index({**question, "gold": row["gold"][qid]}, answer),
                })

    elapsed_ms = (time.perf_counter() - started) * 1000
    quality = {split: probability_metrics(values) for split, values in predictions.items() if values}
    warm_request_ms = request_ms[1:] or request_ms
    result = {
        "schema_version": "nanojev-v2-benchmark-v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "input": str(Path(args.input).resolve()),
        "execution": {
            "device": str(engine.device),
            "precision": engine.precision,
            "load_ms": load_ms,
            "requests": len(request_ms),
            "cold_request_ms": request_ms[0],
            "warm_request_count": len(warm_request_ms),
            "warm_request_p50_ms": percentile(warm_request_ms, 0.50),
            "warm_request_p95_ms": percentile(warm_request_ms, 0.95),
            "warm_request_min_ms": min(warm_request_ms),
            "questions_per_second": sum(item["questions"] for item in quality.values()) / (elapsed_ms / 1000),
            "forward_passes": forward_passes,
            "network_model_calls": network_calls,
        },
        "quality": quality,
        "gates": {
            "zero_invalid_outputs": all(item["invalid_outputs"] == 0 for item in quality.values()),
            "nonempty_quality_reports": bool(quality),
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--splits", default="test,ood")
    parser.add_argument("--batch-states", type=int, default=8)
    parser.add_argument("--batch-questions", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_states < 1 or args.batch_questions < 0:
        parser.error("batch sizes must be positive; batch-questions may be zero")
    result = run(args)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
