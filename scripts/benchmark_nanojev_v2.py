#!/usr/bin/env python3
"""Run fixed NanoJev V2 quality, uncertainty, provenance, and runtime gates."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import resource
import sys
import time
from pathlib import Path

from predict_toy_decisions import DecisionPredictor


METRIC_NAMES = ("accuracy", "nll", "brier", "ece")


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


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
        if not 0 <= gold_index < len(values):
            invalid += 1
            continue
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


def cluster_bootstrap_ci(rows, samples, seed):
    groups = {}
    for row in rows:
        groups.setdefault(row["source_group_id"], []).append(row)
    if samples == 0:
        return None
    if samples < 1 or not groups:
        raise ValueError("bootstrap samples require at least one source group")
    keys = sorted(groups)
    rng = random.Random(seed)
    values = {metric: [] for metric in METRIC_NAMES}
    for _ in range(samples):
        resampled = []
        for _ in keys:
            resampled.extend(groups[rng.choice(keys)])
        metrics = probability_metrics(resampled)
        for metric in METRIC_NAMES:
            if metrics[metric] is not None:
                values[metric].append(metrics[metric])
    return {
        "method": "source_group_cluster_bootstrap",
        "confidence": 0.95,
        "resamples": samples,
        "clusters": len(keys),
        "seed": seed,
        "metrics": {
            metric: {"low": percentile(items, 0.025), "high": percentile(items, 0.975),
                     "valid_resamples": len(items)}
            for metric, items in values.items()
        },
    }


def gold_index(question):
    typ = question["type"]
    if typ == "boolean":
        return 1 if question["gold"] else 0
    if typ == "choice":
        return list(question["criteria"]).index(question["gold"])
    return int(question["gold"])


def resolve_input_files(path, split_filter):
    path = Path(path)
    if path.is_file():
        return [path.resolve()]
    if not path.is_dir():
        raise ValueError(f"input does not exist: {path}")
    names = sorted(split_filter) if split_filter else ["calibration", "dev", "ood", "test", "train"]
    files = [(path / f"{name}.jsonl").resolve() for name in names]
    missing = [str(item) for item in files if not item.is_file()]
    if missing:
        raise ValueError(f"input directory is missing split files: {missing}")
    return files


def load_rows(path, split_filter):
    rows = []
    for source in resolve_input_files(path, split_filter):
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if split_filter and row.get("split") not in split_filter:
                continue
            rows.append(row)
    if not rows:
        raise ValueError("input contains no rows for the requested split")
    return rows


def physical_memory_bytes():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def dependency_versions(names=("torch", "transformers", "safetensors", "numpy")):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def hardware_descriptor(device=None):
    descriptor = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_implementation": platform.python_implementation(),
        "physical_memory_bytes": physical_memory_bytes(),
        "runtime_device": str(device) if device is not None else None,
    }
    descriptor["sha256"] = sha256_bytes(canonical_json(descriptor).encode("utf-8"))
    return descriptor


def benchmark_manifest(input_path, checkpoint, device=None, script_path=None, split_filter=None):
    input_path = Path(input_path).resolve()
    input_files = resolve_input_files(input_path, split_filter or set())
    checkpoint = Path(checkpoint).resolve()
    script_path = Path(script_path or __file__).resolve()
    dataset_manifest_path = (input_path if input_path.is_dir() else input_path.parent) / "manifest.json"
    checkpoint_paths = [
        checkpoint / "config.json",
        checkpoint / "best.safetensors",
        checkpoint / "backbone_config/config.json",
        checkpoint / "tokenizer/tokenizer.json",
        checkpoint / "tokenizer/tokenizer_config.json",
    ]
    missing = [str(path) for path in checkpoint_paths if not path.is_file()]
    if missing:
        raise ValueError(f"checkpoint manifest files are missing: {missing}")
    inputs = [file_identity(path) for path in input_files]
    dependencies = dependency_versions()
    dependency_manifest = {
        "versions": dependencies,
        "sha256": sha256_bytes(canonical_json(dependencies).encode("utf-8")),
    }
    manifest = {
        "schema_version": "nanojev-v2-benchmark-manifest-v1",
        "input": {
            "path": str(input_path),
            "files": inputs,
            "sha256": sha256_bytes(canonical_json(inputs).encode("utf-8")),
        },
        "dataset_manifest": file_identity(dataset_manifest_path) if dataset_manifest_path.is_file() else None,
        "checkpoint": {"directory": str(checkpoint), "files": [file_identity(path) for path in checkpoint_paths]},
        "benchmark_script": file_identity(script_path),
        "dependencies": dependency_manifest,
        "python": {"version": platform.python_version(), "executable": str(Path(sys.executable).resolve())},
        "hardware": hardware_descriptor(device),
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json(manifest).encode("utf-8"))
    return manifest


def resolve_seed_label(checkpoint, explicit):
    if explicit is not None:
        if not explicit.strip():
            raise ValueError("seed-label must be nonempty")
        return explicit.strip()
    config = Path(checkpoint) / "config.json"
    if config.is_file():
        try:
            seed = json.loads(config.read_text(encoding="utf-8")).get("seed")
            if seed is not None:
                return f"seed-{seed}"
        except (OSError, json.JSONDecodeError):
            pass
    return Path(checkpoint).resolve().name


def split_seed(seed, split):
    digest = hashlib.sha256(f"{seed}:{split}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def run(args):
    split_names = args.splits.split(",") if args.splits else ["all"]
    split_filter = set(split_names) if args.splits else set()
    rows = load_rows(args.input, split_filter)
    seed_label = resolve_seed_label(args.checkpoint, args.seed_label)
    load_started = time.perf_counter()
    engine = DecisionPredictor(args.checkpoint, device_name=args.device,
                               precision=args.precision, max_length=args.max_length)
    load_ms = (time.perf_counter() - load_started) * 1000
    predictions = {split: [] for split in split_names}
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
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            source_group = metadata.get("source_group_id") or row["id"]
            family = row.get("family_id") or metadata.get("family_id")
            for qid, question in row["questions"].items():
                answer = answers[qid]
                destination.append({
                    "sample_id": f"{row['id']}:{qid}",
                    "row_id": row["id"],
                    "qid": qid,
                    "split": split,
                    "source_group_id": source_group,
                    "family_id": family,
                    "probabilities": answer["probabilities"],
                    "gold_index": gold_index({**question, "gold": row["gold"][qid]}),
                })

    elapsed_ms = (time.perf_counter() - started) * 1000
    quality = {split: probability_metrics(values) for split, values in predictions.items() if values}
    confidence_intervals = {
        split: cluster_bootstrap_ci(values, args.bootstrap_samples, split_seed(args.bootstrap_seed, split))
        for split, values in predictions.items() if values
    }
    warm_request_ms = request_ms[1:] or request_ms
    result = {
        "schema_version": "nanojev-v2-benchmark-v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "input": str(Path(args.input).resolve()),
        "run": {
            "seed_label": seed_label,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "manifest": benchmark_manifest(args.input, args.checkpoint, engine.device,
                                       split_filter=split_filter),
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
            "peak_rss_bytes": peak_rss_bytes(),
            "forward_passes": forward_passes,
            "network_model_calls": network_calls,
        },
        "quality": quality,
        "confidence_intervals": confidence_intervals,
        "samples": [sample for split in sorted(predictions) for sample in predictions[split]],
        "gates": {
            "zero_invalid_outputs": all(item["invalid_outputs"] == 0 for item in quality.values()),
            "nonempty_quality_reports": bool(quality),
            "manifest_complete": True,
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
    parser.add_argument("--seed-label",
                        help="Training seed or stable run label used for multi-seed comparisons")
    parser.add_argument("--bootstrap-samples", type=int, default=1000,
                        help="Source-group bootstrap resamples; 0 disables confidence intervals")
    parser.add_argument("--bootstrap-seed", type=int, default=20260919)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_states < 1 or args.batch_questions < 0:
        parser.error("batch sizes must be positive; batch-questions may be zero")
    if args.bootstrap_samples < 0:
        parser.error("bootstrap-samples must be nonnegative")
    result = run(args)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
