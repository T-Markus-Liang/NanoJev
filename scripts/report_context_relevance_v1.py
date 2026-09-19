#!/usr/bin/env python3
"""Audit frozen relevance runs without retraining, recalibration, or seed selection."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from benchmark_nanojev_v2 import (cluster_bootstrap_ci, file_identity, probability_metrics,
                                  sha256_file, split_seed)
from build_context_relevance_v1 import validate_cohort
from compare_nanojev_v2 import matched_samples, metric_delta, paired_group_bootstrap


SPLITS = ("calibration", "test", "ood")
THRESHOLD = 0.99  # Frozen protocol, not an optimization parameter.


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def unique_index(rows, key):
    result = {}
    for row in rows:
        identity = row[key]
        if identity in result:
            raise ValueError(f"duplicate {key}: {identity}")
        result[identity] = row
    return result


def join_predictions(records, predictions):
    expected = {f"{row['id']}:irrelevant": row for row in records}
    if len(expected) != len(records):
        raise ValueError("duplicate dataset identity")
    actual = unique_index(predictions, "id")
    if set(actual) != set(expected):
        raise ValueError("prediction and dataset identities differ")
    samples = []
    for identity, row in expected.items():
        pred, meta = actual[identity], row["metadata"]
        if (pred["state_id"] != row["id"] or pred["split"] != row["split"]
                or pred["family_id"] != row["family_id"] or pred["qid"] != "irrelevant"
                or pred["candidate_ids"] != ["false", "true"] or pred["type"] != "boolean"
                or pred["gold_index"] != int(row["gold"]["irrelevant"])):
            raise ValueError("prediction target/order/provenance mismatch")
        probs = pred["student_probs"]
        if len(probs) != 2:
            raise ValueError("expected two probabilities")
        samples.append({"sample_id": identity, "row_id": row["id"], "qid": "irrelevant",
                        "split": row["split"], "source_group_id": meta["source_group_id"],
                        "family_id": row["family_id"], "candidate_kind": meta["candidate_kind"],
                        "wire_format": meta["wire_format"], "gold_index": pred["gold_index"],
                        "probabilities": dict(zip(("false", "true"), probs))})
    if probability_metrics(samples)["invalid_outputs"]:
        raise ValueError("invalid candidate probabilities")
    return samples


def gate_metrics(samples):
    if not samples or probability_metrics(samples)["invalid_outputs"]:
        raise ValueError("gate requires valid nonempty samples")
    drops = [row for row in samples if row["probabilities"]["true"] >= THRESHOLD]
    errors = [row for row in drops if row["gold_index"] == 0]
    groups = {row["source_group_id"] for row in samples}
    bad_groups = {row["source_group_id"] for row in errors}
    return {"threshold": THRESHOLD, "questions": len(samples), "source_groups": len(groups),
            "proposed_drops": len(drops), "false_drops": len(errors),
            "false_drop_groups": len(bad_groups), "false_drop_sample_ids": [r["sample_id"] for r in errors],
            "proposed_drop_rate": len(drops) / len(samples),
            "false_drop_rate_among_proposals": len(errors) / len(drops) if drops else None,
            "retained_or_abstained_rate": 1 - len(drops) / len(samples),
            "zero_error_group_upper95": 1 - 0.05 ** (1 / len(groups)) if not errors else None,
            "upper_bound_scope": "One-sided binomial bound for any false drop in an iid source group; synthetic population only, no production guarantee.",
            "active_deletions": 0, "actual_main_model_tokens_saved": 0}


def breakdown(samples, field):
    groups = defaultdict(list)
    for row in samples:
        groups[row[field]].append(row)
    return {key: {"quality": probability_metrics(rows), "gate": gate_metrics(rows)}
            for key, rows in sorted(groups.items())}


def verify_report(report, data_dir, checkpoint):
    unique_index(report["samples"], "sample_id")
    if probability_metrics(report["samples"])["invalid_outputs"]:
        raise ValueError("invalid benchmark probabilities")
    if Path(report["input"]).resolve() != data_dir.resolve():
        raise ValueError("benchmark input path differs")
    if Path(report["checkpoint"]).resolve() != checkpoint.resolve():
        raise ValueError("benchmark checkpoint differs")
    for entry in report["manifest"]["input"]["files"] + report["manifest"]["checkpoint"]["files"]:
        if sha256_file(entry["path"]) != entry["sha256"]:
            raise ValueError("benchmark source hash differs")


def run(root, bootstrap_samples=1000):
    if bootstrap_samples < 1:
        raise ValueError("bootstrap samples must be positive")
    protocol_path = root / "research/context_relevance_v1_protocol.json"
    protocol = read_json(protocol_path)
    dataset = root / protocol["dataset"]
    baseline_checkpoint = root / protocol["initial_checkpoint"]
    all_records = [row for split in ("train", "dev", *SPLITS) for row in read_lines(dataset / f"{split}.jsonl")]
    validate_cohort(all_records)
    manifest = read_json(dataset / "manifest.json")
    for split, info in manifest["splits"].items():
        if sha256_file(dataset / f"{split}.jsonl") != info["sha256"]:
            raise ValueError("dataset manifest hash mismatch")
    records = [row for row in all_records if row["split"] in SPLITS]
    baseline_path = root / "results/context_relevance_oracle_v1_initialization.json"
    baseline = read_json(baseline_path)
    verify_report(baseline, dataset, baseline_checkpoint)
    baseline_maze_path = root / "results/context_relevance_oracle_v1_maze_initialization.json"
    maze_base = read_json(baseline_maze_path)
    maze_data = root / "dataset/games_v4/data/local_maze_v1"
    verify_report(maze_base, maze_data, baseline_checkpoint)
    result = {"schema_version": "nanojev-context-relevance-report-v1", "protocol": file_identity(protocol_path),
              "data_manifest": file_identity(dataset / "manifest.json"),
              "baseline_receipt": file_identity(baseline_path), "maze_baseline_receipt": file_identity(baseline_maze_path),
              "baseline_quality": baseline["quality"], "maze_baseline_quality": maze_base["quality"],
              "provenance_note": "Current source hashes captured at report time, not a retroactive training-time signature.",
              "source_files": [file_identity(root / "scripts" / name) for name in (
                  "report_context_relevance_v1.py", "build_context_relevance_v1.py", "context_gate_v1.py",
                  "train_pipeline_decisions.py", "benchmark_nanojev_v2.py", "compare_nanojev_v2.py")],
              "uncertainty": "Per-seed source-group intervals; one unchanged initialization, not three independently trained baselines. No pooling repeated groups across seeds.",
              "scope": "Synthetic latest-record dependencies, not general compression, financial performance, or RLCD.",
              "seeds": {}, "deployment": "not promoted; production checkpoint unchanged"}
    for seed in protocol["training_seeds"]:
        folder = root / f"runs/context_relevance_oracle_v1_seed{seed}"
        config, summary = read_json(folder / "config.json"), read_json(folder / "summary.json")
        for key in ("device", "precision", "objective", "loss", "steps", "head_steps", "batch_questions",
                    "microbatch_questions", "max_microbatch_tokens", "max_length", "eval_every",
                    "backbone_lr", "head_lr", "head_warmup_lr"):
            if config[key] != protocol[key]:
                raise ValueError(f"seed {seed}: protocol mismatch {key}")
        if config["seed"] != seed or config["init_checkpoint"] != protocol["initial_checkpoint"]:
            raise ValueError("initialization or seed mismatch")
        if summary["temperature_fitted"] or summary["temperature"] != 1.0:
            raise ValueError("unexpected calibration")
        for name, digest in config["data_sha256"].items():
            if sha256_file(root / name) != digest:
                raise ValueError("training data hash mismatch")
        predictions = [row for split in SPLITS for row in read_lines(folder / f"predictions_{split}.jsonl")]
        samples = join_predictions(records, predictions)
        pairs = matched_samples(baseline, {"samples": samples}, str(seed))
        maze_path = root / f"results/context_relevance_oracle_v1_maze_seed{seed}.json"
        maze = read_json(maze_path)
        verify_report(maze, maze_data, folder)
        maze_pairs = matched_samples(maze_base, maze, str(seed))
        entry = {"summary": summary, "checkpoint": file_identity(folder / "best.safetensors"),
                 "artifacts": [file_identity(folder / name) for name in (
                     "config.json", "summary.json", "initial_dev_metrics.json", "train_log.json",
                     *[f"predictions_{split}.jsonl" for split in SPLITS])],
                 "splits": {}, "maze_receipt": file_identity(maze_path), "maze": {}, "samples": samples}
        for split in SPLITS:
            selected = [r for r in samples if r["split"] == split]
            quality = probability_metrics(selected)
            if not math.isclose(quality["nll"], summary["metrics_by_split"][split]["target_ce"], abs_tol=1e-5):
                raise ValueError("trainer CE and reconstructed NLL disagree")
            entry["splits"][split] = {
                "quality": quality, "gate": gate_metrics(selected),
                "quality_ci95": cluster_bootstrap_ci(selected, bootstrap_samples, split_seed(seed, split)),
                "delta": metric_delta(quality, baseline["quality"][split]),
                "paired_delta_ci95": paired_group_bootstrap({str(seed): pairs}, split, bootstrap_samples, split_seed(seed, split)),
                "by_family": breakdown(selected, "family_id"), "by_kind": breakdown(selected, "candidate_kind"),
                "by_wire": breakdown(selected, "wire_format")}
        for split in ("test", "ood"):
            quality = maze["quality"][split]
            delta = metric_delta(quality, maze_base["quality"][split])
            entry["maze"][split] = {"quality": quality, "delta": delta,
                "within_point_estimate_regression_budget": delta["accuracy"] >= -0.005,
                "paired_delta_ci95": paired_group_bootstrap({str(seed): maze_pairs}, split, bootstrap_samples, split_seed(seed, split))}
        result["seeds"][str(seed)] = entry
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    report = run(args.root, args.bootstrap_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "seeds": list(report["seeds"]), "deployment": report["deployment"]}))
