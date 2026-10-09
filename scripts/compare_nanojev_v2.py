#!/usr/bin/env python3
"""Compare matched NanoJev V2 benchmark reports across checkpoints and seeds."""

import argparse
import json
import math
import random
from pathlib import Path


METRIC_NAMES = ("accuracy", "nll", "brier", "ece")


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
    for row in rows:
        values = list(row["probabilities"].values())
        if not values or not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
            raise ValueError(f"invalid probabilities for {row.get('sample_id')}")
        if abs(math.fsum(values) - 1.0) > 1e-5:
            raise ValueError(f"probabilities do not sum to one for {row.get('sample_id')}")
        gold_index = row["gold_index"]
        if not 0 <= gold_index < len(values):
            raise ValueError(f"invalid gold index for {row.get('sample_id')}")
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
            ece += bucket["n"] / len(confidences) * abs(
                bucket["confidence"] / bucket["n"] - bucket["correct"] / bucket["n"])
    return {"questions": len(rows), "accuracy": mean(correct), "nll": mean(nll),
            "brier": mean(brier), "ece": ece if rows else None}


def metric_delta(candidate, baseline):
    return {metric: candidate[metric] - baseline[metric] for metric in METRIC_NAMES}


def load_report(path):
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "nanojev-v2-benchmark-v1":
        raise ValueError(f"{path}: unsupported benchmark schema")
    seed = report.get("run", {}).get("seed_label")
    samples = report.get("samples")
    if not isinstance(seed, str) or not seed.strip():
        raise ValueError(f"{path}: missing nonempty run.seed_label")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{path}: missing benchmark samples")
    report["_path"] = str(path.resolve())
    return report


def reports_by_seed(paths, arm):
    reports = {}
    for path in paths:
        report = load_report(path)
        seed = report["run"]["seed_label"]
        if seed in reports:
            raise ValueError(f"{arm}: duplicate seed label {seed}")
        reports[seed] = report
    return reports


def identity(sample):
    return (sample["sample_id"], sample["split"], sample["row_id"], sample["qid"],
            sample["source_group_id"], sample.get("family_id"), sample["gold_index"])


def matched_samples(baseline, candidate, seed):
    baseline_by_id = {sample["sample_id"]: sample for sample in baseline["samples"]}
    candidate_by_id = {sample["sample_id"]: sample for sample in candidate["samples"]}
    if set(baseline_by_id) != set(candidate_by_id):
        raise ValueError(f"seed {seed}: baseline and candidate sample IDs differ")
    pairs = []
    for sample_id in sorted(baseline_by_id):
        base = baseline_by_id[sample_id]
        cand = candidate_by_id[sample_id]
        if identity(base) != identity(cand):
            raise ValueError(f"seed {seed}: cohort identity differs for {sample_id}")
        if list(base["probabilities"]) != list(cand["probabilities"]):
            raise ValueError(f"seed {seed}: candidate ordering differs for {sample_id}")
        pairs.append((base, cand))
    return pairs


def bootstrap_mean_ci(values, samples, seed):
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    estimates = [mean([rng.choice(values) for _ in values]) for _ in range(samples)]
    return {"low": percentile(estimates, 0.025), "high": percentile(estimates, 0.975),
            "method": "training_seed_bootstrap", "resamples": samples, "seed": seed}


def aggregate_seed_metrics(by_seed, bootstrap_samples, bootstrap_seed):
    result = {"seed_count": len(by_seed), "by_seed": by_seed, "metrics": {}}
    for offset, metric in enumerate(METRIC_NAMES):
        values = [by_seed[seed][metric] for seed in sorted(by_seed)]
        result["metrics"][metric] = {
            "mean": mean(values),
            "multi_seed_ci95": bootstrap_mean_ci(values, bootstrap_samples, bootstrap_seed + offset),
        }
    return result


def paired_group_bootstrap(pairs_by_seed, split, samples, seed):
    units = {}
    for seed_label, pairs in pairs_by_seed.items():
        for baseline, candidate in pairs:
            if baseline["split"] != split:
                continue
            key = (seed_label, baseline["source_group_id"])
            units.setdefault(key, []).append((baseline, candidate))
    if not units:
        raise ValueError(f"split {split}: no paired source groups")
    if samples == 0:
        return None
    keys = sorted(units)
    rng = random.Random(seed)
    values = {metric: [] for metric in METRIC_NAMES}
    for _ in range(samples):
        baseline_rows, candidate_rows = [], []
        for _ in keys:
            selected = units[rng.choice(keys)]
            baseline_rows.extend(pair[0] for pair in selected)
            candidate_rows.extend(pair[1] for pair in selected)
        delta = metric_delta(probability_metrics(candidate_rows), probability_metrics(baseline_rows))
        for metric in METRIC_NAMES:
            values[metric].append(delta[metric])
    return {
        "method": "paired_seed_source_group_cluster_bootstrap",
        "confidence": 0.95,
        "resamples": samples,
        "clusters": len(keys),
        "seed": seed,
        "metrics": {metric: {"low": percentile(items, 0.025), "high": percentile(items, 0.975)}
                    for metric, items in values.items()},
    }


def compare(baseline_reports, candidate_reports, bootstrap_samples=2000, bootstrap_seed=20260919):
    baseline = reports_by_seed(baseline_reports, "baseline")
    candidate = reports_by_seed(candidate_reports, "candidate")
    if set(baseline) != set(candidate):
        raise ValueError("baseline and candidate seed labels must match exactly")
    if bootstrap_samples < 1:
        raise ValueError("comparison bootstrap samples must be positive")
    seeds = sorted(baseline)
    pairs_by_seed = {seed: matched_samples(baseline[seed], candidate[seed], seed) for seed in seeds}
    splits = sorted({sample["split"] for report in baseline.values() for sample in report["samples"]})
    output = {
        "schema_version": "nanojev-v2-comparison-v1",
        "seeds": seeds,
        "bootstrap": {"samples": bootstrap_samples, "seed": bootstrap_seed},
        "reports": {
            "baseline": {seed: baseline[seed]["_path"] for seed in seeds},
            "candidate": {seed: candidate[seed]["_path"] for seed in seeds},
        },
        "splits": {},
    }
    for split_index, split in enumerate(splits):
        baseline_by_seed, candidate_by_seed, delta_by_seed = {}, {}, {}
        for seed in seeds:
            pairs = [pair for pair in pairs_by_seed[seed] if pair[0]["split"] == split]
            if not pairs:
                raise ValueError(f"seed {seed}: split {split} has no samples")
            baseline_metrics = probability_metrics([pair[0] for pair in pairs])
            candidate_metrics = probability_metrics([pair[1] for pair in pairs])
            baseline_by_seed[seed] = baseline_metrics
            candidate_by_seed[seed] = candidate_metrics
            delta_by_seed[seed] = metric_delta(candidate_metrics, baseline_metrics)
        output["splits"][split] = {
            "baseline": aggregate_seed_metrics(baseline_by_seed, bootstrap_samples,
                                               bootstrap_seed + split_index * 100),
            "candidate": aggregate_seed_metrics(candidate_by_seed, bootstrap_samples,
                                                bootstrap_seed + split_index * 100 + 10),
            "candidate_minus_baseline": aggregate_seed_metrics(delta_by_seed, bootstrap_samples,
                                                               bootstrap_seed + split_index * 100 + 20),
            "paired_source_group_ci95": paired_group_bootstrap(
                pairs_by_seed, split, bootstrap_samples, bootstrap_seed + split_index * 100 + 30),
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, action="append", required=True)
    parser.add_argument("--candidate-report", type=Path, action="append", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260919)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = compare(args.baseline_report, args.candidate_report,
                         args.bootstrap_samples, args.bootstrap_seed)
    except ValueError as error:
        parser.error(str(error))
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
