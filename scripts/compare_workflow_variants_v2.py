#!/usr/bin/env python3
"""Paired, source-clustered robustness analysis; never select a checkpoint or threshold."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random

from benchmark_nanojev_v2 import file_identity, percentile
from build_workflow_challenge_v2 import VARIANTS
from evaluate_pipeline_decisions import divergence, prepare_row
from predict_toy_decisions import reject_nonfinite, unique_object


def index_rows(rows):
    indexed = defaultdict(dict)
    for raw in rows:
        row = prepare_row(raw)
        metadata = row.get("metadata", {})
        name, base = metadata.get("challenge_variant"), metadata.get("base_record_id")
        if name not in VARIANTS or not isinstance(base, str) or not base:
            raise ValueError("missing challenge variant/base identity")
        if row["split"] not in {"test", "ood"} or not metadata.get("source_group_id"):
            raise ValueError("challenge needs held-out split and source group")
        prefix = f"{base}::{name}:"
        if not row["id"].startswith(prefix) or not row["id"][len(prefix):]:
            raise ValueError("question identity does not match challenge record")
        key = (row["split"], row["family_id"], base, row["id"][len(prefix):])
        if name in indexed[key]:
            raise ValueError("duplicate challenge question")
        if row["_gold"] is None:
            raise ValueError("missing independent gold target")
        indexed[key][name] = row
    if not indexed:
        raise ValueError("no predictions")
    source_splits = {}
    for variants in indexed.values():
        if set(variants) != set(VARIANTS):
            raise ValueError("incomplete variant coverage; cannot silently discard unmatched cases")
        original = variants["original"]
        gold = dict(zip(original["candidate_ids"], original["_gold"]))
        group = original["metadata"]["source_group_id"]
        split = original["split"]
        if group in source_splits and source_splits[group] != split:
            raise ValueError("source group crosses splits")
        source_splits[group] = split
        for row in variants.values():
            if (row["_kind"] != original["_kind"] or row["type"] != original["type"]
                    or row["metadata"]["source_group_id"] != group
                    or dict(zip(row["candidate_ids"], row["_gold"])) != gold):
                raise ValueError("variant changed gold semantics or source identity")
    return indexed


def metrics(row):
    p, q = row["student_probs"], row["_gold"]
    best = max(range(len(p)), key=p.__getitem__)
    result = {"brier": divergence(q, p)["expected_brier"], "tv": divergence(q, p)["tv"]}
    if row["_kind"] == "deterministic_truth":
        result["accuracy"] = float(q[best] == 1)
    return result


def paired_summary(pairs, samples, seed):
    groups = defaultdict(list)
    maximum_drift, flips, selected, wrong = 0.0, 0, 0, 0
    for original, other in pairs:
        reference, candidate = metrics(original), metrics(other)
        groups[original["metadata"]["source_group_id"]].append(
            {key: candidate[key] - value for key, value in reference.items()})
        p = dict(zip(original["candidate_ids"], original["student_probs"]))
        r = dict(zip(other["candidate_ids"], other["student_probs"]))
        maximum_drift = max(maximum_drift, max(abs(p[key] - r[key]) for key in p))
        flips += max(p, key=p.get) != max(r, key=r.get)
        if other["_kind"] == "deterministic_truth" and max(r.values()) >= 0.9:
            selected += 1
            wrong += candidate["accuracy"] == 0
    names = sorted({key for rows in groups.values() for row in rows for key in row})
    ordered = sorted(groups)
    # Resample whole source groups, preserving all correlated questions within each group.
    sums = {group: {name: (math.fsum(row[name] for row in groups[group] if name in row),
                           sum(name in row for row in groups[group])) for name in names} for group in ordered}
    rng = random.Random(seed)
    distributions = {name: [] for name in names}
    for _ in range(samples):
        drawn = [rng.choice(ordered) for _ in ordered]
        for name in names:
            denominator = sum(sums[group][name][1] for group in drawn)
            if denominator:
                distributions[name].append(math.fsum(sums[group][name][0] for group in drawn) / denominator)
    return {
        "paired_questions": len(pairs), "source_groups": len(groups),
        "top_choice_flips": flips, "maximum_absolute_probability_drift": maximum_drift,
        "deterministic_confidence_ge_0_9": {"selected": selected, "wrong": wrong,
                                             "error_rate": wrong / selected if selected else None},
        "deltas_variant_minus_original": {
            name: {"mean": math.fsum(sums[group][name][0] for group in ordered) /
                            sum(sums[group][name][1] for group in ordered),
                   "ci95": [percentile(distributions[name], 0.025), percentile(distributions[name], 0.975)],
                   "valid_resamples": len(distributions[name])} for name in names},
    }


def compare(rows, samples=1000, seed=17):
    if type(samples) is not int or samples < 1:
        raise ValueError("positive bootstrap samples required")
    indexed = index_rows(rows)
    groups = defaultdict(list)
    for key, variants in sorted(indexed.items()):
        original = variants["original"]
        for name in VARIANTS[1:]:
            groups[f"{key[0]}/{key[1]}/{original['type']}/{name}"].append((original, variants[name]))
    return {
        "schema_version": "nanojev-workflow-paired-challenge-v1",
        "bootstrap": {"method": "paired_source_group_cluster_bootstrap", "resamples": samples, "seed": seed},
        "scope": "Single checkpoint, fixed synthetic test/OOD groups. Intervals are conditional on this checkpoint, not training-seed uncertainty. Choice reversal leaves Boolean/Score unchanged as controls. No checkpoint or threshold selection.",
        "delta_direction": "Higher accuracy is better; lower Brier/TV is better. A flip is not automatically an error.",
        "base_questions": len(indexed), "variant_questions": sum(len(v) for v in indexed.values()),
        "comparisons": {key: paired_summary(pairs, samples, seed) for key, pairs in sorted(groups.items())},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    rows = [json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_nonfinite)
            for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = compare(rows, args.bootstrap_samples, args.seed)
    report["predictions"] = file_identity(args.predictions)
    report["analysis_script"] = file_identity(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "comparisons": len(report["comparisons"])}))
