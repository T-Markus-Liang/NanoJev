#!/usr/bin/env python3
"""T8g read-only preflight: schema success does not prove evaluation isolation.

Never merges/relabels/repartitions a corpus or imports a model. The only optional output
is a new audit receipt; a blocked corpus exits 2 and is not adapted for training.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from build_engineering_corpus_v1 import check, derive_manifest, trainer_rows
from predict_toy_decisions import read_json
from train_pipeline_decisions import read_training_records, validate_training_row

SCHEMA = "nanojev-engineering-adapter-preflight-v1"
SPLITS = ("train", "dev", "calibration", "test")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def visible_input(row, question):
    """Exclude record/qid/source IDs and gold, none of which enter model input.

    Dict key order is normalized; ordinal criteria list order is retained. This is
    exact canonical-input overlap, not a claim to detect every semantic paraphrase.
    """
    return {"state": row["state"], "type": question["type"],
            "instructions": question["instructions"],
            "criteria": question.get("criteria", {})}


def references(rows, cohort=None):
    buckets = defaultdict(list)
    for row in rows:
        targets = validate_training_row(row)
        for qid, question in row["questions"].items():
            ref = {"id": row["id"], "qid": qid, "split": row["split"],
                   "family_id": row["family_id"], "state_id": row["state_id"],
                   "source_group_id": row.get("metadata", {}).get("source_group_id"),
                   "target": targets[qid]["gold_distribution_probs"]}
            # IDs for candidates matter; compare labelled distributions, not vector order.
            ref["target_by_id"] = row.get("gold_probs", {}).get(qid, row.get("gold", {}).get(qid))
            if cohort is not None:
                ref["cohort"] = cohort
            buckets[digest(visible_input(row, question))].append(ref)
    return buckets


def provenance_is_evaluation_derived(source):
    # Recognizes fact-only aliases, not just one literal filename.
    normalized = re.sub(r"[-_\s]+", " ", str(source).lower())
    return any(marker in normalized for marker in (
        "abstention survey", "workflow challenge", "workflow v2 evaluation",
        "context relevance test", "context relevance ood"))


def audit_rows(rows):
    buckets = references(rows)
    conflicts = []
    for fingerprint, refs in sorted(buckets.items()):
        if len({ref["split"] for ref in refs}) > 1:
            conflicts.append({"input_sha256": fingerprint,
                              "splits": sorted({ref["split"] for ref in refs}),
                              "references": refs,
                              "conflicting_targets": len({canonical(ref["target_by_id"]) for ref in refs}) > 1})
    registries = {"state_id": defaultdict(set), "source_group_id": defaultdict(set)}
    missing_sources, record_ids, duplicate_ids, provenance_hits = [], set(), [], []
    for row in rows:
        if row["id"] in record_ids:
            duplicate_ids.append(row["id"])
        record_ids.add(row["id"])
        metadata = row.get("metadata", {})
        source = metadata.get("source_group_id")
        if not isinstance(source, str) or not source:
            missing_sources.append(row["id"])
        registries["state_id"][row["state_id"]].add(row["split"])
        if source:
            registries["source_group_id"][source].add(row["split"])
        source_id = metadata.get("provenance_source_id", "")
        if metadata.get("derived_from_evaluation_corpus") is True or provenance_is_evaluation_derived(source_id):
            provenance_hits.append({"id": row["id"], "split": row["split"], "source": source_id,
                                    "declared_derived_flag": metadata.get("derived_from_evaluation_corpus")})
    identity_conflicts = {key: {identity: sorted(splits) for identity, splits in sorted(registry.items())
                                if len(splits) > 1} for key, registry in registries.items()}
    reasons = []
    if conflicts:
        reasons.append("canonical_model_input_crosses_splits")
    if any(identity_conflicts.values()):
        reasons.append("declared_identity_crosses_splits")
    if missing_sources:
        reasons.append("missing_source_group_id")
    if duplicate_ids:
        reasons.append("duplicate_record_id")
    if provenance_hits:
        reasons.append("evaluation_derived_provenance_requires_review")
    return {
        "record_count": len(rows), "question_count": sum(len(row["questions"]) for row in rows),
        "counts_by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
        "declared_source_groups": len(registries["source_group_id"]),
        "declared_identity_conflicts": identity_conflicts,
        "missing_source_group_ids": missing_sources, "duplicate_record_ids": duplicate_ids,
        "canonical_input_cross_split_groups": len(conflicts),
        "affected_records": len({ref["id"] for entry in conflicts for ref in entry["references"]}),
        "cross_split_combinations": dict(sorted(Counter("|".join(entry["splits"]) for entry in conflicts).items())),
        "conflicts": conflicts, "evaluation_provenance_hits": provenance_hits,
        "block_reasons": reasons,
    }


def compare_cohorts(left, right):
    a, b = references(left, "original"), references(right, "comparison")
    common = sorted(set(a) & set(b))
    source_ids = lambda rows: {row.get("metadata", {}).get("source_group_id") for row in rows} - {None}
    return {"common_canonical_inputs": len(common),
            "left_affected_records": len({ref["id"] for key in common for ref in a[key]}),
            "right_affected_records": len({ref["id"] for key in common for ref in b[key]}),
            "declared_source_id_intersection": len(source_ids(left) & source_ids(right)),
            "overlap": [{"input_sha256": key, "references": a[key] + b[key]} for key in common]}


def audit_corpus(root, compare_seed=None):
    root = Path(root).resolve(strict=True)
    paths = [root / "manifest.json"] + [root / view / f"{split}.jsonl"
             for view in ("items", "trainer_view") for split in SPLITS]
    hashes_before = {str(path.relative_to(root)): file_hash(path) for path in paths}
    integrity_errors = check(root)
    manifest = read_json(root / "manifest.json")
    # Run the real trainer's directory-level check, not only the per-row validator.
    rows, _ = read_training_records(root / "trainer_view")
    audit = audit_rows(rows)
    reasons = list(audit["block_reasons"])
    if integrity_errors:
        reasons.append("builder_integrity_check_failed")
    report = {"schema_version": SCHEMA, "corpus": str(root),
              "source_hashes": hashes_before, "builder_integrity_errors": integrity_errors,
              "trainer_schema_and_declared_split_checks": "passed",
              "audit": audit, "block_reasons": reasons,
              "pair_count": manifest["pair_count"],
              "catalog_rule_count": len({item["provenance"]["rule_id"] for item in manifest["items"]}),
              "training_authorized": False, "training_performed": False,
              "merged_rows_written": 0, "input_files_changed": False,
              "review_requirements": ["No split/label changes without re-review",
                  "Content-connected pair grouping, not seed/ID-only grouping",
                  "Remove or review evaluation-derived sources in a new corpus version",
                  "Resolve gate scorer serialization and merged heldout composition before adapting",
                  "T9d training protocol remains unapproved"],
              "limitations": ["Canonical equality is a lower bound on leakage; paraphrases may remain",
                  "A clean preflight would not approve labels, corpus scale, training or deployment"]}
    if compare_seed is not None:
        if compare_seed == manifest["seed"]:
            raise ValueError("Comparison seed must differ from corpus seed")
        # Pure in-memory construction, not another materialized training/evaluation cohort.
        other = trainer_rows(derive_manifest(compare_seed))
        comparison = compare_cohorts(rows, other)
        report["cross_seed"] = {"original_seed": manifest["seed"], "comparison_seed": compare_seed,
                                "materialized": False, **comparison}
        if comparison["common_canonical_inputs"]:
            reasons.append("seed_change_does_not_prove_semantic_holdout")
    hashes_after = {str(path.relative_to(root)): file_hash(path) for path in paths}
    if hashes_after != hashes_before:
        raise ValueError("Corpus changed during read-only audit; do not use this receipt")
    report["status"] = "blocked" if reasons else "preflight_passed_not_training_authorized"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--compare-seed", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output and args.output.resolve().is_relative_to(args.corpus.resolve()):
        parser.error("Audit output cannot be inside the read-only input corpus")
    try:
        report = audit_corpus(args.corpus, args.compare_seed)
        serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
            print(json.dumps({"status": report["status"], "output": str(args.output),
                              "block_reasons": report["block_reasons"]}))
        else:
            print(serialized, end="")
        return 2 if report["block_reasons"] else 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "error", "error": str(error), "training_authorized": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
