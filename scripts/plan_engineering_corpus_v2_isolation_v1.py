#!/usr/bin/env python3
"""Build a read-only isolation plan for the blocked engineering corpus.

The plan groups the existing V1 records by both contrastive lineage (``pair_id``)
and exact canonical model-visible input.  It reports connected components and
split/provenance conflicts, but it never rewrites, repartitions, merges, or
materializes a new corpus.  A blocked plan is evidence for the next reviewed
corpus version, not training authorization.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from audit_engineering_corpus_v1 import (
    SPLITS,
    canonical,
    digest,
    file_hash,
    provenance_is_evaluation_derived,
    visible_input,
)
from build_engineering_corpus_v1 import check, derive_manifest, trainer_rows
from train_pipeline_decisions import read_training_records, validate_training_row


SCHEMA = "nanojev-engineering-isolation-plan-v1"


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        parent = self.parent
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _source_paths(root: Path) -> list[Path]:
    return [root / "manifest.json"] + [
        root / "trainer_view" / f"{split}.jsonl" for split in SPLITS
    ]


def _snapshot(paths: list[Path], root: Path) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for path in paths:
        relative = str(path.relative_to(root))
        result[relative] = file_hash(path) if path.is_file() else None
    return result


def _base_report(root: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "plan_only": True,
        "materialized": False,
        "corpus_identity": "unresolved",
        "source_hashes_before": {},
        "source_hashes_after": {},
        "input_files_changed": False,
        "builder_integrity_errors": [],
        "record_count": 0,
        "question_count": 0,
        "component_count": 0,
        "conflicted_component_count": 0,
        "canonical_input_cross_split_groups": 0,
        "evaluation_provenance_record_count": 0,
        "components": [],
        "cross_seed": None,
        "violations": [],
        "block_reasons": [],
        "training_authorized": False,
        "training_performed": False,
        "measurement_authorized": False,
        "measurement_performed": False,
        "deployment_authorized": False,
        "deployment_performed": False,
        "merged_rows_written": 0,
        "status": "blocked_isolation_plan_only",
    }


def _add(report: dict[str, Any], code: str, detail: Any) -> None:
    report["violations"].append({"code": code, "detail": detail})


def _references(rows: list[dict[str, Any]], report: dict[str, Any]) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for row in rows:
        validate_training_row(row)
        metadata = row.get("metadata", {})
        pair_id = metadata.get("pair_id")
        member = metadata.get("member")
        source_group_id = metadata.get("source_group_id")
        provenance_source_id = metadata.get("provenance_source_id", "")
        if not isinstance(pair_id, str) or not pair_id.strip():
            _add(report, "missing_pair_id", row.get("id"))
            pair_id = f"missing:{row.get('id')}"
        if member not in ("base", "variant"):
            _add(report, "missing_member", row.get("id"))
            member = "unknown"
        if not isinstance(source_group_id, str) or not source_group_id.strip():
            _add(report, "missing_source_group_id", row.get("id"))
            source_group_id = None
        for qid, question in row["questions"].items():
            input_sha256 = digest(visible_input(row, question))
            references.append({
                "record_id": row["id"],
                "qid": qid,
                "split": row["split"],
                "pair_id": pair_id,
                "member": member,
                "state_id": row["state_id"],
                "family_id": row["family_id"],
                "source_group_id": source_group_id,
                "provenance_source_id": provenance_source_id,
                "evaluation_derived": bool(
                    metadata.get("derived_from_evaluation_corpus") is True
                    or provenance_is_evaluation_derived(provenance_source_id)
                ),
                "input_sha256": input_sha256,
            })
    return references


def _component_id(pair_ids: list[str], input_hashes: list[str]) -> str:
    payload = {"pair_ids": pair_ids, "input_sha256": input_hashes}
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def component_plan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a deterministic component plan without exposing model-visible text."""
    report: dict[str, Any] = {
        "record_count": len(rows),
        "question_count": 0,
        "components": [],
        "violations": [],
    }
    refs = _references(rows, report)
    report["question_count"] = len(refs)
    if not refs:
        _add(report, "empty_corpus", "no trainer questions")
        report["components"] = []
        return report

    union_find = UnionFind(len(refs))
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, reference in enumerate(refs):
        buckets[("pair", reference["pair_id"])].append(index)
        buckets[("input", reference["input_sha256"])].append(index)
    for indexes in buckets.values():
        first = indexes[0]
        for index in indexes[1:]:
            union_find.union(first, index)

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, reference in enumerate(refs):
        grouped[union_find.find(index)].append(reference)

    components = []
    for references in grouped.values():
        pair_ids = sorted({ref["pair_id"] for ref in references})
        input_hashes = sorted({ref["input_sha256"] for ref in references})
        splits = sorted({ref["split"] for ref in references})
        evaluation_records = sorted({
            ref["record_id"] for ref in references if ref["evaluation_derived"]
        })
        component = {
            "component_id": _component_id(pair_ids, input_hashes),
            "pair_ids": pair_ids,
            "input_sha256": input_hashes,
            "record_ids": sorted({ref["record_id"] for ref in references}),
            "question_count": len(references),
            "split_names": splits,
            "split_conflict": len(splits) > 1,
            "source_group_ids": sorted({
                ref["source_group_id"] for ref in references
                if ref["source_group_id"] is not None
            }),
            "evaluation_derived_record_ids": evaluation_records,
            "safe_action": (
                "quarantine_until_review"
                if len(splits) > 1 or evaluation_records
                else "eligible_for_reassignment_after_review"
            ),
        }
        if component["split_conflict"]:
            _add(report, "component_crosses_splits", {
                "component_id": component["component_id"],
                "splits": splits,
                "pair_ids": pair_ids,
            })
        if evaluation_records:
            _add(report, "evaluation_derived_provenance", {
                "component_id": component["component_id"],
                "record_ids": evaluation_records,
            })
        components.append(component)

    components.sort(key=lambda value: value["component_id"])
    canonical_buckets: dict[str, set[str]] = defaultdict(set)
    pair_splits: dict[str, set[str]] = defaultdict(set)
    source_splits: dict[str, set[str]] = defaultdict(set)
    for reference in refs:
        canonical_buckets[reference["input_sha256"]].add(reference["split"])
        pair_splits[reference["pair_id"]].add(reference["split"])
        if reference["source_group_id"] is not None:
            source_splits[reference["source_group_id"]].add(reference["split"])
    report["canonical_input_cross_split_groups"] = sum(
        len(splits) > 1 for splits in canonical_buckets.values()
    )
    report["evaluation_provenance_record_count"] = len({
        ref["record_id"] for ref in refs if ref["evaluation_derived"]
    })
    report["component_count"] = len(components)
    report["conflicted_component_count"] = sum(
        component["split_conflict"] for component in components
    )
    for pair_id, splits in sorted(pair_splits.items()):
        if len(splits) > 1:
            _add(report, "pair_id_crosses_splits", {"pair_id": pair_id, "splits": sorted(splits)})
    for source_group_id, splits in sorted(source_splits.items()):
        if len(splits) > 1:
            _add(report, "source_group_id_crosses_splits", {
                "source_group_id": source_group_id,
                "splits": sorted(splits),
            })
    report["components"] = components
    return report


def _cross_seed_summary(rows: list[dict[str, Any]], seed: int, compare_seed: int) -> dict[str, Any]:
    if seed == compare_seed:
        raise ValueError("Comparison seed must differ from corpus seed")
    other_rows = trainer_rows(derive_manifest(compare_seed))
    left: dict[str, set[str]] = defaultdict(set)
    right: dict[str, set[str]] = defaultdict(set)
    for collection, target in ((rows, left), (other_rows, right)):
        for row in collection:
            for question in row["questions"].values():
                target[digest(visible_input(row, question))].add(row["id"])
    common = set(left) & set(right)
    return {
        "original_seed": seed,
        "comparison_seed": compare_seed,
        "materialized": False,
        "common_canonical_inputs": len(common),
        "left_affected_records": len({record_id for key in common for record_id in left[key]}),
        "right_affected_records": len({record_id for key in common for record_id in right[key]}),
    }


def plan_corpus(corpus_root: Path, compare_seed: int | None = None) -> dict[str, Any]:
    root = Path(corpus_root).resolve(strict=True)
    report = _base_report(root)
    paths = _source_paths(root)
    report["source_hashes_before"] = _snapshot(paths, root)
    integrity_errors = check(root)
    report["builder_integrity_errors"] = list(integrity_errors)
    if integrity_errors:
        _add(report, "builder_integrity_check_failed", integrity_errors)
    rows, _ = read_training_records(root / "trainer_view")
    component_report = component_plan(rows)
    for key in (
        "record_count", "question_count", "component_count", "conflicted_component_count",
        "canonical_input_cross_split_groups", "evaluation_provenance_record_count",
        "components",
    ):
        report[key] = component_report[key]
    report["violations"].extend(component_report["violations"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    catalog_version = manifest.get("catalog_version", "unknown")
    seed = manifest.get("seed", "unknown")
    report["corpus_identity"] = f"{catalog_version}@seed-{seed}"
    if compare_seed is not None:
        report["cross_seed"] = _cross_seed_summary(rows, manifest["seed"], compare_seed)
        if report["cross_seed"]["common_canonical_inputs"]:
            _add(report, "seed_change_does_not_prove_semantic_holdout", report["cross_seed"])
    else:
        _add(report, "cross_seed_comparison_required", "pass --compare-seed for a semantic-holdout check")
    report["source_hashes_after"] = _snapshot(paths, root)
    report["input_files_changed"] = report["source_hashes_after"] != report["source_hashes_before"]
    if report["input_files_changed"]:
        _add(report, "input_changed_during_plan", "read-only inputs changed")
    report["block_reasons"] = sorted({entry["code"] for entry in report["violations"]})
    report["status"] = (
        "blocked_isolation_plan_only" if report["block_reasons"]
        else "isolation_plan_clean_not_training_authorized"
    )
    report["training_authorized"] = False
    report["training_performed"] = False
    report["measurement_authorized"] = False
    report["measurement_performed"] = False
    report["deployment_authorized"] = False
    report["deployment_performed"] = False
    report["merged_rows_written"] = 0
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--compare-seed", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.corpus.is_dir():
        if args.output.resolve().is_relative_to(args.corpus.resolve()):
            parser.error("plan output cannot be written inside the read-only corpus")
    try:
        report = plan_corpus(args.corpus, args.compare_seed)
        serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
            print(json.dumps({
                "status": report["status"],
                "output": str(args.output),
                "block_reasons": report["block_reasons"],
                "training_authorized": False,
            }, ensure_ascii=False))
        else:
            print(serialized, end="")
        return 2 if report["block_reasons"] else 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({
            "status": "blocked_isolation_plan_only",
            "error": str(error),
            "training_authorized": False,
            "training_performed": False,
            "measurement_authorized": False,
            "measurement_performed": False,
            "deployment_authorized": False,
            "deployment_performed": False,
            "merged_rows_written": 0,
        }, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
