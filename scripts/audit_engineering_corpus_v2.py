#!/usr/bin/env python3
"""Read-only audit for the componented engineering corpus V2.

This is deliberately separate from ``audit_engineering_corpus_v1.py``: the V1
builder contract must remain immutable evidence, while V2 has a different manifest
schema and source-group construction.  The audit never rewrites, merges, relabels,
repartitions or trains.  A V2 corpus can pass internal split isolation and still fail
the independent-heldout check when a different seed reuses canonical inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from audit_engineering_corpus_v1 import SPLITS, audit_rows, compare_cohorts, file_hash
from build_engineering_corpus_v2 import check, derive_manifest, trainer_rows
from train_pipeline_decisions import read_training_records


SCHEMA = "nanojev-engineering-adapter-preflight-v2"


def paths(root: Path) -> list[Path]:
    return [root / "manifest.json"] + [root / "trainer_view" / f"{split}.jsonl" for split in SPLITS]


def snapshot(root: Path) -> dict[str, str]:
    result = {}
    for path in paths(root):
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(path.relative_to(root))] = file_hash(path)
    return result


def audit_corpus(root: Path, compare_seed: int | None = None) -> dict[str, Any]:
    root = Path(root).resolve(strict=True)
    before = snapshot(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    errors = []
    if manifest.get("schema_version") != "nanojev-engineering-judgment-manifest-v2":
        errors.append("wrong V2 manifest schema")
    integrity_errors = check(root)
    errors.extend(integrity_errors)
    rows, _ = read_training_records(root / "trainer_view")
    audit = audit_rows(rows)
    errors.extend(audit["block_reasons"])

    cross_seed = None
    holdout_block_reasons = []
    if compare_seed is None:
        holdout_block_reasons.append("cross_seed_comparison_required")
    else:
        seed = manifest.get("seed")
        if seed == compare_seed:
            raise ValueError("comparison seed must differ from corpus seed")
        other_manifest = derive_manifest(compare_seed)
        other_rows = trainer_rows(other_manifest)
        cross_seed = compare_cohorts(rows, other_rows)
        cross_seed.update({"original_seed": seed, "comparison_seed": compare_seed,
                           "materialized": False})
        if cross_seed["common_canonical_inputs"]:
            holdout_block_reasons.append("seed_change_does_not_prove_semantic_holdout")

    after = snapshot(root)
    if after != before:
        errors.append("input_files_changed_during_audit")
    report = {
        "schema_version": SCHEMA,
        "corpus": str(root),
        "audit_subject": "V2 component-grouped corpus; V1 preserved separately",
        "source_hashes": before,
        "source_hashes_after": after,
        "builder_integrity_errors": integrity_errors,
        "builder": "scripts/build_engineering_corpus_v2.py",
        "builder_sha256": manifest.get("builder_sha256"),
        "audit": audit,
        "component_count": manifest.get("repairs", {}).get("canonical_input_components"),
        "pair_count": manifest.get("pair_count"),
        "item_count": manifest.get("item_count"),
        "counts_by_split": manifest.get("counts_by_split"),
        "cross_seed": cross_seed,
        "block_reasons": sorted(set(errors)),
        "holdout_block_reasons": sorted(set(holdout_block_reasons)),
        "status": "preflight_passed_not_training_authorized" if not errors else "blocked_isolation_plan_only",
        "training_authorized": False,
        "training_performed": False,
        "measurement_authorized": False,
        "measurement_performed": False,
        "deployment_authorized": False,
        "deployment_performed": False,
        "merged_rows_written": 0,
        "review_requirements": [
            "independently author a fresh heldout/OOD cohort; a new seed is not sufficient",
            "freeze gate renderer, source-to-output adapter mapping and endpoint identities",
            "retain V1 as immutable evidence and do not train on it",
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--compare-seed", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = audit_corpus(args.corpus, args.compare_seed)
        serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            if args.output.resolve().is_relative_to(args.corpus.resolve()):
                parser.error("audit output cannot be inside corpus")
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
        else:
            print(serialized, end="")
        if args.output:
            print(json.dumps({"status": report["status"], "output": str(args.output),
                              "block_reasons": report["block_reasons"],
                              "holdout_block_reasons": report["holdout_block_reasons"]}, ensure_ascii=False))
        return 2 if report["block_reasons"] or report["holdout_block_reasons"] else 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "blocked_isolation_plan_only", "error": str(error),
                          "training_authorized": False}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
