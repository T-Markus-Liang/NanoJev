#!/usr/bin/env python3
"""Read-only isolation check: engineering_heldout_v1 vs engineering_judgment_corpus_v2.

Proves, at the canonical model-visible-input level, that the independently
authored heldout cohort shares zero inputs with every corpus v2 split.  The
canonicalization is exactly the auditor's: ``audit_engineering_corpus_v1.
visible_input`` = {state, question type, instructions, criteria} hashed with
sorted keys — the same bytes the trainer/predictor render, excluding record IDs,
qids, source-group IDs and gold, none of which enter model input.  This is an
exact-overlap check, a lower bound on leakage; it does not claim to detect every
paraphrase.

Checks performed:

1. Corpus integrity: ``audit_engineering_corpus_v2.audit_corpus`` (the dedicated
   V2 auditor, in-process) must report no block reasons.  Its own
   holdout_block_reasons are corpus-internal (cross-seed) findings, recorded
   verbatim for context.
2. Heldout rows pass ``train_pipeline_decisions.read_training_records`` and the
   per-row validator, all carry split ``test``, unique IDs and declared
   ``source_group_id``.
3. Zero canonical-input overlap between the heldout and EACH corpus split and
   the corpus as a whole (both trainer-view rows and item-view requests are
   fingerprinted).
4. Zero overlap of declared identities: row ids, state_ids, source_group_ids,
   family_ids.

Emits ``results/heldout_v1_isolation_receipt.json`` (write-once).  Never merges,
relabels, repartitions or trains; input file hashes are snapshotted before and
after.

Usage

    .venv/bin/python scripts/heldout_v1_isolation_check.py \
        --corpus research/engineering_judgment_corpus_v2 \
        --heldout research/engineering_heldout_v1/items.jsonl \
        --output results/heldout_v1_isolation_receipt.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_engineering_corpus_v1 import audit_rows, file_hash, references  # noqa: E402
from audit_engineering_corpus_v2 import audit_corpus  # noqa: E402
from build_engineering_corpus_v2 import visible_fingerprint  # noqa: E402
from train_pipeline_decisions import read_training_records  # noqa: E402

SCHEMA = "nanojev-engineering-heldout-isolation-v1"
SPLITS = ("train", "dev", "calibration", "test")


def heldout_fingerprints(rows):
    refs = references(rows, "heldout")
    return {fp: refs for fp, refs in refs.items()}


def isolation_check(corpus_dir, heldout_path):
    corpus_dir = Path(corpus_dir).resolve(strict=True)
    heldout_path = Path(heldout_path).resolve(strict=True)
    inputs = [heldout_path, corpus_dir / "manifest.json"]
    inputs += [corpus_dir / "trainer_view" / f"{s}.jsonl" for s in SPLITS]
    inputs += [corpus_dir / "items" / f"{s}.jsonl" for s in SPLITS]
    before = {str(p): file_hash(p) for p in inputs}

    corpus_audit = audit_corpus(corpus_dir)  # dedicated V2 auditor, read-only

    corpus_rows, corpus_files = read_training_records(corpus_dir / "trainer_view")
    heldout_rows, _ = read_training_records(heldout_path)

    corpus_refs = references(corpus_rows, "corpus")
    heldout_refs = references(heldout_rows, "heldout")
    common = sorted(set(corpus_refs) & set(heldout_refs))

    per_split = {}
    for split in SPLITS:
        split_rows = [r for r in corpus_rows if r["split"] == split]
        split_refs = references(split_rows, f"corpus:{split}")
        shared = sorted(set(split_refs) & set(heldout_refs))
        per_split[split] = {
            "corpus_questions": sum(len(r["questions"]) for r in split_rows),
            "shared_canonical_inputs": len(shared),
            "shared_input_sha256": shared,
        }

    # Item-view fingerprints cover the same visible content; compute them too so
    # the check binds both published views.
    item_fps = set()
    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["items"]:
        item_fps.add(visible_fingerprint(item))
    heldout_fps = set(heldout_fingerprints(heldout_rows))
    item_view_overlap = sorted(item_fps & heldout_fps)

    def field_values(rows, key):
        out = set()
        for row in rows:
            if key == "source_group_id":
                value = row.get("metadata", {}).get("source_group_id")
            else:
                value = row.get(key)
            if value is not None:
                out.add(value)
        return out

    identity_overlap = {
        key: sorted(field_values(corpus_rows, key) & field_values(heldout_rows, key))
        for key in ("id", "state_id", "family_id", "source_group_id")
    }

    heldout_audit = audit_rows(heldout_rows)
    block_reasons = []
    if corpus_audit["block_reasons"]:
        block_reasons.append("corpus_v2_internal_audit_failed")
    if common or item_view_overlap:
        block_reasons.append("heldout_shares_canonical_inputs_with_corpus")
    if any(identity_overlap.values()):
        block_reasons.append("heldout_shares_declared_identities_with_corpus")
    if heldout_audit["block_reasons"]:
        block_reasons.append("heldout_internal_audit_failed")
    if any(r["split"] != "test" for r in heldout_rows):
        block_reasons.append("heldout_rows_not_all_split_test")
    if len(heldout_rows) < 40:
        block_reasons.append("heldout_below_40_items")

    after = {str(p): file_hash(p) for p in inputs}
    inputs_changed = before != after
    if inputs_changed:
        block_reasons.append("input_files_changed_during_check")

    return {
        "schema_version": SCHEMA,
        "corpus": str(corpus_dir),
        "heldout": str(heldout_path),
        "input_hashes": before,
        "corpus_files_read": [str(p) for p in corpus_files],
        "corpus_v2_audit": {
            "status": corpus_audit["status"],
            "block_reasons": corpus_audit["block_reasons"],
            "holdout_block_reasons": corpus_audit["holdout_block_reasons"],
            "item_count": corpus_audit["item_count"],
            "canonical_input_cross_split_groups":
                corpus_audit["audit"]["canonical_input_cross_split_groups"],
            "evaluation_provenance_hits": corpus_audit["audit"]["evaluation_provenance_hits"],
        },
        "heldout": {
            "path": str(heldout_path),
            "items": len(heldout_rows),
            "questions": heldout_audit["question_count"],
            "source_groups": heldout_audit["declared_source_groups"],
            "counts_by_split": heldout_audit["counts_by_split"],
            "internal_block_reasons": heldout_audit["block_reasons"],
        },
        "isolation": {
            "method": ("exact canonical model-visible-input equality: sha256 of "
                       "{state, question.type, instructions, criteria}, sorted-key JSON; "
                       "checked against trainer_view rows and item-view requests of every "
                       "corpus v2 split"),
            "common_canonical_inputs_total": len(common),
            "common_canonical_inputs_by_split": per_split,
            "item_view_overlap": len(item_view_overlap),
            "heldout_affected_records": len(
                {ref["id"] for key in common for ref in heldout_refs[key]}),
            "corpus_affected_records": len(
                {ref["id"] for key in common for ref in corpus_refs[key]}),
            "identity_overlap": identity_overlap,
            "overlap_details": [
                {"input_sha256": key,
                 "references": corpus_refs[key] + heldout_refs[key]}
                for key in common],
        },
        "input_files_changed": inputs_changed,
        "block_reasons": sorted(set(block_reasons)),
        "status": ("isolation_passed_evaluation_only"
                   if not block_reasons else "isolation_failed"),
        "limitations": [
            "exact-match canonicalization is a lower bound; paraphrase-level leakage "
            "is addressed by independent authorship declared in per-item provenance",
            "passing isolation does not authorize training, merging or deployment",
        ],
        "training_authorized": False,
        "training_performed": False,
        "merged_rows_written": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = isolation_check(args.corpus, args.heldout)
        serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            if args.output.resolve().is_relative_to(args.corpus.resolve()):
                parser.error("receipt cannot live inside the audited corpus")
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
            print(json.dumps({"status": report["status"], "output": str(args.output),
                              "block_reasons": report["block_reasons"],
                              "common_canonical_inputs":
                                  report["isolation"]["common_canonical_inputs_total"]}))
        else:
            print(serialized, end="")
        return 0 if not report["block_reasons"] else 2
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "error", "error": str(error),
                          "training_authorized": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
