#!/usr/bin/env python3
"""Evaluation-only perturbations of frozen workflow test/OOD records; no new labels."""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from benchmark_nanojev_v2 import canonical_json, file_identity, sha256_bytes
from evaluate_raw_dataset_v2 import load_rows
from train_pipeline_decisions import validate_training_row


VARIANTS = ("original", "archived_distractor", "structured_envelope", "choice_order_reversed")
ARCHIVE = (
    "An unrelated archived ticket mentioned an office heater, no authorization and no occupants. "
    "An old warehouse record listed a packed green large item and expedited shipping. "
    "An unrelated bag contained red=9, blue=1 and green=4. These are not the current record."
)


def variant(row, name):
    if row["split"] not in {"test", "ood"}:
        raise ValueError("challenge accepts test/OOD only; never generate training or calibration rows")
    if name not in VARIANTS:
        raise ValueError("unknown challenge variant")
    validate_training_row(row)
    if not row.get("metadata", {}).get("source_group_id"):
        raise ValueError("source group is required for correlated challenge variants")
    out = deepcopy(row)
    out["id"] = f"{row['id']}::{name}"
    # Keep state_id and source_group_id so variants cannot be counted as independent states.
    out["metadata"].update(challenge_variant=name, base_record_id=row["id"])
    if name == "archived_distractor":
        out["state"] = f"Archived unrelated material:\n{ARCHIVE}\n\nCurrent record:\n{row['state']}"
        for question in out["questions"].values():
            question["instructions"] = "Use only the Current record; ignore the archived material. " + question["instructions"]
    elif name == "structured_envelope":
        out["state"] = {"current_record": row["state"], "archived_unrelated_record": ARCHIVE}
        for question in out["questions"].values():
            question["instructions"] = "Use only current_record; ignore archived_unrelated_record. " + question["instructions"]
    elif name == "choice_order_reversed":
        for question in out["questions"].values():
            if question["type"] == "choice":
                question["criteria"] = dict(reversed(list(question["criteria"].items())))
    validate_training_row(out)
    return out


def build(source, output):
    rows, files = load_rows(source, {"test", "ood"})
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty; do not overwrite a frozen challenge")
    generated = [variant(row, name) for row in rows for name in VARIANTS]
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "nanojev-workflow-challenge-v1", "purpose": "evaluation_only",
        "source_files": [file_identity(path) for path in files],
        "builder": file_identity(__file__), "variants": list(VARIANTS),
        "limitations": [
            "Synthetic robustness challenge, not representative production or finance data.",
            "Variants share labels and source groups; do not treat them as independent samples.",
            "Choice reversal does not change Boolean or Score inputs; these are control questions.",
            "OOD inherits Chinese rendering, not a genuinely unseen task family.",
            "No checkpoint, threshold, training, or calibration selection is allowed on this challenge.",
        ], "splits": {},
    }
    for split in ("test", "ood"):
        selected = [row for row in generated if row["split"] == split]
        # Preserve candidate insertion order: sorting JSON keys would erase the reversal challenge.
        content = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in selected)
        (output / f"{split}.jsonl").write_text(content, encoding="utf-8")
        manifest["splits"][split] = {
            "records": len(selected), "questions": sum(len(row["questions"]) for row in selected),
            "source_groups": len({row["metadata"]["source_group_id"] for row in selected}),
            "variants": dict(Counter(row["metadata"]["challenge_variant"] for row in selected)),
            "sha256": sha256_bytes(content.encode("utf-8")),
        }
    (output / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.output_dir), indent=2))
