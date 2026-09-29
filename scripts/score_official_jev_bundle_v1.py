#!/usr/bin/env python3
"""Normalize an official Jev direct receipt to offline-bundle metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from run_offline_benchmark_bundle_v1 import (
    perturbation_metrics,
    retrieval_metrics,
    sha256,
    summarize_records,
)

TRACK_FILES = {
    "official_jevbench_public": "official_jevbench_v1.2.4_public/public_231.jsonl",
    "semif_authored144": "semif_owned/authored144.jsonl",
    "semif_perturbations108": "semif_owned/perturbations108.jsonl",
    "semif_shape777_nogold": "semif_owned/shape777.jsonl",
    "semif_wanli256": "semif_rebuilt_external/wanli256.jsonl",
    "semif_every204": "semif_rebuilt_external/every_rows/inference204.jsonl",
}


def answer_probabilities(answer: dict) -> dict:
    if "noul" in answer:
        p_yes = float(answer["noul"])
        return {"no": 1.0 - p_yes, "yes": p_yes}
    if "probability" in answer:
        p_yes = float(answer["probability"])
        return {"no": 1.0 - p_yes, "yes": p_yes}
    return {str(k): float(v) for k, v in answer["probabilities"].items()}


def load_source_rows(bundle_root: Path) -> dict:
    rows = {}
    for rel in TRACK_FILES.values():
        path = bundle_root / rel
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def normalize_receipt(receipt: Path, bundle_root: Path) -> list[dict]:
    source_rows = load_source_rows(bundle_root)
    latest = {}
    for line in receipt.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        latest[(rec["track"], rec["id"])] = rec
    records = []
    for rec in latest.values():
        if rec.get("error"):
            continue
        source = source_rows.get(rec["id"], {})
        probs = answer_probabilities(rec["answer"])
        records.append({
            "track": rec["track"],
            "file": TRACK_FILES[rec["track"]],
            "id": rec["id"],
            "family": rec.get("family"),
            "split": source.get("split"),
            "group_id": source.get("group_id") or source.get("group")
            or rec["id"].rsplit("/", 1)[0],
            "question_id": rec["id"].rsplit("/", 1)[-1],
            "gold_label": rec.get("gold"),
            "predicted_label": rec.get("pred"),
            "correct": rec.get("correct"),
            "probabilities": probs,
            "confidence": max(probs.values()) if probs else None,
            "latency_s": rec.get("elapsed_ms", 0) / 1000.0,
            "provenance": source.get("provenance"),
            "input_tokens": rec.get("input_tokens"),
            "output_tokens": rec.get("output_tokens"),
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path,
                        default=Path("data/jevbench_offline_bundle_v1"))
    parser.add_argument("--receipt", type=Path,
                        default=Path("research/official_jev_direct_run_v1.jsonl"))
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    records = normalize_receipt(args.receipt.resolve(), args.bundle_root.resolve())
    by_file = defaultdict(list)
    for record in records:
        by_file[record["file"]].append(record)
    file_summaries = {path: summarize_records(rows)
                      for path, rows in sorted(by_file.items())}
    every_path = "semif_rebuilt_external/every_rows/inference204.jsonl"
    every_gold = [r for r in by_file.get(every_path, [])
                  if r.get("gold_label") is not None]
    if every_gold:
        file_summaries["semif_rebuilt_external/every_rows/gold154.jsonl"] = \
            summarize_records(every_gold)
    summary = {
        "schema_version": "nanojev-official-jev-bundle-normalized-v1",
        "model": "jev-latest direct",
        "receipt": str(args.receipt.resolve()),
        "receipt_sha256": sha256(args.receipt),
        "bundle_root": str(args.bundle_root.resolve()),
        "bundle_manifest_sha256": sha256(
            args.bundle_root / "offline_bundle_manifest.json"),
        "overall": summarize_records(records),
        "files": file_summaries,
        "semif_perturbation_stability": perturbation_metrics(records),
        "every_retrieval": retrieval_metrics(records),
        "notes": [
            "official direct receipt scores every204 once; gold154 labels are embedded on 154 of those rows",
            "shape777 and action-firewall rows are unlabeled",
        ],
    }
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = prefix.with_suffix(".predictions.jsonl")
    summary_path = prefix.with_suffix(".summary.json")
    with predictions_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary["predictions_path"] = str(predictions_path)
    summary["predictions_sha256"] = sha256(predictions_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                       allow_nan=False) + "\n")
    print(json.dumps({"output": str(summary_path),
                      "summary_sha256": sha256(summary_path)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
