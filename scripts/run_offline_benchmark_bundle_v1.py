#!/usr/bin/env python3
"""Run the offline JevBench/SemIf bundle against a NanoJev checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

from jevbench_adapter_v1 import task_to_nanojev_payload
from predict_toy_decisions import DecisionPredictor, prepare_examples

BUNDLE_FILES = [
    ("official_jevbench_v1.2.4_public", "official_jevbench_v1.2.4_public/public_231.jsonl"),
    ("semif_owned", "semif_owned/authored144.jsonl"),
    ("semif_owned", "semif_owned/perturbations108.jsonl"),
    ("semif_owned", "semif_owned/shape777.jsonl"),
    ("semif_rebuilt_external", "semif_rebuilt_external/wanli256.jsonl"),
    ("semif_rebuilt_external", "semif_rebuilt_external/every_rows/inference204.jsonl"),
    ("semif_rebuilt_external", "semif_rebuilt_external/every_rows/gold154.jsonl"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(root: Path) -> dict:
    manifest_path = root / "offline_bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file() or sha256(path) != entry["sha256"]:
            failures.append(entry["path"])
    if failures:
        raise ValueError(f"bundle hash verification failed: {failures}")
    return manifest


def semif_payload(row: dict) -> dict:
    criteria = {str(option["id"]): str(option["description"])
                for option in row["options"]}
    question = {"type": "choice", "instructions": row["question"],
                "criteria": criteria}
    return {"states": [{"id": row["id"], "state": row["state"],
                        "questions": {"decision": question}}]}


def row_payload(track: str, row: dict) -> dict:
    if track == "official_jevbench_v1.2.4_public":
        return task_to_nanojev_payload(row)
    return semif_payload(row)


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


def summarize_records(records):
    labeled = [r for r in records if r.get("gold_label") is not None]
    confs = [r["confidence"] for r in records]
    lat = [r["latency_s"] for r in records]
    out = {"rows": len(records), "labeled_rows": len(labeled),
           "mean_confidence": statistics.fmean(confs) if confs else None,
           "coverage_at_0.9": sum(c >= 0.9 for c in confs) / len(confs)
           if confs else None,
           "latency_s": {"wall": sum(lat), "mean": statistics.fmean(lat) if lat else None,
                          "p50": percentile(lat, 0.50), "p95": percentile(lat, 0.95),
                          "min": min(lat) if lat else None,
                          "max": max(lat) if lat else None}}
    if labeled:
        correct = sum(r["correct"] for r in labeled)
        out.update({
            "accuracy": correct / len(labeled),
            "confident_wrong_at_0.9": sum(
                r["confidence"] >= 0.9 and not r["correct"] for r in labeled),
            "nll": statistics.fmean(
                -math.log(max(r["probabilities"].get(r["gold_label"], 0.0), 1e-12))
                for r in labeled),
            "brier": statistics.fmean(
                sum((p - (1.0 if label == r["gold_label"] else 0.0)) ** 2
                    for label, p in r["probabilities"].items())
                for r in labeled),
        })
        by_label = defaultdict(lambda: [0, 0])
        for r in labeled:
            by_label[r["gold_label"]][0] += 1
            by_label[r["gold_label"]][1] += int(r["correct"])
        out["balanced_accuracy"] = statistics.fmean(
            ok / n for n, ok in by_label.values())
    return out


def make_predictor(checkpoint_dir, device, precision, max_length):
    config_path = Path(checkpoint_dir) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("set_head") == "marker":
        from predict_jfast_modernbert import JFastDecisionPredictor
        return JFastDecisionPredictor(checkpoint_dir, device_name=device,
                                      precision=precision), True
    return DecisionPredictor(checkpoint_dir, max_length=max_length,
                             device_name=device, precision=precision), False


def predict_one(predictor, marker_head, payload, max_length, temperature):
    if marker_head:
        return predictor.predict(payload, batch_questions=0,
                                 temperature=temperature)
    return predictor.predict(payload, batch_questions=0,
                             temperature=temperature, shared_prefix=True,
                             shared_prefix_row_tokens=max_length,
                             shared_prefix_max_rows=64,
                             shared_prefix_max_attn_positions=4_000_000)


def retrieval_metrics(records):
    # Every retrieval: rank source files by P(yes) for each family/question.
    inference = [r for r in records
                 if r["file"].endswith("every_rows/inference204.jsonl")]
    gold = [r for r in records
            if r["file"].endswith("every_rows/gold154.jsonl")
            or (r["file"].endswith("every_rows/inference204.jsonl")
                and r.get("gold_label") is not None)]
    gold_by_key = {(r["family"], r["question_id"], r["group_id"]): r["gold_label"]
                   for r in gold}
    # `gold_label` is already the option id ("yes"/"no"), not the raw index.
    by_query = defaultdict(list)
    for r in inference:
        p_yes = r["probabilities"].get("yes", 0.0)
        by_query[(r["family"], r["question_id"])].append((r["group_id"], p_yes))
    recalls1 = []
    recalls3 = []
    reciprocal_ranks = []
    for (family, question_id), ranked in sorted(by_query.items()):
        relevant = {group for (fam, qid, group), label in gold_by_key.items()
                    if fam == family and qid == question_id and label == "yes"}
        if not relevant:
            continue
        ranked.sort(key=lambda x: (-x[1], x[0]))
        groups = [group for group, _ in ranked]
        first = min((groups.index(group) for group in relevant if group in groups),
                    default=None)
        recalls1.append(first is not None and first < 1)
        recalls3.append(first is not None and first < 3)
        reciprocal_ranks.append(0.0 if first is None else 1.0 / (first + 1))
    return {"queries": len(recalls1), "recall_at_1": statistics.fmean(recalls1) if recalls1 else None,
            "recall_at_3": statistics.fmean(recalls3) if recalls3 else None,
            "mrr": statistics.fmean(reciprocal_ranks) if reciprocal_ranks else None}


def perturbation_metrics(records):
    by_id = {r["id"]: r for r in records}
    pairs = []
    for r in records:
        if r["file"].endswith("perturbations108.jsonl"):
            base_id = r.get("provenance", {}).get("base_id")
            base = by_id.get(base_id)
            if base:
                l1 = sum(abs(r["probabilities"].get(k, 0.0) -
                             base["probabilities"].get(k, 0.0))
                         for k in set(r["probabilities"]) | set(base["probabilities"]))
                pairs.append({"id": r["id"], "base_id": base_id,
                              "argmax_flip": r["predicted_label"] != base["predicted_label"],
                              "prob_l1": l1,
                              "max_abs_delta": max(
                                  abs(r["probabilities"].get(k, 0.0) -
                                      base["probabilities"].get(k, 0.0))
                                  for k in set(r["probabilities"]) | set(base["probabilities"]))})
    return {"pairs": len(pairs),
            "argmax_flip_rate": statistics.fmean(x["argmax_flip"] for x in pairs) if pairs else None,
            "mean_prob_l1": statistics.fmean(x["prob_l1"] for x in pairs) if pairs else None,
            "mean_max_abs_delta": statistics.fmean(x["max_abs_delta"] for x in pairs) if pairs else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path,
                        default=Path("data/jevbench_offline_bundle_v1"))
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"],
                        default="fp32")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()

    root = args.bundle_root.resolve()
    manifest = verify_manifest(root)
    predictor, marker_head = make_predictor(
        args.checkpoint_dir, args.device, args.precision, args.max_length)
    all_records = []
    by_file = {}
    run_start = time.perf_counter()
    for track, rel in BUNDLE_FILES:
        path = root / rel
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            payload = row_payload(track, row)
            start = time.perf_counter()
            out = predict_one(predictor, marker_head, payload,
                              args.max_length, args.temperature)
            latency = time.perf_counter() - start
            answer = out["states"][0]["answers"]["decision"]
            probs = answer["probabilities"]
            if track == "official_jevbench_v1.2.4_public" and \
                    row["question"]["type"] == "noul":
                probs = {"no": probs["false"], "yes": probs["true"]}
            labels = list(probs)
            predicted = max(labels, key=probs.get)
            if track == "official_jevbench_v1.2.4_public":
                expected = row.get("expected")
                gold_label = str(expected) if expected is not None else None
            elif "label" in row:
                gold_label = labels[row["label"]]
            else:
                gold_label = None
            record = {"track": track, "file": rel, "id": row["id"],
                      "family": row.get("family"), "split": row.get("split"),
                      "group_id": row.get("group_id") or row.get("group"),
                      "question_id": row["id"].rsplit("/", 1)[-1],
                      "gold_label": gold_label,
                      "predicted_label": predicted,
                      "correct": (predicted == gold_label
                                  if gold_label is not None else None),
                      "probabilities": probs,
                      "confidence": max(probs.values()),
                      "latency_s": latency,
                      "provenance": row.get("provenance")}
            records.append(record)
        by_file[rel] = summarize_records(records)
        all_records.extend(records)
        print(json.dumps({"file": rel, "rows": len(records),
                          "summary": by_file[rel]}), flush=True)

    summary = {"schema_version": "nanojev-offline-bundle-run-v1",
               "bundle_root": str(root),
               "bundle_manifest_sha256": sha256(root / "offline_bundle_manifest.json"),
               "adapter_binding_recorded": manifest.get("adapter_binding"),
               "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
               "device": args.device,
               "precision": args.precision,
               "temperature": args.temperature,
               "wall_s": time.perf_counter() - run_start,
               "files": by_file,
               "overall": summarize_records(all_records),
               "semif_perturbation_stability": perturbation_metrics(all_records),
               "every_retrieval": retrieval_metrics(all_records),
               "notes": ["shape777 and every inference204 are unlabeled; they report distribution/latency only",
                         "official JevBench public rows are diagnostic evidence, not an official milestone"]}
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = prefix.with_suffix(".predictions.jsonl")
    summary_path = prefix.with_suffix(".summary.json")
    with predictions_path.open("w", encoding="utf-8") as stream:
        for record in all_records:
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
