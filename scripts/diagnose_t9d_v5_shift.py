#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 18, 19)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def gold_key(row):
    qid, question = next(iter(row["questions"].items()))
    gold = row["gold"][qid]
    return ("true" if gold else "false") if question["type"] == "boolean" else str(gold)


def candidate_count(row):
    question = next(iter(row["questions"].values()))
    if question["type"] == "boolean":
        return 2
    return len(question["criteria"])


def receipt_answers(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return {state["id"]: next(iter(state["answers"].values())) for state in value["states"]}


def bucket_cardinality(k):
    return str(k) if k <= 12 else "13-32" if k <= 32 else "33-255"


def metric(rows):
    if not rows:
        return None
    return {
        "n": len(rows),
        "accuracy": math.fsum(row["correct"] for row in rows) / len(rows),
        "mean_confidence": statistics.fmean(row["confidence"] for row in rows),
        "mean_confidence_correct": statistics.fmean(row["confidence"] for row in rows if row["correct"])
        if any(row["correct"] for row in rows) else None,
        "mean_confidence_wrong": statistics.fmean(row["confidence"] for row in rows if not row["correct"])
        if any(not row["correct"] for row in rows) else None,
        "coverage_at_0.9": math.fsum(row["confidence"] >= 0.9 for row in rows) / len(rows),
        "protected_errors_at_0.9": sum(row["confidence"] >= 0.9 and not row["correct"] for row in rows),
    }


def grouped(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: metric(values) for name, values in sorted(groups.items())}


def cohort_rows(items, answers_by_seed):
    rows = []
    for item in items:
        predictions = []
        for seed, answers in sorted(answers_by_seed.items()):
            answer = answers[item["id"]]
            probs = answer["probabilities"]
            predictions.append({"seed": seed, "pred": max(probs, key=probs.get),
                                "confidence": max(probs.values())})
        gold = gold_key(item)
        for pred in predictions:
            rows.append({
                "id": item["id"], "seed": pred["seed"], "family": item["family_id"],
                "type": next(iter(item["questions"].values()))["type"],
                "cardinality": candidate_count(item),
                "cardinality_bucket": bucket_cardinality(candidate_count(item)),
                "type_cardinality": f"{next(iter(item['questions'].values()))['type']}:{candidate_count(item)}",
                "correct": pred["pred"] == gold, "confidence": pred["confidence"],
            })
    return rows


def agreement(items, answers_by_seed):
    by_type = defaultdict(list)
    for item in items:
        preds = [max(answers[item["id"]]["probabilities"],
                     key=answers[item["id"]]["probabilities"].get)
                 for answers in answers_by_seed.values()]
        qtype = next(iter(item["questions"].values()))["type"]
        by_type[qtype].append(len(set(preds)) == 1)
    return {qtype: {"n": len(values), "unanimous_rate": math.fsum(values) / len(values)}
            for qtype, values in sorted(by_type.items())}


def cardinality_profile(rows):
    counts = Counter()
    for row in rows:
        counts[(next(iter(row["questions"].values()))["type"], bucket_cardinality(candidate_count(row)))] += 1
    return {f"{qtype}:{bucket}": n for (qtype, bucket), n in sorted(counts.items())}


def input_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def receipt_spec(spec):
    if "=" in spec:
        seed, path = spec.split("=", 1)
        return int(seed), input_path(path)
    match = re.search(r"seed(\d+)", spec)
    if not match:
        raise ValueError(f"receipt seed is not inferable: {spec}")
    return int(match.group(1)), input_path(spec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/t9d_v5_shift_diagnosis_v1.json")
    parser.add_argument("--heldout", type=Path, default=ROOT / "research/engineering_heldout_v2/items.jsonl")
    parser.add_argument("--train", type=Path,
                        default=ROOT / "research/engineering_judgment_corpus_v4/trainer_view/train.jsonl")
    parser.add_argument("--test", type=Path,
                        default=ROOT / "research/engineering_judgment_corpus_v4/trainer_view/test.jsonl")
    parser.add_argument("--receipt", action="append", default=None,
                        help="SEED=PATH or a path containing seedN; repeatable")
    parser.add_argument("--heldout-key", default="heldout_v2")
    parser.add_argument("--schema-version", default="nanojev-t9d-v5-shift-diagnosis-v1")
    args = parser.parse_args()
    heldout_path = input_path(args.heldout)
    v4_train_path = input_path(args.train)
    v4_test_path = input_path(args.test)
    items = load_jsonl(heldout_path)
    receipts = ({seed: ROOT / f"results/heldout_v2_post_t9dv5_seed{seed}.json" for seed in SEEDS}
                if args.receipt is None else dict(receipt_spec(spec) for spec in args.receipt))
    for path in [heldout_path, v4_train_path, v4_test_path, *receipts.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    answers = {seed: receipt_answers(path) for seed, path in receipts.items()}
    expected = {row["id"] for row in items}
    for seed, values in answers.items():
        if set(values) != expected:
            raise ValueError(f"seed{seed} receipt/item ID mismatch")
    rows = cohort_rows(items, answers)
    seed_metrics = grouped(rows, "seed")
    family = grouped(rows, "family")
    ranked = sorted(((name, values["accuracy"], values["n"]) for name, values in family.items()),
                    key=lambda value: (value[1], value[0]))
    output = {
        "schema_version": args.schema_version,
        "status": "aggregate_diagnosis_only_not_training_authorization",
        "inputs_sha256": {
            str(heldout_path.relative_to(ROOT)): sha256(heldout_path),
            str(v4_train_path.relative_to(ROOT)): sha256(v4_train_path),
            str(v4_test_path.relative_to(ROOT)): sha256(v4_test_path),
            **{str(path.relative_to(ROOT)): sha256(path) for path in receipts.values()},
        },
        args.heldout_key: {
            "overall": metric(rows), "by_seed": seed_metrics, "by_type": grouped(rows, "type"),
            "by_type_cardinality": grouped(rows, "type_cardinality"),
            "by_cardinality": grouped(rows, "cardinality"),
            "by_cardinality_bucket": grouped(rows, "cardinality_bucket"),
            "by_family": family, "seed_agreement": agreement(items, answers),
            "lowest_family_aggregates": [
                {"family": name, "accuracy": accuracy, "n_seed_rows": n} for name, accuracy, n in ranked[:5]],
            "highest_family_aggregates": [
                {"family": name, "accuracy": accuracy, "n_seed_rows": n} for name, accuracy, n in ranked[-5:]],
        },
        "shape_comparison": {
            "v4_train": cardinality_profile(load_jsonl(v4_train_path)),
            "v4_test": cardinality_profile(load_jsonl(v4_test_path)),
            args.heldout_key: cardinality_profile(items),
        },
        "interpretation_limits": [
            f"{args.heldout_key} was consumed once for aggregate diagnosis; per-item failures are not emitted.",
            "Family aggregates may identify hypotheses but must not become direct training labels or paraphrase targets.",
            "This receipt cannot distinguish base-model capacity from representation or semantic-domain shift by itself.",
            "Any repair corpus must be independently authored without copying heldout states, rules, vocabulary, or outputs.",
        ],
        "authorization": {"training_authorized": False, "deployment_authorized": False,
                          "jevbench_unlocked": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output),
                      "overall": output[args.heldout_key]["overall"],
                      "by_type": output[args.heldout_key]["by_type"]}, indent=2))


if __name__ == "__main__":
    main()
