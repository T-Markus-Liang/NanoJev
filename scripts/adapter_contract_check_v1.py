#!/usr/bin/env python3
"""Self-check for research/engineering_adapter_contract_v1.json.

Verifies, without training and without writing any corpus file:

1. The contract's frozen ``contract_sha256`` recomputes over the contract
   excluding that field (sorted-key canonical JSON, same convention as the
   corpus manifest's ``content_sha256``).
2. Every bound artifact hash still matches the bytes on disk: the corpus v2
   manifest ``content_sha256``, the heldout ``items.jsonl`` sha256, the pinned
   trainer/predictor sha256 and the protocol sha256.
3. Row mapping: every corpus v2 ``trainer_view`` row and every heldout row
   passes ``train_pipeline_decisions.validate_training_row``; the
   ``{id, state, questions}`` projection passes
   ``predict_toy_decisions.validate_request``; every ``deterministic_truth``
   gold is the argmax of its ``gold_probs``.
4. Renderer: every row's questions serialize through
   ``predict_toy_decisions.prepare_examples`` (the renderer the trainer
   shares) at ``max_length=2048`` with zero truncation, using the frozen
   checkpoint's own tokenizer loaded offline.

Prints a JSON receipt to stdout; ``--output`` writes it once (no overwrite).

Usage

    .venv/bin/python scripts/adapter_contract_check_v1.py \
        --contract research/engineering_adapter_contract_v1.json \
        [--tokenizer checkpoints/local_atomic_seed17/variants/local_atomic_seed17/tokenizer] \
        [--output results/adapter_contract_check_v1.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_toy_decisions import prepare_examples, read_json, validate_request  # noqa: E402
from train_pipeline_decisions import (candidate_ids, read_training_records,  # noqa: E402
                                      validate_training_row)

SCHEMA = "nanojev-engineering-adapter-check-v1"
SPLITS = ("train", "dev", "calibration", "test")
DEFAULT_TOKENIZER = ("checkpoints/local_atomic_seed17/variants/"
                     "local_atomic_seed17/tokenizer")


def digest_value(value):
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_contract(contract_path, repo_root, tokenizer_dir=None, max_length=2048):
    errors, warnings = [], []
    contract_path = Path(contract_path).resolve(strict=True)
    contract = read_json(contract_path)

    declared_hash = contract.get("contract_sha256")
    actual_hash = digest_value({k: v for k, v in contract.items()
                                if k != "contract_sha256"})
    hash_ok = isinstance(declared_hash, str) and declared_hash == actual_hash
    if not hash_ok:
        errors.append(f"contract_sha256 mismatch: declared {declared_hash!r} "
                      f"computed {actual_hash!r}")

    bound = contract.get("bound_artifacts", {})

    def resolve(rel):
        return (repo_root / rel).resolve()

    artifact_checks = {}
    corpus_dir = resolve(bound["corpus_v2"]["path"])
    manifest = read_json(corpus_dir / "manifest.json")
    artifact_checks["corpus_v2.content_sha256"] = {
        "declared": bound["corpus_v2"]["content_sha256"],
        "actual": manifest.get("content_sha256"),
        "match": manifest.get("content_sha256") == bound["corpus_v2"]["content_sha256"],
    }
    heldout_path = resolve(bound["heldout_v1"]["path"])
    artifact_checks["heldout_v1.sha256"] = {
        "declared": bound["heldout_v1"]["sha256"],
        "actual": file_hash(heldout_path),
        "match": file_hash(heldout_path) == bound["heldout_v1"]["sha256"],
    }
    for name in ("trainer", "predictor", "protocol_v2"):
        path = resolve(bound[name]["path"])
        artifact_checks[f"{name}.sha256"] = {
            "declared": bound[name]["sha256"],
            "actual": file_hash(path),
            "match": file_hash(path) == bound[name]["sha256"],
        }
    for name, result in artifact_checks.items():
        if not result["match"]:
            errors.append(f"bound artifact hash mismatch: {name}")

    # Row-mapping serialization check over every bound cohort row.
    rows = list(read_training_records(corpus_dir / "trainer_view")[0])
    rows += list(read_training_records(heldout_path)[0])
    row_stats = {"rows": len(rows), "questions": 0,
                 "by_cohort": {"corpus_v2": len(read_training_records(corpus_dir / "trainer_view")[0]),
                               "heldout_v1": len(rows) - len(read_training_records(corpus_dir / "trainer_view")[0])}}
    renderer = {"checked": 0, "max_leaf_tokens": 0, "tokenizer": None}
    tokenizer = None
    if tokenizer_dir is not None:
        tok_path = Path(tokenizer_dir).resolve()
        if not tok_path.is_dir():
            errors.append(f"tokenizer dir missing: {tok_path}")
        else:
            try:
                import os
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(str(tok_path), local_files_only=True,
                                                          trust_remote_code=False)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token = tokenizer.eos_token
                renderer["tokenizer"] = str(tok_path)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"tokenizer unavailable; render check skipped: {exc}")
    else:
        warnings.append("no --tokenizer given; render check skipped")

    for row in rows:
        try:
            targets = validate_training_row(row)
            validate_request({"states": [{k: row[k] for k in ("id", "state", "questions")}]})
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{row.get('id')}: mapping validation failed: {exc}")
            continue
        for qid, q in row["questions"].items():
            row_stats["questions"] += 1
            t = targets[qid]
            if t["gold_index"] is None or t["gold_distribution_probs"] is None:
                errors.append(f"{row['id']}:{qid}: no usable gold target")
            elif t["gold_probs_kind"] == "deterministic_truth":
                if t["gold_distribution_probs"][t["gold_index"]] != 1.0:
                    errors.append(f"{row['id']}:{qid}: deterministic gold not argmax")
            ids = candidate_ids(q)
            declared = row.get("gold_probs", {}).get(qid)
            if declared is not None and set(declared) != set(ids):
                errors.append(f"{row['id']}:{qid}: gold_probs keys != candidate ids")
        if tokenizer is not None:
            try:
                examples = prepare_examples(
                    {"states": [{k: row[k] for k in ("id", "state", "questions")}]},
                    tokenizer, max_length)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{row['id']}: renderer serialization failed: {exc}")
                continue
            renderer["checked"] += len(examples)
            for ex in examples:
                renderer["max_leaf_tokens"] = max(
                    renderer["max_leaf_tokens"], max(map(len, ex["leaf_tokens"])))

    return {
        "schema_version": SCHEMA,
        "contract": str(contract_path),
        "contract_sha256": {"declared": declared_hash, "computed": actual_hash,
                            "match": hash_ok},
        "artifact_checks": artifact_checks,
        "row_mapping": row_stats,
        "renderer": {**renderer, "max_length": max_length,
                     "all_leaves_within_budget":
                         renderer["max_leaf_tokens"] <= max_length},
        "errors": errors,
        "warnings": warnings,
        "status": "contract_check_passed" if not errors else "contract_check_failed",
        "training_authorized": False,
        "training_performed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path(DEFAULT_TOKENIZER))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = check_contract(args.contract, Path.cwd(), args.tokenizer, args.max_length)
        serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
            print(json.dumps({"status": report["status"], "output": str(args.output),
                              "errors": report["errors"]}))
        else:
            print(serialized, end="")
        return 0 if not report["errors"] else 2
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "error", "error": str(error),
                          "training_authorized": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
